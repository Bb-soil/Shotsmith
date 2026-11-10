"""Focused remote bootstrap + loopback tunnel composition for P1 v3.

All real installation/ComfyUI/tunnel activity is behind explicit live gates at
the operator. Helpers accept explicit runners/process launchers so offline
tests exercise the same protocol without SSH or installs. The ordering is
deliberate: owned root -> pinned setup/base torch -> download complete ->
preflight -> owned ComfyUI -> exact tunnel -> backend readiness.

The local tunnel is owned by an independent monitor process (see
``v3_tunnel_child.py``). The monitor spawns the exact SSH argv, writes the
heartbeat only while that SSH child is alive, and answers a bounded loopback
challenge. A PID plus a heartbeat file written by a daemon thread is not
ownership evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import stat
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from scripts.p1.remote import (
    RemoteIdentity,
    RemoteRunner,
    _validate_remote_absolute_path,
)

DEFAULT_COMFY_REMOTE_PORT = 8188
DEFAULT_LOCAL_TUNNEL_PORT = 8189
TUNNEL_SCHEMA = "anime-video-studio/p1-v3-tunnel-owner/v1"
TUNNEL_ACK_SCHEMA = "anime-video-studio/p1-v3-tunnel-ack/v1"
TUNNEL_HEARTBEAT_SCHEMA = "anime-video-studio/p1-v3-tunnel-heartbeat/v1"
TUNNEL_CONTROL_SCHEMA = "anime-video-studio/p1-v3-tunnel-control/v1"
TUNNEL_CONFIG_SCHEMA = "anime-video-studio/p1-v3-tunnel-config/v1"
TUNNEL_STATE_SCHEMA = "anime-video-studio/p1-v3-tunnel-state/v1"
TUNNEL_STARTING_SCHEMA = "anime-video-studio/p1-v3-tunnel-starting/v1"
TUNNEL_RECOVERY_SCHEMA = "anime-video-studio/p1-v3-tunnel-recovery/v1"
LAUNCH_SCHEMA = "anime-video-studio/p1-v3-launch/v1"
LAUNCH_RESULT_SCHEMA = "anime-video-studio/p1-v3-launch-result/v1"
ROOT_CLAIM_SCHEMA = "anime-video-studio/p1-v3-owned-root/v1"
DEFAULT_TUNNEL_HEARTBEAT_INTERVAL = 2.0
DEFAULT_TUNNEL_FRESHNESS_SECONDS = 30.0
MAX_TUNNEL_JSON_BYTES = 64 * 1024
MAX_TUNNEL_CONTROL_BYTES = 4 * 1024
MAX_TUNNEL_LOG_BYTES = 1024 * 1024
MAX_TUNNEL_ARCHIVE_LOG_TAIL = 64 * 1024

PID_ABSENT = "ABSENT"
PID_ALIVE = "ALIVE"
PID_UNKNOWN = "UNKNOWN"

_TUNNEL_LIFECYCLE_LOCK_NAME = ".tunnel-lifecycle.lock"

_SETUP_RUNNER = (
    "import os,sys;os.chdir(sys.argv[1]);os.execvp('bash',['bash','setup.sh'])"
)


class BootstrapError(Exception):
    pass


def _require_remaining(deadline_monotonic: float | None, label: str) -> float | None:
    """Refuse an expired absolute deadline before any filesystem/network work."""

    if deadline_monotonic is None:
        return None
    if isinstance(deadline_monotonic, bool) or not isinstance(
        deadline_monotonic, (int, float)
    ):
        raise BootstrapError(f"{label} deadline is invalid")
    remaining = float(deadline_monotonic) - time.monotonic()
    if remaining <= 0:
        raise BootstrapError(f"{label} deadline expired before any work")
    return remaining


def _validate_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BootstrapError(f"{name} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number <= 0:
        raise BootstrapError(f"{name} must be finite and positive")
    return number


def _read_small_json(path: Path, name: str) -> dict[str, Any]:
    if os.path.islink(path) or not os.path.isfile(path):
        raise BootstrapError(f"{name} must be a regular non-symlink file")
    if os.path.getsize(path) > MAX_TUNNEL_JSON_BYTES:
        raise BootstrapError(f"{name} exceeds the bound")
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"{name} must be an object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(data) > MAX_TUNNEL_JSON_BYTES:
        raise BootstrapError("tunnel JSON record exceeds the bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _remove_owned_json(path: Path | None, run_id: str, nonce: str) -> None:
    if path is None or not path.is_file() or os.path.islink(path):
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(value, dict):
        return
    if value.get("run_id") == run_id and value.get("nonce") == nonce:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _pid_live(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(code)
                ):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_pid_state(pid: int, api: Any | None = None) -> str:
    """Windows tri-state liveness with injectable API for tests."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return PID_UNKNOWN
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_INVALID_PARAMETER = 87
        if api is None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                ctypes.c_ulong,
                ctypes.c_bool,
                ctypes.c_ulong,
            ]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
            ]
            kernel32.GetExitCodeProcess.restype = ctypes.c_bool
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_bool
            open_process = kernel32.OpenProcess
            get_exit_code = kernel32.GetExitCodeProcess
            close_handle = kernel32.CloseHandle
            get_last_error = ctypes.get_last_error
        else:
            ctypes = getattr(api, "_ctypes", ctypes)
            open_process = api.OpenProcess
            get_exit_code = api.GetExitCodeProcess
            close_handle = api.CloseHandle
            get_last_error = api.GetLastError
        handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return (
                PID_ABSENT
                if int(get_last_error()) == ERROR_INVALID_PARAMETER
                else PID_UNKNOWN
            )
        try:
            code = ctypes.c_ulong()
            if not get_exit_code(handle, ctypes.byref(code)):
                return PID_UNKNOWN
            return PID_ALIVE if code.value == STILL_ACTIVE else PID_ABSENT
        finally:
            try:
                close_handle(handle)
            except Exception:
                pass
    except Exception:
        return PID_UNKNOWN


def pid_state(pid: int) -> str:
    """Tri-state process liveness for ownership decisions.

    Only a documented non-existent PID is ABSENT.  Access denied, platform API
    errors and every ambiguous result are UNKNOWN; a live/reused PID is ALIVE.
    POSIX PermissionError means the process exists but belongs to another user.
    """

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return PID_UNKNOWN
    if os.name == "nt":
        return _windows_pid_state(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return PID_ABSENT
    except PermissionError:
        return PID_ALIVE
    except OSError:
        return PID_UNKNOWN
    return PID_ALIVE


def _finite_timestamp(
    value: Any, label: str, *, allow_zero: bool = True
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BootstrapError(f"{label} is invalid")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        raise BootstrapError(f"{label} is invalid")
    return number


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        # Unreadable is treated as present so the caller fails closed.
        return True


def _is_link_like(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        # An unreadable path is not safe to follow.
        return True
    if stat.S_ISLNK(info.st_mode):
        return True
    try:
        isjunction = getattr(os.path, "isjunction", None)
        return bool(isjunction is not None and isjunction(path))
    except OSError:
        return True


def _same_path(left: Path | str, right: Path | str) -> bool:
    return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(
        os.path.abspath(str(right))
    )


def _assert_safe_path_chain(path: Path, *, directory: bool) -> None:
    """Refuse symlink/junction ancestry before any probe/write operation.

    Existing components are checked root -> leaf with lstat; once a component
    is missing the walk stops, so no descendant is probed through an unchecked
    junction.  The caller may create missing components only after this gate.
    """

    target = Path(path)
    if not target.is_absolute():
        raise BootstrapError("tunnel recovery path must be absolute")
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise BootstrapError(
                "tunnel recovery path ancestry is unreadable"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise BootstrapError(
                "tunnel recovery path has a symlink in its ancestry"
            )
        try:
            isjunction = getattr(os.path, "isjunction", None)
            if isjunction is not None and isjunction(current):
                raise BootstrapError(
                    "tunnel recovery path has a junction in its ancestry"
                )
        except OSError as exc:
            raise BootstrapError(
                "tunnel recovery path ancestry is unreadable"
            ) from exc
        if _same_path(current, target):
            if directory and not stat.S_ISDIR(info.st_mode):
                raise BootstrapError("tunnel recovery directory is not a directory")
            if not directory and not stat.S_ISREG(info.st_mode):
                raise BootstrapError(
                    "tunnel recovery file is not a regular file"
                )


def _write_exclusive_bytes(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise BootstrapError("tunnel archive directory cannot be opened") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise BootstrapError("tunnel archive directory fsync failed") from exc
    finally:
        os.close(fd)


def acquire_tunnel_lifecycle_lock(owner_path: Path) -> Any:
    """Short OS lock for archive-validation -> publish -> spawn -> ACK.

    The run directory and lock path are checked root -> leaf before the lock
    file is opened.  The lock is a distinct local file, never a tunnel sidecar.
    A same-account TOCTOU race against a symlink replacement after this static
    check is a documented local-filesystem limitation, not a security claim.
    """

    from anime_studio.lock import InstanceLock

    run_dir = Path(owner_path).parent
    _assert_safe_path_chain(run_dir, directory=True)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BootstrapError("tunnel run directory cannot be created") from exc
    _assert_safe_path_chain(run_dir, directory=True)
    lock_path = Path(owner_path).with_name(_TUNNEL_LIFECYCLE_LOCK_NAME)
    _assert_safe_path_chain(lock_path, directory=False)
    if _lexists(lock_path) and not stat.S_ISREG(os.lstat(lock_path).st_mode):
        raise BootstrapError("tunnel lifecycle lock is not a regular file")
    try:
        return InstanceLock(lock_path).acquire()
    except Exception as exc:
        raise BootstrapError(
            "tunnel lifecycle lock is busy; concurrent start/archive refused"
        ) from exc


def _archive_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _reap_child(pid: int) -> bool:
    """Reap an exited direct child (POSIX zombie); never kill it."""

    if os.name == "nt":
        return False
    try:
        waited, _status = os.waitpid(int(pid), os.WNOHANG)
    except (ChildProcessError, OSError):
        return False
    return waited == int(pid)


def _terminate_pid(pid: int, *, timeout: float = 5.0) -> bool:
    """Terminate one exact PID and return proof that it is no longer live."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return True
    wait = max(0.5, float(timeout))
    command_timeout = max(2.0, wait)
    for force in (False, True):
        if not _pid_live(pid):
            return True
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=command_timeout,
                    check=False,
                )
            else:
                os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
        except (OSError, subprocess.TimeoutExpired):
            pass
        end = time.monotonic() + wait
        while time.monotonic() < end and _pid_live(pid):
            time.sleep(0.05)
    return not _pid_live(pid)


def _terminate_owned_pids(
    monitor_pid: int | None,
    ssh_pid: int | None,
    *,
    timeout: float = 5.0,
    monitor_process: subprocess.Popen | None = None,
) -> bool:
    """Terminate the exact owned monitor+SSH PIDs and verify both are dead.

    When the monitor is our own Popen, its poll/wait status is authoritative:
    on POSIX a zombie still answers ``os.kill(pid, 0)`` until it is reaped.
    """

    wait = max(0.5, float(timeout))
    if monitor_pid is not None:
        if monitor_process is not None:
            if monitor_process.poll() is None:
                _terminate_pid(int(monitor_pid), timeout=wait)
            try:
                monitor_process.wait(timeout=wait)
            except Exception:
                pass
        elif _pid_live(int(monitor_pid)):
            if not _reap_child(int(monitor_pid)):
                _terminate_pid(int(monitor_pid), timeout=wait)
    if ssh_pid is not None and _pid_live(int(ssh_pid)):
        _terminate_pid(int(ssh_pid), timeout=wait)
    # A second pass covers a child reparented between the two calls.
    if monitor_pid is not None and _pid_live(int(monitor_pid)):
        if monitor_process is None or monitor_process.poll() is None:
            if monitor_process is not None or not _reap_child(int(monitor_pid)):
                _terminate_pid(int(monitor_pid), timeout=wait)
    if ssh_pid is not None and _pid_live(int(ssh_pid)):
        _terminate_pid(int(ssh_pid), timeout=wait)
    monitor_dead = (
        monitor_process.poll() is not None
        if monitor_process is not None
        else (monitor_pid is None or not _pid_live(int(monitor_pid)))
    )
    ssh_dead = ssh_pid is None or not _pid_live(int(ssh_pid))
    return monitor_dead and ssh_dead


def _terminate_process(
    process: subprocess.Popen | None,
    *,
    ssh_pid: int | None = None,
    timeout: float = 5.0,
) -> bool:
    if process is None:
        return ssh_pid is None or not _pid_live(int(ssh_pid))
    monitor_pid = int(process.pid)
    proof = _terminate_owned_pids(
        monitor_pid,
        ssh_pid,
        timeout=timeout,
        monitor_process=process,
    )
    try:
        process.wait(timeout=timeout)
    except Exception:
        pass
    return proof and process.poll() is not None


def ensure_local_port_free(host: str, port: int) -> None:
    if host != "127.0.0.1":
        raise BootstrapError("tunnel host must be loopback")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise BootstrapError(f"local tunnel port {port} is occupied") from exc


def build_tunnel_argv(
    identity: RemoteIdentity,
    *,
    local_port: int,
    remote_port: int,
    ssh_executable: str = "ssh",
    ssh_args_prefix: Sequence[str] = (),
) -> tuple[str, ...]:
    if not (1 <= local_port <= 65535 and 1 <= remote_port <= 65535):
        raise BootstrapError("tunnel ports are invalid")
    if local_port == remote_port:
        raise BootstrapError("tunnel ports must differ")
    if (
        not isinstance(ssh_executable, str)
        or not ssh_executable
        or "\x00" in ssh_executable
        or "\n" in ssh_executable
    ):
        raise BootstrapError("ssh executable is invalid")
    prefix = tuple(ssh_args_prefix)
    if not all(isinstance(item, str) and item and "\x00" not in item for item in prefix):
        raise BootstrapError("ssh argv prefix is invalid")
    identity_args: list[str] = []
    if identity.identity_file is not None:
        identity_args = ["-i", str(identity.identity_file)]
    extra_options: list[str] = []
    global_known_hosts = getattr(identity, "global_known_hosts", None)
    if global_known_hosts is not None:
        extra_options.extend(["-o", f"GlobalKnownHostsFile={global_known_hosts}"])
    host_key_algorithms = getattr(identity, "host_key_algorithms", None)
    if host_key_algorithms:
        extra_options.extend(["-o", f"HostKeyAlgorithms={host_key_algorithms}"])
    return (
        ssh_executable,
        *prefix,
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={identity.known_hosts}",
        *extra_options,
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=10",
        "-p",
        str(identity.port),
        *identity_args,
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        identity.destination,
    )


@dataclass
class OwnedTunnel:
    identity: RemoteIdentity
    local_port: int
    remote_port: int
    log_path: Path
    run_id: str
    nonce: str
    owner_path: Path | None = None
    process: subprocess.Popen | None = None
    popen: Callable[..., Any] | None = None
    attached_pid: int | None = None
    expected_fingerprint: str | None = None
    ssh_executable: str = "ssh"
    ssh_args_prefix: tuple[str, ...] = ()
    heartbeat_interval: float = DEFAULT_TUNNEL_HEARTBEAT_INTERVAL
    monitor_script: Path | None = None
    monitor_pid: int | None = None
    control_port: int | None = None
    _handle_started: bool = False
    _handle_completed: bool = False
    _claim_active: bool = False
    _claim_monitor_pid: int | None = None
    _claim_ssh_pid: int | None = None
    _claim_control_port: int | None = None
    _archive_latched: bool = False

    def _command_argv(self) -> tuple[str, ...]:
        return build_tunnel_argv(
            self.identity,
            local_port=self.local_port,
            remote_port=self.remote_port,
            ssh_executable=self.ssh_executable,
            ssh_args_prefix=self.ssh_args_prefix,
        )

    def _command_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                list(self._command_argv()),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def config_path(self) -> Path:
        base = self.owner_path if self.owner_path is not None else self.log_path
        return base.with_name("tunnel-config.json")

    @property
    def ack_path(self) -> Path | None:
        return (
            self.owner_path.with_name("tunnel-ack.json")
            if self.owner_path is not None
            else None
        )

    @property
    def heartbeat_path(self) -> Path | None:
        return (
            self.owner_path.with_name("tunnel-heartbeat.json")
            if self.owner_path is not None
            else None
        )

    @property
    def state_path(self) -> Path | None:
        return (
            self.owner_path.with_name("tunnel-state.json")
            if self.owner_path is not None
            else None
        )

    @property
    def starting_path(self) -> Path | None:
        return (
            self.owner_path.with_name("tunnel-starting.json")
            if self.owner_path is not None
            else None
        )

    @property
    def recovery_path(self) -> Path | None:
        return (
            self.owner_path.with_name("tunnel-recovery.json")
            if self.owner_path is not None
            else None
        )

    def _owned_record(self, path: Path | None, schema: str) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            return None
        try:
            value = _read_small_json(path, "tunnel ownership record")
        except BootstrapError:
            return None
        if (
            value.get("schema") != schema
            or value.get("run_id") != self.run_id
            or value.get("nonce") != self.nonce
            or value.get("command_sha256") != self._command_sha256()
            or value.get("argv") != list(self._command_argv())
            or value.get("host") != self.identity.host
            or value.get("user") != self.identity.user
            or value.get("port") != int(self.identity.port)
            or value.get("local_port") != int(self.local_port)
            or value.get("remote_port") != int(self.remote_port)
            or value.get("fingerprint") != self.expected_fingerprint
        ):
            return None
        return value

    def _owned_pids(self) -> tuple[int | None, int | None, int | None, bool]:
        if self._handle_completed:
            return None, None, None, False
        identity = self._handle_identity()
        if identity is not None:
            monitor_pid, ssh_pid, control_port = identity
        else:
            monitor_pid = (
                int(self.process.pid) if self.process is not None else self.monitor_pid
            )
            ssh_pid = self.attached_pid
            control_port = self.control_port
        proof = False
        for record in (
            self._owned_record(self.starting_path, TUNNEL_STARTING_SCHEMA),
            self._owned_record(self.owner_path, TUNNEL_SCHEMA),
        ):
            if record is None:
                continue
            record_monitor = record.get("monitor_pid")
            record_ssh = record.get("ssh_pid")
            record_control = record.get("control_port")
            if isinstance(record_monitor, bool) or not isinstance(record_monitor, int):
                continue
            if record_ssh is not None and (
                isinstance(record_ssh, bool)
                or not isinstance(record_ssh, int)
                or record_ssh <= 0
            ):
                continue
            if (
                isinstance(record_control, bool)
                or not isinstance(record_control, int)
                or not (1 <= record_control <= 65535)
            ):
                continue
            if identity is not None:
                # A stale handle must never borrow the fresh attempt's sidecar.
                expected_monitor, expected_ssh, expected_control = identity
                if record_monitor != expected_monitor:
                    continue
                if expected_ssh is not None and record_ssh != expected_ssh:
                    continue
                if expected_control is not None and record_control != expected_control:
                    continue
                self._update_claim(record_ssh, record_control)
                proof = True
                if record_ssh is not None:
                    ssh_pid = record_ssh
                control_port = record_control
            else:
                if self.process is not None and record_monitor != int(self.process.pid):
                    continue
                proof = True
                monitor_pid = record_monitor
                if record_ssh is not None:
                    ssh_pid = record_ssh
                control_port = record_control
            break
        return monitor_pid, ssh_pid, control_port, proof

    def _write_cleanup_state(
        self,
        *,
        monitor_pid: int | None,
        ssh_pid: int | None,
        reason: str,
        failure_stage: str | None = None,
        failure_code: str | None = None,
    ) -> None:
        if self.state_path is None:
            return
        payload = {
            "schema": TUNNEL_STATE_SCHEMA,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "status": "CLEANED",
            "cleanup_confirmed": True,
            "reason": reason,
            "monitor_pid": monitor_pid,
            "ssh_pid": ssh_pid,
            "at": time.time(),
        }
        if failure_stage is not None:
            payload["failure_stage"] = failure_stage
        if failure_code is not None:
            payload["failure_code"] = failure_code
        _atomic_json(self.state_path, payload)

    @staticmethod
    def _post_ack_failure_details(exc: BaseException) -> tuple[str, str]:
        message = str(exc).lower()
        if "terminal state" in message:
            return "post_ack.sidecars", "terminal_state"
        if "heartbeat" in message:
            return "post_ack.sidecars", "heartbeat"
        if "ack" in message:
            return "post_ack.sidecars", "ack"
        if "control" in message:
            return "post_ack.control", "control"
        if "owner" in message or "fingerprint" in message:
            return "post_ack.owner", "owner_identity"
        if "deadline" in message:
            return "post_ack", "deadline"
        return "post_ack.verify", "verification"

    def _write_recovery(
        self,
        *,
        monitor_pid: int | None,
        ssh_pid: int | None,
        control_port: int | None,
        reason: str,
    ) -> dict[str, Any] | None:
        if self.recovery_path is None:
            return None
        payload = {
            "schema": TUNNEL_RECOVERY_SCHEMA,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "status": "CLEANUP_UNCONFIRMED",
            "cleanup_confirmed": False,
            "reason": reason,
            "monitor_pid": monitor_pid,
            "ssh_pid": ssh_pid,
            "control_port": control_port,
            "host": self.identity.host,
            "user": self.identity.user,
            "port": int(self.identity.port),
            "destination": self.identity.destination,
            "fingerprint": self.expected_fingerprint,
            "local_port": int(self.local_port),
            "remote_port": int(self.remote_port),
            "command_sha256": self._command_sha256(),
            "argv": list(self._command_argv()),
            "at": time.time(),
        }
        _atomic_json(self.recovery_path, payload)
        return payload

    def _monitor_script(self) -> Path:
        if self.monitor_script is not None:
            return Path(self.monitor_script)
        return Path(__file__).resolve().parent / "v3_tunnel_child.py"

    def _config_payload(self) -> dict[str, Any]:
        if self.owner_path is None or self.ack_path is None or self.heartbeat_path is None:
            raise BootstrapError("tunnel owner paths are missing")
        if self.state_path is None:
            raise BootstrapError("tunnel state path is missing")
        if self.starting_path is None:
            raise BootstrapError("tunnel starting path is missing")
        argv = list(self._command_argv())
        return {
            "schema": TUNNEL_CONFIG_SCHEMA,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "host": self.identity.host,
            "user": self.identity.user,
            "port": int(self.identity.port),
            "identity_file": (
                str(self.identity.identity_file)
                if self.identity.identity_file is not None
                else None
            ),
            "known_hosts": str(self.identity.known_hosts),
            "global_known_hosts": (
                str(getattr(self.identity, "global_known_hosts"))
                if getattr(self.identity, "global_known_hosts", None) is not None
                else None
            ),
            "host_key_algorithms": getattr(self.identity, "host_key_algorithms", None),
            "fingerprint": self.expected_fingerprint,
            "local_port": int(self.local_port),
            "remote_port": int(self.remote_port),
            "owner": str(self.owner_path),
            "ack": str(self.ack_path),
            "heartbeat": str(self.heartbeat_path),
            "state": str(self.state_path),
            "starting": str(self.starting_path),
            "log": str(self.log_path),
            "argv": argv,
            "command_sha256": self._command_sha256(),
            "heartbeat_interval": float(self.heartbeat_interval),
            "ssh_executable": self.ssh_executable,
            "ssh_args_prefix": list(self.ssh_args_prefix),
        }

    def _canonical_record_paths(self) -> dict[str, Path]:
        assert self.owner_path is not None
        return {
            "owner": self.owner_path,
            "ack": self.owner_path.with_name("tunnel-ack.json"),
            "heartbeat": self.owner_path.with_name("tunnel-heartbeat.json"),
            "state": self.owner_path.with_name("tunnel-state.json"),
            "starting": self.owner_path.with_name("tunnel-starting.json"),
            "recovery": self.owner_path.with_name("tunnel-recovery.json"),
            "config": self.owner_path.with_name("tunnel-config.json"),
        }

    def _snapshot_file(self, path: Path | None, name: str) -> dict[str, Any] | None:
        if path is None:
            return None
        target = Path(path)
        # Gate ancestors and the final target before exists/isfile/stat/read.
        _assert_safe_path_chain(target, directory=False)
        try:
            info = os.lstat(target)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise BootstrapError(f"{name} is unreadable") from exc
        if not stat.S_ISREG(info.st_mode):
            raise BootstrapError(f"{name} is not a regular file")
        if info.st_size > MAX_TUNNEL_JSON_BYTES:
            raise BootstrapError(f"{name} exceeds the bound")
        try:
            with open(target, "rb") as handle:
                data = handle.read(MAX_TUNNEL_JSON_BYTES + 1)
        except OSError as exc:
            raise BootstrapError(f"{name} is unreadable") from exc
        if len(data) > MAX_TUNNEL_JSON_BYTES:
            raise BootstrapError(f"{name} exceeds the bound")
        return {
            "path": target,
            "name": name,
            "data": data,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    @staticmethod
    def _json_from_snapshot(snapshot: Mapping[str, Any], name: str) -> dict[str, Any]:
        try:
            value = json.loads(bytes(snapshot["data"]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapError(f"{name} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise BootstrapError(f"{name} must be an object")
        return value

    def _validate_recovery_config(self, config: Mapping[str, Any]) -> None:
        if self.expected_fingerprint is None:
            raise BootstrapError(
                "trusted fingerprint is missing; refusing tunnel recovery"
            )
        from scripts.p1 import v3_tunnel_child

        try:
            v3_tunnel_child.validate_config(config)
        except Exception as exc:
            raise BootstrapError("trusted tunnel config is invalid") from exc

        expected = {
            "run_id": self.run_id,
            "nonce": self.nonce,
            "host": self.identity.host,
            "user": self.identity.user,
            "port": int(self.identity.port),
            "local_port": int(self.local_port),
            "remote_port": int(self.remote_port),
            "fingerprint": self.expected_fingerprint,
            "command_sha256": self._command_sha256(),
            "argv": list(self._command_argv()),
            "ssh_executable": self.ssh_executable,
            "ssh_args_prefix": list(self.ssh_args_prefix),
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise BootstrapError(
                    f"trusted tunnel config {key} does not match this connection"
                )
        expected_paths = {
            "owner": self.owner_path,
            "ack": self.owner_path.with_name("tunnel-ack.json") if self.owner_path else None,
            "heartbeat": self.owner_path.with_name("tunnel-heartbeat.json") if self.owner_path else None,
            "state": self.owner_path.with_name("tunnel-state.json") if self.owner_path else None,
            "starting": self.owner_path.with_name("tunnel-starting.json") if self.owner_path else None,
            "log": self.log_path,
        }
        for key, value in expected_paths.items():
            if value is None or not _same_path(config.get(key, ""), value):
                raise BootstrapError(f"trusted tunnel config {key} path is invalid")
        identity_file = self.identity.identity_file
        configured_identity = config.get("identity_file")
        if identity_file is None:
            if configured_identity is not None:
                raise BootstrapError("trusted tunnel config identity_file is invalid")
        elif configured_identity is None or not _same_path(configured_identity, identity_file):
            raise BootstrapError("trusted tunnel config identity_file is invalid")
        if not _same_path(config.get("known_hosts", ""), self.identity.known_hosts):
            raise BootstrapError("trusted tunnel config known_hosts is invalid")
        identity_global = getattr(self.identity, "global_known_hosts", None)
        config_global = config.get("global_known_hosts")
        if identity_global is None:
            if config_global is not None:
                raise BootstrapError("trusted tunnel config global_known_hosts is invalid")
        elif config_global is None or not _same_path(config_global, identity_global):
            raise BootstrapError("trusted tunnel config global_known_hosts is invalid")
        if config.get("host_key_algorithms") != getattr(
            self.identity, "host_key_algorithms", None
        ):
            raise BootstrapError("trusted tunnel config host_key_algorithms is invalid")

    def _validate_owned_sidecar(
        self,
        record: Mapping[str, Any],
        *,
        schema: str,
        name: str,
        require_pids: bool = True,
        require_argv: bool = True,
        require_pid_field: bool = True,
        require_destination: bool = True,
        timestamp_field: str | None = None,
    ) -> tuple[int | None, int | None, int | None]:
        if record.get("schema") != schema:
            raise BootstrapError(f"tunnel {name} schema is invalid")
        if record.get("run_id") != self.run_id or record.get("nonce") != self.nonce:
            raise BootstrapError(f"tunnel {name} run/nonce mismatch")
        expected = {
            "host": self.identity.host,
            "user": self.identity.user,
            "port": int(self.identity.port),
            "local_port": int(self.local_port),
            "remote_port": int(self.remote_port),
            "fingerprint": self.expected_fingerprint,
            "command_sha256": self._command_sha256(),
        }
        if require_argv:
            expected["argv"] = list(self._command_argv())
        for key, value in expected.items():
            if record.get(key) != value:
                raise BootstrapError(f"tunnel {name} {key} mismatch")
        if require_destination:
            if record.get("destination") != self.identity.destination:
                raise BootstrapError(f"tunnel {name} destination mismatch")
        elif "destination" in record and record.get("destination") != self.identity.destination:
            raise BootstrapError(f"tunnel {name} destination mismatch")
        if timestamp_field is not None:
            _finite_timestamp(
                record.get(timestamp_field), f"tunnel {name} {timestamp_field}"
            )
        monitor_pid = record.get("monitor_pid")
        ssh_pid = record.get("ssh_pid")
        control_port = record.get("control_port")
        if require_pids:
            for label, pid in (("monitor_pid", monitor_pid), ("ssh_pid", ssh_pid)):
                if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                    raise BootstrapError(f"tunnel {name} {label} is invalid")
            if require_pid_field and record.get("pid") != ssh_pid:
                raise BootstrapError(f"tunnel {name} pid does not match ssh_pid")
            if (
                isinstance(control_port, bool)
                or not isinstance(control_port, int)
                or not (1 <= control_port <= 65535)
            ):
                raise BootstrapError(f"tunnel {name} control_port is invalid")
        return monitor_pid, ssh_pid, control_port

    def _validate_terminal_state(self, state: Mapping[str, Any]) -> tuple[int, int]:
        if state.get("schema") != TUNNEL_STATE_SCHEMA:
            raise BootstrapError("tunnel state schema is invalid")
        if state.get("run_id") != self.run_id or state.get("nonce") != self.nonce:
            raise BootstrapError("tunnel state run/nonce mismatch")
        if state.get("status") != "CLEANED":
            raise BootstrapError("tunnel state is not CLEANED")
        if state.get("cleanup_confirmed") is not True:
            raise BootstrapError("tunnel state cleanup is not confirmed")
        monitor_pid = state.get("monitor_pid")
        ssh_pid = state.get("ssh_pid")
        for label, pid in (("monitor_pid", monitor_pid), ("ssh_pid", ssh_pid)):
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise BootstrapError(f"tunnel state {label} is invalid")
        _finite_timestamp(state.get("at"), "tunnel state at")
        if "started_at" in state:
            _finite_timestamp(state["started_at"], "tunnel state started_at")
        if "returncode" in state and state["returncode"] is not None:
            if isinstance(state["returncode"], bool) or not isinstance(
                state["returncode"], int
            ):
                raise BootstrapError("tunnel state returncode is invalid")
        if "reason" in state and not isinstance(state["reason"], str):
            raise BootstrapError("tunnel state reason is invalid")
        return int(monitor_pid), int(ssh_pid)

    def _assert_no_incomplete_archive(self) -> None:
        assert self.owner_path is not None
        critical = self._archive_critical_path()
        if _lexists(critical):
            raise BootstrapError(
                "tunnel archive has a critical finalization marker; "
                "manual recovery required"
            )
        archive = self.owner_path.parent / "archive"
        if not _lexists(archive):
            return
        _assert_safe_path_chain(archive, directory=True)
        try:
            archive_info = os.lstat(archive)
        except OSError as exc:
            raise BootstrapError("tunnel archive path is unreadable") from exc
        if not stat.S_ISDIR(archive_info.st_mode):
            raise BootstrapError("tunnel archive path is not a directory")
        for child in archive.iterdir():
            if not child.name.startswith("owned-tunnel-"):
                continue
            _assert_safe_path_chain(child, directory=True)
            try:
                child_info = os.lstat(child)
            except OSError as exc:
                raise BootstrapError("tunnel archive entry is unreadable") from exc
            if not stat.S_ISDIR(child_info.st_mode):
                raise BootstrapError("tunnel archive entry is not a directory")
            for marker in (".incomplete", ".retiring", ".retirement_blocked"):
                marker_path = child / marker
                if _lexists(marker_path):
                    raise BootstrapError(
                        "previous tunnel archive is incomplete; manual recovery required"
                    )
            manifest = child / "manifest.json"
            if not _lexists(manifest):
                raise BootstrapError(
                    "previous tunnel archive has no manifest; manual recovery required"
                )
            snapshot = self._snapshot_file(manifest, "tunnel archive manifest")
            if snapshot is None:
                raise BootstrapError(
                    "previous tunnel archive has no manifest; manual recovery required"
                )
            payload = self._json_from_snapshot(snapshot, "tunnel archive manifest")
            if payload.get("status") != "COMPLETE":
                raise BootstrapError(
                    "previous tunnel archive is not complete; manual recovery required"
                )
            retired = child / ".retired"
            if not _lexists(retired):
                raise BootstrapError(
                    "previous tunnel archive has no retired marker; manual recovery required"
                )
            retired_snapshot = self._snapshot_file(retired, "tunnel archive retired marker")
            if retired_snapshot is None or retired_snapshot["size"] > 4096:
                raise BootstrapError(
                    "previous tunnel archive retired marker is invalid; "
                    "manual recovery required"
                )

    def _validate_recovery_bundle(self) -> dict[str, Any] | None:
        """Validate a prior completed terminal attempt before any mutation."""

        assert self.owner_path is not None
        paths = self._canonical_record_paths()
        snapshots = {
            name: self._snapshot_file(path, name)
            for name, path in paths.items()
        }
        owner = snapshots["owner"]
        state = snapshots["state"]
        ack = snapshots["ack"]
        heartbeat = snapshots["heartbeat"]
        starting = snapshots["starting"]
        recovery = snapshots["recovery"]
        config = snapshots["config"]

        if owner is None and state is None:
            if any(item is not None for item in (ack, heartbeat, starting, recovery)):
                raise BootstrapError(
                    "orphan tunnel sidecars exist without owner/state; "
                    "manual recovery required"
                )
            if config is not None:
                # A config-only record may be the result of a crash between
                # config publication and Popen/STARTING.  There is no proof
                # that a child was not spawned, so never overwrite it.
                raise BootstrapError(
                    "tunnel-config.json exists without owner/terminal state; "
                    "manual recovery required"
                )
            return None
        if config is None:
            raise BootstrapError(
                "trusted tunnel-config.json is missing; manual recovery required"
            )
        config_payload = self._json_from_snapshot(config, "tunnel config")
        self._validate_recovery_config(config_payload)

        # Terminal CLEANED/confirmed state is mandatory for every automatic
        # archive, including an owner-only crash.  Unknown cleanup is manual.
        if state is None:
            raise BootstrapError(
                "terminal tunnel-state.json is missing; manual recovery required"
            )
        state_payload = self._json_from_snapshot(state, "tunnel state")
        anchor_pids = self._validate_terminal_state(state_payload)

        owner_pids: tuple[int, int] | None = None
        control_ports: list[int] = []
        if owner is not None:
            owner_payload = self._json_from_snapshot(owner, "tunnel owner")
            monitor_pid, ssh_pid, control_port = self._validate_owned_sidecar(
                owner_payload,
                schema=TUNNEL_SCHEMA,
                name="owner",
                require_destination=True,
                timestamp_field="created_at",
            )
            owner_pids = (int(monitor_pid), int(ssh_pid))
            control_ports.append(int(control_port))
        if owner_pids is not None and owner_pids != anchor_pids:
            raise BootstrapError("tunnel owner and terminal state PIDs disagree")

        if ack is not None:
            ack_payload = self._json_from_snapshot(ack, "tunnel ack")
            if ack_payload.get("status") != "ARMED":
                raise BootstrapError("tunnel ack status is invalid")
            monitor_pid, ssh_pid, control_port = self._validate_owned_sidecar(
                ack_payload,
                schema=TUNNEL_ACK_SCHEMA,
                name="ack",
                require_destination=True,
                timestamp_field="ack_at",
            )
            if (int(monitor_pid), int(ssh_pid)) != anchor_pids:
                raise BootstrapError("tunnel ack PIDs do not match terminal state")
            control_ports.append(int(control_port))
        if heartbeat is not None:
            heartbeat_payload = self._json_from_snapshot(
                heartbeat, "tunnel heartbeat"
            )
            monitor_pid, ssh_pid, control_port = self._validate_owned_sidecar(
                heartbeat_payload,
                schema=TUNNEL_HEARTBEAT_SCHEMA,
                name="heartbeat",
                require_argv=False,
                require_destination=False,
                timestamp_field="at",
            )
            if (int(monitor_pid), int(ssh_pid)) != anchor_pids:
                raise BootstrapError("tunnel heartbeat PIDs do not match terminal state")
            control_ports.append(int(control_port))
        if starting is not None:
            starting_payload = self._json_from_snapshot(starting, "tunnel starting")
            if (
                starting_payload.get("status") != "CLEANED"
                or starting_payload.get("cleanup_confirmed") is not True
            ):
                raise BootstrapError(
                    "tunnel starting is not CLEANED/confirmed; "
                    "manual recovery required"
                )
            monitor_pid, ssh_pid, control_port = self._validate_owned_sidecar(
                starting_payload,
                schema=TUNNEL_STARTING_SCHEMA,
                name="starting",
                require_destination=True,
                timestamp_field="at",
            )
            if (int(monitor_pid), int(ssh_pid)) != anchor_pids:
                raise BootstrapError("tunnel starting PIDs do not match terminal state")
            control_ports.append(int(control_port))
        if recovery is not None:
            recovery_payload = self._json_from_snapshot(recovery, "tunnel recovery")
            if (
                recovery_payload.get("status") != "CLEANED"
                or recovery_payload.get("cleanup_confirmed") is not True
            ):
                raise BootstrapError(
                    "tunnel recovery is not CLEANED/confirmed; manual recovery required"
                )
            monitor_pid, ssh_pid, control_port = self._validate_owned_sidecar(
                recovery_payload,
                schema=TUNNEL_RECOVERY_SCHEMA,
                name="recovery",
                require_pid_field=False,
                require_destination=True,
                timestamp_field="at",
            )
            if (int(monitor_pid), int(ssh_pid)) != anchor_pids:
                raise BootstrapError("tunnel recovery PIDs do not match terminal state")
            control_ports.append(int(control_port))
        if len(set(control_ports)) > 1:
            raise BootstrapError(
                "tunnel sidecar control ports disagree; manual recovery required"
            )

        monitor_state = pid_state(anchor_pids[0])
        ssh_state = pid_state(anchor_pids[1])
        if monitor_state != PID_ABSENT or ssh_state != PID_ABSENT:
            raise BootstrapError(
                "tunnel recovery processes are live or liveness is unproven; "
                "no archive and no kill"
            )

        files = [
            item
            for item in (
                state,
                owner,
                ack,
                heartbeat,
                starting,
                recovery,
                config,
            )
            if item is not None
        ]
        return {
            "run_id": self.run_id,
            "nonce": self.nonce,
            "monitor_pid": anchor_pids[0],
            "ssh_pid": anchor_pids[1],
            "monitor_state": monitor_state,
            "ssh_state": ssh_state,
            "files": files,
            "config_sha256": config["sha256"],
        }

    def _archive_critical_path(self) -> Path:
        assert self.owner_path is not None
        return self.owner_path.parent / ".tunnel-archive-critical"

    def _handle_archive_finalize_failure(
        self, destination: Path, run_dir: Path, exc: BaseException
    ) -> None:
        """Restore a blocking marker if the last marker was already removed.

        A final fsync/error after ``.retiring`` removal must not leave a
        COMPLETE archive that looks safe on the next start.  Restoration is
        best effort; if it cannot be made durable, latch this object and leave a
        critical run-dir sentinel when possible.  This is not a claim of
        universal durability on a failing filesystem.
        """

        marker = destination / ".retirement_blocked"
        marker_gone = not _lexists(marker)
        retiring_gone = not _lexists(destination / ".retiring")
        if not marker_gone and not retiring_gone:
            return
        code = b"retirement_finalize_failed\n"
        restored = False
        if not _lexists(marker):
            try:
                _write_exclusive_bytes(marker, code)
                _fsync_directory(destination)
                restored = True
            except Exception:
                restored = False
        else:
            try:
                _fsync_directory(destination)
                restored = True
            except Exception:
                restored = False
        if restored:
            return
        self._archive_latched = True
        try:
            critical = self._archive_critical_path()
            _assert_safe_path_chain(critical, directory=False)
            if not _lexists(critical):
                _write_exclusive_bytes(critical, code)
                _fsync_directory(run_dir)
        except Exception:
            pass
        raise BootstrapError(
            "tunnel archive finalization failed and blocking marker could not "
            "be restored; manual recovery required; no spawn"
        ) from exc

    def _archive_recovery_bundle(
        self, proof: Mapping[str, Any], *, source: str
    ) -> Path:
        """Copy a fully validated bundle to an exclusive archive, then retire it."""

        assert self.owner_path is not None
        self._assert_no_incomplete_archive()
        run_dir = self.owner_path.parent
        _assert_safe_path_chain(run_dir, directory=True)
        archive_root = run_dir / "archive"
        _assert_safe_path_chain(archive_root, directory=True)
        archive_root.mkdir(parents=True, exist_ok=True)
        _assert_safe_path_chain(archive_root, directory=True)
        if not _same_path(archive_root.parent, run_dir):
            raise BootstrapError("tunnel archive is not a direct child of the run directory")
        destination = archive_root / (
            f"owned-tunnel-{_archive_stamp()}-{uuid.uuid4().hex}"
        )
        _assert_safe_path_chain(destination, directory=True)
        try:
            destination.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError as exc:
            raise BootstrapError("tunnel archive destination already exists") from exc
        _assert_safe_path_chain(destination, directory=True)
        try:
            _write_exclusive_bytes(destination / ".incomplete", b"archiving\n")
            entries: list[dict[str, Any]] = []
            for item in proof["files"]:
                canonical = Path(item["path"])
                _assert_safe_path_chain(canonical, directory=False)
                try:
                    current = canonical.read_bytes()
                except OSError as exc:
                    raise BootstrapError(
                        "tunnel recovery file disappeared during validation"
                    ) from exc
                digest = hashlib.sha256(current).hexdigest()
                if digest != item["sha256"]:
                    raise BootstrapError(
                        "tunnel recovery file changed during validation"
                    )
                archived = destination / canonical.name
                _write_exclusive_bytes(archived, current)
                if hashlib.sha256(archived.read_bytes()).hexdigest() != digest:
                    raise BootstrapError("tunnel archive copy verification failed")
                entries.append(
                    {
                        "name": str(item["name"]),
                        "source_path": str(canonical),
                        "source_sha256": digest,
                        "source_size": len(current),
                        "archive_name": archived.name,
                    }
                )
            log_entry = None
            if _lexists(self.log_path):
                _assert_safe_path_chain(self.log_path, directory=False)
                try:
                    log_info = os.lstat(self.log_path)
                except OSError as exc:
                    raise BootstrapError("tunnel log is unreadable") from exc
                if not stat.S_ISREG(log_info.st_mode):
                    raise BootstrapError("tunnel log is not a regular file")
                size = log_info.st_size
                with self.log_path.open("rb") as handle:
                    if size > MAX_TUNNEL_LOG_BYTES:
                        handle.seek(max(0, size - MAX_TUNNEL_ARCHIVE_LOG_TAIL))
                        log_bytes = handle.read(MAX_TUNNEL_ARCHIVE_LOG_TAIL)
                        log_entry = {
                            "archive_name": "tunnel.log.tail",
                            "copy": "tail_not_full",
                            "source_size": size,
                            "sha256": hashlib.sha256(log_bytes).hexdigest(),
                        }
                    else:
                        log_bytes = handle.read(MAX_TUNNEL_LOG_BYTES + 1)
                        if len(log_bytes) > MAX_TUNNEL_LOG_BYTES:
                            log_bytes = log_bytes[-MAX_TUNNEL_ARCHIVE_LOG_TAIL:]
                            log_entry = {
                                "archive_name": "tunnel.log.tail",
                                "copy": "tail_not_full",
                                "source_size": size,
                                "sha256": hashlib.sha256(log_bytes).hexdigest(),
                            }
                        else:
                            log_entry = {
                                "archive_name": "tunnel.log.current",
                                "copy": "full_current",
                                "source_size": size,
                                "sha256": hashlib.sha256(log_bytes).hexdigest(),
                            }
                _write_exclusive_bytes(destination / log_entry["archive_name"], log_bytes)
            manifest = {
                "schema": "anime-video-studio/p1-v3-tunnel-recovery-archive/v1",
                "status": "COMPLETE",
                "source": source,
                "run_id": proof["run_id"],
                "nonce": proof["nonce"],
                "monitor_pid": proof["monitor_pid"],
                "ssh_pid": proof["ssh_pid"],
                "monitor_liveness": proof["monitor_state"],
                "ssh_liveness": proof["ssh_state"],
                "checked_at": time.time(),
                "archive_created_at": datetime.now(timezone.utc).isoformat(),
                "files": entries,
                "log": log_entry,
            }
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            _write_exclusive_bytes(destination / "manifest.json", manifest_bytes)
            _write_exclusive_bytes(destination / ".retiring", b"retiring\n")
            _write_exclusive_bytes(destination / ".retired", b"retired\n")
            try:
                _fsync_directory(destination)
                _fsync_directory(archive_root)
                (destination / ".incomplete").unlink(missing_ok=True)
                for item in proof["files"]:
                    canonical = Path(item["path"])
                    if not _lexists(canonical):
                        continue
                    _assert_safe_path_chain(canonical, directory=False)
                    try:
                        current = canonical.read_bytes()
                    except OSError as exc:
                        raise BootstrapError(
                            "tunnel recovery file disappeared during retirement"
                        ) from exc
                    if hashlib.sha256(current).hexdigest() != item["sha256"]:
                        _write_exclusive_bytes(
                            destination / ".retirement_blocked",
                            b"canonical file changed during retirement\n",
                        )
                        raise BootstrapError(
                            "tunnel recovery file changed during retirement; "
                            "archive incomplete"
                        )
                    canonical.unlink()
                _fsync_directory(run_dir)
                # .retiring is the LAST blocking marker; removing it is the commit.
                (destination / ".retiring").unlink(missing_ok=True)
                _fsync_directory(destination)
            except Exception as exc:
                self._handle_archive_finalize_failure(destination, run_dir, exc)
                raise
            return destination
        except Exception:
            # Leave .incomplete/.retiring markers in place.  No silent second start.
            raise

    def _handle_identity(self) -> tuple[int, int | None, int | None] | None:
        if self._handle_completed:
            return None
        if self._claim_active:
            monitor_pid = self._claim_monitor_pid
            if (
                isinstance(monitor_pid, bool)
                or not isinstance(monitor_pid, int)
                or monitor_pid <= 0
            ):
                return None
            return monitor_pid, self._claim_ssh_pid, self._claim_control_port
        if not self._handle_started:
            return None
        monitor_pid = self.monitor_pid
        ssh_pid = self.attached_pid
        control_port = self.control_port
        if (
            isinstance(monitor_pid, bool)
            or not isinstance(monitor_pid, int)
            or monitor_pid <= 0
        ):
            return None
        return monitor_pid, ssh_pid, control_port

    def _claim_attempt(self, process: subprocess.Popen) -> None:
        """Capture this object's spawned monitor before ACK/ARMED."""

        self._claim_active = True
        self._claim_monitor_pid = int(process.pid)
        self._claim_ssh_pid = None
        self._claim_control_port = None
        self._handle_completed = False

    def _update_claim(self, ssh_pid: int | None, control_port: int | None) -> None:
        if not self._claim_active:
            return
        if isinstance(ssh_pid, int) and not isinstance(ssh_pid, bool) and ssh_pid > 0:
            self._claim_ssh_pid = int(ssh_pid)
        if (
            isinstance(control_port, int)
            and not isinstance(control_port, bool)
            and 1 <= control_port <= 65535
        ):
            self._claim_control_port = int(control_port)

    def _mark_handle_started(self) -> None:
        self._handle_started = True
        self._handle_completed = False
        self._claim_active = True
        if self._claim_monitor_pid is None and self.monitor_pid is not None:
            self._claim_monitor_pid = int(self.monitor_pid)
        self._update_claim(self.attached_pid, self.control_port)

    def _clear_handle_completed(self) -> None:
        self.process = None
        self.monitor_pid = None
        self.attached_pid = None
        self.control_port = None
        self._claim_active = False
        self._claim_monitor_pid = None
        self._claim_ssh_pid = None
        self._claim_control_port = None
        self._handle_started = False
        self._handle_completed = True

    def _attempt_identity_mismatch(
        self, identity: tuple[int, int | None, int | None]
    ) -> str | None:
        assert self.owner_path is not None
        paths = self._canonical_record_paths()
        for name, schema, timestamp_field in (
            ("owner", TUNNEL_SCHEMA, "created_at"),
            ("starting", TUNNEL_STARTING_SCHEMA, "at"),
        ):
            path = paths[name]
            if not _lexists(path):
                continue
            try:
                snapshot = self._snapshot_file(path, name)
                if snapshot is None:
                    return name
                payload = self._json_from_snapshot(snapshot, name)
                monitor_pid, ssh_pid, control_port = self._validate_owned_sidecar(
                    payload,
                    schema=schema,
                    name=name,
                    require_destination=True,
                    timestamp_field=timestamp_field,
                )
            except BootstrapError:
                return name
            expected_monitor, expected_ssh, expected_control = identity
            if int(monitor_pid) != int(expected_monitor):
                return name
            if expected_ssh is not None and int(ssh_pid) != int(expected_ssh):
                return name
            if expected_control is not None and int(control_port) != int(expected_control):
                return name
            return None
        return None

    def _assert_attempt_scope_for_write(
        self,
        monitor_pid: int,
        ssh_pid: int | None,
        control_port: int | None,
    ) -> None:
        """Refuse state/recovery writes if canonical records are a newer attempt."""

        assert self.owner_path is not None
        paths = self._canonical_record_paths()
        specs = (
            ("owner", TUNNEL_SCHEMA, True, True, True),
            ("ack", TUNNEL_ACK_SCHEMA, True, True, True),
            ("heartbeat", TUNNEL_HEARTBEAT_SCHEMA, False, True, False),
            ("starting", TUNNEL_STARTING_SCHEMA, True, True, True),
            ("recovery", TUNNEL_RECOVERY_SCHEMA, True, False, True),
        )
        for name, schema, require_argv, require_pid_field, require_destination in specs:
            path = paths[name]
            if not _lexists(path):
                continue
            snapshot = self._snapshot_file(path, name)
            if snapshot is None:
                raise BootstrapError(
                    f"canonical tunnel {name} vanished; refusing state/recovery write"
                )
            payload = self._json_from_snapshot(snapshot, name)
            rec_monitor, rec_ssh, rec_control = self._validate_owned_sidecar(
                payload,
                schema=schema,
                name=name,
                require_pids=False,
                require_argv=require_argv,
                require_pid_field=require_pid_field,
                require_destination=require_destination,
            )
            if (
                isinstance(rec_monitor, bool)
                or not isinstance(rec_monitor, int)
                or rec_monitor <= 0
            ):
                raise BootstrapError(
                    "canonical tunnel monitor identity is missing; "
                    "refusing state/recovery write"
                )
            if int(rec_monitor) != int(monitor_pid):
                raise BootstrapError(
                    "canonical tunnel records belong to another monitor; "
                    "refusing state/recovery write"
                )
            if rec_ssh is not None:
                if isinstance(rec_ssh, bool) or not isinstance(rec_ssh, int) or rec_ssh <= 0:
                    raise BootstrapError(
                        "canonical tunnel child identity is invalid; "
                        "refusing state/recovery write"
                    )
                if ssh_pid is None or int(rec_ssh) != int(ssh_pid):
                    raise BootstrapError(
                        "canonical tunnel records belong to another child; "
                        "refusing state/recovery write"
                    )
            if rec_control is not None:
                if (
                    isinstance(rec_control, bool)
                    or not isinstance(rec_control, int)
                    or not (1 <= rec_control <= 65535)
                ):
                    raise BootstrapError(
                        "canonical tunnel control identity is invalid; "
                        "refusing state/recovery write"
                    )
                if control_port is None or int(rec_control) != int(control_port):
                    raise BootstrapError(
                        "canonical tunnel records belong to another control port; "
                        "refusing state/recovery write"
                    )
        state_path = paths["state"]
        if _lexists(state_path):
            snapshot = self._snapshot_file(state_path, "state")
            if snapshot is None:
                raise BootstrapError(
                    "canonical tunnel state vanished; refusing state/recovery write"
                )
            payload = self._json_from_snapshot(snapshot, "tunnel state")
            state_pids = self._validate_terminal_state(payload)
            if state_pids[0] != int(monitor_pid):
                raise BootstrapError(
                    "canonical tunnel state belongs to another monitor; "
                    "refusing state/recovery write"
                )
            if ssh_pid is None or state_pids[1] != int(ssh_pid):
                raise BootstrapError(
                    "canonical tunnel state belongs to another child; "
                    "refusing state/recovery write"
                )

    def _remove_attempt_records(
        self, monitor_pid: int, ssh_pid: int, control_port: int
    ) -> None:
        """Remove only sidecars whose full metadata matches this attempt."""

        assert self.owner_path is not None
        paths = self._canonical_record_paths()
        specs = (
            ("owner", TUNNEL_SCHEMA, True, True, True),
            ("ack", TUNNEL_ACK_SCHEMA, True, True, True),
            ("heartbeat", TUNNEL_HEARTBEAT_SCHEMA, False, True, False),
            ("starting", TUNNEL_STARTING_SCHEMA, True, True, True),
            ("recovery", TUNNEL_RECOVERY_SCHEMA, True, False, True),
        )
        for name, schema, require_argv, require_pid_field, require_destination in specs:
            path = paths[name]
            if not _lexists(path):
                continue
            try:
                snapshot = self._snapshot_file(path, name)
                if snapshot is None:
                    continue
                payload = self._json_from_snapshot(snapshot, name)
                rec_monitor, rec_ssh, rec_control = self._validate_owned_sidecar(
                    payload,
                    schema=schema,
                    name=name,
                    require_argv=require_argv,
                    require_pid_field=require_pid_field,
                    require_destination=require_destination,
                )
            except BootstrapError:
                continue
            if (int(rec_monitor), int(rec_ssh), int(rec_control)) != (
                int(monitor_pid),
                int(ssh_pid),
                int(control_port),
            ):
                continue
            try:
                path.unlink()
            except OSError:
                continue

    def _reset_attempt_identity(self) -> None:
        self.process = None
        self.monitor_pid = None
        self.attached_pid = None
        self.control_port = None
        self._handle_started = False
        self._handle_completed = False
        self._claim_active = False
        self._claim_monitor_pid = None
        self._claim_ssh_pid = None
        self._claim_control_port = None

    def _validate_owner(self, owner: Mapping[str, Any]) -> None:
        expected_argv = list(self._command_argv())
        expected = {
            "schema": TUNNEL_SCHEMA,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "host": self.identity.host,
            "user": self.identity.user,
            "port": int(self.identity.port),
            "destination": self.identity.destination,
            "local_port": int(self.local_port),
            "remote_port": int(self.remote_port),
            "command_sha256": self._command_sha256(),
            "argv": expected_argv,
        }
        for key, value in expected.items():
            if owner.get(key) != value:
                raise BootstrapError(f"tunnel owner {key} does not match this run")
        if owner.get("fingerprint") != self.expected_fingerprint:
            raise BootstrapError("tunnel owner fingerprint does not match the trusted identity")
        _finite_timestamp(owner.get("created_at"), "tunnel owner created_at")
        control_port = owner.get("control_port")
        if (
            isinstance(control_port, bool)
            or not isinstance(control_port, int)
            or not (1 <= control_port <= 65535)
        ):
            raise BootstrapError("tunnel owner control port is invalid")
        monitor_pid = owner.get("monitor_pid")
        ssh_pid = owner.get("ssh_pid")
        if (
            isinstance(monitor_pid, bool)
            or not isinstance(monitor_pid, int)
            or monitor_pid <= 0
        ):
            raise BootstrapError("tunnel owner monitor pid is invalid")
        if isinstance(ssh_pid, bool) or not isinstance(ssh_pid, int) or ssh_pid <= 0:
            raise BootstrapError("tunnel owner ssh pid is invalid")
        if owner.get("pid") != ssh_pid:
            raise BootstrapError("tunnel owner pid does not match the SSH child")
        self.control_port = control_port
        self.monitor_pid = monitor_pid
        self.attached_pid = ssh_pid

    def _heartbeat_age(self, heartbeat: Mapping[str, Any], pid: int) -> float:
        if (
            heartbeat.get("schema") != TUNNEL_HEARTBEAT_SCHEMA
            or heartbeat.get("run_id") != self.run_id
            or heartbeat.get("nonce") != self.nonce
            or heartbeat.get("monitor_pid") != self.monitor_pid
            or heartbeat.get("ssh_pid") != pid
            or heartbeat.get("pid") != pid
            or heartbeat.get("control_port") != self.control_port
            or heartbeat.get("command_sha256") != self._command_sha256()
        ):
            raise BootstrapError("tunnel heartbeat identity mismatch")
        _finite_timestamp(heartbeat.get("at"), "tunnel heartbeat at")
        return time.time() - float(heartbeat["at"])

    def _validate_sidecars(self, *, freshness_seconds: float) -> dict[str, Any]:
        if self.owner_path is None or self.ack_path is None or self.heartbeat_path is None:
            raise BootstrapError("tunnel owner paths are missing")
        if not self.ack_path.is_file():
            raise BootstrapError("tunnel ack is missing")
        ack = _read_small_json(self.ack_path, "tunnel ack")
        if (
            ack.get("schema") != TUNNEL_ACK_SCHEMA
            or ack.get("status") != "ARMED"
            or ack.get("run_id") != self.run_id
            or ack.get("nonce") != self.nonce
            or ack.get("monitor_pid") != self.monitor_pid
            or ack.get("ssh_pid") != self.attached_pid
            or ack.get("pid") != self.attached_pid
            or ack.get("control_port") != self.control_port
            or ack.get("command_sha256") != self._command_sha256()
        ):
            raise BootstrapError("tunnel ack identity mismatch")
        _finite_timestamp(ack.get("ack_at"), "tunnel ack ack_at")
        if not self.heartbeat_path.is_file():
            raise BootstrapError("tunnel heartbeat is missing")
        heartbeat = _read_small_json(self.heartbeat_path, "tunnel heartbeat")
        age = self._heartbeat_age(heartbeat, int(self.attached_pid or 0))
        if age < -5.0 or age > freshness_seconds:
            raise BootstrapError("tunnel heartbeat is stale")
        if self.state_path is not None and self.state_path.is_file():
            # A malformed state is not silently treated as absent: that could
            # resurrect an old terminal attempt under the same run/nonce.
            state = _read_small_json(self.state_path, "tunnel state")
            if (
                state.get("run_id") == self.run_id
                and state.get("nonce") == self.nonce
            ):
                raise BootstrapError("tunnel monitor has recorded a terminal state")
        return ack

    def _control_request(
        self,
        request: Mapping[str, Any],
        *,
        deadline_monotonic: float | None,
        label: str = "tunnel control",
    ) -> dict[str, Any]:
        if self.control_port is None:
            raise BootstrapError("tunnel control port is unknown")
        now = time.monotonic()
        if deadline_monotonic is None:
            end = now + 5.0
        else:
            remaining = _require_remaining(deadline_monotonic, label) or 0.0
            end = min(now + 5.0, now + remaining)
        payload = json.dumps(dict(request), sort_keys=True).encode("utf-8") + b"\n"
        data = bytearray()
        try:
            with socket.create_connection(
                ("127.0.0.1", self.control_port),
                timeout=max(0.001, end - time.monotonic()),
            ) as sock:
                sock.sendall(payload)
                while len(data) <= MAX_TUNNEL_CONTROL_BYTES:
                    now = time.monotonic()
                    if now >= end:
                        raise BootstrapError("tunnel control deadline expired")
                    sock.settimeout(max(0.001, min(0.5, end - now)))
                    chunk = sock.recv(
                        min(1024, MAX_TUNNEL_CONTROL_BYTES + 1 - len(data))
                    )
                    if not chunk:
                        break
                    data.extend(chunk)
                    if b"\n" in chunk:
                        break
        except BootstrapError:
            raise
        except OSError as exc:
            raise BootstrapError("tunnel control connection failed") from exc
        if time.monotonic() > end:
            raise BootstrapError("tunnel control response arrived after the deadline")
        if len(data) > MAX_TUNNEL_CONTROL_BYTES:
            raise BootstrapError("tunnel control response exceeds the bound")
        try:
            value = json.loads(bytes(data).split(b"\n", 1)[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapError("tunnel control response is invalid") from exc
        if not isinstance(value, dict):
            raise BootstrapError("tunnel control response must be an object")
        return value

    def _challenge(
        self,
        *,
        deadline_monotonic: float | None,
        freshness_seconds: float,
        action: str = "status",
    ) -> dict[str, Any]:
        challenge = hashlib.sha256(os.urandom(32)).hexdigest()
        response = self._control_request(
            {
                "action": action,
                "challenge": challenge,
                "run_id": self.run_id,
                "nonce": self.nonce,
            },
            deadline_monotonic=deadline_monotonic,
        )
        expected = {
            "schema": TUNNEL_CONTROL_SCHEMA,
            "challenge": challenge,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "monitor_pid": self.monitor_pid,
            "ssh_pid": self.attached_pid,
            "pid": self.attached_pid,
            "control_port": self.control_port,
            "host": self.identity.host,
            "user": self.identity.user,
            "port": int(self.identity.port),
            "fingerprint": self.expected_fingerprint,
            "local_port": int(self.local_port),
            "remote_port": int(self.remote_port),
            "command_sha256": self._command_sha256(),
            "argv": list(self._command_argv()),
            "alive": True,
        }
        for key, value in expected.items():
            if response.get(key) != value:
                raise BootstrapError(f"tunnel control {key} mismatch")
        at = response.get("at")
        _finite_timestamp(at, "tunnel control timestamp")
        age = time.time() - float(at)
        if age < -5.0 or age > freshness_seconds:
            raise BootstrapError("tunnel control response is stale")
        return response

    @classmethod
    def attach(
        cls,
        *,
        identity: RemoteIdentity,
        owner_path: Path,
        run_id: str,
        nonce: str,
        expected_fingerprint: str | None = None,
        freshness_seconds: float = DEFAULT_TUNNEL_FRESHNESS_SECONDS,
        deadline_monotonic: float | None = None,
        ssh_executable: str = "ssh",
        ssh_args_prefix: Sequence[str] = (),
    ) -> "OwnedTunnel":
        _require_remaining(deadline_monotonic, "tunnel attach")
        freshness = _validate_positive(freshness_seconds, "tunnel freshness")
        owner = _read_small_json(Path(owner_path), "tunnel owner")
        if owner.get("schema") != TUNNEL_SCHEMA:
            raise BootstrapError("tunnel owner schema is invalid")
        if owner.get("run_id") != run_id or owner.get("nonce") != nonce:
            raise BootstrapError("tunnel owner run/nonce mismatch")
        if (
            owner.get("host") != identity.host
            or owner.get("user") != identity.user
            or owner.get("port") != int(identity.port)
            or owner.get("destination") != identity.destination
        ):
            raise BootstrapError("tunnel owner remote identity mismatch")
        fingerprint = owner.get("fingerprint")
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise BootstrapError("tunnel owner fingerprint mismatch")
        local_port = owner.get("local_port")
        remote_port = owner.get("remote_port")
        if (
            isinstance(local_port, bool)
            or not isinstance(local_port, int)
            or not (1 <= local_port <= 65535)
            or isinstance(remote_port, bool)
            or not isinstance(remote_port, int)
            or not (1 <= remote_port <= 65535)
            or local_port == remote_port
        ):
            raise BootstrapError("tunnel owner mapping is invalid")
        tunnel = cls(
            identity=identity,
            local_port=local_port,
            remote_port=remote_port,
            log_path=Path(owner_path).with_name("tunnel.log"),
            run_id=run_id,
            nonce=nonce,
            owner_path=Path(owner_path),
            expected_fingerprint=fingerprint,
            ssh_executable=ssh_executable,
            ssh_args_prefix=tuple(ssh_args_prefix),
        )
        tunnel._validate_owner(owner)
        if not _pid_live(int(tunnel.monitor_pid or 0)):
            raise BootstrapError("tunnel monitor process is not alive")
        tunnel._validate_sidecars(freshness_seconds=freshness)
        tunnel._challenge(
            deadline_monotonic=deadline_monotonic,
            freshness_seconds=freshness,
        )
        tunnel._mark_handle_started()
        return tunnel

    def _control_stop(
        self,
        *,
        control_port: int,
        monitor_pid: int,
        deadline_monotonic: float,
    ) -> int | None:
        """Stop the exact monitor before ACK and return its published SSH PID."""

        challenge = hashlib.sha256(os.urandom(32)).hexdigest()
        saved = self.control_port
        self.control_port = int(control_port)
        try:
            response = self._control_request(
                {
                    "action": "stop",
                    "challenge": challenge,
                    "run_id": self.run_id,
                    "nonce": self.nonce,
                },
                deadline_monotonic=deadline_monotonic,
                label="tunnel cleanup",
            )
        finally:
            self.control_port = saved
        if (
            response.get("schema") != TUNNEL_CONTROL_SCHEMA
            or response.get("challenge") != challenge
            or response.get("run_id") != self.run_id
            or response.get("nonce") != self.nonce
            or response.get("monitor_pid") != int(monitor_pid)
            or response.get("control_port") != int(control_port)
            or response.get("command_sha256") != self._command_sha256()
        ):
            raise BootstrapError("tunnel cleanup control identity mismatch")
        ssh_pid = response.get("ssh_pid")
        if isinstance(ssh_pid, int) and not isinstance(ssh_pid, bool) and ssh_pid > 0:
            return int(ssh_pid)
        return None

    def _cleanup_attempt(
        self,
        process: subprocess.Popen,
        *,
        reason: str,
        deadline_monotonic: float | None,
        failure_stage: str | None = None,
        failure_code: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Best-effort bounded cleanup of exactly this owned attempt.

        Control stop is used when the monitor published its control port. The
        fallback terminates only the exact PIDs proven by the STARTING/owner
        sidecars. Ownership files are removed only after both processes are
        confirmed dead; otherwise a recovery record is written and retained.
        """

        if deadline_monotonic is None:
            timeout = 5.0
        else:
            timeout = max(0.05, min(5.0, deadline_monotonic - time.monotonic()))
        monitor_pid, ssh_pid, control_port, proof = self._owned_pids()
        monitor_pid = int(process.pid)
        # The monitor publishes the SSH PID just after spawn. A parent that
        # timed out between spawn and ACK must give that bounded record a
        # moment to appear before deciding cleanup is unconfirmed.
        publish_end = time.monotonic() + min(0.75, max(0.05, timeout / 2))
        while time.monotonic() < publish_end and ssh_pid is None:
            time.sleep(0.05)
            _mon, ssh_pid, control_port, proof = self._owned_pids()
            monitor_pid = int(process.pid)
        details: dict[str, Any] = {
            "reason": reason,
            "monitor_pid": monitor_pid,
            "ssh_pid": ssh_pid,
            "control_port": control_port,
            "ownership_proof": proof,
        }
        if proof and control_port:
            try:
                stopped_ssh_pid = self._control_stop(
                    control_port=int(control_port),
                    monitor_pid=int(monitor_pid),
                    deadline_monotonic=time.monotonic() + timeout,
                )
            except BootstrapError:
                stopped_ssh_pid = None
            if stopped_ssh_pid is not None:
                ssh_pid = stopped_ssh_pid
                details["ssh_pid"] = ssh_pid
        end = time.monotonic() + (timeout if control_port else min(0.5, timeout))
        while time.monotonic() < end:
            if ssh_pid is None:
                _mon, discovered, _control, _proof = self._owned_pids()
                if discovered is not None:
                    ssh_pid = discovered
                    details["ssh_pid"] = ssh_pid
            monitor_dead = process.poll() is not None or not _pid_live(int(monitor_pid))
            ssh_dead = ssh_pid is None or not _pid_live(int(ssh_pid))
            if monitor_dead and ssh_dead:
                break
            time.sleep(0.05)
        kill_timeout = max(2.0, min(5.0, timeout))
        if proof:
            _terminate_owned_pids(
                monitor_pid,
                ssh_pid,
                timeout=kill_timeout,
                monitor_process=process,
            )
        else:
            _terminate_pid(int(monitor_pid), timeout=kill_timeout)
        # The saved Popen is authoritative for our own monitor: after a
        # successful poll/wait no extra signal is sent to a POSIX zombie.
        monitor_dead = process.poll() is not None or not _pid_live(int(monitor_pid))
        # Without a published SSH PID we cannot prove the child is gone.
        ssh_dead = ssh_pid is not None and not _pid_live(int(ssh_pid))
        confirmed = monitor_dead and ssh_dead
        details["monitor_dead"] = monitor_dead
        details["ssh_dead"] = ssh_dead
        details["cleanup_confirmed"] = confirmed
        if confirmed:
            self._assert_attempt_scope_for_write(
                int(monitor_pid), int(ssh_pid), control_port
            )
            self._remove_attempt_records(
                int(monitor_pid), int(ssh_pid), int(control_port or 0)
            )
            self._write_cleanup_state(
                monitor_pid=monitor_pid,
                ssh_pid=ssh_pid,
                reason=reason,
                failure_stage=failure_stage,
                failure_code=failure_code,
            )
            # A confirmed cleanup of a failed start closes the handle exactly
            # like a successful stop; a later stop must not write an old state.
            self._clear_handle_completed()
        else:
            self._assert_attempt_scope_for_write(
                int(monitor_pid), ssh_pid, control_port
            )
            self._write_recovery(
                monitor_pid=monitor_pid,
                ssh_pid=ssh_pid,
                control_port=control_port,
                reason=reason,
            )
        return confirmed, details


    def _wait_for_ack(
        self,
        process: subprocess.Popen,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        assert self.ack_path is not None and self.heartbeat_path is not None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise BootstrapError("tunnel monitor exited before publishing ack")
            if self.ack_path.is_file():
                try:
                    ack = _read_small_json(self.ack_path, "tunnel ack")
                except BootstrapError:
                    ack = None
                if ack is not None:
                    if (
                        ack.get("schema") == TUNNEL_ACK_SCHEMA
                        and ack.get("status") == "ARMED"
                        and ack.get("run_id") == self.run_id
                        and ack.get("nonce") == self.nonce
                        and isinstance(ack.get("monitor_pid"), int)
                        and ack.get("monitor_pid") > 0
                        and isinstance(ack.get("ssh_pid"), int)
                        and ack.get("ssh_pid") > 0
                        and ack.get("pid") == ack.get("ssh_pid")
                        and isinstance(ack.get("control_port"), int)
                        and 1 <= ack.get("control_port") <= 65535
                        and ack.get("command_sha256") == self._command_sha256()
                        and ack.get("fingerprint") == self.expected_fingerprint
                        and ack.get("argv") == list(self._command_argv())
                    ):
                        if self.heartbeat_path.is_file():
                            try:
                                heartbeat = _read_small_json(
                                    self.heartbeat_path, "tunnel heartbeat"
                                )
                            except BootstrapError:
                                heartbeat = None
                            if heartbeat is not None:
                                self.monitor_pid = int(ack["monitor_pid"])
                                self.control_port = int(ack["control_port"])
                                self.attached_pid = int(ack["ssh_pid"])
                                age = self._heartbeat_age(
                                    heartbeat, int(ack["ssh_pid"])
                                )
                                if -5.0 <= age <= DEFAULT_TUNNEL_FRESHNESS_SECONDS:
                                    return ack
            time.sleep(0.05)
        if process.poll() is not None:
            raise BootstrapError("tunnel monitor exited before publishing ack")
        raise BootstrapError("tunnel monitor ack timed out")

    def start(
        self,
        *,
        deadline_monotonic: float | None = None,
        ack_timeout: float = 20.0,
        lifecycle_lock: Any | None = None,
    ) -> "OwnedTunnel":
        _require_remaining(deadline_monotonic, "tunnel start")
        ack_wait = _validate_positive(ack_timeout, "tunnel ack timeout")
        if self.owner_path is None:
            raise BootstrapError("tunnel start requires an owner record path")
        if self._archive_latched:
            raise BootstrapError(
                "tunnel archive is in critical manual recovery state; no spawn"
            )
        _validate_positive(self.heartbeat_interval, "tunnel heartbeat interval")
        held_lock = lifecycle_lock
        acquired = False
        if held_lock is None:
            held_lock = acquire_tunnel_lifecycle_lock(self.owner_path)
            acquired = True
        try:
            if self.process is not None and self.process.poll() is None:
                raise BootstrapError(
                    "tunnel monitor process is already live; no second start"
                )
            self._assert_no_incomplete_archive()
            prior = self._validate_recovery_bundle()
            # Port ownership is checked before any canonical old artifact is
            # archived.  A busy port therefore leaves the old record untouched.
            ensure_local_port_free("127.0.0.1", self.local_port)
            if prior is not None:
                try:
                    self._archive_recovery_bundle(prior, source="start_preflight")
                except BootstrapError:
                    raise
                except OSError as exc:
                    raise BootstrapError(
                        "tunnel archive failed; canonical files retained; "
                        "manual recovery required"
                    ) from exc
            elif _lexists(self.owner_path):
                raise BootstrapError(
                    "tunnel owner record appeared during preflight; no second start"
                )
            # Only after a validated archive/absence proof may cached identity
            # from the previous attempt be discarded.
            self._reset_attempt_identity()
            # Gate every managed write path before config/log creation.
            _assert_safe_path_chain(self.owner_path.parent, directory=True)
            _assert_safe_path_chain(self.config_path, directory=False)
            _assert_safe_path_chain(self.log_path, directory=False)
            config = self._config_payload()
            _atomic_json(self.config_path, config)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if _lexists(self.log_path):
                    log_info = os.lstat(self.log_path)
                    if not stat.S_ISREG(log_info.st_mode):
                        raise BootstrapError("tunnel log is not a regular file")
                    if log_info.st_size > MAX_TUNNEL_LOG_BYTES:
                        os.replace(
                            self.log_path,
                            self.log_path.with_name(self.log_path.name + ".1"),
                        )
            except BootstrapError:
                raise
            except OSError:
                pass
            handle = self.log_path.open("ab")
            try:
                launcher = self.popen or subprocess.Popen
                kwargs: dict[str, Any] = {
                    "stdout": handle,
                    "stderr": subprocess.STDOUT,
                    "stdin": subprocess.DEVNULL,
                    "close_fds": True,
                }
                if os.name == "nt":
                    kwargs["creationflags"] = (
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        | getattr(subprocess, "DETACHED_PROCESS", 0)
                    )
                else:
                    kwargs["start_new_session"] = True
                process = launcher(
                    [
                        str(self._python_executable()),
                        str(self._monitor_script()),
                        "--config",
                        str(self.config_path),
                    ],
                    **kwargs,
                )
            except BaseException:
                # A custom launcher may have spawned before raising.  Never
                # delete the config without proof that no child exists; a
                # subsequent start will fail closed on config-only and require
                # manual recovery.
                handle.close()
                raise
            finally:
                try:
                    handle.close()
                except OSError:
                    pass
            self.process = process
            self.monitor_pid = int(process.pid)
            # Capture the exact spawned monitor before ACK.  This is a cleanup
            # claim, not proof of ARMED ownership.
            self._claim_attempt(process)
            if deadline_monotonic is not None:
                ack_wait = min(
                    ack_wait, max(0.05, deadline_monotonic - time.monotonic())
                )
            try:
                ack = self._wait_for_ack(
                    process, deadline=time.monotonic() + ack_wait
                )
                self._update_claim(self.attached_pid, self.control_port)
            except BaseException as exc:
                confirmed, _details = self._cleanup_attempt(
                    process,
                    reason=f"ack:{type(exc).__name__}",
                    deadline_monotonic=deadline_monotonic,
                )
                if not confirmed:
                    raise BootstrapError(
                        "tunnel monitor start failed and cleanup is unconfirmed; "
                        f"recovery record: {self.recovery_path}; no blind kill"
                    ) from exc
                raise
            # The monitor is the owner; verify the same sidecars and challenge it
            # directly instead of trusting the ack alone.
            try:
                owner = _read_small_json(self.owner_path, "tunnel owner")
                self._validate_owner(owner)
                self._validate_sidecars(
                    freshness_seconds=DEFAULT_TUNNEL_FRESHNESS_SECONDS
                )
                self._challenge(
                    deadline_monotonic=deadline_monotonic,
                    freshness_seconds=DEFAULT_TUNNEL_FRESHNESS_SECONDS,
                )
            except BaseException as exc:
                failure_stage, failure_code = self._post_ack_failure_details(exc)
                confirmed, _details = self._cleanup_attempt(
                    process,
                    reason=f"post-ack:{type(exc).__name__}",
                    deadline_monotonic=deadline_monotonic,
                    failure_stage=failure_stage,
                    failure_code=failure_code,
                )
                if not confirmed:
                    raise BootstrapError(
                        "tunnel monitor ack verification failed and cleanup is "
                        f"unconfirmed; recovery record: {self.recovery_path}; no blind kill"
                    ) from exc
                raise
            self.attached_pid = int(ack["ssh_pid"])
            self._mark_handle_started()
            return self
        finally:
            if acquired:
                held_lock.release()

    def _python_executable(self) -> str:
        # A venv ``python.exe`` may be a launcher stub whose PID differs from
        # the interpreter that runs the monitor. The monitor only needs the
        # standard library, so prefer the trusted base interpreter.
        base = getattr(sys, "_base_executable", None)
        if isinstance(base, str) and Path(base).is_file():
            return base
        return sys.executable

    def verify_mapping(
        self,
        *,
        deadline_monotonic: float | None = None,
        freshness_seconds: float = DEFAULT_TUNNEL_FRESHNESS_SECONDS,
    ) -> None:
        freshness = _validate_positive(freshness_seconds, "tunnel freshness")
        _require_remaining(deadline_monotonic, "tunnel mapping")
        if self.process is not None:
            if self.process.poll() is not None:
                raise BootstrapError("tunnel monitor process is not alive")
            if self.monitor_pid != int(self.process.pid):
                raise BootstrapError("tunnel monitor pid changed")
        pid = self.monitor_pid
        if isinstance(pid, bool) or not isinstance(pid, int) or not _pid_live(pid):
            raise BootstrapError("recorded tunnel monitor is not alive")
        if self.owner_path is None:
            raise BootstrapError("tunnel owner record path is missing")
        owner = _read_small_json(self.owner_path, "tunnel owner")
        self._validate_owner(owner)
        self._validate_sidecars(freshness_seconds=freshness)
        self._challenge(
            deadline_monotonic=deadline_monotonic,
            freshness_seconds=freshness,
        )

    def readback(
        self,
        *,
        deadline_monotonic: float | None = None,
        freshness_seconds: float = DEFAULT_TUNNEL_FRESHNESS_SECONDS,
    ) -> dict[str, Any]:
        """Verify and return the fresh owned tunnel records.

        The challenge is part of the readback, so a stale heartbeat file or a
        forged PID record is never accepted as ownership.
        """

        freshness = _validate_positive(freshness_seconds, "tunnel freshness")
        _require_remaining(deadline_monotonic, "tunnel readback")
        self.verify_mapping(
            deadline_monotonic=deadline_monotonic,
            freshness_seconds=freshness,
        )
        assert self.owner_path is not None
        assert self.ack_path is not None
        assert self.heartbeat_path is not None
        return {
            "owner": _read_small_json(self.owner_path, "tunnel owner"),
            "ack": _read_small_json(self.ack_path, "tunnel ack"),
            "heartbeat": _read_small_json(
                self.heartbeat_path, "tunnel heartbeat"
            ),
        }

    def health(
        self,
        *,
        timeout: float = 30.0,
        deadline_monotonic: float | None = None,
    ) -> dict:
        """Bounded ComfyUI health through the owned tunnel.

        ``deadline_monotonic`` is an absolute monotonic limit for the whole
        call. Each HTTP attempt is already watchdog-bounded by the shared
        ``v3_backend`` loopback HTTP helper, so a trickled response cannot keep
        the call alive past the total deadline.
        """

        timeout_value = _validate_positive(timeout, "health timeout")
        remaining = _require_remaining(deadline_monotonic, "tunnel health")
        end = time.monotonic() + (
            timeout_value if remaining is None else min(timeout_value, remaining)
        )
        url = f"http://127.0.0.1:{self.local_port}/system_stats"
        while True:
            now = time.monotonic()
            if now >= end:
                break
            try:
                self.verify_mapping(
                    deadline_monotonic=end,
                    freshness_seconds=DEFAULT_TUNNEL_FRESHNESS_SECONDS,
                )
                from scripts.p1 import v3_backend

                data = v3_backend._json_request(
                    base_url=f"http://127.0.0.1:{self.local_port}",
                    method="GET",
                    path="/system_stats",
                    timeout=max(0.001, min(2.0, end - time.monotonic())),
                    max_bytes=256 * 1024,
                )
                if isinstance(data, dict) and data.get("system"):
                    return data
            except Exception:
                pass
            sleep_for = min(0.2, max(0.0, end - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)
        raise BootstrapError(f"ComfyUI/tunnel health timed out at {url}")

    def stop(self, *, timeout: float = 5.0) -> bool:
        if self._handle_completed:
            return True
        lock = None
        if self.owner_path is not None:
            try:
                lock = acquire_tunnel_lifecycle_lock(self.owner_path)
            except BootstrapError:
                # A concurrent start/archive owns the canonical paths.  Do not
                # kill or retire anything from this stale object.
                return False
        try:
            return self._stop_locked(timeout=timeout)
        finally:
            if lock is not None:
                lock.release()

    def _stop_locked(self, *, timeout: float = 5.0) -> bool:
        timeout_value = _validate_positive(timeout, "tunnel stop timeout")
        if self._handle_completed:
            return True
        identity = self._handle_identity()
        if identity is not None:
            if self.owner_path is None or not _lexists(self.owner_path):
                # Terminal state/no owner is left for the next preflight.
                self._clear_handle_completed()
                return True
            mismatch = self._attempt_identity_mismatch(identity)
            if mismatch is not None:
                raise BootstrapError(
                    f"stale tunnel handle does not own current {mismatch} record; "
                    "no cleanup or signal"
                )
        monitor_pid, ssh_pid, control_port, proof = self._owned_pids()
        if monitor_pid is None and ssh_pid is None:
            return True
        control_accepted = not (proof and control_port and ssh_pid)
        if proof and control_port and ssh_pid:
            saved = (self.control_port, self.monitor_pid, self.attached_pid)
            self.control_port = int(control_port)
            self.monitor_pid = int(monitor_pid) if monitor_pid is not None else None
            self.attached_pid = int(ssh_pid)
            try:
                self._challenge(
                    deadline_monotonic=time.monotonic() + timeout_value,
                    freshness_seconds=DEFAULT_TUNNEL_FRESHNESS_SECONDS,
                    action="stop",
                )
                control_accepted = True
            except BootstrapError:
                control_accepted = False
            finally:
                self.control_port, self.monitor_pid, self.attached_pid = saved
        if not control_accepted:
            if (
                self.process is None
                or self.process.poll() is not None
                or int(self.process.pid) != int(monitor_pid or 0)
            ):
                # Never signal an attached/stale handle or a reused PID whose
                # control identity did not answer.  The next preflight will
                # archive only after full provenance and both-PID absence proof.
                return False
        end = time.monotonic() + timeout_value
        while time.monotonic() < end:
            if (
                self.process is None
                and monitor_pid is not None
                and _pid_live(int(monitor_pid))
            ):
                _reap_child(int(monitor_pid))
            monitor_dead = (
                (self.process.poll() is not None)
                if self.process is not None
                else (monitor_pid is None or not _pid_live(int(monitor_pid)))
            )
            ssh_dead = ssh_pid is None or not _pid_live(int(ssh_pid))
            if monitor_dead and ssh_dead:
                break
            time.sleep(0.05)
        if proof:
            _terminate_owned_pids(
                monitor_pid,
                ssh_pid,
                timeout=timeout_value,
                monitor_process=self.process,
            )
        elif self.process is not None and monitor_pid is not None:
            # This monitor was spawned by this object, so killing it is safe,
            # but without a published SSH PID the cleanup is not confirmed.
            if self.process.poll() is None:
                _terminate_pid(int(monitor_pid), timeout=timeout_value)
            try:
                self.process.wait(timeout=timeout_value)
            except Exception:
                pass
        monitor_dead = (
            self.process.poll() is not None
            if self.process is not None
            else (monitor_pid is None or not _pid_live(int(monitor_pid)))
        )
        ssh_dead = ssh_pid is not None and not _pid_live(int(ssh_pid))
        confirmed = monitor_dead and ssh_dead
        if confirmed:
            self._assert_attempt_scope_for_write(
                int(monitor_pid), int(ssh_pid), control_port
            )
            self._remove_attempt_records(
                int(monitor_pid), int(ssh_pid), int(control_port or 0)
            )
            self._write_cleanup_state(
                monitor_pid=monitor_pid, ssh_pid=ssh_pid, reason="stop"
            )
            self._clear_handle_completed()
            return True
        self._assert_attempt_scope_for_write(
            int(monitor_pid), ssh_pid, control_port
        )
        self._write_recovery(
            monitor_pid=monitor_pid,
            ssh_pid=ssh_pid,
            control_port=control_port,
            reason="stop",
        )
        return False


def claim_owned_root(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_root: str,
    run_id: str,
    nonce: str,
    session: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Create/reuse an exact run-owned root claim or refuse a foreign root."""

    _require_remaining(deadline_monotonic, "owned root claim")
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if not isinstance(run_id, str) or not run_id or not isinstance(nonce, str) or not nonce:
        raise BootstrapError("owned root claim identity is invalid")
    payload = {
        "schema": ROOT_CLAIM_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "session": dict(session) if session is not None else None,
    }
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False, newline="\n"
    )
    local = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        try:
            runner.put_file(
                identity,
                str(PurePosixPath(root) / ".p1-owned-root.json"),
                local,
                root=root,
                allow_same_hash=True,
                timeout=timeout,
                deadline_monotonic=deadline_monotonic,
            )
        except Exception as exc:
            raise BootstrapError(
                "remote root is claimed by another run/nonce or cannot be claimed"
            ) from exc
    finally:
        local.unlink(missing_ok=True)
    return payload


def prepare_remote_root(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_root: str,
    timeout: float = 30.0,
    deadline_monotonic: float | None = None,
    run_id: str | None = None,
    nonce: str | None = None,
    session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_remaining(deadline_monotonic, "remote root")
    if timeout <= 0:
        raise BootstrapError("remote root timeout is invalid")
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    mkdir = runner.run_remote(
        identity,
        ("mkdir", "-p", root, str(PurePosixPath(root) / "scripts" / "p1")),
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )
    if mkdir.returncode != 0 or getattr(mkdir, "timed_out", False):
        raise BootstrapError("remote root creation failed")
    df = runner.run_remote(
        identity,
        ("df", "-Pk", root),
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )
    if df.returncode != 0 or getattr(df, "timed_out", False):
        raise BootstrapError("remote disk check failed")
    try:
        available_kb = int([line.split() for line in df.stdout.splitlines() if line.strip()][-1][3])
    except (IndexError, ValueError) as exc:
        raise BootstrapError("remote disk output is invalid") from exc
    if available_kb <= 0:
        raise BootstrapError("remote disk is unavailable")
    if run_id is not None and nonce is not None:
        claim_owned_root(
            runner=runner,
            identity=identity,
            remote_root=root,
            run_id=run_id,
            nonce=nonce,
            session=session,
            timeout=timeout,
            deadline_monotonic=deadline_monotonic,
        )
    return {"status": "ROOT_READY", "remote_root": root, "disk_kb": available_kb}


def render_bootstrap_setup(setup_text: str) -> str:
    """Remove the manifest download step from the bootstrap script.

    The focused download worker is the sole owner of the three manifest
    downloads, target lock, progress and download deadline.  Running the same
    three downloads from ``setup.sh`` first would bypass that ownership.
    """

    if not isinstance(setup_text, str) or not setup_text.strip():
        raise BootstrapError("bootstrap setup text is missing")
    kept: list[str] = []
    for line in setup_text.splitlines():
        stripped = line.strip()
        if "setup_plan.py" in stripped and "--execute" in stripped:
            continue
        if stripped.startswith("# Model downloads are exactly"):
            continue
        kept.append(line)
    rendered = "\n".join(kept) + "\n"
    if "setup_plan.py" in rendered and "--execute" in rendered:
        raise BootstrapError("bootstrap setup still contains a download step")
    return rendered


def stage_bootstrap(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_root: str,
    setup_text: str,
    preflight_text: str,
    launch_manifest: Mapping[str, Any] | None = None,
    timeout: float = 60.0,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    _require_remaining(deadline_monotonic, "bootstrap staging")
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if timeout <= 0:
        raise BootstrapError("staging timeout is invalid")
    root_path = PurePosixPath(root)
    bootstrap_setup = render_bootstrap_setup(setup_text)
    local_setup = Path(tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8").name)
    local_preflight = Path(tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8").name)
    try:
        local_setup.write_text(bootstrap_setup, encoding="utf-8", newline="\n")
        local_preflight.write_text(preflight_text, encoding="utf-8", newline="\n")
        runner.put_file(identity, str(root_path / "setup.sh"), local_setup, root=root, allow_same_hash=True, timeout=timeout, deadline_monotonic=deadline_monotonic)
        runner.put_file(
            identity,
            str(root_path / "preflight.txt"),
            local_preflight,
            root=root,
            allow_same_hash=True,
            timeout=timeout,
            deadline_monotonic=deadline_monotonic,
        )
        for name in (
            "setup_plan.py",
            "preflight.py",
            "v3_preflight_runner.py",
            "v3_remote_fetch_helper.py",
            "v3_remote_launcher.py",
            "v3_runtime_probe.py",
        ):
            runner.put_file(
                identity,
                str(root_path / "scripts" / "p1" / name),
                Path(__file__).resolve().parent / name,
                root=root,
                allow_same_hash=True,
                timeout=timeout,
                deadline_monotonic=deadline_monotonic,
            )
        if launch_manifest is not None:
            handle = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", delete=False, newline="\n"
            )
            local_launch = Path(handle.name)
            try:
                with handle:
                    handle.write(json.dumps(dict(launch_manifest), indent=2, sort_keys=True) + "\n")
                runner.put_file(
                    identity,
                    str(root_path / "comfy-launch.json"),
                    local_launch,
                    root=root,
                    allow_same_hash=True,
                    timeout=timeout,
                    deadline_monotonic=deadline_monotonic,
                )
            finally:
                local_launch.unlink(missing_ok=True)
    finally:
        local_setup.unlink(missing_ok=True)
        local_preflight.unlink(missing_ok=True)
    return {"status": "STAGED", "remote_root": root}


def run_setup(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_root: str,
    timeout: float,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    _require_remaining(deadline_monotonic, "setup")
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if timeout <= 0:
        raise BootstrapError("setup timeout is invalid")
    # The owned root is passed as an argv element, never interpolated into a
    # shell command. The fixed Python runner chdirs and execs ``bash setup.sh``
    # so spaces/semicolons in the path cannot become shell syntax.
    result = runner.run_remote(
        identity,
        ("python3", "-c", _SETUP_RUNNER, root),
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )
    if result.returncode != 0 or getattr(result, "timed_out", False):
        raise BootstrapError("setup/base-torch execution failed")
    return {"status": "SETUP_COMPLETE", "remote_root": root}


def run_preflight(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_root: str,
    workspace: str,
    timeout: float,
    mode: str = "api",
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    _require_remaining(deadline_monotonic, "preflight")
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if timeout <= 0:
        raise BootstrapError("preflight timeout is invalid")
    if mode not in {"offline", "api"}:
        raise BootstrapError("preflight mode is invalid")
    runner_path = str(PurePosixPath(root) / "scripts" / "p1" / "v3_preflight_runner.py")
    result = runner.run_remote(
        identity,
        ("python3", runner_path, mode, workspace),
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )
    if result.returncode != 0 or getattr(result, "timed_out", False):
        raise BootstrapError(f"{mode} preflight execution failed")
    return {
        "status": "OFFLINE_PREFLIGHT_COMPLETE" if mode == "offline" else "PREFLIGHT_COMPLETE",
        "remote_root": root,
        "mode": mode,
    }


def collect_runtime_evidence(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_root: str,
    plan: Mapping[str, Any],
    timeout: float,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Stage the fixed probe plan and collect compact facts under the venv.

    The probe itself streams model bytes and validates object_info remotely.
    Only compact JSON facts cross SSH; no logs, full object_info or secrets.
    """

    from scripts.p1 import v3_readiness

    _require_remaining(deadline_monotonic, "runtime evidence")
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if timeout <= 0:
        raise BootstrapError("runtime evidence timeout is invalid")
    root_path = PurePosixPath(root)
    probe_remote = str(root_path / "scripts" / "p1" / "v3_runtime_probe.py")
    plan_remote = str(root_path / "runtime-evidence-plan.json")
    venv_python = plan.get("venv_python")
    if not isinstance(venv_python, str) or not venv_python:
        raise BootstrapError("runtime plan venv python is missing")
    local_probe = Path(__file__).resolve().parent / "v3_runtime_probe.py"
    if not local_probe.is_file():
        raise BootstrapError("local runtime probe is missing")
    probe_sha = hashlib.sha256(local_probe.read_bytes()).hexdigest()
    plan_bytes = (json.dumps(dict(plan), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    handle = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False)
    local_plan = Path(handle.name)
    try:
        with handle:
            handle.write(plan_bytes)
        try:
            runner.put_file(
                identity,
                plan_remote,
                local_plan,
                root=root,
                allow_same_hash=True,
                timeout=min(60.0, timeout),
                deadline_monotonic=deadline_monotonic,
            )
        except Exception as exc:
            raise BootstrapError("runtime plan staging failed") from exc
    finally:
        local_plan.unlink(missing_ok=True)
    result = runner.run_remote(
        identity,
        (venv_python, probe_remote, "--plan", plan_remote),
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )
    if getattr(result, "timed_out", False) or result.returncode != 0:
        raise BootstrapError("runtime probe failed or timed out; no evidence accepted")
    raw = result.stdout.encode("utf-8")
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("runtime probe output is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != v3_readiness.RUNTIME_EVIDENCE_SCHEMA:
        raise BootstrapError("runtime probe schema is invalid")
    if payload.get("run_id") != plan.get("run_id") or payload.get("nonce") != plan.get("nonce"):
        raise BootstrapError("runtime probe run/nonce identity mismatch")
    for key in ("session_binding_sha256", "source_bundle_sha256", "source_approval_sha256"):
        if payload.get(key) != plan.get(key):
            raise BootstrapError(f"runtime probe {key} mismatch")
    if payload.get("host_key") != plan.get("host_key"):
        raise BootstrapError("runtime probe host key fact mismatch")
    collector = payload.get("collector")
    if not isinstance(collector, Mapping) or collector.get("sha256") != probe_sha:
        raise BootstrapError("runtime probe collector hash mismatch")
    if collector.get("plan_sha256") != plan_sha:
        raise BootstrapError("runtime probe plan hash mismatch")
    return {
        "facts": payload,
        "raw": raw,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "collector_sha256": probe_sha,
        "plan_sha256": plan_sha,
    }


def comfy_launch_manifest(
    *,
    remote_root: str,
    workspace: str,
    run_id: str,
    nonce: str,
    remote_port: int = DEFAULT_COMFY_REMOTE_PORT,
) -> dict[str, Any]:
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if not (1 <= remote_port <= 65535):
        raise BootstrapError("comfy port is invalid")
    root_path = PurePosixPath(root)
    comfy_dir = str(PurePosixPath(workspace) / "ComfyUI")
    python = str(PurePosixPath(workspace) / "p1-venv" / "bin" / "python")
    return {
        "schema": LAUNCH_SCHEMA,
        "kind": "comfy",
        "run_id": run_id,
        "nonce": nonce,
        "root": root,
        "python": python,
        "comfy_dir": comfy_dir,
        "main": str(PurePosixPath(comfy_dir) / "main.py"),
        "port": remote_port,
        "log": str(root_path / "comfy.log"),
        "owner": str(root_path / "comfy-owner.json"),
        "ack": str(root_path / "comfy-ack.json"),
    }


def start_comfy(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_root: str,
    run_id: str,
    nonce: str,
    timeout: float,
    helper_path: str | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    _require_remaining(deadline_monotonic, "comfy start")
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if timeout <= 0:
        raise BootstrapError("comfy start timeout is invalid")
    launch_path = str(PurePosixPath(root) / "comfy-launch.json")
    launcher_path = str(PurePosixPath(root) / "v3_remote_launcher.py")
    result = runner.run_remote(
        identity,
        ("python3", launcher_path, "--launch", launch_path),
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )
    if getattr(result, "timed_out", False) or result.returncode != 0:
        raise BootstrapError("ComfyUI owned start failed")
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("ComfyUI launch result is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != LAUNCH_RESULT_SCHEMA
        or payload.get("kind") != "comfy"
        or payload.get("run_id") != run_id
        or payload.get("nonce") != nonce
        or payload.get("status") not in {"STARTED", "ATTACHED"}
    ):
        raise BootstrapError("ComfyUI launch identity mismatch")
    owner_path = str(PurePosixPath(root) / "comfy-owner.json")
    helper = helper_path or str(PurePosixPath(root) / "scripts" / "p1" / "v3_remote_fetch_helper.py")
    from scripts.p1 import v3_lifecycle

    owner = v3_lifecycle.read_remote_json(
        runner=runner,
        identity=identity,
        helper_path=helper,
        remote_path=owner_path,
        remote_root=root,
        deadline_monotonic=deadline_monotonic,
    )
    if (
        owner.get("schema") != "anime-video-studio/p1-v3-process-owner/v1"
        or owner.get("run_id") != run_id
        or owner.get("nonce") != nonce
        or owner.get("kind") != "comfy"
    ):
        raise BootstrapError("ComfyUI owner identity mismatch")
    return {
        "status": "COMFY_STARTED" if payload.get("status") == "STARTED" else "COMFY_ATTACHED",
        "pid": owner.get("pid"),
        "owner": owner_path,
        "launch": payload,
    }


def bootstrap_worker(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_root: str,
    setup_text: str,
    preflight_text: str,
    workspace: str,
    timeout: float,
    execute_preflight: bool = False,
) -> dict[str, Any]:
    """Backward-compatible composition; preflight is separate by default."""

    prepared = prepare_remote_root(
        runner=runner, identity=identity, remote_root=remote_root
    )
    staged = stage_bootstrap(
        runner=runner,
        identity=identity,
        remote_root=remote_root,
        setup_text=setup_text,
        preflight_text=preflight_text,
    )
    setup = run_setup(
        runner=runner,
        identity=identity,
        remote_root=remote_root,
        timeout=timeout,
    )
    result = {**prepared, **staged, **setup}
    if execute_preflight:
        result.update(
            run_preflight(
                runner=runner,
                identity=identity,
                remote_root=remote_root,
                workspace=workspace,
                timeout=timeout,
                mode="offline",
            )
        )
    return result


__all__ = [
    "BootstrapError",
    "DEFAULT_COMFY_REMOTE_PORT",
    "DEFAULT_LOCAL_TUNNEL_PORT",
    "DEFAULT_TUNNEL_FRESHNESS_SECONDS",
    "DEFAULT_TUNNEL_HEARTBEAT_INTERVAL",
    "OwnedTunnel",
    "bootstrap_worker",
    "build_tunnel_argv",
    "comfy_launch_manifest",
    "ensure_local_port_free",
    "prepare_remote_root",
    "render_bootstrap_setup",
    "run_preflight",
    "run_setup",
    "stage_bootstrap",
    "start_comfy",
]
