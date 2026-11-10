"""Owned localhost tunnel adapter; never stops a foreign ComfyUI/tunnel."""

from __future__ import annotations

import http.client
import json
import os
import socket
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.p1.v3_bootstrap import (
    BootstrapError,
    OwnedTunnel,
    acquire_tunnel_lifecycle_lock,
    pid_state as tunnel_pid_state,
    _windows_pid_state as tunnel_windows_pid_state,
)

from .errors import BlockedError, RemoteFailure


def owner_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "tunnel-owner.json"


def _free_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


ALIVE = "ALIVE"
ABSENT = "ABSENT"
UNKNOWN = "UNKNOWN"

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87


class _WindowsProcessAPI:
    """Tiny injectable wrapper with explicit ctypes signatures."""

    def __init__(self) -> None:
        import ctypes

        self._ctypes = ctypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.OpenProcess.argtypes = [
            ctypes.c_ulong,
            ctypes.c_bool,
            ctypes.c_ulong,
        ]
        self.kernel32.OpenProcess.restype = ctypes.c_void_p
        self.kernel32.GetExitCodeProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        self.kernel32.GetExitCodeProcess.restype = ctypes.c_bool
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = ctypes.c_bool
        self.kernel32.GetLastError.argtypes = []
        self.kernel32.GetLastError.restype = ctypes.c_ulong

    def OpenProcess(self, access, inherit, pid):
        return self.kernel32.OpenProcess(access, inherit, pid)

    def GetExitCodeProcess(self, handle, code_ref):
        return self.kernel32.GetExitCodeProcess(handle, code_ref)

    def CloseHandle(self, handle):
        return self.kernel32.CloseHandle(handle)

    def GetLastError(self):
        return self.kernel32.GetLastError()


def _windows_pid_state(pid: int, api: Any | None = None) -> str:
    """Compatibility wrapper; production uses the P1 tri-state helper."""

    return tunnel_windows_pid_state(pid, api)


def _pid_state(pid: int) -> str:
    """Use the same tri-state decision as the P1 tunnel preflight."""

    return tunnel_pid_state(pid)


def _pid_alive(pid: int) -> bool:
    """Backward-compatible boolean; liveness policy uses _pid_state."""

    return _pid_state(pid) == ALIVE


def _archive_dead_owner(tunnel: OwnedTunnel) -> None:
    """Archive only a fully validated, proved-absent owned tunnel bundle."""

    if tunnel.owner_path is None:
        raise BlockedError("tunnel owner path is missing; manual recovery required")
    lock = None
    try:
        lock = acquire_tunnel_lifecycle_lock(tunnel.owner_path)
        proof = tunnel._validate_recovery_bundle()
        if proof is None:
            raise BlockedError("no owned tunnel bundle is available to archive")
        tunnel._archive_recovery_bundle(proof, source="i1_dead_owner")
    except Exception as exc:
        raise BlockedError(str(exc)) from exc
    finally:
        if lock is not None:
            lock.release()


def _new_tunnel(
    *,
    identity: Any,
    run_dir: Path | str,
    run_id: str,
    nonce: str,
    fingerprint: str,
    local_port: int,
    remote_port: int,
    ssh_executable: str = "ssh",
    ssh_args_prefix: Sequence[str] = (),
) -> OwnedTunnel:
    return OwnedTunnel(
        identity=identity,
        local_port=int(local_port),
        remote_port=int(remote_port),
        log_path=Path(run_dir) / "tunnel.log",
        run_id=run_id,
        nonce=nonce,
        owner_path=owner_path(run_dir),
        expected_fingerprint=fingerprint,
        ssh_executable=ssh_executable,
        ssh_args_prefix=tuple(ssh_args_prefix),
    )


def ensure(
    *,
    identity: Any,
    run_dir: Path | str,
    run_id: str,
    nonce: str,
    fingerprint: str,
    local_port: int,
    remote_port: int,
    deadline_monotonic: float | None = None,
    ssh_executable: str = "ssh",
    ssh_args_prefix: Sequence[str] = (),
) -> OwnedTunnel:
    path = owner_path(run_dir)
    tunnel = _new_tunnel(
        identity=identity,
        run_dir=run_dir,
        run_id=run_id,
        nonce=nonce,
        fingerprint=fingerprint,
        local_port=local_port,
        remote_port=remote_port,
        ssh_executable=ssh_executable,
        ssh_args_prefix=ssh_args_prefix,
    )
    if path.is_file():
        try:
            return OwnedTunnel.attach(
                identity=identity,
                owner_path=path,
                run_id=run_id,
                nonce=nonce,
                expected_fingerprint=fingerprint,
                deadline_monotonic=deadline_monotonic,
                ssh_executable=ssh_executable,
                ssh_args_prefix=ssh_args_prefix,
            )
        except Exception:
            # Do not archive on an arbitrary attach failure.  start validates
            # the full owned bundle, proves both PIDs ABSENT, and only then
            # archives under the lifecycle lock; a live/foreign owner is refused.
            pass
    return tunnel.start(deadline_monotonic=deadline_monotonic)


def stop(
    *,
    identity: Any,
    run_dir: Path | str,
    run_id: str,
    nonce: str,
    fingerprint: str,
    ssh_executable: str = "ssh",
    ssh_args_prefix: Sequence[str] = (),
) -> bool:
    path = owner_path(run_dir)
    if not path.is_file():
        # A terminal state without an owner is not deleted here.  The next
        # ensure preflight validates/archives it before spawning.
        return True
    local_port = 8189
    remote_port = 8188
    try:
        owner_data = json.loads(path.read_text(encoding="utf-8"))
        local_port = int(owner_data.get("local_port", local_port))
        remote_port = int(owner_data.get("remote_port", remote_port))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    tunnel = _new_tunnel(
        identity=identity,
        run_dir=run_dir,
        run_id=run_id,
        nonce=nonce,
        fingerprint=fingerprint,
        local_port=local_port,
        remote_port=remote_port,
        ssh_executable=ssh_executable,
        ssh_args_prefix=ssh_args_prefix,
    )
    try:
        attached = OwnedTunnel.attach(
            identity=identity,
            owner_path=path,
            run_id=run_id,
            nonce=nonce,
            expected_fingerprint=fingerprint,
            ssh_executable=ssh_executable,
            ssh_args_prefix=ssh_args_prefix,
        )
        return attached.stop()
    except Exception:
        try:
            _archive_dead_owner(tunnel)
        except BlockedError:
            return False
        return True


def get_json(*, local_port: int, path: str, timeout: float = 3.0) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", int(local_port), timeout=timeout)
    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        data = response.read(256 * 1024)
        if response.status != 200:
            raise RemoteFailure(f"loopback GET {path} returned {response.status}")
        value = json.loads(data.decode("utf-8"))
        if not isinstance(value, dict):
            raise RemoteFailure(f"loopback GET {path} is not an object")
        return value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteFailure(f"loopback GET {path} failed") from exc
    finally:
        connection.close()


def health(*, local_port: int, timeout: float = 3.0) -> dict[str, Any]:
    return get_json(local_port=local_port, path="/system_stats", timeout=timeout)


__all__ = ["ABSENT", "ALIVE", "UNKNOWN", "ensure", "get_json", "health", "owner_path", "stop"]
