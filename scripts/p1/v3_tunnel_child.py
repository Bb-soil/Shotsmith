"""Focused independent monitor for the local P1 v3 SSH tunnel.

The monitor is the owner of the tunnel identity and lifetime. It spawns the
pinned ``ssh -N -L`` command with a fixed argv, writes the owner/ack/heartbeat
records, and serves a bounded loopback control socket. The heartbeat is written
only while the exact SSH child is alive; when the child dies the heartbeat is
removed and a terminal state is recorded.

This file is deliberately not a generic command runner: every field of the
config is validated, the argv is recomputed from the exact host/user/port/
forward mapping, and no shell is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_SCHEMA = "anime-video-studio/p1-v3-tunnel-config/v1"
OWNER_SCHEMA = "anime-video-studio/p1-v3-tunnel-owner/v1"
ACK_SCHEMA = "anime-video-studio/p1-v3-tunnel-ack/v1"
HEARTBEAT_SCHEMA = "anime-video-studio/p1-v3-tunnel-heartbeat/v1"
STATE_SCHEMA = "anime-video-studio/p1-v3-tunnel-state/v1"
STARTING_SCHEMA = "anime-video-studio/p1-v3-tunnel-starting/v1"
CONTROL_SCHEMA = "anime-video-studio/p1-v3-tunnel-control/v1"

MAX_JSON_BYTES = 64 * 1024
MAX_CONTROL_BYTES = 4 * 1024
MAX_LOG_BYTES = 1024 * 1024
MIN_HEARTBEAT_INTERVAL = 0.05
MAX_HEARTBEAT_INTERVAL = 60.0

_CONFIG_KEYS = {
    "schema",
    "run_id",
    "nonce",
    "host",
    "user",
    "port",
    "identity_file",
    "known_hosts",
    "global_known_hosts",
    "host_key_algorithms",
    "fingerprint",
    "local_port",
    "remote_port",
    "owner",
    "ack",
    "heartbeat",
    "state",
    "starting",
    "log",
    "argv",
    "command_sha256",
    "heartbeat_interval",
    "ssh_executable",
    "ssh_args_prefix",
}


class TunnelChildError(Exception):
    pass


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def command_sha256(argv: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical(list(argv))).hexdigest()


def _atomic(path: Path, payload: Mapping[str, Any], *, bound: int = MAX_JSON_BYTES) -> None:
    data = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(data) > bound:
        raise TunnelChildError("tunnel JSON record exceeds the bound")
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


def _noclobber(path: Path, payload: Mapping[str, Any], *, bound: int = MAX_JSON_BYTES) -> None:
    data = _canonical(payload) + b"\n"
    if len(data) > bound:
        raise TunnelChildError("tunnel JSON record exceeds the bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_json(path: Path) -> dict[str, Any]:
    if os.path.islink(path) or not os.path.isfile(path):
        raise TunnelChildError("tunnel config must be a regular non-symlink file")
    if os.path.getsize(path) > MAX_JSON_BYTES:
        raise TunnelChildError("tunnel config exceeds the bound")
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TunnelChildError("tunnel config is unreadable") from exc
    if not isinstance(value, dict):
        raise TunnelChildError("tunnel config must be an object")
    return value


def _absolute_posix_or_platform(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TunnelChildError(f"{name} must be a non-empty string")
    if os.name == "nt":
        if not os.path.isabs(value):
            raise TunnelChildError(f"{name} must be absolute")
    else:
        if "\\" in value or not os.path.isabs(value):
            raise TunnelChildError(f"{name} must be an absolute POSIX path")
    if os.path.normpath(value) != value or any(part == ".." for part in value.split(os.sep)):
        raise TunnelChildError(f"{name} must be normalized without traversal")
    return value


def _port(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 65535):
        raise TunnelChildError(f"{name} is invalid")
    return value


def _expected_argv(config: Mapping[str, Any]) -> list[str]:
    from scripts.p1.v3_bootstrap import build_tunnel_argv

    class Identity:
        host = config["host"]
        user = config["user"]
        port = config["port"]
        identity_file = (
            Path(config["identity_file"])
            if config["identity_file"] is not None
            else None
        )
        known_hosts = Path(config["known_hosts"])
        global_known_hosts = (
            Path(config["global_known_hosts"])
            if config.get("global_known_hosts") is not None
            else None
        )
        host_key_algorithms = config.get("host_key_algorithms")

        @property
        def destination(self) -> str:
            return f"{self.user}@{self.host}"

    return list(
        build_tunnel_argv(
            Identity(),  # type: ignore[arg-type]
            local_port=config["local_port"],
            remote_port=config["remote_port"],
            ssh_executable=config["ssh_executable"],
            ssh_args_prefix=tuple(config["ssh_args_prefix"]),
        )
    )


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if set(config) != _CONFIG_KEYS:
        raise TunnelChildError("tunnel config has unknown/missing keys")
    if config.get("schema") != CONFIG_SCHEMA:
        raise TunnelChildError("tunnel config schema is invalid")
    for key in ("run_id", "nonce"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise TunnelChildError(f"tunnel config {key} is invalid")
    for key in ("host", "user"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise TunnelChildError(f"tunnel config {key} is invalid")
    port = _port(config.get("port"), "ssh port")
    local_port = _port(config.get("local_port"), "local port")
    remote_port = _port(config.get("remote_port"), "remote port")
    if local_port == remote_port:
        raise TunnelChildError("tunnel ports must differ")
    identity_file = config.get("identity_file")
    if identity_file is not None:
        _absolute_posix_or_platform(identity_file, "identity_file")
    for key in ("known_hosts", "owner", "ack", "heartbeat", "state", "starting", "log"):
        _absolute_posix_or_platform(config.get(key), key)
    if config.get("global_known_hosts") is not None:
        _absolute_posix_or_platform(config.get("global_known_hosts"), "global_known_hosts")
    host_key_algorithms = config.get("host_key_algorithms")
    if host_key_algorithms is not None and (
        not isinstance(host_key_algorithms, str) or not host_key_algorithms
    ):
        raise TunnelChildError("host_key_algorithms is invalid")
    fingerprint = config.get("fingerprint")
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or not fingerprint.startswith("SHA256:")
    ):
        raise TunnelChildError("tunnel fingerprint is invalid")
    interval = config.get("heartbeat_interval")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not MIN_HEARTBEAT_INTERVAL <= float(interval) <= MAX_HEARTBEAT_INTERVAL
    ):
        raise TunnelChildError("heartbeat interval is invalid")
    ssh_executable = config.get("ssh_executable")
    if not isinstance(ssh_executable, str) or not ssh_executable or "\x00" in ssh_executable:
        raise TunnelChildError("ssh executable is invalid")
    prefix = config.get("ssh_args_prefix")
    if not isinstance(prefix, list) or not all(
        isinstance(item, str) and item and "\x00" not in item for item in prefix
    ):
        raise TunnelChildError("ssh argv prefix is invalid")
    argv = config.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise TunnelChildError("tunnel argv is invalid")
    expected = _expected_argv(config)
    if argv != expected:
        raise TunnelChildError("tunnel argv does not match the pinned mapping")
    if config.get("command_sha256") != command_sha256(expected):
        raise TunnelChildError("tunnel command hash is invalid")
    if port == local_port:
        raise TunnelChildError("ssh and local ports must differ")
    return dict(config)


def _terminate(process: subprocess.Popen) -> bool:
    try:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(int(process.pid)), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
            else:
                process.terminate()
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass
    return process.poll() is not None


def _heartbeat_payload(config: Mapping[str, Any], ssh_pid: int, monitor_pid: int, control_port: int) -> dict[str, Any]:
    return {
        "schema": HEARTBEAT_SCHEMA,
        "run_id": config["run_id"],
        "nonce": config["nonce"],
        "monitor_pid": monitor_pid,
        "ssh_pid": ssh_pid,
        "pid": ssh_pid,
        "control_port": control_port,
        "host": config["host"],
        "user": config["user"],
        "port": config["port"],
        "fingerprint": config["fingerprint"],
        "local_port": config["local_port"],
        "remote_port": config["remote_port"],
        "command_sha256": config["command_sha256"],
        "at": time.time(),
    }


def _identity_payload(config: Mapping[str, Any], ssh_pid: int, monitor_pid: int, control_port: int) -> dict[str, Any]:
    return {
        "schema": OWNER_SCHEMA,
        "run_id": config["run_id"],
        "nonce": config["nonce"],
        "monitor_pid": monitor_pid,
        "ssh_pid": ssh_pid,
        "pid": ssh_pid,
        "control_port": control_port,
        "host": config["host"],
        "user": config["user"],
        "port": config["port"],
        "destination": f"{config['user']}@{config['host']}",
        "fingerprint": config["fingerprint"],
        "identity_file": config["identity_file"],
        "known_hosts": config["known_hosts"],
        "local_port": config["local_port"],
        "remote_port": config["remote_port"],
        "command_sha256": config["command_sha256"],
        "argv": list(config["argv"]),
        "created_at": time.time(),
    }


def _starting_payload(
    config: Mapping[str, Any],
    monitor_pid: int,
    control_port: int,
    *,
    ssh_pid: int | None,
    status: str,
    cleanup_confirmed: bool | None = None,
) -> dict[str, Any]:
    return {
        "schema": STARTING_SCHEMA,
        "run_id": config["run_id"],
        "nonce": config["nonce"],
        "status": status,
        "monitor_pid": monitor_pid,
        "ssh_pid": ssh_pid,
        "pid": ssh_pid,
        "control_port": control_port,
        "host": config["host"],
        "user": config["user"],
        "port": config["port"],
        "destination": f"{config['user']}@{config['host']}",
        "fingerprint": config["fingerprint"],
        "local_port": config["local_port"],
        "remote_port": config["remote_port"],
        "command_sha256": config["command_sha256"],
        "argv": list(config["argv"]),
        "cleanup_confirmed": cleanup_confirmed,
        "at": time.time(),
    }


def _control_response(
    request: Mapping[str, Any],
    config: Mapping[str, Any],
    ssh_pid: int,
    monitor_pid: int,
    control_port: int,
    *,
    alive: bool,
    action: str,
) -> dict[str, Any]:
    payload = _identity_payload(config, ssh_pid, monitor_pid, control_port)
    payload.update(
        {
            "schema": CONTROL_SCHEMA,
            "status": action,
            "alive": bool(alive),
            "challenge": request["challenge"],
            "at": time.time(),
        }
    )
    return payload


def _read_request(connection: socket.socket) -> dict[str, Any] | None:
    data = bytearray()
    while len(data) <= MAX_CONTROL_BYTES:
        chunk = connection.recv(min(1024, MAX_CONTROL_BYTES + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            break
    if len(data) > MAX_CONTROL_BYTES:
        return None
    try:
        value = json.loads(bytes(data).split(b"\n", 1)[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    challenge = value.get("challenge")
    if not isinstance(challenge, str) or not (8 <= len(challenge) <= 128):
        return None
    if any(ord(char) < 33 or ord(char) > 126 for char in challenge):
        return None
    return value


def _serve_control(
    connection: socket.socket,
    config: Mapping[str, Any],
    ssh_pid: int,
    monitor_pid: int,
    control_port: int,
    process: subprocess.Popen,
) -> bool:
    """Return True when a validated stop request was received."""

    try:
        connection.settimeout(1.0)
        request = _read_request(connection)
        if request is None:
            return False
        if (
            request.get("run_id") != config["run_id"]
            or request.get("nonce") != config["nonce"]
        ):
            return False
        action = request.get("action")
        if action not in {"status", "stop"}:
            return False
        alive = process.poll() is None
        response = _control_response(
            request,
            config,
            ssh_pid,
            monitor_pid,
            control_port,
            alive=alive,
            action=action,
        )
        connection.sendall(_canonical(response) + b"\n")
        return action == "stop" and alive
    finally:
        try:
            connection.close()
        except OSError:
            pass


def run(config: Mapping[str, Any]) -> int:
    data = validate_config(config)
    monitor_pid = os.getpid()
    control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    control.bind(("127.0.0.1", 0))
    control.listen(8)
    control.settimeout(0.2)
    control_port = int(control.getsockname()[1])

    owner_path = Path(data["owner"])
    ack_path = Path(data["ack"])
    heartbeat_path = Path(data["heartbeat"])
    state_path = Path(data["state"])
    starting_path = Path(data["starting"])
    heartbeat_interval = float(data["heartbeat_interval"])
    stop_requested = {"value": False}

    def _request_stop(signum, frame) -> None:  # noqa: ANN001
        stop_requested["value"] = True

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            signal.signal(signum, _request_stop)
        except (ValueError, OSError):
            pass

    # Durable pre-spawn ownership: the exact monitor/nonce/config is on disk
    # before SSH exists, so a parent can bound its cleanup even if no ACK was
    # ever published.
    _atomic(
        starting_path,
        _starting_payload(
            data, monitor_pid, control_port, ssh_pid=None, status="STARTING"
        ),
    )

    log_path = Path(data["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if log_path.is_file() and log_path.stat().st_size > MAX_LOG_BYTES:
            os.replace(log_path, log_path.with_name(log_path.name + ".1"))
    except OSError:
        pass
    handle = log_path.open("ab")
    try:
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": handle,
            "stderr": subprocess.STDOUT,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(list(data["argv"]), **kwargs)
    except BaseException:
        handle.close()
        control.close()
        raise
    # From this point the child exists. The try/finally must cover the log
    # close, the first SSH_PID metadata write and every later sidecar write so
    # a metadata failure can never leave an untracked SSH child.
    ssh_pid = int(process.pid)
    started_at = time.time()
    try:
        handle.close()
        _atomic(
            starting_path,
            _starting_payload(
                data, monitor_pid, control_port, ssh_pid=ssh_pid, status="SSH_STARTED"
            ),
        )
        # Give ssh a bounded moment to fail on a bad key/host before publishing
        # ownership. A live child is required for every heartbeat. Signal
        # requests are honoured during this wait.
        grace_end = time.monotonic() + min(0.25, heartbeat_interval)
        while time.monotonic() < grace_end and not stop_requested["value"]:
            time.sleep(min(0.02, max(0.0, grace_end - time.monotonic())))
        if process.poll() is not None:
            return 2
        if stop_requested["value"]:
            return 0
        identity = _identity_payload(data, ssh_pid, monitor_pid, control_port)
        _noclobber(owner_path, identity)
        heartbeat = _heartbeat_payload(data, ssh_pid, monitor_pid, control_port)
        _atomic(heartbeat_path, heartbeat)
        ack = dict(identity)
        ack.update({"schema": ACK_SCHEMA, "status": "ARMED", "ack_at": time.time()})
        _noclobber(ack_path, ack)
        last_heartbeat = time.monotonic()
        while not stop_requested["value"]:
            if process.poll() is not None:
                break
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                _atomic(
                    heartbeat_path,
                    _heartbeat_payload(data, ssh_pid, monitor_pid, control_port),
                )
                last_heartbeat = now
            try:
                connection, _ = control.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if _serve_control(
                connection, data, ssh_pid, monitor_pid, control_port, process
            ):
                stop_requested["value"] = True
        return 0
    finally:
        try:
            handle.close()
        except OSError:
            pass
        control.close()
        try:
            heartbeat_path.unlink(missing_ok=True)
        except OSError:
            pass
        terminated = _terminate(process)
        cleanup_status = "CLEANED" if terminated else "CLEANUP_UNCONFIRMED"
        try:
            _atomic(
                starting_path,
                _starting_payload(
                    data,
                    monitor_pid,
                    control_port,
                    ssh_pid=ssh_pid,
                    status=cleanup_status,
                    cleanup_confirmed=terminated,
                ),
            )
        except OSError:
            pass
        try:
            _atomic(
                state_path,
                {
                    "schema": STATE_SCHEMA,
                    "run_id": data["run_id"],
                    "nonce": data["nonce"],
                    "status": cleanup_status,
                    "monitor_pid": monitor_pid,
                    "ssh_pid": ssh_pid,
                    "returncode": process.poll(),
                    "cleanup_confirmed": terminated,
                    "started_at": started_at,
                    "at": time.time(),
                },
            )
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v3_tunnel_child.py")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config = _read_json(Path(args.config))
        return run(config)
    except Exception as exc:
        print(f"tunnel monitor failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
