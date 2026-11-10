from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.i1 import tunnel as i1_tunnel
from scripts.i1.errors import BlockedError
from scripts.p1 import v3_bootstrap

DEAD_MONITOR = 2_000_000_000
DEAD_SSH = 1_999_999_999


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _identity(tmp_path: Path) -> SimpleNamespace:
    identity_file = tmp_path / "id"
    known_hosts = tmp_path / "known"
    identity_file.write_text("x", encoding="utf-8")
    known_hosts.write_text("x", encoding="utf-8")
    return SimpleNamespace(
        host="example.test",
        user="root",
        port=22,
        identity_file=identity_file,
        known_hosts=known_hosts,
        destination="root@example.test",
    )


def _fake_ssh(tmp_path: Path, *, reset: bool = False) -> Path:
    fake = tmp_path / ("fake_ssh_reset.py" if reset else "fake_ssh.py")
    if reset:
        fake.write_text(
            "import sys,time\n"
            "time.sleep(1.5)\n"
            "print('client_loop: send disconnect: Connection reset', file=sys.stderr, flush=True)\n"
            "raise SystemExit(255)\n",
            encoding="utf-8",
        )
    else:
        fake.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    return fake


def _tunnel(
    tmp_path: Path,
    *,
    run_id: str = "r",
    nonce: str = "n",
    fake_ssh: Path | None = None,
    local_port: int | None = None,
    remote_port: int | None = None,
) -> v3_bootstrap.OwnedTunnel:
    return v3_bootstrap.OwnedTunnel(
        identity=_identity(tmp_path),
        local_port=local_port or _free_port(),
        remote_port=remote_port or _free_port(),
        log_path=tmp_path / "tunnel.log",
        run_id=run_id,
        nonce=nonce,
        owner_path=tmp_path / "tunnel-owner.json",
        expected_fingerprint="SHA256:test",
        ssh_executable=sys.executable,
        ssh_args_prefix=(str(fake_ssh or _fake_ssh(tmp_path)),),
        heartbeat_interval=0.1,
    )


def _owner_payload(tunnel: v3_bootstrap.OwnedTunnel, *, monitor: int, ssh: int, fingerprint: str | None = None) -> dict:
    return {
        "schema": v3_bootstrap.TUNNEL_SCHEMA,
        "run_id": tunnel.run_id,
        "nonce": tunnel.nonce,
        "monitor_pid": monitor,
        "ssh_pid": ssh,
        "pid": ssh,
        "control_port": 12345,
        "host": tunnel.identity.host,
        "user": tunnel.identity.user,
        "port": int(tunnel.identity.port),
        "destination": tunnel.identity.destination,
        "fingerprint": fingerprint or tunnel.expected_fingerprint,
        "identity_file": str(tunnel.identity.identity_file),
        "known_hosts": str(tunnel.identity.known_hosts),
        "local_port": int(tunnel.local_port),
        "remote_port": int(tunnel.remote_port),
        "command_sha256": tunnel._command_sha256(),
        "argv": list(tunnel._command_argv()),
        "created_at": time.time(),
    }


def _write_config(tunnel: v3_bootstrap.OwnedTunnel) -> None:
    v3_bootstrap._atomic_json(tunnel.config_path, tunnel._config_payload())


def _write_state(
    tunnel: v3_bootstrap.OwnedTunnel,
    *,
    monitor: int = DEAD_MONITOR,
    ssh: int = DEAD_SSH,
) -> bytes:
    payload = {
        "schema": v3_bootstrap.TUNNEL_STATE_SCHEMA,
        "run_id": tunnel.run_id,
        "nonce": tunnel.nonce,
        "status": "CLEANED",
        "cleanup_confirmed": True,
        "reason": "post-ack:BootstrapError",
        "monitor_pid": monitor,
        "ssh_pid": ssh,
        "at": time.time(),
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    assert tunnel.state_path is not None
    tunnel.state_path.write_bytes(data)
    return data


def _archive_dirs(tmp_path: Path) -> list[Path]:
    archive = tmp_path / "archive"
    return sorted(archive.glob("owned-tunnel-*")) if archive.exists() else []


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_dead_owner_without_trusted_config_is_not_archived(tmp_path: Path) -> None:
    tunnel = _tunnel(tmp_path)
    assert tunnel.owner_path is not None
    owner = _owner_payload(tunnel, monitor=DEAD_MONITOR, ssh=DEAD_SSH)
    tunnel.owner_path.write_text(json.dumps(owner), encoding="utf-8")
    with pytest.raises(BlockedError):
        i1_tunnel._archive_dead_owner(tunnel)
    assert tunnel.owner_path.is_file()
    assert _archive_dirs(tmp_path) == []


def test_dead_owner_full_proof_is_archived(tmp_path: Path) -> None:
    tunnel = _tunnel(tmp_path)
    assert tunnel.owner_path is not None
    _write_config(tunnel)
    tunnel.owner_path.write_text(
        json.dumps(_owner_payload(tunnel, monitor=DEAD_MONITOR, ssh=DEAD_SSH)),
        encoding="utf-8",
    )
    _write_state(tunnel)
    i1_tunnel._archive_dead_owner(tunnel)
    archives = _archive_dirs(tmp_path)
    assert len(archives) == 1
    assert not tunnel.owner_path.exists()
    assert not tunnel.config_path.exists()
    manifest = json.loads((archives[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["monitor_liveness"] == v3_bootstrap.PID_ABSENT
    assert manifest["ssh_liveness"] == v3_bootstrap.PID_ABSENT


def test_dead_owner_without_terminal_state_is_not_archived(tmp_path: Path) -> None:
    tunnel = _tunnel(tmp_path)
    assert tunnel.owner_path is not None
    _write_config(tunnel)
    tunnel.owner_path.write_text(
        json.dumps(_owner_payload(tunnel, monitor=DEAD_MONITOR, ssh=DEAD_SSH)),
        encoding="utf-8",
    )
    owner_before = tunnel.owner_path.read_bytes()
    config_before = tunnel.config_path.read_bytes()
    with pytest.raises(BlockedError):
        i1_tunnel._archive_dead_owner(tunnel)
    assert tunnel.owner_path.read_bytes() == owner_before
    assert tunnel.config_path.read_bytes() == config_before
    assert _archive_dirs(tmp_path) == []


def test_foreign_fingerprint_dead_owner_is_not_archived(tmp_path: Path) -> None:
    tunnel = _tunnel(tmp_path)
    assert tunnel.owner_path is not None
    _write_config(tunnel)
    tunnel.owner_path.write_text(
        json.dumps(
            _owner_payload(
                tunnel,
                monitor=DEAD_MONITOR,
                ssh=DEAD_SSH,
                fingerprint="SHA256:foreign",
            )
        ),
        encoding="utf-8",
    )
    _write_state(tunnel)
    with pytest.raises(BlockedError):
        i1_tunnel._archive_dead_owner(tunnel)
    assert tunnel.owner_path.is_file()
    assert _archive_dirs(tmp_path) == []


def test_live_or_reused_pid_is_never_archived(tmp_path: Path) -> None:
    tunnel = _tunnel(tmp_path)
    assert tunnel.owner_path is not None
    _write_config(tunnel)
    tunnel.owner_path.write_text(
        json.dumps(_owner_payload(tunnel, monitor=os.getpid(), ssh=DEAD_SSH)),
        encoding="utf-8",
    )
    _write_state(tunnel, monitor=os.getpid(), ssh=DEAD_SSH)
    before = tunnel.owner_path.read_bytes()
    with pytest.raises(BlockedError):
        i1_tunnel._archive_dead_owner(tunnel)
    assert tunnel.owner_path.read_bytes() == before
    assert _archive_dirs(tmp_path) == []


class FakeWindowsAPI:
    def __init__(self, *, handle=1, error=0, exit_ok=True, code=259, raise_open=False):
        self.handle = handle
        self.error = error
        self.exit_ok = exit_ok
        self.code = code
        self.raise_open = raise_open

    def OpenProcess(self, access, inherit, pid):
        if self.raise_open:
            raise OSError("injected")
        return self.handle

    def GetExitCodeProcess(self, handle, code_ref):
        if not self.exit_ok:
            return False
        code_ref._obj.value = self.code
        return True

    def CloseHandle(self, handle):
        return True

    def GetLastError(self):
        return self.error


def test_windows_liveness_access_denied_and_api_failures_are_unknown() -> None:
    denied = FakeWindowsAPI(handle=None, error=5)
    assert i1_tunnel._windows_pid_state(123, denied) == i1_tunnel.UNKNOWN
    getexit = FakeWindowsAPI(handle=1, exit_ok=False)
    assert i1_tunnel._windows_pid_state(123, getexit) == i1_tunnel.UNKNOWN
    raised = FakeWindowsAPI(raise_open=True)
    assert i1_tunnel._windows_pid_state(123, raised) == i1_tunnel.UNKNOWN
    exited = FakeWindowsAPI(handle=1, exit_ok=True, code=0)
    assert i1_tunnel._windows_pid_state(123, exited) == i1_tunnel.ABSENT
    missing = FakeWindowsAPI(handle=None, error=87)
    assert i1_tunnel._windows_pid_state(123, missing) == i1_tunnel.ABSENT


def test_bootstrap_windows_pid_state_actual_api_branch() -> None:
    denied = FakeWindowsAPI(handle=None, error=5)
    assert v3_bootstrap._windows_pid_state(123, denied) == v3_bootstrap.PID_UNKNOWN
    getexit = FakeWindowsAPI(handle=1, exit_ok=False)
    assert v3_bootstrap._windows_pid_state(123, getexit) == v3_bootstrap.PID_UNKNOWN
    missing = FakeWindowsAPI(handle=None, error=87)
    assert v3_bootstrap._windows_pid_state(123, missing) == v3_bootstrap.PID_ABSENT
    alive = FakeWindowsAPI(handle=1, exit_ok=True, code=259)
    assert v3_bootstrap._windows_pid_state(123, alive) == v3_bootstrap.PID_ALIVE
    exited = FakeWindowsAPI(handle=1, exit_ok=True, code=0)
    assert v3_bootstrap._windows_pid_state(123, exited) == v3_bootstrap.PID_ABSENT


def test_windows_actual_current_and_nonexistent_pid_states() -> None:
    assert i1_tunnel._pid_state(os.getpid()) == i1_tunnel.ALIVE
    assert i1_tunnel._pid_state(DEAD_MONITOR) == i1_tunnel.ABSENT


def test_unknown_liveness_retains_tunnel_bundle(tmp_path: Path, monkeypatch) -> None:
    tunnel = _tunnel(tmp_path)
    assert tunnel.owner_path is not None
    _write_config(tunnel)
    tunnel.owner_path.write_text(
        json.dumps(_owner_payload(tunnel, monitor=DEAD_MONITOR, ssh=DEAD_SSH)),
        encoding="utf-8",
    )
    _write_state(tunnel)
    before = tunnel.owner_path.read_bytes()
    monkeypatch.setattr(v3_bootstrap, "pid_state", lambda pid: v3_bootstrap.PID_UNKNOWN)
    with pytest.raises(BlockedError):
        i1_tunnel._archive_dead_owner(tunnel)
    assert tunnel.owner_path.read_bytes() == before
    assert _archive_dirs(tmp_path) == []


def test_i1_ensure_stop_ensure_repeats_with_full_archive(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    identity = _identity(tmp_path)
    fake = _fake_ssh(tmp_path)
    local_port = _free_port()
    remote_port = _free_port()
    kwargs = dict(
        identity=identity,
        run_dir=run_dir,
        run_id="r",
        nonce="n",
        fingerprint="SHA256:test",
        local_port=local_port,
        remote_port=remote_port,
        ssh_executable=sys.executable,
        ssh_args_prefix=(str(fake),),
    )
    stop_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"local_port", "remote_port"}
    }
    seen: list[tuple[int | None, int | None]] = []
    try:
        for _ in range(5):
            tunnel = i1_tunnel.ensure(**kwargs)
            seen.append((tunnel.monitor_pid, tunnel.attached_pid))
            assert i1_tunnel.stop(**stop_kwargs) is True
            assert _wait_until(
                lambda: v3_bootstrap.pid_state(int(seen[-1][0] or 0))
                == v3_bootstrap.PID_ABSENT
                and v3_bootstrap.pid_state(int(seen[-1][1] or 0))
                == v3_bootstrap.PID_ABSENT
            )
        # One more start retires the fifth state; the sixth stop leaves a new one.
        tunnel = i1_tunnel.ensure(**kwargs)
        seen.append((tunnel.monitor_pid, tunnel.attached_pid))
        assert len(_archive_dirs(run_dir)) == 5
        assert len({item[0] for item in seen}) == len(seen)
        assert len({item[1] for item in seen}) == len(seen)
    finally:
        i1_tunnel.stop(**stop_kwargs)


def test_i1_stop_does_not_delete_terminal_state_without_owner(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tunnel = _tunnel(run_dir)
    _write_config(tunnel)
    original = _write_state(tunnel)
    # No owner: stop must not delete the terminal state.
    assert (
        i1_tunnel.stop(
            identity=tunnel.identity,
            run_dir=run_dir,
            run_id=tunnel.run_id,
            nonce=tunnel.nonce,
            fingerprint="SHA256:test",
            ssh_executable=sys.executable,
            ssh_args_prefix=tunnel.ssh_args_prefix,
        )
        is True
    )
    assert tunnel.state_path is not None
    assert tunnel.state_path.read_bytes() == original
    assert _archive_dirs(run_dir) == []
