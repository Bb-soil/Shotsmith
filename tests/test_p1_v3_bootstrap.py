from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.p1 import clore_session as cs
from scripts.p1 import v3_bootstrap, v3_lifecycle, v3_operator, v3_remote_launcher

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json"


def _binding(tmp_path: Path):
    quote = cs.CloreQuote.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))
    now = datetime.now(timezone.utc)
    state_path = tmp_path / "session.json"
    state = cs.initialize_state(
        state_path,
        order_id=95001,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        now=now,
    )
    return state_path, state, v3_lifecycle.SessionBinding.from_state(state)


class _PipelineRunner:
    def __init__(self, state, base: Path) -> None:
        self.state = state
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, dict] = {}
        self.commands: list[tuple[str, ...]] = []
        self.puts: list[str] = []

    def _local(self, remote_path: str) -> Path:
        if remote_path == "/workspace":
            relative = ""
        elif remote_path.startswith("/workspace/"):
            relative = remote_path[len("/workspace/") :]
        else:
            raise AssertionError(f"remote path outside /workspace: {remote_path}")
        target = self.base / relative
        return target

    def put_file(self, identity, remote_path, local_path, **kwargs):
        self.puts.append(remote_path)
        target = self._local(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = Path(local_path).read_bytes()
        if target.exists() and target.read_bytes() != data and target.name == ".p1-owned-root.json":
            raise RuntimeError("foreign root claim")
        target.write_bytes(data)
        if target.suffix == ".json":
            self.files[remote_path] = json.loads(target.read_text(encoding="utf-8"))
        else:
            self.files[remote_path] = {"bytes": target.read_bytes()}

    def run_remote(self, identity, args, timeout=None, **kwargs):
        self.commands.append(tuple(args))
        if args[0] == "mkdir":
            return SimpleNamespace(returncode=0, stdout="", timed_out=False)
        if args[0] == "df":
            return SimpleNamespace(
                returncode=0,
                stdout="Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 100000000 1 99999999 1% /workspace\n",
                timed_out=False,
            )
        if args[0] == "python3" and "v3_remote_launcher.py" in args[1]:
            launch_path = str(args[3])
            launch = json.loads(self._local(launch_path).read_text(encoding="utf-8"))
            if launch["kind"] == "download_worker":
                owner = {
                    "schema": "anime-video-studio/p1-v3-process-owner/v1",
                    "kind": "download_worker",
                    "run_id": launch["run_id"],
                    "nonce": launch["nonce"],
                    "command_sha256": v3_remote_launcher.manifest_hash(launch),
                    "pid": 5151,
                }
                status = {"run_id": launch["run_id"], "nonce": launch["nonce"], "status": "COMPLETE"}
                progress = {
                    "run_id": launch["run_id"],
                    "nonce": launch["nonce"],
                    "status": "COMPLETE",
                    "files_done": 3,
                    "files_total": 3,
                    "bytes_done": 10,
                    "bytes_total": 10,
                }
                heartbeat = {
                    "run_id": launch["run_id"],
                    "nonce": launch["nonce"],
                    "status": "COMPLETE",
                    "heartbeat_at": time.time(),
                }
                root_path = launch["root"]
                for name, value in (
                    ("worker-owner.json", owner),
                    ("worker-ack.json", owner),
                    ("status.json", status),
                    ("progress.json", progress),
                    ("heartbeat.json", heartbeat),
                ):
                    remote = str(Path(root_path) / name).replace("\\", "/")
                    target = self._local(remote)
                    target.write_text(json.dumps(value), encoding="utf-8")
                    self.files[remote] = value
            else:
                owner = {
                    "schema": "anime-video-studio/p1-v3-process-owner/v1",
                    "kind": "comfy",
                    "run_id": launch["run_id"],
                    "nonce": launch["nonce"],
                    "command_sha256": v3_remote_launcher.manifest_hash(launch),
                    "pid": 6161,
                }
                remote = str(Path(launch["root"]) / "comfy-owner.json").replace("\\", "/")
                target = self._local(remote)
                target.write_text(json.dumps(owner), encoding="utf-8")
                self.files[remote] = owner
            kind = launch["kind"]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "anime-video-studio/p1-v3-launch-result/v1",
                        "kind": kind,
                        "run_id": launch["run_id"],
                        "nonce": launch["nonce"],
                        "status": "STARTED",
                        "command_sha256": v3_remote_launcher.manifest_hash(launch),
                        "pid": 5151 if kind == "download_worker" else 6161,
                    }
                ),
                timed_out=False,
            )
        return SimpleNamespace(returncode=0, stdout="", timed_out=False)

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout, **kwargs):
        return json.loads(self._local(remote_path).read_text(encoding="utf-8"))


class _FakeTunnel:
    attached_calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False

    @classmethod
    def attach(cls, **kwargs):
        cls.attached_calls.append(kwargs)
        instance = cls(**kwargs)
        instance.started = True
        return instance

    def start(self, **kwargs):
        self.started = True
        return self

    def health(self, *, timeout=30.0, deadline_monotonic=None):
        return {"system": {"comfyui_version": "test"}}

    def verify_mapping(self, **kwargs):
        return None


def test_live_pipeline_order_resume_and_no_duplicate_remote_actions(
    tmp_path: Path, monkeypatch
) -> None:
    _, state, binding = _binding(tmp_path)
    bundle = v3_operator.load_bundle()
    run = tmp_path / "run"
    run.mkdir()
    lifecycle_path = run / "lifecycle.json"
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=run.name,
        session=binding,
        bundle_hashes=v3_operator._bundle_hashes(bundle),
        phase="SUPERVISOR_STARTED",
    )
    v3_lifecycle.write_lifecycle(lifecycle_path, lifecycle)
    runner = _PipelineRunner(state, tmp_path / "remote-root")
    _FakeTunnel.attached_calls = []
    setup_text = v3_operator._setup_plan(bundle, "/workspace").render()
    preflight_text = "\n".join(v3_operator._preflight_lines("/workspace")) + "\n"

    result = v3_operator._advance_live_pipeline(
        target=run,
        lifecycle_path=lifecycle_path,
        lifecycle_value=lifecycle,
        session_binding=binding,
        remote=object(),
        workspace="/workspace",
        bundle=bundle,
        setup_text=setup_text,
        preflight_text=preflight_text,
        runner=runner,
        execute_downloads=True,
        resume_dead=False,
        tunnel_factory=_FakeTunnel,
    )
    assert result.phase == "READY"
    setup_bytes = runner.files["/workspace/setup.sh"]["bytes"].decode("utf-8")
    assert "--execute" not in setup_bytes
    assert "pip install" in setup_bytes
    assert result.pipeline is not None
    assert "tunnel" in result.pipeline
    assert "comfy" in result.pipeline
    commands = runner.commands
    setup_index = next(
        i
        for i, c in enumerate(commands)
        if c[0] == "python3" and "os.execvp" in c[2]
    )
    assert commands[setup_index][3] == "/workspace"
    worker_index = next(i for i, c in enumerate(commands) if c[0] == "python3" and "v3_remote_launcher.py" in c[1])
    offline_index = next(
        i
        for i, c in enumerate(commands)
        if c[0] == "python3" and "v3_preflight_runner.py" in c[1] and c[2] == "offline"
    )
    comfy_index = next(
        i
        for i, c in enumerate(commands)
        if c[0] == "python3" and "v3_remote_launcher.py" in c[1] and i > worker_index
    )
    api_index = next(
        i
        for i, c in enumerate(commands)
        if c[0] == "python3" and "v3_preflight_runner.py" in c[1] and c[2] == "api"
    )
    assert setup_index < worker_index < offline_index < comfy_index < api_index
    assert runner.commands[comfy_index][3].endswith("comfy-launch.json")

    count = len(runner.commands)
    second = v3_operator._advance_live_pipeline(
        target=run,
        lifecycle_path=lifecycle_path,
        lifecycle_value=result,
        session_binding=binding,
        remote=object(),
        workspace="/workspace",
        bundle=bundle,
        setup_text=setup_text,
        preflight_text=preflight_text,
        runner=runner,
        execute_downloads=True,
        resume_dead=False,
        tunnel_factory=_FakeTunnel,
    )
    assert second.phase == "READY"
    assert len(runner.commands) == count


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tunnel_identity(tmp_path: Path):
    identity = SimpleNamespace(
        host="example.test",
        user="root",
        port=22,
        identity_file=tmp_path / "id",
        known_hosts=tmp_path / "known",
        destination="root@example.test",
    )
    identity.identity_file.write_text("x", encoding="utf-8")
    identity.known_hosts.write_text("x", encoding="utf-8")
    return identity


def _fake_ssh(tmp_path: Path) -> Path:
    fake = tmp_path / "fake_ssh.py"
    fake.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    return fake


def _fake_ssh_with_pid(tmp_path: Path) -> tuple[Path, Path]:
    fake = tmp_path / "fake_ssh_pid.py"
    pid_file = tmp_path / "ssh.pid"
    fake.write_text(
        "import os,pathlib,time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    return fake, pid_file


def _tunnel(
    tmp_path: Path,
    *,
    owner: Path,
    local_port: int,
    remote_port: int,
    run_id: str,
    nonce: str,
    fake_ssh: Path | None = None,
    heartbeat_interval: float = 0.1,
) -> v3_bootstrap.OwnedTunnel:
    return v3_bootstrap.OwnedTunnel(
        identity=_tunnel_identity(tmp_path),
        local_port=local_port,
        remote_port=remote_port,
        log_path=tmp_path / "tunnel.log",
        run_id=run_id,
        nonce=nonce,
        owner_path=owner,
        expected_fingerprint="SHA256:test",
        ssh_executable=sys.executable,
        ssh_args_prefix=(str(fake_ssh or _fake_ssh(tmp_path)),),
        heartbeat_interval=heartbeat_interval,
    )


def _attach(tunnel: v3_bootstrap.OwnedTunnel, *, freshness: float = 1.0, nonce: str | None = None):
    return v3_bootstrap.OwnedTunnel.attach(
        identity=tunnel.identity,
        owner_path=tunnel.owner_path,
        run_id=tunnel.run_id,
        nonce=tunnel.nonce if nonce is None else nonce,
        expected_fingerprint="SHA256:test",
        freshness_seconds=freshness,
        deadline_monotonic=time.monotonic() + 5.0,
        ssh_executable=sys.executable,
        ssh_args_prefix=tunnel.ssh_args_prefix,
    )


def _kill(pid: int) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            os.kill(pid, 9)
    except OSError:
        pass


def test_owned_tunnel_refuses_occupied_local_port(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        with pytest.raises(v3_bootstrap.BootstrapError):
            v3_bootstrap.ensure_local_port_free("127.0.0.1", port)
        tunnel = _tunnel(
            tmp_path,
            owner=tmp_path / "tunnel-owner.json",
            local_port=port,
            remote_port=8188,
            run_id="r",
            nonce="n",
        )
        with pytest.raises(v3_bootstrap.BootstrapError):
            tunnel.start()


def test_owned_tunnel_attach_requires_live_monitor_and_exact_identity(tmp_path: Path) -> None:
    owner = tmp_path / "tunnel-owner.json"
    tunnel = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-live",
        nonce="nonce-live",
    )
    try:
        tunnel.start()
        attached = _attach(tunnel)
        attached.verify_mapping()
        assert attached.monitor_pid == tunnel.monitor_pid
        assert attached.attached_pid != attached.monitor_pid
        with pytest.raises(v3_bootstrap.BootstrapError):
            _attach(tunnel, nonce="wrong-nonce")
        with pytest.raises(v3_bootstrap.BootstrapError):
            v3_bootstrap.OwnedTunnel.attach(
                identity=tunnel.identity,
                owner_path=owner,
                run_id=tunnel.run_id,
                nonce=tunnel.nonce,
                expected_fingerprint="SHA256:wrong",
                freshness_seconds=1.0,
                deadline_monotonic=time.monotonic() + 5.0,
                ssh_executable=sys.executable,
                ssh_args_prefix=tunnel.ssh_args_prefix,
            )
    finally:
        tunnel.stop()


def test_forged_python_sleep_record_is_not_tunnel_ownership(tmp_path: Path) -> None:
    owner = tmp_path / "forged-owner.json"
    tunnel = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-forged",
        nonce="nonce-forged",
    )
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        payload = {
            "schema": v3_bootstrap.TUNNEL_SCHEMA,
            "run_id": tunnel.run_id,
            "nonce": tunnel.nonce,
            "monitor_pid": sleeper.pid,
            "ssh_pid": sleeper.pid,
            "pid": sleeper.pid,
            "control_port": _free_port(),
            "host": tunnel.identity.host,
            "user": tunnel.identity.user,
            "port": tunnel.identity.port,
            "destination": tunnel.identity.destination,
            "fingerprint": "SHA256:test",
            "local_port": tunnel.local_port,
            "remote_port": tunnel.remote_port,
            "command_sha256": tunnel._command_sha256(),
            "argv": list(tunnel._command_argv()),
            "created_at": time.time(),
        }
        owner.write_text(json.dumps(payload), encoding="utf-8")
        (tmp_path / "tunnel-ack.json").write_text(
            json.dumps(
                {
                    **payload,
                    "schema": v3_bootstrap.TUNNEL_ACK_SCHEMA,
                    "status": "ARMED",
                    "ack_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        heartbeat = {
            **payload,
            "schema": v3_bootstrap.TUNNEL_HEARTBEAT_SCHEMA,
            "at": time.time(),
        }
        (tmp_path / "tunnel-heartbeat.json").write_text(
            json.dumps(heartbeat), encoding="utf-8"
        )
        with pytest.raises(v3_bootstrap.BootstrapError):
            _attach(tunnel, freshness=1.0)
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=10)


def test_tunnel_monitor_child_death_stops_heartbeat_and_health(tmp_path: Path) -> None:
    owner = tmp_path / "dead-owner.json"
    tunnel = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-dead",
        nonce="nonce-dead",
    )
    try:
        tunnel.start()
        identity = json.loads(owner.read_text(encoding="utf-8"))
        _kill(int(identity["ssh_pid"]))
        deadline = time.time() + 10
        while time.time() < deadline:
            if (tmp_path / "tunnel-state.json").is_file() and not (
                tmp_path / "tunnel-heartbeat.json"
            ).exists():
                break
            time.sleep(0.05)
        assert not (tmp_path / "tunnel-heartbeat.json").exists()
        with pytest.raises(v3_bootstrap.BootstrapError):
            _attach(tunnel, freshness=0.5)
        with pytest.raises(v3_bootstrap.BootstrapError):
            tunnel.health(timeout=1.0, deadline_monotonic=time.monotonic() + 0.5)
    finally:
        tunnel.stop()


def test_tunnel_survives_parent_exit_and_attach_challenges_live_monitor(
    tmp_path: Path,
) -> None:
    identity = _tunnel_identity(tmp_path)
    fake = _fake_ssh(tmp_path)
    owner = tmp_path / "parent-owner.json"
    local_port = _free_port()
    remote_port = _free_port()
    script = (
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "from scripts.p1 import v3_bootstrap\n"
        "identity=SimpleNamespace(host='example.test',user='root',port=22,"
        f"identity_file=Path({str(identity.identity_file)!r}),"
        f"known_hosts=Path({str(identity.known_hosts)!r}),"
        "destination='root@example.test')\n"
        "tunnel=v3_bootstrap.OwnedTunnel(identity=identity,"
        f"local_port={local_port},remote_port={remote_port},"
        f"log_path=Path({str(tmp_path / 'tunnel.log')!r}),"
        "run_id='run-parent',nonce='nonce-parent',"
        f"owner_path=Path({str(owner)!r}),"
        "expected_fingerprint='SHA256:test',"
        "ssh_executable=__import__('sys').executable,"
        f"ssh_args_prefix=({str(fake)!r},),heartbeat_interval=0.1)\n"
        "tunnel.start()\n"
    )
    tunnel = _tunnel(
        tmp_path,
        owner=owner,
        local_port=local_port,
        remote_port=remote_port,
        run_id="run-parent",
        nonce="nonce-parent",
    )
    attached = None
    cleanup_ok = False
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        time.sleep(1.2)
        attached = _attach(tunnel, freshness=1.0)
        attached.verify_mapping()
        assert attached.monitor_pid is not None
    finally:
        target = attached if attached is not None else tunnel
        if target is attached:
            cleanup_ok = target.stop()
        else:
            # The parent subprocess may have failed after spawning the monitor;
            # stop() reads only exact run/nonce/command records and refuses
            # unconfirmed/blind cleanup.
            cleanup_ok = target.stop()
        if cleanup_ok:
            monitor_pid = target.monitor_pid
            ssh_pid = target.attached_pid
            cleanup_ok = (
                (monitor_pid is None or not v3_bootstrap._pid_live(monitor_pid))
                and (ssh_pid is None or not v3_bootstrap._pid_live(ssh_pid))
            )
    assert cleanup_ok
    assert not (tmp_path / "tunnel-recovery.json").exists()
    # The same local/remote port must be reusable after confirmed cleanup.
    # A different run/nonce uses a different canonical directory; the old
    # terminal record is never treated as this run's record.
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _tunnel(
        second_dir,
        owner=second_dir / "parent-owner-2.json",
        local_port=local_port,
        remote_port=remote_port,
        run_id="run-parent-2",
        nonce="nonce-parent-2",
    )
    second.start()
    try:
        second.verify_mapping()
    finally:
        assert second.stop()


def test_tunnel_startup_timeout_between_ssh_spawn_and_ack_cleans_tree(
    tmp_path: Path,
) -> None:
    fake, ssh_pid_path = _fake_ssh_with_pid(tmp_path)
    local_port = _free_port()
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "timeout-owner.json",
        local_port=local_port,
        remote_port=_free_port(),
        run_id="run-timeout",
        nonce="nonce-timeout",
        fake_ssh=fake,
        heartbeat_interval=0.5,
    )
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start(ack_timeout=0.15)
    state = json.loads((tmp_path / "tunnel-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "CLEANED"
    assert state["cleanup_confirmed"] is True
    assert not v3_bootstrap._pid_live(int(state["monitor_pid"]))
    assert isinstance(state["ssh_pid"], int)
    assert not v3_bootstrap._pid_live(int(state["ssh_pid"]))
    if ssh_pid_path.is_file():
        assert not v3_bootstrap._pid_live(int(ssh_pid_path.read_text()))
    assert not (tmp_path / "tunnel-recovery.json").exists()
    for name in (
        "tunnel-owner.json",
        "tunnel-ack.json",
        "tunnel-heartbeat.json",
        "tunnel-starting.json",
    ):
        assert not (tmp_path / name).exists(), name
    v3_bootstrap.ensure_local_port_free("127.0.0.1", local_port)


def test_tunnel_child_metadata_failure_after_ssh_spawn_kills_child(
    tmp_path: Path, monkeypatch,
) -> None:
    import signal as signal_module

    from scripts.p1 import v3_tunnel_child

    fake, _ssh_pid_path = _fake_ssh_with_pid(tmp_path)
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "metadata-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-metadata",
        nonce="nonce-metadata",
        fake_ssh=fake,
    )
    config = tunnel._config_payload()
    starting_path = Path(config["starting"])
    state_path = Path(config["state"])
    real_atomic = v3_tunnel_child._atomic
    real_popen = v3_tunnel_child.subprocess.Popen
    spawned: list[subprocess.Popen] = []
    counts = {"starting": 0}

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    def failing_atomic(path, payload, **kwargs):
        path = Path(path)
        if path == starting_path:
            counts["starting"] += 1
            if counts["starting"] >= 2:
                raise OSError("injected STARTING metadata failure")
        if path == state_path:
            raise OSError("injected state/recovery failure")
        return real_atomic(path, payload, **kwargs)

    signals = [
        getattr(signal_module, name)
        for name in ("SIGTERM", "SIGINT", "SIGBREAK")
        if hasattr(signal_module, name)
    ]
    old_handlers = {signum: signal_module.getsignal(signum) for signum in signals}
    monkeypatch.setattr(v3_tunnel_child, "_atomic", failing_atomic)
    monkeypatch.setattr(v3_tunnel_child.subprocess, "Popen", capture_popen)
    try:
        with pytest.raises(OSError):
            v3_tunnel_child.run(config)
    finally:
        for signum, handler in old_handlers.items():
            signal_module.signal(signum, handler)
    assert spawned
    assert spawned[0].poll() is not None
    with (tmp_path / "tunnel.log").open("ab"):
        pass
    starting = json.loads(starting_path.read_text(encoding="utf-8"))
    assert starting["status"] == "STARTING"
    assert starting["cleanup_confirmed"] is None
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state.get("cleanup_confirmed") is not True
        assert state.get("status") != "CLEANED"


def test_tunnel_cleanup_prefers_own_popen_poll_over_pid_probe(
    tmp_path: Path, monkeypatch,
) -> None:
    owner = tmp_path / "popen-owner.json"
    tunnel = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-popen",
        nonce="nonce-popen",
    )
    monitor_process = subprocess.Popen([sys.executable, "-c", "pass"])
    monitor_process.wait(timeout=10)
    ssh_process = subprocess.Popen([sys.executable, "-c", "pass"])
    ssh_process.wait(timeout=10)
    tunnel.process = monitor_process
    tunnel.monitor_pid = monitor_process.pid
    starting = {
        "schema": v3_bootstrap.TUNNEL_STARTING_SCHEMA,
        "run_id": tunnel.run_id,
        "nonce": tunnel.nonce,
        "status": "SSH_STARTED",
        "monitor_pid": monitor_process.pid,
        "ssh_pid": ssh_process.pid,
        "pid": ssh_process.pid,
        "control_port": _free_port(),
        "host": tunnel.identity.host,
        "user": tunnel.identity.user,
        "port": tunnel.identity.port,
        "destination": tunnel.identity.destination,
        "fingerprint": tunnel.expected_fingerprint,
        "local_port": tunnel.local_port,
        "remote_port": tunnel.remote_port,
        "command_sha256": tunnel._command_sha256(),
        "argv": list(tunnel._command_argv()),
        "cleanup_confirmed": None,
        "at": time.time(),
    }
    (tmp_path / "tunnel-starting.json").write_text(
        json.dumps(starting), encoding="utf-8"
    )
    terminated: list[int] = []

    def fake_terminate(pid, **kwargs):
        terminated.append(int(pid))
        return True

    # Simulate a POSIX zombie: the PID probe stays positive for our monitor
    # even though Popen.poll() has already reaped it.
    monkeypatch.setattr(
        v3_bootstrap,
        "_pid_live",
        lambda pid: int(pid) == int(monitor_process.pid),
    )
    monkeypatch.setattr(v3_bootstrap, "_terminate_pid", fake_terminate)
    confirmed, details = tunnel._cleanup_attempt(
        monitor_process,
        reason="popen-reaped",
        deadline_monotonic=time.monotonic() + 0.3,
    )
    assert confirmed is True
    assert details["monitor_dead"] is True
    assert monitor_process.pid not in terminated
    state = json.loads((tmp_path / "tunnel-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "CLEANED"
    assert state["cleanup_confirmed"] is True
    assert not (tmp_path / "tunnel-starting.json").exists()


def test_tunnel_unconfirmed_cleanup_retains_ownership_and_recovery(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "unconfirmed-owner.json"
    tunnel = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-unconfirmed",
        nonce="nonce-unconfirmed",
    )
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    tunnel.process = sleeper
    tunnel.monitor_pid = sleeper.pid
    starting = {
        "schema": v3_bootstrap.TUNNEL_STARTING_SCHEMA,
        "run_id": tunnel.run_id,
        "nonce": tunnel.nonce,
        "status": "STARTING",
        "monitor_pid": sleeper.pid,
        "ssh_pid": None,
        "pid": None,
        "control_port": _free_port(),
        "host": tunnel.identity.host,
        "user": tunnel.identity.user,
        "port": tunnel.identity.port,
        "destination": tunnel.identity.destination,
        "fingerprint": tunnel.expected_fingerprint,
        "local_port": tunnel.local_port,
        "remote_port": tunnel.remote_port,
        "command_sha256": tunnel._command_sha256(),
        "argv": list(tunnel._command_argv()),
        "cleanup_confirmed": None,
        "at": time.time(),
    }
    (tmp_path / "tunnel-starting.json").write_text(
        json.dumps(starting), encoding="utf-8"
    )
    try:
        confirmed, details = tunnel._cleanup_attempt(
            sleeper,
            reason="synthetic-unconfirmed",
            deadline_monotonic=time.monotonic() + 0.3,
        )
        assert confirmed is False
        assert details["cleanup_confirmed"] is False
        assert not v3_bootstrap._pid_live(sleeper.pid)
        recovery = json.loads(
            (tmp_path / "tunnel-recovery.json").read_text(encoding="utf-8")
        )
        assert recovery["cleanup_confirmed"] is False
        assert recovery["run_id"] == tunnel.run_id
        assert recovery["status"] == "CLEANUP_UNCONFIRMED"
        # Ownership/recovery records are retained; only the exact process we
        # spawned was terminated.
        assert (tmp_path / "tunnel-starting.json").is_file()
        assert not (tmp_path / "tunnel-owner.json").exists()
    finally:
        (tmp_path / "tunnel-recovery.json").unlink(missing_ok=True)
        (tmp_path / "tunnel-starting.json").unlink(missing_ok=True)
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait(timeout=10)


def test_tunnel_health_uses_total_deadline_for_trickled_response(tmp_path: Path) -> None:
    import threading

    owner = tmp_path / "health-owner.json"
    tunnel = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-health",
        nonce="nonce-health",
    )
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tunnel.start()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", tunnel.local_port))
        server.listen(2)

        def trickle() -> None:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            try:
                connection.recv(4096)
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n"
                )
                for _ in range(30):
                    connection.sendall(b"x")
                    time.sleep(0.1)
            except OSError:
                pass
            finally:
                try:
                    connection.close()
                except OSError:
                    pass

        threading.Thread(target=trickle, daemon=True).start()
        started = time.monotonic()
        with pytest.raises(v3_bootstrap.BootstrapError):
            tunnel.health(timeout=30.0, deadline_monotonic=started + 0.6)
        assert time.monotonic() - started < 4.0
    finally:
        server.close()
        tunnel.stop()



def test_run_setup_passes_root_as_argv_and_never_shell_interpolates(
    tmp_path: Path,
) -> None:
    root = "/workspace/owned dir; touch SENTINEL"
    commands: list[tuple[str, ...]] = []

    class Runner:
        def run_remote(self, identity, args, timeout=None, **kwargs):
            commands.append(tuple(args))
            return SimpleNamespace(returncode=0, stdout="", timed_out=False)

    result = v3_bootstrap.run_setup(
        runner=Runner(),
        identity=SimpleNamespace(),
        remote_root=root,
        timeout=10.0,
    )
    assert result["status"] == "SETUP_COMPLETE"
    assert len(commands) == 1
    args = commands[0]
    assert args[0] == "python3"
    assert args[1] == "-c"
    assert "setup.sh" in args[2]
    assert "SENTINEL" not in args[2]
    assert root not in args[2]
    assert args[3] == root
    assert not (tmp_path / "SENTINEL").exists()


def test_owned_root_claim_refuses_foreign_run_before_install(tmp_path: Path) -> None:
    _, state, binding = _binding(tmp_path)
    runner = _PipelineRunner(state, tmp_path / "remote-root")
    identity = SimpleNamespace()
    v3_bootstrap.claim_owned_root(
        runner=runner,
        identity=identity,
        remote_root="/workspace",
        run_id="run-a",
        nonce="nonce-a",
        session=binding.to_dict(),
    )
    with pytest.raises(v3_bootstrap.BootstrapError):
        v3_bootstrap.prepare_remote_root(
            runner=runner,
            identity=identity,
            remote_root="/workspace",
            run_id="run-b",
            nonce="nonce-b",
            session=binding.to_dict(),
        )
    assert not (tmp_path / "remote-root" / "setup.sh").exists()


# --- owned tunnel recovery preflight (I1 2026-11-06) ---

DEAD_MONITOR = 2_000_000_000
DEAD_SSH = 1_999_999_999


def _archive_dirs(root: Path) -> list[Path]:
    archive = root / "archive"
    return sorted(archive.glob("owned-tunnel-*")) if archive.exists() else []


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _fake_ssh_reset(tmp_path: Path) -> Path:
    fake = tmp_path / "fake_ssh_reset.py"
    fake.write_text(
        "import sys,time\n"
        "time.sleep(1.5)\n"
        "print('client_loop: send disconnect: Connection reset', file=sys.stderr, flush=True)\n"
        "raise SystemExit(255)\n",
        encoding="utf-8",
    )
    return fake


def _recovery_state(tunnel: v3_bootstrap.OwnedTunnel) -> dict:
    return {
        "schema": v3_bootstrap.TUNNEL_STATE_SCHEMA,
        "run_id": tunnel.run_id,
        "nonce": tunnel.nonce,
        "status": "CLEANED",
        "cleanup_confirmed": True,
        "reason": "post-ack:BootstrapError",
        "monitor_pid": DEAD_MONITOR,
        "ssh_pid": DEAD_SSH,
        "at": time.time(),
    }


def _write_recovery_case(
    tunnel: v3_bootstrap.OwnedTunnel,
    *,
    state_mutator=None,
    config_mutator=None,
) -> tuple[bytes, bytes]:
    config = tunnel._config_payload()
    if config_mutator is not None:
        config_mutator(config)
    v3_bootstrap._atomic_json(tunnel.config_path, config)
    state = _recovery_state(tunnel)
    if state_mutator is not None:
        state_mutator(state)
    assert tunnel.state_path is not None
    state_bytes = json.dumps(state, sort_keys=True).encode("utf-8")
    tunnel.state_path.write_bytes(state_bytes)
    return tunnel.config_path.read_bytes(), state_bytes


def _forbid_popen(monkeypatch) -> list[object]:
    calls: list[object] = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("monitor Popen must not be reached")

    monkeypatch.setattr(v3_bootstrap.subprocess, "Popen", forbidden)
    return calls


def _make_symlink(link: Path, target: Path, *, directory: bool) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink not available: {type(exc).__name__}")


def test_run_dir_symlink_blocks_before_any_creation(tmp_path: Path, monkeypatch) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    run_link = tmp_path / "run-link"
    _make_symlink(run_link, outside, directory=True)
    tunnel = _tunnel(
        tmp_path,
        owner=run_link / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-symlink",
        nonce="nonce-symlink",
    )
    tunnel.log_path = run_link / "tunnel.log"
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert list(outside.iterdir()) == []


def test_lock_symlink_blocks_before_outside_write(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"sentinel")
    _make_symlink(run_dir / ".tunnel-lifecycle.lock", outside, directory=False)
    tunnel = _tunnel(
        tmp_path,
        owner=run_dir / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-lock",
        nonce="nonce-lock",
    )
    tunnel.log_path = run_dir / "tunnel.log"
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert outside.read_bytes() == b"sentinel"
    assert not (run_dir / "tunnel-config.json").exists()


def test_log_symlink_blocks_before_outside_write(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"sentinel")
    _make_symlink(run_dir / "tunnel.log", outside, directory=False)
    tunnel = _tunnel(
        tmp_path,
        owner=run_dir / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-log",
        nonce="nonce-log",
    )
    tunnel.log_path = run_dir / "tunnel.log"
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert outside.read_bytes() == b"sentinel"
    assert not (run_dir / "tunnel-config.json").exists()


def test_dangling_symlink_is_detected_without_follow(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside_missing = tmp_path / "missing-target.log"
    _make_symlink(run_dir / "tunnel.log", outside_missing, directory=False)
    tunnel = _tunnel(
        tmp_path,
        owner=run_dir / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-dangling",
        nonce="nonce-dangling",
    )
    tunnel.log_path = run_dir / "tunnel.log"
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert not outside_missing.exists()


def test_ancestry_gate_does_not_probe_descendants_through_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    run_link = tmp_path / "run-link"
    _make_symlink(run_link, outside, directory=True)
    tunnel = _tunnel(
        tmp_path,
        owner=run_link / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-order",
        nonce="nonce-order",
    )
    tunnel.log_path = run_link / "tunnel.log"
    probes: list[str] = []
    real_lstat = os.lstat

    def recording_lstat(path, *args, **kwargs):
        probes.append(os.path.abspath(str(path)))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(v3_bootstrap.os, "lstat", recording_lstat)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert probes
    assert not any(
        item.startswith(os.path.abspath(str(run_link)) + os.sep) for item in probes
    )


def test_start_archives_parent_terminal_state_before_spawn(tmp_path: Path, monkeypatch) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-recover",
        nonce="nonce-recover",
    )
    _config_bytes, state_bytes = _write_recovery_case(tunnel)
    real_popen = subprocess.Popen

    def checking_popen(*args, **kwargs):
        assert _archive_dirs(tmp_path), "archive must exist before monitor spawn"
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(v3_bootstrap.subprocess, "Popen", checking_popen)
    try:
        tunnel.start()
        archives = _archive_dirs(tmp_path)
        assert len(archives) == 1
        archived_state = archives[0] / "tunnel-state.json"
        assert archived_state.read_bytes() == state_bytes
        manifest = json.loads((archives[0] / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "COMPLETE"
        assert manifest["monitor_liveness"] == v3_bootstrap.PID_ABSENT
        assert manifest["ssh_liveness"] == v3_bootstrap.PID_ABSENT
        assert not tunnel.state_path.exists()
    finally:
        tunnel.stop()


def test_start_same_object_five_cycles_archives_each_state(tmp_path: Path) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-repeat",
        nonce="nonce-repeat",
    )
    seen: list[tuple[int, int]] = []
    try:
        for cycle in range(5):
            tunnel.start()
            assert tunnel.monitor_pid and tunnel.attached_pid
            seen.append((int(tunnel.monitor_pid), int(tunnel.attached_pid)))
            if cycle:
                assert len(_archive_dirs(tmp_path)) == cycle
            assert tunnel.stop() is True
        tunnel.start()
        seen.append((int(tunnel.monitor_pid or 0), int(tunnel.attached_pid or 0)))
        archives = _archive_dirs(tmp_path)
        assert len(archives) == 5
        assert all((item / "tunnel-state.json").is_file() for item in archives)
        assert len({item[0] for item in seen}) == len(seen)
        assert len({item[1] for item in seen}) == len(seen)
    finally:
        tunnel.stop()


def test_failed_postack_cleanup_closes_handle_and_new_attempt_safe(
    tmp_path: Path, monkeypatch
) -> None:
    owner = tmp_path / "tunnel-owner.json"
    first = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-failpost",
        nonce="nonce-failpost",
    )

    def failing_challenge(*args, **kwargs):
        raise v3_bootstrap.BootstrapError("injected post-ack challenge failure")

    first._challenge = failing_challenge
    with pytest.raises(v3_bootstrap.BootstrapError):
        first.start()
    assert first._handle_completed is True
    assert first._claim_active is False
    assert first.process is None
    assert first.state_path is not None and first.state_path.is_file()

    second = _tunnel(
        tmp_path,
        owner=owner,
        local_port=first.local_port,
        remote_port=first.remote_port,
        run_id="run-failpost",
        nonce="nonce-failpost",
    )
    second.start()
    try:
        assert second.owner_path is not None
        b_files = {
            "owner": second.owner_path.read_bytes(),
            "ack": second.ack_path.read_bytes() if second.ack_path else b"",
            "heartbeat": (
                second.heartbeat_path.read_bytes() if second.heartbeat_path else b""
            ),
            "starting": (
                second.starting_path.read_bytes() if second.starting_path else b""
            ),
        }
        signals: list[tuple[str, object]] = []
        monkeypatch.setattr(
            v3_bootstrap, "_terminate_pid", lambda pid, **k: signals.append(("kill", pid)) or True
        )
        monkeypatch.setattr(
            v3_bootstrap, "_terminate_owned_pids", lambda *a, **k: signals.append(("tree", a)) or True
        )
        monkeypatch.setattr(
            v3_bootstrap.OwnedTunnel,
            "_control_request",
            lambda *a, **k: signals.append(("control", a)) or {},
        )
        assert first.stop() is True
        assert signals == []
        assert second.owner_path.read_bytes() == b_files["owner"]
        assert second.ack_path and second.ack_path.read_bytes() == b_files["ack"]
        assert second.heartbeat_path and second.heartbeat_path.read_bytes() == b_files["heartbeat"]
        assert second.starting_path and second.starting_path.read_bytes() == b_files["starting"]
        monkeypatch.undo()
        second.verify_mapping()
    finally:
        second.stop()


def test_failed_preack_cleanup_closes_handle_and_new_attempt_safe(
    tmp_path: Path, monkeypatch
) -> None:
    owner = tmp_path / "tunnel-owner.json"
    first = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-failpre",
        nonce="nonce-failpre",
        heartbeat_interval=0.5,
    )
    with pytest.raises(v3_bootstrap.BootstrapError):
        first.start(ack_timeout=0.15)
    assert first._handle_completed is True
    assert first._claim_active is False
    assert first.process is None
    second = _tunnel(
        tmp_path,
        owner=owner,
        local_port=first.local_port,
        remote_port=first.remote_port,
        run_id="run-failpre",
        nonce="nonce-failpre",
    )
    second.start()
    try:
        assert second.owner_path is not None
        before = second.owner_path.read_bytes()
        assert first.stop() is True
        assert second.owner_path.read_bytes() == before
        second.verify_mapping()
    finally:
        second.stop()


def test_unconfirmed_cleanup_stale_handle_does_not_overwrite_newer_records(
    tmp_path: Path, monkeypatch
) -> None:
    first = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-unconfirmed",
        nonce="nonce-unconfirmed",
    )
    second = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=first.local_port,
        remote_port=first.remote_port,
        run_id="run-unconfirmed",
        nonce="nonce-unconfirmed",
    )
    owner = _sidecar_payload(
        second,
        v3_bootstrap.TUNNEL_SCHEMA,
        monitor=DEAD_MONITOR,
        ssh=DEAD_SSH + 1,
        control_port=12346,
    )
    owner["created_at"] = time.time()
    starting = _sidecar_payload(
        second,
        v3_bootstrap.TUNNEL_STARTING_SCHEMA,
        monitor=DEAD_MONITOR,
        ssh=DEAD_SSH + 1,
        control_port=12346,
    )
    starting.update(status="SSH_STARTED", cleanup_confirmed=None)
    assert second.owner_path is not None and second.starting_path is not None
    second.owner_path.write_text(json.dumps(owner), encoding="utf-8")
    second.starting_path.write_text(json.dumps(starting), encoding="utf-8")
    owner_before = second.owner_path.read_bytes()
    starting_before = second.starting_path.read_bytes()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    first.process = process
    first.monitor_pid = process.pid
    first.attached_pid = DEAD_SSH
    first.control_port = 12345
    first._claim_active = True
    first._claim_monitor_pid = DEAD_MONITOR
    first._claim_ssh_pid = DEAD_SSH
    first._claim_control_port = 12345
    monkeypatch.setattr(v3_bootstrap, "_terminate_pid", lambda *a, **k: True)
    monkeypatch.setattr(v3_bootstrap, "_terminate_owned_pids", lambda *a, **k: True)
    try:
        with pytest.raises(v3_bootstrap.BootstrapError):
            first._cleanup_attempt(
                process,
                reason="stale-unconfirmed",
                deadline_monotonic=time.monotonic() + 0.15,
            )
        assert second.owner_path.read_bytes() == owner_before
        assert second.starting_path.read_bytes() == starting_before
        assert not (tmp_path / "tunnel-recovery.json").exists()
        assert not (tmp_path / "tunnel-state.json").exists()
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_stopped_handle_is_noop_after_new_attempt(tmp_path: Path) -> None:
    owner = tmp_path / "tunnel-owner.json"
    first = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-stale",
        nonce="nonce-stale",
    )
    first.start()
    assert first.stop() is True
    second = _tunnel(
        tmp_path,
        owner=owner,
        local_port=first.local_port,
        remote_port=first.remote_port,
        run_id="run-stale",
        nonce="nonce-stale",
    )
    second.start()
    try:
        assert second.owner_path is not None and second.ack_path is not None
        owner_before = second.owner_path.read_bytes()
        ack_before = second.ack_path.read_bytes()
        second.verify_mapping()
        assert first.stop() is True
        assert second.owner_path.read_bytes() == owner_before
        assert second.ack_path.read_bytes() == ack_before
        second.verify_mapping()
    finally:
        second.stop()


def test_stale_uncompleted_handle_never_signals_fresh_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    owner = tmp_path / "tunnel-owner.json"
    first = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-stale2",
        nonce="nonce-stale2",
        fake_ssh=_fake_ssh_reset(tmp_path),
    )
    first.start()
    assert first.state_path is not None
    assert _wait_until(
        lambda: first.state_path.is_file()
        and v3_bootstrap.pid_state(int(first.monitor_pid or 0))
        == v3_bootstrap.PID_ABSENT
    )
    second = _tunnel(
        tmp_path,
        owner=owner,
        local_port=first.local_port,
        remote_port=first.remote_port,
        run_id="run-stale2",
        nonce="nonce-stale2",
        fake_ssh=_fake_ssh_reset(tmp_path),
    )
    second.start()
    signals: list[tuple[str, object]] = []
    monkeypatch.setattr(
        v3_bootstrap, "_terminate_pid", lambda pid, **k: signals.append(("kill", pid)) or True
    )
    monkeypatch.setattr(
        v3_bootstrap.OwnedTunnel,
        "_control_request",
        lambda *a, **k: signals.append(("control", a)) or {},
    )
    try:
        assert second.owner_path is not None
        owner_before = second.owner_path.read_bytes()
        with pytest.raises(v3_bootstrap.BootstrapError):
            first.stop()
        assert signals == []
        assert second.owner_path.read_bytes() == owner_before
        monkeypatch.undo()
        second.verify_mapping()
    finally:
        second.stop()


def test_reused_live_pid_without_control_identity_is_not_signalled(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-reuse",
        nonce="nonce-reuse",
    )
    _write_recovery_case(tunnel)
    owner = _sidecar_payload(
        tunnel,
        v3_bootstrap.TUNNEL_SCHEMA,
        monitor=os.getpid(),
        ssh=DEAD_SSH,
        control_port=12345,
    )
    owner["created_at"] = time.time()
    assert tunnel.owner_path is not None
    tunnel.owner_path.write_text(json.dumps(owner), encoding="utf-8")
    tunnel.monitor_pid = os.getpid()
    tunnel.attached_pid = DEAD_SSH
    tunnel.control_port = 12345
    tunnel._handle_started = True
    signals: list[tuple[str, object]] = []
    monkeypatch.setattr(
        v3_bootstrap, "_terminate_pid", lambda pid, **k: signals.append(("kill", pid)) or True
    )

    def failing_challenge(*args, **kwargs):
        raise v3_bootstrap.BootstrapError("control identity mismatch")

    monkeypatch.setattr(v3_bootstrap.OwnedTunnel, "_challenge", failing_challenge)
    before = tunnel.owner_path.read_bytes()
    assert tunnel.stop() is False
    assert signals == []
    assert tunnel.owner_path.read_bytes() == before


def test_start_refuses_live_owner_without_archiving(tmp_path: Path) -> None:
    first = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-live",
        nonce="nonce-live",
    )
    first.start()
    try:
        assert first.owner_path is not None
        before = first.owner_path.read_bytes()
        second = _tunnel(
            tmp_path,
            owner=tmp_path / "tunnel-owner.json",
            local_port=first.local_port,
            remote_port=first.remote_port,
            run_id="run-live",
            nonce="nonce-live",
        )
        with pytest.raises(v3_bootstrap.BootstrapError):
            second.start()
        assert first.owner_path.read_bytes() == before
        assert _archive_dirs(tmp_path) == []
    finally:
        first.stop()


def test_start_archives_after_child_death_owner_present(tmp_path: Path) -> None:
    fake = _fake_ssh_reset(tmp_path)
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-reset",
        nonce="nonce-reset",
        fake_ssh=fake,
    )
    try:
        tunnel.start()
        assert tunnel.state_path is not None
        assert _wait_until(
            lambda: tunnel.state_path.is_file()
            and json.loads(tunnel.state_path.read_text(encoding="utf-8")).get(
                "returncode"
            )
            == 255
        )
        assert _wait_until(
            lambda: v3_bootstrap.pid_state(int(tunnel.monitor_pid or 0))
            == v3_bootstrap.PID_ABSENT
            and v3_bootstrap.pid_state(int(tunnel.attached_pid or 0))
            == v3_bootstrap.PID_ABSENT
        )
        assert "Connection reset" in tunnel.log_path.read_text(
            encoding="utf-8", errors="replace"
        )
        second = _tunnel(
            tmp_path,
            owner=tunnel.owner_path,
            local_port=tunnel.local_port,
            remote_port=tunnel.remote_port,
            run_id=tunnel.run_id,
            nonce=tunnel.nonce,
            fake_ssh=fake,
        )
        second.start()
        archives = _archive_dirs(tmp_path)
        assert len(archives) == 1
        names = {item.name for item in archives[0].iterdir()}
        for name in (
            "tunnel-state.json",
            "tunnel-owner.json",
            "tunnel-ack.json",
            "tunnel-starting.json",
            "tunnel-config.json",
        ):
            assert name in names
        second_owner = (second.owner_path.read_bytes()) if second.owner_path else b""
        second_ack = second.ack_path.read_bytes() if second.ack_path else b""
        with pytest.raises(v3_bootstrap.BootstrapError):
            tunnel.stop()
        assert second.owner_path is not None and second.owner_path.read_bytes() == second_owner
        assert second.ack_path is not None and second.ack_path.read_bytes() == second_ack
        second.verify_mapping()
        second.stop()
    finally:
        # The first monitor is already dead; a stale handle must not act on B.
        pass


def _full_bundle_payloads(tunnel: v3_bootstrap.OwnedTunnel) -> dict[str, dict]:
    state = _recovery_state(tunnel)
    owner = _sidecar_payload(tunnel, v3_bootstrap.TUNNEL_SCHEMA)
    owner["created_at"] = time.time()
    ack = _sidecar_payload(tunnel, v3_bootstrap.TUNNEL_ACK_SCHEMA)
    ack["status"] = "ARMED"
    heartbeat = _sidecar_payload(tunnel, v3_bootstrap.TUNNEL_HEARTBEAT_SCHEMA)
    starting = _sidecar_payload(tunnel, v3_bootstrap.TUNNEL_STARTING_SCHEMA)
    starting.update(status="CLEANED", cleanup_confirmed=True)
    recovery = _sidecar_payload(tunnel, v3_bootstrap.TUNNEL_RECOVERY_SCHEMA)
    recovery.update(status="CLEANED", cleanup_confirmed=True)
    return {
        "state": state,
        "config": tunnel._config_payload(),
        "owner": owner,
        "ack": ack,
        "heartbeat": heartbeat,
        "starting": starting,
        "recovery": recovery,
    }


def _write_full_bundle(tunnel: v3_bootstrap.OwnedTunnel) -> dict[str, bytes]:
    payloads = _full_bundle_payloads(tunnel)
    paths = tunnel._canonical_record_paths()
    written: dict[str, bytes] = {}
    for name, payload in payloads.items():
        data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        paths[name].write_bytes(data)
        written[name] = data
    return written


def test_recovery_mutation_table_blocks_without_changes(
    tmp_path: Path, monkeypatch
) -> None:
    nan = float("nan")
    inf = float("inf")
    cases: list[tuple[str, str, object]] = []
    for record in ("owner", "ack", "heartbeat", "starting", "recovery"):
        cases.extend([
            (record, "schema", "wrong"),
            (record, "run_id", "foreign"),
            (record, "nonce", "foreign"),
            (record, "fingerprint", "SHA256:wrong"),
            (record, "command_sha256", "0" * 64),
            (record, "local_port", 65000),
            ("__control__", "", record),
        ])
    for record in ("owner", "ack", "starting", "recovery"):
        cases.append((record, "destination", "root@foreign"))
    cases.extend([
        ("owner", "created_at", nan),
        ("owner", "created_at", inf),
        ("owner", "created_at", -1),
        ("owner", "created_at", True),
        ("owner", "__missing__", "created_at"),
        ("owner", "monitor_pid", True),
        ("owner", "ssh_pid", None),
        ("ack", "ack_at", nan),
        ("ack", "ack_at", -1),
        ("ack", "ack_at", True),
        ("ack", "__missing__", "ack_at"),
        ("heartbeat", "at", nan),
        ("heartbeat", "at", inf),
        ("heartbeat", "at", -1),
        ("heartbeat", "__missing__", "at"),
        ("starting", "at", nan),
        ("starting", "at", -1),
        ("starting", "__missing__", "at"),
        ("starting", "cleanup_confirmed", False),
        ("recovery", "at", nan),
        ("recovery", "at", -1),
        ("recovery", "__missing__", "at"),
        ("recovery", "cleanup_confirmed", False),
        ("state", "schema", "wrong"),
        ("state", "run_id", "foreign"),
        ("state", "nonce", "foreign"),
        ("state", "status", "RUNNING"),
        ("state", "cleanup_confirmed", False),
        ("state", "at", nan),
        ("state", "at", inf),
        ("state", "at", -1),
        ("state", "at", True),
        ("state", "__missing__", "at"),
        ("state", "started_at", nan),
        ("state", "started_at", -1),
        ("state", "monitor_pid", True),
        ("state", "ssh_pid", None),
        ("config", "schema", "wrong"),
        ("config", "run_id", "foreign"),
        ("config", "nonce", "foreign"),
        ("config", "fingerprint", "SHA256:wrong"),
        ("config", "command_sha256", "0" * 64),
        ("config", "local_port", 65000),
        ("config", "argv", ["wrong"]),
    ])
    for index, (record, field, value) in enumerate(cases):
        case_dir = tmp_path / f"case-{index:03d}"
        case_dir.mkdir()
        tunnel = _tunnel(
            case_dir,
            owner=case_dir / "tunnel-owner.json",
            local_port=_free_port(),
            remote_port=_free_port(),
            run_id="run-mut",
            nonce="nonce-mut",
        )
        payloads = _full_bundle_payloads(tunnel)
        paths = tunnel._canonical_record_paths()
        if record == "__control__":
            payloads[str(value)]["control_port"] = 54321
        elif field == "__missing__":
            payloads[record].pop(str(value), None)
        else:
            payloads[record][field] = value
        before: dict[str, bytes] = {}
        for name, payload in payloads.items():
            data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            paths[name].write_bytes(data)
            before[name] = data
        calls = _forbid_popen(monkeypatch)
        with pytest.raises(v3_bootstrap.BootstrapError):
            tunnel.start()
        assert calls == [], (record, field, value)
        for name, data in before.items():
            assert paths[name].read_bytes() == data, (record, field, value, name)
        assert _archive_dirs(case_dir) == [], (record, field, value)
        monkeypatch.undo()


def test_owner_without_terminal_state_blocks_auto_archive(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-owner-only",
        nonce="nonce-owner-only",
    )
    v3_bootstrap._atomic_json(tunnel.config_path, tunnel._config_payload())
    owner = _sidecar_payload(tunnel, v3_bootstrap.TUNNEL_SCHEMA)
    owner["created_at"] = time.time()
    assert tunnel.owner_path is not None
    tunnel.owner_path.write_text(json.dumps(owner), encoding="utf-8")
    owner_before = tunnel.owner_path.read_bytes()
    config_before = tunnel.config_path.read_bytes()
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.owner_path.read_bytes() == owner_before
    assert tunnel.config_path.read_bytes() == config_before
    assert _archive_dirs(tmp_path) == []


def test_start_fail_closed_table_no_spawn_no_archive(tmp_path: Path, monkeypatch) -> None:
    cases = [
        ("cleanup_false", lambda state: state.update(cleanup_confirmed=False), None),
        ("status_running", lambda state: state.update(status="RUNNING"), None),
        ("foreign_run", lambda state: state.update(run_id="foreign"), None),
        ("bool_pid", lambda state: state.update(monitor_pid=True), None),
        ("null_ssh", lambda state: state.update(ssh_pid=None), None),
        ("live_monitor", lambda state: state.update(monitor_pid=os.getpid()), None),
        ("fingerprint", None, lambda config: config.update(fingerprint="SHA256:wrong")),
        ("argv", None, lambda config: config["argv"].__setitem__(0, "wrong")),
        ("bad_ports", None, lambda config: config.update(local_port=70000)),
    ]
    for name, state_mutator, config_mutator in cases:
        case_dir = tmp_path / name
        case_dir.mkdir()
        tunnel = _tunnel(
            case_dir,
            owner=case_dir / "tunnel-owner.json",
            local_port=_free_port(),
            remote_port=_free_port(),
            run_id="run-case",
            nonce="nonce-case",
        )
        config_bytes, state_bytes = _write_recovery_case(
            tunnel, state_mutator=state_mutator, config_mutator=config_mutator
        )
        calls = _forbid_popen(monkeypatch)
        with pytest.raises(v3_bootstrap.BootstrapError):
            tunnel.start()
        assert calls == []
        assert tunnel.config_path.read_bytes() == config_bytes
        assert tunnel.state_path is not None
        assert tunnel.state_path.read_bytes() == state_bytes
        assert _archive_dirs(case_dir) == []
        monkeypatch.undo()

    # Malformed state must not be ignored or archived.
    case_dir = tmp_path / "malformed"
    case_dir.mkdir()
    tunnel = _tunnel(
        case_dir,
        owner=case_dir / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-case",
        nonce="nonce-case",
    )
    v3_bootstrap._atomic_json(tunnel.config_path, tunnel._config_payload())
    assert tunnel.state_path is not None
    tunnel.state_path.write_bytes(b"{")
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.state_path.read_bytes() == b"{"
    assert _archive_dirs(case_dir) == []
    monkeypatch.undo()


def test_start_fail_closed_unknown_liveness(tmp_path: Path, monkeypatch) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-unknown",
        nonce="nonce-unknown",
    )
    config_bytes, state_bytes = _write_recovery_case(tunnel)
    calls = _forbid_popen(monkeypatch)
    monkeypatch.setattr(
        v3_bootstrap, "pid_state", lambda pid: v3_bootstrap.PID_UNKNOWN
    )
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.config_path.read_bytes() == config_bytes
    assert tunnel.state_path is not None
    assert tunnel.state_path.read_bytes() == state_bytes
    assert _archive_dirs(tmp_path) == []


def test_start_busy_port_does_not_archive(tmp_path: Path, monkeypatch) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-busy",
        nonce="nonce-busy",
    )
    config_bytes, state_bytes = _write_recovery_case(tunnel)
    calls = _forbid_popen(monkeypatch)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", tunnel.local_port))
        sock.listen(1)
        with pytest.raises(v3_bootstrap.BootstrapError):
            tunnel.start()
    assert calls == []
    assert tunnel.config_path.read_bytes() == config_bytes
    assert tunnel.state_path is not None
    assert tunnel.state_path.read_bytes() == state_bytes
    assert _archive_dirs(tmp_path) == []


def test_start_orphan_sidecar_and_incomplete_archive_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    orphan = _tunnel(
        orphan_dir,
        owner=orphan_dir / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-orphan",
        nonce="nonce-orphan",
    )
    (orphan_dir / "tunnel-ack.json").write_text("{}", encoding="utf-8")
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        orphan.start()
    assert calls == []
    assert _archive_dirs(orphan_dir) == []
    monkeypatch.undo()

    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    tunnel = _tunnel(
        incomplete_dir,
        owner=incomplete_dir / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-incomplete",
        nonce="nonce-incomplete",
    )
    archive = incomplete_dir / "archive" / "owned-tunnel-test"
    archive.mkdir(parents=True)
    (archive / ".incomplete").write_text("x", encoding="utf-8")
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert (archive / ".incomplete").is_file()


def test_archive_partial_failure_blocks_without_losing_canonical(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-partial",
        nonce="nonce-partial",
    )
    config_bytes, state_bytes = _write_recovery_case(tunnel)
    calls = _forbid_popen(monkeypatch)
    real_write = v3_bootstrap._write_exclusive_bytes
    count = {"n": 0}

    def failing_write(path, data):
        count["n"] += 1
        if count["n"] == 2:
            raise OSError("injected archive failure")
        return real_write(path, data)

    monkeypatch.setattr(v3_bootstrap, "_write_exclusive_bytes", failing_write)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.config_path.read_bytes() == config_bytes
    assert tunnel.state_path is not None
    assert tunnel.state_path.read_bytes() == state_bytes
    archives = _archive_dirs(tmp_path)
    assert len(archives) == 1
    assert (archives[0] / ".incomplete").is_file()
    # A second attempt must refuse until the incomplete archive is resolved.
    monkeypatch.setattr(v3_bootstrap, "_write_exclusive_bytes", real_write)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.state_path.read_bytes() == state_bytes


def _sidecar_payload(
    tunnel: v3_bootstrap.OwnedTunnel,
    schema: str,
    *,
    monitor: int = DEAD_MONITOR,
    ssh: int = DEAD_SSH,
    control_port: int = 12345,
) -> dict:
    return {
        "schema": schema,
        "run_id": tunnel.run_id,
        "nonce": tunnel.nonce,
        "monitor_pid": monitor,
        "ssh_pid": ssh,
        "pid": ssh,
        "control_port": control_port,
        "host": tunnel.identity.host,
        "user": tunnel.identity.user,
        "port": int(tunnel.identity.port),
        "destination": tunnel.identity.destination,
        "fingerprint": tunnel.expected_fingerprint,
        "local_port": int(tunnel.local_port),
        "remote_port": int(tunnel.remote_port),
        "command_sha256": tunnel._command_sha256(),
        "argv": list(tunnel._command_argv()),
        "ack_at": time.time(),
        "at": time.time(),
    }


def test_exact_config_only_blocks_without_spawn_or_rewrite(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-config-exact",
        nonce="nonce-config-exact",
    )
    v3_bootstrap._atomic_json(tunnel.config_path, tunnel._config_payload())
    before = tunnel.config_path.read_bytes()
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.config_path.read_bytes() == before
    assert _archive_dirs(tmp_path) == []


def test_popen_failure_keeps_config_for_manual_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-popen",
        nonce="nonce-popen",
    )

    def failing_launcher(*args, **kwargs):
        raise OSError("injected launcher failure")

    tunnel.popen = failing_launcher
    with pytest.raises(OSError):
        tunnel.start()
    assert tunnel.config_path.is_file()
    before = tunnel.config_path.read_bytes()
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.config_path.read_bytes() == before


def test_foreign_config_without_owner_state_blocks_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-config",
        nonce="nonce-config",
    )
    config = tunnel._config_payload()
    config["run_id"] = "foreign-run"
    v3_bootstrap._atomic_json(tunnel.config_path, config)
    before = tunnel.config_path.read_bytes()
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.config_path.read_bytes() == before
    assert _archive_dirs(tmp_path) == []


def test_recovery_unconfirmed_blocks_without_archive(tmp_path: Path, monkeypatch) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-recovery",
        nonce="nonce-recovery",
    )
    config_bytes, state_bytes = _write_recovery_case(tunnel)
    tunnel._write_recovery(
        monitor_pid=DEAD_MONITOR,
        ssh_pid=DEAD_SSH,
        control_port=12345,
        reason="post-ack:BootstrapError",
    )
    recovery_bytes = tunnel.recovery_path.read_bytes()
    calls = _forbid_popen(monkeypatch)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel.config_path.read_bytes() == config_bytes
    assert tunnel.state_path is not None
    assert tunnel.state_path.read_bytes() == state_bytes
    assert tunnel.recovery_path.read_bytes() == recovery_bytes
    assert _archive_dirs(tmp_path) == []


def test_current_attempt_terminal_state_still_blocks_sidecars(tmp_path: Path) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-current",
        nonce="nonce-current",
    )
    tunnel.monitor_pid = DEAD_MONITOR
    tunnel.attached_pid = DEAD_SSH
    tunnel.control_port = 12345
    assert tunnel.ack_path is not None and tunnel.heartbeat_path is not None
    assert tunnel.state_path is not None
    ack = _sidecar_payload(tunnel, v3_bootstrap.TUNNEL_ACK_SCHEMA)
    ack["status"] = "ARMED"
    tunnel.ack_path.write_text(json.dumps(ack), encoding="utf-8")
    heartbeat = _sidecar_payload(tunnel, v3_bootstrap.TUNNEL_HEARTBEAT_SCHEMA)
    tunnel.heartbeat_path.write_text(json.dumps(heartbeat), encoding="utf-8")
    tunnel.state_path.write_text(
        json.dumps(_recovery_state(tunnel)), encoding="utf-8"
    )
    with pytest.raises(v3_bootstrap.BootstrapError, match="terminal state"):
        tunnel._validate_sidecars(
            freshness_seconds=v3_bootstrap.DEFAULT_TUNNEL_FRESHNESS_SECONDS
        )
    details = v3_bootstrap.OwnedTunnel._post_ack_failure_details(
        v3_bootstrap.BootstrapError("tunnel monitor has recorded a terminal state")
    )
    assert details == ("post_ack.sidecars", "terminal_state")


def _recovery_bundle_bytes(tunnel: v3_bootstrap.OwnedTunnel) -> dict[str, bytes]:
    return _write_full_bundle(tunnel)


def test_archive_retired_marker_failure_preserves_raw_and_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-late-retired",
        nonce="nonce-late-retired",
    )
    before = _recovery_bundle_bytes(tunnel)
    calls = _forbid_popen(monkeypatch)
    real_write = v3_bootstrap._write_exclusive_bytes

    def failing_write(path, data):
        if Path(path).name == ".retired":
            raise OSError("injected retired marker failure")
        return real_write(path, data)

    monkeypatch.setattr(v3_bootstrap, "_write_exclusive_bytes", failing_write)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    archives = _archive_dirs(tmp_path)
    assert len(archives) == 1
    assert (archives[0] / ".retiring").is_file()
    assert not (archives[0] / ".retired").exists()
    assert (archives[0] / "tunnel-state.json").read_bytes() == before["state"]
    assert (archives[0] / "tunnel-config.json").read_bytes() == before["config"]
    monkeypatch.setattr(v3_bootstrap, "_write_exclusive_bytes", real_write)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []


def test_archive_retiring_unlink_failure_preserves_raw_and_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-late-unlink",
        nonce="nonce-late-unlink",
    )
    before = _recovery_bundle_bytes(tunnel)
    calls = _forbid_popen(monkeypatch)
    real_unlink = os.unlink
    failed = {"value": False}

    def failing_unlink(path, *args, **kwargs):
        if Path(path).name == ".retiring" and not failed["value"]:
            failed["value"] = True
            raise OSError("injected retiring unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(v3_bootstrap.os, "unlink", failing_unlink)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    archives = _archive_dirs(tmp_path)
    assert len(archives) == 1
    assert (archives[0] / ".retiring").is_file()
    assert (archives[0] / "tunnel-state.json").read_bytes() == before["state"]
    assert (archives[0] / "tunnel-config.json").read_bytes() == before["config"]
    monkeypatch.setattr(v3_bootstrap.os, "unlink", real_unlink)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []


def test_archive_last_fsync_failure_restores_blocking_marker(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-last-fsync",
        nonce="nonce-last-fsync",
    )
    before = _recovery_bundle_bytes(tunnel)
    calls = _forbid_popen(monkeypatch)
    real_fsync = v3_bootstrap._fsync_directory
    failed = {"value": False}

    def failing_fsync(path):
        target = Path(path)
        if (
            not failed["value"]
            and target.name.startswith("owned-tunnel-")
            and not (target / ".retiring").exists()
        ):
            failed["value"] = True
            raise OSError("injected final fsync failure")
        return real_fsync(path)

    monkeypatch.setattr(v3_bootstrap, "_fsync_directory", failing_fsync)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    archives = _archive_dirs(tmp_path)
    assert len(archives) == 1
    assert (archives[0] / ".retirement_blocked").is_file()
    assert (archives[0] / "tunnel-state.json").read_bytes() == before["state"]
    assert (archives[0] / "tunnel-config.json").read_bytes() == before["config"]
    monkeypatch.setattr(v3_bootstrap, "_fsync_directory", real_fsync)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    # A new object must also see the blocking marker, not auto-archive.
    fresh = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=tunnel.local_port,
        remote_port=tunnel.remote_port,
        run_id=tunnel.run_id,
        nonce=tunnel.nonce,
    )
    with pytest.raises(v3_bootstrap.BootstrapError):
        fresh.start()
    assert calls == []


def test_archive_last_fsync_marker_restore_failure_latches(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-last-fsync-latch",
        nonce="nonce-last-fsync-latch",
    )
    _recovery_bundle_bytes(tunnel)
    calls = _forbid_popen(monkeypatch)
    real_fsync = v3_bootstrap._fsync_directory
    real_write = v3_bootstrap._write_exclusive_bytes
    failed = {"value": False}

    def failing_fsync(path):
        target = Path(path)
        if (
            not failed["value"]
            and target.name.startswith("owned-tunnel-")
            and not (target / ".retiring").exists()
        ):
            failed["value"] = True
            raise OSError("injected final fsync failure")
        return real_fsync(path)

    def failing_marker_write(path, data):
        if Path(path).name == ".retirement_blocked":
            raise OSError("injected marker restore failure")
        return real_write(path, data)

    monkeypatch.setattr(v3_bootstrap, "_fsync_directory", failing_fsync)
    monkeypatch.setattr(v3_bootstrap, "_write_exclusive_bytes", failing_marker_write)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert tunnel._archive_latched is True
    monkeypatch.setattr(v3_bootstrap, "_fsync_directory", real_fsync)
    monkeypatch.setattr(v3_bootstrap, "_write_exclusive_bytes", real_write)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    assert (tmp_path / ".tunnel-archive-critical").is_file()


def test_archive_partial_canonical_retirement_preserves_raw_and_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    tunnel = _tunnel(
        tmp_path,
        owner=tmp_path / "tunnel-owner.json",
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-late-partial",
        nonce="nonce-late-partial",
    )
    before = _recovery_bundle_bytes(tunnel)
    calls = _forbid_popen(monkeypatch)
    real_unlink = os.unlink
    canonical_unlinks = {"n": 0}

    def failing_unlink(path, *args, **kwargs):
        target = Path(path)
        if target.parent == tmp_path and target.name in {
            "tunnel-state.json",
            "tunnel-owner.json",
            "tunnel-ack.json",
            "tunnel-heartbeat.json",
            "tunnel-starting.json",
            "tunnel-recovery.json",
            "tunnel-config.json",
        }:
            canonical_unlinks["n"] += 1
            if canonical_unlinks["n"] == 3:
                raise OSError("injected canonical retirement failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(v3_bootstrap.os, "unlink", failing_unlink)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []
    archives = _archive_dirs(tmp_path)
    assert len(archives) == 1
    assert (archives[0] / ".retiring").is_file()
    assert (archives[0] / "tunnel-state.json").read_bytes() == before["state"]
    assert (archives[0] / "tunnel-config.json").read_bytes() == before["config"]
    monkeypatch.setattr(v3_bootstrap.os, "unlink", real_unlink)
    with pytest.raises(v3_bootstrap.BootstrapError):
        tunnel.start()
    assert calls == []


def test_concurrent_start_spawns_only_one_monitor(tmp_path: Path, monkeypatch) -> None:
    owner = tmp_path / "tunnel-owner.json"
    first = _tunnel(
        tmp_path,
        owner=owner,
        local_port=_free_port(),
        remote_port=_free_port(),
        run_id="run-race",
        nonce="nonce-race",
    )
    second = _tunnel(
        tmp_path,
        owner=owner,
        local_port=first.local_port,
        remote_port=first.remote_port,
        run_id="run-race",
        nonce="nonce-race",
    )
    barrier = threading.Barrier(2)
    results: list[v3_bootstrap.OwnedTunnel] = []
    errors: list[BaseException] = []
    popen_calls: list[object] = []
    real_popen = subprocess.Popen

    def counting_popen(*args, **kwargs):
        popen_calls.append(args)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(v3_bootstrap.subprocess, "Popen", counting_popen)

    def run(tunnel):
        barrier.wait()
        try:
            tunnel.start()
            results.append(tunnel)
        except BaseException as exc:  # noqa: BLE001 - assertion captures the class
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(item,)) for item in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(results) == 1
    assert len(errors) == 1
    assert len(popen_calls) == 1
    if results:
        results[0].stop()
