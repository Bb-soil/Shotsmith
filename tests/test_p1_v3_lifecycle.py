from __future__ import annotations

import json
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.p1 import clore_session as cs
from scripts.p1 import v3_lifecycle as lc
from scripts.p1.remote import RemoteIdentity, run_stream_to_file


FIXTURE = Path(__file__).parent / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json"


def _session_state(tmp_path: Path) -> tuple[Path, cs.CloreSessionState]:
    quote = cs.CloreQuote.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))
    now = datetime.now(timezone.utc)
    state_path = tmp_path / "state.json"
    state = cs.initialize_state(
        state_path,
        order_id=91001,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        now=now,
    )
    return state_path, state


def _identity(tmp_path: Path) -> RemoteIdentity:
    identity = tmp_path / "id"
    known = tmp_path / "known_hosts"
    identity.write_text("not-a-key", encoding="utf-8")
    known.write_text("[example.test]:22 ssh-ed25519 AAAA\n", encoding="utf-8")
    return RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=identity,
        known_hosts=known,
    )


def _bundle_hashes() -> dict[str, str]:
    return {key: "a" * 64 for key in ("config", "workflow", "manifest", "job", "source", "derived", "policy")}


def _write_supervisor_child(
    path: Path,
    *,
    delay: float = 0.0,
    status: str = "ARMED",
    heartbeat: bool = True,
    tail: float = 1.0,
) -> None:
    path.write_text(
        "import hashlib,json,os,sys,time,pathlib\n"
        "config_path=pathlib.Path(sys.argv[1])\n"
        "spawn_log=pathlib.Path(sys.argv[2])\n"
        "with spawn_log.open('a',encoding='utf-8') as handle: handle.write(f'{os.getpid()}\\n')\n"
        "cfg=json.loads(config_path.read_text())\n"
        "config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest()\n"
        "binding_sha=hashlib.sha256(json.dumps(cfg['session'],sort_keys=True,separators=(',',':')).encode()).hexdigest()\n"
        "common={'run_id':cfg['run_id'],'nonce':cfg['nonce'],'pid':os.getpid(),'config_sha256':config_sha,'binding_sha256':binding_sha}\n"
        f"time.sleep({delay!r})\n"
        + (
            "pathlib.Path(cfg['heartbeat_path']).write_text(json.dumps({'schema':'anime-video-studio/p1-v3-supervisor-heartbeat/v1','at':time.time(),**common}))\n"
            if heartbeat
            else ""
        )
        + "pathlib.Path(cfg['ack_path']).write_text(json.dumps({'schema':'anime-video-studio/p1-v3-supervisor-ack/v1','status':"
        + repr(status)
        + ",'order_id':cfg['session']['order_id'],'created_at':cfg['session']['created_at'],'ack_at':time.time(),**common}))\n"
        f"time.sleep({tail!r})\n",
        encoding="utf-8",
    )


def _write_dead_supervisor_child(path: Path) -> None:
    path.write_text(
        "import os,sys,pathlib\n"
        "pathlib.Path(sys.argv[2]).open('a',encoding='utf-8').write(f'{os.getpid()}\\n')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )


def _kill(pid: int) -> None:
    import subprocess

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


def _supervisor(
    tmp_path: Path,
    child: Path,
    spawn_log: Path,
    *,
    nonce: str = "nonce-retry",
    state_path: Path | None = None,
):
    import sys

    if state_path is None:
        state_path, state = _session_state(tmp_path)
        quote_path = tmp_path / "quote.json"
        quote_path.write_text(json.dumps(state.quote.to_dict()), encoding="utf-8")
    else:
        state = cs.load_state(state_path)
        quote_path = tmp_path / "quote.json"
    binding = lc.SessionBinding.from_state(state)
    supervisor = lc.CloreSupervisor(
        session_state_path=state_path,
        quote_path=quote_path,
        binding=binding,
        log_path=tmp_path / "supervisor.log",
        record_path=tmp_path / "supervisor.json",
        run_id="run-retry",
        nonce=nonce,
        child_argv=(
            sys.executable,
            str(child),
            str(tmp_path / "supervisor-config.json"),
            str(spawn_log),
        ),
    )
    return supervisor


def test_session_binding_is_exact_and_roundtrips(tmp_path: Path) -> None:
    state_path, state = _session_state(tmp_path)
    binding = lc.SessionBinding.from_state(state)
    assert binding.order_id == 91001
    assert binding.server_id == state.server_id
    assert binding.created_at == state.created_at
    assert binding.cap_usd == str(state.cap_usd)
    assert binding.quote_hash
    assert lc.SessionBinding.from_dict(binding.to_dict()) == binding
    binding.assert_matches_state(state)
    lc.assert_deadline(binding)
    lc.assert_budget(binding, state)
    loaded, loaded_binding = lc.load_session_binding(
        state_path,
        expected_order_id=binding.order_id,
        expected_created_at=binding.created_at,
    )
    assert loaded_binding == binding
    with pytest.raises(lc.LifecycleError):
        lc.load_session_binding(
            state_path,
            expected_order_id=999,
            expected_created_at=binding.created_at,
        )


def test_lifecycle_roundtrip_and_incomplete_rejected(tmp_path: Path) -> None:
    _, state = _session_state(tmp_path)
    binding = lc.SessionBinding.from_state(state)
    lifecycle = lc.new_lifecycle(
        run_id="run-1",
        session=binding,
        bundle_hashes=_bundle_hashes(),
    )
    path = tmp_path / "lifecycle.json"
    lc.write_lifecycle(path, lifecycle)
    loaded = lc.load_lifecycle(path)
    assert loaded.run_id == "run-1"
    assert loaded.nonce
    assert loaded.session == binding
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["session"].pop("quote_hash")
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(lc.LifecycleError):
        lc.load_lifecycle(path)


def test_target_lock_is_exclusive_and_owner_bound(tmp_path: Path) -> None:
    first = lc.TargetLock(tmp_path, run_id="run-1", nonce="n1").acquire()
    try:
        first.assert_owned()
        with pytest.raises(lc.LifecycleError):
            lc.TargetLock(tmp_path, run_id="run-2", nonce="n2").acquire()
    finally:
        first.release()
    second = lc.TargetLock(tmp_path, run_id="run-2", nonce="n2").acquire()
    second.release()
    assert not (tmp_path / ".p1-download.owner.json").exists()


def test_progress_recorder_rejects_invalid_counters(tmp_path: Path) -> None:
    recorder = lc.ProgressRecorder(
        path=tmp_path / "progress.json",
        run_id="r",
        nonce="n",
        files_total=1,
        bytes_total=10,
    )
    recorder.callback(5, 10)
    assert json.loads((tmp_path / "progress.json").read_text())["bytes_done"] == 5
    with pytest.raises(lc.LifecycleError):
        recorder.callback(11, 10)


def test_disk_and_deadline_fail_closed(tmp_path: Path) -> None:
    _, state = _session_state(tmp_path)
    binding = lc.SessionBinding.from_state(state)
    with pytest.raises(lc.LifecycleError):
        lc.assert_deadline(binding, now=binding.hard_end_at)
    with pytest.raises(lc.LifecycleError):
        lc.assert_disk(tmp_path / "missing", 1)
    lc.assert_disk(tmp_path, 1)


def test_supervisor_start_is_single_owner_and_readback(tmp_path: Path) -> None:
    import sys
    import time

    state_path, state = _session_state(tmp_path)
    quote_path = tmp_path / "quote.json"
    quote_path.write_text(json.dumps(state.quote.to_dict()), encoding="utf-8")
    record_path = tmp_path / "supervisor.json"
    log_path = tmp_path / "supervisor.log"
    binding = lc.SessionBinding.from_state(state)
    child = tmp_path / "child.py"
    child.write_text(
        "import hashlib,json,sys,time,pathlib,os\n"
        "config_path=pathlib.Path(sys.argv[1])\n"
        "cfg=json.loads(config_path.read_text())\n"
        "config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest()\n"
        "binding_sha=hashlib.sha256(json.dumps(cfg['session'],sort_keys=True,separators=(',',':')).encode()).hexdigest()\n"
        "common={'run_id':cfg['run_id'],'nonce':cfg['nonce'],'pid':os.getpid(),'config_sha256':config_sha,'binding_sha256':binding_sha}\n"
        "pathlib.Path(cfg['heartbeat_path']).write_text(json.dumps({'schema':'anime-video-studio/p1-v3-supervisor-heartbeat/v1','at':time.time(),**common}))\n"
        "pathlib.Path(cfg['ack_path']).write_text(json.dumps({'schema':'anime-video-studio/p1-v3-supervisor-ack/v1','status':'ARMED','order_id':cfg['session']['order_id'],'created_at':cfg['session']['created_at'],'ack_at':time.time(),**common}))\n"
        "time.sleep(1.0)\n",
        encoding="utf-8",
    )
    supervisor = lc.CloreSupervisor(
        session_state_path=state_path,
        quote_path=quote_path,
        binding=binding,
        log_path=log_path,
        record_path=record_path,
        run_id="run-1",
        nonce="nonce-1",
        child_argv=(sys.executable, str(child)),
    )
    # child_argv needs the config path appended by the parent; use a wrapper
    supervisor._child_argv = (sys.executable, str(child), str(supervisor.config_path))
    record = supervisor.start(ack_timeout=5.0)
    assert record.pid > 0
    assert json.loads(record_path.read_text())["nonce"] == "nonce-1"
    assert supervisor.readback() == state
    with pytest.raises(lc.LifecycleError):
        supervisor.start()

    raw = json.loads(record_path.read_text())
    raw["nonce"] = "foreign"
    record_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(lc.LifecycleError):
        supervisor.readback()


def test_supervisor_timeout_retains_same_starting_attempt_and_resumes(tmp_path: Path) -> None:
    child = tmp_path / "slow_child.py"
    spawn_log = tmp_path / "spawn.log"
    _write_supervisor_child(child, delay=1.2)
    first = _supervisor(tmp_path, child, spawn_log)
    with pytest.raises(lc.SupervisorPending) as pending:
        first.start(ack_timeout=0.3)
    assert "no second child" in str(pending.value)
    attempt = json.loads((tmp_path / "supervisor-attempt.json").read_text())
    assert attempt["status"] == "STARTING"
    first_pid = attempt["pid"]
    assert isinstance(first_pid, int)
    assert len(spawn_log.read_text(encoding="utf-8").splitlines()) == 1

    second = _supervisor(
        tmp_path, child, spawn_log, state_path=first.session_state_path
    )
    record = second.start(ack_timeout=5.0)
    assert record.pid > 0
    assert len(spawn_log.read_text(encoding="utf-8").splitlines()) == 1
    published = json.loads((tmp_path / "supervisor.json").read_text())
    assert published["pid"] == record.pid
    attempt_after = json.loads((tmp_path / "supervisor-attempt.json").read_text())
    assert attempt_after["status"] == "ARMED"
    assert attempt_after["pid"] == record.pid
    _kill(record.pid)


def test_supervisor_dead_attempt_is_stable_and_never_respawned(tmp_path: Path) -> None:
    child = tmp_path / "dead_child.py"
    spawn_log = tmp_path / "spawn.log"
    _write_dead_supervisor_child(child)
    first = _supervisor(tmp_path, child, spawn_log)
    with pytest.raises(lc.LifecycleError):
        first.start(ack_timeout=2.0)
    attempt = json.loads((tmp_path / "supervisor-attempt.json").read_text())
    assert attempt["status"] == "DIED_BEFORE_ACK"
    assert not (tmp_path / "supervisor.json").exists()
    assert len(spawn_log.read_text(encoding="utf-8").splitlines()) == 1

    second = _supervisor(
        tmp_path, child, spawn_log, state_path=first.session_state_path
    )
    with pytest.raises(lc.LifecycleError) as manual:
        second.start(ack_timeout=2.0)
    assert "manual recovery" in str(manual.value)
    assert len(spawn_log.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(lc.LifecycleError):
        second.readback()


def test_supervisor_terminal_ack_refuses_ownership_record(tmp_path: Path) -> None:
    child = tmp_path / "terminal_child.py"
    spawn_log = tmp_path / "spawn.log"
    _write_supervisor_child(
        child, delay=0.0, status="LIVE_BLOCKED", heartbeat=False, tail=0.2
    )
    supervisor = _supervisor(tmp_path, child, spawn_log)
    with pytest.raises(lc.LifecycleError) as terminal:
        supervisor.start(ack_timeout=3.0)
    assert "refused activity" in str(terminal.value)
    assert not (tmp_path / "supervisor.json").exists()
    attempt = json.loads((tmp_path / "supervisor-attempt.json").read_text())
    assert attempt["status"] == "REFUSED_LIVE_BLOCKED"
    assert len(spawn_log.read_text(encoding="utf-8").splitlines()) == 1


def test_supervisor_readback_refuses_expired_deadline_before_io(tmp_path: Path) -> None:
    supervisor = _supervisor(
        tmp_path, tmp_path / "unused.py", tmp_path / "spawn.log"
    )
    with pytest.raises(lc.LifecycleError):
        supervisor.readback(deadline_monotonic=time.monotonic() - 0.1)
    with pytest.raises(lc.LifecycleError):
        supervisor.start(deadline_monotonic=time.monotonic() - 0.1)


def test_host_key_fingerprint_matches_only_selected_host_key(
    tmp_path: Path, monkeypatch
) -> None:
    import base64
    import hashlib

    identity = _identity(tmp_path)
    selected = b"selected-key-blob"
    other = b"other-key-blob"
    expected = "SHA256:" + base64.b64encode(
        hashlib.sha256(selected).digest()
    ).decode().rstrip("=")
    matched = (
        f"[example.test]:22 ssh-ed25519 {base64.b64encode(selected).decode()}\n"
    ).encode()

    class Runner:
        ssh_keygen_path = "ssh-keygen"

    monkeypatch.setattr(
        lc,
        "run_bounded_process",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            timed_out=False,
            stdout=matched,
        ),
    )
    lc.verify_host_key(
        runner=Runner(),
        identity=identity,
        expected_fingerprint=expected,
        known_hosts=identity.known_hosts,
    )
    with pytest.raises(lc.LifecycleError):
        lc.verify_host_key(
            runner=Runner(),
            identity=identity,
            expected_fingerprint="SHA256:" + base64.b64encode(
                hashlib.sha256(b"wrong").digest()
            ).decode().rstrip("="),
            known_hosts=identity.known_hosts,
        )
    with pytest.raises(lc.LifecycleError):
        lc.verify_host_key(
            runner=Runner(),
            identity=identity,
            expected_fingerprint="SHA256:" + base64.b64encode(
                hashlib.sha256(other).digest()
            ).decode().rstrip("="),
            known_hosts=identity.known_hosts,
        )


def test_prepare_cli_binds_session_offline_and_draft_refuses_live(tmp_path: Path, capsys, monkeypatch) -> None:
    from scripts.p1 import v3_operator
    monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")

    state_path, state = _session_state(tmp_path)
    run_dir = tmp_path / "run"
    code = v3_operator.main(
        [
            "prepare",
            "--run-dir",
            str(run_dir),
            "--session-state",
            str(state_path),
            "--order-id",
            str(state.order_id),
            "--created-at",
            str(state.created_at),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lifecycle_phase"] == "BOUND_OFFLINE"
    lifecycle = json.loads((run_dir / "lifecycle.json").read_text())
    assert lifecycle["session"]["order_id"] == state.order_id
    assert lifecycle["session"]["quote_hash"]

    code = v3_operator.main(
        [
            "prepare",
            "--live",
            "--run-dir",
            str(tmp_path / "blocked"),
            "--session-state",
            str(state_path),
            "--order-id",
            str(state.order_id),
            "--created-at",
            str(state.created_at),
        ]
    )
    assert code == 2
    assert not (tmp_path / "blocked").exists()


def test_prepare_execute_downloads_requires_live(tmp_path: Path, capsys) -> None:
    from scripts.p1 import v3_operator

    code = v3_operator.main(
        ["prepare", "--execute-downloads", "--run-dir", str(tmp_path / "run")]
    )
    assert code == 2
    assert not (tmp_path / "run").exists()


class _LocalStreamRunner:
    def stream_remote_to_file(self, identity, remote_args, target, *, timeout, max_payload_bytes, **kwargs):
        import subprocess

        completed = subprocess.run(
            [__import__("sys").executable, *remote_args[1:]],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("local helper failed")
        header, payload = completed.stdout.split(b"\n", 1)
        target.write_bytes(payload)
        return header, len(payload)


def test_fetch_exact_stream_full_and_resume(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lc, "_validate_remote_absolute_path", lambda value, name: value)
    source = tmp_path / "artifact.mp4"
    source.write_bytes(b"0123456789" * 100)
    helper = str(Path(__file__).resolve().parents[1] / "scripts" / "p1" / "v3_remote_fetch_helper.py")
    runner = _LocalStreamRunner()
    import hashlib
    full_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    final = tmp_path / "final.mp4"
    result = lc.fetch_exact_stream(
        runner=runner,
        identity=_identity(tmp_path),
        remote_helper_path=helper,
        remote_path=str(source),
        expected_size=source.stat().st_size,
        expected_sha256=full_sha,
        local_path=final,
    )
    assert result["sha256"] == full_sha
    assert final.read_bytes() == source.read_bytes()

    partial = tmp_path / "partial.mp4"
    prefix = source.read_bytes()[:100]
    partial.write_bytes(prefix)
    resumed = lc.fetch_exact_stream(
        runner=runner,
        identity=_identity(tmp_path),
        remote_helper_path=helper,
        remote_path=str(source),
        expected_size=source.stat().st_size,
        expected_sha256=full_sha,
        local_path=partial,
        offset=100,
        expected_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
    )
    assert resumed["size"] == source.stat().st_size
    assert partial.read_bytes() == source.read_bytes()


def test_fetch_exact_stream_rejects_wrong_hash_and_oversize(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lc, "_validate_remote_absolute_path", lambda value, name: value)
    import hashlib
    source = tmp_path / "artifact.mp4"
    source.write_bytes(b"x" * 100)
    helper = str(Path(__file__).resolve().parents[1] / "scripts" / "p1" / "v3_remote_fetch_helper.py")
    runner = _LocalStreamRunner()
    with pytest.raises(lc.LifecycleError):
        lc.fetch_exact_stream(
            runner=runner,
            identity=_identity(tmp_path),
            remote_helper_path=helper,
            remote_path=str(source),
            expected_size=100,
            expected_sha256="0" * 64,
            local_path=tmp_path / "bad.mp4",
        )
    with pytest.raises(lc.LifecycleError):
        lc.fetch_exact_stream(
            runner=runner,
            identity=_identity(tmp_path),
            remote_helper_path=helper,
            remote_path=str(source),
            expected_size=99,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            local_path=tmp_path / "cap.mp4",
        )


def test_remote_worker_stages_fixed_files_and_reads_status(tmp_path: Path) -> None:
    import json
    from types import SimpleNamespace

    puts = []
    commands = []
    root = "/workspace/p1-run"
    files: dict[str, dict] = {}

    class Runner:
        def put_file(self, identity, remote_path, local_path, **kwargs):
            puts.append((remote_path, Path(local_path).name, kwargs))
            if Path(remote_path).name == "launch.json":
                files["launch.json"] = json.loads(Path(local_path).read_text(encoding="utf-8"))

        def run_remote(self, identity, args, timeout=None, **kwargs):
            commands.append(args)
            if args[0] == "df":
                return SimpleNamespace(
                    returncode=0,
                    stdout="Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 1000000 1 999999 1% /workspace\n",
                    timed_out=False,
                )
            if args[0] == "python3" and "v3_remote_launcher.py" in args[1]:
                launch = files["launch.json"]
                command_hash = hashlib.sha256(
                    json.dumps(launch, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                owner = {
                    "schema": "anime-video-studio/p1-v3-process-owner/v1",
                    "kind": "download_worker",
                    "run_id": "r",
                    "nonce": "n",
                    "command_sha256": command_hash,
                    "pid": 12345,
                }
                files["worker-owner.json"] = owner
                files["worker-ack.json"] = dict(owner)
                files["status.json"] = {
                    "run_id": "r",
                    "nonce": "n",
                    "status": "COMPLETE",
                }
                files["progress.json"] = {
                    "run_id": "r",
                    "nonce": "n",
                    "status": "COMPLETE",
                    "files_done": 3,
                    "files_total": 3,
                    "bytes_done": 30,
                    "bytes_total": 30,
                }
                files["heartbeat.json"] = {
                    "run_id": "r",
                    "nonce": "n",
                    "status": "COMPLETE",
                }
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "schema": "anime-video-studio/p1-v3-launch-result/v1",
                            "kind": "download_worker",
                            "run_id": "r",
                            "nonce": "n",
                            "status": "STARTED",
                            "command_sha256": command_hash,
                            "pid": 12345,
                        }
                    ),
                    timed_out=False,
                )
            return SimpleNamespace(returncode=0, stdout="", timed_out=False)

        def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout, **kwargs):
            name = remote_path.rsplit("/", 1)[-1]
            if name not in files:
                raise lc.LifecycleError("missing")
            return files[name]

    result = lc.run_remote_download_worker(
        runner=Runner(),
        identity=_identity(tmp_path),
        remote_dir=root,
        local_worker=Path(__file__).resolve().parents[1] / "scripts" / "p1" / "v3_download_worker.py",
        local_setup_plan=Path(__file__).resolve().parents[1] / "scripts" / "p1" / "setup_plan.py",
        local_manifest=Path(__file__).resolve().parents[1] / "manifests" / "P1.wan22-5b-shot01-v3.draft.json",
        payload={
            "run_id": "r",
            "nonce": "n",
            "manifest_sha256": hashlib.sha256(
                (
                    Path(__file__).resolve().parents[1]
                    / "manifests"
                    / "P1.wan22-5b-shot01-v3.draft.json"
                ).read_bytes()
            ).hexdigest(),
        },
        timeout=5.0,
    )
    assert result["status"] == "COMPLETE"
    assert len([item for item in puts if item[0].startswith(root + "/")]) == 7
    assert all("root" in item[2] for item in puts)
    python_commands = [command for command in commands if command[0] == "python3"]
    assert python_commands
    assert python_commands[0][1].endswith("v3_remote_launcher.py")
    assert "v3_download_worker.py" not in " ".join(python_commands[0])


def test_prepare_resume_preserves_nonce_and_refuses_other_path(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.p1 import v3_operator

    monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
    state_path, state = _session_state(tmp_path)
    args = [
        "prepare",
        "--session-state",
        str(state_path),
        "--order-id",
        str(state.order_id),
        "--created-at",
        str(state.created_at),
    ]
    run = tmp_path / "run"
    assert v3_operator.main(args + ["--run-dir", str(run)]) == 0
    first = json.loads((run / "lifecycle.json").read_text())
    assert v3_operator.main(args + ["--run-dir", str(run)]) == 0
    second = json.loads((run / "lifecycle.json").read_text())
    assert first["nonce"] == second["nonce"]
    assert first["session"] == second["session"]
    other = tmp_path / "other"
    assert v3_operator.main(args + ["--run-dir", str(other)]) == 2
    assert not other.exists()


def test_supervisor_child_cli_validates_full_binding(tmp_path: Path) -> None:
    import subprocess
    import sys
    from datetime import datetime, timezone

    quote = cs.CloreQuote.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))
    now = datetime.now(timezone.utc)
    state_path = tmp_path / "state.json"
    state = cs.initialize_state(
        state_path,
        order_id=93001,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        cap_usd=__import__("decimal").Decimal("1.00"),
        duration_seconds=3600,
        now=now,
    )
    binding = lc.SessionBinding.from_state(state)
    config = tmp_path / "supervisor-config.json"
    config.write_text(
        json.dumps(
            {
                "schema": "anime-video-studio/p1-v3-supervisor-config/v1",
                "run_id": "run-cli",
                "nonce": "nonce-cli",
                "state_path": str(state_path),
                "quote_path": str(tmp_path / "quote.json"),
                "ack_path": str(tmp_path / "ack.json"),
                "heartbeat_path": str(tmp_path / "heartbeat.json"),
                "session": binding.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    child = Path(__file__).resolve().parents[1] / "scripts" / "p1" / "v3_supervisor_child.py"
    ok = subprocess.run(
        [sys.executable, str(child), "--config", str(config), "--validate-config"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert ok.returncode == 0, ok.stderr
    wrong = json.loads(config.read_text())
    wrong["session"]["cap_usd"] = "0.99"
    config.write_text(json.dumps(wrong), encoding="utf-8")
    bad = subprocess.run(
        [sys.executable, str(child), "--config", str(config), "--validate-config"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert bad.returncode == 2

def test_supervisor_child_only_publishes_ready_ack_after_armed(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    from scripts.p1 import v3_supervisor_child

    state_path, state = _session_state(tmp_path)
    binding = lc.SessionBinding.from_state(state)
    config = {
        "state_path": str(state_path),
        "ack_path": str(tmp_path / "ack.json"),
        "heartbeat_path": str(tmp_path / "heartbeat.json"),
        "run_id": "run-child",
        "nonce": "nonce-child",
        "session": binding.to_dict(),
    }

    class FakeBridge:
        async def aclose(self):
            return None

    class FakeSession:
        readback_poll_seconds = 0.01

        def __init__(self, path, bridge):
            self.statuses = iter(["ARMED", "LIVE_BLOCKED"])

        async def evaluate(self):
            return SimpleNamespace(status=next(self.statuses, "LIVE_BLOCKED"))

    monkeypatch.setattr(
        v3_supervisor_child, "_load_bridge", lambda expected_server_id=None: FakeBridge()
    )
    monkeypatch.setattr(v3_supervisor_child.clore_session, "CloreSession", FakeSession)
    assert asyncio.run(v3_supervisor_child._watch(config)) == 2
    ack = json.loads((tmp_path / "ack.json").read_text())
    # A terminal evaluation must replace the ARMED ack and remove the
    # heartbeat so no later readback can accept activity.
    assert ack["status"] == "LIVE_BLOCKED"
    assert not (tmp_path / "heartbeat.json").exists()

    for status in ("LIVE_BLOCKED", "ENDED_REPORTED"):
        (tmp_path / "ack.json").unlink(missing_ok=True)
        (tmp_path / "heartbeat.json").unlink(missing_ok=True)

        class BlockedSession(FakeSession):
            def __init__(self, path, bridge):
                self.statuses = iter([status])

        monkeypatch.setattr(
            v3_supervisor_child.clore_session, "CloreSession", BlockedSession
        )
        assert asyncio.run(v3_supervisor_child._watch(config)) == 2
        blocked_ack = json.loads((tmp_path / "ack.json").read_text())
        assert blocked_ack["status"] == status
        assert not (tmp_path / "heartbeat.json").exists()
