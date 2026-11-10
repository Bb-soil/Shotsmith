from __future__ import annotations

import json
import types
import base64
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.p1 import clore_session as cs
from scripts.p1 import v3_lifecycle
from scripts.p1 import v3_operator
from scripts.p1 import v3_readiness
from scripts.p1.remote import RemoteIdentity
from tests.p1_live_candidate_helpers import install_live_candidate

ROOT = Path(__file__).resolve().parents[1]
MP4_FIXTURE = (
    b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
    b"\x00\x00\x00\x08moov"
    b"\x00\x00\x00\x1dmdatstructural-fixture-not-real-video"
)


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


def _safe_identity(blob: bytes):
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


def _trusted_identity(tmp_path: Path) -> tuple[RemoteIdentity, str]:
    identity = tmp_path / "id_trusted"
    known = tmp_path / "known_hosts_trusted"
    identity.write_text("not-a-key", encoding="utf-8")
    blob = b"operator-status-test-key"
    known.write_text(
        f"[example.test]:22 ssh-ed25519 {base64.b64encode(blob).decode()}\n",
        encoding="utf-8",
    )
    return (
        RemoteIdentity(
            host="example.test",
            user="root",
            port=22,
            identity_file=identity,
            known_hosts=known,
        ),
        _safe_identity(blob),
    )


def _state(tmp_path: Path):
    quote = cs.CloreQuote.from_dict(
        json.loads(
            (ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json").read_text(
                encoding="utf-8"
            )
        )
    )
    now = datetime.now(timezone.utc)
    state_path = tmp_path / "session-state.json"
    state = cs.initialize_state(
        state_path,
        order_id=97001,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        now=now,
    )
    return state_path, state


def _mark_ready(run: Path) -> None:
    path = run / "lifecycle.json"
    state = v3_lifecycle.load_lifecycle(path)
    v3_lifecycle.update_phase(path, state, phase="READY")


def _write_bound_lifecycle(run: Path, identity, fingerprint: str) -> None:
    now = int(time.time())
    binding = v3_lifecycle.SessionBinding(
        order_id=98001,
        server_id=112213,
        created_at=now - 5,
        currency="USD",
        cap_usd="1.00",
        duration_seconds=3600,
        planned_cancel_at=now + 300,
        hard_end_at=now + 600,
        quote_hash="a" * 64,
    )
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=run.name,
        session=binding,
        bundle_hashes={"config": "a" * 64},
        phase="READY",
    )
    path = run / "lifecycle.json"
    v3_lifecycle.write_lifecycle(path, lifecycle)
    lifecycle = v3_lifecycle.update_phase(
        path,
        lifecycle,
        phase="READY",
        pipeline={
            "ssh_identity": {
                "host": identity.host,
                "user": identity.user,
                "port": identity.port,
                "known_hosts": str(identity.known_hosts),
                "fingerprint": fingerprint,
            }
        },
    )


def _write_remote_export(
    run: Path,
    *,
    remote_path: str = "/workspace/result.mp4",
    root: str = "/workspace",
    helper: str = "/workspace/scripts/p1/v3_remote_fetch_helper.py",
    size: int = len(MP4_FIXTURE),
    sha256: str = "b" * 64,
) -> None:
    relative = remote_path[len(root) :].lstrip("/")
    lifecycle = json.loads((run / "lifecycle.json").read_text(encoding="utf-8"))
    (run / "remote-export.json").write_text(
        json.dumps(
            {
                "schema": "anime-video-studio/p1-v3-remote-export/v1",
                "run_id": run.name,
                "nonce": lifecycle["nonce"],
                "session": lifecycle["session"],
                "ssh_identity": lifecycle["pipeline"]["ssh_identity"],
                "job_id": "job-1",
                "prompt_id": "prompt-1",
                "remote_root": root,
                "remote_artifact_path": remote_path,
                "remote_helper_path": helper,
                "relative_path": relative,
                "filename": Path(remote_path).name,
                "size": size,
                "sha256": sha256,
            }
        ),
        encoding="utf-8",
    )


def test_plan_is_offline_and_honest() -> None:
    plan = v3_operator.build_plan()
    assert plan["status"] == "PLAN_OFFLINE"
    assert plan["readiness"]["live_ready"] is False
    assert plan["v3"]["manifest"]["status"] == "DRAFT"
    assert plan["v3"]["workflow"]["prompt_positive_exact"] is True
    assert plan["v3"]["workflow"]["prompt_negative_exact"] is True
    assert plan["v3"]["derived"]["crop_box"] == [14, 6, 927, 1666]
    assert plan["v3"]["derived"]["anisotropic"] is False
    assert plan["collect"]["items"][0] == "artifact.mp4"
    assert plan["collect"]["decoder"]["status"] == "NOT_VERIFIED"
    assert plan["live"]["status"] == "LIVE_NOT_RUN"
    assert "DRAFT" in plan["next_action"] or "readiness" in plan["next_action"]


def test_prepare_writes_lf_runbook_and_durable_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = v3_operator.prepare(run_dir=run_dir)
    assert result["status"] == "PREPARED_OFFLINE"
    setup = (run_dir / "setup.sh").read_bytes()
    assert b"\r\n" not in setup
    assert b"\\\\\n" not in setup
    assert b"pip install \\\n  --no-cache-dir" in setup
    assert (run_dir / "preflight.txt").is_file()
    assert (run_dir / "runbook.ps1").is_file()
    runbook = (run_dir / "runbook.ps1").read_text(encoding="utf-8")
    assert "v3_operator.py" in runbook
    assert "run-once --live" in runbook
    assert "StrictHostKeyChecking" not in runbook
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "heartbeat.json").is_file()
    state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert state["downloads"] == "NOT_RUN_OFFLINE"
    assert state["ssh"] == "NOT_RUN_OFFLINE"
    assert state["decoder"]["status"] == "NOT_VERIFIED"
    assert v3_operator.status(run_dir=run_dir)["status"] == "PREPARED_OFFLINE"


def test_prepare_execute_downloads_is_refused_before_network(tmp_path: Path) -> None:
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=tmp_path / "run", execute_downloads=True)
    assert not (tmp_path / "run").exists()


def test_live_run_is_blocked_before_mutation(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    with pytest.raises(v3_operator.ReadinessBlocked):
        v3_operator.run_once(live=True, run_dir=run_dir)
    assert not run_dir.exists()


def test_offline_run_once_is_live_not_run() -> None:
    result = v3_operator.run_once(live=False)
    assert result["status"] == "LIVE_NOT_RUN"
    assert result["readiness"]["live_ready"] is False


def test_fake_live_coordinator_uses_injected_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = v3_operator.load_bundle()
    bundle = dict(bundle)
    bundle["readiness"] = types.SimpleNamespace(live_ready=True, issues=())
    monkeypatch.setattr(v3_operator, "load_bundle", lambda: bundle)
    quote = cs.CloreQuote.from_dict(
        json.loads(
            (ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json").read_text(
                encoding="utf-8"
            )
        )
    )
    now = datetime.now(timezone.utc)
    state_path = tmp_path / "session.json"
    session = cs.initialize_state(
        state_path,
        order_id=2332282,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        now=now,
    )
    monkeypatch.setattr(
        v3_lifecycle,
        "verify_host_key",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        v3_lifecycle.CloreSupervisor,
        "start",
        lambda self, **kwargs: types.SimpleNamespace(
            to_dict=lambda: {"run_id": self.run_id, "nonce": self.nonce, "pid": 1}
        ),
    )
    monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
    v3_operator.prepare(
        run_dir=tmp_path / "run",
        session_state=state_path,
        order_id=session.order_id,
        created_at=session.created_at,
    )
    _mark_ready(tmp_path / "run")
    identity = tmp_path / "id"
    known = tmp_path / "known_hosts"
    identity.write_text("not-a-key", encoding="utf-8")
    known.write_text("[example.test]:22 ssh-ed25519 AAAA\n", encoding="utf-8")
    remote = RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=identity,
        known_hosts=known,
    )
    quote_path = ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json"
    install_live_candidate(
        run=tmp_path / "run",
        session_state=state_path,
        quote_path=quote_path,
        remote=remote,
        fingerprint="SHA256:trusted",
    )
    monkeypatch.setattr(
        v3_operator,
        "_supervisor_fact_from_disk",
        lambda supervisor: {
            "run_id": supervisor.run_id,
            "nonce": supervisor.nonce,
            "pid": 1,
            "status": "ARMED",
            "binding_sha256": v3_readiness.session_binding_digest(supervisor.binding),
            "config_sha256": "e" * 64,
            "heartbeat_at": time.time(),
        },
    )
    calls: list[str] = []

    def fake_backend(*, bundle, target, binding):
        calls.append(str(target))
        return {"status": "FAKE_LIVE_NOT_VERIFIED", "hard_end_at": binding.hard_end_at}

    result = v3_operator.run_once(
        live=True,
        session_state=state_path,
        order_id=session.order_id,
        created_at=session.created_at,
        remote=remote,
        expected_fingerprint="SHA256:trusted",
        run_dir=tmp_path / "run",
        backend_runner=fake_backend,
        ready_validator=lambda **kwargs: None,
    )
    assert result["status"] == "FAKE_LIVE_NOT_VERIFIED"
    assert calls == [str(tmp_path / "run")]
    state = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert state["order_id"] == session.order_id
    assert state["hard_end_at"] == session.hard_end_at


def test_collect_critical_first_and_decode_not_verified(tmp_path: Path) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    (run / "artifacts").mkdir()
    artifact = run / "artifacts" / "result.mp4"
    artifact.write_bytes(MP4_FIXTURE)
    result = v3_operator.collect(
        run_dir=run,
        artifact=artifact,
    )
    assert result["status"] == "COLLECTED_DECODE_NOT_VERIFIED"
    assert result["completed"][0]["target"] == "artifact.mp4"
    assert result["media"]["status"] == "NOT_VERIFIED"
    assert Path(result["report"]).is_file()
    assert "history.json" in result["not_exported"]
    assert "costs.json" in result["not_exported"]
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["note"].startswith("local staged archive only")


def test_collect_refuses_without_critical_mp4(tmp_path: Path) -> None:
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(run_dir=tmp_path / "run")
    assert not list((tmp_path / "run").glob("collect-*.json"))


def test_collect_remote_requires_allowlisted_path(tmp_path: Path) -> None:
    identity = tmp_path / "id"
    known = tmp_path / "known"
    identity.write_text("x", encoding="utf-8")
    known.write_text("[example.test]:22 ssh-ed25519 AAAA\n", encoding="utf-8")
    remote = RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=identity,
        known_hosts=known,
    )
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(MP4_FIXTURE)
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(
            run_dir=tmp_path / "run",
            artifact=artifact,
            remote=remote,
            remote_evidence=("../../etc/passwd",),
        )


def test_prepare_existing_run_resumes_under_lock_and_preserves_logs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    first = v3_operator.prepare(run_dir=run)
    marker = run / "old.log"
    marker.write_text("preserve me\n", encoding="utf-8")
    second = v3_operator.prepare(run_dir=run)
    assert second["status"] == "PREPARED_OFFLINE"
    assert second["resumed"] is True
    assert second["runbook"] != first["runbook"]
    assert marker.read_text(encoding="utf-8") == "preserve me\n"
    assert (run / "prepare-history.jsonl").is_file()
    assert v3_operator.status(run_dir=run)["status"] == "PREPARED_OFFLINE"


def test_repeated_collect_uses_unique_namespaces_and_reports(tmp_path: Path) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    (run / "artifacts").mkdir()
    artifact = run / "artifacts" / "result.mp4"
    artifact.write_bytes(MP4_FIXTURE)
    first = v3_operator.collect(run_dir=run, artifact=artifact)
    second = v3_operator.collect(run_dir=run, artifact=artifact)
    assert first["report"] != second["report"]
    assert Path(first["report"]).is_file()
    assert Path(second["report"]).is_file()
    assert first["manifest_path"] != second["manifest_path"]
    assert len(list((run / "export").glob(".clore-export-*"))) == 2


def _fake_media_run(*, probe, decode_exit: int = 0):
    class Result:
        def __init__(self, returncode: int, stdout: bytes = b"") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = b""
            self.timed_out = False
            self.reaped = True
            self.readers_done = True

    def run(argv, **kwargs):
        if "ffprobe" in str(argv[0]):
            payload = probe if isinstance(probe, bytes) else json.dumps(probe or {}).encode()
            return Result(0, payload)
        return Result(decode_exit)

    return run


def test_media_requires_full_decode_and_exact_params(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "artifact.mp4"
    path.write_bytes(MP4_FIXTURE)
    good = {
        "streams": [
            {
                "codec_type": "video",
                "width": 704,
                "height": 1280,
                "avg_frame_rate": "24/1",
                "nb_read_frames": "121",
            }
        ],
        "format": {"duration": "5.0416666667"},
    }
    monkeypatch.setattr(v3_operator.shutil, "which", lambda name: name if name in {"ffprobe", "ffmpeg"} else None)
    monkeypatch.setattr(v3_operator, "run_bounded_process", _fake_media_run(probe=good))
    assert v3_operator._inspect_media(path)["status"] == "VERIFIED"

    bad = json.loads(json.dumps(good))
    bad["streams"][0]["width"] = 703
    monkeypatch.setattr(v3_operator, "run_bounded_process", _fake_media_run(probe=bad))
    assert v3_operator._inspect_media(path)["status"] == "NOT_VERIFIED"

    monkeypatch.setattr(v3_operator, "run_bounded_process", _fake_media_run(probe=good, decode_exit=1))
    assert v3_operator._inspect_media(path)["status"] == "NOT_VERIFIED"

    monkeypatch.setattr(v3_operator.shutil, "which", lambda name: name if name == "ffprobe" else None)
    assert v3_operator._inspect_media(path)["status"] == "NOT_VERIFIED"

    def overflow(*args, **kwargs):
        raise v3_operator.RemoteError("output overflow")

    monkeypatch.setattr(v3_operator.shutil, "which", lambda name: name if name in {"ffprobe", "ffmpeg"} else None)
    monkeypatch.setattr(v3_operator, "run_bounded_process", overflow)
    assert v3_operator._inspect_media(path)["status"] == "NOT_VERIFIED"


def test_media_rejects_malformed_schema_and_nonfinite(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "artifact.mp4"
    path.write_bytes(MP4_FIXTURE)
    monkeypatch.setattr(
        v3_operator.shutil,
        "which",
        lambda name: name if name in {"ffprobe", "ffmpeg"} else None,
    )
    for payload in (
        b"[]",
        b'{"streams": "not-a-list", "format": {}}',
        b'{"streams": ["not-an-object"], "format": {"duration": "1"}}',
        b'{"streams": [], "format": {"duration": "NaN"}}',
        b'{"streams": [{"codec_type": "video", "width": 704, "height": 1280, "avg_frame_rate": "24/1", "nb_read_frames": "121"}], "format": {"duration": "Infinity"}}',
    ):
        monkeypatch.setattr(
            v3_operator,
            "run_bounded_process",
            _fake_media_run(probe=payload),
        )
        assert v3_operator._inspect_media(path)["status"] == "NOT_VERIFIED"


def test_decoder_discovery_is_portable(monkeypatch) -> None:
    monkeypatch.setattr(v3_operator.shutil, "which", lambda name: name if name in {"ffprobe", "ffmpeg"} else None)
    assert v3_operator._decoder_status()["status"] == "AVAILABLE"
    monkeypatch.setattr(v3_operator.shutil, "which", lambda name: name if name == "ffprobe" else None)
    assert v3_operator._decoder_status()["status"] == "NOT_VERIFIED"


def test_collect_rejects_outside_artifact_before_mutation(tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(MP4_FIXTURE)
    run = tmp_path / "run"
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(run_dir=run, artifact=outside)
    assert not run.exists()


def test_collect_rejects_foreign_bundle_hash_before_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = v3_operator.load_bundle()
    foreign = dict(bundle)
    foreign["workflow_sha256"] = "0" * 64
    monkeypatch.setattr(v3_operator, "load_bundle", lambda: foreign)
    run = tmp_path / "run"
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(run_dir=run)
    assert not run.exists()


def test_collect_remote_cli_requires_exact_identity_before_mutation(
    tmp_path: Path, capsys
) -> None:
    code = v3_operator.main(
        [
            "collect",
            "--run-dir",
            str(tmp_path / "run"),
            "--remote-artifact-path",
            "/workspace/result.mp4",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "REFUSED"
    assert not (tmp_path / "run").exists()


def test_collect_remote_uses_bounded_fetch_and_collects(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    identity = _identity(tmp_path)
    _write_bound_lifecycle(run, identity, "SHA256:trusted")
    _write_remote_export(run)
    calls: list[str] = []

    class FakeRunner:
        def put_file(self, identity, remote_path, local_path, **kwargs):
            calls.append(f"put:{remote_path}")

    import scripts.p1.v3_lifecycle as lifecycle_module

    def fake_fetch(**kwargs):
        calls.append("fetch")
        target = Path(kwargs["local_path"])
        target.write_bytes(MP4_FIXTURE)
        return {"sha256": "a" * 64}

    monkeypatch.setattr(v3_operator, "RemoteRunner", lambda: FakeRunner())
    monkeypatch.setattr(lifecycle_module, "verify_host_key", lambda **kwargs: None)
    monkeypatch.setattr(lifecycle_module, "fetch_exact_stream", fake_fetch)
    result = v3_operator.collect(
        run_dir=run,
        remote=identity,
        remote_artifact_path="/workspace/result.mp4",
        remote_helper_path="/workspace/scripts/p1/v3_remote_fetch_helper.py",
        remote_expected_size=len(MP4_FIXTURE),
        remote_expected_sha256="b" * 64,
        expected_fingerprint="SHA256:trusted",
    )
    assert calls[0] == "put:/workspace/scripts/p1/v3_remote_fetch_helper.py"
    assert "fetch" in calls
    assert result["completed"][0]["target"] == "artifact.mp4"


def test_collect_remote_hostkey_refusal_before_fetch(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    identity = _identity(tmp_path)

    class FakeRunner:
        def put_file(self, *args):
            raise AssertionError("put_file must not run before host key")

    import scripts.p1.v3_lifecycle as lifecycle_module

    def fail_host_key(**kwargs):
        raise lifecycle_module.LifecycleError("mismatch")

    monkeypatch.setattr(v3_operator, "RemoteRunner", lambda: FakeRunner())
    monkeypatch.setattr(lifecycle_module, "verify_host_key", fail_host_key)
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(
            run_dir=run,
            remote=identity,
            remote_artifact_path="/workspace/result.mp4",
            remote_helper_path="/workspace/scripts/p1/v3_remote_fetch_helper.py",
            remote_expected_size=10,
            remote_expected_sha256="b" * 64,
            expected_fingerprint="SHA256:trusted",
        )


def test_collect_does_not_import_user_structured_evidence(tmp_path: Path) -> None:
    import inspect

    assert "evidence_dir" not in inspect.signature(v3_operator.collect).parameters
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    (run / "artifacts").mkdir()
    artifact = run / "artifacts" / "result.mp4"
    artifact.write_bytes(MP4_FIXTURE)
    user_file = run / "user-evidence.json"
    user_file.write_text(
        '{"token": "synthetic-secret", "url": "https://secret.invalid/x", "unknown": 1',
        encoding="utf-8",
    )
    result = v3_operator.collect(run_dir=run, artifact=artifact)
    assert not any(item["target"] == "user-evidence.json" for item in result["completed"])
    dumped = json.dumps(result)
    assert "synthetic-secret" not in dumped
    assert "secret.invalid" not in dumped
    assert (run / "user-evidence.json").is_file()


def test_nonempty_run_without_run_json_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "foreign.txt").write_text("x", encoding="utf-8")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)
    assert not (run / "setup.sh").exists()


def test_foreign_bundle_hash_blocks_resume(tmp_path: Path) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    state = json.loads((run / "run.json").read_text(encoding="utf-8"))
    state["bundle_hashes"]["workflow"] = "0" * 64
    (run / "run.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)


@pytest.mark.parametrize(
    "mutation",
    ["absent", "null", "partial", "malformed"],
)
def test_incomplete_immutable_identity_is_rejected_without_writing(
    tmp_path: Path, mutation: str
) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    state = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if mutation == "absent":
        state.pop("bundle_hashes")
    elif mutation == "null":
        state["bundle_hashes"] = None
    elif mutation == "partial":
        state["bundle_hashes"] = {"workflow": state["bundle_hashes"]["workflow"]}
    else:
        state["bundle_hashes"] = "not-a-mapping"
    path = run / "run.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)
    assert path.read_bytes() == before


def test_lock_symlink_external_empty_bytes_unchanged(tmp_path: Path) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    external = tmp_path / "external.lock"
    external.write_bytes(b"")
    lock = run / "operator.lock"
    lock.unlink()
    try:
        lock.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)
    assert external.read_bytes() == b""


def test_collect_target_symlink_does_not_create_external_lock(tmp_path: Path) -> None:
    external = tmp_path / "external-run"
    external.mkdir()
    target = tmp_path / "run-link"
    try:
        target.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(run_dir=target)
    assert not (external / "operator.lock").exists()


def test_dangling_lock_or_target_fails_closed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    lock = run / "operator.lock"
    lock.unlink()
    try:
        lock.symlink_to(tmp_path / "missing.lock")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)
    assert not (tmp_path / "missing.lock").exists()

    dangling_target = tmp_path / "dangling-run"
    try:
        dangling_target.symlink_to(tmp_path / "missing-run", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=dangling_target)
    assert not (tmp_path / "missing-run").exists()


def test_lock_only_nonempty_run_is_rejected_before_lock(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lock = run / "operator.lock"
    lock.write_bytes(b"")
    before = lock.read_bytes()
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)
    assert lock.read_bytes() == before
    assert not (run / "run.json").exists()


def test_external_state_symlink_is_not_read_and_creates_no_lock(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    external = tmp_path / "external-state.json"
    external.write_text(
        json.dumps(
            {
                "schema": v3_operator.SCHEMA,
                "task_id": v3_operator.TASK_ID,
                "run_id": run.name,
                "bundle_hashes": v3_operator._bundle_hashes(v3_operator.load_bundle()),
            }
        ),
        encoding="utf-8",
    )
    try:
        (run / "run.json").symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")
    before = external.read_bytes()
    read_calls: list[str] = []
    original_read = v3_operator._read_json

    def spy_read(path):
        if Path(path).name == "run.json":
            read_calls.append(str(path))
            raise AssertionError("external state must not be read")
        return original_read(path)

    monkeypatch.setattr(v3_operator, "_read_json", spy_read)
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)
    assert read_calls == []
    assert external.read_bytes() == before
    assert not (run / "operator.lock").exists()


def test_dangling_state_symlink_is_rejected_before_read_and_lock(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    try:
        (run / "run.json").symlink_to(tmp_path / "missing-state.json")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)
    assert not (run / "operator.lock").exists()
    assert not (tmp_path / "missing-state.json").exists()


def test_default_artifact_symlink_parent_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (run / "artifacts").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(run_dir=run)


def test_history_symlink_is_rejected_on_resume(tmp_path: Path) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("", encoding="utf-8")
    history = run / "prepare-history.jsonl"
    history.unlink()
    try:
        history.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(run_dir=run)


def test_remote_collect_is_disabled_before_mutation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(
            run_dir=run,
            remote_artifact_path="artifact.mp4",
        )
    assert not run.exists()


def test_copy_bounded_honours_absolute_deadline(tmp_path: Path) -> None:
    import time

    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * 4096)
    target = tmp_path / "target.bin"
    with pytest.raises(v3_operator.OperatorError):
        v3_operator._copy_bounded(
            source,
            target,
            max_bytes=8192,
            deadline_monotonic=time.monotonic() - 1,
        )
    assert not target.exists()


def test_cli_collect_error_is_structured(tmp_path: Path, capsys) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(MP4_FIXTURE)
    code = v3_operator.main(
        [
            "collect",
            "--run-dir",
            str(tmp_path / "run"),
            "--artifact",
            str(outside),
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "REFUSED"
    assert payload["error_type"] == "OperatorError"
    assert not (tmp_path / "run").exists()

def test_status_remote_reads_all_three_fresh_progress_files(tmp_path: Path) -> None:
    run = tmp_path / "run"
    state_path, state = _state(tmp_path)
    v3_operator.prepare(
        run_dir=run,
        session_state=state_path,
        order_id=state.order_id,
        created_at=state.created_at,
    )
    lifecycle = json.loads((run / "lifecycle.json").read_text())
    identity, fingerprint = _trusted_identity(tmp_path)
    files = {
        "worker-owner.json": {
            "schema": "anime-video-studio/p1-v3-process-owner/v1",
            "kind": "download_worker",
            "run_id": lifecycle["run_id"],
            "nonce": lifecycle["nonce"],
            "command_sha256": "c" * 64,
            "pid": 123,
        },
        "status.json": {"run_id": lifecycle["run_id"], "nonce": lifecycle["nonce"], "status": "RUNNING"},
        "progress.json": {
            "run_id": lifecycle["run_id"],
            "nonce": lifecycle["nonce"],
            "status": "RUNNING",
            "files_done": 1,
            "files_total": 3,
            "bytes_done": 10,
            "bytes_total": 30,
        },
        "heartbeat.json": {
            "run_id": lifecycle["run_id"],
            "nonce": lifecycle["nonce"],
            "status": "RUNNING",
            "heartbeat_at": time.time(),
        },
    }
    calls: list[str] = []

    class Runner:
        ssh_keygen_path = __import__("shutil").which("ssh-keygen")

        def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout, **kwargs):
            calls.append(Path(remote_path).name)
            return files[Path(remote_path).name]

    result = v3_operator.status(
        run_dir=run,
        remote=identity,
        expected_fingerprint=fingerprint,
        remote_runner=Runner(),
    )
    assert result["remote_status"]["progress"]["files_done"] == 1
    assert result["remote_status"]["progress"]["bytes_total"] == 30
    assert result["remote_status"]["fresh"] is True
    assert calls == ["worker-owner.json", "status.json", "progress.json", "heartbeat.json"]

    class NoNetwork:
        def read_remote_exact_json(self, *args, **kwargs):
            raise AssertionError("default status must not read remote state")

    local = v3_operator.status(run_dir=run, remote_runner=NoNetwork())
    assert local["remote_status"] is None


def test_backend_uses_planned_cancel_not_only_hard_end(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.p1 import v3_backend

    bundle = v3_operator.load_bundle()
    target = tmp_path / "run"
    target.mkdir()
    now = int(time.time())
    posted: list[str] = []

    clock = {"now": now}

    class FakeProcess:
        base_url = "http://127.0.0.1:1"

        @classmethod
        def build(cls, **kwargs):
            return cls()

        def start(self, *, deadline=None):
            clock["now"] = now + 130

        def stop(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def health(self, *, deadline=None):
            raise v3_backend.BackendDeadlineExpired("deadline consumed")

        def post_job(self, *args, **kwargs):
            posted.append("post")
            raise AssertionError("must not POST after the planned cancel deadline")

    monkeypatch.setattr(v3_backend.BackendProcess, "build", classmethod(lambda cls, **kwargs: FakeProcess()))
    monkeypatch.setattr(v3_backend, "BackendClient", FakeClient)
    monkeypatch.setattr(v3_backend, "check_comfy_queue", lambda *args, **kwargs: None)
    monkeypatch.setattr(v3_operator.time, "time", lambda: clock["now"])
    binding = v3_lifecycle.SessionBinding(
        order_id=97001,
        server_id=112213,
        created_at=now - 10,
        currency="USD",
        cap_usd="1.00",
        duration_seconds=3600,
        planned_cancel_at=now + 130,
        hard_end_at=now + 1000,
        quote_hash="a" * 64,
    )
    result = v3_operator._run_backend_once(
        bundle=bundle,
        target=target,
        config_path=v3_operator.V3_CONFIG,
        binding=binding,
        backend_runner=None,
    )
    assert result["status"] == "DEADLINE_BEFORE_POST"
    assert result["no_post"] is True
    assert posted == []


def test_run_once_same_run_resumes_without_duplicate_supervisor(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = dict(v3_operator.load_bundle())
    bundle["readiness"] = types.SimpleNamespace(live_ready=True, issues=())
    monkeypatch.setattr(v3_operator, "load_bundle", lambda: bundle)
    monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
    monkeypatch.setattr(v3_lifecycle, "_pid_live", lambda pid: True)
    monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
    state_path, state = _state(tmp_path)
    run = tmp_path / "run"
    v3_operator.prepare(
        run_dir=run,
        session_state=state_path,
        order_id=state.order_id,
        created_at=state.created_at,
    )
    _mark_ready(run)
    nonce = json.loads((run / "lifecycle.json").read_text())["nonce"]

    starts: list[str] = []

    def fake_start(self, **kwargs):
        starts.append(self.nonce)
        from scripts.p1.v3_lifecycle import SupervisorRecord

        self.config_path.write_text(
            json.dumps(self._config_payload()), encoding="utf-8"
        )
        config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        binding_sha256 = hashlib.sha256(
            json.dumps(
                self.binding.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        record = SupervisorRecord(
            run_id=self.run_id,
            nonce=self.nonce,
            pid=4321,
            state_path=str(self.session_state_path),
            quote_path=str(self.quote_path),
            log_path=str(self.log_path),
            started_at=datetime.now(timezone.utc).isoformat(),
            order_id=self.binding.order_id,
            created_at=self.binding.created_at,
            config_sha256=config_sha256,
            binding_sha256=binding_sha256,
        )
        self.record_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
        self.ack_path.write_text(
            json.dumps(
                {
                    "schema": "anime-video-studio/p1-v3-supervisor-ack/v1",
                    "status": "ARMED",
                    "run_id": self.run_id,
                    "nonce": self.nonce,
                    "order_id": self.binding.order_id,
                    "created_at": self.binding.created_at,
                    "ack_at": time.time(),
                    "pid": 4321,
                    "config_sha256": config_sha256,
                    "binding_sha256": binding_sha256,
                }
            ),
            encoding="utf-8",
        )
        self.heartbeat_path.write_text(
            json.dumps(
                {
                    "schema": "anime-video-studio/p1-v3-supervisor-heartbeat/v1",
                    "run_id": self.run_id,
                    "nonce": self.nonce,
                    "at": time.time(),
                    "pid": 4321,
                    "config_sha256": config_sha256,
                    "binding_sha256": binding_sha256,
                }
            ),
            encoding="utf-8",
        )
        return record

    monkeypatch.setattr(v3_lifecycle.CloreSupervisor, "start", fake_start)
    identity, fingerprint = _trusted_identity(tmp_path)
    install_live_candidate(
        run=run,
        session_state=state_path,
        quote_path=ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json",
        remote=identity,
        fingerprint=fingerprint,
    )
    calls: list[str] = []

    def fake_backend(**kwargs):
        calls.append(str(kwargs["target"]))
        return {"status": "FAKE_RESUMED"}

    first = v3_operator.run_once(
        live=True,
        session_state=state_path,
        order_id=state.order_id,
        created_at=state.created_at,
        remote=identity,
        expected_fingerprint=fingerprint,
        run_dir=run,
        backend_runner=fake_backend,
        ready_validator=lambda **kwargs: None,
    )
    second = v3_operator.run_once(
        live=True,
        session_state=state_path,
        order_id=state.order_id,
        created_at=state.created_at,
        remote=identity,
        expected_fingerprint=fingerprint,
        run_dir=run,
        backend_runner=fake_backend,
        ready_validator=lambda **kwargs: None,
    )
    assert first["status"] == "FAKE_RESUMED"
    assert second["status"] == "FAKE_RESUMED"
    assert starts == [nonce]
    assert calls == [str(run), str(run)]
    assert json.loads((run / "lifecycle.json").read_text())["nonce"] == nonce

def test_operator_backend_forwards_one_run_deadline_to_all_steps(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.p1 import v3_backend

    target = tmp_path / "run"
    target.mkdir()
    now = int(time.time())
    binding = v3_lifecycle.SessionBinding(
        order_id=97001,
        server_id=112213,
        created_at=now - 10,
        currency="USD",
        cap_usd="1.00",
        duration_seconds=3600,
        planned_cancel_at=now + 300,
        hard_end_at=now + 600,
        quote_hash="a" * 64,
    )
    captured: dict[str, object] = {}

    class FakeProcess:
        base_url = "http://127.0.0.1:1"

        @classmethod
        def build(cls, **kwargs):
            captured["build_run_dir"] = kwargs.get("run_dir")
            return cls()

        def start(self, *, deadline=None):
            captured["start"] = deadline
            return self

        def stop(self):
            return True

    class FakeClient:
        def __init__(self, base_url, *, timeout=None, deadline=None):
            captured["client"] = deadline

        def health(self, *, deadline=None):
            captured["health"] = deadline
            return {"status": "ok"}

        def post_job(self, **kwargs):
            captured["post"] = kwargs.get("deadline")
            return {"job": {"job_id": "job-1"}}

        def wait_terminal(self, job_id, *, deadline, poll=0.5, session_deadline=None):
            captured["wait"] = session_deadline
            return {"status": "succeeded", "job_id": job_id}

        def get_result(self, job_id, *, deadline=None):
            captured["result"] = deadline
            return {"job_id": job_id, "result": {"ok": True}}

    def fake_queue(base_url, token=None, *, deadline=None, timeout=10.0):
        captured["queue"] = deadline
        return {"running": 0, "pending": 0}

    monkeypatch.setattr(v3_backend.BackendProcess, "build", classmethod(lambda cls, **kwargs: FakeProcess.build(**kwargs)))
    monkeypatch.setattr(v3_backend, "BackendClient", FakeClient)
    monkeypatch.setattr(v3_backend, "check_comfy_queue", fake_queue)
    result = v3_operator._run_backend_once(
        bundle={"job": {"prompt": "deadline forwarding", "seed": 1}},
        target=target,
        config_path=v3_operator.V3_CONFIG,
        binding=binding,
    )
    assert result["status"] == "succeeded"
    deadline = v3_backend.RunDeadline(binding)
    for key in ("start", "client", "health", "queue", "post", "wait", "result"):
        value = captured[key]
        assert value is not None, key
        assert value.effective_deadline_at == deadline.effective_deadline_at, key
        assert value.binding == binding, key
    assert captured["build_run_dir"] == target

def test_collect_defaults_to_exact_backend_job_artifact(tmp_path: Path) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    job_id = "job-abc"
    artifact = run / "artifacts" / job_id / "result.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(MP4_FIXTURE)
    (run / "job-id.json").write_text(
        json.dumps({"job_id": job_id, "idempotency_key": "key"}), encoding="utf-8"
    )
    (run / "backend-result.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "job_id": job_id,
                "result": {
                    "job_id": job_id,
                    "mock": False,
                    "result": {
                        "artifacts": [
                            {
                                "kind": "video",
                                "mock": False,
                                "media_type": "video/mp4",
                                "relative_path": f"{job_id}/result.mp4",
                                "path": str(artifact),
                                "sha256": hashlib.sha256(MP4_FIXTURE).hexdigest(),
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    result = v3_operator.collect(run_dir=run)
    assert result["status"] == "COLLECTED_DECODE_NOT_VERIFIED"
    assert result["completed"][0]["target"] == "artifact.mp4"
    exported = Path(result["manifest_path"]).parent / "artifact.mp4"
    assert exported.read_bytes() == MP4_FIXTURE

def test_session_claim_is_stable_and_full_binding_exclusive(
    tmp_path: Path, monkeypatch
) -> None:
    import threading

    monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
    now = int(time.time())
    base = dict(
        order_id=99001,
        server_id=112213,
        created_at=now - 5,
        currency="USD",
        duration_seconds=3600,
        planned_cancel_at=now + 300,
        hard_end_at=now + 600,
        quote_hash="a" * 64,
    )
    first_binding = v3_lifecycle.SessionBinding(cap_usd="1.00", **base)
    second_binding = v3_lifecycle.SessionBinding(cap_usd="2.00", **base)
    assert v3_operator._session_key(first_binding) == v3_operator._session_key(second_binding)
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def claim(binding, run_name: str) -> None:
        barrier.wait()
        try:
            v3_operator._register_session_owner(
                binding, run_dir=tmp_path / run_name, nonce=f"nonce-{run_name}"
            )
            outcomes.append(f"ok:{run_name}")
        except v3_operator.OperatorError:
            outcomes.append(f"refused:{run_name}")

    threads = [
        threading.Thread(target=claim, args=(first_binding, "run-a")),
        threading.Thread(target=claim, args=(second_binding, "run-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(outcomes) == 2
    assert sum(item.startswith("ok:") for item in outcomes) == 1
    assert sum(item.startswith("refused:") for item in outcomes) == 1

def test_run_once_rejects_stale_ready_without_owned_tunnel_before_post(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = dict(v3_operator.load_bundle())
    bundle["readiness"] = types.SimpleNamespace(live_ready=True, issues=())
    monkeypatch.setattr(v3_operator, "load_bundle", lambda: bundle)
    monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
    monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
    state_path, state = _state(tmp_path)
    run = tmp_path / "run"
    v3_operator.prepare(
        run_dir=run,
        session_state=state_path,
        order_id=state.order_id,
        created_at=state.created_at,
    )
    _mark_ready(run)
    identity, fingerprint = _trusted_identity(tmp_path)
    calls: list[str] = []

    def fake_backend(**kwargs):
        calls.append(str(kwargs["target"]))
        return {"status": "SHOULD_NOT_RUN"}

    with pytest.raises(v3_operator.OperatorError):
        v3_operator.run_once(
            live=True,
            session_state=state_path,
            order_id=state.order_id,
            created_at=state.created_at,
            remote=identity,
            expected_fingerprint=fingerprint,
            run_dir=run,
            backend_runner=fake_backend,
        )
    assert calls == []

def test_collect_remote_refuses_without_pipeline_export_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    identity = _identity(tmp_path)
    monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
    calls: list[str] = []

    class Runner:
        def put_file(self, *args, **kwargs):
            calls.append("put")

    with pytest.raises(v3_operator.OperatorError):
        v3_operator.collect(
            run_dir=run,
            remote=identity,
            remote_artifact_path="/workspace/result.mp4",
            remote_helper_path="/workspace/scripts/p1/v3_remote_fetch_helper.py",
            remote_expected_size=len(MP4_FIXTURE),
            remote_expected_sha256="b" * 64,
            expected_fingerprint="SHA256:trusted",
            remote_runner=Runner(),
        )
    assert calls == []
    assert not (run / "remote-export.json").exists()

def test_remote_export_producer_from_comfy_history_then_collect(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace
    import subprocess

    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    local_root = tmp_path / "remote-root"
    artifact_rel = "ComfyUI/output/shot/result.mp4"
    local_artifact = local_root / artifact_rel
    local_artifact.parent.mkdir(parents=True)
    local_artifact.write_bytes(MP4_FIXTURE)
    sha = hashlib.sha256(MP4_FIXTURE).hexdigest()
    (run / "job-id.json").write_text(
        json.dumps({"job_id": "job-remote-1", "idempotency_key": "key"}), encoding="utf-8"
    )
    (run / "backend-result.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "job_id": "job-remote-1",
                "result": {
                    "job_id": "job-remote-1",
                    "result": {"prompt_id": "prompt-1", "artifacts": []},
                },
            }
        ),
        encoding="utf-8",
    )
    identity, fingerprint = _trusted_identity(tmp_path)
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=run.name,
        session=v3_lifecycle.SessionBinding(
            order_id=98001,
            server_id=112213,
            created_at=int(time.time()) - 5,
            currency="USD",
            cap_usd="1.00",
            duration_seconds=3600,
            planned_cancel_at=int(time.time()) + 300,
            hard_end_at=int(time.time()) + 600,
            quote_hash="a" * 64,
        ),
        bundle_hashes={"config": "a" * 64},
    )
    lifecycle = v3_lifecycle.update_phase(
        run / "lifecycle.json",
        lifecycle,
        phase="READY",
        pipeline={
            "ssh_identity": {
                "host": identity.host,
                "user": identity.user,
                "port": identity.port,
                "known_hosts": str(identity.known_hosts),
                "fingerprint": fingerprint,
            }
        },
    )

    class Transport:
        async def _history_async(self, prompt_id):
            assert prompt_id == "prompt-1"
            return {"outputs": {}, "status": {"status_str": "success"}}

        def _status_from_entry(self, prompt_id, entry):
            return SimpleNamespace(
                state="succeeded",
                artifact={
                    "scheme": "comfy-http",
                    "filename": "result.mp4",
                    "subfolder": "shot",
                    "type": "output",
                    "media_type": "video/mp4",
                }
            )

    class Runner:
        ssh_keygen_path = __import__("shutil").which("ssh-keygen")

        def run_remote(self, identity, args, timeout=None, **kwargs):
            assert args[0] == "python3" and args[2] == "metadata"
            remote_path = args[3]
            local = local_root / remote_path.removeprefix("/workspace/")
            completed = subprocess.run(
                [__import__("sys").executable, str(ROOT / "scripts" / "p1" / "v3_remote_fetch_helper.py"), "metadata", str(local), str(local_root)],
                capture_output=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            payload = json.loads(completed.stdout)
            payload["path"] = remote_path
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(payload), timed_out=False)

        def put_file(self, *args, **kwargs):
            return None

    runner = Runner()
    export = v3_operator._produce_remote_export(
        target=run,
        remote=identity,
        workspace="/workspace",
        config_path=v3_operator.V3_CONFIG,
        lifecycle=lifecycle,
        transport=Transport(),
        runner=runner,
    )
    assert export["job_id"] == "job-remote-1"
    assert export["prompt_id"] == "prompt-1"
    assert export["remote_artifact_path"] == f"/workspace/{artifact_rel}"
    assert export["size"] == len(MP4_FIXTURE)
    assert export["sha256"] == sha
    assert (run / "remote-export.json").is_file()

    import scripts.p1.v3_lifecycle as lifecycle_module

    monkeypatch.setattr(lifecycle_module, "verify_host_key", lambda **kwargs: None)

    def fake_fetch(**kwargs):
        Path(kwargs["local_path"]).write_bytes(MP4_FIXTURE)
        return {"sha256": sha}

    monkeypatch.setattr(lifecycle_module, "fetch_exact_stream", fake_fetch)
    result = v3_operator.collect(
        run_dir=run,
        remote=identity,
        remote_artifact_path=export["remote_artifact_path"],
        remote_helper_path=export["remote_helper_path"],
        remote_expected_size=export["size"],
        remote_expected_sha256=export["sha256"],
        expected_fingerprint=fingerprint,
        remote_runner=runner,
    )
    assert result["completed"][0]["target"] == "artifact.mp4"

def test_remote_export_history_respects_absolute_deadline(tmp_path: Path) -> None:
    import threading

    target = tmp_path / "run"
    v3_operator.prepare(run_dir=target)
    (target / "job-id.json").write_text(json.dumps({"job_id": "job-1"}), encoding="utf-8")
    (target / "backend-result.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "result": {"result": {"prompt_id": "prompt-1"}},
            }
        ),
        encoding="utf-8",
    )
    now = int(time.time())
    binding = v3_lifecycle.SessionBinding(
        order_id=99001,
        server_id=112213,
        created_at=now - 5,
        currency="USD",
        cap_usd="1.00",
        duration_seconds=3600,
        planned_cancel_at=now + 300,
        hard_end_at=now + 600,
        quote_hash="a" * 64,
    )
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=target.name, session=binding, bundle_hashes={"config": "a" * 64}
    )
    started = threading.Event()
    release = threading.Event()

    class SlowTransport:
        async def _history_async(self, prompt_id):
            started.set()
            await __import__("asyncio").sleep(5)
            return {}

        def _status_from_entry(self, prompt_id, entry):
            return SimpleNamespace(state="succeeded", artifact={})

    try:
        with pytest.raises(v3_operator.OperatorError):
            v3_operator._produce_remote_export(
                target=target,
                remote=_identity(tmp_path),
                workspace="/workspace",
                config_path=v3_operator.V3_CONFIG,
                lifecycle=lifecycle,
                transport=SlowTransport(),
                runner=object(),
                deadline_monotonic=time.monotonic() + 0.1,
            )
        assert started.is_set()
        assert not [t for t in threading.enumerate() if t.name.startswith("p1-history")]
    finally:
        release.set()

