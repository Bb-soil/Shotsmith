from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

from scripts.i1 import profile as profile_mod
from scripts.i1 import setup as setup_mod
from scripts.i1 import worker
from scripts.i1 import worker_recover

ROOT = Path(__file__).resolve().parents[1]


def _tiny_profile(tmp_path: Path) -> tuple[profile_mod.Profile, dict[str, bytes]]:
    base = json.loads(
        (ROOT / "config" / "i1" / "wan22-14b-ab.profile.json").read_text(encoding="utf-8")
    )
    payloads = {}
    for index, item in enumerate(base["models"]):
        data = f"tiny-model-{index}".encode()
        payloads[item["target"]] = data
        item["size_bytes"] = len(data)
        item["sha256"] = hashlib.sha256(data).hexdigest()
        item["url"] = f"https://example.invalid/{index}.bin"
    base["disk"]["install_overhead_bytes"] = 0
    base["disk"]["reserve_bytes"] = 0
    base["profile_id"] = "tiny-14b"
    base["source_inventory"]["sha256"] = "a" * 64
    base["workflow"]["api_sha256"] = "b" * 64
    if base["workflow"].get("ui_sha256"):
        base["workflow"]["ui_sha256"] = "c" * 64
    profile = profile_mod.validate_profile(base)
    return profile, payloads


def test_setup_script_has_no_download_or_torch_reinstall(tmp_path: Path) -> None:
    base = json.loads(
        (ROOT / "config" / "i1" / "wan22-14b-ab.profile.json").read_text(encoding="utf-8")
    )
    profile = profile_mod.validate_profile(base)
    text = setup_mod.render_setup(profile, workspace="/tmp/workspace with space;safe")
    assert "setup_plan" in text
    assert "--execute" not in text
    assert "git reset" not in text
    assert "git clean" not in text
    assert "pip install --upgrade torch" not in text
    assert "cmp -s" in text
    assert "torch-before.json" in text
    assert "torch-after.json" in text
    assert "7a28332047b6af638dc6c7f72ef3481a718769fd" in text
    assert "/tmp/workspace with space;safe" in text
    assert "custom_nodes" in text
    assert "unsafe symlink before mutation" in text
    assert "I1_STOCK_NODES_PY" in text
    assert "anime-video-studio/i1-stock-nodes/v1" in text
    assert "__pycache__" in text
    assert "GIT_NO_REPLACE_OBJECTS" in text
    assert "*.example" not in text
    assert text.index("I1_STOCK_NODES_PY") < text.index("pip install")
    assert "realpath" in text


def _job(tmp_path: Path, profile: profile_mod.Profile, payloads: dict[str, bytes]) -> tuple[Path, dict]:
    workspace = tmp_path / "workspace"
    owned = workspace / ".avs-i1"
    comfy = workspace / "ComfyUI"
    owned.mkdir(parents=True)
    attempt = owned / "attempts" / "0001"
    attempt.mkdir(parents=True)
    comfy.mkdir(parents=True)
    profile_path = attempt / "profile.json"
    profile_path.write_text(json.dumps(profile.data), encoding="utf-8")
    payload = {
        "schema": worker.JOB_SCHEMA,
        "run_id": "run-1",
        "nonce": "nonce-1",
        "workspace": str(workspace),
        "owned_root": str(attempt),
        "attempt_dir": str(attempt),
        "comfy_root": str(comfy),
        "profile_path": str(profile_path),
        "profile_sha256": profile.sha256,
        "target_home": str(tmp_path),
        "target_id": "t" * 64,
        "target_lock_path": str(
            tmp_path / ".cache" / "anime-video-studio" / "i1" / "targets" / ("t" * 64 + ".lock")
        ),
        "deadline_epoch": 9_999_999_999,
        "log": str(attempt / "download.log"),
        "status_path": str(attempt / "status.json"),
        "progress_path": str(attempt / "progress.json"),
        "heartbeat_path": str(attempt / "heartbeat.json"),
        "required_bytes": profile.disk_required_bytes,
    }
    job_path = attempt / "job.json"
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    return job_path, payload


def test_worker_complete_progress_and_target_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, payloads = _tiny_profile(tmp_path)
    job_path, payload = _job(tmp_path, profile, payloads)
    seen = []

    def fake_download(spec, *, root, progress, stage, log, **kwargs):
        seen.append(spec.model_id)
        stage("DOWNLOADING")
        target = Path(root) / spec.target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[spec.target])
        network_progress = kwargs.get("network_progress")
        if network_progress is not None:
            network_progress(0, spec.size_bytes)
            network_progress(spec.size_bytes, spec.size_bytes)
        progress(spec.size_bytes, spec.size_bytes)
        stage("VERIFYING")
        stage("DONE")
        from scripts.p1.setup_plan import DownloadResult

        return DownloadResult(spec, target, False, spec.sha256, spec.size_bytes)

    monkeypatch.setattr(worker.setup_plan, "download_verified", fake_download)
    assert worker.run(job_path, concurrency=1) == 0
    status = json.loads((Path(payload["status_path"])).read_text(encoding="utf-8"))
    progress = json.loads((Path(payload["progress_path"])).read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETE"
    assert progress["files_done"] == 4
    assert progress["bytes_done"] == progress["bytes_total"]
    assert seen == [item["id"] for item in profile.models]
    assert not (Path(payload["workspace"]) / ".avs-i1-download.lock").exists()


def test_download_verified_stage_callback_is_backward_compatible(tmp_path: Path) -> None:
    from scripts.p1 import setup_plan

    data = b"stage-callback-payload"
    spec = setup_plan.DownloadSpec(
        model_id="tiny",
        url="https://example.invalid/tiny.bin",
        target="models/tiny.bin",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )

    class Response:
        status = 200
        headers = {}

        def __init__(self):
            self.sent = False

        def read(self, n):
            if self.sent:
                return b""
            self.sent = True
            return data

        def close(self):
            pass

    stages = []
    progress = []

    def opener(url, headers):
        return Response()

    result = setup_plan.download_verified(
        spec,
        root=tmp_path,
        opener=opener,
        stage=stages.append,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert result.sha256 == spec.sha256
    assert stages == ["DOWNLOADING", "VERIFYING", "DONE"]
    assert progress[-1] == (len(data), len(data))


def test_remaining_disk_excludes_existing_final_and_counts_partial(tmp_path: Path) -> None:
    profile, payloads = _tiny_profile(tmp_path)
    data = dict(profile.data)
    data["disk"] = {"install_overhead_bytes": 1, "reserve_bytes": 2}
    profile = profile_mod.validate_profile(data)
    comfy = tmp_path / "ComfyUI"
    items = list(profile.models)
    first = comfy / items[0]["target"]
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"x" * items[0]["size_bytes"])
    second = comfy / items[1]["target"]
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(b"x" * (items[1]["size_bytes"] - 2))
    (second.with_name(second.name + ".part")).write_bytes(b"x" * 2)
    remaining = worker._remaining_need(profile, comfy)
    # first final is skipped; second partial needs the rest; other two full sizes
    expected = (
        (items[1]["size_bytes"] - 2)
        + items[2]["size_bytes"]
        + items[3]["size_bytes"]
        + 3
    )
    assert remaining == expected


def test_target_lock_is_shared_across_workspaces_and_profile_names(tmp_path: Path) -> None:
    home = tmp_path / "home"
    lock = home / ".cache" / "anime-video-studio" / "i1" / "targets" / ("t" * 64 + ".lock")
    first = worker.TargetLock(
        tmp_path / "workspace-a", "r1", "n1", lock_dir=lock, target_id_value="t" * 64
    ).acquire()
    try:
        with pytest.raises(worker.WorkerError):
            worker.TargetLock(
                tmp_path / "workspace-b",
                "r2",
                "n2",
                lock_dir=lock,
                target_id_value="t" * 64,
            ).acquire()
        other = worker.TargetLock(
            tmp_path / "workspace-b",
            "r2",
            "n2",
            lock_dir=home / ".cache" / "anime-video-studio" / "i1" / "targets" / ("u" * 64 + ".lock"),
            target_id_value="u" * 64,
        ).acquire()
        other.release()
    finally:
        first.release()


def test_worker_second_lock_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock1 = worker.TargetLock(workspace, "r", "n").acquire()
    try:
        with pytest.raises(worker.WorkerError):
            worker.TargetLock(workspace, "r2", "n2").acquire()
    finally:
        lock1.release()


def test_failed_lock_does_not_open_http_or_leave_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, _ = _tiny_profile(tmp_path)
    job_path, payload = _job(tmp_path, profile, {})
    target_lock = Path(payload["target_lock_path"])
    held = worker.TargetLock(
        Path(payload["workspace"]),
        "other-run",
        "other-nonce",
        lock_dir=target_lock,
        target_id_value=str(payload["target_id"]),
    ).acquire()
    calls: list[str] = []
    monkeypatch.setattr(
        worker.setup_plan,
        "download_verified",
        lambda *args, **kwargs: calls.append("http"),
    )
    try:
        with pytest.raises(worker.WorkerError):
            worker.run(job_path)
    finally:
        held.release()
    assert calls == []
    status = json.loads(Path(payload["status_path"]).read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert not [t for t in threading.enumerate() if t.name.startswith("i1-download-")]


def test_one_file_error_cancels_other_and_drains_all_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, payloads = _tiny_profile(tmp_path)
    job_path, payload = _job(tmp_path, profile, payloads)
    both_started = threading.Event()
    lock = threading.Lock()
    started: list[str] = []

    def fake_download(spec, *, root, progress, stage, log, **kwargs):
        with lock:
            started.append(spec.model_id)
            if len(started) >= 2:
                both_started.set()
        both_started.wait(5.0)
        if spec.model_id == profile.models[0]["id"]:
            raise RuntimeError("injected first-file failure")
        cancel = kwargs.get("cancel")
        while cancel is not None and not cancel():
            time.sleep(0.01)
        raise worker.WorkerError("cancelled sibling")

    monkeypatch.setattr(worker.setup_plan, "download_verified", fake_download)
    with pytest.raises(RuntimeError):
        worker.run(job_path)
    assert not [t for t in threading.enumerate() if t.name.startswith("i1-download-")]
    assert not (Path(payload["workspace"]) / ".avs-i1-download.lock").exists()
    status = json.loads(Path(payload["status_path"]).read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"


def test_first_error_is_visible_while_sibling_holds_actual_target_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, payloads = _tiny_profile(tmp_path)
    job_path, payload = _job(tmp_path, profile, payloads)
    both_started = threading.Event()
    hold_sibling = threading.Event()
    lock = threading.Lock()
    started: list[str] = []

    def fake_download(spec, *, root, progress, stage, log, **kwargs):
        with lock:
            started.append(spec.model_id)
            if len(started) >= 2:
                both_started.set()
        both_started.wait(5.0)
        if spec.model_id == profile.models[0]["id"]:
            raise RuntimeError("injected first-file failure")
        hold_sibling.wait(10.0)
        raise worker.WorkerError("sibling drained")

    monkeypatch.setattr(worker.setup_plan, "download_verified", fake_download)
    outcome: dict[str, BaseException] = {}

    def run_worker() -> None:
        try:
            worker.run(job_path)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run_worker, name="test-error-early")
    thread.start()
    try:
        deadline = time.monotonic() + 8.0
        status: dict = {}
        while time.monotonic() < deadline:
            try:
                status = json.loads(Path(payload["status_path"]).read_text(encoding="utf-8"))
                if status.get("status") == "FAILED":
                    break
            except (OSError, json.JSONDecodeError):
                pass
            time.sleep(0.02)
        assert status.get("status") == "FAILED"
        assert thread.is_alive() is True
        actual_lock = Path(payload["target_lock_path"])
        assert actual_lock != Path(payload["workspace"]) / ".avs-i1-download.lock"
        with pytest.raises(worker.WorkerError):
            worker.TargetLock(
                Path(payload["workspace"]),
                "second-installer",
                "second-nonce",
                lock_dir=actual_lock,
                target_id_value=str(payload["target_id"]),
            ).acquire()
        assert "i1-download-" in "\n".join(t.name for t in threading.enumerate())
    finally:
        hold_sibling.set()
        thread.join(15.0)
    assert thread.is_alive() is False
    assert isinstance(outcome.get("error"), RuntimeError)
    lock_after = worker.TargetLock(
        Path(payload["workspace"]),
        "after-release",
        "after-nonce",
        lock_dir=Path(payload["target_lock_path"]),
        target_id_value=str(payload["target_id"]),
    ).acquire()
    lock_after.release()


def test_expired_deadline_refuses_before_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, payloads = _tiny_profile(tmp_path)
    job_path, payload = _job(tmp_path, profile, payloads)
    payload["deadline_epoch"] = time.time() - 1
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        worker.setup_plan,
        "download_verified",
        lambda *args, **kwargs: calls.append("http"),
    )
    with pytest.raises(worker.WorkerError):
        worker.run(job_path)
    assert calls == []


def test_complete_is_not_published_while_a_download_thread_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, payloads = _tiny_profile(tmp_path)
    job_path, payload = _job(tmp_path, profile, payloads)
    release = threading.Event()
    slow_started = threading.Event()
    first_id = profile.models[0]["id"]

    def fake_download(spec, *, root, progress, stage, log, **kwargs):
        target = Path(root) / spec.target
        target.parent.mkdir(parents=True, exist_ok=True)
        if spec.model_id != first_id:
            slow_started.set()
            release.wait(10.0)
        target.write_bytes(payloads[spec.target])
        progress(len(payloads[spec.target]), len(payloads[spec.target]))
        from scripts.p1.setup_plan import DownloadResult

        return DownloadResult(spec, target, False, spec.sha256, spec.size_bytes)

    monkeypatch.setattr(worker.setup_plan, "download_verified", fake_download)
    outcome: dict[str, int] = {}

    def run_worker() -> None:
        outcome["code"] = worker.run(job_path)

    thread = threading.Thread(target=run_worker, name="test-worker-run")
    thread.start()
    try:
        assert slow_started.wait(5.0)
        status_path = Path(payload["status_path"])
        if status_path.exists():
            assert json.loads(status_path.read_text(encoding="utf-8"))["status"] != "COMPLETE"
    finally:
        release.set()
        thread.join(15.0)
    assert thread.is_alive() is False
    assert outcome.get("code") == 0
    assert json.loads(Path(payload["status_path"]).read_text(encoding="utf-8"))["status"] == "COMPLETE"


def _recovery_state(tmp_path: Path, *, pid: int, age: float) -> tuple[Path, dict, dict]:
    workspace = tmp_path / "workspace"
    root = workspace / ".avs-i1"
    attempt = root / "attempts" / "0001"
    attempt.mkdir(parents=True)
    owner = {
        "schema": "anime-video-studio/p1-v3-process-owner/v1",
        "kind": "download_worker",
        "run_id": "r",
        "nonce": "n",
        "command_sha256": "a" * 64,
        "pid": pid,
        "ack_at": 1.0,
    }
    (attempt / "worker-owner.json").write_text(json.dumps(owner), encoding="utf-8")
    (attempt / "worker-ack.json").write_text(json.dumps(owner), encoding="utf-8")
    (attempt / "status.json").write_text(
        json.dumps({"schema": "anime-video-studio/i1-download-status/v1", "run_id": "r", "nonce": "n", "status": "RUNNING"}),
        encoding="utf-8",
    )
    import time

    (attempt / "heartbeat.json").write_text(
        json.dumps({"run_id": "r", "nonce": "n", "heartbeat_at": time.time() - age}),
        encoding="utf-8",
    )
    target_home = tmp_path / "home"
    target_home.mkdir()
    target_id = "t" * 64
    lock = target_home / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{target_id}.lock"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"\0")
    lock.with_name(lock.name + ".owner.json").write_text(
        json.dumps(
            {"run_id": "r", "nonce": "n", "target_id": target_id, "pid": pid}
        ),
        encoding="utf-8",
    )
    return root, owner, {"workspace": workspace, "attempt": attempt, "lock": lock}


def test_recovery_requires_same_run_dead_pid_stale_heartbeat(tmp_path: Path) -> None:
    root, owner, meta = _recovery_state(tmp_path, pid=999999, age=120)
    result = worker_recover.recover(root, meta["attempt"], meta["lock"], "r", "n", 30)
    assert result["status"] == "RECOVERED_INTERRUPTED"
    status = json.loads((meta["attempt"] / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert status["recovered_interrupted"] is True
    assert (meta["attempt"] / "recovery.json").is_file()
    assert (meta["attempt"] / "worker-owner.json").is_file()


def test_recovery_refuses_live_pid_or_foreign_run(tmp_path: Path) -> None:
    root, _, meta = _recovery_state(tmp_path, pid=os.getpid(), age=120)
    with pytest.raises(worker_recover.RecoveryError):
        worker_recover.recover(root, meta["attempt"], meta["lock"], "r", "n", 30)
    root2, _, meta2 = _recovery_state(tmp_path / "other", pid=999999, age=120)
    with pytest.raises(worker_recover.RecoveryError):
        worker_recover.recover(root2, meta2["attempt"], meta2["lock"], "foreign", "n", 30)
