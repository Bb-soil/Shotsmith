from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.p1 import setup_plan, v3_download_worker as worker
from scripts.p1 import v3_lifecycle as lc


def _manifest(path: Path) -> None:
    models = []
    for index in range(3):
        models.append(
            {
                "id": f"m{index}",
                "download_url": f"https://example.invalid/m{index}.bin",
                "target": f"models/m{index}.bin",
                "size_bytes": 10 + index,
                "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            }
        )
    path.write_text(json.dumps({"models": models}), encoding="utf-8")


def _payload(tmp_path: Path) -> tuple[Path, dict]:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)
    payload = {
        "schema": "anime-video-studio/p1-v3-download-job/v1",
        "run_id": "run-1",
        "nonce": "nonce-1",
        "workspace": str(tmp_path / "workspace"),
        "root": str(tmp_path / "ComfyUI"),
        "deadline_epoch": time.time() + 60,
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "status_path": str(tmp_path / "workspace" / "status.json"),
        "progress_path": str(tmp_path / "workspace" / "progress.json"),
        "heartbeat_path": str(tmp_path / "workspace" / "heartbeat.json"),
    }
    payload_path = tmp_path / "job.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return payload_path, payload


def test_download_worker_complete_and_releases_lock(tmp_path: Path, monkeypatch) -> None:
    payload_path, payload = _payload(tmp_path)
    calls: list[str] = []

    def fake_download(spec, *, root, progress, log):
        calls.append(spec.model_id)
        target = Path(root) / spec.target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * spec.size_bytes)
        progress(spec.size_bytes, spec.size_bytes)
        return SimpleNamespace(
            size_bytes=spec.size_bytes,
            sha256=spec.sha256,
            skipped=False,
        )

    monkeypatch.setattr(setup_plan, "download_verified", fake_download)
    assert worker.run(payload_path) == 0
    status = json.loads(Path(payload["status_path"]).read_text())
    progress = json.loads(Path(payload["progress_path"]).read_text())
    assert status["status"] == "COMPLETE"
    assert progress["status"] == "COMPLETE"
    assert progress["files_done"] == 3
    assert calls == ["m0", "m1", "m2"]
    assert not (tmp_path / "workspace" / ".p1-download.lock").exists()


def test_download_worker_rejects_expired_deadline(tmp_path: Path) -> None:
    payload_path, payload = _payload(tmp_path)
    payload["deadline_epoch"] = time.time() - 1
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(worker.WorkerError):
        worker.run(payload_path)
    assert not (tmp_path / "workspace" / ".p1-download.lock").exists()


def test_download_worker_rejects_existing_owner(tmp_path: Path, monkeypatch) -> None:
    payload_path, payload = _payload(tmp_path)
    held = worker.TargetLock(tmp_path / "workspace", run_id="other", nonce="other").acquire()
    try:
        with pytest.raises(worker.WorkerError):
            worker.run(payload_path)
    finally:
        held.release()
    status = json.loads(Path(payload["status_path"]).read_text())
    assert status["status"] == "FAILED"
    assert not (tmp_path / "workspace" / ".p1-download.lock").exists()

def test_worker_main_records_same_run_failure_before_recorder(tmp_path: Path) -> None:
    payload_path, payload = _payload(tmp_path)
    payload["deadline_epoch"] = time.time() - 1
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    assert worker.main([str(payload_path)]) == 2
    status = json.loads(Path(payload["status_path"]).read_text())
    assert status["run_id"] == "run-1"
    assert status["nonce"] == "nonce-1"
    assert status["status"] == "FAILED"
    progress = json.loads(Path(payload["progress_path"]).read_text())
    heartbeat = json.loads(Path(payload["heartbeat_path"]).read_text())
    assert progress["status"] == "FAILED"
    assert heartbeat["status"] == "FAILED"
