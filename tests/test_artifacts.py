from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from anime_studio.artifacts import LocalArtifactStore
from anime_studio.errors import ArtifactError
from anime_studio.states import JobStatus
from tests.helpers import make_request, make_stack, wait_direct


class SlowArtifactStore:
    """Test double that crosses the deadline while materializing."""

    def __init__(self, real: LocalArtifactStore, delay: float) -> None:
        self.real = real
        self.delay = delay

    def fetch_result(self, ref: dict[str, Any], job_id: str) -> dict[str, Any]:
        time.sleep(self.delay)
        return self.real.fetch_result(ref, job_id)


class BlockingArtifactStore:
    """Test double that holds the first finalizer inside artifact IO."""

    def __init__(self, real: LocalArtifactStore) -> None:
        self.real = real
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._calls_lock = threading.Lock()

    def fetch_result(self, ref: dict[str, Any], job_id: str) -> dict[str, Any]:
        with self._calls_lock:
            self.calls += 1
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise TimeoutError("test artifact barrier was not released")
        return self.real.fetch_result(ref, job_id)


def test_success_writes_only_a_small_mock_json_artifact(tmp_path) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(
        make_request(prompt="MOCK artifact boundary", duration_seconds=0.3),
        "artifact-key-0001",
    )
    final = wait_direct(
        stack.worker,
        job_id=job["job_id"],
        orchestrator=stack.orchestrator,
        timeout=15.0,
    )
    assert final["status"] == "succeeded", final
    artifact = final["result"]["artifacts"][0]
    path = Path(artifact["path"])
    assert path.exists()
    assert path.name == "result.mock.json"
    assert artifact["mock"] is True
    assert artifact["media_type"] == "application/json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["mock"] is True
    assert "NOT A REAL VIDEO" in body["note"]
    assert not list(Path(tmp_path / "artifacts").rglob("*.mp4"))
    assert not list(Path(tmp_path / "artifacts").rglob("*.mov"))


def test_real_or_unknown_artifact_scheme_is_rejected(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ArtifactError):
        store.fetch_result({"scheme": "https", "url": "https://example.invalid"}, "job-1")
    with pytest.raises(ArtifactError):
        store.fetch_result({"scheme": "mock"}, "job-2")


def test_concurrent_artifact_writes_use_unique_temps(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    ref = {
        "scheme": "mock",
        "mock": True,
        "content": {"mock": True, "note": "MOCK ARTIFACT - NOT A REAL VIDEO"},
    }
    barrier = threading.Barrier(2)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def write() -> None:
        barrier.wait(timeout=10)
        try:
            results.append(store.fetch_result(ref, "same-job"))
        except BaseException as exc:  # noqa: BLE001 - test captures all failures
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert errors == []
    assert len(results) == 2
    assert results[0]["path"] == results[1]["path"]
    path = Path(results[0]["path"])
    assert path.exists()
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == results[0]["sha256"]
    assert not list(path.parent.glob("*.tmp"))


def test_deadline_crossed_during_artifact_io_is_not_committed_as_success(
    tmp_path,
) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(
        make_request(
            prompt="MOCK deadline during artifact IO",
            timeout_seconds=1,
            duration_seconds=0.1,
        ),
        "artifact-deadline-0001",
    )
    with stack.db.read() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)
        ).fetchone()
    payload = json.loads(row["payload_canonical"])
    initial = stack.adapter.submit(job["correlation_key"], payload)
    deadline = time.monotonic() + 10
    remote = None
    while time.monotonic() < deadline:
        remote = stack.adapter.get(initial.prompt_id)
        if remote is not None and remote.state == "succeeded":
            break
        time.sleep(0.02)
    assert remote is not None and remote.state == "succeeded"

    with stack.db.transaction() as conn:
        stack.orchestrator._transition(
            conn,
            job["job_id"],
            JobStatus.DISPATCHING.value,
            "dispatch_started",
        )
    linked = stack.orchestrator.link_remote(
        job["job_id"], remote, apply_result=False
    )
    assert linked is not None and linked["status"] == "submitted"

    stack.orchestrator.artifacts = SlowArtifactStore(stack.artifacts, delay=1.2)
    final = stack.orchestrator.apply_remote_state(job["job_id"], remote)
    assert final is not None
    assert final["status"] == "timed_out", final
    events = stack.orchestrator.get_events(
        job["job_id"], after=0, limit=200
    )["events"]
    assert not any(event["type"] == "job_succeeded" for event in events)
    assert any(event["type"] == "job_timed_out" for event in events)


def test_concurrent_completion_and_cancel_finalize_once(tmp_path) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(
        make_request(prompt="MOCK completion/cancel race", duration_seconds=0.1),
        "artifact-race-0001",
    )
    with stack.db.read() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)
        ).fetchone()
    payload = json.loads(row["payload_canonical"])
    initial = stack.adapter.submit(job["correlation_key"], payload)
    deadline = time.monotonic() + 10
    remote = None
    while time.monotonic() < deadline:
        remote = stack.adapter.get(initial.prompt_id)
        if remote is not None and remote.state == "succeeded":
            break
        time.sleep(0.02)
    assert remote is not None and remote.state == "succeeded"

    with stack.db.transaction() as conn:
        stack.orchestrator._transition(
            conn,
            job["job_id"],
            JobStatus.DISPATCHING.value,
            "dispatch_started",
        )
    linked = stack.orchestrator.link_remote(
        job["job_id"], remote, apply_result=False
    )
    assert linked is not None and linked["prompt_id"] == remote.prompt_id
    assert linked["status"] == "submitted"

    blocking = BlockingArtifactStore(stack.artifacts)
    stack.orchestrator.artifacts = blocking
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def completion() -> None:
        barrier.wait(timeout=10)
        try:
            stack.orchestrator.apply_remote_state(job["job_id"], remote)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def cancel() -> None:
        barrier.wait(timeout=10)
        try:
            stack.orchestrator.request_cancel(job["job_id"])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    first = threading.Thread(target=completion)
    second = threading.Thread(target=cancel)
    first.start()
    second.start()
    assert blocking.entered.wait(timeout=10)
    time.sleep(0.1)
    blocking.release.set()
    first.join(timeout=15)
    second.join(timeout=15)

    assert errors == []
    assert blocking.calls == 1
    final = stack.orchestrator.get_job(job["job_id"])
    assert final is not None
    assert final["status"] == "succeeded", final
    events = stack.orchestrator.get_events(
        job["job_id"], after=0, limit=200
    )["events"]
    assert sum(1 for event in events if event["type"] == "job_succeeded") == 1
    assert sum(
        1 for event in events if event["type"] == "late_event_ignored"
    ) <= 1
    artifact = final["result"]["artifacts"][0]
    path = Path(artifact["path"])
    assert path.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    assert not list(path.parent.glob("*.tmp"))


def test_artifact_io_error_maps_to_artifact_error(tmp_path, monkeypatch) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    ref = {"scheme": "mock", "mock": True, "content": {"mock": True}}

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_atomic_write", fail_write)
    with pytest.raises(ArtifactError) as exc_info:
        store.fetch_result(ref, "job-io-error")
    assert exc_info.value.code == "ARTIFACT_IO_ERROR"
