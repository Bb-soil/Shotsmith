from __future__ import annotations

import threading
from typing import Any

import pytest

from anime_studio.artifacts import LocalArtifactStore
from anime_studio.comfy import RemoteStatus
from anime_studio.db import Database
from anime_studio.errors import InstanceAlreadyRunning
from anime_studio.lock import InstanceLock
from anime_studio.orchestrator import Orchestrator
from anime_studio.worker import Worker
from tests.helpers import make_request


class BlockingAdapter:
    mode = "TEST"
    execution_fingerprint = "TEST"
    supports_reconcile_by_correlation_key = False
    supports_idempotent_submit = False

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def submit(
        self,
        correlation_key: str,
        payload: dict[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        del deadline_ms
        return RemoteStatus(
            prompt_id="shutdown-prompt",
            correlation_key=correlation_key,
            state="queued",
            progress=0,
        )

    def get(self, prompt_id: str) -> RemoteStatus | None:
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise TimeoutError("shutdown test adapter was not released")
        return RemoteStatus(
            prompt_id=prompt_id,
            correlation_key="",
            state="running",
            progress=10,
        )

    def cancel(self, prompt_id: str) -> RemoteStatus | None:
        return None

    def reconcile(self, correlation_key: str) -> RemoteStatus | None:
        return None


def test_worker_stop_reports_false_and_lock_is_not_released(tmp_path) -> None:
    adapter = BlockingAdapter()
    db = Database(tmp_path / "shutdown.sqlite3")
    db.init()
    orchestrator = Orchestrator(
        db,
        adapter,
        LocalArtifactStore(tmp_path / "artifacts"),
        lease_seconds=0.5,
    )
    orchestrator.create_job(
        make_request(prompt="shutdown test"), "shutdown-key-0001"
    )
    worker = Worker(orchestrator, poll_interval=0.01, lease_seconds=0.5)
    lock_path = tmp_path / "instance.lock"
    lock = InstanceLock(lock_path).acquire()
    worker.start()
    assert adapter.entered.wait(timeout=5)
    assert worker.stop(timeout=0.05) is False
    with pytest.raises(InstanceAlreadyRunning):
        InstanceLock(lock_path).acquire()
    adapter.release.set()
    assert worker.stop(timeout=5) is True
    lock.release()
    reacquired = InstanceLock(lock_path).acquire()
    reacquired.release()
