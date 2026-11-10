from __future__ import annotations

from typing import Any

from anime_studio.artifacts import LocalArtifactStore
from anime_studio.comfy import RemoteStatus
from anime_studio.db import Database
from anime_studio.orchestrator import Orchestrator
from anime_studio.states import JobStatus
from anime_studio.timeutils import now_iso, now_ms
from anime_studio.worker import Worker
from tests.helpers import make_request


class NoReconcileAdapter:
    mode = "NO_RECONCILE"
    supports_reconcile_by_correlation_key = False
    supports_idempotent_submit = False

    def __init__(self) -> None:
        self.submit_calls = 0

    def submit(
        self,
        correlation_key: str,
        payload: dict[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        del deadline_ms
        self.submit_calls += 1
        raise AssertionError("must not submit without reconciliation")

    def get(self, prompt_id: str) -> RemoteStatus | None:
        return None

    def cancel(self, prompt_id: str) -> RemoteStatus | None:
        return None

    def reconcile(self, correlation_key: str) -> RemoteStatus | None:
        return None


def _force_dispatching(orchestrator: Orchestrator, job_id: str) -> None:
    with orchestrator.db.transaction() as conn:
        orchestrator._transition(
            conn,
            job_id,
            JobStatus.DISPATCHING.value,
            "dispatch_started",
            {"test": True},
            updates={
                "dispatch_started_at": now_iso(),
                "lease_owner": "old-worker",
                "lease_expires_at_ms": now_ms() - 1,
            },
        )


def test_stale_dispatch_without_reconciliation_is_explicit(tmp_path) -> None:
    db = Database(tmp_path / "p0.sqlite3")
    db.init()
    adapter = NoReconcileAdapter()
    orchestrator = Orchestrator(
        db,
        adapter,  # type: ignore[arg-type]
        LocalArtifactStore(tmp_path / "artifacts"),
        lease_seconds=0.5,
    )
    worker = Worker(orchestrator, poll_interval=0.02, lease_seconds=0.5)
    job, _ = orchestrator.create_job(
        make_request(prompt="MOCK ambiguous dispatch"), "ambiguity-key-0001"
    )
    _force_dispatching(orchestrator, job["job_id"])

    worker.recover_stale_dispatches()
    final = orchestrator.get_job(job["job_id"])
    assert final is not None
    assert final["status"] == "failed"
    assert final["error"]["code"] == "AMBIGUOUS_DISPATCH"
    assert "Manual reconciliation" in final["error"]["message"]
    assert adapter.submit_calls == 0
    events = orchestrator.get_events(job["job_id"], after=0, limit=100)
    assert any(event["type"] == "dispatch_ambiguous" for event in events["events"])


def test_link_remote_binds_prompt_id_to_internal_job(tmp_path) -> None:
    db = Database(tmp_path / "p0.sqlite3")
    db.init()
    orchestrator = Orchestrator(
        db,
        NoReconcileAdapter(),  # type: ignore[arg-type]
        LocalArtifactStore(tmp_path / "artifacts"),
    )
    job, _ = orchestrator.create_job(
        make_request(prompt="MOCK prompt mapping"), "mapping-key-0001"
    )
    _force_dispatching(orchestrator, job["job_id"])
    remote = RemoteStatus(
        prompt_id="mock-prompt-123",
        correlation_key=job["correlation_key"],
        state="queued",
        progress=0,
    )
    linked = orchestrator.link_remote(job["job_id"], remote)
    assert linked is not None
    assert linked["status"] == "submitted"
    assert linked["prompt_id"] == "mock-prompt-123"
    assert linked["correlation_key"] == job["correlation_key"]
