from __future__ import annotations

from anime_studio.states import JobStatus
from anime_studio.timeutils import now_ms
from tests.helpers import make_request, make_stack


def test_cancel_in_dispatch_window_prevents_remote_submit(tmp_path) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(
        make_request(prompt="MOCK cancel in dispatch window"),
        "cancel-window-key-0001",
    )
    with stack.db.transaction() as conn:
        stack.orchestrator._transition(
            conn,
            job["job_id"],
            JobStatus.DISPATCHING.value,
            "dispatch_started",
            updates={
                "lease_owner": "old-worker",
                "lease_expires_at_ms": now_ms() - 1,
            },
        )

    cancelled = stack.orchestrator.request_cancel_after_dispatch(job["job_id"])
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert stack.adapter.count_remote_jobs() == 0
    events = stack.orchestrator.get_events(job["job_id"], after=0, limit=100)
    cancelled_events = [
        event for event in events["events"] if event["type"] == "job_cancelled"
    ]
    assert cancelled_events[-1]["data"]["reason"] == "cancel_before_remote_submit"
