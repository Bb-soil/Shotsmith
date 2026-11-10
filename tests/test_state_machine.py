from __future__ import annotations

import time

import pytest

from anime_studio.comfy import RemoteStatus
from anime_studio.states import (
    IllegalTransition,
    JobStatus,
    can_transition,
    is_terminal,
)
from tests.helpers import make_request, make_stack


def test_transition_table_is_explicit() -> None:
    assert can_transition("queued", "dispatching")
    assert can_transition("dispatching", "submitted")
    assert can_transition("submitted", "running")
    assert can_transition("running", "succeeded")
    assert can_transition("submitted", "cancelled")
    assert can_transition("dispatching", "timed_out")
    assert not can_transition("queued", "succeeded")
    assert not can_transition("succeeded", "running")
    assert not can_transition("cancelled", "running")
    assert is_terminal("succeeded")
    assert is_terminal("failed")
    assert is_terminal("cancelled")
    assert is_terminal("timed_out")
    assert not is_terminal("running")


def test_illegal_transition_raises_inside_transaction(tmp_path) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(make_request(), "state-key-0001")
    with stack.db.transaction() as conn:
        with pytest.raises(IllegalTransition):
            stack.orchestrator._transition(
                conn, job["job_id"], "succeeded", "should_not_happen"
            )


def test_late_success_does_not_change_failed_job(tmp_path) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(make_request(), "state-key-0002")
    job_id = job["job_id"]
    stack.orchestrator.fail_job(job_id, "TEST_FAILURE", "deliberate failure")
    remote = RemoteStatus(
        prompt_id="mock-late-1",
        correlation_key=job["correlation_key"],
        state="succeeded",
        progress=100,
        artifact={"scheme": "mock", "mock": True, "content": {"mock": True}},
    )
    stack.orchestrator.apply_remote_state(job_id, remote)
    final = stack.orchestrator.get_job(job_id)
    assert final is not None
    assert final["status"] == "failed"
    events = stack.orchestrator.get_events(job_id, after=0, limit=100)
    assert any(event["type"] == "late_event_ignored" for event in events["events"])
    assert not any(event["type"] == "job_succeeded" for event in events["events"])


def test_remote_success_after_deadline_is_timed_out_not_succeeded(tmp_path) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(
        make_request(prompt="MOCK deadline commit guard", timeout_seconds=1),
        "deadline-commit-0001",
    )
    prompt_id = "mock-deadline-commit"
    with stack.db.transaction() as conn:
        stack.orchestrator._transition(
            conn,
            job["job_id"],
            JobStatus.DISPATCHING.value,
            "dispatch_started",
        )
        stack.orchestrator._transition(
            conn,
            job["job_id"],
            JobStatus.SUBMITTED.value,
            "remote_submitted",
            updates={
                "prompt_id": prompt_id,
                "remote_state": "running",
                "progress": 50,
            },
        )
    time.sleep(1.1)
    remote = RemoteStatus(
        prompt_id=prompt_id,
        correlation_key=job["correlation_key"],
        state="succeeded",
        progress=100,
        artifact={"scheme": "mock", "mock": True, "content": {"mock": True}},
    )
    final = stack.orchestrator.apply_remote_state(job["job_id"], remote)
    assert final is not None
    assert final["status"] == "timed_out"
    events = stack.orchestrator.get_events(
        job["job_id"], after=0, limit=100
    )["events"]
    types = [event["type"] for event in events]
    assert "job_timed_out" in types
    assert "job_succeeded" not in types


def test_late_success_after_timeout_is_ignored(tmp_path) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(
        make_request(prompt="MOCK late success after timeout"),
        "timeout-late-0001",
    )
    stack.orchestrator.mark_timeout(
        job["job_id"],
        deadline_at=job["deadline_at"],
        had_prompt_id=False,
        cancel_attempted=False,
        remote_state=None,
    )
    remote = RemoteStatus(
        prompt_id="mock-late-timeout",
        correlation_key=job["correlation_key"],
        state="succeeded",
        progress=100,
        artifact={"scheme": "mock", "mock": True, "content": {"mock": True}},
    )
    stack.orchestrator.apply_remote_state(job["job_id"], remote)
    final = stack.orchestrator.get_job(job["job_id"])
    assert final is not None
    assert final["status"] == "timed_out"
    events = stack.orchestrator.get_events(
        job["job_id"], after=0, limit=100
    )["events"]
    types = [event["type"] for event in events]
    assert "late_event_ignored" in types
    assert "job_succeeded" not in types


def test_late_success_does_not_change_cancelled_job(tmp_path) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(make_request(), "state-key-0003")
    job_id = job["job_id"]
    cancelled = stack.orchestrator.request_cancel(job_id)
    assert cancelled["status"] == "cancelled"
    remote = RemoteStatus(
        prompt_id="mock-late-2",
        correlation_key=job["correlation_key"],
        state="succeeded",
        progress=100,
        artifact={"scheme": "mock", "mock": True, "content": {"mock": True}},
    )
    stack.orchestrator.apply_remote_state(job_id, remote)
    final = stack.orchestrator.get_job(job_id)
    assert final is not None
    assert final["status"] == "cancelled"
    events = stack.orchestrator.get_events(job_id, after=0, limit=100)
    assert any(event["type"] == "late_event_ignored" for event in events["events"])
