from __future__ import annotations

from pathlib import Path

from anime_studio.states import JobStatus
from anime_studio.timeutils import now_ms
from tests.comfy_stub import run_stub
from tests.helpers import make_request, wait_direct
from tests.p1_helpers import make_real_stack


def _job(stack, prompt: str = "REAL fixture prompt"):
    job, _ = stack.orchestrator.create_job(
        make_request(prompt=prompt, seed=42, timeout_seconds=30),
        "real-e2e-" + prompt.replace(" ", "-")[:40],
    )
    return job


def test_real_submit_materializes_non_mock_result_and_dynamic_metadata(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        job = _job(stack)
        stack.worker.tick()
        dispatched = stack.orchestrator.get_job(job["job_id"])
        assert dispatched is not None
        assert dispatched["prompt_id"] == "prompt-1"
        assert dispatched["mock"] is False
        assert dispatched["mode"] == "COMFY_HTTP"

        state.set_success()
        final = wait_direct(
            stack.worker,
            job_id=job["job_id"],
            orchestrator=stack.orchestrator,
        )
        assert final["status"] == JobStatus.SUCCEEDED.value
        assert final["result"]["mock"] is False
        assert final["result"]["mode"] == "COMFY_HTTP"
        assert final["result"]["artifacts"]
        artifact = final["result"]["artifacts"][0]
        assert artifact["mock"] is False
        assert artifact["media_type"] == "video/mp4"
        assert Path(artifact["path"]).suffix == ".mp4"
        assert Path(artifact["path"]).exists()
        assert not list((tmp_path / "artifacts").rglob("*.mock.json"))
        events = stack.orchestrator.get_events(job["job_id"], after=0, limit=100)
        succeeded = [e for e in events["events"] if e["type"] == "job_succeeded"]
        assert succeeded and succeeded[-1]["data"]["mock"] is False
        assert state.prompt_requests == 1


def test_real_validation_failure_is_terminal_without_resubmit(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_mode = "validation"
        stack = make_real_stack(tmp_path, base_url=base_url)
        job = _job(stack, "REAL validation")
        stack.worker.tick()
        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.FAILED.value
        assert final["error"]["code"] == "REMOTE_VALIDATION_REJECTED"
        assert state.prompt_requests == 1
        stack.worker.tick()
        assert state.prompt_requests == 1


def test_real_ambiguous_submit_stays_ambiguous_and_is_not_resubmitted(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_mode = "ambiguous_500"
        stack = make_real_stack(tmp_path, base_url=base_url)
        job = _job(stack, "REAL ambiguous")
        stack.worker.tick()
        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.FAILED.value
        assert final["error"]["code"] == "AMBIGUOUS_DISPATCH"
        events = stack.orchestrator.get_events(job["job_id"], after=0, limit=100)
        assert any(e["type"] == "dispatch_ambiguous" for e in events["events"])
        stack.worker.tick()
        assert state.prompt_requests == 1


def test_real_crash_window_without_reconcile_preserves_ambiguity(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        job = _job(stack, "REAL crash window")
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
        stack.worker.recover_stale_dispatches()
        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.FAILED.value
        assert final["error"]["code"] == "AMBIGUOUS_DISPATCH"
        assert "not confirmed" in final["error"]["message"].lower()
        assert state.prompt_requests == 0


def test_real_lost_prompt_is_not_success(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url, prompt_lost_grace_seconds=0)
        job = _job(stack, "REAL lost")
        stack.worker.tick()
        state.queue_pending = []
        state.history_missing = True
        stack.worker.tick()
        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.FAILED.value
        assert final["error"]["code"] == "REMOTE_PROMPT_LOST"
        events = stack.orchestrator.get_events(job["job_id"], after=0, limit=100)
        assert any(e["type"] == "remote_prompt_lost" for e in events["events"])


def test_real_missing_outputs_is_not_success(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        job = _job(stack, "REAL missing outputs")
        stack.worker.tick()
        state.history[state.prompt_id] = {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {},
        }
        stack.worker.tick()
        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.FAILED.value
        assert final["error"]["code"] == "REMOTE_RESULT_MISSING"
        assert not list((tmp_path / "artifacts").rglob("result.mock.json"))

def test_real_pending_cancel_end_to_end(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        job = _job(stack, "REAL pending cancel")
        stack.worker.tick()
        assert stack.orchestrator.get_job(job["job_id"])["prompt_id"] == "prompt-1"
        cancelled = stack.orchestrator.request_cancel(job["job_id"])
        assert cancelled["status"] == JobStatus.CANCELLED.value
        assert state.delete_calls == [["prompt-1"]]
        assert state.interrupt_calls == 0


def test_real_shared_running_cancel_is_unconfirmed_not_cancelled(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url, exclusive_worker=False)
        job = _job(stack, "REAL shared cancel")
        stack.worker.tick()
        state.queue_pending = []
        state.queue_running = ["prompt-1"]
        result = stack.orchestrator.request_cancel(job["job_id"])
        assert result["status"] in {JobStatus.SUBMITTED.value, JobStatus.RUNNING.value}
        events = stack.orchestrator.get_events(job["job_id"], after=0, limit=100)
        assert any(e["type"] == "cancel_unconfirmed" for e in events["events"])
        assert state.interrupt_calls == 0

def test_real_crash_window_cancel_without_prompt_is_not_cancelled(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        job = _job(stack, "REAL crash cancel")
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
        requested = stack.orchestrator.request_cancel(job["job_id"])
        assert requested["cancel_requested"] is True
        stack.worker.recover_stale_dispatches()
        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.FAILED.value
        assert final["error"]["code"] == "AMBIGUOUS_DISPATCH"
        assert final["status"] != JobStatus.CANCELLED.value
        assert state.prompt_requests == 0
        assert state.interrupt_calls == 0

def test_real_malformed_submit_becomes_terminal_ambiguous_not_stuck(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_mode = "malformed_200"
        stack = make_real_stack(tmp_path, base_url=base_url)
        job = _job(stack, "REAL malformed")
        stack.worker.tick()
        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.FAILED.value
        assert final["error"]["code"] == "AMBIGUOUS_DISPATCH"
        assert state.prompt_requests == 1
        stack.worker.tick()
        assert state.prompt_requests == 1
