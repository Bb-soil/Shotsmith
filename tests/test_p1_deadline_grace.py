"""Regressions for the P1 offline review findings F1 and F2.

All tests use the loopback ComfyUI stub. No external network, GPU, model or
RunPod call is made. Wall-clock and monotonic clocks are injected where the
deadline/grace boundary matters so the tests do not depend on sleeps.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from anime_studio.artifacts import LocalArtifactStore
from anime_studio.comfy import RemoteStatus
from anime_studio.db import Database
from anime_studio.errors import (
    RemoteAmbiguousSubmit,
    RemoteDeadlineExpired,
    RemotePromptLost,
    RemoteProtocolError,
    RemoteTransientError,
)
from anime_studio.orchestrator import Orchestrator
from anime_studio.states import JobStatus
from anime_studio.timeutils import ms_to_iso, now_ms
from anime_studio.worker import Worker
from tests.comfy_stub import run_stub
from tests.helpers import make_request
from tests.p1_helpers import make_real_stack, make_transport


def _wait_for_upload(state: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.upload_requests == 1:
            return
        time.sleep(0.01)
    assert state.upload_requests == 1


class _BindingProxy:
    """Delegates to WorkflowBinding but advances a fake clock after bind()."""

    def __init__(self, inner: Any, clock: list[int], deadline_ms: int) -> None:
        self._inner = inner
        self._clock = clock
        self._deadline_ms = deadline_ms

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def bind(self, *args: Any, **kwargs: Any) -> Any:
        bound = self._inner.bind(*args, **kwargs)
        self._clock[0] = self._deadline_ms + 1
        return bound


class _RecordingIdempotentAdapter:
    """Minimal adapter used to prove recovery passes the stored deadline."""

    mode = "TEST"
    execution_fingerprint = "TEST"
    supports_reconcile_by_correlation_key = True
    supports_idempotent_submit = True

    def __init__(self) -> None:
        self.deadline_ms: int | None = None
        self.submit_calls = 0

    def reconcile(self, correlation_key: str) -> RemoteStatus | None:
        return None

    def submit(
        self,
        correlation_key: str,
        payload: dict[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        self.deadline_ms = deadline_ms
        self.submit_calls += 1
        return RemoteStatus(
            prompt_id="recovery-prompt",
            correlation_key=correlation_key,
            state="queued",
            progress=0,
        )

    def get(self, prompt_id: str) -> RemoteStatus | None:
        return None

    def cancel(self, prompt_id: str) -> RemoteStatus | None:
        return None


class _ReconcileDeadlineAdapter:
    """Reconcile crosses the job deadline; submit must never be called."""

    mode = "TEST"
    execution_fingerprint = "TEST"

    def __init__(
        self,
        clock: list[int],
        *,
        remote: RemoteStatus | None = None,
        supports_reconcile: bool = True,
    ) -> None:
        self._clock = clock
        self._remote = remote
        self.supports_reconcile_by_correlation_key = supports_reconcile
        self.supports_idempotent_submit = True
        self.submit_calls = 0
        self.cancel_calls: list[str] = []

    def reconcile(self, correlation_key: str) -> RemoteStatus | None:
        self._clock[0] += 1000
        return self._remote

    def submit(
        self,
        correlation_key: str,
        payload: dict[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        del deadline_ms
        self.submit_calls += 1
        raise AssertionError("expired recovery must never submit")

    def get(self, prompt_id: str) -> RemoteStatus | None:
        return None

    def cancel(self, prompt_id: str) -> RemoteStatus | None:
        self.cancel_calls.append(prompt_id)
        return RemoteStatus(
            prompt_id=prompt_id,
            correlation_key="",
            state="cancelled",
            progress=0,
        )


def _force_stale_dispatching(orchestrator: Orchestrator, job_id: str) -> None:
    with orchestrator.db.transaction() as conn:
        orchestrator._transition(
            conn,
            job_id,
            JobStatus.DISPATCHING.value,
            "dispatch_started",
            {"test": True},
            updates={
                "lease_owner": "old-worker",
                "lease_expires_at_ms": now_ms() - 1,
            },
        )


# --------------------------------------------------------------------- F1
def test_expired_deadline_is_rejected_before_upload_and_prompt(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(RemoteDeadlineExpired) as exc_info:
            transport.submit(
                "corr-expired",
                {"prompt": "expired", "seed": 1},
                deadline_ms=now_ms() - 1,
            )
        assert exc_info.value.code == "DEADLINE_EXPIRED_BEFORE_SEND"
        assert state.upload_requests == 0
        assert state.prompt_requests == 0


def test_sufficient_deadline_sends_exactly_one_prompt(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        remote = transport.submit(
            "corr-enough",
            {"prompt": "enough budget", "seed": 1},
            deadline_ms=now_ms() + 5000,
        )
        assert remote.prompt_id == "prompt-1"
        assert state.upload_requests == 1
        assert state.prompt_requests == 1


def test_binding_latency_crossing_deadline_sends_no_prompt(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        clock = [now_ms()]
        deadline = clock[0] + 1000
        transport._wall_clock = lambda: clock[0]
        transport.binding = _BindingProxy(transport.binding, clock, deadline)
        with pytest.raises(RemoteDeadlineExpired):
            transport.submit(
                "corr-binding",
                {"prompt": "binding latency", "seed": 1},
                deadline_ms=deadline,
            )
        assert state.upload_requests == 1
        assert state.prompt_requests == 0


def test_cached_upload_still_checks_deadline_after_binding(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        transport._uploaded_image_name = "source.png"
        clock = [now_ms()]
        deadline = clock[0] + 1000
        transport._wall_clock = lambda: clock[0]
        transport.binding = _BindingProxy(transport.binding, clock, deadline)
        with pytest.raises(RemoteDeadlineExpired):
            transport.submit(
                "corr-cached",
                {"prompt": "cached upload", "seed": 1},
                deadline_ms=deadline,
            )
        assert state.upload_requests == 0
        assert state.prompt_requests == 0


def test_async_helper_rechecks_deadline_immediately_before_prompt(
    tmp_path: Path,
) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(RemoteDeadlineExpired):
            asyncio.run(
                transport._submit_prompt_async(
                    {"prompt": {}},
                    "corr-helper",
                    deadline_ms=now_ms() - 1,
                )
            )
        assert state.prompt_requests == 0


def test_deadline_context_reaches_the_async_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, _state):
        transport = make_transport(tmp_path, base_url=base_url)
        observed: dict[str, int | None] = {}
        original = transport._submit_prompt_async

        async def observe(body: dict[str, Any], correlation_key: str):
            observed["deadline_ms"] = transport._active_deadline_ms.get()
            return await original(body, correlation_key)

        monkeypatch.setattr(transport, "_submit_prompt_async", observe)
        deadline = now_ms() + 5000
        transport.submit(
            "corr-context",
            {"prompt": "context propagation", "seed": 1},
            deadline_ms=deadline,
        )
        assert observed["deadline_ms"] == deadline


class _SendBarrierTransport(httpx.AsyncBaseTransport):
    """Test-only async transport with a deterministic post-send barrier.

    The barrier is set when the transport receives the request, i.e. after the
    client has crossed its send boundary. The delayed response then exceeds the
    HTTP budget. This removes the old host-dependent race without changing any
    production timeout or the real transport.
    """

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.sent = asyncio.Event()
        self.requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        self.sent.set()
        await asyncio.sleep(self.delay_seconds)
        return httpx.Response(200, json={"prompt_id": "post-send", "node_errors": {}})


def test_post_send_timeout_remains_ambiguous_with_valid_deadline(
    tmp_path: Path,
) -> None:
    transport = make_transport(
        tmp_path, base_url="http://127.0.0.1:1", read_timeout_seconds=0.2
    )
    fake = _SendBarrierTransport(delay_seconds=1.0)
    transport._async_transport = fake

    with pytest.raises(RemoteAmbiguousSubmit):
        asyncio.run(
            transport._submit_prompt_async(
                {"prompt": {"1": {"class_type": "Test", "inputs": {}}}, "client_id": "test"},
                "corr-ambiguous",
                deadline_ms=now_ms() + 5000,
            )
        )

    assert fake.sent.is_set(), "test did not cross the post-send boundary"
    assert fake.requests == 1


def test_pre_send_deadline_never_reaches_transport(tmp_path: Path) -> None:
    transport = make_transport(
        tmp_path, base_url="http://127.0.0.1:1", read_timeout_seconds=0.2
    )
    fake = _SendBarrierTransport(delay_seconds=1.0)
    transport._async_transport = fake

    with pytest.raises(RemoteDeadlineExpired):
        asyncio.run(
            transport._submit_prompt_async(
                {"prompt": {"1": {"class_type": "Test", "inputs": {}}}, "client_id": "test"},
                "corr-pre-send",
                deadline_ms=now_ms() - 1,
            )
        )

    assert not fake.sent.is_set(), "pre-send deadline must not reach the transport"
    assert fake.requests == 0


def test_worker_upload_crossing_deadline_times_out_without_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        job, _ = stack.orchestrator.create_job(
            make_request(prompt="F1 worker deadline", timeout_seconds=30),
            "f1-worker-deadline-0001",
        )
        clock = [now_ms()]
        deadline = clock[0] + 1000
        with stack.db.transaction() as conn:
            conn.execute(
                "UPDATE jobs SET deadline_ts=?, deadline_at=? WHERE job_id=?",
                (deadline, ms_to_iso(deadline), job["job_id"]),
            )
        stack.transport._wall_clock = lambda: clock[0]
        monkeypatch.setattr("anime_studio.worker.now_ms", lambda: clock[0])

        original_upload = stack.transport._upload_input

        def upload_then_cross(deadline_ms: int | None = None) -> str:
            name = original_upload(deadline_ms)
            clock[0] = deadline + 1
            return name

        monkeypatch.setattr(stack.transport, "_upload_input", upload_then_cross)
        stack.worker.tick()
        _wait_for_upload(state)

        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.TIMED_OUT.value
        assert final["error"]["code"] == "DEADLINE_EXCEEDED"
        assert state.upload_requests == 1
        assert state.prompt_requests == 0
        events = stack.orchestrator.get_events(job["job_id"], after=0, limit=100)
        assert not any(event["type"] == "job_succeeded" for event in events["events"])
        timed_out = [
            event for event in events["events"] if event["type"] == "job_timed_out"
        ]
        assert timed_out
        assert timed_out[-1]["data"]["had_prompt_id"] is False
        assert timed_out[-1]["data"]["classification"] == "confirmed_not_sent"
        assert timed_out[-1]["data"]["accepted"] is False
        assert timed_out[-1]["data"]["prompt_requests"] == 0


def test_build_request_crossing_wall_deadline_does_not_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        clock = [now_ms()]
        deadline = clock[0] + 5000
        transport._wall_clock = lambda: clock[0]
        original_build = httpx.AsyncClient.build_request

        def build_request(self, method, url, **kwargs):
            request = original_build(self, method, url, **kwargs)
            if request.url.path == "/prompt":
                clock[0] = deadline
            return request

        monkeypatch.setattr(httpx.AsyncClient, "build_request", build_request)
        with pytest.raises(RemoteDeadlineExpired):
            transport.submit(
                "corr-build-wall",
                {"prompt": "build wall", "seed": 1},
                deadline_ms=deadline,
            )
        assert state.upload_requests == 1
        assert state.prompt_requests == 0


def test_build_request_crossing_monotonic_budget_does_not_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(
            tmp_path, base_url=base_url, read_timeout_seconds=0.01
        )
        original_build = httpx.AsyncClient.build_request

        def build_request(self, method, url, **kwargs):
            request = original_build(self, method, url, **kwargs)
            if request.url.path == "/prompt":
                time.sleep(0.05)
            return request

        monkeypatch.setattr(httpx.AsyncClient, "build_request", build_request)
        # The job deadline is still valid; only the HTTP request budget expired
        # before send, so keep the conservative ambiguous classification.
        with pytest.raises(RemoteAmbiguousSubmit):
            transport.submit(
                "corr-build-mono",
                {"prompt": "build mono", "seed": 1},
                deadline_ms=now_ms() + 5000,
            )
        assert state.prompt_requests == 0


def test_worker_build_request_crossing_deadline_times_out_without_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        job, _ = stack.orchestrator.create_job(
            make_request(prompt="F1 build request deadline", timeout_seconds=30),
            "f1-build-request-deadline-0001",
        )
        clock = [now_ms()]
        deadline = clock[0] + 1000
        with stack.db.transaction() as conn:
            conn.execute(
                "UPDATE jobs SET deadline_ts=?, deadline_at=? WHERE job_id=?",
                (deadline, ms_to_iso(deadline), job["job_id"]),
            )
        stack.transport._wall_clock = lambda: clock[0]
        monkeypatch.setattr("anime_studio.worker.now_ms", lambda: clock[0])
        original_build = httpx.AsyncClient.build_request

        def build_request(self, method, url, **kwargs):
            request = original_build(self, method, url, **kwargs)
            if request.url.path == "/prompt":
                clock[0] = deadline
            return request

        monkeypatch.setattr(httpx.AsyncClient, "build_request", build_request)
        stack.worker.tick()
        _wait_for_upload(state)

        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.TIMED_OUT.value
        assert final["error"]["code"] == "DEADLINE_EXCEEDED"
        assert state.upload_requests == 1
        assert state.prompt_requests == 0
        events = stack.orchestrator.get_events(job["job_id"], after=0, limit=100)
        assert not any(event["type"] == "job_succeeded" for event in events["events"])
        timed_out = [
            event for event in events["events"] if event["type"] == "job_timed_out"
        ]
        assert timed_out
        assert timed_out[-1]["data"]["accepted"] is False
        assert timed_out[-1]["data"]["classification"] == "confirmed_not_sent"


def test_worker_build_request_crossing_monotonic_job_budget_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        job, _ = stack.orchestrator.create_job(
            make_request(prompt="F1 build mono deadline", timeout_seconds=30),
            "f1-build-mono-deadline-0001",
        )
        clock = [now_ms()]
        deadline = clock[0] + 200  # job-limited monotonic budget
        with stack.db.transaction() as conn:
            conn.execute(
                "UPDATE jobs SET deadline_ts=?, deadline_at=? WHERE job_id=?",
                (deadline, ms_to_iso(deadline), job["job_id"]),
            )
        stack.transport._wall_clock = lambda: clock[0]
        monkeypatch.setattr("anime_studio.worker.now_ms", lambda: clock[0])
        original_build = httpx.AsyncClient.build_request

        def build_request(self, method, url, **kwargs):
            request = original_build(self, method, url, **kwargs)
            if request.url.path == "/prompt":
                time.sleep(0.3)  # exceeds the 200 ms job-derived budget
            return request

        monkeypatch.setattr(httpx.AsyncClient, "build_request", build_request)
        stack.worker.tick()

        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.TIMED_OUT.value
        assert final["error"]["code"] == "DEADLINE_EXCEEDED"
        assert state.prompt_requests == 0
        events = stack.orchestrator.get_events(job["job_id"], after=0, limit=100)
        assert not any(event["type"] == "job_succeeded" for event in events["events"])


def test_recovery_submit_receives_the_stored_absolute_deadline(tmp_path: Path) -> None:
    db = Database(tmp_path / "recovery.sqlite3")
    db.init()
    adapter = _RecordingIdempotentAdapter()
    orchestrator = Orchestrator(
        db,
        adapter,  # type: ignore[arg-type]
        LocalArtifactStore(tmp_path / "artifacts"),
        lease_seconds=0.5,
    )
    job, _ = orchestrator.create_job(
        make_request(prompt="recovery deadline", timeout_seconds=30),
        "f1-recovery-deadline-0001",
    )
    _force_stale_dispatching(orchestrator, job["job_id"])
    worker = Worker(orchestrator, poll_interval=0.01, lease_seconds=0.5)

    worker.recover_stale_dispatches()

    assert adapter.submit_calls == 1
    assert adapter.deadline_ms == job["deadline_ts"]


def _recovery_case(
    tmp_path: Path,
    adapter: _ReconcileDeadlineAdapter,
    clock: list[int],
    *,
    prompt_id: str | None = None,
):
    db = Database(tmp_path / "recovery.sqlite3")
    db.init()
    orchestrator = Orchestrator(
        db,
        adapter,  # type: ignore[arg-type]
        LocalArtifactStore(tmp_path / "artifacts"),
        lease_seconds=0.5,
    )
    job, _ = orchestrator.create_job(
        make_request(prompt="recovery reconcile deadline", timeout_seconds=30),
        "recovery-reconcile-deadline-0001",
    )
    deadline = clock[0] + 1000
    with db.transaction() as conn:
        conn.execute(
            "UPDATE jobs SET deadline_ts=?, deadline_at=? WHERE job_id=?",
            (deadline, ms_to_iso(deadline), job["job_id"]),
        )
        if prompt_id is not None:
            conn.execute(
                "UPDATE jobs SET prompt_id=? WHERE job_id=?",
                (prompt_id, job["job_id"]),
            )
    _force_stale_dispatching(orchestrator, job["job_id"])
    worker = Worker(orchestrator, poll_interval=0.01, lease_seconds=0.5)
    return orchestrator, worker, job


def test_expired_recovery_reconcile_none_unsupported_is_timeout_not_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [now_ms()]
    adapter = _ReconcileDeadlineAdapter(
        clock, remote=None, supports_reconcile=False
    )
    orchestrator, worker, job = _recovery_case(tmp_path, adapter, clock)
    monkeypatch.setattr("anime_studio.worker.now_ms", lambda: clock[0])

    worker.recover_stale_dispatches()

    final = orchestrator.get_job(job["job_id"])
    assert final is not None
    assert final["status"] == JobStatus.TIMED_OUT.value
    assert final["error"]["code"] == "DEADLINE_EXCEEDED"
    assert adapter.submit_calls == 0
    assert adapter.cancel_calls == []
    events = orchestrator.get_events(job["job_id"], after=0, limit=100)
    assert not any(event["type"] == "dispatch_ambiguous" for event in events["events"])
    timed_out = [
        event for event in events["events"] if event["type"] == "job_timed_out"
    ]
    assert timed_out
    data = timed_out[-1]["data"]
    assert data["had_prompt_id"] is False
    assert data["cancel_attempted"] is False
    assert "classification" not in data
    assert "accepted" not in data


def test_expired_recovery_with_known_remote_keeps_identity_and_cancels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [now_ms()]
    remote = RemoteStatus(
        prompt_id="reconciled-1",
        correlation_key="",
        state="queued",
        progress=0,
    )
    adapter = _ReconcileDeadlineAdapter(clock, remote=remote)
    orchestrator, worker, job = _recovery_case(tmp_path, adapter, clock)
    monkeypatch.setattr("anime_studio.worker.now_ms", lambda: clock[0])

    worker.recover_stale_dispatches()

    final = orchestrator.get_job(job["job_id"])
    assert final is not None
    assert final["status"] == JobStatus.TIMED_OUT.value
    assert final["error"]["code"] == "DEADLINE_EXCEEDED"
    assert final["prompt_id"] == "reconciled-1"
    assert adapter.submit_calls == 0
    assert adapter.cancel_calls == ["reconciled-1"]
    events = orchestrator.get_events(job["job_id"], after=0, limit=100)
    timed_out = [
        event for event in events["events"] if event["type"] == "job_timed_out"
    ]
    assert timed_out
    data = timed_out[-1]["data"]
    assert data["had_prompt_id"] is True
    assert data["cancel_attempted"] is True
    assert data["remote_state"] == "cancelled"
    assert "classification" not in data
    assert "accepted" not in data


def test_expired_recovery_with_existing_prompt_id_cancels_known_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [now_ms()]
    adapter = _ReconcileDeadlineAdapter(clock, remote=None)
    orchestrator, worker, job = _recovery_case(
        tmp_path, adapter, clock, prompt_id="existing-1"
    )
    monkeypatch.setattr("anime_studio.worker.now_ms", lambda: clock[0])

    worker.recover_stale_dispatches()

    final = orchestrator.get_job(job["job_id"])
    assert final is not None
    assert final["status"] == JobStatus.TIMED_OUT.value
    assert final["error"]["code"] == "DEADLINE_EXCEEDED"
    assert final["prompt_id"] == "existing-1"
    assert adapter.submit_calls == 0
    assert adapter.cancel_calls == ["existing-1"]
    events = orchestrator.get_events(job["job_id"], after=0, limit=100)
    timed_out = [
        event for event in events["events"] if event["type"] == "job_timed_out"
    ]
    assert timed_out
    data = timed_out[-1]["data"]
    assert data["had_prompt_id"] is True
    assert data["cancel_attempted"] is True
    assert data["remote_state"] == "cancelled"
    assert "classification" not in data
    assert "accepted" not in data


# --------------------------------------------------------------------- F2
def _grace_transport(tmp_path: Path, base_url: str):
    transport = make_transport(
        tmp_path, base_url=base_url, prompt_lost_grace_seconds=5.0
    )
    clock = [0.0]
    transport._clock = lambda: clock[0]
    return transport, clock


def test_missing_running_missing_starts_a_new_grace_interval(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        transport, clock = _grace_transport(tmp_path, base_url)
        first = transport.get("prompt-gap")
        assert first is not None and first.state == "queued"

        clock[0] = 10.0
        state.queue_running = ["prompt-gap"]
        visible = transport.get("prompt-gap")
        assert visible is not None and visible.state == "running"

        clock[0] = 10.1
        state.queue_running = []
        second_gap = transport.get("prompt-gap")
        assert second_gap is not None and second_gap.state == "queued"

        clock[0] = 15.2  # 5.1 seconds of continuous absence from 10.1
        with pytest.raises(RemotePromptLost):
            transport.get("prompt-gap")


def test_missing_pending_missing_starts_a_new_grace_interval(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        transport, clock = _grace_transport(tmp_path, base_url)
        assert transport.get("prompt-pending").state == "queued"

        clock[0] = 10.0
        state.queue_pending = ["prompt-pending"]
        visible = transport.get("prompt-pending")
        assert visible is not None and visible.state == "queued"

        clock[0] = 10.1
        state.queue_pending = []
        second_gap = transport.get("prompt-pending")
        assert second_gap is not None and second_gap.state == "queued"

        clock[0] = 15.2
        with pytest.raises(RemotePromptLost):
            transport.get("prompt-pending")


def test_missing_history_presence_resets_the_grace_interval(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        transport, clock = _grace_transport(tmp_path, base_url)
        assert transport.get("prompt-history").state == "queued"

        clock[0] = 10.0
        state.history["prompt-history"] = {
            "status": {"status_str": "running", "completed": False, "messages": []},
            "outputs": {},
        }
        visible = transport.get("prompt-history")
        assert visible is not None and visible.state == "running"

        clock[0] = 10.1
        state.history.pop("prompt-history")
        second_gap = transport.get("prompt-history")
        assert second_gap is not None and second_gap.state == "queued"

        clock[0] = 15.2
        with pytest.raises(RemotePromptLost):
            transport.get("prompt-history")


def test_uninterrupted_missing_longer_than_grace_is_lost(tmp_path: Path) -> None:
    with run_stub() as (base_url, _state):
        transport, clock = _grace_transport(tmp_path, base_url)
        assert transport.get("prompt-gone").state == "queued"
        clock[0] = 5.1
        with pytest.raises(RemotePromptLost):
            transport.get("prompt-gone")


def test_malformed_queue_is_not_presence_or_absence_evidence(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        transport, clock = _grace_transport(tmp_path, base_url)
        assert transport.get("prompt-malformed").state == "queued"

        clock[0] = 10.0
        state.queue_mode = "missing_keys"
        with pytest.raises(RemoteProtocolError):
            transport.get("prompt-malformed")

        # The malformed response neither reset the timer nor raised LOST by
        # itself. The earlier confirmed absence is still older than grace.
        state.queue_mode = "normal"
        clock[0] = 10.1
        with pytest.raises(RemotePromptLost):
            transport.get("prompt-malformed")


def test_transient_error_does_not_reset_the_absence_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, _state):
        transport, clock = _grace_transport(tmp_path, base_url)
        assert transport.get("prompt-transient").state == "queued"

        original_history = transport._history

        def transient_history(prompt_id: str):
            raise RemoteTransientError(
                "synthetic transient history failure",
                code="REMOTE_TRANSIENT_ERROR",
            )

        monkeypatch.setattr(transport, "_history", transient_history)
        clock[0] = 10.0
        with pytest.raises(RemoteTransientError):
            transport.get("prompt-transient")

        monkeypatch.setattr(transport, "_history", original_history)
        clock[0] = 15.1
        with pytest.raises(RemotePromptLost):
            transport.get("prompt-transient")


def test_worker_lost_prompt_is_terminal_and_not_resubmitted(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(
            tmp_path, base_url=base_url, prompt_lost_grace_seconds=5.0
        )
        clock = [0.0]
        stack.transport._clock = lambda: clock[0]
        job, _ = stack.orchestrator.create_job(
            make_request(prompt="F2 worker grace", timeout_seconds=30),
            "f2-worker-grace-0001",
        )
        stack.worker.tick()
        assert state.prompt_requests == 1

        state.history_missing = True
        state.queue_pending = []
        stack.worker.tick()  # first observed absence; grace starts
        assert stack.orchestrator.get_job(job["job_id"])["status"] in {
            JobStatus.SUBMITTED.value,
            JobStatus.RUNNING.value,
        }

        clock[0] = 10.0
        state.queue_running = [state.prompt_id]
        stack.worker.tick()  # confirmed running presence clears the timer
        assert stack.orchestrator.get_job(job["job_id"])["status"] == JobStatus.RUNNING.value

        clock[0] = 10.1
        state.queue_running = []
        stack.worker.tick()  # new contiguous absence; must not be lost yet
        assert stack.orchestrator.get_job(job["job_id"])["status"] in {
            JobStatus.SUBMITTED.value,
            JobStatus.RUNNING.value,
        }

        clock[0] = 15.2
        stack.worker.tick()
        final = stack.orchestrator.get_job(job["job_id"])
        assert final is not None
        assert final["status"] == JobStatus.FAILED.value
        assert final["error"]["code"] == "REMOTE_PROMPT_LOST"
        assert state.prompt_requests == 1

        stack.worker.tick()
        assert state.prompt_requests == 1
