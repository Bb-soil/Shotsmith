from __future__ import annotations

import calendar
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.p1.stop_pod as stop_pod_module
from scripts.p1.runpodctl_bridge import (
    ProviderAuthError,
    ProviderContractUnverified,
    ProviderError,
    ProviderPod,
    ProviderTransientError,
)
from scripts.p1.stop_pod import (
    MAX_SESSION_SECONDS,
    StopController,
    StopStateError,
    initialize_state,
    load_state,
    main as stop_main,
    save_state,
)


CREATED = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
POD_ID = "pod-1"


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.monotonic_value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def mono(self) -> float:
        return self.monotonic_value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)
        self.monotonic_value += seconds


class FakeProvider:
    def __init__(
        self,
        *,
        pod: ProviderPod | None = None,
        get_results: list[object] | None = None,
        stop_error: Exception | None = None,
        get_error: Exception | None = None,
        on_stop: object | None = None,
    ) -> None:
        if get_results is not None:
            self.get_results = list(get_results)
        elif pod is not None:
            self.get_results = [pod]
        elif get_error is not None:
            self.get_results = [get_error]
        else:
            self.get_results = [None]
        self.stop_error = stop_error
        self.on_stop = on_stop
        self.ops: list[tuple[str, str]] = []
        self.get_timeouts: list[float | None] = []
        self.get_count = 0
        self.stop_count = 0

    def get(
        self, pod_id: str, *, timeout_seconds: float | None = None
    ) -> ProviderPod | None:
        self.ops.append(("get", pod_id))
        self.get_timeouts.append(timeout_seconds)
        index = min(self.get_count, len(self.get_results) - 1)
        self.get_count += 1
        result = self.get_results[index]
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[return-value]

    def stop(
        self, pod_id: str, *, timeout_seconds: float | None = None
    ) -> None:
        self.ops.append(("stop", pod_id))
        self.stop_count += 1
        if self.on_stop is not None:
            self.on_stop()
        if self.stop_error is not None:
            raise self.stop_error


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / "stop-state.json"


def _controller(
    state_path: Path,
    provider: FakeProvider,
    clock: FakeClock | None = None,
    *,
    readback_timeout_seconds: float = 120.0,
    readback_poll_interval_seconds: float = 5.0,
) -> StopController:
    return StopController(
        state_path,
        pod_id=POD_ID,
        provider=provider,
        clock=clock,
        monotonic=clock.mono if clock is not None else None,
        readback_timeout_seconds=readback_timeout_seconds,
        readback_poll_interval_seconds=readback_poll_interval_seconds,
    )


def _initialized(
    tmp_path: Path,
    provider: FakeProvider,
    clock: FakeClock | None = None,
    *,
    readback_timeout_seconds: float = 120.0,
    readback_poll_interval_seconds: float = 5.0,
) -> tuple[StopController, object, Path]:
    state_path = _state_path(tmp_path)
    state = initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    if clock is None:
        clock = FakeClock(CREATED)
    return (
        _controller(
            state_path,
            provider,
            clock,
            readback_timeout_seconds=readback_timeout_seconds,
            readback_poll_interval_seconds=readback_poll_interval_seconds,
        ),
        state,
        state_path,
    )


def _running(created_at: datetime | None = CREATED) -> ProviderPod:
    return ProviderPod(POD_ID, "RUNNING", created_at)


def _stopped(created_at: datetime | None = CREATED) -> ProviderPod:
    return ProviderPod(POD_ID, "STOPPED", created_at)


def test_deadline_is_at_most_sixty_minutes_and_restart_cannot_extend(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    state = initialize_state(
        state_path,
        pod_id=POD_ID,
        created_at=CREATED,
        now=CREATED,
    )
    assert state.deadline_at == CREATED + timedelta(minutes=60)

    again = initialize_state(
        state_path,
        pod_id=POD_ID,
        created_at=CREATED,
        now=CREATED + timedelta(minutes=5),
    )
    assert again.deadline_at == state.deadline_at

    with pytest.raises(StopStateError):
        initialize_state(
            state_path,
            pod_id=POD_ID,
            created_at=CREATED,
            deadline_at=CREATED + timedelta(minutes=30),
            now=CREATED,
        )
    with pytest.raises(StopStateError):
        initialize_state(
            state_path,
            pod_id="pod-2",
            created_at=CREATED,
            now=CREATED,
        )


def test_deadline_over_sixty_minutes_is_rejected(tmp_path) -> None:
    with pytest.raises(StopStateError):
        initialize_state(
            _state_path(tmp_path),
            pod_id=POD_ID,
            created_at=CREATED,
            deadline_at=CREATED + timedelta(minutes=61),
            now=CREATED,
        )
    assert MAX_SESSION_SECONDS == 60 * 60


def test_dry_run_before_deadline_performs_no_provider_operation(tmp_path) -> None:
    provider = FakeProvider(pod=_running())
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=10))
    assert outcome.status == "ARMED"
    assert provider.ops == []


def test_overdue_stop_confirms_running_then_stopped(tmp_path) -> None:
    provider = FakeProvider(get_results=[_running(), _stopped()])
    controller, state, state_path = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "STOPPED"
    assert provider.ops == [("get", POD_ID), ("stop", POD_ID), ("get", POD_ID)]
    assert provider.stop_count == 1
    assert load_state(state_path).status == "STOPPED"
    assert load_state(state_path).deadline_at == state.deadline_at


def test_explicit_early_stop_is_allowed(tmp_path) -> None:
    provider = FakeProvider(get_results=[_running(), _stopped()])
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(
        now=CREATED + timedelta(minutes=1),
        explicit=True,
    )
    assert outcome.status == "STOPPED"
    assert provider.ops == [("get", POD_ID), ("stop", POD_ID), ("get", POD_ID)]


def test_stop_failure_is_live_blocked_not_success(tmp_path) -> None:
    provider = FakeProvider(
        get_results=[_running()],
        stop_error=ProviderError("stop failed"),
    )
    controller, _, state_path = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "LIVE_BLOCKED"
    assert provider.ops == [("get", POD_ID), ("stop", POD_ID)]
    assert provider.stop_count == 1
    persisted = load_state(state_path)
    assert persisted.status == "LIVE_BLOCKED"
    assert persisted.readback_deadline_at is not None


def test_readback_failure_is_live_blocked(tmp_path) -> None:
    provider = FakeProvider(
        get_results=[_running(), ProviderError("readback failed")]
    )
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "LIVE_BLOCKED"
    assert provider.ops == [("get", POD_ID), ("stop", POD_ID), ("get", POD_ID)]


def test_still_running_readback_is_bounded_and_live_blocked(tmp_path) -> None:
    clock = FakeClock(CREATED)
    provider = FakeProvider(get_results=[_running()])
    controller, state, state_path = _initialized(
        tmp_path,
        provider,
        clock,
        readback_timeout_seconds=10,
        readback_poll_interval_seconds=5,
    )
    outcome = controller.evaluate(
        now=CREATED + timedelta(minutes=61), sleeper=clock.sleep
    )

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.stop_count == 1
    assert provider.get_count >= 3
    assert "last status: RUNNING" in outcome.message
    persisted = load_state(state_path)
    assert persisted.status == "LIVE_BLOCKED"
    assert persisted.deadline_at == state.deadline_at
    assert persisted.readback_deadline_at is not None
    assert clock.sleeps and all(second <= 5 for second in clock.sleeps)


@pytest.mark.parametrize(
    ("pod", "expected_stop"),
    [
        (None, False),
        (ProviderPod("other-pod", "STOPPED", CREATED), False),
        (ProviderPod(POD_ID, "TERMINATED", CREATED), False),
        (ProviderPod(POD_ID, "ERROR", CREATED), False),
    ],
)
def test_pre_read_missing_wrong_or_ineligible_never_issues_stop(
    tmp_path, pod, expected_stop
) -> None:
    provider = FakeProvider(get_results=[pod])
    controller, _, state_path = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.ops == [("get", POD_ID)]
    assert provider.stop_count == 0
    assert expected_stop is False
    assert load_state(state_path).status == "LIVE_BLOCKED"


def test_pre_read_missing_created_at_blocks_without_stop(tmp_path) -> None:
    provider = FakeProvider(get_results=[_stopped(created_at=None)])
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "LIVE_BLOCKED"
    assert provider.ops == [("get", POD_ID)]
    assert provider.stop_count == 0


def test_pre_read_created_at_mismatch_blocks_without_stop(tmp_path) -> None:
    provider = FakeProvider(
        get_results=[_stopped(created_at=CREATED + timedelta(seconds=1))]
    )
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "LIVE_BLOCKED"
    assert provider.ops == [("get", POD_ID)]
    assert provider.stop_count == 0


def test_pre_read_timezone_equivalent_created_at_is_accepted(tmp_path) -> None:
    offset = timezone(timedelta(hours=2))
    provider = FakeProvider(
        get_results=[
            _running(created_at=CREATED.astimezone(offset)),
            _stopped(created_at=CREATED.astimezone(offset)),
        ]
    )
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "STOPPED"
    assert provider.stop_count == 1


def test_pre_read_stopped_skips_redundant_stop(tmp_path) -> None:
    provider = FakeProvider(get_results=[_stopped()])
    controller, _, state_path = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "STOPPED"
    assert provider.ops == [("get", POD_ID)]
    assert provider.stop_count == 0
    assert load_state(state_path).status == "STOPPED"


def test_confirmed_stopped_is_not_rechecked_on_restart(tmp_path) -> None:
    provider = FakeProvider(get_results=[_stopped()])
    controller, _, state_path = _initialized(tmp_path, provider)
    assert controller.evaluate(now=CREATED + timedelta(minutes=61)).status == "STOPPED"
    restarted = _controller(state_path, provider)
    before = list(provider.ops)
    assert restarted.evaluate(explicit=True).status == "STOPPED"
    assert provider.ops == before


def test_dry_run_view_rejects_mismatched_created_or_deadline(tmp_path) -> None:
    provider = FakeProvider()
    controller, _, _ = _initialized(tmp_path, provider)
    with pytest.raises(StopStateError):
        controller.dry_run_view(created_at=CREATED + timedelta(seconds=1))
    with pytest.raises(StopStateError):
        controller.dry_run_view(deadline_at=CREATED + timedelta(minutes=30))


def test_state_deadline_tampering_is_rejected(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["deadline_at"] = (CREATED + timedelta(hours=2)).isoformat().replace(
        "+00:00", "Z"
    )
    state_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StopStateError):
        load_state(state_path)


def test_readback_deadline_tampering_is_rejected(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["readback_deadline_at"] = (
        CREATED + timedelta(hours=2)
    ).isoformat().replace("+00:00", "Z")
    state_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StopStateError):
        load_state(state_path)


def test_stop_intent_is_persisted_before_external_stop(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    observed: list[str] = []
    provider = FakeProvider(
        get_results=[_running(), _stopped()],
        on_stop=lambda: observed.append(load_state(state_path).status),
    )
    controller = _controller(state_path, provider, FakeClock(CREATED))
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))

    assert outcome.status == "STOPPED"
    assert observed == ["STOP_REQUESTED"]
    persisted = load_state(state_path)
    assert persisted.status == "STOPPED"
    assert persisted.readback_deadline_at is not None


def test_restart_with_stop_intent_is_readback_only_and_never_stops(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(minutes=61))
    save_state(state_path, requested)
    provider = FakeProvider(get_results=[_running()])
    clock = FakeClock(CREATED + timedelta(minutes=60))
    controller = _controller(state_path, provider, clock)

    outcome = controller.evaluate(
        now=CREATED + timedelta(minutes=60), sleeper=clock.sleep
    )

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.ops
    assert all(operation == "get" for operation, _ in provider.ops)
    assert provider.stop_count == 0


def test_restart_with_stop_intent_can_confirm_stopped_without_stop(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(minutes=61))
    save_state(state_path, requested)
    provider = FakeProvider(get_results=[_stopped()])
    controller = _controller(state_path, provider, FakeClock(CREATED + timedelta(minutes=60)))

    outcome = controller.evaluate(now=CREATED + timedelta(minutes=60))

    assert outcome.status == "STOPPED"
    assert provider.ops == [("get", POD_ID)]
    assert provider.stop_count == 0


def test_readback_deadline_is_not_reset_by_restart_config(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    original_deadline = CREATED + timedelta(minutes=60, seconds=5)
    requested = _requested(state, original_deadline)
    save_state(state_path, requested)
    provider = FakeProvider(get_results=[_stopped()])
    controller = _controller(
        state_path,
        provider,
        FakeClock(CREATED + timedelta(minutes=60)),
        readback_timeout_seconds=300,
    )

    assert controller.evaluate(now=CREATED + timedelta(minutes=60)).status == "STOPPED"
    assert load_state(state_path).readback_deadline_at == original_deadline


def test_live_blocked_restart_and_explicit_do_not_retry(tmp_path) -> None:
    provider = FakeProvider(
        get_results=[_running()],
        stop_error=ProviderError("early failure"),
    )
    controller, _, state_path = _initialized(tmp_path, provider)
    assert (
        controller.evaluate(
            now=CREATED + timedelta(minutes=1),
            explicit=True,
        ).status
        == "LIVE_BLOCKED"
    )
    before = list(provider.ops)
    assert controller.evaluate(now=CREATED + timedelta(minutes=2)).status == "LIVE_BLOCKED"
    assert controller.evaluate(explicit=True).status == "LIVE_BLOCKED"
    assert provider.ops == before
    assert load_state(state_path).status == "LIVE_BLOCKED"


def test_pending_running_unknown_then_stopped_polls_to_success(tmp_path) -> None:
    clock = FakeClock(CREATED)
    provider = FakeProvider(
        get_results=[
            _running(),
            ProviderPod(POD_ID, "INITIALIZING", CREATED, "awaiting_container"),
            _running(),
            ProviderPod(POD_ID, "UNKNOWN", CREATED, "runtime_unavailable"),
            _stopped(),
        ]
    )
    controller, state, state_path = _initialized(tmp_path, provider, clock)
    outcome = controller.evaluate(
        now=CREATED + timedelta(minutes=61), sleeper=clock.sleep
    )

    assert outcome.status == "STOPPED"
    assert provider.stop_count == 1
    assert provider.get_count == 5
    assert load_state(state_path).deadline_at == state.deadline_at


def test_transient_read_errors_are_retried_within_bound(tmp_path) -> None:
    clock = FakeClock(CREATED)
    provider = FakeProvider(
        get_results=[
            _running(),
            ProviderTransientError("network_error"),
            _stopped(),
        ]
    )
    controller, _, _ = _initialized(tmp_path, provider, clock)
    outcome = controller.evaluate(
        now=CREATED + timedelta(minutes=61), sleeper=clock.sleep
    )
    assert outcome.status == "STOPPED"
    assert provider.stop_count == 1
    assert provider.get_count == 3


def test_terminal_read_error_fails_closed_without_retry(tmp_path) -> None:
    provider = FakeProvider(
        get_results=[_running(), ProviderAuthError("unauthorized")]
    )
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "LIVE_BLOCKED"
    assert provider.stop_count == 1
    assert provider.get_count == 2


@pytest.mark.parametrize(
    "readback",
    [
        None,
        ProviderPod(POD_ID, "TERMINATED", CREATED),
        ProviderPod(POD_ID, "ERROR", CREATED),
        ProviderError("readback failed"),
    ],
)
def test_missing_terminated_error_or_failed_readback_blocks(
    tmp_path, readback
) -> None:
    provider = FakeProvider(get_results=[_running(), readback])
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "LIVE_BLOCKED"
    assert provider.stop_count == 1
    assert provider.get_count == 2


def test_identity_mismatch_during_readback_blocks(tmp_path) -> None:
    provider = FakeProvider(
        get_results=[
            _running(),
            _stopped(created_at=CREATED + timedelta(seconds=2)),
        ]
    )
    controller, _, _ = _initialized(tmp_path, provider)
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))
    assert outcome.status == "LIVE_BLOCKED"
    assert "createdAt" in outcome.message
    assert provider.stop_count == 1


def test_readback_command_timeout_follows_remaining_budget(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(minutes=60, seconds=10))
    save_state(state_path, requested)
    clock = FakeClock(CREATED + timedelta(minutes=60))
    provider = FakeProvider(get_results=[_running()])
    controller = _controller(
        state_path,
        provider,
        clock,
        readback_timeout_seconds=10,
        readback_poll_interval_seconds=5,
    )

    outcome = controller.evaluate(
        now=CREATED + timedelta(minutes=60), sleeper=clock.sleep
    )

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.stop_count == 0
    timeouts = [float(value) for value in provider.get_timeouts if value is not None]
    assert timeouts, "readback must have been attempted"
    assert all(0 < value <= 10 for value in timeouts)
    assert timeouts[0] <= 10
    assert timeouts[-1] <= timeouts[0]


def test_watch_waits_until_persisted_deadline_with_fake_clock(tmp_path) -> None:
    provider = FakeProvider(get_results=[_running(), _stopped()])
    clock = FakeClock(CREATED + timedelta(minutes=1))
    controller, state, state_path = _initialized(tmp_path, provider, clock)
    outcome = controller.watch(poll_interval_seconds=600, sleeper=clock.sleep)
    assert outcome.status == "STOPPED"
    assert provider.ops == [("get", POD_ID), ("stop", POD_ID), ("get", POD_ID)]
    assert all(second <= 600 for second in clock.sleeps)
    assert sum(clock.sleeps) == 3540
    assert load_state(state_path).deadline_at == state.deadline_at


def test_watch_restart_overdue_is_immediate_and_keeps_deadline(tmp_path) -> None:
    provider = FakeProvider(get_results=[_running(), _stopped()])
    state_path = _state_path(tmp_path)
    state = initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    clock = FakeClock(state.deadline_at + timedelta(seconds=1))
    controller = _controller(state_path, provider, clock)
    outcome = controller.watch(poll_interval_seconds=600, sleeper=clock.sleep)
    assert outcome.status == "STOPPED"
    assert clock.sleeps == []
    assert provider.ops == [("get", POD_ID), ("stop", POD_ID), ("get", POD_ID)]
    assert load_state(state_path).deadline_at == state.deadline_at


def test_watch_already_stopped_performs_no_actions(tmp_path) -> None:
    provider = FakeProvider(get_results=[_stopped()])
    controller, _, _ = _initialized(tmp_path, provider)
    assert controller.evaluate(now=CREATED + timedelta(minutes=61)).status == "STOPPED"
    provider.ops.clear()
    clock = FakeClock(CREATED + timedelta(minutes=1))
    restarted = _controller(_state_path(tmp_path), provider, clock)
    assert restarted.watch(poll_interval_seconds=600, sleeper=clock.sleep).status == "STOPPED"
    assert clock.sleeps == []
    assert provider.ops == []


def test_concurrent_initialization_keeps_one_identity(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    barrier = threading.Barrier(2)

    def attempt(pod_id: str):
        barrier.wait()
        try:
            return initialize_state(
                state_path,
                pod_id=pod_id,
                created_at=CREATED,
                now=CREATED,
            )
        except StopStateError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (POD_ID, "pod-2")))

    states = [result for result in results if not isinstance(result, Exception)]
    errors = [result for result in results if isinstance(result, StopStateError)]
    assert len(states) == 1
    assert len(errors) == 1
    assert load_state(state_path).pod_id == states[0].pod_id


def test_concurrent_evaluate_cannot_regress_confirmed_stopped(tmp_path) -> None:
    class BlockingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__(get_results=[_running(), _stopped()])
            self.stop_event = threading.Event()
            self.release = threading.Event()

        def stop(self, pod_id: str, *, timeout_seconds: float | None = None) -> None:
            self.ops.append(("stop", pod_id))
            self.stop_count += 1
            self.stop_event.set()
            self.release.wait(timeout=2)

    provider = BlockingProvider()
    state_path = _state_path(tmp_path)
    state = initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    first = _controller(state_path, provider, FakeClock(state.deadline_at))
    second = _controller(state_path, provider, FakeClock(state.deadline_at))
    second_started = threading.Event()
    second_done = threading.Event()

    def run_second():
        second_started.set()
        try:
            return second.evaluate(now=state.deadline_at + timedelta(seconds=1))
        finally:
            second_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            first.evaluate,
            now=state.deadline_at + timedelta(seconds=1),
        )
        assert provider.stop_event.wait(timeout=2)
        second_future = executor.submit(run_second)
        assert second_started.wait(timeout=2)
        time.sleep(0.05)
        assert not second_done.is_set()
        provider.release.set()
        assert first_future.result(timeout=2).status == "STOPPED"
        assert second_future.result(timeout=2).status == "STOPPED"

    assert provider.ops == [("get", POD_ID), ("stop", POD_ID), ("get", POD_ID)]
    assert load_state(state_path).status == "STOPPED"


def test_cli_execute_unverified_backend_blocks_before_state_write(
    tmp_path, capsys, monkeypatch
) -> None:
    class MissingBackend:
        def __init__(self, executable: str = "runpodctl") -> None:
            pass

        def verify_contract(self) -> str:
            raise ProviderContractUnverified("runpodctl executable was not found")

    monkeypatch.setattr(stop_pod_module, "RunpodctlProvider", MissingBackend)
    state_path = _state_path(tmp_path)
    code = stop_main(
        [
            "--state",
            str(state_path),
            "--pod-id",
            "pod-test01",
            "--created-at",
            "2026-10-21T12:00:00Z",
            "--execute",
        ]
    )
    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "LIVE_BLOCKED"
    assert payload["state"] is None
    assert not state_path.exists()


def test_cli_dry_run_does_not_construct_provider_or_write_state(
    tmp_path, capsys, monkeypatch
) -> None:
    class FailIfConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run must not construct a provider")

    monkeypatch.setattr(stop_pod_module, "RunpodctlProvider", FailIfConstructed)
    state_path = _state_path(tmp_path)
    code = stop_main(
        [
            "--state",
            str(state_path),
            "--pod-id",
            "pod-test01",
            "--created-at",
            "2026-10-21T12:00:00Z",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ARMED"
    assert payload["readback_deadline_at"] is None
    assert not state_path.exists()


def test_cli_once_uses_full_bounded_readback_not_single_get(tmp_path, capsys, monkeypatch) -> None:
    class CliProvider(FakeProvider):
        def __init__(self, executable: str = "runpodctl") -> None:
            self.executable = executable
            super().__init__(get_results=[_running(), _stopped()])

        def verify_contract(self) -> str:
            self.ops.append(("verify", self.executable))
            return "v3.0.1"

    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    monkeypatch.setattr(stop_pod_module, "RunpodctlProvider", CliProvider)

    code = stop_main(
        [
            "--state",
            str(state_path),
            "--pod-id",
            POD_ID,
            "--execute",
            "--once",
            "--explicit-stop",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "STOPPED"
    assert load_state(state_path).status == "STOPPED"


def test_cli_version_failure_blocks_before_state(tmp_path, capsys, monkeypatch) -> None:
    class UnsupportedProvider:
        def __init__(self, executable: str = "runpodctl") -> None:
            pass

        def verify_contract(self) -> str:
            raise ProviderContractUnverified("synthetic version failure")

    monkeypatch.setattr(stop_pod_module, "RunpodctlProvider", UnsupportedProvider)
    state_path = _state_path(tmp_path)
    code = stop_main(
        [
            "--state",
            str(state_path),
            "--pod-id",
            POD_ID,
            "--created-at",
            "2026-10-21T12:00:00Z",
            "--execute",
        ]
    )
    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "LIVE_BLOCKED"
    assert payload["state"] is None
    assert not state_path.exists()


def test_concurrent_initialization_with_different_times_keeps_first(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    barrier = threading.Barrier(2)

    def attempt(created_at: datetime):
        barrier.wait()
        try:
            return initialize_state(
                state_path,
                pod_id=POD_ID,
                created_at=created_at,
                now=CREATED,
            )
        except StopStateError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                attempt,
                (CREATED, CREATED + timedelta(minutes=1)),
            )
        )
    states = [result for result in results if not isinstance(result, Exception)]
    errors = [result for result in results if isinstance(result, StopStateError)]
    assert len(states) == 1
    assert len(errors) == 1
    persisted = load_state(state_path)
    assert persisted.created_at == states[0].created_at
    assert persisted.deadline_at == states[0].created_at + timedelta(minutes=60)


def _ns(second: int = 0, fraction: int = 0) -> int:
    base = calendar.timegm(CREATED.utctimetuple()) * 1_000_000_000
    return base + second * 1_000_000_000 + fraction


def test_fractional_created_at_ns_mismatch_blocks_before_stop(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(
        state_path,
        pod_id=POD_ID,
        created_at="2026-10-21T12:00:00.100Z",
        now=CREATED,
    )
    provider = FakeProvider(
        get_results=[
            ProviderPod(
                POD_ID,
                "RUNNING",
                CREATED,
                created_at_ns=_ns(fraction=900_000_000),
            )
        ]
    )
    controller = _controller(state_path, provider, FakeClock(CREATED))
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.ops == [("get", POD_ID)]
    assert provider.stop_count == 0


def test_equivalent_fractional_created_at_timezones_can_confirm_stopped(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(
        state_path,
        pod_id=POD_ID,
        created_at="2026-10-21T12:00:00.100Z",
        now=CREATED,
    )
    offset = timezone(timedelta(hours=2))
    provider = FakeProvider(
        get_results=[
            ProviderPod(
                POD_ID,
                "STOPPED",
                CREATED.astimezone(offset),
                created_at_ns=_ns(fraction=100_000_000),
            )
        ]
    )
    controller = _controller(state_path, provider, FakeClock(CREATED))
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))

    assert outcome.status == "STOPPED"
    assert provider.stop_count == 0


def test_one_nanosecond_created_at_mismatch_blocks_before_stop(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(
        state_path,
        pod_id=POD_ID,
        created_at="2026-10-21T12:00:00.100000000Z",
        now=CREATED,
    )
    provider = FakeProvider(
        get_results=[
            ProviderPod(
                POD_ID,
                "RUNNING",
                CREATED,
                created_at_ns=_ns(fraction=100_000_001),
            )
        ]
    )
    controller = _controller(state_path, provider, FakeClock(CREATED))
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.stop_count == 0


def test_legacy_seconds_identity_rejects_fractional_provider(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(
        state_path,
        pod_id=POD_ID,
        created_at="2026-10-21T12:00:00Z",
        now=CREATED,
    )
    state = load_state(state_path)
    assert state.created_at_ns == _ns()
    provider = FakeProvider(
        get_results=[
            ProviderPod(
                POD_ID,
                "RUNNING",
                CREATED,
                created_at_ns=_ns(fraction=1),
            )
        ]
    )
    controller = _controller(state_path, provider, FakeClock(CREATED))
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.stop_count == 0


def test_expired_readback_bound_does_not_issue_get(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(milliseconds=1))
    save_state(state_path, requested)
    provider = FakeProvider(get_results=[_stopped()])
    controller = _controller(
        state_path, provider, FakeClock(CREATED + timedelta(milliseconds=500))
    )
    outcome = controller.evaluate(now=CREATED)

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.get_count == 0
    assert provider.stop_count == 0


def test_tiny_positive_readback_budget_never_enlarged(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(milliseconds=1))
    save_state(state_path, requested)
    provider = FakeProvider(get_results=[_stopped()])
    controller = _controller(state_path, provider, FakeClock(CREATED))
    outcome = controller.evaluate(now=CREATED)

    assert outcome.status == "STOPPED"
    timeout = provider.get_timeouts[0]
    assert timeout is not None and 0 < timeout <= 0.001


def test_response_crossing_readback_budget_is_not_success(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(seconds=1))
    save_state(state_path, requested)
    clock = FakeClock(CREATED)

    class CrossingProvider(FakeProvider):
        def get(self, pod_id: str, *, timeout_seconds: float | None = None):
            pod = super().get(pod_id, timeout_seconds=timeout_seconds)
            clock.now += timedelta(seconds=2)
            clock.monotonic_value += 2
            return pod

    provider = CrossingProvider(get_results=[_stopped()])
    controller = _controller(state_path, provider, clock)
    outcome = controller.evaluate(now=CREATED)

    assert outcome.status == "LIVE_BLOCKED"
    assert load_state(state_path).status == "LIVE_BLOCKED"
    assert provider.get_count == 1
    assert provider.stop_count == 0


def test_unknown_error_code_not_persisted_in_state_or_cli_output(
    tmp_path, capsys, monkeypatch
) -> None:
    from types import SimpleNamespace

    from scripts.p1.runpodctl_bridge import RunpodctlProvider

    secret_code = "syntheticsecretcode"

    def runner(args, **kwargs):
        if "--version" in args:
            return SimpleNamespace(
                returncode=0, stdout="runpodctl v3.0.1\n", stderr=""
            )
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=json.dumps(
                {
                    "error": "synthetic-error-must-not-leak",
                    "code": secret_code,
                    "status": 401,
                }
            ),
        )

    class CliProvider:
        def __init__(self, executable: str = "runpodctl") -> None:
            self._provider = RunpodctlProvider(runner=runner)

        def verify_contract(self) -> str:
            return self._provider.verify_contract()

        def get(self, pod_id: str, *, timeout_seconds: float | None = None):
            return self._provider.get(pod_id, timeout_seconds=timeout_seconds)

        def stop(self, pod_id: str, *, timeout_seconds: float | None = None) -> None:
            return self._provider.stop(pod_id, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(stop_pod_module, "RunpodctlProvider", CliProvider)
    state_path = _state_path(tmp_path)
    code = stop_main(
        [
            "--state",
            str(state_path),
            "--pod-id",
            POD_ID,
            "--created-at",
            "2026-10-21T12:00:00Z",
            "--execute",
            "--explicit-stop",
        ]
    )

    captured = capsys.readouterr()
    assert code == 3
    assert secret_code not in captured.out
    assert secret_code not in captured.err
    assert "synthetic-error-must-not-leak" not in captured.out
    persisted = load_state(state_path)
    assert persisted.status == "LIVE_BLOCKED"
    assert secret_code not in (persisted.last_error or "")
    assert "synthetic-error-must-not-leak" not in (persisted.last_error or "")


def test_restart_fractional_expired_readback_does_not_get_or_stop(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(milliseconds=1))
    save_state(state_path, requested)
    provider = FakeProvider(get_results=[_stopped()])
    clock = FakeClock(CREATED + timedelta(milliseconds=500))
    controller = _controller(state_path, provider, clock)
    outcome = controller.evaluate(now=CREATED)

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.get_count == 0
    assert provider.stop_count == 0


def test_tiny_fractional_remaining_budget_is_not_enlarged(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(milliseconds=1))
    save_state(state_path, requested)
    provider = FakeProvider(get_results=[_stopped()])
    clock = FakeClock(CREATED + timedelta(microseconds=500))
    controller = _controller(state_path, provider, clock)
    outcome = controller.evaluate(now=CREATED)

    assert outcome.status == "STOPPED"
    timeout = provider.get_timeouts[0]
    assert timeout is not None
    assert 0 < timeout <= 0.0005


def test_late_stopped_crossing_wall_fractional_end_is_rejected(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = _requested(state, CREATED + timedelta(milliseconds=1))
    save_state(state_path, requested)
    clock = FakeClock(CREATED + timedelta(microseconds=500))

    class LateStoppedProvider(FakeProvider):
        def get(self, pod_id: str, *, timeout_seconds: float | None = None):
            pod = super().get(pod_id, timeout_seconds=timeout_seconds)
            clock.now += timedelta(milliseconds=1)
            # monotonic clock intentionally frozen: wall crossing must reject.
            return pod

    provider = LateStoppedProvider(get_results=[_stopped()])
    controller = _controller(state_path, provider, clock)
    outcome = controller.evaluate(now=CREATED)

    assert outcome.status == "LIVE_BLOCKED"
    assert load_state(state_path).status == "LIVE_BLOCKED"
    assert provider.get_count == 1
    assert provider.stop_count == 0

def _requested(state, deadline, *, started: datetime | None = None):
    if started is None:
        started = max(state.created_at, deadline - timedelta(seconds=10))
    return state.__class__(
        pod_id=state.pod_id,
        created_at=state.created_at,
        deadline_at=state.deadline_at,
        status="STOP_REQUESTED",
        updated_at=state.created_at + timedelta(minutes=60),
        last_error=None,
        readback_started_at=started,
        readback_deadline_at=deadline,
        created_at_ns=state.created_at_ns,
    )


def test_readback_anchor_is_precise_and_exactly_max_bound(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    clock = FakeClock(CREATED + timedelta(microseconds=500_000))
    captured = []

    provider = FakeProvider(
        get_results=[_running(), _stopped()],
        on_stop=lambda: captured.append(load_state(state_path)),
    )
    controller = _controller(
        state_path,
        provider,
        clock,
        readback_timeout_seconds=600,
    )
    outcome = controller.evaluate(now=CREATED + timedelta(minutes=61))

    assert outcome.status == "STOPPED"
    assert captured
    anchored = captured[0]
    assert anchored.status == "STOP_REQUESTED"
    assert anchored.readback_started_at == CREATED + timedelta(microseconds=500_000)
    assert (
        anchored.readback_deadline_at - anchored.readback_started_at
    ).total_seconds() == 600


def test_readback_interval_over_max_is_rejected_on_load(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["status"] = "STOP_REQUESTED"
    raw["updated_at"] = "2026-10-21T12:00:00Z"
    raw["readback_started_at"] = "2026-10-21T12:00:00Z"
    raw["readback_deadline_at"] = "2026-10-21T12:10:01Z"
    state_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(StopStateError):
        load_state(state_path)


def test_legacy_stop_intent_without_anchor_fails_closed_manual(tmp_path) -> None:
    state_path = _state_path(tmp_path)
    initialize_state(state_path, pod_id=POD_ID, created_at=CREATED, now=CREATED)
    state = load_state(state_path)
    requested = state.__class__(
        pod_id=state.pod_id,
        created_at=state.created_at,
        deadline_at=state.deadline_at,
        status="STOP_REQUESTED",
        updated_at=CREATED,
        last_error=None,
        readback_started_at=None,
        readback_deadline_at=CREATED + timedelta(milliseconds=1),
        created_at_ns=state.created_at_ns,
    )
    save_state(state_path, requested)
    provider = FakeProvider(get_results=[_stopped()])
    controller = _controller(state_path, provider, FakeClock(CREATED))
    outcome = controller.evaluate(now=CREATED)

    assert outcome.status == "LIVE_BLOCKED"
    assert provider.get_count == 0
    assert provider.stop_count == 0
    assert "anchor" in outcome.message
