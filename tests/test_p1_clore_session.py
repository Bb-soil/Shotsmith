from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from scripts.p1 import clore_session as cs
from scripts.p1.clore_bridge import (
    EXPECTED_CURRENCY,
    CloreCancelAck,
    CloreCancelAmbiguous,
    CloreCurrencyMismatch,
    CloreNotFound,
    CloreOrder,
    CloreTransportError,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "p1" / "clore-v1"
CREATED = 1761090000
START = datetime.fromtimestamp(CREATED, tz=timezone.utc)


class FakeClock:
    def __init__(self, start: datetime = START) -> None:
        self.now = start
        self.mono = 0.0

    def clock(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.mono

    async def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)
        self.mono += seconds


class FakeBridge:
    def __init__(
        self,
        order: CloreOrder | None = None,
        *,
        expired_order: CloreOrder | None = None,
        cancel_ack: CloreCancelAck | None = None,
        cancel_exc: Exception | None = None,
        cancel_hook: Any = None,
        read_errors: list[Exception] | None = None,
        read_errors_after_cancel: list[Exception] | None = None,
        missing_after_cancel: bool = False,
    ) -> None:
        self.order = order or _order()
        self.expired_order = expired_order or _order(expired=True)
        self.cancel_ack = cancel_ack or CloreCancelAck(0, True, False, "ack")
        self.cancel_exc = cancel_exc
        self.cancel_hook = cancel_hook
        self.read_errors = list(read_errors or [])
        self.read_errors_after_cancel = list(read_errors_after_cancel or [])
        self.missing_after_cancel = missing_after_cancel
        self.read_calls = 0
        self.cancel_calls = 0

    async def read_order(
        self, order_id: int, *, deadline_monotonic: float | None = None
    ) -> CloreOrder:
        self.read_calls += 1
        if self.read_errors:
            raise self.read_errors.pop(0)
        if self.cancel_calls and self.read_errors_after_cancel:
            raise self.read_errors_after_cancel.pop(0)
        if self.missing_after_cancel and self.cancel_calls:
            raise CloreNotFound("missing")
        if self.cancel_calls:
            return self.expired_order
        return self.order

    async def cancel_order(
        self, order_id: int, *, deadline_monotonic: float | None = None
    ) -> CloreCancelAck:
        self.cancel_calls += 1
        if self.cancel_hook is not None:
            self.cancel_hook()
        if self.cancel_exc is not None:
            raise self.cancel_exc
        return self.cancel_ack


def _quote(**overrides: Any) -> cs.CloreQuote:
    data = json.loads((FIXTURE_DIR / "quote-ready-example.json").read_text(encoding="utf-8"))
    data.update(overrides)
    return cs.CloreQuote.from_dict(data)


def _owner_quote() -> cs.CloreQuote:
    data = json.loads(
        (FIXTURE_DIR / "quote-owner-example.json").read_text(encoding="utf-8")
    )
    return cs.CloreQuote.from_dict(data)


def _order(**overrides: Any) -> CloreOrder:
    base = CloreOrder(
        order_id=7001,
        server_id=112213,
        created_at=CREATED,
        price_per_day=Decimal("13.2"),
        currency=EXPECTED_CURRENCY,
        expired=False,
        fee_percent=Decimal("5"),
        creation_fee=Decimal("0.10"),
        spend=Decimal("0"),
        spot=False,
        mrl_seconds=518400,
    )
    return replace(base, **overrides)


def _session(
    tmp_path: Path,
    bridge: FakeBridge,
    *,
    clock: FakeClock | None = None,
    readback_timeout_seconds: float = 120.0,
    quote: cs.CloreQuote | None = None,
) -> cs.CloreSession:
    quote = quote or _quote()
    clock = clock or FakeClock()
    state = cs.initialize_state(
        tmp_path / "state.json",
        order_id=7001,
        created_at=CREATED,
        quote=quote,
        now=START + timedelta(seconds=5),
    )
    assert state.status == "ARMED"
    return cs.CloreSession(
        tmp_path / "state.json",
        bridge=bridge,
        clock=clock.clock,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        readback_timeout_seconds=readback_timeout_seconds,
        readback_poll_seconds=5.0,
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_owner_screenshot_projection_is_not_launch_ready(tmp_path: Path) -> None:
    quote = _owner_quote()
    projection = quote.project_total(60)
    assert projection == Decimal("0.7275")
    blockers = quote.launch_blockers(cap_usd=Decimal("1.00"), minutes=60)
    assert any("fresh" in blocker for blocker in blockers)
    assert any("minimum rental" in blocker for blocker in blockers)
    with pytest.raises(cs.CloreLaunchBlocked) as exc_info:
        cs.initialize_state(
            tmp_path / "unused.json",
            order_id=7001,
            created_at=CREATED,
            quote=quote,
            now=START,
        )
    assert exc_info.value.blockers


def test_ready_quote_projects_and_launches(tmp_path: Path) -> None:
    quote = _quote()
    assert quote.project_total(60) == Decimal("0.7275")
    assert quote.launch_blockers(cap_usd=Decimal("1.00"), minutes=60) == ()
    state = cs.initialize_state(
        tmp_path / "state.json",
        order_id=7001,
        created_at=CREATED,
        quote=quote,
        now=START,
    )
    assert state.provider == "clore"
    assert state.status == "ARMED"
    assert state.cap_usd == Decimal("1.00")
    assert state.hard_end_at == CREATED + 3600


def test_six_hour_minimum_is_explicit_launch_blocker_no_guess(tmp_path: Path) -> None:
    quote = _quote(min_rental_hours="6")
    blockers = quote.launch_blockers(cap_usd=Decimal("1.00"), minutes=60)
    assert any("minimum rental" in blocker for blocker in blockers)
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "state-six.json",
            order_id=7001,
            created_at=CREATED,
            quote=quote,
            now=START,
        )
    assert not (tmp_path / "state-six.json").exists()


def test_currency_or_cap_mismatch_cannot_launch(tmp_path: Path) -> None:
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "state-currency.json",
            order_id=7001,
            created_at=CREATED,
            currency="bitcoin",
            quote=_quote(),
            now=START,
        )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "state-cap.json",
            order_id=7001,
            created_at=CREATED,
            quote=_quote(),
            cap_usd=Decimal("0.50"),
            now=START,
        )


def test_state_is_immutable_and_cannot_rebind_or_extend(tmp_path: Path) -> None:
    state_path = tmp_path / "immutable-state.json"
    first = cs.initialize_state(
        state_path, order_id=7001, created_at=CREATED, quote=_quote(), now=START
    )
    assert (
        cs.initialize_state(
            state_path, order_id=7001, created_at=CREATED, quote=_quote(), now=START
        )
        == first
    )
    with pytest.raises(cs.CloreStateError):
        cs.initialize_state(
            state_path, order_id=7002, created_at=CREATED, quote=_quote(), now=START
        )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            state_path,
            order_id=7001,
            created_at=CREATED,
            quote=_quote(),
            planned_cancel_at=CREATED + 3301,
            now=START,
        )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            state_path,
            order_id=7001,
            created_at=CREATED,
            quote=_quote(),
            hard_end_at=CREATED + 3601,
            now=START,
        )


def test_runpod_state_legacy_is_never_rebound(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({"schema": "anime-video-studio/p1-stop-state/v1", "pod_id": "abc"}),
        encoding="utf-8",
    )
    with pytest.raises(cs.CloreStateError, match="RunPod"):
        cs.load_state(path)
    with pytest.raises(cs.CloreStateError, match="RunPod"):
        cs.initialize_state(
            path, order_id=7001, created_at=CREATED, quote=_quote(), now=START
        )


def test_deadline_tampering_is_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path, FakeBridge())
    data = json.loads(session.state_path.read_text(encoding="utf-8"))
    data["hard_end_at"] = CREATED + 3601
    session.state_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(cs.CloreStateError):
        cs.load_state(session.state_path)


def test_before_planned_is_armed_and_does_not_mutate(tmp_path: Path) -> None:
    bridge = FakeBridge()
    session = _session(tmp_path, bridge)
    clock = FakeClock(START + timedelta(seconds=600))
    session._clock = clock.clock
    session._monotonic = clock.monotonic
    outcome = _run(session.evaluate())
    assert outcome.status == "ARMED"
    assert bridge.cancel_calls == 0
    assert bridge.read_calls == 1


def test_planned_cancel_persists_intent_before_post_and_ends_provisionally(
    tmp_path: Path,
) -> None:
    bridge = FakeBridge()
    original_hook = bridge.cancel_hook

    def hook() -> None:
        persisted = cs.load_state(tmp_path / "state.json")
        assert persisted.status == "STOP_REQUESTED"
        assert persisted.cancel_intent_at is not None
        assert persisted.readback_started_at is not None
        if original_hook is not None:
            original_hook()

    bridge.cancel_hook = hook
    clock = FakeClock(START + timedelta(seconds=3300))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert outcome.billing_verified is False
    assert bridge.cancel_calls == 1
    assert bridge.read_calls == 2
    assert cs.load_state(session.state_path).cancel_reason == "planned cancellation deadline"


def test_cancel_ambiguous_is_readback_only_and_never_retried(tmp_path: Path) -> None:
    bridge = FakeBridge(cancel_exc=CloreCancelAmbiguous("ambiguous test"))
    clock = FakeClock(START + timedelta(seconds=3300))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    persisted = cs.load_state(session.state_path)
    assert persisted.status == "ENDED_REPORTED"
    assert "ambiguous" in (persisted.cancel_reason or "").lower() or (
        "planned" in (persisted.cancel_reason or "").lower()
    )


def test_code_zero_without_terminal_readback_is_live_blocked(tmp_path: Path) -> None:
    bridge = FakeBridge(expired_order=_order(expired=False))
    clock = FakeClock(START + timedelta(seconds=3300))
    session = _session(tmp_path, bridge, clock=clock, readback_timeout_seconds=0.01)
    outcome = _run(session.evaluate())
    assert outcome.status == "LIVE_BLOCKED"
    assert outcome.billing_verified is False
    assert bridge.cancel_calls == 1


def test_missing_row_after_cancel_is_live_blocked_not_ended(tmp_path: Path) -> None:
    bridge = FakeBridge(missing_after_cancel=True)
    clock = FakeClock(START + timedelta(seconds=3300))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "LIVE_BLOCKED"
    assert "missing" in outcome.message


def test_identity_mismatch_forbids_mutation(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(created_at=CREATED + 1))
    clock = FakeClock(START + timedelta(seconds=3300))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "LIVE_BLOCKED"
    assert bridge.cancel_calls == 0


def test_currency_or_price_breach_triggers_exact_identity_cancel(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(currency="bitcoin"))
    clock = FakeClock(START + timedelta(seconds=600))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    persisted = cs.load_state(session.state_path)
    assert "currency" in (persisted.cancel_reason or "").lower()


def test_bridge_currency_mismatch_still_triggers_exact_identity_cancel(
    tmp_path: Path,
) -> None:
    class CurrencyBridge(FakeBridge):
        async def read_order(
            self, order_id: int, *, deadline_monotonic: float | None = None
        ) -> CloreOrder:
            self.read_calls += 1
            if self.cancel_calls:
                return _order(expired=True)
            raise CloreCurrencyMismatch(
                "currency changed", order=_order(currency="bitcoin")
            )

    bridge = CurrencyBridge()
    clock = FakeClock(START + timedelta(seconds=600))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    assert "currency" in (cs.load_state(session.state_path).cancel_reason or "").lower()


def test_budget_actual_spend_over_cap_cancels_early(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(spend=Decimal("1.01")))
    clock = FakeClock(START + timedelta(seconds=600))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    assert "spend" in (cs.load_state(session.state_path).cancel_reason or "").lower()


def test_budget_missing_spend_cancels_early(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(spend=None))
    clock = FakeClock(START + timedelta(seconds=600))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    assert "missing" in (cs.load_state(session.state_path).cancel_reason or "").lower()


def test_hard_end_passed_makes_no_provider_call(tmp_path: Path) -> None:
    bridge = FakeBridge()
    clock = FakeClock(START + timedelta(seconds=3600))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "LIVE_BLOCKED"
    assert bridge.read_calls == 0
    assert bridge.cancel_calls == 0


def test_expired_readback_bound_makes_no_provider_call(tmp_path: Path) -> None:
    bridge = FakeBridge()
    session = _session(tmp_path, bridge)
    state = cs.load_state(session.state_path)
    expired = replace(
        state,
        status="STOP_REQUESTED",
        cancel_reason="test",
        cancel_intent_at=cs._format_utc(START),
        readback_started_at=cs._format_utc(START),
        readback_deadline_at=cs._format_utc(START + timedelta(seconds=1)),
    )
    cs.save_state(session.state_path, expired)
    clock = FakeClock(START + timedelta(seconds=2))
    session._clock = clock.clock
    session._monotonic = clock.monotonic
    outcome = _run(session.evaluate())
    assert outcome.status == "LIVE_BLOCKED"
    assert bridge.read_calls == 0


def test_watch_uses_fake_clock_and_cancels_at_planned_not_60(tmp_path: Path) -> None:
    bridge = FakeBridge()
    clock = FakeClock()
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.watch(poll_interval_seconds=300.0))
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    assert clock.now >= START + timedelta(seconds=3300)
    assert clock.now < START + timedelta(seconds=3600)


def test_watch_budget_break_cancels_before_planned(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(spend=Decimal("1.01")))
    clock = FakeClock()
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.watch(poll_interval_seconds=300.0))
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    assert clock.now < START + timedelta(seconds=3300)


def test_early_stop_is_independent_of_generation_or_export(tmp_path: Path) -> None:
    bridge = FakeBridge()
    clock = FakeClock(START + timedelta(seconds=300))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.stop_now(reason="generation error"))
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1


def test_transient_readback_error_is_bounded_then_reported(tmp_path: Path) -> None:
    bridge = FakeBridge(
        read_errors_after_cancel=[CloreTransportError("transient")]
    )
    clock = FakeClock(START + timedelta(seconds=3300))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1


def test_no_create_generate_or_topup_interface_exists(tmp_path: Path) -> None:
    session = _session(tmp_path, FakeBridge())
    for forbidden in ("create_order", "generate", "topup", "extend"):
        assert not hasattr(session, forbidden)


def test_restart_with_stop_intent_is_readback_only(tmp_path: Path) -> None:
    clock = FakeClock(START + timedelta(seconds=3300))
    session = _session(tmp_path, FakeBridge(), clock=clock)
    state = cs.load_state(session.state_path)
    requested = replace(
        state,
        status="STOP_REQUESTED",
        cancel_intent_at=cs._format_utc(clock.now),
        cancel_reason="simulated crash after intent",
        readback_started_at=cs._format_utc(clock.now),
        readback_deadline_at=cs._format_utc(
            clock.now + timedelta(seconds=120)
        ),
    )
    cs.save_state(session.state_path, requested)

    second_bridge = FakeBridge(order=_order(expired=True))
    restarted = cs.CloreSession(
        tmp_path / "state.json",
        bridge=second_bridge,
        clock=clock.clock,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    restarted_outcome = _run(restarted.evaluate())
    assert restarted_outcome.status == "ENDED_REPORTED"
    assert second_bridge.cancel_calls == 0
    assert second_bridge.read_calls == 1


def test_creation_fee_is_added_once_only_after_ten_minutes(tmp_path: Path) -> None:
    bridge = FakeBridge()
    session = _session(tmp_path, bridge)
    state = cs.load_state(session.state_path)
    before = session._actual_total(_order(spend=Decimal("0.50")), state, START + timedelta(minutes=9))
    after = session._actual_total(_order(spend=Decimal("0.50")), state, START + timedelta(minutes=11))
    assert before == Decimal("0.50")
    assert after == Decimal("0.60")
    # The documented spend already includes the renter fee; no second fee is added.
    assert after != Decimal("0.625")


def test_concurrent_stop_failfast_while_read_paused_keeps_loop_alive(
    tmp_path: Path,
) -> None:
    class PausedBridge(FakeBridge):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def read_order(
            self, order_id: int, *, deadline_monotonic: float | None = None
        ) -> CloreOrder:
            self.read_calls += 1
            if not self.cancel_calls:
                self.entered.set()
                await self.release.wait()
            if self.cancel_calls:
                return _order(expired=True)
            return _order()

    bridge = PausedBridge()
    clock = FakeClock(START + timedelta(seconds=3300))
    session = _session(tmp_path, bridge, clock=clock)

    async def scenario() -> tuple[Any, Any, int]:
        first = asyncio.create_task(session.evaluate())
        await bridge.entered.wait()
        heartbeat = 0
        blocked: Any = None
        try:
            await session.stop_now()
        except cs.CloreSessionBusy as exc:
            blocked = exc
        heartbeat += 1
        bridge.release.set()
        outcome = await first
        return outcome, blocked, heartbeat

    outcome, blocked, heartbeat = _run(scenario())
    assert isinstance(blocked, cs.CloreSessionBusy)
    assert heartbeat == 1
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1


def test_budget_t30_spend_089_cancels_before_overcap(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(spend=Decimal("0.89")))
    clock = FakeClock(START + timedelta(minutes=30))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    assert "upper bound" in (cs.load_state(session.state_path).cancel_reason or "")


def test_provider_fee_percent_drift_triggers_cancel(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(fee_percent=Decimal("6")))
    clock = FakeClock(START + timedelta(minutes=10))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1
    assert "fee percent" in (cs.load_state(session.state_path).cancel_reason or "")


def test_read_crossing_planned_deadline_cancels_immediately(tmp_path: Path) -> None:
    class AdvancingBridge(FakeBridge):
        async def read_order(
            self, order_id: int, *, deadline_monotonic: float | None = None
        ) -> CloreOrder:
            self.read_calls += 1
            if self.cancel_calls:
                return _order(expired=True)
            return _order()

    bridge = AdvancingBridge()
    clock = FakeClock(START + timedelta(seconds=54 * 60))

    async def scenario() -> cs.CloreSessionOutcome:
        session = _session(tmp_path, bridge, clock=clock)
        original_read = bridge.read_order

        async def slow_read(order_id: int, *, deadline_monotonic: float | None = None) -> CloreOrder:
            if not bridge.cancel_calls:
                clock.now += timedelta(seconds=120)
            return await original_read(order_id, deadline_monotonic=deadline_monotonic)

        bridge.read_order = slow_read  # type: ignore[method-assign]
        return await session.evaluate()

    outcome = _run(scenario())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1


def test_read_crossing_hard_end_makes_no_cancel(tmp_path: Path) -> None:
    bridge = FakeBridge()
    clock = FakeClock(START + timedelta(seconds=50 * 60))
    session = _session(tmp_path, bridge, clock=clock)
    original_read = bridge.read_order

    async def slow_read(order_id: int, *, deadline_monotonic: float | None = None) -> CloreOrder:
        clock.now += timedelta(seconds=11 * 60)
        return await original_read(order_id, deadline_monotonic=deadline_monotonic)

    bridge.read_order = slow_read  # type: ignore[method-assign]
    outcome = _run(session.evaluate())
    assert outcome.status == "LIVE_BLOCKED"
    assert bridge.cancel_calls == 0


def test_readback_late_expired_after_bound_is_not_accepted(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(expired=True))
    session = _session(tmp_path, bridge)
    state = cs.load_state(session.state_path)
    requested = replace(
        state,
        status="STOP_REQUESTED",
        cancel_intent_at=cs._format_utc(START),
        cancel_reason="test",
        readback_started_at=cs._format_utc(START),
        readback_deadline_at=cs._format_utc(START + timedelta(seconds=1)),
    )
    cs.save_state(session.state_path, requested)
    clock = FakeClock(START)
    session._clock = clock.clock
    session._monotonic = clock.monotonic
    original_read = bridge.read_order

    async def late_read(order_id: int, *, deadline_monotonic: float | None = None) -> CloreOrder:
        clock.now += timedelta(seconds=2)
        return await original_read(order_id, deadline_monotonic=deadline_monotonic)

    bridge.read_order = late_read  # type: ignore[method-assign]
    outcome = _run(session.evaluate())
    assert outcome.status == "LIVE_BLOCKED"
    assert "late terminal" in outcome.message


def test_existing_cli_binding_mismatch_fails_before_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path = tmp_path / "state.json"
    cs.initialize_state(
        state_path,
        order_id=7001,
        created_at=CREATED,
        quote=_quote(),
        now=START + timedelta(seconds=5),
    )

    class ExplodingBridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("bridge must not be constructed before validation")

    monkeypatch.setattr(cs, "CloreBridge", ExplodingBridge)
    code = cs.main(
        [
            "--live",
            "--state",
            str(state_path),
            "--quote",
            str(FIXTURE_DIR / "quote-ready-example.json"),
            "--order-id",
            "7002",
            "--created-at",
            str(CREATED),
        ]
    )
    assert code == 2
    assert "mismatch" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ["--order-id", "7002"],
        ["--created-at", str(CREATED + 1)],
        ["--cap-usd", "0.50"],
        ["--duration-seconds", "1800"],
        ["--server-id", "9"],
    ],
)
def test_existing_binding_immutable_flags_fail_before_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
) -> None:
    state_path = tmp_path / "state.json"
    cs.initialize_state(
        state_path,
        order_id=7001,
        created_at=CREATED,
        quote=_quote(),
        now=START + timedelta(seconds=5),
    )

    class ExplodingBridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("bridge must not be constructed before validation")

    monkeypatch.setattr(cs, "CloreBridge", ExplodingBridge)
    base = [
        "--live",
        "--state",
        str(state_path),
        "--quote",
        str(FIXTURE_DIR / "quote-ready-example.json"),
        "--order-id",
        "7001",
        "--created-at",
        str(CREATED),
    ]
    # Replace the default order/created-at when the parameter targets them.
    if extra[0] == "--order-id":
        base[base.index("--order-id") + 1] = extra[1]
    elif extra[0] == "--created-at":
        base[base.index("--created-at") + 1] = extra[1]
    else:
        base.extend(extra)
    assert cs.main(base) == 2


def test_existing_quote_or_creation_mismatch_fails_before_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    cs.initialize_state(
        state_path,
        order_id=7001,
        created_at=CREATED,
        quote=_quote(),
        now=START + timedelta(seconds=5),
    )

    class ExplodingBridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("bridge must not be constructed before validation")

    monkeypatch.setattr(cs, "CloreBridge", ExplodingBridge)
    code = cs.main(
        [
            "--live",
            "--state",
            str(state_path),
            "--quote",
            str(FIXTURE_DIR / "quote-owner-example.json"),
            "--order-id",
            "7001",
            "--created-at",
            str(CREATED),
        ]
    )
    assert code == 2


def test_quote_projection_adds_other_fees_and_reserve_once() -> None:
    quote = _quote(
        mandatory_other_fees="0.20",
        reserve="0.05",
    )
    assert quote.project_total(60) == Decimal("0.9275")
    assert quote.launch_blockers(cap_usd=Decimal("1.00"), minutes=60) == ()


def test_stop_now_ignores_budget_breach_and_cancels_bound_order(tmp_path: Path) -> None:
    bridge = FakeBridge(order=_order(spend=Decimal("2.00")))
    clock = FakeClock(START + timedelta(seconds=300))
    session = _session(tmp_path, bridge, clock=clock)
    outcome = _run(session.stop_now(reason="operator stop during breach"))
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancel_calls == 1


def test_existing_runpod_state_rejected_before_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"schema": "anime-video-studio/p1-stop-state/v1", "pod_id": "x"}),
        encoding="utf-8",
    )

    class ExplodingBridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("bridge must not be constructed")

    monkeypatch.setattr(cs, "CloreBridge", ExplodingBridge)
    code = cs.main(
        [
            "--live",
            "--state",
            str(state_path),
            "--quote",
            str(FIXTURE_DIR / "quote-ready-example.json"),
            "--order-id",
            "7001",
            "--created-at",
            str(CREATED),
        ]
    )
    assert code == 2


def test_new_blocked_quote_fails_before_bridge_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "new-state.json"

    class ExplodingBridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("bridge must not be constructed")

    monkeypatch.setattr(cs, "CloreBridge", ExplodingBridge)
    code = cs.main(
        [
            "--live",
            "--state",
            str(state_path),
            "--quote",
            str(FIXTURE_DIR / "quote-owner-example.json"),
            "--order-id",
            "7001",
            "--created-at",
            str(CREATED),
        ]
    )
    assert code == 2
    assert not state_path.exists()


def test_initialize_rejects_already_passed_hard_end(tmp_path: Path) -> None:
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "late.json",
            order_id=7001,
            created_at=CREATED,
            quote=_quote(),
            now=START + timedelta(seconds=3600),
        )

@pytest.mark.parametrize("fee", ["6", "100"])
def test_non_five_provider_fee_blocks_new_quote_before_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fee: str
) -> None:
    quote = _quote(provider_fee_percent=fee)
    assert quote.launch_blockers(cap_usd=Decimal("1.00"), minutes=60)
    quote_path = tmp_path / "quote.json"
    quote_path.write_text(json.dumps(quote.to_dict()), encoding="utf-8")
    state_path = tmp_path / "state.json"

    class ExplodingBridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("bridge must not be constructed")

    monkeypatch.setattr(cs, "CloreBridge", ExplodingBridge)
    code = cs.main(
        [
            "--live",
            "--state",
            str(state_path),
            "--quote",
            str(quote_path),
            "--order-id",
            "7001",
            "--created-at",
            str(CREATED),
        ]
    )
    assert code == 2
    assert not state_path.exists()
def test_cli_live_defaults_to_watch_and_once_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import types
    created = int(datetime.now(timezone.utc).timestamp()) - 5
    calls: list[str] = []

    class FakeBridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def aclose(self) -> None:
            pass

    class FakeSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def initialize(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def watch(self) -> Any:
            calls.append("watch")
            return types.SimpleNamespace(
                status="ENDED_REPORTED",
                message="watched",
                billing_verified=False,
                breach=None,
            )

        async def evaluate(self) -> Any:
            calls.append("evaluate")
            return types.SimpleNamespace(
                status="ARMED",
                message="armed once",
                billing_verified=False,
                breach=None,
            )

    monkeypatch.setattr(cs, "CloreBridge", FakeBridge)
    monkeypatch.setattr(cs, "CloreSession", FakeSession)
    base = [
        "--live",
        "--state",
        str(tmp_path / "state.json"),
        "--quote",
        str(FIXTURE_DIR / "quote-ready-example.json"),
        "--order-id",
        "7001",
        "--created-at",
        str(created),
    ]
    assert cs.main(base) == 0
    assert "watch" in calls
    assert cs.main(base + ["--once"]) == 3
    assert "WARNING: --once" in capsys.readouterr().err
def test_cli_watch_once_conflict_fails_before_quote_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class ExplodingBridge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("bridge must not be constructed")

    monkeypatch.setattr(cs, "CloreBridge", ExplodingBridge)
    code = cs.main(
        [
            "--live",
            "--watch",
            "--once",
            "--quote",
            str(tmp_path / "does-not-exist.json"),
            "--state",
            str(tmp_path / "state.json"),
            "--order-id",
            "7001",
            "--created-at",
            str(CREATED),
        ]
    )
    assert code == 2
    assert "mutually exclusive" in capsys.readouterr().err