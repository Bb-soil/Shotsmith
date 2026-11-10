from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.p1 import clore_session as cs
from scripts.p1 import owner_risk, v3_lifecycle, v3_operator
from scripts.p1.clore_bridge import CloreCancelAck, CloreOrder

ROOT = Path(__file__).resolve().parents[1]
PENDING = ROOT / "data" / "p1-quotes" / "session-02.pending.json"


@pytest.fixture(autouse=True)
def _isolate_owner_risk_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(owner_risk, "DEFAULT_REGISTRY_ROOT", tmp_path / "registry")


def _iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat()


def _write_fresh_quote(
    path: Path,
    *,
    now: datetime | None = None,
    price_per_day: str = "12",
    currency: str = "USD-Blockchain",
    fresh_verified: bool = True,
    fees_known: bool = False,
    tax_fx_known: bool = False,
    min_rental_hours: str | None = None,
    mandatory_other_fees: str | None = None,
) -> Path:
    current = now or datetime.now(timezone.utc)
    payload = {
        "observed_at": _iso(current),
        "observation_origin": "owner_forwarded_official_response_not_direct_account_audit",
        "provenance": {
            "kind": "owner_forwarded_official_response",
            "not_provider_api_audit": True,
        },
        "quote": {
            "price_per_day": price_per_day,
            "price_per_hour": "0.50",
            "price_per_minute": "0.008333333333333333333333333333",
            "currency": currency,
            "creation_fee": "0.10",
            "mandatory_other_fees": mandatory_other_fees,
            "reserve": "0.10",
            "fresh_verified": fresh_verified,
            "fees_known": fees_known,
            "tax_fx_known": tax_fx_known,
            "min_rental_hours": min_rental_hours,
            "provider_fee_percent": "5",
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _admission(
    tmp_path: Path,
    quote_path: Path,
    *,
    now: datetime | None = None,
    output: str = "admission.json",
    valid_hours: int = 2,
    server_id: int = owner_risk.ALLOWED_SERVER_ID,
    hoster_id: int = owner_risk.ALLOWED_HOSTER_ID,
):
    return owner_risk.create_admission(
        quote=quote_path,
        output_path=tmp_path / output,
        now=now,
        valid_hours=valid_hours,
        server_id=server_id,
        hoster_id=hoster_id,
    )


def _state_with_risk(
    tmp_path: Path,
    quote_path: Path,
    admission,
    *,
    now: datetime | None = None,
    order_id: int = 97001,
    state_name: str = "state.json",
):
    current = now or datetime.now(timezone.utc)
    quote = cs.CloreQuote.from_dict(
        json.loads(quote_path.read_text(encoding="utf-8"))["quote"]
    )
    return cs.initialize_state(
        tmp_path / state_name,
        order_id=order_id,
        server_id=111947,
        created_at=int(current.timestamp()) - 5,
        quote=quote,
        owner_risk_admission=admission,
        now=current,
    )


class _FakeOrderBridge:
    def __init__(self, order: CloreOrder) -> None:
        self.order = order
        self.cancelled: list[int] = []
        self.cancelled_at: int | None = None

    async def read_order(self, order_id: int, *, deadline_monotonic=None):
        del deadline_monotonic
        assert order_id == self.order.order_id
        expired = self.cancelled_at is not None
        return replace(self.order, expired=expired)

    async def cancel_order(self, order_id: int, *, deadline_monotonic=None):
        del deadline_monotonic
        self.cancelled.append(order_id)
        self.cancelled_at = 1
        return CloreCancelAck(
            code=0, accepted=True, ambiguous=False, reason="fake accepted"
        )


def _order_for(state: cs.CloreSessionState, **overrides):
    base = CloreOrder(
        order_id=state.order_id,
        server_id=state.server_id,
        created_at=state.created_at,
        price_per_day=Decimal("12"),
        currency="USD-Blockchain",
        expired=False,
        fee_percent=Decimal("5"),
        creation_fee=Decimal("0.10"),
        spend=Decimal("0"),
        spot=False,
        mrl_seconds=336 * 3600,
    )
    return replace(base, **overrides)


def test_pending_artifact_is_non_authorizing_and_does_not_rewrite_quote(tmp_path: Path) -> None:
    before = hashlib.sha256(PENDING.read_bytes()).hexdigest()
    output = tmp_path / "pending.json"
    payload = owner_risk.create_pending(pending_quote=PENDING, output_path=output)
    assert payload["authorizes_session"] is False
    assert payload["provider_billing_verified"] is False
    assert payload["status"] == "PENDING_QUOTE_NON_AUTHORIZING"
    assert payload["quote"]["fresh_verified"] is False
    assert payload["quote"]["fees_known"] is False
    assert payload["quote"]["tax_fx_known"] is False
    assert payload["quote"]["min_rental_hours"] is None
    assert payload["policy_forecast_60m_usd"] == Decimal("0.725")
    assert payload["observed_projection_60m_usd"] == Decimal("0.7565")
    assert hashlib.sha256(PENDING.read_bytes()).hexdigest() == before


def test_valid_admission_allows_only_exact_scope_and_preserves_flags(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    assert admission.scope["server_id"] == 111947
    assert admission.scope["currency"] == "USD-Blockchain"
    assert admission.scope["price_per_day"] == "12"
    assert admission.projection["total_60m"] == "0.725"
    assert admission.verification["provider_billing_verified"] is False
    assert admission.verification["tax_fx_verified"] is False
    assert admission.verification["minimum_billing_verified"] is False
    assert admission.verification["financial_status"] == "OWNER_ACCEPTED_FINANCIAL_RISK"

    state = _state_with_risk(tmp_path, quote_path, admission, now=now)
    binding = v3_lifecycle.SessionBinding.from_state(state)
    assert state.server_id == 111947
    assert state.quote.fees_known is False
    assert state.quote.tax_fx_known is False
    assert state.quote.min_rental_hours is None
    assert state.quote.mandatory_other_fees is None
    assert state.quote.price_per_day == Decimal("12")
    assert state.owner_risk_sha256 == owner_risk.admission_digest(admission)
    assert binding.owner_risk_admission_sha256 == state.owner_risk_sha256
    assert v3_lifecycle.SessionBinding.from_dict(binding.to_dict()) == binding
    assert state.quote.launch_blockers(admission=state.owner_risk) == ()
    assert state.quote.launch_blockers()
    v3_lifecycle.assert_budget(binding, state, now=now)
    loaded = cs.load_state(tmp_path / "state.json")
    assert loaded.owner_risk_sha256 == state.owner_risk_sha256
    assert loaded.quote.fees_known is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"server_id": 112213},
        {"hoster_id": 1},
    ],
)
def test_admission_rejects_wrong_server_or_hoster(tmp_path: Path, kwargs) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    with pytest.raises(owner_risk.OwnerRiskError):
        _admission(tmp_path, quote_path, now=now, **kwargs)


@pytest.mark.parametrize(
    "quote_kwargs",
    [
        {"price_per_day": "13.20"},
        {"currency": "USD"},
        {"fresh_verified": False},
        {"fees_known": True},
        {"tax_fx_known": True},
        {"min_rental_hours": "1"},
    ],
)
def test_admission_rejects_wrong_financial_scope(tmp_path: Path, quote_kwargs) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now, **quote_kwargs)
    with pytest.raises(owner_risk.OwnerRiskError):
        _admission(tmp_path, quote_path, now=now)


def test_admission_rejects_stale_or_wrong_document(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    stale = _write_fresh_quote(
        tmp_path / "stale.json", now=now - timedelta(minutes=16)
    )
    with pytest.raises(owner_risk.OwnerRiskError):
        _admission(tmp_path, stale, now=now)
    fresh = _write_fresh_quote(tmp_path / "fresh.json", now=now)
    with pytest.raises(owner_risk.OwnerRiskError):
        owner_risk.create_admission(
            quote=fresh,
            output_path=tmp_path / "wrong-doc.json",
            approval_doc=ROOT / "docs" / "P1-session-02-approval-20261026.md",
            now=now,
        )


def test_tampered_state_or_admission_fails_closed(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    state = _state_with_risk(tmp_path, quote_path, admission, now=now)
    state_path = tmp_path / "state.json"
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["owner_risk"]["scope"]["price_per_day"] = "13"
    state_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(cs.CloreStateError):
        cs.load_state(state_path)

    artifact = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    artifact["scope"]["currency"] = "USD"
    (tmp_path / "tampered.json").write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        owner_risk.load_admission(tmp_path / "tampered.json", now=now)


def test_owner_risk_cap_and_duration_are_exact(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    quote = cs.CloreQuote.from_dict(
        json.loads(quote_path.read_text(encoding="utf-8"))["quote"]
    )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "low-cap.json",
            order_id=1,
            server_id=111947,
            created_at=int(now.timestamp()) - 5,
            quote=quote,
            cap_usd=Decimal("0.50"),
            owner_risk_admission=admission,
            now=now,
        )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "short.json",
            order_id=1,
            server_id=111947,
            created_at=int(now.timestamp()) - 5,
            quote=quote,
            duration_seconds=1800,
            owner_risk_admission=admission,
            now=now,
        )


def test_bind_claim_single_use_and_second_order_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(timezone.utc)
    registry = tmp_path / "registry"
    monkeypatch.setattr(owner_risk, "DEFAULT_REGISTRY_ROOT", registry)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    state_path = tmp_path / "bound.json"
    created = int(now.timestamp()) - 5
    first = v3_operator.bind_session(
        session_state=state_path,
        quote=quote_path,
        order_id=97001,
        created_at=created,
        owner_risk_admission=tmp_path / "admission.json",
    )
    assert first["financial_status"] == "OWNER_ACCEPTED_FINANCIAL_RISK"
    assert first["provider_billing_verified"] is False
    observation = json.loads(
        Path(first["quote_observation"]).read_text(encoding="utf-8")
    )
    quote_document = json.loads(quote_path.read_text(encoding="utf-8"))
    assert observation["observed_at"] == quote_document["observed_at"]
    assert "artifact_created_at" in observation
    binding = v3_lifecycle.SessionBinding.from_dict(first["session"])
    bound, reserve = owner_risk.claim_paths(
        owner_risk.permission_id_for_admission(admission), registry_root=registry
    )
    assert bound.is_file()
    assert reserve.is_file()
    before = bound.read_bytes()

    same = v3_operator.bind_session(
        session_state=state_path,
        quote=quote_path,
        order_id=97001,
        created_at=created,
        owner_risk_admission=tmp_path / "admission.json",
    )
    assert same["session"] == first["session"]
    assert bound.read_bytes() == before
    assert binding.owner_risk_admission_sha256 is not None

    with pytest.raises(v3_operator.OperatorError):
        v3_operator.bind_session(
            session_state=tmp_path / "second.json",
            quote=quote_path,
            order_id=97002,
            created_at=created + 1,
            owner_risk_admission=tmp_path / "admission.json",
        )

    copied = tmp_path / "copied-admission.json"
    shutil.copyfile(tmp_path / "admission.json", copied)
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.bind_session(
            session_state=tmp_path / "third.json",
            quote=quote_path,
            order_id=97003,
            created_at=created + 2,
            owner_risk_admission=copied,
        )

    second_quote = _write_fresh_quote(
        tmp_path / "quote-2.json", now=now + timedelta(seconds=1)
    )
    with pytest.raises(owner_risk.OwnerRiskError):
        owner_risk.create_admission(
            quote=second_quote,
            output_path=tmp_path / "admission-2.json",
            now=now,
        )


def test_expired_admission_blocks_new_work_but_watch_still_cancels(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now, valid_hours=1)
    quote = cs.CloreQuote.from_dict(
        json.loads(quote_path.read_text(encoding="utf-8"))["quote"]
    )
    bind_at = now + timedelta(minutes=15)
    created = int(bind_at.timestamp())
    state_path = tmp_path / "state.json"
    state = cs.initialize_state(
        state_path,
        order_id=97001,
        server_id=111947,
        created_at=created,
        quote=quote,
        owner_risk_admission=admission,
        now=bind_at,
    )
    expired_at = now + timedelta(minutes=65)
    with pytest.raises(cs.CloreStateError):
        cs.load_state(state_path, now=expired_at)
    expired = cs.load_state(state_path, safety_mode=True, now=expired_at)
    binding = v3_lifecycle.SessionBinding.from_state(expired)
    with pytest.raises(v3_lifecycle.LifecycleError):
        v3_lifecycle.assert_budget(binding, expired, now=expired_at)
    assert expired.quote.launch_blockers(
        admission=expired.owner_risk, now=expired_at
    )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "expired-new.json",
            order_id=97002,
            server_id=111947,
            created_at=int(expired_at.timestamp()) - 5,
            quote=expired.quote,
            owner_risk_admission=expired.owner_risk,
            now=expired_at,
        )

    bridge = _FakeOrderBridge(_order_for(expired, price_per_day=Decimal("12.01")))
    session = cs.CloreSession(
        state_path, bridge=bridge, clock=lambda: expired_at
    )
    outcome = asyncio.run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancelled == [expired.order_id]


def test_owner_risk_stop_now_cancels_exact_order_and_reads_back(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    state = _state_with_risk(tmp_path, quote_path, admission, now=now)
    bridge = _FakeOrderBridge(_order_for(state))
    session = cs.CloreSession(tmp_path / "state.json", bridge=bridge)
    outcome = asyncio.run(session.stop_now(reason="owner-risk explicit stop"))
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancelled == [state.order_id]
    assert cs.load_state(tmp_path / "state.json", safety_mode=True).status == "ENDED_REPORTED"


@pytest.mark.parametrize(
    "override,expect_cancel",
    [
        ({"price_per_day": Decimal("12.01")}, True),
        ({"spend": Decimal("1.01")}, True),
        ({"spend": None}, True),
    ],
)
def test_watch_price_or_budget_threat_cancels_exact_order(
    tmp_path: Path, override, expect_cancel: bool
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    state = _state_with_risk(tmp_path, quote_path, admission, now=now)
    bridge = _FakeOrderBridge(_order_for(state, **override))
    session = cs.CloreSession(tmp_path / "state.json", bridge=bridge)
    outcome = asyncio.run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bool(bridge.cancelled) is expect_cancel
    assert bridge.cancelled == [state.order_id]


def test_cli_pending_admit_and_bind_are_offline_and_no_manual_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(owner_risk, "DEFAULT_REGISTRY_ROOT", tmp_path / "registry")
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    pending_out = tmp_path / "pending-cli.json"
    assert (
        v3_operator.main(
            [
                "owner-risk-pending",
                "--pending-quote",
                str(PENDING),
                "--output",
                str(pending_out),
                "--json",
            ]
        )
        == 0
    )
    pending = json.loads(pending_out.read_text(encoding="utf-8"))
    assert pending["authorizes_session"] is False

    admission_out = tmp_path / "admission-cli.json"
    assert (
        v3_operator.main(
            [
                "owner-risk-admit",
                "--quote",
                str(quote_path),
                "--expected-quote-sha256",
                hashlib.sha256(quote_path.read_bytes()).hexdigest(),
                "--output",
                str(admission_out),
                "--json",
            ]
        )
        == 0
    )
    assert admission_out.is_file()
    capsys.readouterr()
    created = int(now.timestamp()) - 5
    assert (
        v3_operator.main(
            [
                "session-bind",
                "--session-state",
                str(tmp_path / "cli-state.json"),
                "--quote",
                str(quote_path),
                "--order-id",
                "97101",
                "--created-at",
                str(created),
                "--owner-risk-admission",
                str(admission_out),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["financial_status"] == "OWNER_ACCEPTED_FINANCIAL_RISK"
    assert payload["provider_billing_verified"] is False


def test_public_initialize_uses_same_claim_and_rejects_copied_state(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    quote = cs.CloreQuote.from_dict(
        json.loads(quote_path.read_text(encoding="utf-8"))["quote"]
    )
    state_path = tmp_path / "state.json"
    first = cs.initialize_state(
        state_path,
        order_id=97001,
        server_id=111947,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        owner_risk_admission=admission,
        now=now,
    )
    same = cs.initialize_state(
        state_path,
        order_id=97001,
        server_id=111947,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        owner_risk_admission=admission,
        now=now,
    )
    assert same.owner_risk_sha256 == first.owner_risk_sha256
    copied = tmp_path / "copied-state.json"
    shutil.copyfile(state_path, copied)
    with pytest.raises(cs.CloreStateError):
        cs.load_state(copied)
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "second.json",
            order_id=97002,
            server_id=111947,
            created_at=int(now.timestamp()) - 5,
            quote=quote,
            owner_risk_admission=admission,
            now=now,
        )
    assert not (tmp_path / "second.json").exists()


def test_bind_rejects_stale_original_quote_before_state(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    later = now + timedelta(minutes=20)
    with pytest.raises(owner_risk.OwnerRiskError):
        owner_risk.load_admission(tmp_path / "admission.json", now=later)
    state_path = tmp_path / "stale-bind.json"
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.bind_session(
            session_state=state_path,
            quote=quote_path,
            order_id=97001,
            created_at=int(later.timestamp()) - 5,
            owner_risk_admission=tmp_path / "admission.json",
            now=later,
        )
    assert not state_path.exists()

    future_quote = _write_fresh_quote(
        tmp_path / "future-quote.json", now=now + timedelta(minutes=10)
    )
    with pytest.raises(owner_risk.OwnerRiskError):
        owner_risk.create_admission(
            quote=future_quote,
            output_path=tmp_path / "future-admission.json",
            now=now,
        )


def test_stable_consent_id_ignores_metadata_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = _write_fresh_quote(tmp_path / "quote.json", now=now)
    admission = _admission(tmp_path, quote_path, now=now)
    state_path = tmp_path / "state.json"
    v3_operator.bind_session(
        session_state=state_path,
        quote=quote_path,
        order_id=97001,
        created_at=int(now.timestamp()) - 5,
        owner_risk_admission=tmp_path / "admission.json",
        now=now,
    )
    altered_scope = dict(admission.scope)
    altered_scope["mrl_semantics"] = "MINIMUM"
    assert (
        owner_risk.admission_id_for(
            approval_doc=admission.approval_doc,
            session_approval_doc=admission.session_approval_doc,
            scope=altered_scope,
        )
        == admission.admission_id
    )
    original = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    for name, mutate in (
        ("nonce", lambda data: data["scope"].__setitem__("nonce", "x")),
        (
            "provenance",
            lambda data: data["scope"].__setitem__(
                "hoster_id_provenance", "cosmetic"
            ),
        ),
        (
            "mrl",
            lambda data: data["scope"].__setitem__("mrl_semantics", "MINIMUM"),
        ),
    ):
        payload = json.loads(json.dumps(original))
        mutate(payload)
        path = tmp_path / f"tampered-{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(owner_risk.OwnerRiskError):
            owner_risk.load_admission(path, now=now)

    alternate = json.loads(json.dumps(original))
    alternate["scope"]["price_per_day"] = "1.2E+1"
    alternate["scope"]["provider_fee_percent"] = "5.0"
    alternate["scope"]["creation_fee"] = "0.100"
    alternate["scope"]["reserve"] = "0.10"
    alternate["scope"]["code_cap_usd"] = "1.0E0"
    alternate["scope"]["owner_ceiling_usd"] = "2.000E0"
    alternate["admission_id"] = owner_risk.admission_id_for(
        approval_doc=alternate["approval_doc"],
        session_approval_doc=alternate["session_approval_doc"],
        scope=alternate["scope"],
    )
    alternate_path = tmp_path / "alternate-numeric.json"
    alternate_path.write_text(json.dumps(alternate), encoding="utf-8")
    alternate_admission = owner_risk.load_admission(alternate_path, now=now)
    assert alternate_admission.admission_id == admission.admission_id
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.bind_session(
            session_state=tmp_path / "alternate-second.json",
            quote=quote_path,
            order_id=97002,
            created_at=int(now.timestamp()) - 4,
            owner_risk_admission=alternate_path,
            now=now,
        )
    assert not (tmp_path / "alternate-second.json").exists()
