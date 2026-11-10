from __future__ import annotations

import asyncio
import hashlib
import json

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.p1 import clore_session as cs
from scripts.p1 import marketplace_quote, owner_risk, v3_lifecycle, v3_operator
from scripts.p1.clore_bridge import CloreCancelAck, CloreOrder


@pytest.fixture(autouse=True)
def _isolate_owner_risk_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(owner_risk, "DEFAULT_REGISTRY_ROOT", tmp_path / "registry")


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _write_scoped_quote(
    path: Path,
    *,
    now: datetime,
    server_id: int = 222333,
    hoster_id: int = 777,
    price_per_day: str = "8",
    mrl_hours: int = 72,
    currency: str = "USD-Blockchain",
) -> dict:
    facts = {
        "server_id": server_id,
        "hoster_id": hoster_id,
        "mrl_raw_hours": mrl_hours,
        "mrl_raw_seconds": mrl_hours * 3600,
        "mrl_semantics": "MAXIMUM_RENTAL_LENGTH",
        "currency": currency,
        "price_per_day": price_per_day,
        "availability": {
            "listed": True,
            "rented": False,
            "online": None,
            "online_source": "not_exposed_by_marketplace",
        },
    }
    document = marketplace_quote.build_quote_document(
        facts=facts, requested_at=now, observed_at=now
    )
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def _rehash(document: dict) -> dict:
    facts = {
        "server_id": document["server_id"],
        "hoster_id": document["hoster_id"],
        "mrl_raw_hours": document["mrl_raw_hours"],
        "mrl_raw_seconds": document["mrl_raw_seconds"],
        "mrl_semantics": document["mrl_semantics"],
        "currency": document["currency"],
        "price_per_day": document["price_per_day"],
        "availability": dict(document["availability"]),
    }
    collector = {
        "collector_id": owner_risk.MARKET_COLLECTOR_ID,
        "collector_version": owner_risk.MARKET_COLLECTOR_VERSION,
        "method": owner_risk.MARKET_METHOD,
        "base_url": owner_risk.MARKET_BASE_URL,
        "endpoint": owner_risk.MARKET_ENDPOINT,
        "path": owner_risk.MARKET_PATH,
    }
    document["facts_sha256"] = marketplace_quote._sha_mapping(
        {
            "facts": facts,
            "quote": document["quote"],
            "proposal": document["proposal"],
            "requested_at": document["requested_at"],
            "observed_at": document["observed_at"],
            "observation_origin": document["observation_origin"],
            "collector": collector,
        }
    )
    return document


def _write_legacy_quote(path: Path, *, now: datetime) -> dict:
    document = {
        "observed_at": _iso(now),
        "observation_origin": "owner_forwarded_official_response_not_direct_account_audit",
        "provenance": {
            "kind": "owner_forwarded_official_response",
            "not_provider_api_audit": True,
        },
        "quote": {
            "price_per_day": "12",
            "price_per_hour": None,
            "price_per_minute": None,
            "currency": "USD-Blockchain",
            "creation_fee": "0.10",
            "mandatory_other_fees": None,
            "reserve": "0.10",
            "fresh_verified": True,
            "fees_known": False,
            "tax_fx_known": False,
            "min_rental_hours": None,
            "provider_fee_percent": "5",
        },
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def _scoped_admission(
    tmp_path: Path,
    quote_path: Path,
    *,
    now: datetime,
    server_id: int = 222333,
    hoster_id: int = 777,
    currency: str = "USD-Blockchain",
    name: str = "admission.json",
):
    return owner_risk.create_scoped_admission(
        quote=quote_path,
        output_path=tmp_path / name,
        server_id=server_id,
        hoster_id=hoster_id,
        currency=currency,
        approval_confirm=owner_risk.APPROVAL_CONFIRMATION,
        now=now,
    )


class _FakeBridge:
    def __init__(self, order: CloreOrder) -> None:
        self.order = order
        self.cancelled: list[int] = []
        self.expired = False

    async def read_order(self, order_id: int, *, deadline_monotonic=None):
        del deadline_monotonic
        assert order_id == self.order.order_id
        return replace(self.order, expired=self.expired)

    async def cancel_order(self, order_id: int, *, deadline_monotonic=None):
        del deadline_monotonic
        self.cancelled.append(order_id)
        self.expired = True
        return CloreCancelAck(
            code=0, accepted=True, ambiguous=False, reason="fake accepted"
        )


def _order(state: cs.CloreSessionState, price: str) -> CloreOrder:
    return CloreOrder(
        order_id=state.order_id,
        server_id=state.server_id,
        created_at=state.created_at,
        price_per_day=Decimal(price),
        currency=state.currency,
        expired=False,
        fee_percent=Decimal("5"),
        creation_fee=Decimal("0.10"),
        spend=Decimal("0"),
        spot=False,
        mrl_seconds=72 * 3600,
    )


def test_scoped_new_server_and_rate_round_trip_and_watch(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    quote_path = tmp_path / "quote.json"
    document = _write_scoped_quote(quote_path, now=now)
    admission = _scoped_admission(tmp_path, quote_path, now=now)
    assert admission.scope["server_id"] == 222333
    assert admission.scope["hoster_id"] == 777
    assert admission.scope["price_per_day"] == "8"
    assert admission.scope["currency"] == "USD-Blockchain"
    assert admission.projection["total_60m"] == "0.55"
    assert admission.verification["provider_billing_verified"] is False
    assert admission.verification["tax_fx_verified"] is False
    assert admission.verification["minimum_billing_verified"] is False
    assert admission.permission_id is not None
    assert owner_risk.permission_id_for_admission(admission) == admission.permission_id
    assert not (tmp_path / "registry" / "claims").exists()

    quote = cs.CloreQuote.from_dict(document["quote"])
    state = cs.initialize_state(
        tmp_path / "state.json",
        order_id=12345,
        server_id=222333,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        owner_risk_admission=admission,
        now=now,
    )
    assert state.server_id == 222333
    assert state.currency == "USD-Blockchain"
    assert state.quote.price_per_day == Decimal("8")
    binding = v3_lifecycle.SessionBinding.from_state(state)
    assert binding.server_id == 222333
    assert binding.currency == "USD-Blockchain"
    assert binding.owner_risk_admission_sha256 == state.owner_risk_sha256
    loaded = cs.load_state(tmp_path / "state.json", now=now)
    assert loaded.server_id == 222333
    assert loaded.quote.price_per_day == Decimal("8")

    bridge = _FakeBridge(_order(loaded, "8"))
    session = cs.CloreSession(tmp_path / "state.json", bridge=bridge, clock=lambda: now)
    assert asyncio.run(session.evaluate()).status == "ARMED"
    outcome = asyncio.run(session.stop_now(reason="scoped test stop"))
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancelled == [state.order_id]


def test_two_scoped_proposals_share_one_root_permission_and_only_one_binds(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    quote_one = tmp_path / "quote-one.json"
    quote_two = tmp_path / "quote-two.json"
    _write_scoped_quote(quote_one, now=now, price_per_day="8")
    _write_scoped_quote(quote_two, now=now, price_per_day="9")
    first = _scoped_admission(tmp_path, quote_one, now=now, name="admission-one.json")
    second = _scoped_admission(tmp_path, quote_two, now=now, name="admission-two.json")
    assert first.permission_id == second.permission_id
    assert first.admission_id != second.admission_id
    assert not (tmp_path / "registry" / "claims").exists()

    bound = v3_operator.bind_session(
        session_state=tmp_path / "state-one.json",
        quote=quote_one,
        order_id=13001,
        created_at=int(now.timestamp()) - 5,
        owner_risk_admission=tmp_path / "admission-one.json",
        now=now,
    )
    assert bound["status"] == "SESSION_BOUND"
    claim_dir = tmp_path / "registry" / "claims"
    assert (claim_dir / f"{first.permission_id}.json").is_file()
    assert (claim_dir / f"{first.permission_id}.reserve").is_file()

    with pytest.raises(v3_operator.OperatorError):
        v3_operator.bind_session(
            session_state=tmp_path / "state-two.json",
            quote=quote_two,
            order_id=13002,
            created_at=int(now.timestamp()) - 4,
            owner_risk_admission=tmp_path / "admission-two.json",
            now=now,
        )
    assert not (tmp_path / "state-two.json").exists()

    third_quote = tmp_path / "quote-three.json"
    _write_scoped_quote(third_quote, now=now, price_per_day="7")
    _scoped_admission(tmp_path, third_quote, now=now, name="admission-three.json")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.bind_session(
            session_state=tmp_path / "state-three.json",
            quote=third_quote,
            order_id=13003,
            created_at=int(now.timestamp()) - 3,
            owner_risk_admission=tmp_path / "admission-three.json",
            now=now,
        )
    assert not (tmp_path / "state-three.json").exists()
    assert len(list(claim_dir.glob("*.json"))) == 1


def test_scoped_price_over_cap_is_refused_without_lowering_or_artifact(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = tmp_path / "quote.json"
    document = _write_scoped_quote(quote_path, now=now, price_per_day="30")
    assert document["proposal"]["fits_code_cap"] is False
    with pytest.raises(owner_risk.OwnerRiskError) as excinfo:
        _scoped_admission(tmp_path, quote_path, now=now)
    assert "cap" in str(excinfo.value).lower()
    assert not (tmp_path / "admission.json").exists()
    assert not (tmp_path / "registry" / "claims").exists()


def test_non_usd_scoped_currency_is_refused_even_when_consistent(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = tmp_path / "btc-quote.json"
    document = _write_scoped_quote(
        quote_path, now=now, price_per_day="0.0002", currency="BTC"
    )
    assert document["currency"] == "BTC"
    assert document["quote"]["currency"] == "BTC"
    assert document["proposal"]["currency"] == "BTC"
    with pytest.raises(owner_risk.OwnerRiskError) as excinfo:
        _scoped_admission(tmp_path, quote_path, now=now, currency="BTC")
    assert "usd-blockchain" in str(excinfo.value).lower()
    assert not (tmp_path / "admission.json").exists()
    assert not (tmp_path / "registry" / "claims").exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document.__setitem__("observation_origin", "manual_snapshot"),
        lambda document: document["provenance"].__setitem__(
            "observation_origin", "manual_snapshot"
        ),
        lambda document: document["provenance"].__setitem__(
            "kind", "owner_forwarded_official_response"
        ),
        lambda document: document["provenance"].__setitem__(
            "base_url", "https://example.invalid/v1/"
        ),
        lambda document: document["provenance"].__setitem__("method", "POST"),
        lambda document: document["provenance"].__setitem__("endpoint", "my_orders"),
        lambda document: document["provenance"].__setitem__("path", "/v1/my_orders"),
    ],
)
def test_scoped_quote_rejects_manual_or_wrong_collector_contract_even_with_hash(
    tmp_path: Path, mutate
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = tmp_path / "quote.json"
    document = _write_scoped_quote(quote_path, now=now)
    mutate(document)
    _rehash(document)
    quote_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        _scoped_admission(tmp_path, quote_path, now=now)
    assert not (tmp_path / "admission.json").exists()
    assert not (tmp_path / "registry" / "claims").exists()


def test_scoped_quote_hash_covers_quote_and_proposal_crossfields(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    quote_path = tmp_path / "quote.json"
    document = _write_scoped_quote(quote_path, now=now)
    document["quote"]["price_per_day"] = "8.5"
    quote_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        _scoped_admission(tmp_path, quote_path, now=now)

    document = _write_scoped_quote(quote_path, now=now)
    document["quote"]["price_per_day"] = "8.5"
    _rehash(document)
    quote_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        _scoped_admission(tmp_path, quote_path, now=now)

    document = _write_scoped_quote(quote_path, now=now)
    document["proposal"]["server_id"] = 999999
    _rehash(document)
    quote_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        _scoped_admission(tmp_path, quote_path, now=now)
    assert not (tmp_path / "admission.json").exists()
    assert not (tmp_path / "registry" / "claims").exists()


def test_scoped_quote_rejects_noncanonical_or_inconsistent_timestamps(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    quote_path = tmp_path / "quote.json"
    document = _write_scoped_quote(quote_path, now=now)
    document["requested_at"] = "2026-10-27T12:00:01+00:00"
    document["observed_at"] = "2026-10-27T12:00:00Z"
    _rehash(document)
    quote_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        _scoped_admission(tmp_path, quote_path, now=now)

    document = _write_scoped_quote(quote_path, now=now)
    document["observed_at"] = "2026-10-27T12:05:00Z"
    _rehash(document)
    quote_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        _scoped_admission(tmp_path, quote_path, now=now)
    assert not (tmp_path / "admission.json").exists()
    assert not (tmp_path / "registry" / "claims").exists()


def test_scoped_approval_requires_explicit_confirmation(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    quote_path = tmp_path / "quote.json"
    _write_scoped_quote(quote_path, now=now)
    with pytest.raises(owner_risk.OwnerRiskError):
        owner_risk.create_scoped_admission(
            quote=quote_path,
            output_path=tmp_path / "admission.json",
            server_id=222333,
            hoster_id=777,
            approval_confirm=None,
            now=now,
        )
    assert not (tmp_path / "admission.json").exists()


def test_legacy_admission_cannot_transfer_to_new_server_or_quote(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    legacy_quote = tmp_path / "legacy-quote.json"
    _write_legacy_quote(legacy_quote, now=now)
    legacy = owner_risk.create_admission(
        quote=legacy_quote,
        output_path=tmp_path / "legacy-admission.json",
        now=now,
    )
    assert legacy.scope["server_id"] == 111947

    new_quote = tmp_path / "new-quote.json"
    _write_scoped_quote(new_quote, now=now, price_per_day="8")
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.bind_session(
            session_state=tmp_path / "new-state.json",
            quote=new_quote,
            order_id=14001,
            created_at=int(now.timestamp()) - 5,
            owner_risk_admission=tmp_path / "legacy-admission.json",
            server_id=222333,
            now=now,
        )
    assert not (tmp_path / "new-state.json").exists()
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.bind_session(
            session_state=tmp_path / "legacy-new-server.json",
            quote=legacy_quote,
            order_id=14002,
            created_at=int(now.timestamp()) - 5,
            owner_risk_admission=tmp_path / "legacy-admission.json",
            server_id=222333,
            now=now,
        )
    assert not (tmp_path / "legacy-new-server.json").exists()


def test_scoped_approval_tamper_and_quote_change_fail_closed(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    quote_path = tmp_path / "quote.json"
    _write_scoped_quote(quote_path, now=now)
    _scoped_admission(tmp_path, quote_path, now=now)
    original = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    original["approval"]["explicit"] = False
    tampered = tmp_path / "tampered-approval.json"
    tampered.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        owner_risk.load_admission(tampered, now=now)

    original = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    original["quote"]["sha256"] = "0" * 64
    tampered = tmp_path / "tampered-quote.json"
    tampered.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(owner_risk.OwnerRiskError):
        owner_risk.load_admission(tampered, now=now)


def test_owner_cli_fresh_quote_and_scoped_approval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    quote_path = tmp_path / "quote.json"
    assert (
        v3_operator.main(
            [
                "fresh-quote",
                "--server-id",
                "222333",
                "--output",
                str(quote_path),
            ]
        )
        == 2
    )
    assert not quote_path.exists()
    refused = json.loads(capsys.readouterr().out)
    assert refused["status"] == "REFUSED"

    now = datetime.now(timezone.utc)
    _write_scoped_quote(quote_path, now=now)
    admission_path = tmp_path / "admission.json"
    assert (
        v3_operator.main(
            [
                "owner-risk-approve",
                "--quote",
                str(quote_path),
                "--output",
                str(admission_path),
                "--server-id",
                "222333",
                "--confirm-approval",
                owner_risk.APPROVAL_CONFIRMATION,
                "--expected-quote-sha256",
                hashlib.sha256(quote_path.read_bytes()).hexdigest(),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "OWNER_ACCEPTED_FINANCIAL_RISK"
    assert payload["server_id"] == 222333
    assert payload["price_per_day"] == "8"
    assert payload["projection"]["total_60m"] == "0.55"
    assert payload["verification"]["provider_billing_verified"] is False
    assert admission_path.is_file()

    second = tmp_path / "second-admission.json"
    assert (
        v3_operator.main(
            [
                "owner-risk-approve",
                "--quote",
                str(quote_path),
                "--output",
                str(second),
                "--server-id",
                "222333",
                "--confirm-approval",
                "WRONG",
                "--json",
            ]
        )
        == 2
    )
    assert not second.exists()
