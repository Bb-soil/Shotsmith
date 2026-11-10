from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.p1 import clore_session as cs
from scripts.p1 import marketplace_quote, owner_risk, v3_lifecycle
from scripts.p1.clore_bridge import CloreCancelAck, CloreOrder

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "p1" / "legacy-owner-risk"
IDS = json.loads((FIXTURE / "legacy-ids.json").read_text(encoding="utf-8"))
ADMISSION_ID = IDS["admission_id"]
NOW = datetime(2026, 10, 27, 6, 1, tzinfo=timezone.utc)
LEGACY_STATE = ROOT / "data" / "p1-legacy-fixture" / "state.json"


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(owner_risk, "DEFAULT_REGISTRY_ROOT", tmp_path / "registry")


@pytest.fixture
def legacy_state_path():
    LEGACY_STATE.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_STATE.write_bytes((FIXTURE / "legacy-state.json").read_bytes())
    try:
        yield LEGACY_STATE
    finally:
        shutil.rmtree(LEGACY_STATE.parent, ignore_errors=True)


def _install_claim(tmp_path: Path, *, status: str) -> Path:
    claims = tmp_path / "registry" / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    suffix = ".json" if status == "BOUND" else ".reserve"
    target = claims / f"{ADMISSION_ID}{suffix}"
    shutil.copyfile(FIXTURE / f"{ADMISSION_ID}{suffix}", target)
    return target


def _install_pair(tmp_path: Path) -> tuple[Path, Path]:
    claims = tmp_path / "registry" / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    bound = claims / f"{ADMISSION_ID}.json"
    reserve = claims / f"{ADMISSION_ID}.reserve"
    shutil.copyfile(FIXTURE / f"{ADMISSION_ID}.json", bound)
    shutil.copyfile(FIXTURE / f"{ADMISSION_ID}.reserve", reserve)
    return bound, reserve


def _load_admission():
    return owner_risk.load_admission(FIXTURE / "legacy-admission.json", now=NOW)


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


def _new_scoped_quote(
    tmp_path: Path, *, price: str = "9.6", name: str = "new-scoped-quote.json"
) -> Path:
    facts = {
        "server_id": 120777,
        "hoster_id": 47111,
        "mrl_raw_hours": 72,
        "mrl_raw_seconds": 72 * 3600,
        "mrl_semantics": "MAXIMUM_RENTAL_LENGTH",
        "currency": "USD-Blockchain",
        "price_per_day": price,
        "availability": {
            "listed": True,
            "rented": False,
            "online": None,
            "online_source": "not_exposed_by_marketplace",
        },
    }
    document = marketplace_quote.build_quote_document(
        facts=facts, requested_at=NOW, observed_at=NOW
    )
    path = tmp_path / name
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def test_legacy_bound_claim_is_visible_and_exact_recovery_load_works(
    tmp_path: Path, legacy_state_path: Path
) -> None:
    _install_claim(tmp_path, status="BOUND")
    admission = _load_admission()
    assert admission.admission_id == ADMISSION_ID
    assert admission.permission_id is None

    state = cs.load_state(legacy_state_path, safety_mode=True, now=NOW)
    assert state.order_id == IDS["order_id"]
    assert state.owner_risk_sha256 == IDS["admission_sha256"]
    binding = v3_lifecycle.SessionBinding.from_state(state)
    assert binding.owner_risk_admission_sha256 == IDS["admission_sha256"]
    assert cs.load_state(legacy_state_path, now=NOW).order_id == state.order_id

    same = cs.initialize_state(
        legacy_state_path,
        order_id=state.order_id,
        server_id=state.server_id,
        created_at=state.created_at,
        quote=state.quote,
        owner_risk_admission=admission,
        now=NOW,
    )
    assert same.owner_risk_sha256 == IDS["admission_sha256"]
    permission_id = owner_risk.permission_id_for_admission(admission)
    claims = tmp_path / "registry" / "claims"
    assert not (claims / f"{permission_id}.json").exists()
    assert not (claims / f"{permission_id}.reserve").exists()
    assert (claims / f"{ADMISSION_ID}.json").is_file()


def test_legacy_bound_claim_blocks_second_order_and_copied_state(
    tmp_path: Path, legacy_state_path: Path
) -> None:
    _install_claim(tmp_path, status="BOUND")
    admission = _load_admission()
    state = cs.load_state(legacy_state_path, safety_mode=True, now=NOW)
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "second-state.json",
            order_id=state.order_id + 1,
            server_id=state.server_id,
            created_at=state.created_at + 1,
            quote=state.quote,
            owner_risk_admission=admission,
            now=NOW,
        )
    assert not (tmp_path / "second-state.json").exists()

    copied = tmp_path / "copied-state.json"
    shutil.copyfile(legacy_state_path, copied)
    with pytest.raises(cs.CloreStateError):
        cs.load_state(copied, safety_mode=True, now=NOW)


def test_legacy_bound_claim_blocks_new_scoped_proposal(
    tmp_path: Path, legacy_state_path: Path
) -> None:
    _install_claim(tmp_path, status="BOUND")
    quote = _new_scoped_quote(tmp_path)
    scoped = owner_risk.create_scoped_admission(
        quote=quote,
        output_path=tmp_path / "scoped-admission.json",
        server_id=120777,
        hoster_id=47111,
        approval_confirm=owner_risk.APPROVAL_CONFIRMATION,
        now=NOW,
    )
    scoped_quote = cs.CloreQuote.from_dict(
        json.loads(quote.read_text(encoding="utf-8"))["quote"]
    )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "scoped-state.json",
            order_id=99001,
            server_id=120777,
            created_at=int(NOW.timestamp()) - 5,
            quote=scoped_quote,
            owner_risk_admission=scoped,
            now=NOW,
        )
    assert not (tmp_path / "scoped-state.json").exists()
    claims = tmp_path / "registry" / "claims"
    assert not (claims / f"{owner_risk.permission_id_for_admission(scoped)}.json").exists()
    assert not (claims / f"{owner_risk.permission_id_for_admission(scoped)}.reserve").exists()

    second_quote = _new_scoped_quote(
        tmp_path, price="8.5", name="new-scoped-quote-2.json"
    )
    second_scoped = owner_risk.create_scoped_admission(
        quote=second_quote,
        output_path=tmp_path / "scoped-admission-2.json",
        server_id=120777,
        hoster_id=47111,
        approval_confirm=owner_risk.APPROVAL_CONFIRMATION,
        now=NOW,
    )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "scoped-state-2.json",
            order_id=99002,
            server_id=120777,
            created_at=int(NOW.timestamp()) - 4,
            quote=cs.CloreQuote.from_dict(
                json.loads(second_quote.read_text(encoding="utf-8"))["quote"]
            ),
            owner_risk_admission=second_scoped,
            now=NOW,
        )
    assert not (tmp_path / "scoped-state-2.json").exists()
    assert not (claims / f"{owner_risk.permission_id_for_admission(second_scoped)}.json").exists()
    assert not (claims / f"{owner_risk.permission_id_for_admission(second_scoped)}.reserve").exists()


def test_legacy_bound_reserved_pair_is_one_transaction_and_exact_recovery_works(
    tmp_path: Path, legacy_state_path: Path
) -> None:
    bound, reserve = _install_pair(tmp_path)
    assert bound.is_file() and reserve.is_file()
    admission = _load_admission()
    state = cs.load_state(legacy_state_path, safety_mode=True, now=NOW)
    assert state.order_id == IDS["order_id"]
    assert cs.load_state(legacy_state_path, now=NOW).owner_risk_sha256 == IDS["admission_sha256"]

    same = cs.initialize_state(
        legacy_state_path,
        order_id=state.order_id,
        server_id=state.server_id,
        created_at=state.created_at,
        quote=state.quote,
        owner_risk_admission=admission,
        now=NOW,
    )
    assert same.owner_risk_sha256 == IDS["admission_sha256"]
    assert same.cap_usd == state.cap_usd
    assert same.hard_end_at == state.hard_end_at
    permission_id = owner_risk.permission_id_for_admission(admission)
    claims = tmp_path / "registry" / "claims"
    assert not (claims / f"{permission_id}.json").exists()
    assert not (claims / f"{permission_id}.reserve").exists()
    assert bound.is_file() and reserve.is_file()


def test_legacy_pair_after_expiry_safety_load_and_exact_cancel_readback(
    tmp_path: Path, legacy_state_path: Path
) -> None:
    _install_pair(tmp_path)
    expired_now = datetime(2026, 10, 27, 7, 15, tzinfo=timezone.utc)
    state = cs.load_state(legacy_state_path, safety_mode=True, now=expired_now)
    bridge = _FakeBridge(
        CloreOrder(
            order_id=state.order_id,
            server_id=state.server_id,
            created_at=state.created_at,
            price_per_day=Decimal("12.01"),
            currency=state.currency,
            expired=False,
            fee_percent=Decimal("5"),
            creation_fee=Decimal("0.10"),
            spend=Decimal("0"),
            spot=False,
            mrl_seconds=336 * 3600,
        )
    )
    session = cs.CloreSession(legacy_state_path, bridge=bridge, clock=lambda: expired_now)
    outcome = asyncio.run(session.evaluate())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancelled == [state.order_id]
    assert cs.load_state(legacy_state_path, safety_mode=True, now=expired_now).status == "ENDED_REPORTED"


def test_legacy_pair_second_order_new_proposal_and_copied_state_reject(
    tmp_path: Path, legacy_state_path: Path
) -> None:
    _install_pair(tmp_path)
    admission = _load_admission()
    state = cs.load_state(legacy_state_path, safety_mode=True, now=NOW)
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "second-state.json",
            order_id=state.order_id + 1,
            server_id=state.server_id,
            created_at=state.created_at + 1,
            quote=state.quote,
            owner_risk_admission=admission,
            now=NOW,
        )
    assert not (tmp_path / "second-state.json").exists()

    copied = tmp_path / "copied-state.json"
    shutil.copyfile(legacy_state_path, copied)
    with pytest.raises(cs.CloreStateError):
        cs.load_state(copied, safety_mode=True, now=NOW)

    quote = _new_scoped_quote(tmp_path)
    scoped = owner_risk.create_scoped_admission(
        quote=quote,
        output_path=tmp_path / "scoped-admission.json",
        server_id=120777,
        hoster_id=47111,
        approval_confirm=owner_risk.APPROVAL_CONFIRMATION,
        now=NOW,
    )
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            tmp_path / "scoped-state.json",
            order_id=99003,
            server_id=120777,
            created_at=int(NOW.timestamp()) - 5,
            quote=cs.CloreQuote.from_dict(
                json.loads(quote.read_text(encoding="utf-8"))["quote"]
            ),
            owner_risk_admission=scoped,
            now=NOW,
        )
    assert not (tmp_path / "scoped-state.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("admission_sha256", "0" * 64),
        ("quote_sha256", "0" * 64),
        ("order_id", IDS["order_id"] + 1),
        ("created_at", IDS["created_at"] + 1),
        ("state_path", "C:\\foreign-state.json"),
        ("state_sha256", "0" * 64),
        ("session_binding_sha256", "0" * 64),
    ],
)
def test_legacy_pair_conflicts_fail_closed(
    tmp_path: Path, legacy_state_path: Path, field: str, value
) -> None:
    _bound, reserve = _install_pair(tmp_path)
    payload = json.loads(reserve.read_text(encoding="utf-8"))
    payload[field] = value
    reserve.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(cs.CloreStateError):
        cs.load_state(legacy_state_path, safety_mode=True, now=NOW)
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            legacy_state_path,
            order_id=IDS["order_id"],
            server_id=IDS["server_id"],
            created_at=IDS["created_at"],
            quote=cs.CloreQuote.from_dict(
                json.loads((FIXTURE / "legacy-quote.json").read_text(encoding="utf-8"))["quote"]
            ),
            owner_risk_admission=_load_admission(),
            now=NOW,
        )


def test_legacy_reserved_claim_blocks_new_work_but_safety_load_works(
    tmp_path: Path, legacy_state_path: Path
) -> None:
    _install_claim(tmp_path, status="RESERVED")
    admission = _load_admission()
    state = cs.load_state(legacy_state_path, safety_mode=True, now=NOW)
    assert state.order_id == IDS["order_id"]
    with pytest.raises(cs.CloreStateError):
        cs.load_state(legacy_state_path, now=NOW)
    with pytest.raises(cs.CloreLaunchBlocked):
        cs.initialize_state(
            legacy_state_path,
            order_id=state.order_id,
            server_id=state.server_id,
            created_at=state.created_at,
            quote=state.quote,
            owner_risk_admission=admission,
            now=NOW,
        )
    claims = tmp_path / "registry" / "claims"
    assert not (claims / f"{owner_risk.permission_id_for_admission(admission)}.json").exists()
    assert not (claims / f"{owner_risk.permission_id_for_admission(admission)}.reserve").exists()
    assert (claims / f"{ADMISSION_ID}.reserve").is_file()
