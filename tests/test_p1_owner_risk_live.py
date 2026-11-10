from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.p1 import clore_session as cs
from scripts.p1 import marketplace_quote, owner_risk, v3_lifecycle, v3_operator, v3_readiness
from scripts.p1.clore_bridge import CloreCancelAck, CloreOrder
from tests.comfy_stub import run_stub
from tests.helpers import db_scalar
from tests.p1_live_candidate_helpers import install_live_candidate
from tests.test_p1_v3_live_coldstart import _fake_supervisor_start, _trusted_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_owner_risk_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(owner_risk, "DEFAULT_REGISTRY_ROOT", tmp_path / "registry")


def _write_quote(
    path: Path,
    now: datetime,
    *,
    server_id: int = 111947,
    hoster_id: int = 46993,
    price_per_day: str = "12",
    mrl_hours: int = 336,
) -> Path:
    facts = {
        "server_id": server_id,
        "hoster_id": hoster_id,
        "mrl_raw_hours": mrl_hours,
        "mrl_raw_seconds": mrl_hours * 3600,
        "mrl_semantics": "MAXIMUM_RENTAL_LENGTH",
        "currency": "USD-Blockchain",
        "price_per_day": price_per_day,
        "availability": {
            "listed": True,
            "rented": False,
            "online": None,
            "online_source": "not_exposed_by_marketplace",
        },
    }
    if server_id == 111947 and hoster_id == 46993:
        payload = {
            "schema": marketplace_quote.QUOTE_SCHEMA,
            "status": marketplace_quote.STATUS,
            "authorizes_session": False,
            "provider_billing_verified": False,
            "observed_at": now.astimezone(timezone.utc).isoformat(),
            "requested_at": now.astimezone(timezone.utc).isoformat(),
            "server_id": server_id,
            "hoster_id": hoster_id,
            "currency": "USD-Blockchain",
            "price_per_day": price_per_day,
            "mrl_raw_hours": mrl_hours,
            "mrl_raw_seconds": mrl_hours * 3600,
            "mrl_semantics": "MAXIMUM_RENTAL_LENGTH",
            "availability": facts["availability"],
            "observation_origin": "owner_forwarded_official_response_not_direct_account_audit",
            "provenance": {
                "kind": "owner_forwarded_official_response",
                "not_provider_api_audit": True,
            },
            "quote": {
                "price_per_day": price_per_day,
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
    else:
        payload = marketplace_quote.build_quote_document(
            facts=facts, requested_at=now, observed_at=now
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class _StopBridge:
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


@pytest.mark.parametrize(
    "server_id,price_per_day,hoster_id",
    [(111947, "12", None), (222333, "8", 777)],
    ids=["legacy-fixed", "scoped-new-server"],
)
def test_seeded_generation_ready_owner_risk_one_post_collect_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server_id: int,
    price_per_day: str,
    hoster_id: int | None,
) -> None:
    # This test deliberately starts at an already-bound READY lifecycle to keep
    # the owner-risk path focused. The full prepare/probe/candidate flow is
    # covered by test_p1_v3_two_phase_integration; it is not duplicated here.
    # Before a candidate exists, source/readiness may be admitted by owner risk
    # but the existing backend must not create a jobs DB or POST.
    now = datetime.now(timezone.utc)
    quote_path = _write_quote(
        tmp_path / "quote.json",
        now,
        server_id=server_id,
        hoster_id=hoster_id or 46993,
        price_per_day=price_per_day,
    )
    admission_path = tmp_path / "admission.json"
    if hoster_id is None:
        admission = owner_risk.create_admission(
            quote=quote_path, output_path=admission_path, now=now
        )
    else:
        admission = owner_risk.create_scoped_admission(
            quote=quote_path,
            output_path=admission_path,
            server_id=server_id,
            hoster_id=hoster_id,
            approval_confirm=owner_risk.APPROVAL_CONFIRMATION,
            now=now,
        )
    quote = cs.CloreQuote.from_dict(
        json.loads(quote_path.read_text(encoding="utf-8"))["quote"]
    )
    state_path = tmp_path / "session.json"
    state = cs.initialize_state(
        state_path,
        order_id=98001,
        server_id=server_id,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        owner_risk_admission=admission,
        now=now,
    )
    binding = v3_lifecycle.SessionBinding.from_state(state)
    bundle = dict(v3_operator.load_bundle())
    source_hashes = v3_operator._bundle_hashes(bundle)
    run = tmp_path / "run"
    run.mkdir()
    v3_operator._ensure_run_document(run, source_hashes)
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=run.name,
        session=binding,
        bundle_hashes=source_hashes,
        phase="READY",
    )
    v3_lifecycle.write_lifecycle(run / "lifecycle.json", lifecycle)
    remote, fingerprint = _trusted_identity(tmp_path)
    pre_approval = v3_readiness.create_source_approval(
        bundle=bundle,
        source_hashes=source_hashes,
        output_path=run / "source-approval.json",
    )
    pre_observation = v3_readiness.create_quote_observation(
        quote_path=quote_path,
        session_state=state_path,
        binding=binding,
    )
    pre_host = v3_readiness.make_host_key_fact(remote, fingerprint)
    pre_gate = v3_readiness.evaluate_source_gate(
        bundle=bundle,
        source_hashes=source_hashes,
        approval=pre_approval,
        quote_observation=pre_observation,
        session_state=state,
        binding=binding,
        host_key_fact=pre_host,
        remote=remote,
        expected_fingerprint=fingerprint,
    )
    assert pre_gate.passed, pre_gate.issues
    with pytest.raises(v3_operator.ReadinessBlocked):
        v3_operator.run_once(
            live=True,
            session_state=state_path,
            order_id=state.order_id,
            created_at=state.created_at,
            remote=remote,
            expected_fingerprint=fingerprint,
            run_dir=run,
            ready_validator=lambda **kwargs: None,
        )
    assert not (run / "jobs.sqlite3").exists()

    with run_stub() as (base_url, stub):
        original_candidate_config = v3_readiness._candidate_config

        def candidate_config_with_stub(*, live_dir: Path, source_hashes: dict[str, str]):
            config = original_candidate_config(
                live_dir=live_dir, source_hashes=source_hashes
            )
            config["runtime"]["base_url"] = base_url
            return config

        monkeypatch.setattr(
            v3_readiness, "_candidate_config", candidate_config_with_stub
        )
        installed = install_live_candidate(
            run=run,
            session_state=state_path,
            quote_path=quote_path,
            remote=remote,
            fingerprint=fingerprint,
        )
        assert (
            installed["binding"].owner_risk_admission_sha256
            == state.owner_risk_sha256
        )
        assert (
            installed["candidate"].binding["session"][
                "owner_risk_admission_sha256"
            ]
            == state.owner_risk_sha256
        )
        source_gate = v3_readiness.evaluate_source_gate(
            bundle=bundle,
            source_hashes=source_hashes,
            approval=installed["approval"],
            quote_observation=installed["observation"],
            session_state=state,
            binding=binding,
            host_key_fact=installed["host_key_fact"],
            remote=remote,
            expected_fingerprint=fingerprint,
        )
        assert source_gate.passed, source_gate.issues
        assert source_gate.facts["financial_status"] == "OWNER_ACCEPTED_FINANCIAL_RISK"
        stub.set_success("result.mp4")
        stub.history[stub.prompt_id]["outputs"] = {
            "58": {
                "images": [
                    {"filename": "result.mp4", "subfolder": "", "type": "output"}
                ]
            }
        }
        monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
        monkeypatch.setattr(v3_lifecycle, "_pid_live", lambda pid: True)
        monkeypatch.setattr(v3_lifecycle.CloreSupervisor, "start", _fake_supervisor_start)
        monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
        monkeypatch.setattr(v3_operator, "_produce_remote_export", lambda **kwargs: None)
        result = v3_operator.run_once(
            live=True,
            session_state=state_path,
            order_id=state.order_id,
            created_at=state.created_at,
            remote=remote,
            expected_fingerprint=fingerprint,
            run_dir=run,
            ready_validator=lambda **kwargs: None,
        )
        assert result["status"] == "succeeded", result
        assert stub.prompt_requests == 1
        assert db_scalar(run / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 1
        assert json.loads((run / "run.json").read_text(encoding="utf-8"))[
            "financial_status"
        ] == "OWNER_ACCEPTED_FINANCIAL_RISK"
        second = v3_operator.run_once(
            live=True,
            session_state=state_path,
            order_id=state.order_id,
            created_at=state.created_at,
            remote=remote,
            expected_fingerprint=fingerprint,
            run_dir=run,
            ready_validator=lambda **kwargs: None,
        )
        assert second["status"] == "succeeded", second
        assert stub.prompt_requests == 1
        assert db_scalar(run / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 1
        supervisor_config = json.loads(
            (run / "supervisor-config.json").read_text(encoding="utf-8")
        )
        assert (
            supervisor_config["session"]["owner_risk_admission_sha256"]
            == state.owner_risk_sha256
        )
        collected = v3_operator.collect(run_dir=run)
        assert collected["status"].startswith("COLLECTED")
        candidates = []
        for candidate in run.rglob("runtime.json"):
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("schema") == "anime-video-studio/p1-v3-runtime/v1":
                candidates.append(payload)
        assert candidates
        assert candidates[-1]["financial_status"] == "OWNER_ACCEPTED_FINANCIAL_RISK"
        assert candidates[-1]["provider_billing_verified"] is False

    order = CloreOrder(
        order_id=state.order_id,
        server_id=state.server_id,
        created_at=state.created_at,
        price_per_day=Decimal(price_per_day),
        currency="USD-Blockchain",
        expired=False,
        fee_percent=Decimal("5"),
        creation_fee=Decimal("0.10"),
        spend=Decimal("0"),
        spot=False,
        mrl_seconds=336 * 3600,
    )
    bridge = _StopBridge(order)
    stop_session = cs.CloreSession(state_path, bridge=bridge)
    outcome = asyncio.run(stop_session.stop_now())
    assert outcome.status == "ENDED_REPORTED"
    assert bridge.cancelled == [state.order_id]
