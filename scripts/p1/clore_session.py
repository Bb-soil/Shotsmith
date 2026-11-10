"""Fail-closed Clore session deadline/budget/cancel controller.

This is an isolated Clore implementation. It does not import or modify the
RunPod controller/bridge, the application, the setup/preflight helpers, the
workflow, the manifest, or the accepted F1-F4 test harness.

The controller owns one immutable Clore order identity and one immutable
session budget. It can only read the exact order and cancel that exact order.
There is no create/extend/top-up/re-generate path.

States are deliberately explicit:

``ARMED``
    Identity and launch gates passed; no mutating call has been made.
``STOP_REQUESTED``
    Durable cancellation intent exists. A restart performs readback only.
``ENDED_REPORTED``
    The provider row reported ``expired=true``; this is provisional and
    billing remains unverified.
``LIVE_BLOCKED``
    No safe terminal proof exists. Manual My Orders -> exact ID -> Terminate is
    required.

``expired=true`` is never called a confirmed billing stop, and a missing row,
code 0, SSH disconnect, or provider error is never treated as ended.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Mapping, Protocol

# Allow ``python scripts/p1/clore_session.py`` as well as package/test imports.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from anime_studio.errors import InstanceAlreadyRunning
from anime_studio.lock import InstanceLock

from scripts.p1 import owner_risk

from scripts.p1.clore_bridge import (
    EXPECTED_CURRENCY,
    EXPECTED_SERVER_ID,
    CloreAuthError,
    CloreBridge,
    CloreBudgetError,
    CloreCancelAck,
    CloreCancelAmbiguous,
    CloreCurrencyMismatch,
    CloreError,
    CloreNotFound,
    CloreOrder,
    CloreProviderError,
    CloreProtocolError,
    CloreProviderError,
    CloreRateLimitError,
    CloreTransportError,
)

STATE_SCHEMA = "anime-video-studio/p1-clore-state/v1"
CAP_USD = Decimal("1.00")
MAX_DURATION_SECONDS = 60 * 60
PLANNED_CANCEL_OFFSET_SECONDS = 55 * 60
MAX_READBACK_SECONDS = 600.0
DEFAULT_READBACK_SECONDS = 120.0
DEFAULT_READBACK_POLL_SECONDS = 5.0
DEFAULT_WATCH_POLL_SECONDS = 30.0
STATUSES = frozenset({"ARMED", "STOP_REQUESTED", "ENDED_REPORTED", "LIVE_BLOCKED"})
_RENTER_FEE_RATE = Decimal("0.05")
_MIN_REQUEST_INTERVAL = 1.1


class CloreSessionError(CloreError):
    """Invalid session configuration or persisted state."""


class CloreLaunchBlocked(CloreSessionError):
    """Fresh quote/identity/budget inputs do not permit a launch."""

    def __init__(self, blockers: tuple[str, ...]) -> None:
        self.blockers = blockers
        super().__init__("; ".join(blockers) if blockers else "launch blocked")


class CloreStateError(CloreSessionError):
    """Persisted state is malformed, mutable, or belongs to another provider."""


class CloreSessionBusy(CloreSessionError):
    """Another controller operation is already in progress; fail fast."""


class CloreOrderSource(Protocol):
    async def read_order(
        self,
        order_id: int,
        *,
        deadline_monotonic: float | None = None,
    ) -> CloreOrder:
        ...

    async def cancel_order(
        self,
        order_id: int,
        *,
        deadline_monotonic: float | None = None,
    ) -> CloreCancelAck:
        ...


@dataclass(frozen=True)
class CloreQuote:
    """Inputs for a fresh, all-in quote projection.

    Optional values are allowed so an owner screenshot can be projected while
    still being explicitly not launch-ready. Launch readiness requires every
    field below to be present and verified.
    """

    price_per_day: Decimal | None = None
    price_per_hour: Decimal | None = None
    price_per_minute: Decimal | None = None
    currency: str = EXPECTED_CURRENCY
    creation_fee: Decimal | None = None
    mandatory_other_fees: Decimal | None = None
    reserve: Decimal | None = None
    fresh_verified: bool = False
    fees_known: bool = False
    tax_fx_known: bool = False
    min_rental_hours: Decimal | None = None
    provider_fee_percent: Decimal | None = None

    def __post_init__(self) -> None:
        for name in (
            "price_per_day",
            "price_per_hour",
            "price_per_minute",
            "creation_fee",
            "mandatory_other_fees",
            "reserve",
            "min_rental_hours",
            "provider_fee_percent",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Decimal)
                or not value.is_finite()
                or value < 0
            ):
                raise CloreSessionError(
                    f"quote.{name} must be a finite non-negative Decimal"
                )
        if not isinstance(self.currency, str):
            raise CloreSessionError("quote.currency must be a string")
        for name in ("fresh_verified", "fees_known", "tax_fx_known"):
            if not isinstance(getattr(self, name), bool):
                raise CloreSessionError(f"quote.{name} must be a boolean")
        if self.provider_fee_percent is not None and self.provider_fee_percent > Decimal(
            "100"
        ):
            raise CloreSessionError("quote.provider_fee_percent must be <= 100")

    def rental_per_minute(self) -> Decimal | None:
        projections: list[Decimal] = []
        if self.price_per_day is not None:
            projections.append(self.price_per_day / Decimal(1440))
        if self.price_per_hour is not None:
            projections.append(self.price_per_hour / Decimal(60))
        if self.price_per_minute is not None:
            projections.append(self.price_per_minute)
        return max(projections) if projections else None

    def project_total(self, minutes: int) -> Decimal | None:
        """Worst-case projection; returns None when a required input is absent.

        ``minutes`` is rounded up by the caller. Rental is the maximum of the
        available hourly/minute/day projections; the 5% renter base fee is
        applied once without assuming a PoH reduction. Creation fee is reserved
        only when the session crosses the documented >10 minute threshold.
        """

        if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 0:
            raise CloreSessionError("minutes must be a non-negative integer")
        if self.creation_fee is None or self.mandatory_other_fees is None or self.reserve is None:
            return None
        projections: list[Decimal] = []
        minute_decimal = Decimal(minutes)
        if self.price_per_day is not None:
            projections.append(self.price_per_day * minute_decimal / Decimal(1440))
        if self.price_per_hour is not None:
            projections.append(self.price_per_hour * minute_decimal / Decimal(60))
        if self.price_per_minute is not None:
            projections.append(self.price_per_minute * minute_decimal)
        if not projections:
            return None
        rental = max(projections)
        base_fee = rental * _RENTER_FEE_RATE
        creation = self.creation_fee if minutes > 10 else Decimal("0")
        return (
            rental
            + base_fee
            + creation
            + self.mandatory_other_fees
            + self.reserve
        )

    def launch_blockers(
        self,
        *,
        cap_usd: Decimal = CAP_USD,
        minutes: int = 60,
        admission: owner_risk.OwnerRiskAdmission | Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        """One policy entry: strict when no admission, scoped otherwise."""

        if admission is None:
            return owner_risk.strict_blockers(
                self, cap_usd=cap_usd, minutes=minutes
            )
        return owner_risk.policy_blockers(
            self,
            admission,
            cap_usd=cap_usd,
            minutes=minutes,
            now=now,
        )

    def to_dict(self) -> dict[str, Any]:
        def text(value: Decimal | None) -> str | None:
            return None if value is None else str(value)

        return {
            "price_per_day": text(self.price_per_day),
            "price_per_hour": text(self.price_per_hour),
            "price_per_minute": text(self.price_per_minute),
            "currency": self.currency,
            "creation_fee": text(self.creation_fee),
            "mandatory_other_fees": text(self.mandatory_other_fees),
            "reserve": text(self.reserve),
            "fresh_verified": self.fresh_verified,
            "fees_known": self.fees_known,
            "tax_fx_known": self.tax_fx_known,
            "min_rental_hours": text(self.min_rental_hours),
            "provider_fee_percent": text(self.provider_fee_percent),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CloreQuote":
        if not isinstance(data, Mapping):
            raise CloreSessionError("quote must be an object")
        allowed = {
            "price_per_day",
            "price_per_hour",
            "price_per_minute",
            "currency",
            "creation_fee",
            "mandatory_other_fees",
            "reserve",
            "fresh_verified",
            "fees_known",
            "tax_fx_known",
            "min_rental_hours",
            "provider_fee_percent",
        }
        unknown = set(data) - allowed
        if unknown:
            raise CloreSessionError(f"quote has unknown keys: {sorted(unknown)}")
        currency = data.get("currency", EXPECTED_CURRENCY)
        if not isinstance(currency, str):
            raise CloreSessionError("quote.currency must be a string")
        return cls(
            price_per_day=_optional_decimal(data.get("price_per_day"), "quote.price_per_day"),
            price_per_hour=_optional_decimal(data.get("price_per_hour"), "quote.price_per_hour"),
            price_per_minute=_optional_decimal(data.get("price_per_minute"), "quote.price_per_minute"),
            currency=currency,
            creation_fee=_optional_decimal(data.get("creation_fee"), "quote.creation_fee"),
            mandatory_other_fees=_optional_decimal(
                data.get("mandatory_other_fees"), "quote.mandatory_other_fees"
            ),
            reserve=_optional_decimal(data.get("reserve"), "quote.reserve"),
            fresh_verified=_required_bool(data.get("fresh_verified"), "quote.fresh_verified"),
            fees_known=_required_bool(data.get("fees_known"), "quote.fees_known"),
            tax_fx_known=_required_bool(data.get("tax_fx_known"), "quote.tax_fx_known"),
            min_rental_hours=_optional_decimal(
                data.get("min_rental_hours"), "quote.min_rental_hours"
            ),
            provider_fee_percent=_optional_decimal(
                data.get("provider_fee_percent"), "quote.provider_fee_percent"
            ),
        )


@dataclass(frozen=True)
class CloreSessionState:
    schema: str
    provider: str
    order_id: int
    server_id: int
    created_at: int
    currency: str
    quote: CloreQuote
    cap_usd: Decimal
    duration_seconds: int
    planned_cancel_at: int
    hard_end_at: int
    status: str
    updated_at: str
    cancel_intent_at: str | None = None
    cancel_reason: str | None = None
    readback_started_at: str | None = None
    readback_deadline_at: str | None = None
    last_error: str | None = None
    billing_verified: bool = False
    owner_risk: Mapping[str, Any] | None = None
    owner_risk_sha256: str | None = None


@dataclass(frozen=True)
class CloreSessionOutcome:
    status: str
    message: str
    state: CloreSessionState
    billing_verified: bool = False
    breach: str | None = None


_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


@contextmanager
def _state_lock(state_path: Path) -> Iterator[None]:
    """OS + in-process exclusion across one read/validate/mutate/write scope."""

    lock_path = Path(str(state_path) + ".lock")
    if lock_path.is_symlink():
        raise CloreStateError("state lock path must not be a symlink")
    key = str(state_path.resolve())
    with _LOCAL_LOCKS_GUARD:
        local_lock = _LOCAL_LOCKS.setdefault(key, threading.Lock())
    if not local_lock.acquire(blocking=False):
        raise CloreSessionBusy(
            "another Clore session operation is already in progress"
        )
    lock = None
    try:
        try:
            lock = InstanceLock(lock_path).acquire()
        except InstanceAlreadyRunning as exc:
            raise CloreSessionBusy(
                "another Clore session controller holds the state lock"
            ) from exc
        yield
    finally:
        if lock is not None:
            lock.release()
        local_lock.release()


def _optional_decimal(value: Any, name: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CloreSessionError(f"{name} must be a finite decimal or null")
    try:
        if isinstance(value, Decimal):
            number = value
        elif isinstance(value, int):
            number = Decimal(value)
        elif isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                raise CloreSessionError(f"{name} must be a finite decimal or null")
            number = Decimal(str(value))
        elif isinstance(value, str):
            if not value.strip():
                raise CloreSessionError(f"{name} must be a finite decimal or null")
            number = Decimal(value.strip())
        else:
            raise CloreSessionError(f"{name} must be a finite decimal or null")
    except (InvalidOperation, ValueError) as exc:
        raise CloreSessionError(f"{name} must be a finite decimal or null") from exc
    if not number.is_finite() or number < 0:
        raise CloreSessionError(f"{name} must be a finite non-negative decimal")
    return number


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CloreSessionError(f"{name} must be a boolean")
    return value


def _required_int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CloreSessionError(f"{name} must be an integer")
    if maximum is not None and value > maximum:
        raise CloreSessionError(f"{name} must not exceed {maximum}")
    return value


def _finite_positive(value: Any, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CloreSessionError(f"{name} must be a finite positive number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number <= 0:
        raise CloreSessionError(f"{name} must be a finite positive number")
    if maximum is not None and number > maximum:
        raise CloreSessionError(f"{name} must not exceed {maximum:g}")
    return number


def _coerce_utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise CloreStateError(f"{name} must be timezone-aware UTC")
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise CloreStateError(f"{name} must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise CloreStateError(f"{name} must include a UTC offset")
    else:
        raise CloreStateError(f"{name} must be an ISO-8601 UTC timestamp")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def binding_payload(state: CloreSessionState) -> dict[str, Any]:
    """Canonical session binding shared by lifecycle/readiness/claims."""

    quote = state.quote.to_dict()
    quote_hash = hashlib.sha256(
        json.dumps(
            quote, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "order_id": state.order_id,
        "server_id": state.server_id,
        "created_at": state.created_at,
        "currency": state.currency,
        "cap_usd": str(state.cap_usd),
        "duration_seconds": state.duration_seconds,
        "planned_cancel_at": state.planned_cancel_at,
        "hard_end_at": state.hard_end_at,
        "quote_hash": quote_hash,
    }
    if state.owner_risk_sha256 is not None:
        payload["owner_risk_admission_sha256"] = state.owner_risk_sha256
    return payload


def binding_digest(state: CloreSessionState) -> str:
    return hashlib.sha256(
        json.dumps(
            binding_payload(state),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _coerce_epoch(value: Any, name: str) -> int:
    return _required_int(value, name, minimum=1)


def _quote_to_dict(quote: CloreQuote) -> dict[str, Any]:
    return quote.to_dict()


def _quote_from_state(data: Any) -> CloreQuote:
    if not isinstance(data, Mapping):
        raise CloreStateError("state.quote must be an object")
    try:
        return CloreQuote.from_dict(data)
    except CloreSessionError as exc:
        raise CloreStateError(f"state.quote is invalid: {exc}") from exc


def _state_admission(
    state: CloreSessionState,
) -> owner_risk.OwnerRiskAdmission | None:
    if state.owner_risk is None:
        if state.owner_risk_sha256 is not None:
            raise CloreStateError("state owner_risk digest requires an owner_risk block")
        return None
    if not isinstance(state.owner_risk, Mapping):
        raise CloreStateError("state.owner_risk must be an object")
    if not isinstance(state.owner_risk_sha256, str) or len(state.owner_risk_sha256) != 64:
        raise CloreStateError("state.owner_risk_sha256 is missing/invalid")
    try:
        admission = owner_risk.OwnerRiskAdmission.from_dict(state.owner_risk)
        digest = owner_risk.admission_digest(admission)
    except owner_risk.OwnerRiskError as exc:
        raise CloreStateError(f"state.owner_risk is invalid: {exc}") from exc
    if digest != state.owner_risk_sha256.lower():
        raise CloreStateError("state owner_risk digest mismatch")
    return admission


def _state_to_dict(state: CloreSessionState) -> dict[str, Any]:
    payload = {
        "schema": state.schema,
        "provider": state.provider,
        "order_id": state.order_id,
        "server_id": state.server_id,
        "created_at": state.created_at,
        "currency": state.currency,
        "quote": _quote_to_dict(state.quote),
        "cap_usd": str(state.cap_usd),
        "duration_seconds": state.duration_seconds,
        "planned_cancel_at": state.planned_cancel_at,
        "hard_end_at": state.hard_end_at,
        "status": state.status,
        "updated_at": state.updated_at,
        "cancel_intent_at": state.cancel_intent_at,
        "cancel_reason": state.cancel_reason,
        "readback_started_at": state.readback_started_at,
        "readback_deadline_at": state.readback_deadline_at,
        "last_error": state.last_error,
        "billing_verified": state.billing_verified,
    }
    if state.owner_risk is not None or state.owner_risk_sha256 is not None:
        payload["owner_risk"] = (
            dict(state.owner_risk) if isinstance(state.owner_risk, Mapping) else None
        )
        payload["owner_risk_sha256"] = state.owner_risk_sha256
    return payload


def _validate_state(state: CloreSessionState) -> CloreSessionState:
    if state.schema != STATE_SCHEMA:
        raise CloreStateError(
            "state schema is not the Clore schema; RunPod state cannot be rebound"
        )
    if state.provider != "clore":
        raise CloreStateError("state provider must be clore")
    if state.order_id <= 0:
        raise CloreStateError("state order id must be positive")
    admission = _state_admission(state)
    expected_currency = EXPECTED_CURRENCY
    if admission is None:
        if state.server_id != EXPECTED_SERVER_ID:
            raise CloreStateError("state server id does not match the selected server")
    else:
        issues = owner_risk.validate_admission(admission, allow_expired=True)
        if issues:
            raise CloreStateError(
                "state owner_risk is invalid: " + "; ".join(issues)
            )
        if state.server_id != int(admission.scope["server_id"]):
            raise CloreStateError("state server id does not match the owner-risk admission")
        expected_currency = str(admission.scope["currency"])
    if state.created_at <= 0:
        raise CloreStateError("state creation time must be positive")
    if not isinstance(state.currency, str) or state.currency != expected_currency:
        raise CloreStateError("state currency does not match the scoped admission")
    if state.quote.currency != state.currency:
        raise CloreStateError("quote and state currency differ")
    if state.cap_usd <= 0 or state.cap_usd > CAP_USD:
        raise CloreStateError("state cap must be positive and at most USD 1.00")
    if not (0 < state.duration_seconds <= MAX_DURATION_SECONDS):
        raise CloreStateError("state duration must be at most 3600 seconds")
    if state.hard_end_at > state.created_at + MAX_DURATION_SECONDS:
        raise CloreStateError("state hard end exceeds creation + 60 minutes")
    if state.hard_end_at <= state.created_at:
        raise CloreStateError("state hard end must be after creation")
    if state.planned_cancel_at < state.created_at:
        raise CloreStateError("planned cancel must not precede creation")
    if state.planned_cancel_at > state.created_at + PLANNED_CANCEL_OFFSET_SECONDS:
        raise CloreStateError("planned cancel exceeds creation + 55 minutes")
    if state.planned_cancel_at > state.hard_end_at:
        raise CloreStateError("planned cancel must not exceed the hard end")
    if not isinstance(state.status, str) or state.status not in STATUSES:
        raise CloreStateError("state status is not recognized")
    if not isinstance(state.billing_verified, bool):
        raise CloreStateError("billing_verified must be a boolean")
    _coerce_utc(state.updated_at, "updated_at")
    if state.cancel_intent_at is not None:
        _coerce_utc(state.cancel_intent_at, "cancel_intent_at")
    if state.status == "STOP_REQUESTED" and state.cancel_intent_at is None:
        raise CloreStateError("STOP_REQUESTED state requires a durable cancel intent")
    if state.readback_started_at is None and state.readback_deadline_at is None:
        if state.status == "STOP_REQUESTED":
            raise CloreStateError("STOP_REQUESTED state requires a readback bound")
    elif state.status == "ARMED":
        raise CloreStateError("ARMED state must not carry readback anchors")
    elif state.readback_started_at is None or state.readback_deadline_at is None:
        raise CloreStateError("readback anchors must be present together")
    else:
        started = _coerce_utc(state.readback_started_at, "readback_started_at")
        deadline = _coerce_utc(state.readback_deadline_at, "readback_deadline_at")
        if deadline <= started:
            raise CloreStateError("readback deadline must be after its start")
        if deadline > datetime.fromtimestamp(
            state.created_at, tz=timezone.utc
        ) + timedelta(seconds=MAX_DURATION_SECONDS):
            raise CloreStateError(
                "readback deadline exceeds creation + 60 minutes"
            )
    if state.status == "ENDED_REPORTED" and state.billing_verified:
        raise CloreStateError("ended-reported state cannot claim billing verification")
    if state.status == "LIVE_BLOCKED" and not state.last_error:
        raise CloreStateError("LIVE_BLOCKED state requires a reason")
    if admission is None:
        blockers = state.quote.launch_blockers(cap_usd=state.cap_usd, minutes=60)
    else:
        blockers = owner_risk.policy_blockers(
            state.quote,
            admission,
            cap_usd=state.cap_usd,
            minutes=60,
            allow_expired=True,
        )
    if blockers:
        raise CloreStateError(
            "state quote is not launch-ready: " + "; ".join(blockers)
        )
    return state


def load_state(
    path: Path | str,
    *,
    safety_mode: bool = False,
    now: datetime | None = None,
    owner_risk_registry_root: Path | str | None = None,
) -> CloreSessionState:
    state_path = Path(path)
    if state_path.is_symlink():
        raise CloreStateError("state path must not be a symlink")
    try:
        data = json.loads(
            state_path.read_text(encoding="utf-8"), parse_float=Decimal
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloreStateError(f"cannot read Clore state: {type(exc).__name__}") from exc
    if not isinstance(data, Mapping):
        raise CloreStateError("state must be a JSON object")
    if data.get("schema") != STATE_SCHEMA:
        raise CloreStateError(
            "state schema is not the Clore schema; RunPod state cannot be rebound"
        )
    try:
        quote = _quote_from_state(data.get("quote"))
        cap = _optional_decimal(data.get("cap_usd"), "state.cap_usd")
        if cap is None:
            raise CloreStateError("state.cap_usd is missing")
        owner_risk_data = data.get("owner_risk")
        owner_risk_sha = data.get("owner_risk_sha256")
        if owner_risk_data is not None and not isinstance(owner_risk_data, Mapping):
            raise CloreStateError("state.owner_risk must be an object or null")
        if owner_risk_sha is not None and not isinstance(owner_risk_sha, str):
            raise CloreStateError("state.owner_risk_sha256 must be a string or null")
        state = CloreSessionState(
            schema=STATE_SCHEMA,
            provider=data.get("provider"),
            order_id=_required_int(data.get("order_id"), "state.order_id", minimum=1),
            server_id=_required_int(data.get("server_id"), "state.server_id", minimum=1),
            created_at=_coerce_epoch(data.get("created_at"), "state.created_at"),
            currency=data.get("currency"),
            quote=quote,
            cap_usd=cap,
            duration_seconds=_required_int(
                data.get("duration_seconds"), "state.duration_seconds", minimum=1
            ),
            planned_cancel_at=_coerce_epoch(
                data.get("planned_cancel_at"), "state.planned_cancel_at"
            ),
            hard_end_at=_coerce_epoch(data.get("hard_end_at"), "state.hard_end_at"),
            status=data.get("status"),
            updated_at=data.get("updated_at"),
            cancel_intent_at=data.get("cancel_intent_at"),
            cancel_reason=data.get("cancel_reason"),
            readback_started_at=data.get("readback_started_at"),
            readback_deadline_at=data.get("readback_deadline_at"),
            last_error=data.get("last_error"),
            billing_verified=data.get("billing_verified", False),
            owner_risk=dict(owner_risk_data) if owner_risk_data is not None else None,
            owner_risk_sha256=owner_risk_sha,
        )
    except CloreStateError:
        raise
    except (TypeError, ValueError, CloreSessionError) as exc:
        raise CloreStateError("state fields are invalid") from exc
    for field_name in ("updated_at",):
        if not isinstance(getattr(state, field_name), str):
            raise CloreStateError(f"state.{field_name} must be a string")
    if state.last_error is not None and not isinstance(state.last_error, str):
        raise CloreStateError("state.last_error must be a string or null")
    if state.cancel_reason is not None and not isinstance(state.cancel_reason, str):
        raise CloreStateError("state.cancel_reason must be a string or null")
    state = _validate_state(state)
    if state.owner_risk is not None:
        admission = _state_admission(state)
        assert admission is not None
        if not safety_mode:
            issues = owner_risk.validate_admission(
                admission, now=now, allow_expired=False
            )
            if issues:
                raise CloreStateError(
                    "state owner_risk is not valid for new work: "
                    + "; ".join(issues)
                )
        try:
            owner_risk.assert_bound_claim(
                admission,
                order_id=state.order_id,
                server_id=state.server_id,
                created_at=state.created_at,
                state_path=state_path,
                binding_sha256=binding_digest(state),
                allow_reserved=safety_mode,
                registry_root=owner_risk_registry_root,
            )
        except owner_risk.OwnerRiskError as exc:
            raise CloreStateError(
                f"state owner_risk claim is not valid: {exc}"
            ) from exc
    return state


def save_state(path: Path | str, state: CloreSessionState) -> None:
    _validate_state(state)
    state_path = Path(path)
    if state_path.is_symlink():
        raise CloreStateError("state path must not be a symlink")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_state_to_dict(state), indent=2, sort_keys=True) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=state_path.parent,
        prefix=state_path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp_name = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, state_path)
    finally:
        try:
            temp_name.unlink(missing_ok=True)
        except OSError:
            pass


def initialize_state(
    path: Path | str,
    *,
    order_id: int,
    server_id: int = EXPECTED_SERVER_ID,
    created_at: int,
    currency: str = EXPECTED_CURRENCY,
    quote: CloreQuote,
    cap_usd: Decimal = CAP_USD,
    duration_seconds: int = MAX_DURATION_SECONDS,
    planned_cancel_at: int | None = None,
    hard_end_at: int | None = None,
    now: datetime | None = None,
    owner_risk_admission: owner_risk.OwnerRiskAdmission | Mapping[str, Any] | None = None,
    owner_risk_registry_root: Path | str | None = None,
) -> CloreSessionState:
    """Create one Clore state, or return the identical persisted state.

    Launch readiness is checked before any file is created. Quote, identity,
    duration, and money can never be changed by a later initialization/restart.
    """

    state_path = Path(path)
    with _state_lock(state_path):
        return _initialize_unlocked(
            state_path,
            order_id=order_id,
            server_id=server_id,
            created_at=created_at,
            currency=currency,
            quote=quote,
            cap_usd=cap_usd,
            duration_seconds=duration_seconds,
            planned_cancel_at=planned_cancel_at,
            hard_end_at=hard_end_at,
            now=now,
            owner_risk_admission=owner_risk_admission,
            owner_risk_registry_root=owner_risk_registry_root,
        )


def _initialize_unlocked(
    path: Path | str,
    *,
    order_id: int,
    server_id: int,
    created_at: int,
    currency: str,
    quote: CloreQuote,
    cap_usd: Decimal,
    duration_seconds: int,
    planned_cancel_at: int | None,
    hard_end_at: int | None,
    now: datetime | None,
    owner_risk_admission: owner_risk.OwnerRiskAdmission | Mapping[str, Any] | None,
    owner_risk_registry_root: Path | str | None,
) -> CloreSessionState:
    order_id = _required_int(order_id, "order_id", minimum=1)
    server_id = _required_int(server_id, "server_id", minimum=1)
    clock = _coerce_utc(now, "now") if now is not None else datetime.now(timezone.utc)
    admission: owner_risk.OwnerRiskAdmission | None = None
    if owner_risk_admission is not None:
        try:
            admission = (
                owner_risk_admission
                if isinstance(owner_risk_admission, owner_risk.OwnerRiskAdmission)
                else owner_risk.OwnerRiskAdmission.from_dict(owner_risk_admission)
            )
        except owner_risk.OwnerRiskError as exc:
            raise CloreLaunchBlocked((f"owner-risk admission is invalid: {exc}",)) from exc
        issues = owner_risk.validate_admission(admission, now=clock)
        if issues:
            raise CloreLaunchBlocked(issues)
        if server_id != int(admission.scope["server_id"]):
            raise CloreLaunchBlocked(
                ("server id does not match the owner-risk admission",)
            )
    elif server_id != EXPECTED_SERVER_ID:
        raise CloreLaunchBlocked(("server id does not match the selected Clore server",))
    expected_currency = (
        str(admission.scope["currency"])
        if admission is not None
        else EXPECTED_CURRENCY
    )
    if currency != expected_currency:
        raise CloreLaunchBlocked(
            ("currency does not match the scoped admission",)
        )
    cap = _optional_decimal(cap_usd, "cap_usd")
    if cap is None or cap <= 0 or cap > CAP_USD:
        raise CloreLaunchBlocked(("session cap must be positive and at most USD 1.00",))
    if admission is not None and cap != CAP_USD:
        raise CloreLaunchBlocked(
            ("owner-risk session cap must be exactly USD 1.00",)
        )
    duration = _required_int(duration_seconds, "duration_seconds", minimum=1)
    if duration > MAX_DURATION_SECONDS:
        raise CloreLaunchBlocked(("session duration must not exceed 60 minutes",))
    if admission is not None and duration != MAX_DURATION_SECONDS:
        raise CloreLaunchBlocked(
            ("owner-risk session duration must be exactly 3600 seconds",)
        )
    created = _coerce_epoch(created_at, "created_at")
    planned = (
        created + PLANNED_CANCEL_OFFSET_SECONDS
        if planned_cancel_at is None
        else _coerce_epoch(planned_cancel_at, "planned_cancel_at")
    )
    hard_end = (
        created + duration
        if hard_end_at is None
        else _coerce_epoch(hard_end_at, "hard_end_at")
    )
    if planned < created or planned > created + PLANNED_CANCEL_OFFSET_SECONDS:
        raise CloreLaunchBlocked(
            ("planned cancel must be within creation + 55 minutes",)
        )
    if hard_end <= created or hard_end > created + MAX_DURATION_SECONDS:
        raise CloreLaunchBlocked(
            ("hard end must be after creation and within creation + 60 minutes",)
        )
    if admission is not None and hard_end != created + MAX_DURATION_SECONDS:
        raise CloreLaunchBlocked(
            ("owner-risk hard end must be exactly provider creation + 3600 seconds",)
        )
    if planned > hard_end:
        raise CloreLaunchBlocked(("planned cancel must not exceed the hard end",))
    if admission is None:
        blockers = quote.launch_blockers(cap_usd=cap, minutes=60)
    else:
        blockers = owner_risk.policy_blockers(
            quote, admission, cap_usd=cap, minutes=60, now=clock
        )
    if blockers:
        raise CloreLaunchBlocked(blockers)

    state_path = Path(path)
    state_exists = state_path.exists()
    if not state_exists and clock >= datetime.fromtimestamp(
        hard_end, tz=timezone.utc
    ):
        raise CloreLaunchBlocked(
            ("session hard end has already passed; refusing a new launch",)
        )
    owner_risk_block = admission.to_dict() if admission is not None else None
    owner_risk_sha = (
        owner_risk.admission_digest(admission) if admission is not None else None
    )
    reservation: owner_risk.ClaimReservation | None = None
    if admission is not None:
        if not state_exists:
            fresh = owner_risk.validate_fresh_quote(admission, now=clock)
            if fresh:
                raise CloreLaunchBlocked(fresh)
        try:
            reservation = owner_risk.reserve_admission(
                admission,
                order_id=order_id,
                created_at=created,
                state_path=state_path,
                registry_root=owner_risk_registry_root,
            )
        except owner_risk.OwnerRiskError as exc:
            raise CloreLaunchBlocked(
                (f"owner-risk claim refused: {exc}",)
            ) from exc
    if state_exists:
        existing = load_state(
            state_path,
            safety_mode=admission is not None,
            now=clock,
            owner_risk_registry_root=owner_risk_registry_root,
        )
        comparisons = {
            "order_id": existing.order_id,
            "server_id": existing.server_id,
            "created_at": existing.created_at,
            "currency": existing.currency,
            "quote": existing.quote,
            "cap_usd": existing.cap_usd,
            "duration_seconds": existing.duration_seconds,
            "planned_cancel_at": existing.planned_cancel_at,
            "hard_end_at": existing.hard_end_at,
            "owner_risk": existing.owner_risk,
            "owner_risk_sha256": existing.owner_risk_sha256,
        }
        requested = {
            "order_id": order_id,
            "server_id": server_id,
            "created_at": created,
            "currency": currency,
            "quote": quote,
            "cap_usd": cap,
            "duration_seconds": duration,
            "planned_cancel_at": planned,
            "hard_end_at": hard_end,
            "owner_risk": owner_risk_block,
            "owner_risk_sha256": owner_risk_sha,
        }
        if comparisons != requested:
            raise CloreStateError(
                "persisted Clore identity/deadline/quote/budget cannot be rebound or extended"
            )
        if reservation is not None and not reservation.bound:
            owner_risk.complete_claim(
                reservation,
                state_path=state_path,
                    state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                    session_binding_sha256=binding_digest(existing),
                )
        return existing

    state = CloreSessionState(
        schema=STATE_SCHEMA,
        provider="clore",
        order_id=order_id,
        server_id=server_id,
        created_at=created,
        currency=currency,
        quote=quote,
        cap_usd=cap,
        duration_seconds=duration,
        planned_cancel_at=planned,
        hard_end_at=hard_end,
        status="ARMED",
        updated_at=_format_utc(clock),
        owner_risk=owner_risk_block,
        owner_risk_sha256=owner_risk_sha,
    )
    _validate_state(state)
    save_state(state_path, state)
    if reservation is not None:
        owner_risk.complete_claim(
            reservation,
            state_path=state_path,
            state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
            session_binding_sha256=binding_digest(state),
        )
    return state


class CloreSession:
    def __init__(
        self,
        state_path: Path | str,
        *,
        bridge: CloreOrderSource,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        readback_timeout_seconds: float = DEFAULT_READBACK_SECONDS,
        readback_poll_seconds: float = DEFAULT_READBACK_POLL_SECONDS,
    ) -> None:
        self.state_path = Path(state_path)
        self.bridge = bridge
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self.readback_timeout_seconds = _finite_positive(
            readback_timeout_seconds,
            "readback_timeout_seconds",
            maximum=MAX_READBACK_SECONDS,
        )
        self.readback_poll_seconds = _finite_positive(
            readback_poll_seconds, "readback_poll_seconds"
        )

    def _now(self, value: datetime | None = None) -> datetime:
        return _coerce_utc(value if value is not None else self._clock(), "now")

    def _monotonic_deadline(
        self, deadline: datetime, *, now: datetime | None = None
    ) -> float:
        current = self._now(now)
        remaining = (deadline - current).total_seconds()
        if remaining <= 0:
            raise CloreBudgetError("persisted wall deadline already expired")
        return self._monotonic() + remaining

    def load(self) -> CloreSessionState:
        return load_state(
            self.state_path, safety_mode=True, now=self._now()
        )

    def initialize(self, **kwargs: Any) -> CloreSessionState:
        return initialize_state(self.state_path, **kwargs)

    def _blocked(self, state: CloreSessionState, message: str) -> CloreSessionOutcome:
        safe = " ".join(str(message).split())[:300]
        updated = replace(
            state,
            status="LIVE_BLOCKED",
            updated_at=_format_utc(self._now()),
            last_error=safe,
        )
        save_state(self.state_path, updated)
        return CloreSessionOutcome("LIVE_BLOCKED", safe, updated, False, state.cancel_reason)

    def _ended(self, state: CloreSessionState) -> CloreSessionOutcome:
        message = (
            "provider reported expired=true; terminal is provisional and billing "
            "is unverified"
        )
        updated = replace(
            state,
            status="ENDED_REPORTED",
            updated_at=_format_utc(self._now()),
            last_error=None,
            billing_verified=False,
        )
        save_state(self.state_path, updated)
        return CloreSessionOutcome(
            "ENDED_REPORTED", message, updated, False, state.cancel_reason
        )

    @staticmethod
    def _identity_error(order: CloreOrder, state: CloreSessionState) -> str | None:
        if order.order_id != state.order_id:
            return "provider order id does not match the persisted identity"
        if order.server_id != state.server_id:
            return "provider server id does not match the persisted identity"
        if order.created_at != state.created_at:
            return "provider creation time does not match the persisted identity"
        return None

    @staticmethod
    def _elapsed_minutes(state: CloreSessionState, now: datetime) -> int:
        elapsed_seconds = max(
            0,
            int(
                (
                    now - datetime.fromtimestamp(state.created_at, tz=timezone.utc)
                ).total_seconds()
            ),
        )
        return (elapsed_seconds + 59) // 60

    @staticmethod
    def _expected_price_per_day(state: CloreSessionState) -> Decimal | None:
        admission = _state_admission(state)
        return owner_risk.expected_price_per_day(state.quote, admission)

    def _quote_breach(
        self, order: CloreOrder, state: CloreSessionState
    ) -> str | None:
        if order.currency != state.currency:
            return "provider currency changed after the immutable quote"
        admission = _state_admission(state)
        expected = self._expected_price_per_day(state)
        if expected is None:
            return "provider price per day cannot be verified against the immutable quote"
        if order.price_per_day != expected:
            return "provider price per day changed after the immutable quote"
        expected_fee = state.quote.provider_fee_percent
        if expected_fee is None and admission is not None:
            expected_fee = Decimal(str(admission.scope.get("provider_fee_percent")))
        if expected_fee is not None:
            if order.fee_percent is None:
                return "provider fee percent is missing; fee semantics are ambiguous"
            if order.fee_percent != expected_fee:
                return "provider fee percent changed after the immutable quote"
        if order.spot is True:
            return "provider order is spot, but the approved profile is on-demand"
        quote_creation = state.quote.creation_fee
        if quote_creation is None and admission is not None:
            quote_creation = Decimal(str(admission.scope.get("creation_fee")))
        if (
            quote_creation is not None
            and order.creation_fee is not None
            and order.creation_fee > 0
            and order.creation_fee != quote_creation
        ):
            return "provider creation fee changed after the immutable quote"
        return None

    def _actual_total(
        self, order: CloreOrder, state: CloreSessionState, now: datetime
    ) -> Decimal | None:
        if order.spend is None:
            return None
        elapsed = self._elapsed_minutes(state, now)
        creation = Decimal("0")
        if elapsed > 10:
            if order.creation_fee is not None and order.creation_fee > 0:
                creation = order.creation_fee
            elif state.quote.creation_fee is not None:
                creation = state.quote.creation_fee
            else:
                admission = _state_admission(state)
                if admission is not None:
                    creation = Decimal(str(admission.scope.get("creation_fee")))
        return order.spend + creation

    def _budget_breach(
        self, order: CloreOrder, state: CloreSessionState, now: datetime
    ) -> str | None:
        if order.spend is None:
            return "provider spend is missing; budget cannot be verified"
        actual = self._actual_total(order, state, now)
        if actual is None:
            return "provider spend cannot be represented as a finite decimal"
        if actual > state.cap_usd:
            return "actual provider spend exceeded the session cap"

        quote = state.quote
        admission = _state_admission(state)
        if admission is not None:
            inputs = owner_risk.budget_inputs(quote, admission)
            if inputs is None:
                return "budget inputs became incomplete"
            rate, creation_fee, other_fee, reserve = inputs
        else:
            rate = quote.rental_per_minute()
            if (
                rate is None
                or quote.creation_fee is None
                or quote.mandatory_other_fees is None
                or quote.reserve is None
            ):
                return "budget inputs became incomplete"
            creation_fee = quote.creation_fee
            other_fee = quote.mandatory_other_fees
            reserve = quote.reserve
        elapsed = max(1, self._elapsed_minutes(state, now))
        observed_total_rate = order.spend / Decimal(elapsed)
        quote_total_rate = rate * (Decimal("1") + _RENTER_FEE_RATE)
        exposure_rate = max(quote_total_rate, observed_total_rate)

        planned = datetime.fromtimestamp(state.planned_cancel_at, tz=timezone.utc)
        hard_end = datetime.fromtimestamp(state.hard_end_at, tz=timezone.utc)
        exposure_end = min(
            planned + timedelta(seconds=self.readback_timeout_seconds),
            hard_end,
        )
        remaining_seconds = max(0.0, (exposure_end - now).total_seconds())
        remaining_minutes = Decimal(str(math.ceil(remaining_seconds / 60.0)))
        future_rental = exposure_rate * remaining_minutes
        creation = (
            creation_fee
            if self._elapsed_minutes(state, exposure_end) > 10
            else Decimal("0")
        )
        upper = order.spend + future_rental + creation + other_fee + reserve
        if upper > state.cap_usd:
            return "conservative remaining-exposure upper bound exceeds the session cap"
        return None

    async def _cancel_locked(
        self,
        state: CloreSessionState,
        *,
        now: datetime,
        reason: str,
    ) -> CloreSessionOutcome:
        if state.status != "ARMED":
            raise CloreStateError("only ARMED state may start a cancellation")
        now = self._now()
        hard_end = datetime.fromtimestamp(state.hard_end_at, tz=timezone.utc)
        if now >= hard_end:
            return self._blocked(
                state,
                "session hard end passed before cancellation; no provider call "
                "is attempted; manual exact-order Terminate required",
            )
        readback_deadline = min(
            now + timedelta(seconds=self.readback_timeout_seconds),
            hard_end,
        )
        if readback_deadline <= now:
            return self._blocked(
                state,
                "no readback budget remains; manual exact-order Terminate required",
            )
        requested = replace(
            state,
            status="STOP_REQUESTED",
            updated_at=_format_utc(now),
            cancel_intent_at=_format_utc(now),
            cancel_reason=reason,
            readback_started_at=_format_utc(now),
            readback_deadline_at=_format_utc(readback_deadline),
            last_error=None,
        )
        # Durable intent is written BEFORE the mutating POST.
        save_state(self.state_path, requested)
        try:
            cancel_deadline = self._monotonic_deadline(readback_deadline)
        except CloreBudgetError:
            requested = replace(
                requested,
                updated_at=_format_utc(self._now()),
                last_error="cancellation request budget expired before send; "
                "readback only",
            )
            save_state(self.state_path, requested)
            return await self._readback_locked(requested)
        try:
            acknowledgement = await self.bridge.cancel_order(
                state.order_id,
                deadline_monotonic=cancel_deadline,
            )
            if not acknowledgement.accepted:
                if acknowledgement.ambiguous:
                    raise CloreCancelAmbiguous(acknowledgement.reason)
                raise CloreProviderError(acknowledgement.reason)
        except CloreCancelAmbiguous as exc:
            requested = replace(
                requested,
                updated_at=_format_utc(self._now()),
                last_error=" ".join(str(exc).split())[:300],
            )
            save_state(self.state_path, requested)
            return await self._readback_locked(requested)
        except (CloreAuthError, CloreProviderError, CloreProtocolError) as exc:
            return self._blocked(
                requested,
                f"cancellation rejected: {type(exc).__name__}; manual exact-order "
                "Terminate required",
            )
        except CloreBudgetError:
            requested = replace(
                requested,
                updated_at=_format_utc(self._now()),
                last_error="cancellation budget expired before or during send; "
                "readback only",
            )
            save_state(self.state_path, requested)
            return await self._readback_locked(requested)
        except CloreError as exc:
            return self._blocked(
                requested,
                f"cancellation error: {type(exc).__name__}; manual exact-order "
                "Terminate required",
            )
        except Exception as exc:
            return self._blocked(
                requested,
                f"cancellation failed: {type(exc).__name__}; manual exact-order "
                "Terminate required",
            )
        if self._now() >= hard_end:
            return self._blocked(
                requested,
                "cancellation POST crossed the hard end; terminal not accepted; "
                "manual exact-order Terminate required",
            )
        return await self._readback_locked(requested)

    async def _readback_locked(
        self, state: CloreSessionState
    ) -> CloreSessionOutcome:
        if state.status != "STOP_REQUESTED":
            raise CloreStateError("readback requires STOP_REQUESTED state")
        if state.readback_deadline_at is None:
            return self._blocked(
                state,
                "readback bound is missing; manual exact-order Terminate required",
            )
        deadline = _coerce_utc(state.readback_deadline_at, "readback_deadline_at")
        now = self._now()
        remaining_wall = max(0.0, (deadline - now).total_seconds())
        if remaining_wall <= 0:
            return self._blocked(
                state,
                "readback bound expired before terminal proof; manual "
                "My Orders exact-ID Terminate required",
            )
        monotonic_deadline = self._monotonic() + remaining_wall
        last_error: str | None = None
        while True:
            now = self._now()
            remaining = min(
                monotonic_deadline - self._monotonic(),
                (deadline - now).total_seconds(),
            )
            if remaining <= 0:
                return self._blocked(
                    state,
                    "readback bound expired before terminal proof; manual "
                    "My Orders exact-ID Terminate required",
                )
            try:
                order = await self.bridge.read_order(
                    state.order_id,
                    deadline_monotonic=monotonic_deadline,
                )
            except CloreBudgetError:
                return self._blocked(
                    state,
                    "readback bound expired during the provider read; late "
                    "terminal is not accepted; manual exact-ID Terminate required",
                )
            except CloreNotFound:
                return self._blocked(
                    state,
                    "exact order row is missing from readback; missing row is not "
                    "terminal proof; manual My Orders exact-ID Terminate required",
                )
            except (CloreAuthError, CloreProtocolError) as exc:
                return self._blocked(
                    state,
                    f"readback cannot be trusted ({type(exc).__name__}); manual "
                    "My Orders exact-ID Terminate required",
                )
            except (CloreTransportError, CloreRateLimitError, CloreProviderError) as exc:
                last_error = type(exc).__name__
            except Exception as exc:
                return self._blocked(
                    state,
                    f"readback failed: {type(exc).__name__}; manual exact-order "
                    "Terminate required",
                )
            else:
                now_after = self._now()
                if (
                    now_after >= deadline
                    or self._monotonic() >= monotonic_deadline
                ):
                    return self._blocked(
                        state,
                        "readback response crossed the bound; late terminal is "
                        "not accepted; manual exact-ID Terminate required",
                    )
                identity_error = self._identity_error(order, state)
                if identity_error is not None:
                    return self._blocked(state, identity_error)
                if order.expired:
                    return self._ended(state)
                last_error = "provider still reports expired=false"
            now = self._now()
            remaining = min(
                monotonic_deadline - self._monotonic(),
                (deadline - now).total_seconds(),
            )
            if remaining <= 0:
                return self._blocked(
                    state,
                    "readback bound expired without terminal proof; manual "
                    "My Orders exact-ID Terminate required",
                )
            await self._sleep(min(self.readback_poll_seconds, remaining))

    async def evaluate(
        self,
        *,
        now: datetime | None = None,
    ) -> CloreSessionOutcome:
        with _state_lock(self.state_path):
            return await self._evaluate_locked(now=now)

    async def _evaluate_locked(
        self,
        *,
        now: datetime | None = None,
    ) -> CloreSessionOutcome:
        state = self.load()
        current = self._now(now)
        if state.status == "ENDED_REPORTED":
            return CloreSessionOutcome(
                "ENDED_REPORTED",
                "provider-reported expired=true; billing unverified",
                state,
                False,
                state.cancel_reason,
            )
        if state.status == "LIVE_BLOCKED":
            return CloreSessionOutcome(
                "LIVE_BLOCKED",
                state.last_error or "manual intervention required",
                state,
                False,
                state.cancel_reason,
            )
        if state.status == "STOP_REQUESTED":
            return await self._readback_locked(state)

        hard_end = datetime.fromtimestamp(state.hard_end_at, tz=timezone.utc)
        if current >= hard_end:
            return self._blocked(
                state,
                "session hard end passed without cancellation intent; no provider "
                "call is attempted; manual exact-order Terminate required",
            )

        read_deadline = self._monotonic_deadline(hard_end)
        try:
            order = await self.bridge.read_order(
                state.order_id,
                deadline_monotonic=read_deadline,
            )
        except CloreBudgetError:
            current = self._now()
            if current >= hard_end:
                return self._blocked(
                    state,
                    "pre-cancel read consumed the hard end; no mutation is "
                    "attempted; manual exact-order Terminate required",
                )
            if current >= datetime.fromtimestamp(
                state.planned_cancel_at, tz=timezone.utc
            ):
                return self._blocked(
                    state,
                    "pre-cancel read consumed the planned deadline; no safe "
                    "mutation remains; manual exact-order Terminate required",
                )
            return CloreSessionOutcome(
                "ARMED",
                "pre-cancel read budget expired; will retry within the deadline",
                state,
            )
        except CloreNotFound:
            return self._blocked(
                state,
                "exact order row is missing before cancellation; no mutation is "
                "attempted; manual My Orders exact-ID Terminate required",
            )
        except CloreCurrencyMismatch as exc:
            observed = exc.order
            if observed is None:
                return self._blocked(
                    state,
                    "provider currency mismatch without a normalized order; "
                    "manual exact-order Terminate required",
                )
            identity_error = self._identity_error(observed, state)
            if identity_error is not None:
                return self._blocked(state, identity_error)
            if observed.expired:
                return self._ended(state)
            return await self._cancel_locked(
                state,
                now=current,
                reason="provider currency changed after the immutable quote",
            )
        except (
            CloreTransportError,
            CloreProtocolError,
            CloreAuthError,
            CloreProviderError,
            CloreRateLimitError,
        ) as exc:
            current = self._now()
            planned = datetime.fromtimestamp(state.planned_cancel_at, tz=timezone.utc)
            if current >= planned:
                return self._blocked(
                    state,
                    f"pre-cancel readback failed ({type(exc).__name__}); no "
                    "mutation is attempted; manual exact-order Terminate required",
                )
            return CloreSessionOutcome(
                "ARMED",
                f"pre-cancel readback failed ({type(exc).__name__}); will retry "
                "within the persisted deadline",
                state,
                False,
                None,
            )
        except Exception as exc:
            return self._blocked(
                state,
                f"pre-cancel readback failed: {type(exc).__name__}; manual "
                "exact-order Terminate required",
            )

        current = self._now()
        if current >= hard_end:
            return self._blocked(
                state,
                "pre-cancel read crossed the hard end; no mutation is attempted; "
                "manual exact-order Terminate required",
            )
        identity_error = self._identity_error(order, state)
        if identity_error is not None:
            return self._blocked(state, identity_error)
        if order.expired:
            return self._ended(state)

        breach = self._quote_breach(order, state)
        if breach is None:
            breach = self._budget_breach(order, state, current)
        if breach is not None:
            return await self._cancel_locked(state, now=current, reason=breach)

        planned = datetime.fromtimestamp(state.planned_cancel_at, tz=timezone.utc)
        if current >= planned:
            return await self._cancel_locked(
                state, now=current, reason="planned cancellation deadline"
            )
        return CloreSessionOutcome(
            "ARMED", "before planned cancellation; no mutation performed", state
        )

    async def stop_now(self, *, reason: str = "operator stop") -> CloreSessionOutcome:
        """Explicit early stop for finish/error; never waits for generation/export."""

        with _state_lock(self.state_path):
            state = self.load()
            current = self._now()
            if state.status in ("ENDED_REPORTED", "LIVE_BLOCKED"):
                return CloreSessionOutcome(
                    state.status,
                    state.last_error or "no further mutation",
                    state,
                    state.billing_verified,
                    state.cancel_reason,
                )
            if state.status == "STOP_REQUESTED":
                return await self._readback_locked(state)
            hard_end = datetime.fromtimestamp(state.hard_end_at, tz=timezone.utc)
            if current >= hard_end:
                return self._blocked(
                    state,
                    "session hard end passed; no provider call is attempted; "
                    "manual exact-order Terminate required",
                )
            read_deadline = self._monotonic_deadline(hard_end)
            try:
                order = await self.bridge.read_order(
                    state.order_id,
                    deadline_monotonic=read_deadline,
                )
            except CloreBudgetError:
                return self._blocked(
                    state,
                    "early-stop pre-read consumed the hard end; no mutation is "
                    "attempted; manual exact-order Terminate required",
                )
            except CloreCurrencyMismatch as exc:
                observed = exc.order
                if observed is None:
                    return self._blocked(
                        state,
                        "provider currency mismatch without a normalized order; "
                        "manual exact-order Terminate required",
                    )
                identity_error = self._identity_error(observed, state)
                if identity_error is not None:
                    return self._blocked(state, identity_error)
                if observed.expired:
                    return self._ended(state)
                return await self._cancel_locked(
                    state,
                    now=current,
                    reason="provider currency changed after the immutable quote",
                )
            except Exception as exc:
                return self._blocked(
                    state,
                    f"early-stop pre-read failed: {type(exc).__name__}; manual "
                    "exact-order Terminate required",
                )
            current = self._now()
            if current >= hard_end:
                return self._blocked(
                    state,
                    "early-stop pre-read crossed the hard end; no mutation is "
                    "attempted; manual exact-order Terminate required",
                )
            identity_error = self._identity_error(order, state)
            if identity_error is not None:
                return self._blocked(state, identity_error)
            if order.expired:
                return self._ended(state)
            return await self._cancel_locked(state, now=current, reason=reason)

    async def watch(
        self,
        *,
        poll_interval_seconds: float = DEFAULT_WATCH_POLL_SECONDS,
    ) -> CloreSessionOutcome:
        poll = _finite_positive(poll_interval_seconds, "poll_interval_seconds")
        while True:
            state = self.load()
            if state.status in ("ENDED_REPORTED", "LIVE_BLOCKED"):
                return CloreSessionOutcome(
                    state.status,
                    state.last_error or "no further mutation",
                    state,
                    state.billing_verified,
                    state.cancel_reason,
                )
            current = self._now()
            if state.status == "STOP_REQUESTED":
                return await self.evaluate(now=current)
            hard_end = datetime.fromtimestamp(state.hard_end_at, tz=timezone.utc)
            planned = datetime.fromtimestamp(state.planned_cancel_at, tz=timezone.utc)
            if current >= hard_end:
                return await self.evaluate(now=current)
            # Read/validate/budget-check at each poll, but never hold the state
            # lock while sleeping.
            outcome = await self.evaluate(now=current)
            if outcome.status != "ARMED":
                return outcome
            current = self._now()
            remaining = min(
                (planned - current).total_seconds(),
                (hard_end - current).total_seconds(),
            )
            if remaining <= 0:
                return await self.evaluate(now=self._now())
            await self._sleep(min(poll, remaining))


# ---------------------------------------------------------------------------
# Dry-run CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/p1/clore_session.py",
        description=(
            "Clore session budget/deadline plan; dry-run by default. "
            "No environment, network, provider, or state write without --live."
        ),
    )
    parser.add_argument("--state", default=None)
    parser.add_argument("--quote", default=None, help="fresh quote JSON file")
    parser.add_argument("--cap-usd", default=None)
    parser.add_argument("--server-id", type=int, default=EXPECTED_SERVER_ID)
    parser.add_argument("--order-id", type=int, default=None)
    parser.add_argument("--created-at", type=int, default=None)
    parser.add_argument("--duration-seconds", type=int, default=MAX_DURATION_SECONDS)
    parser.add_argument("--planned-cancel-at", type=int, default=None)
    parser.add_argument("--hard-end-at", type=int, default=None)
    parser.add_argument("--readback-timeout", type=float, default=DEFAULT_READBACK_SECONDS)
    parser.add_argument("--readback-poll", type=float, default=DEFAULT_READBACK_POLL_SECONDS)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="compatibility; live defaults to watch",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="explicit single evaluation; external supervisor required",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicit future live opt-in; not used by the offline evidence run",
    )
    return parser


def _load_quote_document(path: Path) -> tuple[CloreQuote, Decimal | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloreSessionError(f"cannot read quote: {type(exc).__name__}") from exc
    profile_cap: Decimal | None = None
    if isinstance(data, Mapping) and isinstance(data.get("quote"), Mapping):
        profile_cap = _optional_decimal(
            data.get("session_cap_usd"), "profile.session_cap_usd"
        )
        data = data["quote"]
    return CloreQuote.from_dict(data), profile_cap


def _validate_existing_binding(
    state: CloreSessionState,
    *,
    args: argparse.Namespace,
    quote: CloreQuote,
    cap: Decimal,
) -> None:
    if args.order_id is None or args.created_at is None:
        raise CloreSessionError(
            "existing state requires explicit --order-id and --created-at"
        )
    comparisons = {
        "server_id": (state.server_id, args.server_id),
        "order_id": (state.order_id, args.order_id),
        "created_at": (state.created_at, args.created_at),
        "currency": (state.currency, quote.currency),
        "quote": (state.quote, quote),
        "cap_usd": (state.cap_usd, cap),
        "duration_seconds": (state.duration_seconds, args.duration_seconds),
    }
    for name, (actual, requested) in comparisons.items():
        if actual != requested:
            raise CloreSessionError(
                f"existing Clore binding mismatch for {name}; refusing a different session"
            )
    if args.planned_cancel_at is not None and state.planned_cancel_at != args.planned_cancel_at:
        raise CloreSessionError(
            "existing Clore binding mismatch for planned_cancel_at"
        )
    if args.hard_end_at is not None and state.hard_end_at != args.hard_end_at:
        raise CloreSessionError("existing Clore binding mismatch for hard_end_at")


def _validate_new_binding(
    *,
    args: argparse.Namespace,
    quote: CloreQuote,
    cap: Decimal,
) -> tuple[int, int]:
    if args.order_id is None or args.created_at is None:
        raise CloreSessionError("new session requires --order-id and --created-at")
    if args.server_id != EXPECTED_SERVER_ID:
        raise CloreLaunchBlocked(
            ("server id does not match the selected Clore server",)
        )
    if cap <= 0 or cap > CAP_USD:
        raise CloreSessionError("cap_usd must be positive and at most USD 1.00")
    if not (0 < args.duration_seconds <= MAX_DURATION_SECONDS):
        raise CloreSessionError("duration must be at most 3600 seconds")
    planned = (
        args.created_at + PLANNED_CANCEL_OFFSET_SECONDS
        if args.planned_cancel_at is None
        else args.planned_cancel_at
    )
    hard_end = (
        args.created_at + args.duration_seconds
        if args.hard_end_at is None
        else args.hard_end_at
    )
    if planned < args.created_at or planned > args.created_at + PLANNED_CANCEL_OFFSET_SECONDS:
        raise CloreSessionError("planned cancel must be within creation + 55 minutes")
    if hard_end <= args.created_at or hard_end > args.created_at + MAX_DURATION_SECONDS:
        raise CloreSessionError("hard end must be within creation + 60 minutes")
    if planned > hard_end:
        raise CloreSessionError("planned cancel must not exceed the hard end")
    blockers = quote.launch_blockers(cap_usd=cap, minutes=60)
    if blockers:
        raise CloreLaunchBlocked(blockers)
    return planned, hard_end


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.watch and args.once:
        print(
            "clore session refused: --watch and --once are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    try:
        if args.quote is None:
            raise CloreSessionError("--quote is required")
        quote, profile_cap = _load_quote_document(Path(args.quote))
        cap = (
            _optional_decimal(args.cap_usd, "cap_usd")
            if args.cap_usd is not None
            else profile_cap
        )
        if cap is None:
            cap = CAP_USD
        if cap <= 0 or cap > CAP_USD:
            raise CloreSessionError("cap_usd must be positive and at most USD 1.00")

        state_path = Path(args.state) if args.state is not None else None
        existing_state: CloreSessionState | None = None
        planned_for_new: int | None = None
        hard_for_new: int | None = None
        if state_path is not None and state_path.exists():
            existing_state = load_state(state_path, safety_mode=True)
            _validate_existing_binding(
                existing_state, args=args, quote=quote, cap=cap
            )
        elif args.live:
            planned_for_new, hard_for_new = _validate_new_binding(
                args=args, quote=quote, cap=cap
            )

        blockers = quote.launch_blockers(cap_usd=cap, minutes=60)
        projection = quote.project_total(60)
        plan = {
            "status": "CLORE_PLAN" if not args.live else "CLORE_LIVE_OPT_IN",
            "provider": "clore",
            "currency": quote.currency,
            "cap_usd": str(cap),
            "projection_60m": None if projection is None else str(projection),
            "launch_ready": not blockers,
            "launch_blockers": list(blockers),
            "existing_state": (
                None
                if existing_state is None
                else {
                    "order_id": existing_state.order_id,
                    "created_at": existing_state.created_at,
                    "status": existing_state.status,
                }
            ),
            "note": (
                "dry-run: no environment read, no network/provider call, no state "
                "write; quote projection is not live readiness"
            ),
        }
        if not args.live:
            _print_json(plan)
            return 0

        if state_path is None or args.order_id is None or args.created_at is None:
            raise CloreSessionError(
                "--live requires --state, --order-id and --created-at"
            )

        async def _run_live() -> CloreSessionOutcome:
            # Client construction happens only after local binding/quote checks.
            expected_server = (
                existing_state.server_id if existing_state is not None else args.server_id
            )
            expected_currency = (
                existing_state.currency if existing_state is not None else quote.currency
            )
            bridge = CloreBridge(
                allow_live=True,
                expected_server_id=expected_server,
                expected_currency=expected_currency,
            )
            try:
                session = CloreSession(
                    state_path,
                    bridge=bridge,
                    readback_timeout_seconds=args.readback_timeout,
                    readback_poll_seconds=args.readback_poll,
                )
                if existing_state is None:
                    session.initialize(
                        order_id=args.order_id,
                        server_id=args.server_id,
                        created_at=args.created_at,
                        currency=quote.currency,
                        quote=quote,
                        cap_usd=cap,
                        duration_seconds=args.duration_seconds,
                        planned_cancel_at=planned_for_new,
                        hard_end_at=hard_for_new,
                    )
                if args.watch and args.once:
                    raise CloreSessionError(
                        "--watch and --once are mutually exclusive"
                    )
                if args.once:
                    print(
                        "WARNING: --once exits after one evaluation; an external "
                        "supervisor must own the deadline/cancel",
                        file=sys.stderr,
                    )
                    return await session.evaluate()
                return await session.watch()
            finally:
                await bridge.aclose()

        outcome = asyncio.run(_run_live())
        _print_json(
            {
                "status": outcome.status,
                "message": outcome.message,
                "billing_verified": outcome.billing_verified,
                "breach": outcome.breach,
            }
        )
        return 0 if outcome.status == "ENDED_REPORTED" else 3
    except (CloreSessionError, CloreError) as exc:
        print(f"clore session refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
