"""Owner financial-risk admission for the single approved P1 session.

This module is intentionally small and offline-only. It uses only the local
``anime_studio.lock.InstanceLock`` for the central claim-registry exclusion; it
does not read environment secrets, call a provider, start SSH, or mutate a quote.
It provides one policy entrypoint used by session initialization/loading,
readiness, supervisor, budget watch and operator flow.


Strict quote policy is unchanged when no admission is supplied. The admission
only waives the explicitly listed financial unknowns for one scoped session:
minimum billed duration, additional mandatory fees and tax/FX/payment fees.
Freshness, server/currency/identity, exact rate, deadline, source, host,
supervisor, runtime and model gates remain strict.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "anime-video-studio/p1-owner-risk-admission/v1"
PENDING_SCHEMA = "anime-video-studio/p1-owner-risk-pending/v1"
CLAIM_SCHEMA = "anime-video-studio/p1-owner-risk-claim/v1"
RESERVE_SCHEMA = "anime-video-studio/p1-owner-risk-reserve/v1"
PROPOSAL_SCHEMA = "anime-video-studio/p1-owner-risk-proposal/v1"
APPROVAL_SCHEMA = "anime-video-studio/p1-owner-risk-approval/v1"
MARKET_QUOTE_SCHEMA = "anime-video-studio/p1-marketplace-quote/v1"
APPROVAL_CONFIRMATION = "APPROVE_ONE_SCOPED_PROPOSAL"
MARKET_COLLECTOR_ID = "anime-video-studio/p1-marketplace-quote-collector"
MARKET_COLLECTOR_VERSION = "1"
MARKET_OBSERVATION_ORIGIN = "authenticated_marketplace_price_get"
MARKET_BASE_URL = "https://api.clore.ai/v1/"
MARKET_ENDPOINT = "marketplace"
MARKET_METHOD = "GET"
MARKET_PATH = "/v1/marketplace"
MARKET_FRESHNESS = "observed_at is the successful GET completion time"


RISK_DOC = ROOT / "docs" / "P1-owner-financial-risk-20261026.md"
SESSION_APPROVAL_DOC = ROOT / "docs" / "P1-session-02-approval-20261026.md"
RISK_DOC_REL = "docs/P1-owner-financial-risk-20261026.md"
SESSION_APPROVAL_DOC_REL = "docs/P1-session-02-approval-20261026.md"
DEFAULT_REGISTRY_ROOT = ROOT / "data" / "p1-admissions"
CONSENT_PURPOSE = "session02_owner_risk_one_session"

# Only authorizing identity fields are part of the stable consent id. Descriptive
# provenance/raw-mrl/assumption metadata is checked but cannot mint a new consent.
AUTHORIZING_SCOPE_KEYS = (
    "server_id",
    "hoster_id",
    "currency",
    "price_per_day",
    "provider_fee_percent",
    "creation_fee",
    "reserve",
    "code_cap_usd",
    "owner_ceiling_usd",
    "duration_seconds",
    "orders",
    "generations",
    "automatic_retries",
)
NONAUTHORIZING_SCOPE_KEYS = (
    "purpose",
    "hoster_id_provenance",
    "mrl_semantics",
    "provider_reported_mrl_seconds",
    "mandatory_other_fees_assumed",
    "min_billing_seconds_assumed",
)
ALLOWED_SCOPE_KEYS = frozenset(AUTHORIZING_SCOPE_KEYS + NONAUTHORIZING_SCOPE_KEYS)
INT_SCOPE_KEYS = (
    "server_id",
    "hoster_id",
    "duration_seconds",
    "orders",
    "generations",
    "automatic_retries",
)
DECIMAL_SCOPE_KEYS = (
    "price_per_day",
    "provider_fee_percent",
    "creation_fee",
    "reserve",
    "code_cap_usd",
    "owner_ceiling_usd",
)

ALLOWED_SERVER_ID = 111947
ALLOWED_HOSTER_ID = 46993
ALLOWED_CURRENCY = "USD-Blockchain"
ALLOWED_PRICE_PER_DAY = Decimal("12")
ALLOWED_PROVIDER_FEE_PERCENT = Decimal("5")
ALLOWED_CREATION_FEE = Decimal("0.10")
ALLOWED_RESERVE = Decimal("0.10")
ASSUMED_OTHER_FEES = Decimal("0")
ASSUMED_MIN_BILLING_SECONDS = 60
PROVIDER_REPORTED_MRL_SECONDS = 336 * 60 * 60
CODE_CAP_USD = Decimal("1.00")
OWNER_CEILING_USD = Decimal("2.00")
DURATION_SECONDS = 3600
ORDERS = 1
GENERATIONS = 1
AUTOMATIC_RETRIES = 0
QUOTE_MAX_AGE_SECONDS = 15 * 60
MAX_ADMISSION_HOURS = 24
TOTAL_60M = Decimal("0.725")
_HEX64 = 64


class OwnerRiskError(Exception):
    """An admission/claim is missing, malformed, tampered or not reusable."""


class OwnerRiskClaimError(OwnerRiskError):
    """A consent has already been claimed or a claim is inconsistent."""


@dataclass(frozen=True)
class OwnerRiskAdmission:
    schema: str
    admission_id: str
    created_at: datetime
    expires_at: datetime
    approval_doc: Mapping[str, Any]
    session_approval_doc: Mapping[str, Any]
    quote: Mapping[str, Any]
    scope: Mapping[str, Any]
    projection: Mapping[str, Any]
    financial_unknowns: tuple[str, ...]
    verification: Mapping[str, Any]
    provenance: Mapping[str, Any]
    permission_id: str | None = None
    approval: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "admission_id": self.admission_id,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "approval_doc": dict(self.approval_doc),
            "session_approval_doc": dict(self.session_approval_doc),
            "quote": dict(self.quote),
            "scope": dict(self.scope),
            "projection": dict(self.projection),
            "financial_unknowns": list(self.financial_unknowns),
            "verification": dict(self.verification),
            "provenance": dict(self.provenance),
        }
        if self.permission_id is not None:
            payload["permission_id"] = self.permission_id
        if self.approval is not None:
            payload["approval"] = dict(self.approval)
        return _jsonable(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OwnerRiskAdmission":
        if not isinstance(data, Mapping):
            raise OwnerRiskError("owner-risk admission must be an object")
        allowed = {
            "schema",
            "admission_id",
            "created_at",
            "expires_at",
            "approval_doc",
            "session_approval_doc",
            "quote",
            "scope",
            "projection",
            "financial_unknowns",
            "verification",
            "provenance",
            "permission_id",
            "approval",
        }
        required = allowed - {"permission_id", "approval"}
        if set(data) - allowed or required - set(data):
            raise OwnerRiskError("owner-risk admission has unknown/missing keys")

        for key in (
            "approval_doc",
            "session_approval_doc",
            "quote",
            "scope",
            "projection",
            "verification",
            "provenance",
        ):
            if not isinstance(data.get(key), Mapping):
                raise OwnerRiskError(f"owner-risk admission {key} must be an object")
        unknown_scope = set(data["scope"]) - ALLOWED_SCOPE_KEYS
        if unknown_scope:
            raise OwnerRiskError(
                f"owner-risk scope has unknown keys: {sorted(unknown_scope)}"
            )
        unknowns = data.get("financial_unknowns")
        if not isinstance(unknowns, list) or any(
            not isinstance(value, str) or not value.strip() for value in unknowns
        ):
            raise OwnerRiskError(
                "owner-risk admission financial_unknowns must be a string list"
            )
        permission_id = data.get("permission_id")
        if permission_id is not None:
            permission_id = _required_hex(permission_id, "permission_id")
        approval = data.get("approval")
        if approval is not None and not isinstance(approval, Mapping):
            raise OwnerRiskError("owner-risk admission approval must be an object")
        return cls(
            schema=_required_str(data.get("schema"), "schema"),
            admission_id=_required_hex(data.get("admission_id"), "admission_id"),
            created_at=_parse_moment(data.get("created_at"), "created_at"),
            expires_at=_parse_moment(data.get("expires_at"), "expires_at"),
            approval_doc=dict(data["approval_doc"]),
            session_approval_doc=dict(data["session_approval_doc"]),
            quote=dict(data["quote"]),
            scope=dict(data["scope"]),
            projection=dict(data["projection"]),
            financial_unknowns=tuple(unknowns),
            verification=dict(data["verification"]),
            provenance=dict(data["provenance"]),
            permission_id=permission_id,
            approval=dict(approval) if approval is not None else None,
        )


def admission_digest(admission: OwnerRiskAdmission | Mapping[str, Any]) -> str:
    payload = admission.to_dict() if isinstance(admission, OwnerRiskAdmission) else _jsonable(admission)
    return _sha_mapping(payload)


def admission_id_for(
    *,
    approval_doc: Mapping[str, Any],
    session_approval_doc: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> str:
    """Stable consent identity: documents + session scope, not path/quote time."""

    unknown = set(scope) - ALLOWED_SCOPE_KEYS
    if unknown:
        raise OwnerRiskError(f"owner-risk scope has unknown keys: {sorted(unknown)}")
    missing = set(AUTHORIZING_SCOPE_KEYS) - set(scope)
    if missing:
        raise OwnerRiskError(
            f"owner-risk authorizing scope is missing: {sorted(missing)}"
        )
    core = {
        "schema": SCHEMA,
        "purpose": CONSENT_PURPOSE,
        "approval_doc_sha256": _required_hex(
            approval_doc.get("sha256"), "approval_doc.sha256"
        ),
        "session_approval_doc_sha256": _required_hex(
            session_approval_doc.get("sha256"), "session_approval_doc.sha256"
        ),
        "authorizing_scope": _canonical_authorizing_scope(scope),
    }
    return _sha_mapping(core)


def permission_id_for(
    *,
    approval_doc: Mapping[str, Any],
    session_approval_doc: Mapping[str, Any],
) -> str:
    """Stable overall owner permission, independent of server/rate/proposal.

    The single approved order/generation permission is anchored only to the
    fixed owner-risk document and the session02 authorization document plus the
    fixed purpose. A new proposal cannot mint a second permission by changing a
    server, rate, currency, quote timestamp or local metadata representation.
    """

    core = {
        "schema": SCHEMA,
        "purpose": CONSENT_PURPOSE,
        "approval_doc_sha256": _required_hex(
            approval_doc.get("sha256"), "approval_doc.sha256"
        ),
        "session_approval_doc_sha256": _required_hex(
            session_approval_doc.get("sha256"), "session_approval_doc.sha256"
        ),
    }
    return _sha_mapping(core)


def permission_id_for_admission(
    admission: OwnerRiskAdmission | Mapping[str, Any],
) -> str:
    value = _coerce_admission(admission)
    expected = permission_id_for(
        approval_doc=value.approval_doc,
        session_approval_doc=value.session_approval_doc,
    )
    if value.permission_id is not None and value.permission_id != expected:
        raise OwnerRiskError("owner-risk permission id does not match root documents")
    return expected


def proposal_payload_for(
    *,
    approval_doc: Mapping[str, Any],
    session_approval_doc: Mapping[str, Any],
    quote: Mapping[str, Any],
    scope: Mapping[str, Any],
    projection: Mapping[str, Any],
    financial_unknowns: list[str],
    verification: Mapping[str, Any],
    provenance: Mapping[str, Any],
    created_at: str,
    expires_at: str,
) -> dict[str, Any]:
    return {
        "schema": PROPOSAL_SCHEMA,
        "permission_id": permission_id_for(
            approval_doc=approval_doc, session_approval_doc=session_approval_doc
        ),
        "approval_doc": dict(approval_doc),
        "session_approval_doc": dict(session_approval_doc),
        "quote": dict(quote),
        "scope": dict(scope),
        "projection": dict(projection),
        "financial_unknowns": list(financial_unknowns),
        "verification": dict(verification),
        "provenance": dict(provenance),
        "created_at": created_at,
        "expires_at": expires_at,
    }


def proposal_digest_for_admission(
    admission: OwnerRiskAdmission | Mapping[str, Any],
) -> str:
    value = _coerce_admission(admission)
    return _sha_mapping(
        proposal_payload_for(
            approval_doc=value.approval_doc,
            session_approval_doc=value.session_approval_doc,
            quote=value.quote,
            scope=value.scope,
            projection=value.projection,
            financial_unknowns=list(value.financial_unknowns),
            verification=value.verification,
            provenance=value.provenance,
            created_at=_iso(value.created_at),
            expires_at=_iso(value.expires_at),
        )
    )



def _is_dynamic(value: OwnerRiskAdmission) -> bool:
    return value.approval is not None


def _canonical_authorizing_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize typed authorizing values before hashing.

    ``12``, ``12.0``, ``1.2E+1`` and ``1.200E1`` all become the same canonical
    decimal string; integer fields reject bool/string/float and are hashed as int.
    """

    canonical: dict[str, Any] = {}
    for key in AUTHORIZING_SCOPE_KEYS:
        value = scope[key]
        if key in INT_SCOPE_KEYS:
            canonical[key] = _required_int(
                value,
                f"scope.{key}",
                minimum=0 if key == "automatic_retries" else 1,
            )
        elif key in DECIMAL_SCOPE_KEYS:
            number = _decimal(value, f"scope.{key}")
            assert number is not None
            canonical[key] = format(number.normalize(), "f")
        else:
            canonical[key] = _required_str(value, f"scope.{key}")
    return canonical


def validate_fresh_quote(
    admission: OwnerRiskAdmission | Mapping[str, Any],
    *,
    now: datetime | float | None = None,
) -> tuple[str, ...]:
    """Validate the original market observation time, not a later bind time."""

    try:
        value = _coerce_admission(admission)
        observed = _parse_moment(value.quote.get("observed_at"), "quote.observed_at")
    except OwnerRiskError as exc:
        return (str(exc),)
    current = _now(now)
    age = (current - observed).total_seconds()
    issues: list[str] = []
    if age < -60:
        issues.append("owner-risk quote observed_at is in the future")
    if age > QUOTE_MAX_AGE_SECONDS:
        issues.append("owner-risk original quote observation is stale")
    return tuple(issues)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_text(value: Any) -> str:
    return json.dumps(
        _jsonable(value), indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_mapping(value: Mapping[str, Any]) -> str:
    text = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _sha_bytes(text.encode("utf-8"))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise OwnerRiskError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _required_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OwnerRiskError(f"{name} must be a non-empty string")
    return value.strip()


def _required_hex(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HEX64
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise OwnerRiskError(f"{name} must be 64 hex characters")
    return value.lower()


def _required_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OwnerRiskError(f"{name} must be an integer >= {minimum}")
    return value


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OwnerRiskError(f"{name} must be a boolean")
    return value


def _decimal(value: Any, name: str, *, optional: bool = False) -> Decimal | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise OwnerRiskError(f"{name} must be a finite decimal")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise OwnerRiskError(f"{name} must be a finite decimal") from exc
    if not number.is_finite() or number < 0:
        raise OwnerRiskError(f"{name} must be a finite non-negative decimal")
    return number


def _parse_moment(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise OwnerRiskError(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise OwnerRiskError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise OwnerRiskError(f"{name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _now(value: datetime | float | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise OwnerRiskError("now must be timezone-aware")
        return value.astimezone(timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _assert_no_secret_keys(value: Any, *, where: str = "object") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(
                marker in lowered
                for marker in (
                    "token",
                    "secret",
                    "password",
                    "passphrase",
                    "private_key",
                    "privatekey",
                    "api_key",
                    "apikey",
                    "authorization",
                )
            ):
                raise OwnerRiskError(f"{where} contains a secret-like key")
            _assert_no_secret_keys(item, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secret_keys(item, where=f"{where}[{index}]")


def _assert_safe_path(path: Path, *, must_exist: bool) -> Path:
    candidate = Path(path)
    current = candidate
    while True:
        try:
            if current.is_symlink() or bool(getattr(current, "is_junction", lambda: False)()):
                raise OwnerRiskError(f"path has a symlink/junction component: {current}")
        except OSError as exc:
            raise OwnerRiskError(f"path cannot be inspected: {candidate}") from exc
        if current.parent == current:
            break
        current = current.parent
    if candidate.exists():
        if candidate.is_dir():
            if must_exist:
                raise OwnerRiskError(f"path is a directory, not a file: {candidate}")
        elif not candidate.is_file():
            raise OwnerRiskError(f"path is not a regular file: {candidate}")
    elif must_exist:
        raise OwnerRiskError(f"file is missing: {candidate}")
    return candidate


def _atomic_create(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise
    except OSError as exc:
        raise OwnerRiskError(f"cannot create {path.name}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OwnerRiskError(f"cannot read {path.name}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise OwnerRiskError(f"{path.name} must contain an object")
    return value


def _read_quote_document(path: Path) -> tuple[dict[str, Any], datetime, Mapping[str, Any]]:
    data = _read_json(path)
    _assert_no_secret_keys(data, where="quote")
    quote = data.get("quote") if isinstance(data.get("quote"), Mapping) else data
    if not isinstance(quote, Mapping):
        raise OwnerRiskError("quote document must contain an object")
    observed_raw = data.get("observed_at")
    if not isinstance(observed_raw, str) or not observed_raw.strip():
        raise OwnerRiskError("fresh quote document requires top-level observed_at")
    observed = _parse_moment(observed_raw, "quote.observed_at")
    provenance = data.get("provenance")
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, Mapping):
        raise OwnerRiskError("quote.provenance must be an object")
    origin = (
        data.get("observation_origin")
        or provenance.get("observation_origin")
        or provenance.get("kind")
        or provenance.get("source")
    )
    if not isinstance(origin, str) or not origin.strip():
        raise OwnerRiskError("fresh quote requires observation_origin/provenance")
    return dict(quote), observed, dict(provenance)


def _quote_decimal(quote: Mapping[str, Any], name: str, *, optional: bool = True) -> Decimal | None:
    return _decimal(quote.get(name), f"quote.{name}", optional=optional)


def _quote_bool(quote: Mapping[str, Any], name: str, *, default: bool = False) -> bool:
    value = quote.get(name, default)
    if not isinstance(value, bool):
        raise OwnerRiskError(f"quote.{name} must be a boolean")
    return value


def _validate_quote_for_admission(quote: Mapping[str, Any]) -> None:
    currency = _required_str(quote.get("currency"), "quote.currency")
    if currency != ALLOWED_CURRENCY:
        raise OwnerRiskError("quote currency is not the approved USD-Blockchain identifier")
    if not _quote_bool(quote, "fresh_verified"):
        raise OwnerRiskError("admission requires a fresh verified quote")
    if _quote_bool(quote, "fees_known") or _quote_bool(quote, "tax_fx_known"):
        raise OwnerRiskError("financial unknown flags must remain false for this admission")
    if quote.get("min_rental_hours") is not None:
        minimum = _quote_decimal(quote, "min_rental_hours")
        assert minimum is not None
        if minimum * Decimal(3600) > Decimal(ASSUMED_MIN_BILLING_SECONDS):
            raise OwnerRiskError("known minimum rental exceeds the 60-minute session")
    day = _quote_decimal(quote, "price_per_day")
    if day is None or day != ALLOWED_PRICE_PER_DAY:
        raise OwnerRiskError("admission requires exact 12 USD/day")
    hour = _quote_decimal(quote, "price_per_hour")
    minute = _quote_decimal(quote, "price_per_minute")
    tolerance = Decimal("0.000001")
    if hour is not None and abs(hour * Decimal(24) - ALLOWED_PRICE_PER_DAY) > tolerance:
        raise OwnerRiskError("quote hourly rate is inconsistent with exact 12 USD/day")
    if minute is not None and abs(minute * Decimal(1440) - ALLOWED_PRICE_PER_DAY) > tolerance:
        raise OwnerRiskError("quote minute rate is inconsistent with exact 12 USD/day")
    fee = _quote_decimal(quote, "provider_fee_percent")
    if fee != ALLOWED_PROVIDER_FEE_PERCENT:
        raise OwnerRiskError("provider fee percent must be exactly 5")
    creation = _quote_decimal(quote, "creation_fee")
    if creation is not None and creation != ALLOWED_CREATION_FEE:
        raise OwnerRiskError("known creation fee differs from the approved 0.10")
    reserve = _quote_decimal(quote, "reserve")
    if reserve is not None and reserve != ALLOWED_RESERVE:
        raise OwnerRiskError("known reserve differs from the approved 0.10")
    other = _quote_decimal(quote, "mandatory_other_fees")
    if other is not None and other != 0:
        raise OwnerRiskError("known mandatory other fees are outside this risk scope")


def projection_for_quote(
    quote: Any,
    admission: OwnerRiskAdmission | Mapping[str, Any] | None,
    *,
    minutes: int = 60,
) -> dict[str, Any] | None:
    if admission is None:
        return None
    admission_obj = _coerce_admission(admission)
    scope = admission_obj.scope
    day = _decimal(scope.get("price_per_day"), "scope.price_per_day")
    if day is None:
        return None
    minute_decimal = Decimal(minutes)
    rental = day * minute_decimal / Decimal(1440)
    fee_percent = _decimal(scope.get("provider_fee_percent"), "scope.provider_fee_percent")
    if fee_percent is None:
        return None
    renter_fee = rental * fee_percent / Decimal(100)
    creation = (
        _decimal(scope.get("creation_fee"), "scope.creation_fee")
        if minutes > 10
        else Decimal("0")
    )
    reserve = _decimal(scope.get("reserve"), "scope.reserve")
    other = _decimal(
        scope.get("mandatory_other_fees_assumed"),
        "scope.mandatory_other_fees_assumed",
    )
    if creation is None or reserve is None or other is None:
        return None
    total = rental + renter_fee + creation + reserve + other
    return {
        "minutes": minutes,
        "rental": rental,
        "renter_fee": renter_fee,
        "creation_fee": creation,
        "reserve": reserve,
        "other_fees_assumed": other,
        "total": total,
        "assumptions": [
            f"minimum billing {ASSUMED_MIN_BILLING_SECONDS}s assumed, not verified",
            "additional mandatory fees assumed 0 and covered by reserve, not factual",
            "tax/FX/payment fees assumed 0 and excluded, not factual",
            "creation fee 0.10 used from approved checkout/assumption",
            "mrl 336h recorded as raw fact with unverified semantics; not used as minimum",
        ],
    }


def effective_projection(
    quote: Any,
    admission: OwnerRiskAdmission | Mapping[str, Any] | None,
    *,
    minutes: int = 60,
) -> Decimal | None:
    facts = projection_for_quote(quote, admission, minutes=minutes)
    return None if facts is None else facts["total"]


def expected_price_per_day(
    quote: Any, admission: OwnerRiskAdmission | Mapping[str, Any] | None
) -> Decimal | None:
    if admission is not None:
        scope = _coerce_admission(admission).scope
        return _decimal(scope.get("price_per_day"), "scope.price_per_day")
    if getattr(quote, "price_per_day", None) is not None:
        return quote.price_per_day
    if getattr(quote, "price_per_hour", None) is not None:
        return quote.price_per_hour * Decimal(24)
    if getattr(quote, "price_per_minute", None) is not None:
        return quote.price_per_minute * Decimal(1440)
    return None


def budget_inputs(
    quote: Any, admission: OwnerRiskAdmission | Mapping[str, Any] | None
) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
    if admission is None:
        return None
    admission_obj = _coerce_admission(admission)
    scope = admission_obj.scope
    day = _decimal(scope.get("price_per_day"), "scope.price_per_day")
    creation = _decimal(scope.get("creation_fee"), "scope.creation_fee")
    other = _decimal(
        scope.get("mandatory_other_fees_assumed"), "scope.mandatory_other_fees_assumed"
    )
    reserve = _decimal(scope.get("reserve"), "scope.reserve")
    fee = _decimal(scope.get("provider_fee_percent"), "scope.provider_fee_percent")
    if None in (day, creation, other, reserve, fee) or fee != Decimal("5"):
        return None
    quote_creation = getattr(quote, "creation_fee", None)
    quote_reserve = getattr(quote, "reserve", None)
    quote_other = getattr(quote, "mandatory_other_fees", None)
    quote_day = getattr(quote, "price_per_day", None)
    if quote_day is not None and quote_day != day:
        return None
    if quote_creation is not None and quote_creation != creation:
        return None
    if quote_reserve is not None and quote_reserve != reserve:
        return None
    if quote_other is not None and quote_other != other:
        return None
    rate = day / Decimal(1440)
    return rate, creation, other, reserve


def _coerce_admission(
    admission: OwnerRiskAdmission | Mapping[str, Any]
) -> OwnerRiskAdmission:
    if isinstance(admission, OwnerRiskAdmission):
        return admission
    if isinstance(admission, Mapping):
        return OwnerRiskAdmission.from_dict(admission)
    raise OwnerRiskError("admission is not a mapping/dataclass")


def strict_blockers(
    quote: Any, *, cap_usd: Decimal, minutes: int = 60
) -> tuple[str, ...]:
    blockers: list[str] = []
    if getattr(quote, "currency", None) != ALLOWED_CURRENCY:
        blockers.append("quote currency is not the approved USD-Blockchain identifier")
    if not getattr(quote, "fresh_verified", False):
        blockers.append("quote is not a fresh verified quote")
    if not getattr(quote, "fees_known", False):
        blockers.append("mandatory provider fees are unknown")
    if not getattr(quote, "tax_fx_known", False):
        blockers.append("tax/FX/payment-fee inputs are unverified")
    if not any(
        getattr(quote, name, None) is not None
        for name in ("price_per_day", "price_per_hour", "price_per_minute")
    ):
        blockers.append("quote has no price projection input")
    if getattr(quote, "creation_fee", None) is None:
        blockers.append("creation fee is unknown")
    if getattr(quote, "mandatory_other_fees", None) is None:
        blockers.append("mandatory other fees are unknown")
    if getattr(quote, "reserve", None) is None:
        blockers.append("budget reserve is unknown")
    minimum = getattr(quote, "min_rental_hours", None)
    if minimum is None:
        blockers.append("minimum rental duration is unknown")
    elif minimum > Decimal("1"):
        blockers.append(
            "minimum rental exceeds the 60-minute session; owner agreement required"
        )
    if getattr(quote, "provider_fee_percent", None) != Decimal("5"):
        blockers.append(
            "provider fee percent must be exactly 5 for the approved profile"
        )
    projected = quote.project_total(minutes)
    if projected is None:
        blockers.append("budget projection is incomplete")
    elif projected > cap_usd:
        blockers.append(
            f"projected total {projected} exceeds the session cap {cap_usd}"
        )
    return tuple(blockers)


def validate_admission(
    admission: OwnerRiskAdmission | Mapping[str, Any],
    *,
    now: datetime | float | None = None,
    allow_expired: bool = False,
    quote_sha256: str | None = None,
) -> tuple[str, ...]:
    issues: list[str] = []
    try:
        value = _coerce_admission(admission)
    except OwnerRiskError as exc:
        return (str(exc),)
    if value.schema != SCHEMA:
        issues.append("owner-risk admission schema is invalid")
    try:
        expected_id = admission_id_for(
            approval_doc=value.approval_doc,
            session_approval_doc=value.session_approval_doc,
            scope=value.scope,
        )
        if value.admission_id != expected_id:
            issues.append("owner-risk admission id does not match consent documents/scope")
        _required_hex(value.approval_doc.get("sha256"), "approval_doc.sha256")
        _required_hex(
            value.session_approval_doc.get("sha256"), "session_approval_doc.sha256"
        )
        if str(value.approval_doc.get("sha256")).lower() != _sha_file(RISK_DOC):
            issues.append("owner-risk approval document hash is not the pinned decision")
        if (
            str(value.session_approval_doc.get("sha256")).lower()
            != _sha_file(SESSION_APPROVAL_DOC)
        ):
            issues.append("owner-risk session approval document hash is not current")
        expected_permission = permission_id_for(
            approval_doc=value.approval_doc,
            session_approval_doc=value.session_approval_doc,
        )
        if value.permission_id is not None and value.permission_id != expected_permission:
            issues.append("owner-risk permission id does not match root documents")
        _required_hex(value.quote.get("sha256"), "quote.sha256")
        _parse_moment(value.quote.get("observed_at"), "quote.observed_at")
        if not isinstance(value.quote.get("observation_origin"), str) or not value.quote[
            "observation_origin"
        ].strip():
            issues.append("owner-risk quote observation_origin is missing")
        scope = value.scope
        if scope.get("purpose") != CONSENT_PURPOSE:
            issues.append("owner-risk scope purpose is not the fixed session02 purpose")
        if _decimal(scope.get("code_cap_usd"), "scope.code_cap") != CODE_CAP_USD:
            issues.append("owner-risk code cap is not 1.00")
        if _decimal(scope.get("owner_ceiling_usd"), "scope.owner_ceiling") != OWNER_CEILING_USD:
            issues.append("owner-risk owner ceiling is not 2.00")
        if _required_int(scope.get("duration_seconds"), "scope.duration", minimum=1) != DURATION_SECONDS:
            issues.append("owner-risk duration is not 3600")
        if _required_int(scope.get("orders"), "scope.orders", minimum=1) != ORDERS:
            issues.append("owner-risk orders is not 1")
        if _required_int(scope.get("generations"), "scope.generations", minimum=1) != GENERATIONS:
            issues.append("owner-risk generations is not 1")
        if _required_int(scope.get("automatic_retries"), "scope.retries", minimum=0) != AUTOMATIC_RETRIES:
            issues.append("owner-risk retries is not 0")
        if _decimal(scope.get("provider_fee_percent"), "scope.provider_fee_percent") != ALLOWED_PROVIDER_FEE_PERCENT:
            issues.append("owner-risk provider fee is not exactly 5")
        if _decimal(scope.get("creation_fee"), "scope.creation_fee") != ALLOWED_CREATION_FEE:
            issues.append("owner-risk creation fee is not 0.10")
        if _decimal(scope.get("reserve"), "scope.reserve") != ALLOWED_RESERVE:
            issues.append("owner-risk reserve is not 0.10")
        if _decimal(scope.get("mandatory_other_fees_assumed"), "scope.other") != ASSUMED_OTHER_FEES:
            issues.append("owner-risk other-fee assumption is not 0")
        if _required_int(scope.get("min_billing_seconds_assumed"), "scope.min", minimum=1) != ASSUMED_MIN_BILLING_SECONDS:
            issues.append("owner-risk minimum-billing assumption is not 60s")
        if _is_dynamic(value):
            issues.extend(_validate_dynamic_admission(value, expected_permission))
        else:
            if scope.get("hoster_id_provenance") != "owner_forwarded_response":
                issues.append("owner-risk hoster provenance is not fixed/owner-forwarded")
            if scope.get("mrl_semantics") != "UNVERIFIED_MAX_OR_MIN":
                issues.append("owner-risk mrl semantics must remain unverified")
            if _required_int(scope.get("server_id"), "scope.server_id", minimum=1) != ALLOWED_SERVER_ID:
                issues.append("owner-risk server is not the approved 111947")
            if _required_int(scope.get("hoster_id"), "scope.hoster_id", minimum=1) != ALLOWED_HOSTER_ID:
                issues.append("owner-risk hoster is not the approved 46993")
            if scope.get("currency") != ALLOWED_CURRENCY:
                issues.append("owner-risk currency is not USD-Blockchain")
            if _decimal(scope.get("price_per_day"), "scope.price_per_day") != ALLOWED_PRICE_PER_DAY:
                issues.append("owner-risk day rate is not exactly 12")
            if scope.get("provider_reported_mrl_seconds") != PROVIDER_REPORTED_MRL_SECONDS:
                issues.append("owner-risk raw mrl is not the recorded 336h fact")
            projection = value.projection
            total = _decimal(projection.get("total_60m"), "projection.total_60m")
            if total != TOTAL_60M:
                issues.append("owner-risk 60m projection is not 0.725")
        if value.verification.get("provider_billing_verified") is not False:
            issues.append("owner-risk must not claim provider billing verification")
        if value.verification.get("tax_fx_verified") is not False:
            issues.append("owner-risk must not claim tax/FX verification")
        if value.verification.get("minimum_billing_verified") is not False:
            issues.append("owner-risk must not claim minimum billing verification")
        if value.verification.get("financial_flags_unchanged") is not True:
            issues.append("owner-risk must preserve unknown financial flags")
        if not isinstance(value.verification.get("financial_status"), str):
            issues.append("owner-risk financial_status is missing")
        if value.expires_at <= value.created_at:
            issues.append("owner-risk expires_at must be after created_at")
        current = _now(now)
        if not allow_expired and current >= value.expires_at:
            issues.append("owner-risk admission has expired for new work")
        if quote_sha256 is not None and quote_sha256.lower() != str(
            value.quote.get("sha256")
        ).lower():
            issues.append("owner-risk quote hash differs from the current quote")
    except OwnerRiskError as exc:
        issues.append(str(exc))
    return tuple(issues)


def _validate_dynamic_admission(
    value: OwnerRiskAdmission, expected_permission: str
) -> list[str]:
    issues: list[str] = []
    scope = value.scope
    approval = value.approval
    assert approval is not None
    allowed = {
        "schema",
        "approved_by",
        "approved_at",
        "explicit",
        "confirmation",
        "proposal_sha256",
        "quote_sha256",
        "quote_observed_at",
        "risk_doc_sha256",
        "session_approval_doc_sha256",
        "scope_sha256",
        "permission_id",
    }
    if set(approval) != allowed:
        issues.append("owner-risk explicit approval has unknown/missing keys")
        return issues
    if approval.get("schema") != APPROVAL_SCHEMA:
        issues.append("owner-risk explicit approval schema is invalid")
    if approval.get("explicit") is not True:
        issues.append("owner-risk explicit approval flag must be true")
    if approval.get("confirmation") != APPROVAL_CONFIRMATION:
        issues.append("owner-risk explicit approval confirmation phrase is invalid")
    if not isinstance(approval.get("approved_by"), str) or not approval["approved_by"].strip():
        issues.append("owner-risk explicit approval owner is missing")
    try:
        approved_at = _parse_moment(approval.get("approved_at"), "approval.approved_at")
        if approved_at > value.created_at + timedelta(seconds=5):
            issues.append("owner-risk explicit approval time is after admission creation")
    except OwnerRiskError as exc:
        issues.append(str(exc))
    if approval.get("permission_id") != expected_permission:
        issues.append("owner-risk explicit approval permission id mismatch")
    if approval.get("risk_doc_sha256") != str(value.approval_doc.get("sha256")).lower():
        issues.append("owner-risk explicit approval risk-document hash mismatch")
    if approval.get("session_approval_doc_sha256") != str(
        value.session_approval_doc.get("sha256")
    ).lower():
        issues.append("owner-risk explicit approval session-document hash mismatch")
    if approval.get("quote_sha256") != str(value.quote.get("sha256")).lower():
        issues.append("owner-risk explicit approval quote hash mismatch")
    try:
        approved_observed = _parse_moment(
            approval.get("quote_observed_at"), "approval.quote_observed_at"
        )
        admission_observed = _parse_moment(
            value.quote.get("observed_at"), "quote.observed_at"
        )
        if approved_observed != admission_observed:
            issues.append("owner-risk explicit approval quote timestamp mismatch")
    except OwnerRiskError as exc:
        issues.append(str(exc))
    if approval.get("proposal_sha256") != proposal_digest_for_admission(value):
        issues.append("owner-risk explicit approval proposal hash mismatch")
    if approval.get("scope_sha256") != _sha_mapping(dict(scope)):
        issues.append("owner-risk explicit approval scope hash mismatch")

    if scope.get("hoster_id_provenance") != "fresh_marketplace_quote":
        issues.append("owner-risk dynamic hoster provenance is not a fresh marketplace quote")
    if scope.get("mrl_semantics") not in (
        "UNVERIFIED_MAX_OR_MIN",
        "MAXIMUM_RENTAL_LENGTH",
    ):
        issues.append("owner-risk dynamic mrl semantics is not recognized")
    if _required_int(scope.get("server_id"), "scope.server_id", minimum=1) <= 0:
        issues.append("owner-risk dynamic server is invalid")
    if _required_int(scope.get("hoster_id"), "scope.hoster_id", minimum=1) <= 0:
        issues.append("owner-risk dynamic hoster is invalid")
    currency = scope.get("currency")
    if not isinstance(currency, str) or not currency.strip():
        issues.append("owner-risk dynamic currency is missing")
    elif currency != ALLOWED_CURRENCY:
        issues.append(
            "owner-risk dynamic currency must be exactly USD-Blockchain; "
            "crypto conversion/FX is not authorized"
        )

    day = _decimal(scope.get("price_per_day"), "scope.price_per_day")
    if day is None or day <= 0:
        issues.append("owner-risk dynamic day rate must be positive")
    raw_mrl = scope.get("provider_reported_mrl_seconds")
    if isinstance(raw_mrl, bool) or not isinstance(raw_mrl, int) or raw_mrl <= 0:
        issues.append("owner-risk dynamic raw mrl must be a positive integer")
    if day is not None and day > 0:
        total = (
            day * Decimal(60) / Decimal(1440) * (Decimal(1) + ALLOWED_PROVIDER_FEE_PERCENT / Decimal(100))
            + ALLOWED_CREATION_FEE
            + ALLOWED_RESERVE
            + ASSUMED_OTHER_FEES
        )
        projection_total = _decimal(
            value.projection.get("total_60m"), "projection.total_60m"
        )
        if projection_total != total:
            issues.append("owner-risk dynamic projection does not match the scoped rate")
        if total > CODE_CAP_USD:
            issues.append("owner-risk dynamic projection exceeds the code cap")
        rental = day * Decimal(60) / Decimal(1440)
        if _decimal(value.projection.get("rental_60m"), "projection.rental_60m") != rental:
            issues.append("owner-risk dynamic rental projection mismatch")
    return issues


def policy_blockers(
    quote: Any,
    admission: OwnerRiskAdmission | Mapping[str, Any] | None,
    *,
    cap_usd: Decimal,
    minutes: int = 60,
    now: datetime | float | None = None,
    allow_expired: bool = False,
    quote_sha256: str | None = None,
) -> tuple[str, ...]:
    """One policy entry: strict when no admission, scoped otherwise."""

    if admission is None:
        return strict_blockers(quote, cap_usd=cap_usd, minutes=minutes)
    blockers = list(
        validate_admission(
            admission,
            now=now,
            allow_expired=allow_expired,
            quote_sha256=quote_sha256,
        )
    )
    if blockers:
        return tuple(blockers)
    value = _coerce_admission(admission)
    dynamic = _is_dynamic(value)
    scope = value.scope
    expected_currency = (
        str(scope.get("currency")) if dynamic else ALLOWED_CURRENCY
    )
    expected_day = _decimal(
        scope.get("price_per_day"), "scope.price_per_day"
    )
    if getattr(quote, "currency", None) != expected_currency:
        blockers.append("quote currency does not match the scoped admission")
    if not getattr(quote, "fresh_verified", False):
        blockers.append("quote is not a fresh verified quote")
    day = getattr(quote, "price_per_day", None)
    if day is None or day != expected_day:
        blockers.append("quote exact scoped day rate is missing or changed")
    if getattr(quote, "provider_fee_percent", None) not in (
        None,
        ALLOWED_PROVIDER_FEE_PERCENT,
    ):
        blockers.append("provider fee percent must be exactly 5 for the approved profile")
    if getattr(quote, "fees_known", False):
        blockers.append("financial unknown flags must remain false for owner-risk admission")
    if getattr(quote, "tax_fx_known", False):
        blockers.append("tax/FX flag must remain false for owner-risk admission")
    minimum = getattr(quote, "min_rental_hours", None)
    if minimum is not None and minimum * Decimal(3600) > Decimal(ASSUMED_MIN_BILLING_SECONDS):
        blockers.append("known minimum rental exceeds the 60-minute session")
    creation = getattr(quote, "creation_fee", None)
    if creation is not None and creation != ALLOWED_CREATION_FEE:
        blockers.append("known creation fee differs from the approved 0.10")
    reserve = getattr(quote, "reserve", None)
    if reserve is not None and reserve != ALLOWED_RESERVE:
        blockers.append("known reserve differs from the approved 0.10")
    other = getattr(quote, "mandatory_other_fees", None)
    if other is not None and other != ASSUMED_OTHER_FEES:
        blockers.append("known mandatory other fees are outside this risk scope")
    if cap_usd <= 0 or cap_usd > CODE_CAP_USD:
        blockers.append("session cap must be positive and at most USD 1.00")
    projected = effective_projection(quote, value, minutes=minutes)
    if projected is None:
        blockers.append("owner-risk budget projection is incomplete")
    elif projected > cap_usd:
        blockers.append(f"projected total {projected} exceeds the session cap {cap_usd}")
    return tuple(blockers)


def _quote_has_secret_keys(quote: Mapping[str, Any]) -> bool:
    try:
        _assert_no_secret_keys(quote, where="quote")
    except OwnerRiskError:
        return True
    return False


def create_pending(*, pending_quote: Path | str, output_path: Path | str) -> dict[str, Any]:
    quote_path = _assert_safe_path(Path(pending_quote), must_exist=True)
    data = _read_json(quote_path)
    _assert_no_secret_keys(data, where="pending quote")
    quote = data.get("quote") if isinstance(data.get("quote"), Mapping) else data
    if not isinstance(quote, Mapping):
        raise OwnerRiskError("pending quote must contain an object")
    day = _quote_decimal(quote, "price_per_day")
    hour = _quote_decimal(quote, "price_per_hour")
    minute = _quote_decimal(quote, "price_per_minute")
    creation = _quote_decimal(quote, "creation_fee")
    reserve = _quote_decimal(quote, "reserve")
    hourly_rates = [
        value
        for value in (
            day / Decimal(24) if day is not None else None,
            hour,
            minute * Decimal(60) if minute is not None else None,
        )
        if value is not None
    ]
    observed_total = None
    if hourly_rates:
        rental_60m = max(hourly_rates)
        observed_total = (
            rental_60m * Decimal("1.05")
            + (creation if creation is not None else Decimal("0"))
            + (reserve if reserve is not None else Decimal("0"))
        )
    payload = {
        "schema": PENDING_SCHEMA,
        "status": "PENDING_QUOTE_NON_AUTHORIZING",
        "authorizes_session": False,
        "provider_billing_verified": False,
        "financial_status": "OWNER_RISK_PENDING_NO_AUTH",
        "observed_at": data.get("observed_at"),
        "quote_path": str(quote_path),
        "quote_sha256": _sha_file(quote_path),
        "quote": {str(key): _jsonable(value) for key, value in quote.items()},
        "observed_projection_60m_usd": observed_total,
        "policy_forecast_60m_usd": TOTAL_60M,
        "assumptions": [
            "exact 12 USD/day + known 5% renter fee",
            "creation 0.10 assumption if not verified",
            "reserve 0.10",
            "additional fees/tax/FX assumed 0 and disclosed, not factual",
            "minimum billing 60s assumed, not verified",
            "mrl 336h recorded as raw fact, semantics unverified",
        ],
        "provenance": {
            "kind": "pending_quote_read_only",
            "note": "This is not a fresh quote and does not authorize session-bind/Pay.",
        },
    }
    target = Path(output_path)
    text = _json_text(payload)
    if target.exists():
        existing = _read_json(target)
        if existing == json.loads(text):
            return existing
        raise OwnerRiskError("pending artifact already exists and is immutable")
    _atomic_create(target, text)
    return payload


def create_admission(
    *,
    quote: Path | str,
    output_path: Path | str,
    approval_doc: Path | str = RISK_DOC,
    session_approval_doc: Path | str = SESSION_APPROVAL_DOC,
    server_id: int = ALLOWED_SERVER_ID,
    hoster_id: int = ALLOWED_HOSTER_ID,
    valid_hours: int = 2,
    now: datetime | float | None = None,
    observation_origin: str | None = None,
    expected_quote_sha256: str | None = None,
) -> OwnerRiskAdmission:
    current = _now(now)
    if isinstance(valid_hours, bool) or not isinstance(valid_hours, int) or not 1 <= valid_hours <= MAX_ADMISSION_HOURS:
        raise OwnerRiskError("valid_hours must be between 1 and 24")
    if server_id != ALLOWED_SERVER_ID:
        raise OwnerRiskError("only the approved server 111947 may be admitted")
    if hoster_id != ALLOWED_HOSTER_ID:
        raise OwnerRiskError("only the approved hoster 46993 may be admitted")
    risk_path = _assert_safe_path(Path(approval_doc), must_exist=True).resolve()
    session_path = _assert_safe_path(Path(session_approval_doc), must_exist=True).resolve()
    if risk_path != RISK_DOC.resolve():
        raise OwnerRiskError("approval document is not the fixed owner-risk document")
    if session_path != SESSION_APPROVAL_DOC.resolve():
        raise OwnerRiskError("session approval document is not the current session02 document")
    quote_path = _assert_safe_path(Path(quote), must_exist=True).resolve()
    quote_sha = _sha_file(quote_path)
    if expected_quote_sha256 is not None:
        if not isinstance(expected_quote_sha256, str) or len(expected_quote_sha256) != 64:
            raise OwnerRiskError("expected quote hash must be 64 hex characters")
        if quote_sha != expected_quote_sha256.lower():
            raise OwnerRiskError("fresh machine quote hash differs from the manager hash")
    quote_data, observed, provenance = _read_quote_document(quote_path)
    age = (current - observed).total_seconds()
    if age < -60 or age > QUOTE_MAX_AGE_SECONDS:
        raise OwnerRiskError("fresh machine quote is stale or has a future timestamp")
    _validate_quote_for_admission(quote_data)
    origin = observation_origin or (
        provenance.get("observation_origin")
        or provenance.get("kind")
        or provenance.get("source")
    )
    if not isinstance(origin, str) or not origin.strip():
        raise OwnerRiskError("observation_origin is required")
    approval_block = {
        "path": RISK_DOC_REL,
        "sha256": _sha_file(risk_path),
    }
    session_block = {
        "path": SESSION_APPROVAL_DOC_REL,
        "sha256": _sha_file(session_path),
    }
    scope = {
        "purpose": CONSENT_PURPOSE,
        "server_id": server_id,
        "hoster_id": hoster_id,
        "hoster_id_provenance": "owner_forwarded_response",
        "currency": ALLOWED_CURRENCY,
        "price_per_day": str(ALLOWED_PRICE_PER_DAY),
        "provider_fee_percent": str(ALLOWED_PROVIDER_FEE_PERCENT),
        "creation_fee": str(ALLOWED_CREATION_FEE),
        "reserve": str(ALLOWED_RESERVE),
        "mandatory_other_fees_assumed": str(ASSUMED_OTHER_FEES),
        "min_billing_seconds_assumed": ASSUMED_MIN_BILLING_SECONDS,
        "provider_reported_mrl_seconds": PROVIDER_REPORTED_MRL_SECONDS,
        "mrl_semantics": "UNVERIFIED_MAX_OR_MIN",
        "code_cap_usd": str(CODE_CAP_USD),
        "owner_ceiling_usd": str(OWNER_CEILING_USD),
        "duration_seconds": DURATION_SECONDS,
        "orders": ORDERS,
        "generations": GENERATIONS,
        "automatic_retries": AUTOMATIC_RETRIES,
    }
    # Build the fixed 60-minute forecast from the validated scope constants.
    rental = ALLOWED_PRICE_PER_DAY * Decimal(60) / Decimal(1440)
    renter_fee = rental * ALLOWED_PROVIDER_FEE_PERCENT / Decimal(100)
    projection = {
        "rental_60m": str(rental),
        "renter_fee_60m": str(renter_fee),
        "creation_fee": str(ALLOWED_CREATION_FEE),
        "reserve": str(ALLOWED_RESERVE),
        "other_fees_assumed": str(ASSUMED_OTHER_FEES),
        "total_60m": str(TOTAL_60M),
        "assumptions": [
            f"minimum billing {ASSUMED_MIN_BILLING_SECONDS}s assumed, not verified",
            "additional mandatory fees assumed 0 and covered by reserve, not factual",
            "tax/FX/payment fees assumed 0 and excluded, not factual",
            "creation fee 0.10 used from approved checkout/assumption",
            "mrl 336h recorded as raw fact with unverified semantics; not used as minimum",
        ],
    }
    admission_id = admission_id_for(
        approval_doc=approval_block,
        session_approval_doc=session_block,
        scope=scope,
    )
    payload = {
        "schema": SCHEMA,
        "admission_id": admission_id,
        "permission_id": permission_id_for(
            approval_doc=approval_block, session_approval_doc=session_block
        ),
        "created_at": _iso(current),
        "expires_at": _iso(current + timedelta(hours=valid_hours)),
        "approval_doc": approval_block,
        "session_approval_doc": session_block,
        "quote": {
            "path": str(quote_path),
            "sha256": _sha_file(quote_path),
            "observed_at": _iso(observed),
            "observation_origin": origin,
        },
        "scope": scope,
        "projection": projection,
        "financial_unknowns": [
            "minimum_billed_duration",
            "additional_mandatory_fees",
            "tax",
            "fx",
            "payment_fees",
        ],
        "verification": {
            "provider_billing_verified": False,
            "tax_fx_verified": False,
            "minimum_billing_verified": False,
            "financial_flags_unchanged": True,
            "financial_status": "OWNER_ACCEPTED_FINANCIAL_RISK",
            "hard_cap_guaranteed": False,
        },
        "provenance": {
            "kind": "owner_forwarded_official_response+owner_attestation",
            "not_provider_api_audit": True,
            "observation_origin": origin,
            "mrl_raw_seconds": PROVIDER_REPORTED_MRL_SECONDS,
            "note": "Owner accepted billing-minimum/fees/tax/FX uncertainty; not a price/identity/deadline change and not a hard billing cap.",
        },
    }
    value = OwnerRiskAdmission.from_dict(payload)
    issues = validate_admission(value, now=current)
    if issues:
        raise OwnerRiskError("owner-risk admission refused: " + "; ".join(issues))
    target = Path(output_path)
    text = _json_text(value.to_dict())
    if target.exists():
        existing = load_admission(target, now=current, allow_expired=True, verify_docs=True, verify_quote=True)
        if (
            existing.admission_id == value.admission_id
            and existing.quote.get("sha256") == value.quote.get("sha256")
            and existing.scope == value.scope
        ):
            return existing
        raise OwnerRiskError(
            "same consent already has an immutable admission with different quote/scope"
        )
    if claim_exists(
        permission_id_for(
            approval_doc=approval_block, session_approval_doc=session_block
        )
    ):
        raise OwnerRiskError("owner-risk root permission has already been claimed")
    _atomic_create(target, text)
    return load_admission(target, now=current)


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _projection_for_day(day: Decimal) -> dict[str, Any]:
    rental = day * Decimal(60) / Decimal(1440)
    renter_fee = rental * ALLOWED_PROVIDER_FEE_PERCENT / Decimal(100)
    total = rental + renter_fee + ALLOWED_CREATION_FEE + ALLOWED_RESERVE + ASSUMED_OTHER_FEES
    return {
        "rental_60m": _decimal_text(rental),
        "renter_fee_60m": _decimal_text(renter_fee),
        "creation_fee": str(ALLOWED_CREATION_FEE),
        "reserve": str(ALLOWED_RESERVE),
        "other_fees_assumed": str(ASSUMED_OTHER_FEES),
        "total_60m": _decimal_text(total),

        "assumptions": [
            f"minimum billing {ASSUMED_MIN_BILLING_SECONDS}s assumed, not verified",
            "additional mandatory fees assumed 0 and covered by reserve, not factual",
            "tax/FX/payment fees assumed 0 and excluded, not factual",
            "creation fee 0.10 used from approved checkout/assumption",
            "raw marketplace mrl is a maximum rental length; not used as a minimum bill",
        ],
    }


def _read_scoped_quote_document(
    path: Path, *, now: datetime | float | None = None
) -> tuple[dict[str, Any], dict[str, Any], datetime, int, int, str, Decimal, int, str]:
    data = _read_json(path)
    _assert_no_secret_keys(data, where="marketplace quote")
    if data.get("schema") != MARKET_QUOTE_SCHEMA:
        raise OwnerRiskError("fresh marketplace quote schema is not recognized")
    if data.get("status") != "FRESH_MARKETPLACE_QUOTE_READ_ONLY":
        raise OwnerRiskError("fresh marketplace quote status is not read-only fresh")
    if data.get("authorizes_session") is not False:
        raise OwnerRiskError("fresh marketplace quote must not authorize a session")
    if data.get("provider_billing_verified") is not False:
        raise OwnerRiskError("fresh marketplace quote must not claim billing verification")
    if data.get("observation_origin") != MARKET_OBSERVATION_ORIGIN:
        raise OwnerRiskError("fresh marketplace quote observation origin is not the collector")
    requested = _parse_moment(data.get("requested_at"), "quote.requested_at")
    observed = _parse_moment(data.get("observed_at"), "quote.observed_at")
    if data.get("requested_at") != _iso(requested) or data.get("observed_at") != _iso(observed):
        raise OwnerRiskError("fresh marketplace quote timestamps are not canonical UTC")
    current = _now(now)
    if observed < requested:
        raise OwnerRiskError("fresh marketplace quote observed_at precedes requested_at")
    if (observed - requested).total_seconds() > 300:
        raise OwnerRiskError("fresh marketplace quote request window is too long")
    age = (current - observed).total_seconds()
    if age < -60 or age > QUOTE_MAX_AGE_SECONDS:
        raise OwnerRiskError("fresh marketplace quote is stale or has a future timestamp")
    if (current - requested).total_seconds() < -60:
        raise OwnerRiskError("fresh marketplace quote requested_at is in the future")
    server_id = _required_int(data.get("server_id"), "quote.server_id", minimum=1)
    hoster_id = _required_int(data.get("hoster_id"), "quote.hoster_id", minimum=1)
    currency = _required_str(data.get("currency"), "quote.currency")
    day = _decimal(data.get("price_per_day"), "quote.price_per_day")
    if day is None or day <= 0:
        raise OwnerRiskError("quote.price_per_day must be a positive decimal")
    raw_mrl = data.get("mrl_raw_seconds")
    if isinstance(raw_mrl, bool) or not isinstance(raw_mrl, int) or raw_mrl <= 0:
        raise OwnerRiskError("quote.mrl_raw_seconds must be a positive integer")
    semantics = _required_str(data.get("mrl_semantics"), "quote.mrl_semantics")
    if semantics not in ("UNVERIFIED_MAX_OR_MIN", "MAXIMUM_RENTAL_LENGTH"):
        raise OwnerRiskError("quote.mrl_semantics is not recognized")
    facts_sha = _required_hex(data.get("facts_sha256"), "quote.facts_sha256")
    mrl_hours = data.get("mrl_raw_hours")
    if isinstance(mrl_hours, bool) or not isinstance(mrl_hours, int) or mrl_hours <= 0:
        raise OwnerRiskError("quote.mrl_raw_hours must be a positive integer")
    if mrl_hours * 3600 != raw_mrl:
        raise OwnerRiskError("quote raw mrl hours/seconds are inconsistent")
    availability = data.get("availability")
    if not isinstance(availability, Mapping):
        raise OwnerRiskError("quote.availability must be an object")
    if availability.get("listed") is not True:
        raise OwnerRiskError("marketplace quote is not listed as available")
    if availability.get("rented") is not False:
        raise OwnerRiskError("marketplace server is rented/unavailable")
    online = availability.get("online")
    if online is not None and online is not True:
        raise OwnerRiskError("marketplace server is not online")
    expected_online_field = (
        "marketplace" if online is not None else "not_exposed_by_marketplace"
    )
    if availability.get("online_source") != expected_online_field:
        raise OwnerRiskError("marketplace availability online source is inconsistent")
    provenance = data.get("provenance")
    required_provenance = {
        "kind",
        "collector_id",
        "collector_version",
        "method",
        "base_url",
        "endpoint",
        "path",
        "http_status",
        "authenticated",
        "price_only",
        "fees_observed",
        "provider_response_persisted",
        "not_provider_billing_audit",
        "observation_origin",
        "freshness",
        "online_field",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != required_provenance:
        raise OwnerRiskError("fresh marketplace collector provenance is incomplete/unknown")
    expected_provenance = {
        "kind": "clore_marketplace_get",
        "collector_id": MARKET_COLLECTOR_ID,
        "collector_version": MARKET_COLLECTOR_VERSION,
        "method": MARKET_METHOD,
        "base_url": MARKET_BASE_URL,
        "endpoint": MARKET_ENDPOINT,
        "path": MARKET_PATH,
        "http_status": 200,
        "authenticated": True,
        "price_only": True,
        "fees_observed": False,
        "provider_response_persisted": False,
        "not_provider_billing_audit": True,
        "observation_origin": MARKET_OBSERVATION_ORIGIN,
        "freshness": MARKET_FRESHNESS,
        "online_field": expected_online_field,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise OwnerRiskError(
                f"fresh marketplace collector provenance {key} is not the fixed contract"
            )
    aliases = [
        data.get("observation_origin"),
        provenance.get("observation_origin"),
        provenance.get("kind"),
    ]
    if "source" in provenance:
        aliases.append(provenance.get("source"))
    if any(
        alias not in (None, MARKET_OBSERVATION_ORIGIN, "clore_marketplace_get")
        for alias in aliases
    ):
        raise OwnerRiskError("fresh marketplace observation aliases conflict")
    quote = data.get("quote")
    if not isinstance(quote, Mapping):
        raise OwnerRiskError("fresh marketplace quote.quote must be an object")
    proposal = data.get("proposal")
    if not isinstance(proposal, Mapping):
        raise OwnerRiskError("fresh marketplace quote.proposal must be an object")
    public_facts = {
        "server_id": server_id,
        "hoster_id": hoster_id,
        "mrl_raw_hours": mrl_hours,
        "mrl_raw_seconds": raw_mrl,
        "mrl_semantics": semantics,
        "currency": currency,
        "price_per_day": _decimal_text(day),
        "availability": dict(availability),
    }
    collector_core = {
        "collector_id": MARKET_COLLECTOR_ID,
        "collector_version": MARKET_COLLECTOR_VERSION,
        "method": MARKET_METHOD,
        "base_url": MARKET_BASE_URL,
        "endpoint": MARKET_ENDPOINT,
        "path": MARKET_PATH,
    }
    integrity = {
        "facts": public_facts,
        "quote": dict(quote),
        "proposal": dict(proposal),
        "requested_at": data.get("requested_at"),
        "observed_at": data.get("observed_at"),
        "observation_origin": MARKET_OBSERVATION_ORIGIN,
        "collector": collector_core,
    }
    if facts_sha != _sha_mapping(integrity):
        raise OwnerRiskError(
            "fresh marketplace facts hash does not cover the quote/proposal contract"
        )
    if proposal.get("schema") != PROPOSAL_SCHEMA:
        raise OwnerRiskError("fresh marketplace proposal schema is not recognized")
    if proposal.get("authorizes_session") is not False:
        raise OwnerRiskError("fresh marketplace proposal must not authorize a session")
    for key, expected in (
        ("server_id", server_id),
        ("hoster_id", hoster_id),
        ("currency", currency),
        ("price_per_day", _decimal_text(day)),
        ("provider_reported_mrl_seconds", raw_mrl),
        ("mrl_semantics", semantics),
    ):
        if proposal.get(key) != expected:
            raise OwnerRiskError(
                f"fresh marketplace proposal {key} differs from the quote facts"
            )
    proposal_projection = proposal.get("projection")
    if not isinstance(proposal_projection, Mapping):
        raise OwnerRiskError("fresh marketplace proposal projection is missing")
    expected_projection = _projection_for_day(day)
    if _decimal(
        proposal_projection.get("total_60m"), "proposal.projection.total_60m"
    ) != _decimal(expected_projection.get("total_60m"), "projection.total_60m"):
        raise OwnerRiskError("fresh marketplace proposal projection differs from the rate")
    expected_fit = _decimal(
        expected_projection.get("total_60m"), "projection.total_60m"
    ) <= CODE_CAP_USD
    if proposal.get("fits_code_cap") is not expected_fit:
        raise OwnerRiskError("fresh marketplace proposal code-cap fit is inconsistent")
    _validate_scoped_quote_fields(quote, currency=currency, day=day)
    return (
        data,
        dict(quote),
        observed,
        server_id,
        hoster_id,
        currency,
        day,
        raw_mrl,
        semantics,
    )



def _validate_scoped_quote_fields(
    quote: Mapping[str, Any], *, currency: str, day: Decimal
) -> None:
    if quote.get("currency") != currency:
        raise OwnerRiskError("quote currency differs from the marketplace facts")
    if quote.get("fresh_verified") is not True:
        raise OwnerRiskError("fresh marketplace quote must set fresh_verified=true")
    if quote.get("fees_known") is not False or quote.get("tax_fx_known") is not False:
        raise OwnerRiskError("financial unknown flags must remain false")
    if _decimal(quote.get("price_per_day"), "quote.price_per_day") != day:
        raise OwnerRiskError("quote daily rate differs from the marketplace facts")
    fee = _decimal(
        quote.get("provider_fee_percent"),
        "quote.provider_fee_percent",
        optional=True,
    )
    if fee is not None and fee != ALLOWED_PROVIDER_FEE_PERCENT:
        raise OwnerRiskError("quote provider fee is not exactly 5")
    creation = _decimal(
        quote.get("creation_fee"), "quote.creation_fee", optional=True
    )
    if creation is not None and creation != ALLOWED_CREATION_FEE:
        raise OwnerRiskError("quote creation fee differs from the approved assumption")
    reserve = _decimal(
        quote.get("reserve"), "quote.reserve", optional=True
    )
    if reserve is not None and reserve != ALLOWED_RESERVE:
        raise OwnerRiskError("quote reserve differs from the approved assumption")
    other = _decimal(
        quote.get("mandatory_other_fees"),
        "quote.mandatory_other_fees",
        optional=True,
    )
    if other is not None and other != ASSUMED_OTHER_FEES:
        raise OwnerRiskError("quote mandatory other fees are outside this risk scope")
    minimum = _decimal(
        quote.get("min_rental_hours"), "quote.min_rental_hours", optional=True
    )
    if minimum is not None and minimum * Decimal(3600) > Decimal(
        ASSUMED_MIN_BILLING_SECONDS
    ):
        raise OwnerRiskError("known minimum rental exceeds the 60-minute session")
    for name in ("price_per_hour", "price_per_minute"):
        value = _decimal(quote.get(name), f"quote.{name}", optional=True)

        if value is None:
            continue
        if name == "price_per_hour":
            implied = value * Decimal(24)
        else:
            implied = value * Decimal(1440)
        if abs(implied - day) > Decimal("0.000001"):
            raise OwnerRiskError(f"quote.{name} is inconsistent with the daily rate")


def create_scoped_admission(
    *,
    quote: Path | str,
    output_path: Path | str,
    server_id: int,
    hoster_id: int | None = None,
    currency: str = ALLOWED_CURRENCY,
    approval_doc: Path | str = RISK_DOC,
    session_approval_doc: Path | str = SESSION_APPROVAL_DOC,
    valid_hours: int = 2,
    now: datetime | float | None = None,
    approved_by: str = "owner",
    approval_confirm: str | None = None,
    expected_quote_sha256: str | None = None,
) -> OwnerRiskAdmission:
    """Create one explicit, data-driven scoped proposal admission.

    This is the dynamic replacement for the historical fixed 111947/12/336h
    admission. It never creates a claim, never calls a provider and never
    approves anything implicitly. A new server/rate/quote requires a new
    explicit call with the confirmation phrase.
    """

    if approval_confirm != APPROVAL_CONFIRMATION:
        raise OwnerRiskError(
            "explicit scoped approval confirmation phrase is required"
        )
    current = _now(now)
    if isinstance(valid_hours, bool) or not isinstance(valid_hours, int) or not 1 <= valid_hours <= MAX_ADMISSION_HOURS:
        raise OwnerRiskError("valid_hours must be between 1 and 24")
    server_id = _required_int(server_id, "server_id", minimum=1)
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise OwnerRiskError("approved_by must be a non-empty owner label")
    if not isinstance(currency, str) or not currency.strip():
        raise OwnerRiskError("currency must be a non-empty string")
    risk_path = _assert_safe_path(Path(approval_doc), must_exist=True).resolve()
    session_path = _assert_safe_path(Path(session_approval_doc), must_exist=True).resolve()
    if risk_path != RISK_DOC.resolve():
        raise OwnerRiskError("approval document is not the fixed owner-risk document")
    if session_path != SESSION_APPROVAL_DOC.resolve():
        raise OwnerRiskError("session approval document is not the current session02 document")
    quote_path = _assert_safe_path(Path(quote), must_exist=True).resolve()
    quote_sha = _sha_file(quote_path)
    if expected_quote_sha256 is not None:
        if not isinstance(expected_quote_sha256, str) or len(expected_quote_sha256) != 64:
            raise OwnerRiskError("expected quote hash must be 64 hex characters")
        if quote_sha != expected_quote_sha256.lower():
            raise OwnerRiskError("fresh marketplace quote hash differs from the supplied hash")
    (
        data,
        quote_data,
        observed,
        quote_server,
        quote_hoster,
        quote_currency,
        day,
        raw_mrl,
        semantics,
    ) = _read_scoped_quote_document(quote_path, now=current)
    if quote_server != server_id:
        raise OwnerRiskError("fresh marketplace quote is for a different server id")
    if hoster_id is not None and quote_hoster != _required_int(
        hoster_id, "hoster_id", minimum=1
    ):
        raise OwnerRiskError("fresh marketplace quote is for a different hoster id")
    if quote_currency != currency.strip():
        raise OwnerRiskError("fresh marketplace quote currency differs from the request")
    if quote_currency != ALLOWED_CURRENCY:
        raise OwnerRiskError(
            "scoped admission supports only the exact USD-Blockchain currency; "
            "crypto conversion/FX is not authorized"
        )

    age = (current - observed).total_seconds()
    if age < -60 or age > QUOTE_MAX_AGE_SECONDS:
        raise OwnerRiskError("fresh marketplace quote is stale or has a future timestamp")
    projection = _projection_for_day(day)
    total = _decimal(projection.get("total_60m"), "projection.total_60m")
    if total is None or total > CODE_CAP_USD:
        raise OwnerRiskError(
            "scoped 60-minute projection exceeds the 1.00 USD code cap; refusing to lower the rate/currency/mrl"
        )
    provenance_raw = data.get("provenance")
    if not isinstance(provenance_raw, Mapping):
        provenance_raw = {}
    origin = (
        data.get("observation_origin")
        or provenance_raw.get("observation_origin")
        or provenance_raw.get("kind")
    )
    if not isinstance(origin, str) or not origin.strip():
        raise OwnerRiskError("fresh marketplace quote observation origin is missing")
    approval_block = {
        "path": RISK_DOC_REL,
        "sha256": _sha_file(risk_path),
    }
    session_block = {
        "path": SESSION_APPROVAL_DOC_REL,
        "sha256": _sha_file(session_path),
    }
    scope = {
        "purpose": CONSENT_PURPOSE,
        "server_id": server_id,
        "hoster_id": quote_hoster,
        "hoster_id_provenance": "fresh_marketplace_quote",
        "currency": quote_currency,
        "price_per_day": format(day.normalize(), "f"),
        "provider_fee_percent": str(ALLOWED_PROVIDER_FEE_PERCENT),
        "creation_fee": str(ALLOWED_CREATION_FEE),
        "reserve": str(ALLOWED_RESERVE),
        "mandatory_other_fees_assumed": str(ASSUMED_OTHER_FEES),
        "min_billing_seconds_assumed": ASSUMED_MIN_BILLING_SECONDS,
        "provider_reported_mrl_seconds": raw_mrl,
        "mrl_semantics": semantics,
        "code_cap_usd": str(CODE_CAP_USD),
        "owner_ceiling_usd": str(OWNER_CEILING_USD),
        "duration_seconds": DURATION_SECONDS,
        "orders": ORDERS,
        "generations": GENERATIONS,
        "automatic_retries": AUTOMATIC_RETRIES,
    }
    quote_block = {
        "path": str(quote_path),
        "sha256": quote_sha,
        "observed_at": _iso(observed),
        "observation_origin": origin,
    }
    financial_unknowns = [
        "minimum_billed_duration",
        "additional_mandatory_fees",
        "tax",
        "fx",
        "payment_fees",
    ]
    verification = {
        "provider_billing_verified": False,
        "tax_fx_verified": False,
        "minimum_billing_verified": False,
        "financial_flags_unchanged": True,
        "financial_status": "OWNER_ACCEPTED_FINANCIAL_RISK",
        "hard_cap_guaranteed": False,
    }
    provenance = {
        "kind": "fresh_marketplace_quote+explicit_owner_approval",
        "not_provider_billing_audit": True,
        "observation_origin": origin,
        "marketplace_facts_sha256": str(data.get("facts_sha256")),
        "mrl_raw_seconds": raw_mrl,
        "mrl_semantics": semantics,
        "note": "Explicit local owner attestation over a fresh GET-only marketplace quote; not a signature or billing audit.",
    }
    created_iso = _iso(current)
    expires_iso = _iso(current + timedelta(hours=valid_hours))
    proposal_payload = proposal_payload_for(
        approval_doc=approval_block,
        session_approval_doc=session_block,
        quote=quote_block,
        scope=scope,
        projection=projection,
        financial_unknowns=financial_unknowns,
        verification=verification,
        provenance=provenance,
        created_at=created_iso,
        expires_at=expires_iso,
    )
    approval = {
        "schema": APPROVAL_SCHEMA,
        "approved_by": approved_by.strip(),
        "approved_at": _iso(current),
        "explicit": True,
        "confirmation": APPROVAL_CONFIRMATION,
        "proposal_sha256": _sha_mapping(proposal_payload),
        "quote_sha256": quote_sha,
        "quote_observed_at": _iso(observed),
        "risk_doc_sha256": approval_block["sha256"],
        "session_approval_doc_sha256": session_block["sha256"],
        "scope_sha256": _sha_mapping(scope),
        "permission_id": permission_id_for(
            approval_doc=approval_block, session_approval_doc=session_block
        ),
    }
    payload = {
        "schema": SCHEMA,
        "admission_id": admission_id_for(
            approval_doc=approval_block,
            session_approval_doc=session_block,
            scope=scope,
        ),
        "permission_id": approval["permission_id"],
        "created_at": created_iso,
        "expires_at": expires_iso,

        "approval_doc": approval_block,
        "session_approval_doc": session_block,
        "quote": quote_block,
        "scope": scope,
        "projection": projection,
        "financial_unknowns": financial_unknowns,
        "verification": verification,
        "provenance": provenance,
        "approval": approval,
    }
    value = OwnerRiskAdmission.from_dict(payload)
    issues = validate_admission(value, now=current)
    if issues:
        raise OwnerRiskError("scoped owner-risk admission refused: " + "; ".join(issues))
    target = Path(output_path)
    text = _json_text(value.to_dict())
    if target.exists():
        existing = load_admission(
            target,
            now=current,
            allow_expired=True,
            verify_docs=True,
            verify_quote=True,
        )
        if (
            existing.admission_id == value.admission_id
            and existing.quote.get("sha256") == value.quote.get("sha256")
            and existing.scope == value.scope
            and existing.approval == value.approval
        ):
            return existing
        raise OwnerRiskError(
            "scoped admission already exists and is immutable; use a new output path"
        )
    _atomic_create(target, text)
    return load_admission(target, now=current)


def load_admission(
    path: Path | str,
    *,
    now: datetime | float | None = None,
    allow_expired: bool = False,
    verify_docs: bool = True,
    verify_quote: bool = True,
    require_fresh: bool = True,
) -> OwnerRiskAdmission:
    admission_path = _assert_safe_path(Path(path), must_exist=True)
    value = OwnerRiskAdmission.from_dict(_read_json(admission_path))
    issues = validate_admission(value, now=now, allow_expired=allow_expired)
    if issues:
        raise OwnerRiskError("owner-risk admission invalid: " + "; ".join(issues))
    if require_fresh:
        fresh = validate_fresh_quote(value, now=now)
        if fresh:
            raise OwnerRiskError("owner-risk admission invalid: " + "; ".join(fresh))
    if verify_docs:
        for block, expected_rel, expected_path in (
            (value.approval_doc, RISK_DOC_REL, RISK_DOC),
            (value.session_approval_doc, SESSION_APPROVAL_DOC_REL, SESSION_APPROVAL_DOC),
        ):
            if block.get("path") != expected_rel:
                raise OwnerRiskError("owner-risk approval path is not the fixed document")
            path_value = _assert_safe_path(Path(expected_path), must_exist=True)
            if _sha_file(path_value) != str(block.get("sha256")).lower():
                raise OwnerRiskError("owner-risk approval document hash mismatch")
    if verify_quote:
        quote_path = _assert_safe_path(Path(str(value.quote.get("path"))), must_exist=True)
        if _sha_file(quote_path) != str(value.quote.get("sha256")).lower():
            raise OwnerRiskError("owner-risk quote file hash mismatch")
    return value


def _claims_dir(registry_root: Path | str | None = None) -> Path:
    root = Path(registry_root) if registry_root is not None else DEFAULT_REGISTRY_ROOT
    _assert_safe_path(root, must_exist=False)
    claims = root / "claims"
    if claims.exists() and not claims.is_dir():
        raise OwnerRiskError("owner-risk registry claims path is not a directory")
    claims.mkdir(parents=True, exist_ok=True)
    return claims


@contextmanager
def _permission_lock(registry_root: Path | str | None = None):
    """Central registry lock shared by new and legacy claim scans/writes."""

    claims = _claims_dir(registry_root)
    lock_path = claims / ".permission.lock"
    from anime_studio.errors import InstanceAlreadyRunning
    from anime_studio.lock import InstanceLock

    try:
        lock = InstanceLock(lock_path).acquire()
    except InstanceAlreadyRunning as exc:
        raise OwnerRiskClaimError(
            "owner-risk permission registry is busy; retry or recover manually"
        ) from exc
    try:
        yield claims
    finally:
        lock.release()


def _legacy_admission_digest(value: OwnerRiskAdmission) -> str:
    """Digest exactly as the accepted pre-scoped owner_risk code computed it."""

    payload = value.to_dict()
    payload.pop("permission_id", None)
    payload.pop("approval", None)
    return _sha_mapping(payload)


def _legacy_claim_records(
    claims: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    """Read old admission-id-named claims; fail closed on malformed legacy files."""

    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(claims.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix not in (".json", ".reserve"):
            continue
        stem = path.stem
        if len(stem) != _HEX64 or any(
            char not in "0123456789abcdefABCDEF" for char in stem
        ):
            continue
        payload = _read_json(path)
        if "permission_id" in payload:
            continue
        if payload.get("schema") not in (CLAIM_SCHEMA, RESERVE_SCHEMA):
            raise OwnerRiskClaimError(
                f"legacy claim {path.name} has an unknown schema; manual recovery required"
            )
        _validate_claim_payload(
            payload,
            schema=payload["schema"],
            admission_id=payload.get("admission_id"),
        )
        if payload.get("status") not in ("BOUND", "RESERVED"):
            raise OwnerRiskClaimError(
                f"legacy claim {path.name} has an unknown status; manual recovery required"
            )
        records.append((path, payload))
    return records


def _legacy_record_matches(
    payload: Mapping[str, Any],
    value: OwnerRiskAdmission,
    *,
    order_id: int,
    created_at: int,
    state_path: str,
    binding_sha256: str | None,
    allow_reserved: bool,
) -> bool:
    if payload.get("admission_id") != value.admission_id:
        return False
    if payload.get("admission_sha256") != _legacy_admission_digest(value):
        return False
    if payload.get("quote_sha256") != str(value.quote.get("sha256")).lower():
        return False
    if payload.get("server_id") != int(value.scope["server_id"]):
        return False
    if payload.get("order_id") != order_id:
        return False
    if payload.get("created_at") != created_at:
        return False
    if payload.get("state_path") != state_path:
        return False
    if payload.get("status") == "BOUND":
        if binding_sha256 is not None and payload.get("session_binding_sha256") != binding_sha256:
            return False
        return True
    return payload.get("status") == "RESERVED" and allow_reserved


def _legacy_pair_consistent(
    bound: Mapping[str, Any], reserve: Mapping[str, Any]
) -> None:
    immutable = (
        "admission_id",
        "admission_sha256",
        "quote_sha256",
        "server_id",
        "order_id",
        "created_at",
        "state_path",
    )
    for key in immutable:
        if bound.get(key) != reserve.get(key):
            raise OwnerRiskClaimError(
                "legacy owner-risk BOUND/RESERVED pair has conflicting immutable fields; "
                "manual recovery required"
            )
    for key in ("state_sha256", "session_binding_sha256"):
        if (
            key in reserve
            and key in bound
            and reserve.get(key) != bound.get(key)
        ):
            raise OwnerRiskClaimError(
                "legacy owner-risk BOUND/RESERVED pair has conflicting state proof; "
                "manual recovery required"
            )


def _legacy_claim_transactions(
    claims: Path,
) -> list[dict[str, Any]]:
    """Coalesce a consistent old BOUND/.json + RESERVED/.reserve pair.

    The accepted old complete_claim wrote BOUND <admission_id>.json and left
    RESERVED <same_id>.reserve. That is one transaction with BOUND authority,
    not two competing claims. Conflicting/foreign/multiple groups fail closed.
    """

    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, payload in _legacy_claim_records(claims):
        grouped.setdefault(str(payload["admission_id"]), []).append((path, payload))
    transactions: list[dict[str, Any]] = []
    for admission_id, items in grouped.items():
        if len(items) == 1:
            path, payload = items[0]
            transactions.append(
                {
                    "admission_id": admission_id,
                    "bound_path": path if payload["status"] == "BOUND" else None,
                    "reserve_path": path if payload["status"] == "RESERVED" else None,
                    "authority": payload if payload["status"] == "BOUND" else None,
                    "reserve": payload if payload["status"] == "RESERVED" else None,
                }
            )
            continue
        if len(items) != 2:
            raise OwnerRiskClaimError(
                "legacy owner-risk has multiple records for one admission; "
                "manual recovery required"
            )
        bound_items = [item for item in items if item[1].get("status") == "BOUND"]
        reserve_items = [item for item in items if item[1].get("status") == "RESERVED"]
        if len(bound_items) != 1 or len(reserve_items) != 1:
            raise OwnerRiskClaimError(
                "legacy owner-risk pair is not one BOUND + one RESERVED; "
                "manual recovery required"
            )
        bound_path, bound_payload = bound_items[0]
        reserve_path, reserve_payload = reserve_items[0]
        _legacy_pair_consistent(bound_payload, reserve_payload)
        transactions.append(
            {
                "admission_id": admission_id,
                "bound_path": bound_path,
                "reserve_path": reserve_path,
                "authority": bound_payload,
                "reserve": reserve_payload,
            }
        )
    return transactions


def _legacy_transaction_matches(
    transaction: Mapping[str, Any],
    value: OwnerRiskAdmission,
    *,
    order_id: int,
    created_at: int,
    state_path: str,
    binding_sha256: str | None,
    allow_reserved: bool,
) -> bool:
    authority = transaction.get("authority")
    reserve = transaction.get("reserve")
    payload = authority if authority is not None else reserve
    if not isinstance(payload, Mapping):
        return False
    if not _legacy_record_matches(
        payload,
        value,
        order_id=order_id,
        created_at=created_at,
        state_path=state_path,
        binding_sha256=binding_sha256,
        allow_reserved=allow_reserved,
    ):
        return False
    if authority is not None and isinstance(reserve, Mapping):
        for key in ("state_sha256", "session_binding_sha256"):
            if (
                key in reserve
                and key in authority
                and reserve.get(key) != authority.get(key)
            ):
                return False
    return True



def claim_paths(
    permission_id: str, *, registry_root: Path | str | None = None
) -> tuple[Path, Path]:
    _required_hex(permission_id, "permission_id")
    claims = _claims_dir(registry_root)
    return claims / f"{permission_id}.json", claims / f"{permission_id}.reserve"


def claim_exists(
    permission_id: str, *, registry_root: Path | str | None = None
) -> bool:
    try:
        _required_hex(permission_id, "permission_id")
    except OwnerRiskError:
        return False
    root = Path(registry_root) if registry_root is not None else DEFAULT_REGISTRY_ROOT
    claims = root / "claims"
    if not claims.is_dir():
        return False
    return (claims / f"{permission_id}.json").is_file() or (
        claims / f"{permission_id}.reserve"
    ).is_file()


def _validate_claim_payload(
    payload: Mapping[str, Any], *, schema: str, admission_id: str
) -> None:
    if payload.get("schema") != schema:
        raise OwnerRiskError("owner-risk claim schema is invalid")
    if payload.get("admission_id") != admission_id:
        raise OwnerRiskError("owner-risk claim admission id mismatch")
    permission_id = payload.get("permission_id")
    if permission_id is not None:
        _required_hex(permission_id, "permission_id")
    for name in ("admission_sha256", "quote_sha256"):
        _required_hex(payload.get(name), name)
    _required_int(payload.get("server_id"), "server_id", minimum=1)
    _required_int(payload.get("order_id"), "order_id", minimum=1)
    _required_int(payload.get("created_at"), "created_at", minimum=1)
    _required_str(payload.get("state_path"), "state_path")


@dataclass(frozen=True)
class ClaimReservation:
    admission_id: str
    permission_id: str
    admission_sha256: str
    quote_sha256: str
    server_id: int
    order_id: int
    created_at: int
    state_path: str
    reserve_path: Path
    bound_path: Path
    payload: Mapping[str, Any]

    @property
    def bound(self) -> bool:
        return self.payload.get("status") == "BOUND"


def reserve_admission(
    admission: OwnerRiskAdmission | Mapping[str, Any],
    *,
    order_id: int,
    created_at: int,
    state_path: Path | str,
    registry_root: Path | str | None = None,
    now: datetime | float | None = None,
) -> ClaimReservation:
    value = _coerce_admission(admission)
    _required_int(order_id, "order_id", minimum=1)
    _required_int(created_at, "created_at", minimum=1)
    state_text = str(Path(state_path).resolve())
    permission_id = permission_id_for_admission(value)
    artifact_sha = admission_digest(value)
    with _permission_lock(registry_root) as claims:
        bound_path = claims / f"{permission_id}.json"
        reserve_path = claims / f"{permission_id}.reserve"
        legacy_transactions = _legacy_claim_transactions(claims)
        if bound_path.is_file():
            payload = _read_json(bound_path)
            _validate_claim_payload(
                payload, schema=CLAIM_SCHEMA, admission_id=value.admission_id
            )
            if legacy_transactions:
                raise OwnerRiskClaimError(
                    "legacy owner-risk claim also exists; manual recovery required"
                )
            if (
                payload.get("admission_sha256") != artifact_sha
                or payload.get("quote_sha256") != str(value.quote.get("sha256")).lower()
                or payload.get("server_id") != int(value.scope["server_id"])
                or payload.get("order_id") != order_id
                or payload.get("created_at") != created_at
                or payload.get("state_path") != state_text
            ):
                raise OwnerRiskClaimError(
                    "owner-risk consent is already bound to another order/session"
                )
            return ClaimReservation(
                admission_id=value.admission_id,
                permission_id=permission_id,
                admission_sha256=artifact_sha,
                quote_sha256=str(value.quote.get("sha256")),
                server_id=int(value.scope["server_id"]),
                order_id=order_id,
                created_at=created_at,
                state_path=state_text,
                reserve_path=reserve_path,
                bound_path=bound_path,
                payload=payload,
            )
        if reserve_path.is_file():
            payload = _read_json(reserve_path)
            _validate_claim_payload(
                payload, schema=RESERVE_SCHEMA, admission_id=value.admission_id
            )
            if legacy_transactions:
                raise OwnerRiskClaimError(
                    "legacy owner-risk claim also exists; manual recovery required"
                )
            if (
                payload.get("admission_sha256") != artifact_sha
                or payload.get("quote_sha256") != str(value.quote.get("sha256")).lower()
                or payload.get("server_id") != int(value.scope["server_id"])
                or payload.get("order_id") != order_id
                or payload.get("created_at") != created_at
                or payload.get("state_path") != state_text
            ):
                raise OwnerRiskClaimError(
                    "owner-risk consent is reserved for another order/session"
                )
            return ClaimReservation(
                admission_id=value.admission_id,
                permission_id=permission_id,
                admission_sha256=artifact_sha,
                quote_sha256=str(value.quote.get("sha256")),
                server_id=int(value.scope["server_id"]),
                order_id=order_id,
                created_at=created_at,
                state_path=state_text,
                reserve_path=reserve_path,
                bound_path=bound_path,
                payload=payload,
            )
        if legacy_transactions:
            if _is_dynamic(value):
                raise OwnerRiskClaimError(
                    "legacy owner-risk claim exists; a new scoped proposal cannot "
                    "consume it; manual recovery required"
                )
            matches = [
                transaction
                for transaction in legacy_transactions
                if _legacy_transaction_matches(
                    transaction,
                    value,
                    order_id=order_id,
                    created_at=created_at,
                    state_path=state_text,
                    binding_sha256=None,
                    allow_reserved=False,
                )
            ]
            if (
                len(legacy_transactions) == 1
                and len(matches) == 1
                and matches[0].get("authority") is not None
            ):
                if not Path(state_path).is_file():
                    raise OwnerRiskClaimError(
                        "legacy BOUND claim exists but its state is missing; "
                        "manual recovery required"
                    )
                transaction = matches[0]
                return ClaimReservation(
                    admission_id=value.admission_id,
                    permission_id=permission_id,
                    admission_sha256=artifact_sha,
                    quote_sha256=str(value.quote.get("sha256")),
                    server_id=int(value.scope["server_id"]),
                    order_id=order_id,
                    created_at=created_at,
                    state_path=state_text,
                    reserve_path=transaction["reserve_path"],
                    bound_path=transaction["bound_path"],
                    payload=transaction["authority"],
                )
            raise OwnerRiskClaimError(
                "legacy owner-risk RESERVED/unknown claim requires manual recovery; "
                "refusing new permission consumption"
            )
        reserve_payload = {
            "schema": RESERVE_SCHEMA,
            "status": "RESERVED",
            "permission_id": permission_id,
            "admission_id": value.admission_id,
            "admission_sha256": artifact_sha,
            "quote_sha256": str(value.quote.get("sha256")),
            "server_id": value.scope.get("server_id"),
            "order_id": order_id,
            "created_at": created_at,
            "state_path": state_text,
            "reserved_at": _iso(_now(now)),
        }
        _atomic_create(reserve_path, _json_text(reserve_payload))
        return ClaimReservation(
            admission_id=value.admission_id,
            permission_id=permission_id,
            admission_sha256=artifact_sha,
            quote_sha256=str(value.quote.get("sha256")),
            server_id=int(value.scope["server_id"]),
            order_id=order_id,
            created_at=created_at,
            state_path=state_text,
            reserve_path=reserve_path,
            bound_path=bound_path,
            payload=reserve_payload,
        )



def assert_bound_claim(
    admission: OwnerRiskAdmission | Mapping[str, Any],
    *,
    order_id: int,
    server_id: int,
    created_at: int,
    state_path: Path | str,
    binding_sha256: str | None = None,
    allow_reserved: bool = False,
    registry_root: Path | str | None = None,
) -> Mapping[str, Any]:
    """Validate the exact single-use claim for an existing bound state."""

    value = _coerce_admission(admission)
    _required_int(order_id, "order_id", minimum=1)
    _required_int(server_id, "server_id", minimum=1)
    _required_int(created_at, "created_at", minimum=1)
    state_text = str(Path(state_path).resolve())
    permission_id = permission_id_for_admission(value)
    with _permission_lock(registry_root) as claims:
        bound_path = claims / f"{permission_id}.json"
        reserve_path = claims / f"{permission_id}.reserve"
        legacy_transactions = _legacy_claim_transactions(claims)
        if bound_path.is_file():
            payload = _read_json(bound_path)
            _validate_claim_payload(
                payload, schema=CLAIM_SCHEMA, admission_id=value.admission_id
            )
            if legacy_transactions:
                raise OwnerRiskClaimError(
                    "legacy owner-risk claim also exists; manual recovery required"
                )
            if payload.get("permission_id") not in (None, permission_id):
                raise OwnerRiskClaimError("owner-risk claim permission id mismatch")
            if payload.get("status") != "BOUND":
                raise OwnerRiskClaimError("owner-risk claim is not BOUND")
            if binding_sha256 is not None and payload.get("session_binding_sha256") != binding_sha256:
                raise OwnerRiskClaimError("owner-risk claim binding hash mismatch")
        elif allow_reserved and reserve_path.is_file():
            payload = _read_json(reserve_path)
            _validate_claim_payload(
                payload, schema=RESERVE_SCHEMA, admission_id=value.admission_id
            )
            if legacy_transactions:
                raise OwnerRiskClaimError(
                    "legacy owner-risk claim also exists; manual recovery required"
                )
            if payload.get("permission_id") not in (None, permission_id):
                raise OwnerRiskClaimError("owner-risk claim permission id mismatch")
        elif legacy_transactions:
            matches = [
                transaction
                for transaction in legacy_transactions
                if _legacy_transaction_matches(
                    transaction,
                    value,
                    order_id=order_id,
                    created_at=created_at,
                    state_path=state_text,
                    binding_sha256=binding_sha256,
                    allow_reserved=allow_reserved,
                )
            ]
            if len(legacy_transactions) == 1 and len(matches) == 1:
                transaction = matches[0]
                return (
                    transaction["authority"]
                    if transaction.get("authority") is not None
                    else transaction["reserve"]
                )
            raise OwnerRiskClaimError(
                "legacy owner-risk claim requires manual recovery; refusing to assume it is free"
            )
        else:
            raise OwnerRiskClaimError("owner-risk BOUND claim is missing")

        if (
            payload.get("admission_sha256") != admission_digest(value)
            or payload.get("quote_sha256") != str(value.quote.get("sha256")).lower()
            or payload.get("server_id") != server_id
            or payload.get("order_id") != order_id
            or payload.get("created_at") != created_at
            or payload.get("state_path") != state_text
        ):
            raise OwnerRiskClaimError("owner-risk claim identity/state path mismatch")
        return payload



def complete_claim(
    reservation: ClaimReservation,
    *,
    state_path: Path | str,
    state_sha256: str,
    session_binding_sha256: str,
    now: datetime | float | None = None,
) -> dict[str, Any]:
    state_text = str(Path(state_path).resolve())
    if state_text != reservation.state_path:
        raise OwnerRiskClaimError("owner-risk claim state path changed")
    _required_hex(state_sha256, "state_sha256")
    _required_hex(session_binding_sha256, "session_binding_sha256")
    with _permission_lock(reservation.bound_path.parent.parent) as claims:
        if _legacy_claim_transactions(claims):
            raise OwnerRiskClaimError(
                "legacy owner-risk claim appeared; manual recovery required"
            )
        payload = {
            "schema": CLAIM_SCHEMA,
            "status": "BOUND",
            "permission_id": reservation.permission_id,
            "admission_id": reservation.admission_id,
            "admission_sha256": reservation.admission_sha256,
            "quote_sha256": reservation.quote_sha256,
            "server_id": reservation.server_id,
            "order_id": reservation.order_id,
            "created_at": reservation.created_at,
            "state_path": state_text,
            "state_sha256": state_sha256,
            "session_binding_sha256": session_binding_sha256,
            "bound_at": _iso(_now(now)),
        }
        text = _json_text(payload)
        if reservation.bound_path.exists():
            existing = _read_json(reservation.bound_path)
            _validate_claim_payload(existing, schema=CLAIM_SCHEMA, admission_id=reservation.admission_id)
            if existing.get("permission_id") not in (None, reservation.permission_id):
                raise OwnerRiskClaimError("owner-risk claim permission id mismatch")
            if (
                existing.get("admission_sha256") != reservation.admission_sha256
                or existing.get("state_path") != state_text
                or existing.get("state_sha256") != state_sha256
                or existing.get("session_binding_sha256") != session_binding_sha256
            ):
                raise OwnerRiskClaimError("owner-risk claim is already bound with different state")
            return existing
        try:
            _atomic_create(reservation.bound_path, text)
        except FileExistsError:
            existing = _read_json(reservation.bound_path)
            _validate_claim_payload(existing, schema=CLAIM_SCHEMA, admission_id=reservation.admission_id)
            if existing.get("permission_id") not in (None, reservation.permission_id):
                raise OwnerRiskClaimError("owner-risk claim permission id mismatch")
            if (
                existing.get("admission_sha256") != reservation.admission_sha256
                or existing.get("state_path") != state_text
                or existing.get("state_sha256") != state_sha256
                or existing.get("session_binding_sha256") != session_binding_sha256
            ):
                raise OwnerRiskClaimError("owner-risk claim is already bound with different state")
            return existing
        return payload




__all__ = [
    "ALLOWED_CREATION_FEE",
    "ALLOWED_CURRENCY",
    "ALLOWED_HOSTER_ID",
    "ALLOWED_PRICE_PER_DAY",
    "ALLOWED_PROVIDER_FEE_PERCENT",
    "ALLOWED_RESERVE",
    "ALLOWED_SERVER_ID",
    "APPROVAL_CONFIRMATION",
    "APPROVAL_SCHEMA",
    "ASSUMED_MIN_BILLING_SECONDS",
    "ASSUMED_OTHER_FEES",

    "AUTOMATIC_RETRIES",
    "CODE_CAP_USD",
    "CLAIM_SCHEMA",
    "ClaimReservation",
    "CONSENT_PURPOSE",
    "DEFAULT_REGISTRY_ROOT",
    "DURATION_SECONDS",
    "GENERATIONS",
    "MARKET_QUOTE_SCHEMA",
    "ORDERS",
    "OWNER_CEILING_USD",

    "OwnerRiskAdmission",
    "OwnerRiskClaimError",
    "OwnerRiskError",
    "PENDING_SCHEMA",
    "PROVIDER_REPORTED_MRL_SECONDS",
    "RESERVE_SCHEMA",
    "RISK_DOC",
    "SCHEMA",
    "SESSION_APPROVAL_DOC",
    "TOTAL_60M",
    "admission_digest",
    "admission_id_for",
    "assert_bound_claim",
    "budget_inputs",
    "claim_exists",
    "claim_paths",
    "complete_claim",
    "create_admission",
    "create_pending",
    "create_scoped_admission",
    "effective_projection",
    "expected_price_per_day",
    "load_admission",
    "permission_id_for",
    "permission_id_for_admission",
    "policy_blockers",
    "projection_for_quote",
    "proposal_digest_for_admission",
    "proposal_payload_for",
    "reserve_admission",

    "strict_blockers",
    "validate_admission",
    "validate_fresh_quote",
]
