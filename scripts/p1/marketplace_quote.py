"""Read-only fresh marketplace quote collector.

This module is deliberately small and fail-closed. It performs exactly one
``GET marketplace`` request through the existing ``CloreBridge`` transport,
selects one exact server id and persists only whitelisted public facts. It never
creates an order, cancels anything, mutates an account, reads an SSH key or
writes a provider response body.

The output is a machine quote/proposal preview, not an approval. It is not
``VERIFIED_PROVIDER_BILLING``: fees, tax/FX and minimum billing remain unknown.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

import httpx

# Allow ``python scripts/p1/marketplace_quote.py`` as well as package imports.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.p1 import owner_risk
from scripts.p1.clore_bridge import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    CloreBridge,
    CloreError,
)

QUOTE_SCHEMA = owner_risk.MARKET_QUOTE_SCHEMA
STATUS = "FRESH_MARKETPLACE_QUOTE_READ_ONLY"
_ENTRY_INTERVAL = 1.1


class MarketplaceQuoteError(Exception):
    """A quote could not be obtained or did not match the pinned contract."""


def _now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise MarketplaceQuoteError("collector clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha_mapping(value: Mapping[str, Any]) -> str:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _contract_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MarketplaceQuoteError(f"provider field {name} must be an integer")
    return value


def _contract_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise MarketplaceQuoteError(f"provider field {name} must be a boolean")
    return value


def _contract_decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise MarketplaceQuoteError(f"provider field {name} must be a decimal")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MarketplaceQuoteError(
            f"provider field {name} must be a decimal"
        ) from exc
    if not number.is_finite() or number <= 0:
        raise MarketplaceQuoteError(
            f"provider field {name} must be a positive finite decimal"
        )
    return number


def _parse_body(body: bytes) -> Mapping[str, Any]:
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise MarketplaceQuoteError("provider response was not valid UTF-8") from None
    try:
        payload = json.loads(text, parse_float=Decimal)
    except (TypeError, ValueError):
        raise MarketplaceQuoteError("provider response was not valid JSON") from None
    if not isinstance(payload, Mapping):
        raise MarketplaceQuoteError("provider response must be a JSON object")
    return payload


def _check_code(payload: Mapping[str, Any]) -> None:
    code = payload.get("code")
    if isinstance(code, bool) or not isinstance(code, int):
        raise MarketplaceQuoteError("provider response code must be an integer")
    if code == 0:
        return
    if code == 3:
        raise MarketplaceQuoteError("provider authentication failed")
    if code == 5:
        raise MarketplaceQuoteError("provider rate limit exceeded")
    raise MarketplaceQuoteError("provider returned an error code")


def _reject_stale_metadata(payload: Mapping[str, Any], *, now: datetime) -> None:
    for key in ("observed_at", "timestamp", "updated_at", "collected_at"):
        if key not in payload:
            continue
        value = payload[key]
        parsed: datetime | None = None
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError:
                raise MarketplaceQuoteError(
                    f"provider cache metadata {key} is not ISO-8601"
                ) from None
            if parsed.tzinfo is None:
                raise MarketplaceQuoteError(
                    f"provider cache metadata {key} must include a UTC offset"
                )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                raise MarketplaceQuoteError(
                    f"provider cache metadata {key} is not a valid epoch"
                ) from None
        else:
            raise MarketplaceQuoteError(
                f"provider cache metadata {key} has an invalid type"
            )
        parsed = parsed.astimezone(timezone.utc)
        age = (now - parsed).total_seconds()
        if age < -60 or age > owner_risk.QUOTE_MAX_AGE_SECONDS:
            raise MarketplaceQuoteError(
                f"provider cache metadata {key} is stale or in the future"
            )


def _select_public_facts(
    payload: Mapping[str, Any], *, server_id: int, currency: str
) -> dict[str, Any]:
    my_servers = payload.get("my_servers")
    if my_servers is not None:
        if not isinstance(my_servers, list):
            raise MarketplaceQuoteError("provider my_servers field must be a list")
        for item in my_servers:
            if _contract_int(item, "my_servers[]", minimum=1) == server_id:
                raise MarketplaceQuoteError(
                    "requested server is owned by the account and cannot be rented"
                )
    rows = payload.get("servers")
    if not isinstance(rows, list):
        raise MarketplaceQuoteError("provider servers field must be a list")
    selected: Mapping[str, Any] | None = None
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise MarketplaceQuoteError("provider server row must be an object")
        row_id = _contract_int(row.get("id"), "servers[].id", minimum=1)
        if row_id in seen:
            raise MarketplaceQuoteError("provider response contains a duplicate server id")
        seen.add(row_id)
        if row_id == server_id:
            selected = row
    if selected is None:
        raise MarketplaceQuoteError("requested server id is absent from the marketplace")
    owner = _contract_int(selected.get("owner"), "servers[].owner", minimum=1)
    mrl_hours = _contract_int(selected.get("mrl"), "servers[].mrl", minimum=1)
    rented = _contract_bool(selected.get("rented"), "servers[].rented")
    if rented:
        raise MarketplaceQuoteError("requested marketplace server is rented")
    online = selected.get("online")
    if online is not None:
        online = _contract_bool(online, "servers[].online")
        if not online:
            raise MarketplaceQuoteError("requested marketplace server is offline")
    price = selected.get("price")
    if not isinstance(price, Mapping):
        raise MarketplaceQuoteError("provider servers[].price must be an object")
    on_demand = price.get("on_demand")
    if not isinstance(on_demand, Mapping):
        raise MarketplaceQuoteError("provider price.on_demand must be an object")
    if currency not in on_demand:
        raise MarketplaceQuoteError(
            "requested currency is not accepted for this server"
        )
    day = _contract_decimal(on_demand.get(currency), "price.on_demand[currency]")
    facts = {
        "server_id": server_id,
        "hoster_id": owner,
        "mrl_raw_hours": mrl_hours,
        "mrl_raw_seconds": mrl_hours * 3600,
        "mrl_semantics": "MAXIMUM_RENTAL_LENGTH",
        "currency": currency,
        "price_per_day": format(day.normalize(), "f"),
        "availability": {
            "listed": True,
            "rented": False,
            "online": online,
            "online_source": (
                "marketplace" if online is not None else "not_exposed_by_marketplace"
            ),
        },
    }
    return facts


def _proposal_preview(facts: Mapping[str, Any]) -> dict[str, Any]:
    day = Decimal(str(facts["price_per_day"]))
    rental = day * Decimal(60) / Decimal(1440)
    renter = rental * owner_risk.ALLOWED_PROVIDER_FEE_PERCENT / Decimal(100)
    total = (
        rental
        + renter
        + owner_risk.ALLOWED_CREATION_FEE
        + owner_risk.ALLOWED_RESERVE
        + owner_risk.ASSUMED_OTHER_FEES
    )
    return {
        "schema": owner_risk.PROPOSAL_SCHEMA,
        "authorizes_session": False,
        "server_id": facts["server_id"],
        "hoster_id": facts["hoster_id"],
        "hoster_id_provenance": "fresh_marketplace_quote",
        "currency": facts["currency"],
        "price_per_day": facts["price_per_day"],
        "provider_fee_percent": str(owner_risk.ALLOWED_PROVIDER_FEE_PERCENT),
        "creation_fee": str(owner_risk.ALLOWED_CREATION_FEE),
        "reserve": str(owner_risk.ALLOWED_RESERVE),
        "mandatory_other_fees_assumed": str(owner_risk.ASSUMED_OTHER_FEES),
        "min_billing_seconds_assumed": owner_risk.ASSUMED_MIN_BILLING_SECONDS,
        "provider_reported_mrl_seconds": facts["mrl_raw_seconds"],
        "mrl_semantics": facts["mrl_semantics"],
        "code_cap_usd": str(owner_risk.CODE_CAP_USD),
        "owner_ceiling_usd": str(owner_risk.OWNER_CEILING_USD),
        "duration_seconds": owner_risk.DURATION_SECONDS,
        "orders": owner_risk.ORDERS,
        "generations": owner_risk.GENERATIONS,
        "automatic_retries": owner_risk.AUTOMATIC_RETRIES,
        "projection": {
            "rental_60m": format(rental.normalize(), "f"),
            "renter_fee_60m": format(renter.normalize(), "f"),
            "creation_fee": str(owner_risk.ALLOWED_CREATION_FEE),
            "reserve": str(owner_risk.ALLOWED_RESERVE),
            "other_fees_assumed": str(owner_risk.ASSUMED_OTHER_FEES),
            "total_60m": format(total.normalize(), "f"),
        },
        "fits_code_cap": total <= owner_risk.CODE_CAP_USD,
    }


def build_quote_document(
    *,
    facts: Mapping[str, Any],
    requested_at: datetime,
    observed_at: datetime,
) -> dict[str, Any]:
    day = str(facts["price_per_day"])
    proposal = _proposal_preview(facts)
    quote = {
        "price_per_day": day,
        "price_per_hour": None,
        "price_per_minute": None,
        "currency": facts["currency"],
        "creation_fee": None,
        "mandatory_other_fees": None,
        "reserve": None,
        "fresh_verified": True,
        "fees_known": False,
        "tax_fx_known": False,
        "min_rental_hours": None,
        "provider_fee_percent": None,
    }
    requested_iso = _iso(requested_at)
    observed_iso = _iso(observed_at)
    public_facts = {
        "server_id": facts["server_id"],
        "hoster_id": facts["hoster_id"],
        "mrl_raw_hours": facts["mrl_raw_hours"],
        "mrl_raw_seconds": facts["mrl_raw_seconds"],
        "mrl_semantics": facts["mrl_semantics"],
        "currency": facts["currency"],
        "price_per_day": day,
        "availability": dict(facts["availability"]),
    }
    collector = {
        "collector_id": owner_risk.MARKET_COLLECTOR_ID,
        "collector_version": owner_risk.MARKET_COLLECTOR_VERSION,
        "method": owner_risk.MARKET_METHOD,
        "base_url": owner_risk.MARKET_BASE_URL,
        "endpoint": owner_risk.MARKET_ENDPOINT,
        "path": owner_risk.MARKET_PATH,
    }
    integrity = {
        "facts": public_facts,
        "quote": quote,
        "proposal": proposal,
        "requested_at": requested_iso,
        "observed_at": observed_iso,
        "observation_origin": owner_risk.MARKET_OBSERVATION_ORIGIN,
        "collector": collector,
    }
    provenance = {
        "kind": "clore_marketplace_get",
        **collector,
        "http_status": 200,
        "authenticated": True,
        "price_only": True,
        "fees_observed": False,
        "provider_response_persisted": False,
        "not_provider_billing_audit": True,
        "observation_origin": owner_risk.MARKET_OBSERVATION_ORIGIN,
        "freshness": owner_risk.MARKET_FRESHNESS,
        "online_field": (
            "marketplace"
            if facts["availability"].get("online") is not None
            else "not_exposed_by_marketplace"
        ),
    }
    return {
        "schema": QUOTE_SCHEMA,
        "status": STATUS,
        "authorizes_session": False,
        "provider_billing_verified": False,
        "requested_at": requested_iso,
        "observed_at": observed_iso,
        "observation_origin": owner_risk.MARKET_OBSERVATION_ORIGIN,
        "server_id": facts["server_id"],
        "hoster_id": facts["hoster_id"],
        "currency": facts["currency"],
        "price_per_day": day,
        "mrl_raw_hours": facts["mrl_raw_hours"],
        "mrl_raw_seconds": facts["mrl_raw_seconds"],
        "mrl_semantics": facts["mrl_semantics"],
        "availability": dict(facts["availability"]),
        "facts_sha256": _sha_mapping(integrity),
        "quote": quote,
        "proposal": proposal,
        "provenance": provenance,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MarketplaceQuoteError("existing quote artifact cannot be read") from exc
        if existing == text:
            return
        raise MarketplaceQuoteError(
            "quote artifact already exists and is immutable; use a new output path"
        )
    except OSError as exc:
        raise MarketplaceQuoteError("cannot create quote artifact") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


async def collect_marketplace_quote(
    *,
    server_id: int,
    output_path: Path | str,
    currency: str = owner_risk.ALLOWED_CURRENCY,
    allow_live: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
    api_key: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    env_getter: Callable[[str], str | None] | None = None,
    now: datetime | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    min_request_interval_seconds: float = _ENTRY_INTERVAL,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    if isinstance(server_id, bool) or not isinstance(server_id, int) or server_id <= 0:
        raise MarketplaceQuoteError("server_id must be a positive integer")
    if not isinstance(currency, str) or not currency.strip():
        raise MarketplaceQuoteError("currency must be a non-empty string")
    requested_at = _now(now)
    try:
        bridge = CloreBridge(
            api_key=api_key,
            api_key_env=api_key_env,
            allow_live=allow_live,
            transport=transport,
            env_getter=env_getter,
            expected_server_id=server_id,
            expected_currency=currency,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            min_request_interval_seconds=min_request_interval_seconds,
            clock=clock,
            sleep=sleep,
        )
    except CloreError as exc:
        raise MarketplaceQuoteError(str(exc)) from None
    try:
        try:
            body = await bridge.read_marketplace()
        except CloreError as exc:
            raise MarketplaceQuoteError(str(exc)) from None
        observed_at = _now(now)
    finally:
        try:
            await bridge.aclose()
        except CloreError as exc:
            raise MarketplaceQuoteError(str(exc)) from None
    payload = _parse_body(body)
    _check_code(payload)
    _reject_stale_metadata(payload, now=observed_at)
    facts = _select_public_facts(
        payload, server_id=server_id, currency=currency.strip()
    )
    document = build_quote_document(
        facts=facts, requested_at=requested_at, observed_at=observed_at
    )
    _write_json(Path(output_path), document)
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/p1/marketplace_quote.py",
        description=(
            "Read-only fresh Clore marketplace quote. Requires --live-read for "
            "the owner's real GET; tests inject a fake transport."
        ),
    )
    parser.add_argument("--server-id", type=int, required=True)
    parser.add_argument("--currency", default=owner_risk.ALLOWED_CURRENCY)
    parser.add_argument("--output", required=True)
    parser.add_argument("--live-read", action="store_true")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.live_read:
        print(
            json.dumps(
                {
                    "status": "REFUSED",
                    "error": "fresh marketplace quote requires explicit --live-read",
                },
                ensure_ascii=False,
            ),
            end="",
        )
        return 2
    try:
        document = asyncio.run(
            collect_marketplace_quote(
                server_id=args.server_id,
                currency=args.currency,
                output_path=Path(args.output),
                allow_live=True,
                api_key_env=args.api_key_env,
                timeout_seconds=args.timeout_seconds,
                max_response_bytes=args.max_response_bytes,
            )
        )
    except MarketplaceQuoteError as exc:
        print(
            json.dumps(
                {
                    "status": "REFUSED",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                ensure_ascii=False,
            ),
            end="",
        )
        return 2
    summary = {
        "status": document["status"],
        "path": str(Path(args.output)),
        "server_id": document["server_id"],
        "hoster_id": document["hoster_id"],
        "currency": document["currency"],
        "price_per_day": document["price_per_day"],
        "mrl_raw_seconds": document["mrl_raw_seconds"],
        "facts_sha256": document["facts_sha256"],
        "observed_at": document["observed_at"],
        "authorizes_session": False,
        "provider_billing_verified": False,
        "fits_code_cap": document["proposal"]["fits_code_cap"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), end="")
    return 0 if document["proposal"]["fits_code_cap"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
