"""Isolated, fail-closed Clore.ai order bridge.

This module is deliberately limited to the two public renter operations needed
by the Clore adaptation:

* ``GET my_orders?return_completed=true`` to read one exact order;
* ``POST cancel_order`` with a body containing only the exact order id.

It never creates, extends, tops up, starts, bulk-cancels or deletes orders. It
does not install or use the Clore SDK/CLI. The default constructor refuses to
read the environment or open a network client unless a caller explicitly opts
into live mode. Tests inject an in-process ``httpx`` transport.

The public contract is pinned from the 2026-10-21 documentation snapshot, not
from an account call. Documentation is dynamic; fixtures are
``SOURCE_RECONSTRUCTION_NOT_LIVE`` and must never be described as captures.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Awaitable, Callable, Mapping

import httpx

CLORE_BASE_URL = "https://api.clore.ai/v1/"
DEFAULT_API_KEY_ENV = "CLORE_API_KEY"
EXPECTED_SERVER_ID = 112213
EXPECTED_CURRENCY = "USD-Blockchain"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RESPONSE_BYTES = 1_000_000
DEFAULT_MAX_HEADER_BYTES = 64 * 1024
MIN_REQUEST_INTERVAL_SECONDS = 1.1

_CODE_MESSAGES = {
    0: "normal",
    1: "provider database error",
    2: "invalid input data",
    3: "invalid API token",
    4: "invalid endpoint",
    5: "rate limit exceeded",
    6: "provider rejected request",
}
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class CloreError(Exception):
    """Base class; messages never include raw provider bodies or credentials."""


class CloreConfigError(CloreError):
    """Invalid local configuration."""


class CloreLiveOptInRequired(CloreConfigError):
    """The caller did not explicitly opt into live environment/network use."""


class CloreBudgetError(CloreError):
    """A finite time/size/rate budget cannot accommodate the operation."""


class CloreTransportError(CloreError):
    """Network/TLS/timeout failure, sanitized to an exception type."""


class CloreProtocolError(CloreError):
    """The provider response does not match the pinned public contract."""


class CloreOversizedResponse(CloreProtocolError):
    """The provider response exceeded a fixed size bound."""


class CloreAuthError(CloreError):
    """Authentication failed; no retry is safe."""


class CloreRateLimitError(CloreError):
    """The provider rate limiter rejected the request."""


class CloreProviderError(CloreError):
    """A documented non-auth provider error code."""


class CloreNotFound(CloreError):
    """The exact order id is absent from the bounded response."""


class CloreOrderMismatch(CloreError):
    """The selected order identity does not match the pinned server/profile."""


class CloreCurrencyMismatch(CloreProtocolError):
    """The order currency is not the approved exact currency identifier.

    The normalized order is attached when available so the session can perform
    an exact-identity, budget-safe cancellation instead of withholding a cancel
    and leaving an active order billing.
    """

    def __init__(self, message: str, *, order: "CloreOrder | None" = None) -> None:
        super().__init__(message)
        self.order = order


class CloreCleanupError(CloreError):
    """A response/client cleanup did not complete within its bound."""


class CloreCancelAmbiguous(CloreError):
    """Cancellation was sent but no definitive acceptance/terminal state exists."""


@dataclass(frozen=True)
class CloreOrder:
    """Only the normalized fields needed by the Clore session.

    Raw rows, other orders, headers, and provider error text are intentionally
    not retained.
    """

    order_id: int
    server_id: int
    created_at: int
    price_per_day: Decimal
    currency: str
    expired: bool
    fee_percent: Decimal | None = None
    creation_fee: Decimal | None = None
    spend: Decimal | None = None
    spot: bool | None = None
    mrl_seconds: int | None = None


@dataclass(frozen=True)
class CloreCancelAck:
    """A provider acknowledgement, not proof of termination."""

    code: int
    accepted: bool
    ambiguous: bool
    reason: str


class MonotonicRateLimiter:
    """Serialize all requests with a >=1.1s monotonic interval.

    The wait is performed through the injected sleep function and is bounded by
    the caller's remaining total request budget. If the wait itself cannot fit,
    no request is sent.
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if (
            isinstance(min_interval_seconds, bool)
            or not isinstance(min_interval_seconds, (int, float))
            or not math.isfinite(float(min_interval_seconds))
            or float(min_interval_seconds) < MIN_REQUEST_INTERVAL_SECONDS
        ):
            raise CloreConfigError(
                "min_interval_seconds must be a finite number >= 1.1"
            )
        self.min_interval_seconds = float(min_interval_seconds)
        self._clock = clock
        self._sleep = sleep or asyncio.sleep
        self._last_request_at: float | None = None
        self._slot_lock = asyncio.Lock()

    async def wait(self, *, remaining_budget_seconds: float) -> None:
        """Reserve one send slot under an async lock, then sleep to it.

        The lock is never held across a blocking thread primitive. Lock
        acquisition itself is bounded by the caller's remaining budget.
        Reservation (not completion) assigns the slot, so three concurrent
        callers receive three distinct >=1.1s send times.
        """

        _finite_positive(remaining_budget_seconds, "remaining_budget_seconds")
        try:
            await asyncio.wait_for(
                self._slot_lock.acquire(),
                timeout=remaining_budget_seconds,
            )
        except asyncio.TimeoutError:
            raise CloreBudgetError(
                "rate-limit slot lock did not fit in the remaining request budget"
            ) from None
        try:
            now = self._clock()
            if self._last_request_at is None:
                send_at = now
            else:
                send_at = max(
                    now,
                    self._last_request_at + self.min_interval_seconds,
                )
            wait_for = send_at - now
            if wait_for >= remaining_budget_seconds:
                raise CloreBudgetError(
                    "rate-limit wait does not fit in the remaining request budget"
                )
            self._last_request_at = send_at
        finally:
            self._slot_lock.release()
        if wait_for > 0:
            await self._sleep(wait_for)

    def mark_send_boundary(self) -> None:
        """Record the actual monotonic instant immediately before send."""

        self._last_request_at = self._clock()


def _finite_positive(value: Any, name: str, *, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (value > 0)
    ):
        raise CloreConfigError(f"{name} must be a finite positive number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise CloreConfigError(f"{name} must be a finite positive number")
    if maximum is not None and number > maximum:
        raise CloreConfigError(f"{name} must not exceed {maximum:g}")
    return number


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CloreConfigError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise CloreConfigError(f"{name} must not exceed {maximum}")
    return value


def _contract_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CloreProtocolError(f"provider field {name} must be an integer")
    return value


def _contract_decimal(
    value: Any,
    name: str,
    *,
    minimum: Decimal = Decimal("0"),
    maximum: Decimal | None = None,
) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise CloreProtocolError(f"provider field {name} must be a finite decimal")
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise CloreProtocolError(
                f"provider field {name} must be a finite decimal"
            )
        number = Decimal(str(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise CloreProtocolError(
                f"provider field {name} must be a finite decimal"
            )
        try:
            number = Decimal(text)
        except InvalidOperation as exc:
            raise CloreProtocolError(
                f"provider field {name} must be a finite decimal"
            ) from exc
    else:
        raise CloreProtocolError(f"provider field {name} must be a finite decimal")
    if not number.is_finite():
        raise CloreProtocolError(f"provider field {name} must be a finite decimal")
    if number < minimum:
        raise CloreProtocolError(f"provider field {name} is below its minimum")
    if maximum is not None and number > maximum:
        raise CloreProtocolError(f"provider field {name} is above its maximum")
    return number


def _contract_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CloreProtocolError(f"provider field {name} must be a boolean")
    return value


def _parse_json_object(body: bytes) -> Mapping[str, Any]:
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise CloreProtocolError("provider response was not valid UTF-8") from None
    try:
        payload = json.loads(text, parse_float=Decimal)
    except (TypeError, ValueError):
        raise CloreProtocolError("provider response was not valid JSON") from None
    if not isinstance(payload, Mapping):
        raise CloreProtocolError("provider response must be a JSON object")
    return payload


def _code_error(code: Any) -> Exception | None:
    if isinstance(code, bool) or not isinstance(code, int):
        return CloreProtocolError("provider response code must be an integer")
    if code == 0:
        return None
    if code == 3:
        return CloreAuthError(_CODE_MESSAGES[3])
    if code == 5:
        return CloreRateLimitError(_CODE_MESSAGES[5])
    if code in (1, 2, 4, 6):
        return CloreProviderError(_CODE_MESSAGES[code])
    return CloreProtocolError("provider returned an unknown error code")


class CloreBridge:
    """Bound, exact-id Clore client.

    Construction with no injected transport requires ``allow_live=True`` and
    then reads only the named environment variable. Construction with an
    injected transport never reads the environment and never creates a network
    connection outside that transport.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        allow_live: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        env_getter: Callable[[str], str | None] | None = None,
        expected_server_id: int = EXPECTED_SERVER_ID,
        expected_currency: str = EXPECTED_CURRENCY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
        min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if not isinstance(api_key_env, str) or not _ENV_RE.fullmatch(api_key_env):
            raise CloreConfigError("api_key_env must be an environment variable name")
        if (
            isinstance(expected_server_id, bool)
            or not isinstance(expected_server_id, int)
            or expected_server_id <= 0
        ):
            raise CloreConfigError("expected_server_id must be a positive integer")
        if not isinstance(expected_currency, str) or not expected_currency.strip():
            raise CloreConfigError("expected_currency must be a non-empty string")
        self.expected_server_id = expected_server_id
        self.expected_currency = expected_currency.strip()
        self.timeout_seconds = _finite_positive(
            timeout_seconds, "timeout_seconds", maximum=300.0
        )
        self.max_response_bytes = _positive_int(
            max_response_bytes, "max_response_bytes", maximum=10_000_000
        )
        self.max_header_bytes = _positive_int(
            max_header_bytes, "max_header_bytes", maximum=1_000_000
        )
        self._clock = clock
        self._send_lock = asyncio.Lock()
        self._env_getter = env_getter or os.environ.get
        self._limiter = MonotonicRateLimiter(
            min_interval_seconds=min_request_interval_seconds,
            clock=clock,
            sleep=sleep,
        )

        if transport is None:
            if not allow_live:
                raise CloreLiveOptInRequired(
                    "explicit live opt-in is required; environment and network "
                    "were not accessed"
                )
            key = self._env_getter(api_key_env)
            if not key:
                raise CloreConfigError(
                    "Clore API key environment variable is not set"
                )
        else:
            # A fake/in-process transport must never cause an environment read.
            key = api_key if api_key is not None else "offline-fixture-key"

        self._client = httpx.AsyncClient(
            transport=transport,
            base_url=CLORE_BASE_URL,
            headers={"auth": key},
            follow_redirects=False,
            timeout=None,
            trust_env=False,
        )

    async def aclose(self, timeout_seconds: float = 1.0) -> None:
        timeout = _finite_positive(
            timeout_seconds, "cleanup timeout_seconds", maximum=5.0
        )
        try:
            await asyncio.wait_for(self._client.aclose(), timeout=timeout)
        except asyncio.TimeoutError:
            raise CloreCleanupError("client close exceeded its bound") from None
        except httpx.HTTPError:
            raise CloreCleanupError("client close failed") from None

    async def __aenter__(self) -> "CloreBridge":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise CloreBudgetError("request budget expired before completion")
        return remaining

    async def _read_body(self, response: httpx.Response, deadline: float) -> bytes:
        total = 0
        chunks: list[bytes] = []
        async for chunk in response.aiter_bytes():
            if not isinstance(chunk, (bytes, bytearray)):
                raise CloreProtocolError("provider body chunk has an invalid type")
            total += len(chunk)
            if total > self.max_response_bytes:
                raise CloreOversizedResponse(
                    "provider response exceeded the configured size bound"
                )
            self._remaining(deadline)
            chunks.append(bytes(chunk))
        self._remaining(deadline)
        return b"".join(chunks)

    async def _close_response(
        self, response: httpx.Response, deadline: float
    ) -> None:
        remaining = max(0.0, deadline - self._clock())
        timeout = min(1.0, remaining)
        try:
            await asyncio.wait_for(response.aclose(), timeout=timeout)
        except asyncio.TimeoutError:
            raise CloreCleanupError("response close exceeded its bound") from None
        except httpx.HTTPError:
            raise CloreCleanupError("response close failed") from None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        deadline_monotonic: float | None = None,
    ) -> tuple[int, bytes]:
        configured_deadline = self._clock() + self.timeout_seconds
        if deadline_monotonic is not None:
            if (
                isinstance(deadline_monotonic, bool)
                or not isinstance(deadline_monotonic, (int, float))
                or not math.isfinite(float(deadline_monotonic))
            ):
                raise CloreBudgetError(
                    "external deadline must be a finite monotonic timestamp"
                )
            deadline = min(configured_deadline, float(deadline_monotonic))
        else:
            deadline = configured_deadline
        if deadline <= self._clock():
            raise CloreBudgetError("request budget expired before send")
        try:
            await asyncio.wait_for(
                self._send_lock.acquire(),
                timeout=self._remaining(deadline),
            )
        except asyncio.TimeoutError:
            raise CloreBudgetError(
                "send admission did not fit in the remaining request budget"
            ) from None
        try:
            await self._limiter.wait(
                remaining_budget_seconds=self._remaining(deadline)
            )
            if self._remaining(deadline) <= 0:
                raise CloreBudgetError(
                    "request budget expired after rate-limit wait"
                )
            request = self._client.build_request(
                method,
                path,
                params=params,
                json=json_body,
            )
            if self._remaining(deadline) <= 0:
                raise CloreBudgetError(
                    "request budget expired while building request"
                )
            self._limiter.mark_send_boundary()
            try:
                send_timeout = self._remaining(deadline)
                response = await asyncio.wait_for(
                    self._client.send(request, stream=True),
                    timeout=send_timeout,
                )
            except asyncio.TimeoutError:
                raise CloreTransportError("provider request timed out") from None
            except httpx.HTTPError as exc:
                raise CloreTransportError(
                    f"provider transport error ({type(exc).__name__})"
                ) from None
        finally:
            self._send_lock.release()

        try:
            try:
                self._remaining(deadline)
                header_bytes = 0
                for name, value in response.headers.multi_items():
                    header_bytes += len(str(name).encode("utf-8")) + len(
                        str(value).encode("utf-8")
                    )
                    if header_bytes > self.max_header_bytes:
                        raise CloreOversizedResponse(
                            "provider response headers exceeded the configured bound"
                        )
                status = response.status_code
                if 300 <= status < 400:
                    raise CloreProtocolError("provider redirects are forbidden")
                try:
                    body_timeout = self._remaining(deadline)
                    body = await asyncio.wait_for(
                        self._read_body(response, deadline),
                        timeout=body_timeout,
                    )
                except asyncio.TimeoutError:
                    raise CloreTransportError(
                        "provider response timed out"
                    ) from None
                except httpx.HTTPError as exc:
                    raise CloreTransportError(
                        f"provider response stream error ({type(exc).__name__})"
                    ) from None
            except BaseException:
                try:
                    await self._close_response(response, deadline)
                except BaseException:
                    pass
                raise
            else:
                await self._close_response(response, deadline)
        except CloreCleanupError:
            raise
        except CloreError:
            raise

        if status in (401, 403):
            raise CloreAuthError("provider authentication failed")
        if status == 429:
            raise CloreRateLimitError(_CODE_MESSAGES[5])
        if 500 <= status <= 599:
            raise CloreTransportError(
                f"provider server error (HTTP {status})"
            )
        if status != 200:
            raise CloreProtocolError(
                f"unexpected provider HTTP status {status}"
            )
        return status, body

    async def read_order(
        self,
        order_id: int,
        *,
        deadline_monotonic: float | None = None,
    ) -> CloreOrder:
        """Read exactly one order id from a bounded my_orders response."""

        if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id <= 0:
            raise CloreConfigError("order_id must be a positive integer")
        _status, body = await self._request(
            "GET",
            "my_orders",
            params={"return_completed": "true"},
            deadline_monotonic=deadline_monotonic,
        )
        payload = _parse_json_object(body)
        error = _code_error(payload.get("code"))
        if error is not None:
            raise error
        orders = payload.get("orders")
        if not isinstance(orders, list):
            raise CloreProtocolError(
                "provider response orders field must be a list"
            )
        seen: set[int] = set()
        selected: Mapping[str, Any] | None = None
        for row in orders:
            if not isinstance(row, Mapping):
                raise CloreProtocolError("provider order row must be an object")
            row_id = _contract_int(row.get("id"), "id", minimum=1)
            if row_id in seen:
                raise CloreProtocolError("provider response contains a duplicate order id")
            seen.add(row_id)
            if row_id == order_id:
                selected = row
        if selected is None:
            raise CloreNotFound("exact order id is absent from the bounded response")

        server_id = _contract_int(selected.get("si"), "si", minimum=1)
        if server_id != self.expected_server_id:
            raise CloreOrderMismatch(
                "order server id does not match the selected Clore server"
            )
        created_at = _contract_int(selected.get("ct"), "ct", minimum=1)
        price = _contract_decimal(selected.get("price"), "price")
        currency = selected.get("currency")
        if not isinstance(currency, str):
            raise CloreProtocolError("provider field currency must be a string")
        expired = _contract_bool(selected.get("expired"), "expired")

        fee = selected.get("fee")
        fee_percent = (
            None
            if fee is None
            else _contract_decimal(
                fee, "fee", maximum=Decimal("100")
            )
        )
        creation_fee = selected.get("creation_fee")
        creation = (
            None
            if creation_fee is None
            else _contract_decimal(creation_fee, "creation_fee")
        )
        spend = selected.get("spend")
        spend_value = (
            None if spend is None else _contract_decimal(spend, "spend")
        )
        spot = selected.get("spot")
        spot_value = None if spot is None else _contract_bool(spot, "spot")
        mrl = selected.get("mrl")
        mrl_value = (
            None if mrl is None else _contract_int(mrl, "mrl", minimum=0)
        )

        order = CloreOrder(
            order_id=order_id,
            server_id=server_id,
            created_at=created_at,
            price_per_day=price,
            currency=currency,
            expired=expired,
            fee_percent=fee_percent,
            creation_fee=creation,
            spend=spend_value,
            spot=spot_value,
            mrl_seconds=mrl_value,
        )
        if currency != self.expected_currency:
            raise CloreCurrencyMismatch(
                "order currency is not the approved exact currency identifier",
                order=order,
            )
        return order

    async def read_marketplace(
        self,
        *,
        deadline_monotonic: float | None = None,
    ) -> bytes:
        """GET the fixed public marketplace endpoint; never a caller URL.

        The raw body is returned only to the local collector, which whitelists
        public facts and never persists the provider response. No mutation,
        query parameters or arbitrary path is possible through this method.
        """

        _status, body = await self._request(
            "GET",
            "marketplace",
            deadline_monotonic=deadline_monotonic,
        )
        return body

    async def cancel_order(
        self,
        order_id: int,
        *,
        deadline_monotonic: float | None = None,
    ) -> CloreCancelAck:
        """Send exactly ``{"id": order_id}`` and classify the response.

        A code 0 is only an acknowledgement. Timeouts, 5xx, DB errors, rate
        limits, malformed/oversized bodies and unknown codes are ambiguous and
        are never retried here.
        """

        if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id <= 0:
            raise CloreConfigError("order_id must be a positive integer")
        try:
            _status, body = await self._request(
                "POST",
                "cancel_order",
                json_body={"id": order_id},
                deadline_monotonic=deadline_monotonic,
            )
        except CloreAuthError:
            raise
        except CloreProviderError:
            raise
        except CloreTransportError as exc:
            raise CloreCancelAmbiguous(
                "cancellation transport outcome is ambiguous; readback only"
            ) from None
        except CloreCleanupError:
            raise CloreCancelAmbiguous(
                "cancellation response cleanup was not confirmed; readback only"
            ) from None
        except (CloreProtocolError, CloreOversizedResponse):
            raise CloreCancelAmbiguous(
                "cancellation response could not be trusted; readback only"
            ) from None
        except CloreRateLimitError:
            raise CloreCancelAmbiguous(
                "cancellation was rate-limited; no retry is attempted"
            ) from None
        except CloreBudgetError:
            # No request was sent if the budget/rate limiter refused first.
            raise

        try:
            payload = _parse_json_object(body)
            code = payload.get("code")
            if isinstance(code, bool) or not isinstance(code, int):
                raise CloreCancelAmbiguous(
                    "cancellation response code is missing or invalid; readback only"
                )
            if code == 0:
                return CloreCancelAck(
                    code=0,
                    accepted=True,
                    ambiguous=False,
                    reason=(
                        "provider acknowledged cancellation request; "
                        "terminal not confirmed"
                    ),
                )
            if code == 3:
                raise CloreAuthError(_CODE_MESSAGES[3])
            if code in (1, 5):
                raise CloreCancelAmbiguous(
                    f"{_CODE_MESSAGES[code]}; cancellation is not confirmed"
                )
            if code in (2, 4, 6):
                raise CloreProviderError(_CODE_MESSAGES[code])
            raise CloreCancelAmbiguous(
                "unknown cancellation response code; cancellation is not confirmed"
            )
        except (CloreAuthError, CloreProviderError, CloreCancelAmbiguous):
            raise
        except CloreError:
            raise CloreCancelAmbiguous(
                "cancellation response could not be trusted; readback only"
            ) from None


__all__ = [
    "CLORE_BASE_URL",
    "DEFAULT_API_KEY_ENV",
    "EXPECTED_CURRENCY",
    "EXPECTED_SERVER_ID",
    "MIN_REQUEST_INTERVAL_SECONDS",
    "CloreAuthError",
    "CloreBridge",
    "CloreBudgetError",
    "CloreCancelAck",
    "CloreCancelAmbiguous",
    "CloreCleanupError",
    "CloreConfigError",
    "CloreCurrencyMismatch",
    "CloreError",
    "CloreLiveOptInRequired",
    "CloreNotFound",
    "CloreOrder",
    "CloreOrderMismatch",
    "CloreOversizedResponse",
    "CloreProtocolError",
    "CloreProviderError",
    "CloreRateLimitError",
    "CloreTransportError",
    "MonotonicRateLimiter",
]
