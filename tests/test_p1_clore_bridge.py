from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from scripts.p1 import clore_bridge as cb

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "p1" / "clore-v1"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _read_with(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> Any:
    bridge = cb.CloreBridge(
        transport=httpx.MockTransport(handler),
        api_key="fixture-key",
        timeout_seconds=kwargs.pop("timeout_seconds", 2.0),
        **kwargs,
    )
    try:
        return await bridge.read_order(7001)
    finally:
        await bridge.aclose()


async def _cancel_with(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> Any:
    bridge = cb.CloreBridge(
        transport=httpx.MockTransport(handler),
        api_key="fixture-key",
        timeout_seconds=kwargs.pop("timeout_seconds", 2.0),
        **kwargs,
    )
    try:
        return await bridge.cancel_order(7001)
    finally:
        await bridge.aclose()


def test_reads_exact_order_and_normalizes_public_fields() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/v1/my_orders"
        assert request.url.params["return_completed"] == "true"
        assert request.headers["auth"] == "fixture-key"
        assert "authorization" not in request.headers
        return httpx.Response(200, json=_fixture("my_orders.json"))

    order = _run(_read_with(handler))
    assert order.order_id == 7001
    assert order.server_id == 112213
    assert order.created_at == 1761090000
    assert order.price_per_day == cb.Decimal("13.2")
    assert order.fee_percent == cb.Decimal("5")
    assert order.creation_fee == cb.Decimal("0.1")
    assert order.spend == cb.Decimal("0.123456")
    assert order.currency == "USD-Blockchain"
    assert order.expired is False
    assert order.spot is False
    assert len(requests) == 1


def test_expected_server_can_be_parameterized_only_by_caller() -> None:
    payload = _fixture("my_orders.json")
    payload["orders"][1]["si"] = 111947

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    order = _run(_read_with(handler, expected_server_id=111947))
    assert order.server_id == 111947
    with pytest.raises(cb.CloreOrderMismatch):
        _run(_read_with(handler))


def test_missing_exact_order_is_not_found_never_first_or_latest() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _fixture("my_orders.json")
        payload["orders"] = [row for row in payload["orders"] if row["id"] != 7001]
        return httpx.Response(200, json=payload)

    with pytest.raises(cb.CloreNotFound):
        _run(_read_with(handler))


def test_duplicate_rows_are_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _fixture("my_orders.json")
        payload["orders"].append(copy.deepcopy(payload["orders"][1]))
        return httpx.Response(200, json=payload)

    with pytest.raises(cb.CloreProtocolError, match="duplicate"):
        _run(_read_with(handler))


def test_other_row_malformed_rejects_entire_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _fixture("my_orders.json")
        payload["orders"][0]["id"] = True
        return httpx.Response(200, json=payload)

    with pytest.raises(cb.CloreProtocolError, match="integer"):
        _run(_read_with(handler))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", True),
        ("si", 112214),
        ("ct", 1.5),
        ("price", "NaN"),
        ("fee", "Infinity"),
        ("spend", "NaN"),
        ("currency", "USD"),
        ("expired", 0),
        ("spot", "false"),
        ("mrl", True),
    ],
)
def test_wrong_selected_field_types_or_identity_are_rejected(
    field: str, value: Any
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload = _fixture("my_orders.json")
        payload["orders"][1][field] = value
        return httpx.Response(200, json=payload)

    with pytest.raises(cb.CloreError):
        _run(_read_with(handler))


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (3, cb.CloreAuthError),
        (5, cb.CloreRateLimitError),
        (6, cb.CloreProviderError),
        (1, cb.CloreProviderError),
        (99, cb.CloreProtocolError),
        (True, cb.CloreProtocolError),
        ("0", cb.CloreProtocolError),
    ],
)
def test_code_errors_are_detected_even_with_http_200(
    code: Any, expected: type[Exception]
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": code, "orders": []})

    with pytest.raises(expected) as exc_info:
        _run(_read_with(handler))
    assert "raw-provider-error-text" not in str(exc_info.value)


def test_http_status_errors_are_sanitized() -> None:
    def unauthorized(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "secret-token"})

    with pytest.raises(cb.CloreAuthError) as exc_info:
        _run(_read_with(unauthorized))
    assert "secret-token" not in str(exc_info.value)

    def redirect(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.invalid/"})

    with pytest.raises(cb.CloreProtocolError, match="redirect"):
        _run(_read_with(redirect))

    def server_error(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="secret-provider-body")

    with pytest.raises(cb.CloreTransportError) as exc_info:
        _run(_read_with(server_error))
    assert "secret-provider-body" not in str(exc_info.value)


def test_cancel_posts_only_exact_id_and_code_zero_is_only_ack() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/cancel_order"
        assert request.headers["auth"] == "fixture-key"
        assert json.loads(request.content) == {"id": 7001}
        return httpx.Response(200, json=_fixture("cancel_code0.json"))

    ack = _run(_cancel_with(handler))
    assert ack.accepted is True
    assert ack.ambiguous is False
    assert "terminal not confirmed" in ack.reason
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (3, cb.CloreAuthError),
        (2, cb.CloreProviderError),
        (4, cb.CloreProviderError),
        (6, cb.CloreProviderError),
        (1, cb.CloreCancelAmbiguous),
        (5, cb.CloreCancelAmbiguous),
        (99, cb.CloreCancelAmbiguous),
        ("0", cb.CloreCancelAmbiguous),
    ],
)
def test_cancel_nonzero_or_malformed_code_is_never_success(
    code: Any, expected: type[Exception]
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": code, "error": "secret-error"})

    with pytest.raises(expected) as exc_info:
        _run(_cancel_with(handler))
    assert "secret-error" not in str(exc_info.value)


@pytest.mark.parametrize(
    "body",
    [
        b"\xff\xfe",
        b"{not-json",
        b"[]",
        b'{"code":"0"}',
        b'{"no_code":0}',
    ],
)
def test_cancel_post_send_malformed_or_schema_invalid_is_ambiguous(
    body: bytes,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=body)

    with pytest.raises(cb.CloreCancelAmbiguous) as exc_info:
        _run(_cancel_with(handler))
    assert calls == 1
    assert "secret" not in str(exc_info.value)


def test_cancel_timeout_is_ambiguous_and_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("secret-timeout")

    with pytest.raises(cb.CloreCancelAmbiguous) as exc_info:
        _run(_cancel_with(handler))
    assert calls == 1
    assert "secret-timeout" not in str(exc_info.value)


def test_oversized_and_header_limited_responses_are_rejected() -> None:
    def oversized(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"code":0,"orders":[]}' * 20)

    with pytest.raises(cb.CloreOversizedResponse):
        _run(_read_with(oversized, max_response_bytes=32))

    def headers(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_fixture("my_orders.json"),
            headers={"x-long": "x" * 200},
        )

    with pytest.raises(cb.CloreOversizedResponse):
        _run(_read_with(headers, max_header_bytes=32))


def test_shared_rate_limiter_waits_at_least_1_1_seconds_between_get_and_post() -> None:
    class FakeTime:
        now = 0.0

        def clock(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.now += seconds

    fake = FakeTime()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.method == "GET":
            return httpx.Response(200, json=_fixture("my_orders.json"))
        return httpx.Response(200, json=_fixture("cancel_code0.json"))

    async def scenario() -> None:
        bridge = cb.CloreBridge(
            transport=httpx.MockTransport(handler),
            api_key="fixture-key",
            timeout_seconds=5.0,
            clock=fake.clock,
            sleep=fake.sleep,
        )
        try:
            await bridge.read_order(7001)
            await bridge.cancel_order(7001)
        finally:
            await bridge.aclose()

    _run(scenario())
    assert calls == 2
    assert fake.now >= 1.1


def test_three_concurrent_requests_get_distinct_spaced_send_slots() -> None:
    class FakeTime:
        now = 0.0

        def clock(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.now += seconds

    fake = FakeTime()
    send_times: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        send_times.append(fake.now)
        return httpx.Response(200, json=_fixture("my_orders.json"))

    async def scenario() -> None:
        bridge = cb.CloreBridge(
            transport=httpx.MockTransport(handler),
            api_key="fixture-key",
            timeout_seconds=10.0,
            clock=fake.clock,
            sleep=fake.sleep,
        )
        try:
            await asyncio.gather(
                bridge.read_order(7001),
                bridge.read_order(7001),
                bridge.read_order(7001),
            )
        finally:
            await bridge.aclose()

    _run(scenario())
    assert len(send_times) == 3
    assert send_times[1] - send_times[0] >= 1.1
    assert send_times[2] - send_times[1] >= 1.1


def test_external_deadline_shorter_than_rate_wait_sends_no_request() -> None:
    class FakeTime:
        now = 0.0

        def clock(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.now += seconds

    fake = FakeTime()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_fixture("my_orders.json"))

    async def scenario() -> None:
        bridge = cb.CloreBridge(
            transport=httpx.MockTransport(handler),
            api_key="fixture-key",
            timeout_seconds=10.0,
            clock=fake.clock,
            sleep=fake.sleep,
        )
        try:
            await bridge.read_order(7001)
            with pytest.raises(cb.CloreBudgetError):
                await bridge.read_order(7001, deadline_monotonic=0.5)
        finally:
            await bridge.aclose()

    _run(scenario())
    assert calls == 1


def test_external_deadline_crossed_during_response_is_rejected() -> None:
    class FakeTime:
        now = 0.0

        def clock(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.now += seconds

    fake = FakeTime()

    def handler(_request: httpx.Request) -> httpx.Response:
        fake.now += 2.0
        return httpx.Response(200, json=_fixture("my_orders.json"))

    async def scenario() -> None:
        bridge = cb.CloreBridge(
            transport=httpx.MockTransport(handler),
            api_key="fixture-key",
            timeout_seconds=10.0,
            clock=fake.clock,
            sleep=fake.sleep,
        )
        try:
            with pytest.raises(cb.CloreBudgetError):
                await bridge.read_order(7001, deadline_monotonic=1.0)
        finally:
            await bridge.aclose()

    _run(scenario())


def test_rate_limit_wait_that_does_not_fit_budget_sends_no_request() -> None:
    class FakeTime:
        now = 0.0

        def clock(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.now += seconds

    fake = FakeTime()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_fixture("my_orders.json"))

    async def scenario() -> None:
        bridge = cb.CloreBridge(
            transport=httpx.MockTransport(handler),
            api_key="fixture-key",
            timeout_seconds=1.0,
            clock=fake.clock,
            sleep=fake.sleep,
        )
        try:
            await bridge.read_order(7001)
            with pytest.raises(cb.CloreBudgetError):
                await bridge.read_order(7001)
        finally:
            await bridge.aclose()

    _run(scenario())
    assert calls == 1


def test_default_constructor_requires_live_opt_in_without_env_read() -> None:
    calls: list[str] = []

    def env_getter(name: str) -> str | None:
        calls.append(name)
        raise AssertionError("environment must not be read")

    with pytest.raises(cb.CloreLiveOptInRequired):
        cb.CloreBridge(env_getter=env_getter)
    assert calls == []


def test_fake_transport_never_reads_environment() -> None:
    calls: list[str] = []

    def env_getter(name: str) -> str | None:
        calls.append(name)
        raise AssertionError("environment must not be read")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture("my_orders.json"))

    async def scenario() -> None:
        bridge = cb.CloreBridge(
            transport=httpx.MockTransport(handler), env_getter=env_getter
        )
        try:
            await bridge.read_order(7001)
        finally:
            await bridge.aclose()

    _run(scenario())
    assert calls == []

class _HangingResponse(httpx.Response):
    async def aclose(self) -> None:
        await asyncio.sleep(10)


class _BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"code":0,"orders":'
        raise httpx.ReadError("secret-stream-body")

    async def aclose(self) -> None:
        return None


def test_hanging_response_close_is_bounded_and_explicit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _HangingResponse(200, json=_fixture("my_orders.json"))

    started = __import__("time").monotonic()
    with pytest.raises(cb.CloreCleanupError):
        _run(_read_with(handler, timeout_seconds=0.2))
    assert __import__("time").monotonic() - started < 2.0


def test_hanging_close_after_cancel_is_ambiguous_no_retry() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _HangingResponse(200, json=_fixture("cancel_code0.json"))

    with pytest.raises(cb.CloreCancelAmbiguous):
        _run(_cancel_with(handler, timeout_seconds=0.2))
    assert calls == 1


def test_partial_stream_error_is_sanitized_and_postsend_ambiguous() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=_BrokenStream())

    with pytest.raises(cb.CloreCancelAmbiguous) as exc_info:
        _run(_cancel_with(handler))
    assert calls == 1
    assert "secret-stream-body" not in str(exc_info.value)


def test_client_aclose_is_bounded() -> None:
    class HangingClient:
        async def aclose(self) -> None:
            await asyncio.sleep(10)

    async def scenario() -> None:
        bridge = cb.CloreBridge(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
            api_key="fixture-key",
        )
        bridge._client = HangingClient()  # type: ignore[assignment]
        with pytest.raises(cb.CloreCleanupError):
            await bridge.aclose(timeout_seconds=0.05)

    _run(scenario())