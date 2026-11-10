from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from scripts.p1 import marketplace_quote

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "marketplace.json"
SOURCE_META = (
    ROOT
    / "docs"
    / "p1-evidence"
    / "p1-scoped-proposal-20261027"
    / "public-clore-api-20261027.meta.txt"
)
NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
SERVER_ID = 120777
HOSTER_ID = 47111
CURRENCY = "USD-Blockchain"
DAY_RATE = 9.6


def _fixture_body() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _body(
    *,
    server_id: int = SERVER_ID,
    hoster_id: int = HOSTER_ID,
    price: object = DAY_RATE,
    mrl: object = 72,
    rented: object = False,
    online: object = None,
    code: object = 0,
    extra_rows: list[dict] | None = None,
    my_servers: object = None,
    servers: object = None,
) -> dict:
    payload = _fixture_body()
    row = {
        "id": server_id,
        "owner": hoster_id,
        "mrl": mrl,
        "price": {"on_demand": {CURRENCY: price}, "spot": {}},
        "rented": rented,
        "specs": {},
    }
    if online is not None:
        row["online"] = online
    rows = [row]
    if extra_rows:
        rows.extend(extra_rows)
    payload["servers"] = rows
    payload["code"] = code
    if my_servers is not None:
        payload["my_servers"] = my_servers
    if servers is not None:
        payload["servers"] = servers
    return payload


def _collect(tmp_path: Path, payload: dict, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://api.clore.ai/v1/marketplace"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    output = tmp_path / "quote.json"
    document = asyncio.run(
        marketplace_quote.collect_marketplace_quote(
            server_id=kwargs.pop("server_id", SERVER_ID),
            output_path=output,
            transport=transport,
            now=NOW,
            clock=lambda: 0.0,
            sleep=lambda seconds: asyncio.sleep(0),
            **kwargs,
        )
    )
    return output, document


def test_fresh_quote_is_bounded_public_facts_only(tmp_path: Path) -> None:
    output, document = _collect(tmp_path, _body())
    raw = output.read_text(encoding="utf-8")
    assert "offline-fixture-key" not in raw
    assert '"auth"' not in raw.lower()
    assert "authorization" not in raw.lower()
    assert document["schema"] == marketplace_quote.QUOTE_SCHEMA
    assert document["status"] == "FRESH_MARKETPLACE_QUOTE_READ_ONLY"
    assert document["authorizes_session"] is False
    assert document["provider_billing_verified"] is False
    assert document["server_id"] == SERVER_ID
    assert document["hoster_id"] == HOSTER_ID
    assert document["price_per_day"] == "9.6"
    assert document["mrl_raw_seconds"] == 72 * 3600
    assert document["mrl_semantics"] == "MAXIMUM_RENTAL_LENGTH"
    assert document["availability"]["rented"] is False
    assert document["availability"]["online"] is None
    assert len(document["facts_sha256"]) == 64
    assert document["quote"]["fresh_verified"] is True
    assert document["quote"]["fees_known"] is False
    assert document["quote"]["tax_fx_known"] is False
    assert document["quote"]["creation_fee"] is None
    assert document["quote"]["reserve"] is None
    assert document["quote"]["min_rental_hours"] is None
    assert document["quote"]["provider_fee_percent"] is None
    assert document["proposal"]["authorizes_session"] is False
    assert document["proposal"]["fits_code_cap"] is True
    assert document["proposal"]["projection"]["total_60m"] == "0.62"


def test_marketplace_fixture_matches_public_contract_shape() -> None:
    payload = _fixture_body()
    assert payload["code"] == 0
    assert isinstance(payload["my_servers"], list)
    assert isinstance(payload["servers"], list)
    assert payload["servers"]
    for row in payload["servers"]:
        assert isinstance(row["id"], int) and not isinstance(row["id"], bool)
        assert isinstance(row["owner"], int) and not isinstance(row["owner"], bool)
        assert isinstance(row["mrl"], int) and row["mrl"] > 0
        assert isinstance(row["rented"], bool)
        assert isinstance(row["price"], dict)
        assert isinstance(row["price"]["on_demand"], dict)
        assert row["price"]["on_demand"]
        for currency, price in row["price"]["on_demand"].items():
            assert isinstance(currency, str) and currency
            assert isinstance(price, (int, float)) and not isinstance(price, bool)
            assert price > 0
    meta = SOURCE_META.read_text(encoding="utf-8")
    assert "URL=https://docs.clore.ai/for-hosts/api.md" in meta
    assert "SHA256=90563d6154f1871fd134815d670e731dbf7bd6ae0c9774c92bf949ee02d746d7" in meta
    readme = (FIXTURE.parent / "README.md").read_text(encoding="utf-8")
    assert "SOURCE_RECONSTRUCTION_FROM_CLORE_PUBLIC_API_DOCS_20261027" in readme
    local_pin = (ROOT / "docs" / "P1-clore-contract-20261021.md").read_text(
        encoding="utf-8"
    )
    assert "90563d6154f1871fd134815d670e731dbf7bd6ae0c9774c92bf949ee02d746d7" in local_pin
    owner_forwarded = (
        ROOT / "docs" / "P1-owner-financial-risk-20261026.md"
    ).read_text(encoding="utf-8")
    assert "111947/owner46993/rented=false/12USD-Blockchain" in owner_forwarded


def test_fresh_quote_does_not_emit_synthetic_secret_strings(tmp_path: Path) -> None:
    payload = _body()
    payload["servers"][0]["specs"]["env"] = "SYNTHETIC_SECRET_SHOULD_NOT_APPEAR"
    output, _document = _collect(tmp_path, payload)
    raw = output.read_text(encoding="utf-8")
    assert "SYNTHETIC_SECRET_SHOULD_NOT_APPEAR" not in raw
    assert "offline-fixture-key" not in raw


@pytest.mark.parametrize(
    "payload",
    [
        _body(code=3),
        _body(server_id=999999),
        _body(rented=True),
        _body(online=False),
        _body(my_servers=[SERVER_ID]),
        _body(mrl="72"),
        _body(price="not-a-price"),
        _body(servers="not-a-list"),
        _body(
            extra_rows=[
                {
                    "id": SERVER_ID,
                    "owner": HOSTER_ID,
                    "mrl": 72,
                    "price": {"on_demand": {CURRENCY: 9.0}, "spot": {}},
                    "rented": False,
                    "specs": {},
                }
            ]
        ),
        {"code": 0, "servers": []},
    ],
)
def test_fresh_quote_fails_closed_without_artifact(
    tmp_path: Path, payload: dict
) -> None:
    output = tmp_path / "quote.json"
    with pytest.raises(marketplace_quote.MarketplaceQuoteError):
        asyncio.run(
            marketplace_quote.collect_marketplace_quote(
                server_id=SERVER_ID,
                output_path=output,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=payload)
                ),
                now=NOW,
                clock=lambda: 0.0,
                sleep=lambda seconds: asyncio.sleep(0),
            )
        )
    assert not output.exists()


def test_fresh_quote_currency_unknown_and_malformed_are_refused(tmp_path: Path) -> None:
    row = {
        "id": SERVER_ID,
        "owner": HOSTER_ID,
        "mrl": 72,
        "price": {"on_demand": {"bitcoin": 0.001}, "spot": {}},
        "rented": False,
        "specs": {},
    }
    with pytest.raises(marketplace_quote.MarketplaceQuoteError):
        _collect(tmp_path, {"code": 0, "servers": [row]})
    assert not (tmp_path / "quote.json").exists()


@pytest.mark.parametrize(
    "metadata",
    [
        {"timestamp": (NOW - timedelta(minutes=30)).isoformat()},
        {"updated_at": (NOW + timedelta(minutes=5)).isoformat()},
        {"observed_at": "not-a-timestamp"},
    ],
)
def test_fresh_quote_rejects_stale_or_invalid_cache_metadata(
    tmp_path: Path, metadata: dict
) -> None:
    payload = _body()
    payload.update(metadata)
    with pytest.raises(marketplace_quote.MarketplaceQuoteError):
        _collect(tmp_path, payload)
    assert not (tmp_path / "quote.json").exists()


def test_fresh_quote_rejects_redirect_oversize_timeout_and_auth(
    tmp_path: Path,
) -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.invalid/"})

    def oversized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 4096)

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("synthetic timeout")

    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": 3})

    for name, handler, kwargs in (
        ("redirect", redirect, {}),
        ("oversize", oversized, {"max_response_bytes": 1024}),
        ("timeout", timeout, {"timeout_seconds": 1.0}),
        ("auth", unauthorized, {}),
    ):
        output = tmp_path / f"{name}.json"
        with pytest.raises(marketplace_quote.MarketplaceQuoteError) as excinfo:
            asyncio.run(
                marketplace_quote.collect_marketplace_quote(
                    server_id=SERVER_ID,
                    output_path=output,
                    transport=httpx.MockTransport(handler),
                    now=NOW,
                    clock=lambda: 0.0,
                    sleep=lambda seconds: asyncio.sleep(0),
                    **kwargs,
                )
            )
        assert not output.exists()
        assert "offline-fixture-key" not in str(excinfo.value)
        assert "SYNTHETIC" not in str(excinfo.value)


def test_fresh_quote_missing_credential_is_refused_without_env_or_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "quote.json"
    with pytest.raises(marketplace_quote.MarketplaceQuoteError):
        asyncio.run(
            marketplace_quote.collect_marketplace_quote(
                server_id=SERVER_ID,
                output_path=output,
                allow_live=True,
                env_getter=lambda name: None,
                now=NOW,
                clock=lambda: 0.0,
                sleep=lambda seconds: asyncio.sleep(0),
            )
        )
    assert not output.exists()


def test_fresh_quote_requires_explicit_live_read(tmp_path: Path, capsys) -> None:
    output = tmp_path / "quote.json"
    assert marketplace_quote.main(
        ["--server-id", str(SERVER_ID), "--output", str(output)]
    ) == 2
    assert not output.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "REFUSED"
