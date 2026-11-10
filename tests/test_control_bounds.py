from __future__ import annotations

import time

import pytest

from anime_studio import comfy_http
from anime_studio.errors import (
    RemoteAmbiguousSubmit,
    RemoteProtocolError,
    RemoteTransientError,
)
from tests.comfy_stub import run_stub
from tests.p1_helpers import make_transport


LIMIT = 1024
TOTAL = 256 * 1024


@pytest.fixture(autouse=True)
def small_control_limit(monkeypatch):
    monkeypatch.setattr(comfy_http, "MAX_CONTROL_RESPONSE_BYTES", LIMIT)


def _assert_early_abort(state) -> None:
    assert state.oversized_written < TOTAL
    assert state.oversized_written >= LIMIT


def test_oversized_upload_fails_before_prompt(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.oversized_path = "upload"
        state.oversized_total = TOTAL
        transport = make_transport(tmp_path, base_url=base_url)
        started = time.monotonic()
        with pytest.raises(RemoteTransientError):
            transport.submit("corr-upload-oversize", {"prompt": "oversize", "seed": 1})
        assert time.monotonic() - started < 5
        assert state.prompt_requests == 0
        _assert_early_abort(state)


def test_oversized_submit_is_ambiguous_not_rejected(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.oversized_path = "prompt"
        state.oversized_total = TOTAL
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(RemoteAmbiguousSubmit):
            transport.submit("corr-submit-oversize", {"prompt": "oversize", "seed": 1})
        assert state.prompt_requests == 1
        _assert_early_abort(state)


def test_oversized_queue_and_history_are_protocol_errors(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.oversized_path = "history"
        state.oversized_total = TOTAL
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(RemoteProtocolError):
            transport.get("prompt-1")
        _assert_early_abort(state)

    with run_stub() as (base_url, state):
        state.oversized_path = "queue"
        state.oversized_total = TOTAL
        state.history_missing = True
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(RemoteProtocolError):
            transport.get("prompt-1")
        _assert_early_abort(state)


def test_oversized_cancel_responses_do_not_confirm_cancel(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.oversized_path = "queue_delete"
        state.oversized_total = TOTAL
        state.queue_pending = ["prompt-1"]
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(RemoteProtocolError):
            transport.cancel("prompt-1")
        assert state.interrupt_calls == 0
        _assert_early_abort(state)

    with run_stub() as (base_url, state):
        state.oversized_path = "interrupt"
        state.oversized_total = TOTAL
        state.queue_running = ["prompt-1"]
        transport = make_transport(tmp_path, base_url=base_url, exclusive_worker=True)
        with pytest.raises(RemoteProtocolError):
            transport.cancel("prompt-1")
        assert state.interrupt_calls == 1
        _assert_early_abort(state)


def test_control_total_deadline_on_slow_queue(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.queue_delay_seconds = 0.5
        state.history_missing = True
        transport = make_transport(tmp_path, base_url=base_url, read_timeout_seconds=0.1)
        with pytest.raises(RemoteTransientError):
            transport.get("prompt-1")
