from __future__ import annotations

import pytest

from anime_studio.errors import (
    RemoteCancelUnsupported,
    RemoteProtocolError,
)
from tests.comfy_stub import run_stub
from tests.p1_helpers import make_transport


@pytest.mark.parametrize("mode", ["missing_keys", "bad_type", "bad_entry"])
def test_malformed_queue_is_not_treated_as_empty(tmp_path, mode: str) -> None:
    with run_stub() as (base_url, state):
        state.queue_mode = mode
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(RemoteProtocolError):
            transport.get("prompt-1")
        with pytest.raises(RemoteProtocolError):
            transport.cancel("prompt-1")


def test_post_delete_malformed_queue_is_not_confirmed_cancellation(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        state.queue_pending = ["prompt-1"]
        state.queue_after_delete_mode = "bad_entry"
        with pytest.raises(RemoteProtocolError):
            transport.cancel("prompt-1")
        assert state.interrupt_calls == 0
        assert state.delete_calls == [["prompt-1"]]


def test_pending_to_running_race_does_not_claim_cancel(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url, exclusive_worker=True)
        state.queue_pending = ["prompt-1"]
        state.queue_after_delete_mode = "running"
        with pytest.raises(RemoteCancelUnsupported):
            transport.cancel("prompt-1")
        assert state.interrupt_calls == 0


def test_pending_to_completed_race_returns_success_not_cancelled(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        state.queue_pending = ["prompt-1"]
        state.set_success()
        state.queue_pending = ["prompt-1"]
        state.queue_after_delete_mode = "empty"
        result = transport.cancel("prompt-1")
        assert result is not None
        assert result.state == "succeeded"
        assert state.interrupt_calls == 0
