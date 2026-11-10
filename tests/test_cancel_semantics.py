from __future__ import annotations

from pathlib import Path

import pytest

from anime_studio.comfy_http import ComfyHttpTransport
from anime_studio.errors import RemoteCancelUnsupported, RemotePromptLost
from anime_studio.p1_config import load_config
from anime_studio.workflow import load_workflow, workflow_sha256
from tests.comfy_stub import run_stub
from tests.p1_helpers import make_manifest, write_config, write_manifest, write_workflow


def _transport(
    tmp_path: Path, base_url: str, *, exclusive_worker: bool = True
) -> ComfyHttpTransport:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        make_manifest(workflow_sha256=workflow_sha256(workflow_path)),
    )
    config_path = write_config(
        tmp_path,
        base_url=base_url,
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        workflow_sha256=workflow_sha256(workflow_path),
        exclusive_worker=exclusive_worker,
    )
    config = load_config(config_path, base_dir=tmp_path)
    workflow, _ = load_workflow(workflow_path)
    return ComfyHttpTransport(
        config,
        workflow,
        prompt_lost_grace_seconds=0,
    )


def test_pending_cancel_is_targeted_and_does_not_interrupt(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url)
        state.queue_pending = ["prompt-1"]
        result = transport.cancel("prompt-1")
        assert result is not None and result.state == "cancelled"
        assert state.delete_calls == [["prompt-1"]]
        assert state.interrupt_calls == 0


def test_shared_running_cancel_refuses_global_interrupt(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url, exclusive_worker=False)
        state.queue_running = ["prompt-1"]
        with pytest.raises(RemoteCancelUnsupported):
            transport.cancel("prompt-1")
        assert state.interrupt_calls == 0


def test_exclusive_running_interrupt_only_when_single_own_prompt(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url, exclusive_worker=True)
        state.queue_running = ["prompt-1"]
        state.set_cancelled()
        state.queue_running = ["prompt-1"]
        result = transport.cancel("prompt-1")
        assert result is not None and result.state == "cancelled"
        assert state.interrupt_calls == 1


def test_exclusive_interrupt_refuses_foreign_active_prompt(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url, exclusive_worker=True)
        state.queue_running = ["prompt-1", "another-job"]
        with pytest.raises(RemoteCancelUnsupported):
            transport.cancel("prompt-1")
        assert state.interrupt_calls == 0


def test_unconfirmed_interrupt_is_not_reported_cancelled(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url, exclusive_worker=True)
        state.queue_running = ["prompt-1"]
        state.interrupt_cancels = False
        with pytest.raises(RemoteCancelUnsupported):
            transport.cancel("prompt-1")
        assert state.interrupt_calls == 1


def test_cancel_without_prompt_or_history_does_not_claim_remote_absent(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url)
        state.queue_pending = []
        state.queue_running = []
        state.history_missing = True
        with pytest.raises(RemotePromptLost):
            transport.cancel("unknown-prompt")
        assert state.interrupt_calls == 0
        assert state.delete_calls == []
