from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from anime_studio.comfy_http import ComfyHttpTransport, redact_error, sanitize_diagnostics
from anime_studio.errors import (
    RemoteAmbiguousSubmit,
    RemoteAuthError,
    RemotePromptLost,
    RemoteProtocolError,
    RemoteTransientError,
    RemoteValidationRejected,
)
from anime_studio.p1_config import load_config
from anime_studio.workflow import load_workflow, workflow_sha256
from tests.comfy_stub import run_stub
from tests.p1_helpers import (
    make_manifest,
    make_transport,
    write_config,
    write_manifest,
    write_workflow,
)


def _transport(
    tmp_path: Path,
    base_url: str,
    *,
    exclusive_worker: bool = True,
    prompt_lost_grace_seconds: float = 0.0,
    template: bool = False,
    read_timeout_seconds: float = 2.0,
) -> ComfyHttpTransport:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path, template=template)
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
        read_timeout_seconds=read_timeout_seconds,
    )
    config = load_config(config_path, base_dir=tmp_path)
    workflow, _ = load_workflow(workflow_path)
    return ComfyHttpTransport(
        config,
        workflow,
        prompt_lost_grace_seconds=prompt_lost_grace_seconds,
    )


def test_submit_uploads_fixed_image_and_binds_declared_fields(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url)
        remote = transport.submit("corr-1", {"prompt": "bound prompt", "seed": 7})
        assert remote.prompt_id == "prompt-1"
        assert state.upload_requests == 1
        assert state.prompt_requests == 1
        body = state.prompt_bodies[0]
        assert body["extra_data"]["correlation_key"] == "corr-1"
        workflow = body["prompt"]
        assert workflow["1"]["inputs"]["image"] == "source.png"
        assert workflow["2"]["inputs"]["text"] == "bound prompt"
        assert workflow["6"]["inputs"]["seed"] == 7
        assert workflow["5"]["inputs"]["width"] == 704
        assert workflow["5"]["inputs"]["height"] == 1280
        assert workflow["5"]["inputs"]["length"] == 81
        assert workflow["5"]["inputs"]["fps"] == 24


def test_validation_rejection_is_confirmed_and_has_no_prompt(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_mode = "validation"
        transport = _transport(tmp_path, base_url)
        with pytest.raises(RemoteValidationRejected):
            transport.submit("corr-validation", {"prompt": "bad", "seed": 1})
        assert state.prompt_requests == 1


def test_valid_prompt_id_with_node_errors_is_accepted_with_diagnostics(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_mode = "node_errors"
        transport = _transport(tmp_path, base_url)
        remote = transport.submit("corr-node-errors", {"prompt": "partial", "seed": 1})
        assert remote.prompt_id == "prompt-1"
        assert remote.diagnostics is not None
        assert remote.diagnostics["partial_acceptance"] is True
        assert "node_errors" in remote.diagnostics
        assert state.prompt_requests == 1
        # Identity is in queue, so the prompt can be polled/cancelled normally.
        queued = transport.get(remote.prompt_id)
        assert queued is not None and queued.state == "queued"
        cancelled = transport.cancel(remote.prompt_id)
        assert cancelled is not None and cancelled.state == "cancelled"
        assert state.interrupt_calls == 0
        assert state.delete_calls == [["prompt-1"]]


def test_transient_connect_failure_is_distinct(tmp_path) -> None:
    transport = _transport(tmp_path, "http://127.0.0.1:1")
    with pytest.raises(RemoteTransientError):
        transport.submit("corr-connect", {"prompt": "connect", "seed": 1})


def test_ambiguous_500_and_malformed_200_are_not_marked_rejected(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_mode = "ambiguous_500"
        transport = _transport(tmp_path, base_url)
        with pytest.raises(RemoteAmbiguousSubmit):
            transport.submit("corr-500", {"prompt": "ambiguous", "seed": 1})

    with run_stub() as (base_url, state):
        state.prompt_mode = "malformed_200"
        transport = _transport(tmp_path, base_url)
        with pytest.raises(RemoteAmbiguousSubmit):
            transport.submit("corr-malformed", {"prompt": "ambiguous", "seed": 1})


def test_missing_prompt_id_is_ambiguous_not_rejected(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_mode = "no_prompt_id"
        transport = _transport(tmp_path, base_url)
        with pytest.raises(RemoteAmbiguousSubmit):
            transport.submit("corr-no-id", {"prompt": "ambiguous", "seed": 1})


def test_queue_progress_history_success_and_output_ref(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url)
        remote = transport.submit("corr-progress", {"prompt": "progress", "seed": 2})
        queued = transport.get(remote.prompt_id)
        assert queued is not None and queued.state == "queued"

        state.queue_pending = []
        state.queue_running = [state.prompt_id]
        running = transport.get(remote.prompt_id)
        assert running is not None and running.state == "running"

        state.set_success("result.mp4")
        succeeded = transport.get(remote.prompt_id)
        assert succeeded is not None
        assert succeeded.state == "succeeded"
        assert succeeded.artifact is not None
        assert succeeded.artifact["scheme"] == "comfy-http"
        assert succeeded.artifact["filename"] == "result.mp4"
        assert succeeded.artifact["type"] == "output"


def test_history_execution_error_and_missing_outputs(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url)
        remote = transport.submit("corr-error", {"prompt": "error", "seed": 2})
        state.set_error("bad execution")
        failed = transport.get(remote.prompt_id)
        assert failed is not None and failed.state == "failed"
        assert failed.error_code == "REMOTE_EXECUTION_ERROR"

    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url)
        remote = transport.submit("corr-empty", {"prompt": "empty", "seed": 2})
        state.history[state.prompt_id] = {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {},
        }
        failed = transport.get(remote.prompt_id)
        assert failed is not None and failed.state == "failed"
        assert failed.error_code == "REMOTE_RESULT_MISSING"


def test_lost_prompt_is_distinct_from_absent_cancel_confirmation(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url, prompt_lost_grace_seconds=0)
        remote = transport.submit("corr-lost", {"prompt": "lost", "seed": 2})
        state.history_missing = True
        state.queue_pending = []
        with pytest.raises(RemotePromptLost):
            transport.get(remote.prompt_id)


def test_reconcile_is_explicitly_unsupported(tmp_path) -> None:
    with run_stub() as (base_url, _state):
        transport = _transport(tmp_path, base_url)
        assert transport.supports_reconcile_by_correlation_key is False
        assert transport.supports_idempotent_submit is False
        assert transport.reconcile("corr-any") is None


def test_template_workflow_is_rejected_before_upload(tmp_path) -> None:
    with run_stub() as (base_url, state):
        transport = _transport(tmp_path, base_url, template=True)
        with pytest.raises(RemoteProtocolError) as exc_info:
            transport.submit("corr-template", {"prompt": "template", "seed": 1})
        assert exc_info.value.code == "WORKFLOW_TEMPLATE_NOT_EXECUTABLE"
        assert state.upload_requests == 0
        assert state.prompt_requests == 0


def test_redaction_removes_tokens_and_headers() -> None:
    text = redact_error(
        "Authorization: Bearer super-secret token=abc123 url=https://x/?token=abc123",
        "super-secret",
    )
    assert "super-secret" not in text
    assert "abc123" not in text


def test_non_loopback_requires_authenticated_channel(tmp_path) -> None:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        make_manifest(workflow_sha256=workflow_sha256(workflow_path)),
    )
    config_path = write_config(
        tmp_path,
        base_url="https://example.invalid",
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        workflow_sha256=workflow_sha256(workflow_path),
    )
    config = load_config(config_path, base_dir=tmp_path)
    workflow, _ = load_workflow(workflow_path)
    with pytest.raises(RemoteAuthError):
        ComfyHttpTransport(config, workflow, token=None)

def test_submit_redirect_is_ambiguous_and_forbidden(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_mode = "redirect"
        transport = _transport(tmp_path, base_url)
        with pytest.raises(RemoteAmbiguousSubmit):
            transport.submit("corr-redirect", {"prompt": "redirect", "seed": 1})

def test_submit_read_timeout_is_ambiguous(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.prompt_delay_seconds = 0.5
        transport = _transport(tmp_path, base_url, read_timeout_seconds=0.1)
        with pytest.raises(RemoteAmbiguousSubmit):
            transport.submit("corr-read-timeout", {"prompt": "ambiguous", "seed": 1})


def test_submit_write_error_is_ambiguous(tmp_path, monkeypatch) -> None:
    with run_stub() as (base_url, _state):
        transport = _transport(tmp_path, base_url)
        original = transport._request_bytes

        async def fail_prompt(method, path, **kwargs):
            if path == "/prompt":
                raise httpx.WriteError("fixture write failure")
            return await original(method, path, **kwargs)

        monkeypatch.setattr(transport, "_request_bytes", fail_prompt)
        with pytest.raises(RemoteAmbiguousSubmit):
            transport.submit("corr-write-error", {"prompt": "ambiguous", "seed": 1})


def test_upload_response_delay_is_bounded_by_total_deadline(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.upload_response_delay_seconds = 0.5
        transport = make_transport(
            tmp_path, base_url=base_url, upload_timeout_seconds=0.2
        )
        with pytest.raises(RemoteTransientError):
            transport.submit("corr-upload-slow", {"prompt": "slow upload", "seed": 1})
        assert state.prompt_requests == 0


def test_upload_read_delay_is_bounded_by_total_deadline(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.upload_read_delay_seconds = 0.5
        transport = make_transport(
            tmp_path, base_url=base_url, upload_timeout_seconds=0.2
        )
        with pytest.raises(RemoteTransientError):
            transport.submit("corr-upload-read", {"prompt": "slow upload", "seed": 1})
        assert state.prompt_requests == 0


def test_diagnostics_are_bounded_and_redacted() -> None:
    token = "SYNTHETIC_TRANSPORT_TOKEN"
    nested = {
        "token": "SYNTHETIC_NESTED_TOKEN",
        "password": "SYNTHETIC_PASSWORD",
        "authorization": "Bearer " + token,
        "url": "https://example.invalid/?api_key=SYNTHETIC_QUERY_SECRET",
        "note": "Authorization: Bearer " + token,
        "deep": {"a": {"b": {"c": {"d": {"e": {"f": {"g": "too deep"}}}}}}},
        "many": list(range(100)),
    }
    sanitized = sanitize_diagnostics(nested, token)
    text = __import__("json").dumps(sanitized, ensure_ascii=False)
    assert token not in text
    assert "SYNTHETIC_NESTED_TOKEN" not in text
    assert "SYNTHETIC_PASSWORD" not in text
    assert "SYNTHETIC_QUERY_SECRET" not in text
    assert "TRUNCATED" in text or "many" in text
    assert len(text.encode("utf-8")) <= 8192


def test_diagnostics_fallback_unicode_and_escaping_stays_within_bytes() -> None:
    token = "SYNTHETIC_UNICODE_TOKEN_777"
    noisy = (
        "\u4e2d" * 100
        + "\U0001f600" * 50
        + '"quote\\backslash?token=' + token
        + "\u4e2d" * 700
        + "\U0001f600" * 250
    )
    payload = {
        "partial_acceptance": True,
        "node_errors": [noisy for _ in range(50)],
        "note": "Authorization: Bearer " + token,
    }
    source_text = payload["node_errors"][0]
    assert "\u4e2d" in source_text
    assert "\U0001f600" in source_text
    sanitized = sanitize_diagnostics(payload, token)
    encoded = __import__("json").dumps(
        sanitized, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    assert len(encoded) <= 8192
    assert sanitized["partial_acceptance"] is True
    assert "truncated" in sanitized
    preview = sanitized["preview"]
    assert "\u4e2d" in preview
    assert "\U0001f600" in preview
    assert '\\"quote' in preview
    assert "\\\\backslash" in preview
    assert "[REDACTED]" in preview
    assert token not in encoded.decode("utf-8")


def test_diagnostics_sanitizes_unpaired_surrogate_keys_and_values() -> None:
    payload = {
        "partial_acceptance": True,
        "\ud800key": "\udfffvalue",
        "nested": {"\ud801key": ["ok", "\ud802value"]},
    }
    sanitized = sanitize_diagnostics(payload, "SYNTHETIC_SURROGATE_TOKEN")
    encoded = __import__("json").dumps(
        sanitized, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    assert sanitized["partial_acceptance"] is True
    assert len(encoded) <= 8192
    decoded = encoded.decode("utf-8")
    assert "?" in decoded
    assert not any(0xD800 <= ord(char) <= 0xDFFF for char in decoded)
