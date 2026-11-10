"""Real ComfyUI HTTP transport, isolated behind the P0 ComfyAdapter protocol.

This module is prepared for P1 but is never selected by default. It is built
only with an explicit live opt-in and a validated readiness manifest. All
offline contract tests use a loopback HTTP stub.

ComfyUI has no durable correlation-key reconciliation primitive. Therefore
``supports_reconcile_by_correlation_key`` and ``supports_idempotent_submit`` are
both false: an ambiguous submission is recorded and never automatically retried.
Progress is intentionally coarse: queue pending/running plus terminal history.
ComfyUI's built-in HTTP API does not expose per-step sampler progress.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx

from .comfy import RemoteStatus
from .errors import (
    RemoteAmbiguousSubmit,
    RemoteAuthError,
    RemoteCancelUnsupported,
    RemoteDeadlineExpired,
    RemotePromptLost,
    RemoteProtocolError,
    RemoteTransientError,
    RemoteValidationRejected,
)
from .p1_config import (
    P1RuntimeConfig,
    execution_fingerprint,
    sha256_file,
)
from .timeutils import ms_to_iso, now_ms
from .workflow import (
    TEMPLATE_MARKER,
    WorkflowBinding,
    load_workflow,
    safe_remote_filename,
    workflow_sha256,
)


_CREDENTIAL_RE = re.compile(
    r"(?i)(authorization|bearer|access[_-]?token|token|api[_-]?key|password|secret)"
    r"\s*[:=]\s*\S+"
)
_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)(authorization|bearer|access[_-]?token|token|api[_-]?key|apikey|"
    r"password|secret|credential)"
)
MAX_CONTROL_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_DIAGNOSTICS_BYTES = 8192
MAX_DIAGNOSTICS_DEPTH = 6
MAX_DIAGNOSTICS_ITEMS = 50
MAX_DIAGNOSTICS_STRING = 1000


class _ControlTooLarge(Exception):
    pass


class _DeadlineBeforeRequest(asyncio.TimeoutError):
    """The request deadline expired before any bytes were sent.

    Kept distinct from a timeout while ``client.send`` is in flight: only this
    pre-send failure can be classified as confirmed-not-sent. When the request
    budget came from the persisted job deadline, the subtype below lets the
    caller distinguish a job timeout from a plain HTTP read/write budget.
    """


class _JobDeadlineBeforeRequest(_DeadlineBeforeRequest):
    """The persisted job deadline expired while preparing the request."""


def _sanitize_diagnostics_value(value: Any, token: str | None, depth: int = 0) -> Any:
    if depth > MAX_DIAGNOSTICS_DEPTH:
        return "[TRUNCATED_DEPTH]"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, child) in enumerate(value.items()):
            if index >= MAX_DIAGNOSTICS_ITEMS:
                result["__truncated__"] = "item limit"
                break
            raw_key = str(key)
            safe_key = redact_error(raw_key, token)[:200]
            safe_key = safe_key.encode("utf-8", errors="replace").decode("utf-8")
            if _CREDENTIAL_KEY_RE.search(raw_key):
                result[safe_key] = "[REDACTED]"
            else:
                result[safe_key] = _sanitize_diagnostics_value(child, token, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        result_list = [
            _sanitize_diagnostics_value(item, token, depth + 1)
            for item in list(value)[:MAX_DIAGNOSTICS_ITEMS]
        ]
        if len(value) > MAX_DIAGNOSTICS_ITEMS:
            result_list.append("[TRUNCATED_ITEMS]")
        return result_list
    if isinstance(value, str):
        safe = redact_error(value, token)[:MAX_DIAGNOSTICS_STRING]
        return safe.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return redact_error(str(value), token)[:MAX_DIAGNOSTICS_STRING]


def _diagnostics_bytes(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def sanitize_diagnostics(value: Any, token: str | None = None) -> Any:
    normalized = _sanitize_diagnostics_value(value, token)
    text = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    if len(text.encode("utf-8")) <= MAX_DIAGNOSTICS_BYTES:
        return normalized

    original_bytes = len(text.encode("utf-8"))
    wrapper: dict[str, Any] = {
        "truncated": True,
        "original_bytes": original_bytes,
        "preview": "",
    }
    if isinstance(normalized, dict) and "partial_acceptance" in normalized:
        wrapper["partial_acceptance"] = normalized["partial_acceptance"]

    # The preview is JSON-escaped inside the wrapper. Binary-search the largest
    # character prefix whose *whole encoded wrapper* still fits the byte limit.
    low, high = 0, len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = dict(wrapper)
        candidate["preview"] = text[:middle]
        if _diagnostics_bytes(candidate) <= MAX_DIAGNOSTICS_BYTES:
            best = candidate["preview"]
            low = middle + 1
        else:
            high = middle - 1
    bounded = dict(wrapper)
    bounded["preview"] = best
    if _diagnostics_bytes(bounded) > MAX_DIAGNOSTICS_BYTES:
        # Wrapper/partial_acceptance alone must still fit; drop preview.
        bounded["preview"] = ""
    return bounded




def redact_error(value: Any, token: str | None = None) -> str:
    text = str(value)
    if token:
        text = text.replace(token, "[REDACTED]")
    text = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)\S+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", text)
    text = _CREDENTIAL_RE.sub(r"\1=[REDACTED]", text)
    text = re.sub(
        r"([?&](?:token|access_token|api[_-]?key|key|password|secret)=)[^&\s]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def _json_or_none(response: httpx.Response) -> dict[str, Any] | None:
    try:
        value = response.json()
    except (ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class ComfyHttpTransport:
    mode = "COMFY_HTTP"
    supports_reconcile_by_correlation_key = False
    supports_idempotent_submit = False

    def __init__(
        self,
        config: P1RuntimeConfig,
        workflow: Mapping[str, Any],
        *,
        token: str | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        prompt_lost_grace_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], int] = now_ms,
    ) -> None:
        self.config = config
        self.workflow = workflow
        self.binding: WorkflowBinding = config.binding
        input_sha = None
        input_size = None
        if config.input_image_path is not None and config.input_image_path.is_file():
            try:
                input_sha = sha256_file(config.input_image_path)
                input_size = config.input_image_path.stat().st_size
            except Exception:
                input_sha = None
                input_size = None
        manifest_sha = (
            sha256_file(config.manifest_path)
            if config.manifest_path.is_file()
            else "MISSING"
        )
        workflow_sha = (
            workflow_sha256(config.workflow_path)
            if config.workflow_path.is_file()
            else (config.workflow_sha256 or "MISSING")
        )
        self.execution_fingerprint = execution_fingerprint(
            config,
            workflow_sha256=workflow_sha,
            manifest_sha256=manifest_sha,
            input_image_sha256=input_sha,
            input_image_size=input_size,
        )
        self.prompt_lost_grace_seconds = prompt_lost_grace_seconds
        self._clock = clock
        self._wall_clock = wall_clock
        self._prompt_first_missing: dict[str, float] = {}
        self._uploaded_image_name: str | None = None
        self._async_transport = async_transport
        # Per-call deadline context for the async submit helper. A ContextVar
        # keeps concurrent/threaded callers isolated while preserving the
        # existing two-argument helper signature used by diagnostics/tests.
        self._active_deadline_ms: ContextVar[int | None] = ContextVar(
            "comfy_http_submit_deadline", default=None
        )

        if token is None and config.token_env:
            token = os.environ.get(config.token_env)
        if not config.loopback and not token:
            raise RemoteAuthError(
                "Non-loopback ComfyUI requires an authentication token from "
                f"environment variable {config.token_env or '<unset>'}; "
                "public/unauthenticated live endpoints are not allowed.",
                code="REMOTE_AUTH_ERROR",
            )
        self._token = token

    def __repr__(self) -> str:
        return f"ComfyHttpTransport(base_url={self.config.base_url!r}, mode='COMFY_HTTP')"

    def close(self) -> None:
        # Per-request AsyncClients are closed by their async context managers.
        return None

    # --------------------------------------------------------------- helpers
    def _message(self, value: Any) -> str:
        return redact_error(value, self._token)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _remaining_job_seconds(self, deadline_ms: int | None) -> float | None:
        if deadline_ms is None:
            return None
        # persisted deadlines are wall-clock epoch milliseconds; convert the
        # remaining wall budget to a monotonic request budget without extending
        # the job's absolute deadline.
        return (int(deadline_ms) - int(self._wall_clock())) / 1000.0

    def _ensure_job_deadline(
        self, deadline_ms: int | None, stage: str
    ) -> None:
        if deadline_ms is not None and int(self._wall_clock()) >= int(deadline_ms):
            raise RemoteDeadlineExpired(
                f"Job deadline expired {stage}; no /prompt was sent. "
                "This is a confirmed-not-sent timeout, not an ambiguous dispatch.",
                code="DEADLINE_EXPIRED_BEFORE_SEND",
            )

    def _mark_present(self, prompt_id: str) -> None:
        """Clear the missing interval after any confirmed prompt presence."""

        self._prompt_first_missing.pop(prompt_id, None)

    def _protocol_error(self, response: httpx.Response, action: str) -> RemoteProtocolError:
        return RemoteProtocolError(
            f"{action}: unexpected HTTP {response.status_code}", code="REMOTE_PROTOCOL_ERROR"
        )

    @staticmethod
    def _parse_json_body(body: bytes, action: str) -> dict[str, Any]:
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteProtocolError(
                f"{action}: response is not valid UTF-8 JSON",
                code="REMOTE_PROTOCOL_ERROR",
            ) from exc
        if not isinstance(data, dict):
            raise RemoteProtocolError(
                f"{action}: response is not a JSON object", code="REMOTE_PROTOCOL_ERROR"
            )
        return data

    async def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        deadline: float,
        action: str,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        files: Any = None,
        data: Any = None,
        job_deadline_ms: int | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineBeforeRequest(
                f"{action}: deadline expired before request"
            )
        timeout = httpx.Timeout(
            connect=min(self.config.connect_timeout_seconds, remaining),
            read=remaining,
            write=remaining,
            pool=remaining,
        )
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        async with httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
            transport=self._async_transport,
        ) as client:
            request = client.build_request(
                method,
                path,
                json=json_body,
                params=params,
                files=files,
                data=data,
            )
            # build_request/client preparation can consume a material part of
            # the budget. Re-check both the persisted wall deadline and the
            # monotonic budget immediately before the real send; use the fresh
            # remaining value for wait_for.
            if job_deadline_ms is not None and int(self._wall_clock()) >= int(
                job_deadline_ms
            ):
                raise _JobDeadlineBeforeRequest(
                    f"{action}: job deadline expired while preparing the request"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _DeadlineBeforeRequest(
                    f"{action}: deadline expired while preparing the request"
                )
            response = await asyncio.wait_for(
                client.send(request, stream=True), timeout=remaining
            )
            try:
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError as exc:
                        raise _ControlTooLarge(
                            f"{action}: malformed Content-Length"
                        ) from exc
                    if declared_size < 0 or declared_size > MAX_CONTROL_RESPONSE_BYTES:
                        raise _ControlTooLarge(
                            f"{action}: declared body exceeds the control limit"
                        )
                body = bytearray()
                total = 0
                iterator = response.aiter_bytes().__aiter__()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError(f"{action}: total deadline exceeded")
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_CONTROL_RESPONSE_BYTES:
                        raise _ControlTooLarge(
                            f"{action}: streamed body exceeds the control limit"
                        )
                    body.extend(chunk)
                if time.monotonic() > deadline:
                    raise asyncio.TimeoutError(f"{action}: total deadline exceeded")
                return (
                    response.status_code,
                    {key: value for key, value in response.headers.items()},
                    bytes(body),
                )
            finally:
                await response.aclose()

    def _queue(self) -> list[tuple[str, str]]:
        return asyncio.run(self._queue_async())

    async def _queue_async(self) -> list[tuple[str, str]]:
        deadline = time.monotonic() + self.config.read_timeout_seconds
        try:
            status, _headers, body = await self._request_bytes(
                "GET", "/queue", deadline=deadline, action="queue"
            )
        except asyncio.TimeoutError as exc:
            raise RemoteTransientError(
                self._message("queue request exceeded the total deadline"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except _ControlTooLarge as exc:
            raise RemoteProtocolError(str(exc)) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RemoteTransientError(
                self._message(f"queue connect failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            raise RemoteTransientError(
                self._message(f"queue read failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        if status in {401, 403}:
            raise RemoteAuthError("ComfyUI queue endpoint rejected authentication")
        if status >= 500:
            raise RemoteTransientError(
                f"ComfyUI queue endpoint returned HTTP {status}"
            )
        if status != 200:
            raise RemoteProtocolError(f"queue: unexpected HTTP {status}")
        data = self._parse_json_body(body, "queue")
        result: list[tuple[str, str]] = []
        for key, state in (("queue_running", "running"), ("queue_pending", "queued")):
            if key not in data:
                raise RemoteProtocolError(
                    f"queue response is missing {key!r}; absence is not a valid "
                    "empty queue"
                )
            entries = data[key]
            if not isinstance(entries, list):
                raise RemoteProtocolError(f"queue.{key} must be a list")
            for index, entry in enumerate(entries):
                prompt_id: Any = None
                if isinstance(entry, list):
                    if len(entry) < 2:
                        raise RemoteProtocolError(
                            f"queue.{key}[{index}] is a malformed entry"
                        )
                    prompt_id = entry[1]
                elif isinstance(entry, Mapping):
                    prompt_id = entry.get("prompt_id")
                else:
                    raise RemoteProtocolError(
                        f"queue.{key}[{index}] has an unsupported type"
                    )
                if not isinstance(prompt_id, str) or not prompt_id:
                    raise RemoteProtocolError(
                        f"queue.{key}[{index}] has an invalid prompt id"
                    )
                if len(prompt_id) > 256 or any(
                    ord(char) < 32 or ord(char) == 127 for char in prompt_id
                ):
                    raise RemoteProtocolError(
                        f"queue.{key}[{index}] has an unsafe prompt id"
                    )
                result.append((state, prompt_id))
        return result

    def _queue_state(self, prompt_id: str) -> str | None:
        for state, candidate in self._queue():
            if candidate == prompt_id:
                return state
        return None

    def _history(self, prompt_id: str) -> dict[str, Any] | None:
        return asyncio.run(self._history_async(prompt_id))

    async def _history_async(self, prompt_id: str) -> dict[str, Any] | None:
        deadline = time.monotonic() + self.config.read_timeout_seconds
        try:
            status, _headers, body = await self._request_bytes(
                "GET",
                f"/history/{prompt_id}",
                deadline=deadline,
                action="history",
            )
        except asyncio.TimeoutError as exc:
            raise RemoteTransientError(
                self._message("history request exceeded the total deadline"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except _ControlTooLarge as exc:
            raise RemoteProtocolError(str(exc)) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RemoteTransientError(
                self._message(f"history connect failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            raise RemoteTransientError(
                self._message(f"history read failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        if status in {401, 403}:
            raise RemoteAuthError("ComfyUI history endpoint rejected authentication")
        if status == 404:
            return None
        if status >= 500:
            raise RemoteTransientError(
                f"ComfyUI history endpoint returned HTTP {status}"
            )
        if status != 200:
            raise RemoteProtocolError(f"history: unexpected HTTP {status}")
        data = self._parse_json_body(body, "history")
        entry = data.get(prompt_id)
        if entry is None:
            # Some ComfyUI versions return the entry at top level for /history/{id}.
            entry = data if data.get("outputs") is not None or data.get("status") is not None else None
        if entry is None:
            return None
        if not isinstance(entry, Mapping):
            raise RemoteProtocolError("history entry must be an object")
        return dict(entry)

    @staticmethod
    def _status_value(entry: Mapping[str, Any]) -> tuple[str | None, str | None]:
        status = entry.get("status")
        status_str = None
        messages: Any = None
        if isinstance(status, Mapping):
            status_str = status.get("status_str")
            messages = status.get("messages")
        elif isinstance(status, str):
            status_str = status
        error_message = None
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, list) and len(message) >= 2:
                    event, payload = message[0], message[1]
                    if event in {"execution_error", "execution_interrupted"}:
                        if isinstance(payload, Mapping):
                            error_message = payload.get("exception_message") or payload.get(
                                "exception_type"
                            )
                        if not error_message:
                            error_message = str(event)
                        if event == "execution_interrupted":
                            status_str = "cancelled"
                        break
        return (
            status_str if isinstance(status_str, str) else None,
            str(error_message) if error_message else None,
        )

    def _output_ref(self, entry: Mapping[str, Any]) -> dict[str, Any] | None:
        outputs = entry.get("outputs")
        if not isinstance(outputs, Mapping):
            return None
        node = outputs.get(self.binding.output.node_id)
        if not isinstance(node, Mapping):
            return None
        for key in self.binding.output.keys:
            values = node.get(key)
            if not isinstance(values, list) or not values:
                continue
            value = values[0]
            if not isinstance(value, Mapping):
                continue
            filename = value.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            try:
                safe_name = safe_remote_filename(filename)
            except Exception as exc:
                raise RemoteProtocolError(f"unsafe output filename: {exc}") from exc
            suffix = Path(safe_name).suffix.lower()
            if suffix != ".mp4":
                raise RemoteProtocolError(
                    "P1 output reference must be an MP4 file",
                    code="REMOTE_PROTOCOL_ERROR",
                )
            subfolder = value.get("subfolder", "")
            if not isinstance(subfolder, str):
                raise RemoteProtocolError("output subfolder must be a string")
            artifact_type = value.get("type", "output")
            if artifact_type != "output":
                raise RemoteProtocolError("output reference type must be output")
            return {
                "scheme": "comfy-http",
                "filename": safe_name,
                "subfolder": subfolder,
                "type": "output",
                "media_type": "video/mp4",
            }
        return None

    def _status_from_entry(self, prompt_id: str, entry: Mapping[str, Any]) -> RemoteStatus:
        status_str, error_message = self._status_value(entry)
        artifact = self._output_ref(entry)
        if status_str == "cancelled":
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key="",
                state="cancelled",
                progress=0,
                updated_at=ms_to_iso(now_ms()),
            )
        if status_str in {"error", "failed"} or error_message:
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key="",
                state="failed",
                progress=0,
                error_code="REMOTE_EXECUTION_ERROR",
                error_message=self._message(
                    error_message or "ComfyUI reported an execution error"
                ),
                updated_at=ms_to_iso(now_ms()),
            )
        if artifact is not None:
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key="",
                state="succeeded",
                progress=100,
                artifact=artifact,
                updated_at=ms_to_iso(now_ms()),
            )
        if status_str == "success" or entry.get("completed") is True:
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key="",
                state="failed",
                progress=0,
                error_code="REMOTE_RESULT_MISSING",
                error_message=(
                    "ComfyUI reported completion without a non-empty configured "
                    "output reference; empty outputs are not success."
                ),
                updated_at=ms_to_iso(now_ms()),
            )
        return RemoteStatus(
            prompt_id=prompt_id,
            correlation_key="",
            state="running",
            progress=10,
            updated_at=ms_to_iso(now_ms()),
        )

    # ---------------------------------------------------------------- upload
    def _upload_input(self, deadline_ms: int | None = None) -> str:
        self._ensure_job_deadline(deadline_ms, "before input upload")
        if self._uploaded_image_name is not None:
            self._ensure_job_deadline(
                deadline_ms, "before reusing the uploaded input image"
            )
            return self._uploaded_image_name
        path = self.config.input_image_path
        if path is None:
            raise RemoteProtocolError(
                "P1 input_image_path is not configured", code="P1_INPUT_IMAGE_MISSING"
            )
        if not path.is_file():
            raise RemoteProtocolError(
                f"P1 input image is missing: {path.name}", code="P1_INPUT_IMAGE_MISSING"
            )
        if path.suffix.lower() not in self.config.input_image_suffixes:
            raise RemoteProtocolError(
                "P1 input image suffix is not allowed", code="P1_INPUT_IMAGE_INVALID"
            )
        remaining = self._remaining_job_seconds(deadline_ms)
        upload_budget = self.config.upload_timeout_seconds
        job_limited = False
        if remaining is not None:
            job_limited = remaining <= upload_budget
            upload_budget = min(upload_budget, remaining)
        if upload_budget <= 0:
            raise RemoteDeadlineExpired(
                "Job deadline expired before input upload; no /prompt was sent.",
                code="DEADLINE_EXPIRED_BEFORE_SEND",
            )
        deadline = time.monotonic() + upload_budget
        try:
            # Bounded read: never call read_bytes() on an untrusted/racing file.
            with open(path, "rb") as handle:
                content = handle.read(self.config.max_upload_bytes + 1)
                extra = handle.read(1)
        except OSError as exc:
            raise RemoteProtocolError(f"Cannot read input image: {exc}") from exc
        if not content or len(content) > self.config.max_upload_bytes or extra:
            raise RemoteProtocolError(
                "P1 input image is empty or exceeds max_upload_bytes",
                code="P1_INPUT_IMAGE_INVALID",
            )
        if time.monotonic() > deadline:
            if job_limited or (
                deadline_ms is not None
                and int(self._wall_clock()) >= int(deadline_ms)
            ):
                raise RemoteDeadlineExpired(
                    "Job deadline expired while reading the input image; "
                    "no /prompt was sent.",
                    code="DEADLINE_EXPIRED_BEFORE_SEND",
                )
            raise RemoteTransientError(
                "Input image read exceeded the upload total time limit",
                code="REMOTE_TRANSIENT_ERROR",
            )
        self._ensure_job_deadline(deadline_ms, "after reading the input image")
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")
        try:
            name = asyncio.run(
                self._upload_input_async(
                    path.name, content, media_type, deadline, deadline_ms
                )
            )
        except _JobDeadlineBeforeRequest as exc:
            raise RemoteDeadlineExpired(
                "Job deadline expired before the input upload request; "
                "no /prompt was sent.",
                code="DEADLINE_EXPIRED_BEFORE_SEND",
            ) from exc
        except _DeadlineBeforeRequest as exc:
            if job_limited or (
                deadline_ms is not None
                and int(self._wall_clock()) >= int(deadline_ms)
            ):
                raise RemoteDeadlineExpired(
                    "Job deadline expired before the input upload request; "
                    "no /prompt was sent.",
                    code="DEADLINE_EXPIRED_BEFORE_SEND",
                ) from exc
            raise RemoteTransientError(
                "Upload request budget expired before send; no prompt was sent",
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except asyncio.TimeoutError as exc:
            if job_limited or (
                deadline_ms is not None
                and int(self._wall_clock()) >= int(deadline_ms)
            ):
                raise RemoteDeadlineExpired(
                    "Job deadline expired during input upload; no /prompt was sent.",
                    code="DEADLINE_EXPIRED_BEFORE_SEND",
                ) from exc
            raise RemoteTransientError(
                "Upload exceeded the total time limit; no prompt was sent",
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RemoteTransientError(
                self._message(f"upload connect failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            raise RemoteTransientError(
                self._message(f"upload read/write failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except httpx.HTTPError as exc:
            raise RemoteTransientError(
                self._message(f"upload HTTP failure: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        self._ensure_job_deadline(deadline_ms, "after input upload")
        return name

    async def _upload_input_async(
        self,
        filename: str,
        content: bytes,
        media_type: str,
        deadline: float,
        job_deadline_ms: int | None = None,
    ) -> str:
        try:
            status, _headers, body = await self._request_bytes(
                "POST",
                "/upload/image",
                deadline=deadline,
                action="upload",
                files={"image": (filename, content, media_type)},
                data={"overwrite": "false", "type": "input"},
                job_deadline_ms=job_deadline_ms,
            )
        except _ControlTooLarge as exc:
            raise RemoteTransientError(
                f"Upload response exceeded the control limit before /prompt: {exc}",
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RemoteTransientError(
                self._message(f"upload connect failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            raise RemoteTransientError(
                self._message(f"upload read/write failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except httpx.HTTPError as exc:
            raise RemoteTransientError(
                self._message(f"upload HTTP failure: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc

        if status in {401, 403}:
            raise RemoteAuthError("ComfyUI upload endpoint rejected authentication")
        if status >= 500:
            raise RemoteTransientError(
                f"ComfyUI upload returned HTTP {status}; no prompt was sent"
            )
        if status >= 400:
            raise RemoteValidationRejected(
                self._message(f"ComfyUI upload rejected with HTTP {status}"),
                code="INPUT_UPLOAD_REJECTED",
            )
        data = self._parse_json_body(body, "upload")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise RemoteProtocolError("ComfyUI upload response is missing name")
        try:
            safe_name = safe_remote_filename(name)
        except Exception as exc:
            raise RemoteProtocolError(
                f"ComfyUI returned an unsafe upload name: {exc}"
            ) from exc
        self._uploaded_image_name = safe_name
        return safe_name

    # ---------------------------------------------------------------- submit
    def submit(
        self,
        correlation_key: str,
        payload: dict[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        if not isinstance(payload, dict):
            raise RemoteProtocolError("Job payload must be an object")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RemoteValidationRejected("Prompt must be a non-empty string")
        seed = payload.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RemoteValidationRejected("Seed must be an integer")
        if TEMPLATE_MARKER in json.dumps(self.workflow, ensure_ascii=False):
            raise RemoteProtocolError(
                "Configured workflow is TEMPLATE_NOT_EXECUTABLE",
                code="WORKFLOW_TEMPLATE_NOT_EXECUTABLE",
            )
        self._ensure_job_deadline(deadline_ms, "before input upload")
        image_name = self._upload_input(deadline_ms)
        self._ensure_job_deadline(deadline_ms, "after input upload")
        try:
            bound = self.binding.bind(
                self.workflow,
                prompt=prompt,
                seed=seed,
                image_name=image_name,
                width=self.config.width,
                height=self.config.height,
                frames=self.config.frames,
                fps=self.config.fps,
            )
        except Exception as exc:
            raise RemoteValidationRejected(
                self._message(f"Workflow binding failed: {exc}")
            ) from exc
        self._ensure_job_deadline(deadline_ms, "after workflow binding")

        body = {
            "prompt": bound,
            "client_id": self.config.client_id,
            "extra_data": {"correlation_key": correlation_key},
        }
        token = self._active_deadline_ms.set(deadline_ms)
        try:
            return asyncio.run(self._submit_prompt_async(body, correlation_key))
        finally:
            self._active_deadline_ms.reset(token)

    async def _submit_prompt_async(
        self,
        body: dict[str, Any],
        correlation_key: str,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        if deadline_ms is None:
            deadline_ms = self._active_deadline_ms.get()
        # Re-check immediately before the real request, inside the event loop.
        self._ensure_job_deadline(deadline_ms, "before the /prompt request")
        request_budget = self.config.read_timeout_seconds
        remaining = self._remaining_job_seconds(deadline_ms)
        job_limited = remaining is not None and remaining <= request_budget
        if remaining is not None:
            request_budget = min(request_budget, remaining)
        if request_budget <= 0:
            raise RemoteDeadlineExpired(
                "Job deadline expired before the /prompt request; "
                "no /prompt was sent.",
                code="DEADLINE_EXPIRED_BEFORE_SEND",
            )
        deadline = time.monotonic() + request_budget
        try:
            status, _headers, response_body = await self._request_bytes(
                "POST",
                "/prompt",
                deadline=deadline,
                action="submit",
                json_body=body,
                job_deadline_ms=deadline_ms,
            )
        except _JobDeadlineBeforeRequest as exc:
            # The persisted job deadline expired during request preparation,
            # before client.send; this is confirmed-not-sent.
            raise RemoteDeadlineExpired(
                "Job deadline expired before the /prompt request was sent.",
                code="DEADLINE_EXPIRED_BEFORE_SEND",
            ) from exc
        except _DeadlineBeforeRequest as exc:
            # A plain HTTP read/write budget expired before send. It is not a
            # persisted job timeout, so preserve conservative ambiguity.
            if job_limited or (
                deadline_ms is not None
                and int(self._wall_clock()) >= int(deadline_ms)
            ):
                raise RemoteDeadlineExpired(
                    "Job deadline expired before the /prompt request was sent.",
                    code="DEADLINE_EXPIRED_BEFORE_SEND",
                ) from exc
            raise RemoteAmbiguousSubmit(
                "Submit request budget expired before send; acceptance is unknown",
                code="AMBIGUOUS_DISPATCH",
            ) from exc
        except _ControlTooLarge as exc:
            raise RemoteAmbiguousSubmit(
                f"Submit response exceeded the control limit after send: {exc}",
                code="AMBIGUOUS_DISPATCH",
            ) from exc
        except asyncio.TimeoutError as exc:
            raise RemoteAmbiguousSubmit(
                "Submit control response exceeded the total deadline; acceptance is unknown",
                code="AMBIGUOUS_DISPATCH",
            ) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RemoteTransientError(
                self._message(f"submit connect failed: {type(exc).__name__}"),
                code="REMOTE_TRANSIENT_ERROR",
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
            raise RemoteAmbiguousSubmit(
                self._message(f"submit outcome is ambiguous: {type(exc).__name__}"),
                code="AMBIGUOUS_DISPATCH",
            ) from exc

        if status in {401, 403}:
            raise RemoteAuthError("ComfyUI submit endpoint rejected authentication")
        if 300 <= status < 400:
            raise RemoteAmbiguousSubmit(
                "ComfyUI submit returned a redirect; redirects are forbidden",
                code="AMBIGUOUS_DISPATCH",
            )
        if status >= 500:
            raise RemoteAmbiguousSubmit(
                f"ComfyUI submit returned HTTP {status}; acceptance is unknown",
                code="AMBIGUOUS_DISPATCH",
            )
        if status in {400, 422}:
            detail = ""
            try:
                error_data = self._parse_json_body(response_body, "submit-error")
                detail = str(
                    error_data.get("error")
                    or error_data.get("message")
                    or ""
                )
            except RemoteProtocolError:
                pass
            raise RemoteValidationRejected(
                self._message(
                    detail or f"ComfyUI rejected the prompt with HTTP {status}"
                ),
                code="REMOTE_VALIDATION_REJECTED",
            )
        if status != 200:
            raise RemoteProtocolError(
                f"ComfyUI submit returned unexpected HTTP {status}"
            )
        try:
            data = self._parse_json_body(response_body, "submit")
        except RemoteProtocolError as exc:
            raise RemoteAmbiguousSubmit(
                "ComfyUI returned HTTP 200 with malformed/oversized JSON; "
                "acceptance is unknown",
                code="AMBIGUOUS_DISPATCH",
            ) from exc
        node_errors = data.get("node_errors")
        prompt_id = data.get("prompt_id")
        # A prompt_id is authoritative identity even when node_errors is present:
        # ComfyUI can partially accept/queue a prompt while reporting diagnostics.
        # Never mark a valid accepted prompt as accepted=False; keep diagnostics.
        if isinstance(prompt_id, str) and prompt_id:
            diagnostics: dict[str, Any] | None = None
            if node_errors:
                diagnostics = sanitize_diagnostics(
                    {
                        "node_errors": node_errors,
                        "partial_acceptance": True,
                        "note": (
                            "ComfyUI returned a valid prompt_id together with "
                            "node_errors; identity is preserved and diagnostics "
                            "are redacted/bounded before persistence."
                        ),
                    },
                    self._token,
                )
                if not isinstance(diagnostics, dict) or not diagnostics.get(
                    "partial_acceptance"
                ):
                    diagnostics = {
                        "partial_acceptance": True,
                        "node_errors": "[TRUNCATED_OR_REDACTED]",
                    }
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key=correlation_key,
                state="queued",
                progress=0,
                diagnostics=diagnostics,
                updated_at=ms_to_iso(now_ms()),
            )
        if node_errors:
            raise RemoteValidationRejected(
                self._message(
                    json.dumps(
                        sanitize_diagnostics(node_errors, self._token),
                        ensure_ascii=False,
                    )
                )
            )
        raise RemoteAmbiguousSubmit(
            "ComfyUI returned no prompt_id; acceptance is unknown",
            code="AMBIGUOUS_DISPATCH",
        )

    # ------------------------------------------------------------------- get
    def get(self, prompt_id: str) -> RemoteStatus | None:
        entry = self._history(prompt_id)
        if entry is not None:
            # Any history entry confirms presence; a later new absence must get
            # its own full grace interval.
            self._mark_present(prompt_id)
            return self._status_from_entry(prompt_id, entry)
        queue_state = self._queue_state(prompt_id)
        if queue_state == "queued":
            self._mark_present(prompt_id)
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key="",
                state="queued",
                progress=0,
                updated_at=ms_to_iso(now_ms()),
            )
        if queue_state == "running":
            self._mark_present(prompt_id)
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key="",
                state="running",
                progress=10,
                updated_at=ms_to_iso(now_ms()),
            )
        first = self._prompt_first_missing.setdefault(prompt_id, self._clock())
        if self._clock() - first < self.prompt_lost_grace_seconds:
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key="",
                state="queued",
                progress=0,
                updated_at=ms_to_iso(now_ms()),
            )
        raise RemotePromptLost(
            "Prompt is absent from both ComfyUI queue and history after the grace "
            "period; this is a lost prompt, not a confirmed success or failure."
        )

    # ---------------------------------------------------------------- cancel
    def _post_queue_delete(self, prompt_id: str) -> None:
        asyncio.run(
            self._post_control_async(
                "/queue",
                action="queue delete",
                json_body={"delete": [prompt_id]},
            )
        )

    def _post_interrupt(self) -> None:
        asyncio.run(self._post_control_async("/interrupt", action="interrupt"))

    async def _post_control_async(
        self,
        path: str,
        *,
        action: str,
        json_body: Any = None,
    ) -> None:
        deadline = time.monotonic() + self.config.read_timeout_seconds
        try:
            status, _headers, _body = await self._request_bytes(
                "POST",
                path,
                deadline=deadline,
                action=action,
                json_body=json_body,
            )
        except _ControlTooLarge as exc:
            raise RemoteProtocolError(str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise RemoteTransientError(
                self._message(f"{action} exceeded the total deadline")
            ) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RemoteTransientError(
                self._message(f"{action} connect failed: {type(exc).__name__}")
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            raise RemoteTransientError(
                self._message(f"{action} read failed: {type(exc).__name__}")
            ) from exc
        if status in {401, 403}:
            raise RemoteAuthError(f"ComfyUI {action} rejected authentication")
        if status >= 500:
            raise RemoteTransientError(f"ComfyUI {action} returned HTTP {status}")
        if status not in {200, 204}:
            raise RemoteProtocolError(f"{action}: unexpected HTTP {status}")

    def cancel(self, prompt_id: str) -> RemoteStatus | None:
        queue_state = self._queue_state(prompt_id)
        if queue_state == "queued":
            self._post_queue_delete(prompt_id)
            post_state = self._queue_state(prompt_id)
            if post_state == "running":
                raise RemoteCancelUnsupported(
                    "Pending prompt started running after the delete; refusing a "
                    "global interrupt and not claiming cancellation."
                )
            if post_state is not None:
                raise RemoteCancelUnsupported(
                    "Pending cancellation could not be confirmed in the ComfyUI queue"
                )
            # A queued -> completed race may have produced history after the
            # delete. Never claim cancelled when history shows success/error.
            entry = self._history(prompt_id)
            if entry is not None:
                status = self._status_from_entry(prompt_id, entry)
                if status.state in {"succeeded", "failed", "cancelled"}:
                    return status
                raise RemoteCancelUnsupported(
                    "Pending prompt disappeared from queue but history is not "
                    "terminal; cancellation is not confirmed."
                )
            return RemoteStatus(
                prompt_id=prompt_id,
                correlation_key="",
                state="cancelled",
                progress=0,
                updated_at=ms_to_iso(now_ms()),
            )
        if queue_state == "running":
            if not self.config.exclusive_worker:
                raise RemoteCancelUnsupported(
                    "Cancel of a running job in shared ComfyUI mode would use the "
                    "global /interrupt and could cancel another job; refusing."
                )
            running = [pid for state, pid in self._queue() if state == "running"]
            if running != [prompt_id]:
                raise RemoteCancelUnsupported(
                    "Exclusive interrupt refused: the only active prompt does not "
                    "belong to this job (race limitation documented)."
                )
            self._post_interrupt()
            entry = self._history(prompt_id)
            if entry is not None:
                status = self._status_from_entry(prompt_id, entry)
                if status.state == "cancelled":
                    return status
                if status.state in {"succeeded", "failed"}:
                    return status
            raise RemoteCancelUnsupported(
                "ComfyUI interrupt was issued but cancellation was not confirmed "
                "by queue/history; job must not be marked cancelled."
            )

        entry = self._history(prompt_id)
        if entry is not None:
            return self._status_from_entry(prompt_id, entry)
        raise RemotePromptLost(
            "Prompt is absent from queue and history; cancellation/absence is not "
            "confirmed and no remote-absent claim is made."
        )

    def reconcile(self, correlation_key: str) -> RemoteStatus | None:
        # ComfyUI does not expose a durable correlation-key lookup. Returning
        # None is deliberate; callers must preserve ambiguity, never resubmit.
        return None


def build_transport_from_config(
    config_path: Path | str,
    *,
    token: str | None = None,
    async_transport: httpx.AsyncBaseTransport | None = None,
) -> ComfyHttpTransport:
    """Load config/workflow and construct the real transport.

    This performs no network request. Authorization and readiness must be
    checked by the caller before calling this helper in a live context.
    """

    from .p1_config import load_config

    config = load_config(config_path)
    workflow, _ = load_workflow(config.workflow_path)
    return ComfyHttpTransport(
        config, workflow, token=token, async_transport=async_transport
    )


__all__ = [
    "ComfyHttpTransport",
    "build_transport_from_config",
    "redact_error",
]
