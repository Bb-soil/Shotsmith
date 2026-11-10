"""Small bounded ComfyUI HTTP client used by KF1.

It intentionally supports only the endpoints needed for keyframes: upload,
prompt submit, queue/history readback, PNG view and the explicit /free request.
It does not import or publish through the legacy P1 live-candidate path.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from anime_studio.workflow import safe_relative_subfolder, safe_remote_filename

from .errors import AmbiguousDispatch, TransientRemoteError
from .normalize import validate_png_bytes

MAX_CONTROL_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_VIEW_BYTES = 100 * 1024 * 1024


class ComfyError(Exception):
    pass


class ValidationRejected(ComfyError):
    pass


class PromptLost(ComfyError):
    pass


class HistoryContractError(ComfyError):
    pass


@dataclass(frozen=True)
class Submission:
    prompt_id: str
    status_code: int


def _loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ComfyError("base_url должен быть http(s) URL")
    if not _loopback(parsed.hostname):
        raise ComfyError("KF1 принимает только loopback ComfyUI (owned tunnel)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ComfyError("base_url не должен содержать userinfo/query/fragment")
    return value.rstrip("/")


class ComfyClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = validate_base_url(base_url)
        self.token = token
        self.timeout_seconds = float(timeout_seconds)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ComfyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: Any = None,
        files: Any = None,
        data: Any = None,
        max_bytes: int = MAX_CONTROL_BYTES,
        timeout: float | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        deadline = time.monotonic() + (timeout or self.timeout_seconds)
        try:
            with self._client.stream(
                method,
                path,
                json=json_body,
                params=params,
                files=files,
                data=data,
                headers=self._headers(),
                timeout=httpx.Timeout(max(0.05, deadline - time.monotonic())),
            ) as response:
                if response.is_redirect or 300 <= response.status_code < 400:
                    raise ComfyError("ComfyUI redirect запрещён")
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        size = int(declared)
                    except ValueError as exc:
                        raise ComfyError("Некорректный Content-Length") from exc
                    if size < 0 or size > max_bytes:
                        raise ComfyError("Ответ ComfyUI превышает лимит")
                body = bytearray()
                for chunk in response.iter_bytes():
                    if time.monotonic() > deadline:
                        raise TimeoutError("ComfyUI response deadline exceeded")
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise ComfyError("Ответ ComfyUI превышает лимит")
                return response.status_code, dict(response.headers), bytes(body)
        except httpx.HTTPError as exc:
            raise TransientRemoteError(f"ComfyUI HTTP error: {type(exc).__name__}") from exc

    def upload_png(
        self, path: Path | str, *, remote_name: str | None = None
    ) -> dict[str, Any]:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ComfyError(f"upload: нет обычного файла {source}")
        data = source.read_bytes()
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise ComfyError("upload: PNG пуст или слишком большой")
        validate_png_bytes(data)
        name = safe_remote_filename(remote_name or source.name)
        status, _headers, body = self._request(
            "POST",
            "/upload/image",
            files={"image": (name, data, "image/png")},
            data={"overwrite": "false", "type": "input"},
            max_bytes=MAX_CONTROL_BYTES,
            timeout=max(self.timeout_seconds, 120.0),
        )
        if status in {401, 403}:
            raise ComfyError("upload: authentication rejected")
        if status >= 400:
            raise ComfyError(f"upload: HTTP {status}")
        payload = _json_object(body, "upload")
        uploaded = payload.get("name")
        if not isinstance(uploaded, str) or not uploaded:
            raise ComfyError("upload: response без name")
        uploaded = safe_remote_filename(uploaded)
        if uploaded != name:
            raise ComfyError("upload: сервер переименовал файл; exact target не подтверждён")
        artifact_type = payload.get("type", "input")
        if artifact_type != "input":
            raise ComfyError("upload: response type должен быть input")
        subfolder = payload.get("subfolder", "")
        if not isinstance(subfolder, str):
            raise ComfyError("upload: response subfolder некорректен")
        subfolder = safe_relative_subfolder(subfolder)
        path_prefix = f"input/{subfolder}/{uploaded}" if subfolder else f"input/{uploaded}"
        return {
            "name": uploaded,
            "requested_name": name,
            "subfolder": subfolder,
            "type": "input",
            "path": path_prefix,
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    def submit(
        self,
        workflow: Mapping[str, Any],
        *,
        client_id: str,
        correlation_key: str,
        nonce: str,
    ) -> Submission:
        body = {
            "prompt": dict(workflow),
            "client_id": client_id,
            "extra_data": {
                "correlation_key": correlation_key,
                "nonce": nonce,
                "source": "anime-video-studio-kf1",
            },
        }
        try:
            status, _headers, response = self._request(
                "POST", "/prompt", json_body=body, max_bytes=MAX_CONTROL_BYTES
            )
        except TimeoutError as exc:
            raise AmbiguousDispatch("POST /prompt таймаут: принятие неизвестно") from exc
        except TransientRemoteError as exc:
            # A connect failure may be confirmed-not-sent, but a read/write
            # failure after send is ambiguous.  The client cannot distinguish
            # reliably; fail closed and let the caller keep AMBIGUOUS.
            raise AmbiguousDispatch(f"POST /prompt transport: {exc}") from exc
        if status in {400, 422}:
            detail = ""
            try:
                payload = _json_object(response, "submit-error")
                detail = str(payload.get("error") or payload.get("message") or "")
            except ComfyError:
                pass
            raise ValidationRejected(detail or f"ComfyUI validation HTTP {status}")
        if status != 200:
            raise AmbiguousDispatch(f"POST /prompt HTTP {status}: принятие неизвестно")
        try:
            payload = _json_object(response, "submit")
        except ComfyError as exc:
            raise AmbiguousDispatch("POST /prompt malformed JSON: принятие неизвестно") from exc
        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise AmbiguousDispatch("POST /prompt без prompt_id: принятие неизвестно")
        return Submission(prompt_id=prompt_id, status_code=200)

    def object_info(self) -> dict[str, Any]:
        status, _headers, body = self._request(
            "GET", "/object_info", max_bytes=16 * 1024 * 1024
        )
        if status != 200:
            raise TransientRemoteError(f"object_info HTTP {status}")
        payload = _json_object(body, "object_info")
        if not isinstance(payload, dict) or not payload:
            raise ComfyError("object_info пуст/не объект")
        return payload
    def history(self, prompt_id: str) -> dict[str, Any] | None:
        if not isinstance(prompt_id, str) or not prompt_id or "/" in prompt_id:
            raise ComfyError("history: некорректный prompt_id")
        status, _headers, body = self._request(
            "GET", f"/history/{prompt_id}", max_bytes=MAX_CONTROL_BYTES
        )
        if status == 404:
            return None
        if status != 200:
            raise TransientRemoteError(f"history HTTP {status}")
        try:
            payload = _json_object(body, "history")
        except ComfyError as exc:
            raise TransientRemoteError("history malformed JSON") from exc
        entry = payload.get(prompt_id)
        if entry is None:
            return None
        if not isinstance(entry, Mapping):
            raise TransientRemoteError("history entry is not an object")
        return dict(entry)

    def history_all(self) -> dict[str, Any]:
        status, _headers, body = self._request(
            "GET", "/history", max_bytes=MAX_CONTROL_BYTES
        )
        if status != 200:
            raise TransientRemoteError(f"history_all HTTP {status}")
        return _json_object(body, "history_all")

    def queue(self) -> dict[str, Any]:
        status, _headers, body = self._request("GET", "/queue", max_bytes=MAX_CONTROL_BYTES)
        if status != 200:
            raise TransientRemoteError(f"queue HTTP {status}")
        payload = _json_object(body, "queue")
        for key in ("queue_running", "queue_pending"):
            if not isinstance(payload.get(key), list):
                raise TransientRemoteError(f"queue: нет {key}")
        return payload

    def queue_ids(self) -> tuple[list[str], list[str]]:
        payload = self.queue()
        running = [_queue_id(item) for item in payload["queue_running"]]
        pending = [_queue_id(item) for item in payload["queue_pending"]]
        return running, pending

    def find_prompt_by_correlation(
        self, correlation_key: str, nonce: str
    ) -> str | None:
        data = self.history_all()
        matches: list[str] = []
        for prompt_id, entry in data.items():
            if not isinstance(prompt_id, str) or not isinstance(entry, Mapping):
                continue
            prompt = entry.get("prompt")
            if not isinstance(prompt, list) or len(prompt) < 4:
                continue
            if prompt[1] != prompt_id:
                continue
            extra = prompt[3]
            if not isinstance(extra, Mapping):
                continue
            if (
                extra.get("correlation_key") != correlation_key
                or extra.get("nonce") != nonce
                or extra.get("source") != "anime-video-studio-kf1"
            ):
                continue
            matches.append(prompt_id)
        if len(matches) > 1:
            raise AmbiguousDispatch("Найдено несколько prompt_id с exact correlation/nonce")
        return matches[0] if matches else None

    def view(self, output: Mapping[str, Any]) -> bytes:
        try:
            filename = safe_remote_filename(output.get("filename", ""))
            subfolder = safe_relative_subfolder(output.get("subfolder", ""))
        except Exception as exc:
            raise ComfyError(f"output path unsafe: {exc}") from exc
        artifact_type = output.get("type", "output")
        if artifact_type != "output":
            raise ComfyError("output type must be output")
        if Path(filename).suffix.lower() != ".png":
            raise ComfyError("SaveImage output must be .png")
        status, _headers, body = self._request(
            "GET",
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": artifact_type},
            max_bytes=MAX_VIEW_BYTES,
            timeout=max(self.timeout_seconds, 120.0),
        )
        if status != 200:
            raise TransientRemoteError(f"view HTTP {status}")
        validate_png_bytes(body, max_bytes=MAX_VIEW_BYTES)
        return body

    def free(self) -> bool:
        status, _headers, _body = self._request(
            "POST",
            "/free",
            json_body={"unload_models": True, "free_memory": True},
            max_bytes=MAX_CONTROL_BYTES,
        )
        if status != 200:
            raise TransientRemoteError(f"/free HTTP {status}")
        return True


def _json_object(body: bytes, action: str) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComfyError(f"{action}: не JSON") from exc
    if not isinstance(payload, dict):
        raise ComfyError(f"{action}: не JSON-объект")
    return payload


def _queue_id(item: Any) -> str:
    if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], str):
        return item[1]
    if isinstance(item, Mapping) and isinstance(item.get("prompt_id"), str):
        return item["prompt_id"]
    raise TransientRemoteError("queue entry has no prompt id")


def history_status(entry: Mapping[str, Any]) -> tuple[str | None, str | None]:
    status = entry.get("status")
    status_str = None
    completed = False
    messages = None
    if isinstance(status, Mapping):
        status_str = status.get("status_str")
        completed = status.get("completed") is True
        messages = status.get("messages")
    error = None
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, list) and len(message) >= 2:
                event, payload = message[0], message[1]
                if event in {"execution_error", "execution_interrupted"}:
                    if isinstance(payload, Mapping):
                        error = payload.get("exception_message") or payload.get("exception_type")
                    if not error:
                        error = str(event)
                    if event == "execution_interrupted":
                        status_str = "cancelled"
                    break
    if error or status_str in {"error", "failed", "cancelled"}:
        return (str(status_str) if status_str else "error", str(error) if error else status_str)
    if completed and status_str == "success":
        return ("success", None)
    return (None, None)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def output_from_history(
    entry: Mapping[str, Any],
    *,
    node_id: str,
    keys: list[str],
    expected_output: Mapping[str, Any] | None = None,
    expected_graph_sha256: str | None = None,
    expected_prompt_id: str | None = None,
    expected_correlation_key: str | None = None,
    expected_nonce: str | None = None,
    expected_source: str | None = None,
) -> dict[str, Any]:
    status, error = history_status(entry)
    if status != "success" or error:
        raise ComfyError("history entry is not a successful terminal result")
    prompt = entry.get("prompt")
    if not isinstance(prompt, list) or len(prompt) < 4:
        raise HistoryContractError("history entry lacks pinned prompt tuple")
    if expected_prompt_id is not None and prompt[1] != expected_prompt_id:
        raise HistoryContractError("history prompt_id tuple mismatch")
    if expected_graph_sha256 is not None:
        if _canonical_sha(prompt[2]) != expected_graph_sha256:
            raise ComfyError("history graph hash differs from the durable attempt")
    if (
        expected_correlation_key is not None
        or expected_nonce is not None
        or expected_source is not None
    ):
        extra = prompt[3]
        if not isinstance(extra, Mapping):
            raise HistoryContractError("history extra_data is missing")
        if (
            expected_correlation_key is not None
            and extra.get("correlation_key") != expected_correlation_key
        ):
            raise HistoryContractError("history correlation_key mismatch")
        if expected_nonce is not None and extra.get("nonce") != expected_nonce:
            raise HistoryContractError("history nonce mismatch")
        if expected_source is not None and extra.get("source") != expected_source:
            raise HistoryContractError("history owner/source mismatch")
    outputs = entry.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ComfyError("history entry без outputs")
    node = outputs.get(str(node_id))
    if not isinstance(node, Mapping):
        raise ComfyError(f"history entry без output node {node_id}")
    candidates = []
    for key in keys:
        values = node.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            filename = value.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            safe = safe_remote_filename(filename)
            if Path(safe).suffix.lower() != ".png":
                continue
            subfolder = value.get("subfolder", "")
            if not isinstance(subfolder, str):
                continue
            safe_subfolder = safe_relative_subfolder(subfolder)
            if value.get("type", "output") != "output":
                continue
            candidates.append(
                {
                    "filename": safe,
                    "subfolder": safe_subfolder,
                    "type": "output",
                    "media_type": "image/png",
                }
            )
    if len(candidates) != 1:
        raise ComfyError("SaveImage должен вернуть ровно один PNG")
    result = candidates[0]
    if expected_output:
        subfolder = expected_output.get("subfolder")
        prefix = expected_output.get("filename_prefix")
        if subfolder is not None and result["subfolder"] != subfolder:
            raise ComfyError("SaveImage subfolder не совпадает с binding")
        if prefix is not None and not result["filename"].startswith(str(prefix) + "_"):
            raise ComfyError("SaveImage filename prefix не совпадает с binding")
    return result


__all__ = [
    "AmbiguousDispatch",
    "ComfyClient",
    "ComfyError",
    "HistoryContractError",
    "PromptLost",
    "Submission",
    "TransientRemoteError",
    "ValidationRejected",
    "history_status",
    "output_from_history",
    "validate_base_url",
]



