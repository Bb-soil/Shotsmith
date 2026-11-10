"""Loopback-only ComfyUI HTTP stub used by offline contract tests.

No test in this repository contacts RunPod, a GPU worker or any external host.
The server binds an ephemeral 127.0.0.1 port and is shut down by the context
manager.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator


def _box(box_type: bytes, payload: bytes = b"") -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + box_type + payload


# Structural only: ftyp + moov + mdat boxes. This is not decoded media and is
# explicitly labelled as such by HttpArtifactStore.
MP4_FIXTURE = (
    _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    + _box(b"moov")
    + _box(b"mdat", b"structural-fixture-not-real-video")
)


@dataclass
class StubState:
    prompt_mode: str = "accepted"
    prompt_id: str = "prompt-1"
    prompt_requests: int = 0
    prompt_delay_seconds: float = 0.0
    node_errors_payload: dict[str, Any] | None = None
    queue_mode: str = "normal"
    queue_after_delete_mode: str | None = None
    queue_delay_seconds: float = 0.0
    oversized_path: str | None = None
    oversized_total: int = 8192
    oversized_written: int = 0
    oversized_content_length: str | None = None
    prompt_bodies: list[dict[str, Any]] = field(default_factory=list)
    upload_requests: int = 0
    upload_status: int = 200
    upload_name: str = "source.png"
    upload_read_delay_seconds: float = 0.0
    upload_response_delay_seconds: float = 0.0
    queue_pending: list[str] = field(default_factory=list)
    queue_running: list[str] = field(default_factory=list)
    history: dict[str, dict[str, Any]] = field(default_factory=dict)
    history_missing: bool = False
    delete_calls: list[list[str]] = field(default_factory=list)
    interrupt_calls: int = 0
    interrupt_cancels: bool = True
    view_status: int = 200
    view_body: bytes = MP4_FIXTURE
    view_content_length: str | None = None
    view_delay_seconds: float = 0.0
    view_chunks: list[bytes] | None = None
    view_chunk_delay_seconds: float = 0.0
    view_redirect: str | None = None
    view_calls: int = 0

    def accept(self) -> None:
        self.prompt_mode = "accepted"
        self.queue_pending = [self.prompt_id]

    def set_success(self, filename: str = "result.mp4") -> None:
        self.history[self.prompt_id] = {
            "status": {
                "status_str": "success",
                "completed": True,
                "messages": [],
            },
            "outputs": {
                "7": {
                    "gifs": [
                        {
                            "filename": filename,
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            },
        }
        self.queue_pending = []
        self.queue_running = []

    def set_error(self, message: str = "simulated execution error") -> None:
        self.history[self.prompt_id] = {
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [["execution_error", {"exception_message": message}]],
            },
            "outputs": {},
        }
        self.queue_pending = []
        self.queue_running = []

    def set_cancelled(self) -> None:
        self.history[self.prompt_id] = {
            "status": {"status_str": "cancelled", "completed": False, "messages": []},
            "outputs": {},
        }
        self.queue_pending = []
        self.queue_running = []


class _Handler(BaseHTTPRequestHandler):
    server_version = "ComfyStub/1"
    protocol_version = "HTTP/1.0"

    @property
    def state(self) -> StubState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length)

    def _oversized(self) -> None:
        state = self.state
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if state.oversized_content_length is not None:
            self.send_header("Content-Length", state.oversized_content_length)
        self.end_headers()
        chunk = b"{" + b"x" * 255
        written = 0
        try:
            while written < state.oversized_total:
                self.wfile.write(chunk)
                self.wfile.flush()
                written += len(chunk)
                state.oversized_written = written
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return
        self.close_connection = True

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        state = self.state
        if state.oversized_path == "upload" and self.path == "/upload/image":
            self._body()
            self._oversized()
            return
        if state.oversized_path == "prompt" and self.path == "/prompt":
            self._body()
            state.prompt_requests += 1
            self._oversized()
            return
        if state.oversized_path == "queue_delete" and self.path == "/queue":
            self._body()
            self._oversized()
            return
        if state.oversized_path == "interrupt" and self.path == "/interrupt":
            state.interrupt_calls += 1
            self._oversized()
            return
        if self.path == "/upload/image":
            if state.upload_read_delay_seconds:
                time.sleep(state.upload_read_delay_seconds)
            body = self._body()
            state.upload_requests += 1
            if state.upload_response_delay_seconds:
                time.sleep(state.upload_response_delay_seconds)
            if state.upload_status != 200:
                self._json(state.upload_status, {"error": "upload rejected"})
                return
            if not body:
                self._json(400, {"error": "empty upload"})
                return
            self._json(
                200,
                {"name": state.upload_name, "subfolder": "", "type": "input"},
            )
            return
        if self.path == "/prompt":
            raw = self._body()
            state.prompt_requests += 1
            try:
                state.prompt_bodies.append(json.loads(raw.decode("utf-8")))
            except ValueError:
                state.prompt_bodies.append({})
            if state.prompt_mode == "validation":
                self._json(
                    400,
                    {
                        "error": {"type": "prompt_outputs_failed_validation"},
                        "node_errors": {"5": {"errors": ["bad input"]}},
                    },
                )
                return
            if state.prompt_mode == "node_errors":
                state.queue_pending = [state.prompt_id]
                self._json(
                    200,
                    {
                        "prompt_id": state.prompt_id,
                        "node_errors": state.node_errors_payload
                        or {"5": {"errors": ["bad input"]}},
                    },
                )
                return
            if state.prompt_mode == "ambiguous_500":
                self._json(500, {"error": "worker overloaded"})
                return
            if state.prompt_mode == "malformed_200":
                data = b"{not-json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if state.prompt_mode == "no_prompt_id":
                self._json(200, {"number": 1, "node_errors": {}})
                return
            if state.prompt_delay_seconds:
                time.sleep(state.prompt_delay_seconds)
            if state.prompt_mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "https://example.invalid/prompt")
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True
                return
            state.queue_pending = [state.prompt_id]
            self._json(
                200,
                {"prompt_id": state.prompt_id, "number": 1, "node_errors": {}},
            )
            return
        if self.path == "/queue":
            raw = self._body()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except ValueError:
                payload = {}
            delete = payload.get("delete", [])
            if isinstance(delete, list):
                state.delete_calls.append([str(item) for item in delete])
                state.queue_pending = [
                    item for item in state.queue_pending if item not in delete
                ]
                if state.queue_after_delete_mode:
                    state.queue_mode = state.queue_after_delete_mode
            self._json(200, {})
            return
        if self.path == "/interrupt":
            state.interrupt_calls += 1
            if state.interrupt_cancels:
                state.set_cancelled()
            self._json(200, {})
            return
        self._json(404, {"error": "not found"})

    def do_GET(self) -> None:  # noqa: N802
        state = self.state
        if state.oversized_path == "queue" and self.path == "/queue":
            self._oversized()
            return
        if self.path == "/queue" and state.queue_delay_seconds:
            time.sleep(state.queue_delay_seconds)
        if state.oversized_path == "history" and self.path.startswith("/history/"):
            self._oversized()
            return
        if self.path == "/queue":
            if state.queue_mode == "missing_keys":
                self._json(200, {})
                return
            if state.queue_mode == "bad_type":
                self._json(200, {"queue_running": {}, "queue_pending": []})
                return
            if state.queue_mode == "bad_entry":
                self._json(
                    200,
                    {"queue_running": [["bad"]], "queue_pending": []},
                )
                return
            if state.queue_mode == "running":
                pending = []
                running = [[0, state.prompt_id, {}, {}, []]]
            else:
                pending = [
                    [index, prompt_id, {}, {}, []]
                    for index, prompt_id in enumerate(state.queue_pending)
                ]
                running = [
                    [index, prompt_id, {}, {}, []]
                    for index, prompt_id in enumerate(state.queue_running)
                ]
            self._json(
                200,
                {
                    "queue_running": running,
                    "queue_pending": pending,
                },
            )
            return
        if self.path.startswith("/history/"):
            prompt_id = self.path.rsplit("/", 1)[-1]
            if state.history_missing:
                self._json(200, {})
                return
            entry = state.history.get(prompt_id)
            self._json(200, {prompt_id: entry} if entry is not None else {})
            return
        if self.path.startswith("/view"):
            state.view_calls += 1
            if state.view_redirect:
                self.send_response(302)
                self.send_header("Location", state.view_redirect)
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True
                return
            if state.view_status != 200:
                self._json(state.view_status, {"error": "view failed"})
                return
            body = state.view_body
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            if state.view_content_length == "OMIT":
                pass
            elif state.view_content_length is not None:
                self.send_header("Content-Length", state.view_content_length)
            else:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state.view_chunks is not None:
                for chunk in state.view_chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if state.view_chunk_delay_seconds:
                        time.sleep(state.view_chunk_delay_seconds)
            elif state.view_delay_seconds:
                split = max(1, len(body) // 2)
                self.wfile.write(body[:split])
                self.wfile.flush()
                time.sleep(state.view_delay_seconds)
                self.wfile.write(body[split:])
            else:
                self.wfile.write(body)
            return
        if self.path == "/system_stats":
            self._json(200, {"system": {"comfyui_version": "stub"}})
            return
        self._json(404, {"error": "not found"})


@contextmanager
def run_stub(state: StubState | None = None) -> Iterator[tuple[str, StubState]]:
    state = state or StubState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


__all__ = ["MP4_FIXTURE", "StubState", "run_stub"]
