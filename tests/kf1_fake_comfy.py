from __future__ import annotations

import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from PIL import Image
import io


def _png(width: int = 704, height: int = 1280, color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _generic_object_info(*, load_image_files=None) -> dict:
    from tests.kf1_object_info import build_object_info

    return build_object_info(load_image_files=load_image_files or [])

class FakeComfy:
    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode
        self.uploads: dict[str, bytes] = {}
        self.prompts: dict[str, dict[str, Any]] = {}
        self.history: dict[str, dict[str, Any]] = {}
        self.requests: list[tuple[str, str]] = []
        self.free_count = 0
        self.queue_running: list[Any] = []
        self.queue_pending: list[Any] = []
        self.output_bytes = _png()
        self.output_map: dict[str, bytes] = {}
        self.fixed_output = False
        self.object_info = _generic_object_info()
        self.list_uploaded_images = True
        self.uploaded_image_names: list[str] = []
        self.prompt_count = 0
        self.upload_count = 0
        self.view_count = 0
        self.view_redirect = False
        self.history_failures = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeComfy":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return None

            def _send(self, status: int, payload: Any = None, *, raw: bytes | None = None) -> None:
                body = raw if raw is not None else json.dumps(payload or {}).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json" if raw is None else "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> bytes:
                length = int(self.headers.get("Content-Length", "0") or 0)
                return self.rfile.read(length)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                outer.requests.append(("GET", parsed.path))
                if parsed.path == "/object_info":
                    self._send(200, outer.object_info)
                    return
                if parsed.path == "/queue":
                    self._send(
                        200,
                        {
                            "queue_running": outer.queue_running,
                            "queue_pending": outer.queue_pending,
                        },
                    )
                    return
                if parsed.path == "/history":
                    self._send(200, outer.history)
                    return
                if parsed.path.startswith("/history/"):
                    prompt_id = parsed.path.rsplit("/", 1)[-1]
                    if outer.history_failures > 0:
                        outer.history_failures -= 1
                        self._send(500, {"error": "transient"})
                        return
                    entry = outer.history.get(prompt_id)
                    if entry is None:
                        self._send(404, {})
                    else:
                        self._send(200, {prompt_id: entry})
                    return
                if parsed.path == "/view":
                    outer.view_count += 1
                    if outer.view_redirect:
                        self.send_response(302)
                        self.send_header("Location", "http://127.0.0.1/foreign.png")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    filename = parse_qs(parsed.query).get("filename", [""])[0]
                    self._send(200, raw=outer.output_map.get(filename, outer.output_bytes))
                    return
                self._send(404, {})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                outer.requests.append(("POST", parsed.path))
                body = self._body()
                if parsed.path == "/upload/image":
                    match = re.search(rb'filename="([^"]+)"', body)
                    name = match.group(1).decode("utf-8") if match else f"upload-{uuid.uuid4().hex}.png"
                    start = body.find(b"\r\n\r\n", body.find(name.encode("utf-8")))
                    content = b""
                    if start >= 0:
                        start += 4
                        end = body.find(b"\r\n--", start)
                        content = body[start:end if end >= 0 else len(body)]
                    outer.uploads[name] = content
                    outer.upload_count += 1
                    if outer.list_uploaded_images:
                        outer.uploaded_image_names.append(name)
                        load_image = outer.object_info.get("LoadImage")
                        if isinstance(load_image, dict):
                            required = load_image.get("input", {}).get("required", {})
                            if "image" in required:
                                required["image"] = [
                                    sorted(set(outer.uploaded_image_names)),
                                    {"image_upload": True},
                                ]
                    self._send(
                        200,
                        {"name": name, "subfolder": "", "type": "input"},
                    )
                    return
                if parsed.path == "/prompt":
                    try:
                        payload = json.loads(body.decode("utf-8"))
                    except Exception:
                        self._send(400, {"error": "bad json"})
                        return
                    if outer.mode == "reject":
                        self._send(422, {"error": "validation rejected"})
                        return
                    outer.prompt_count += 1
                    prompt_id = f"prompt-{outer.prompt_count}"
                    outer.prompts[prompt_id] = payload
                    prompt = payload.get("prompt") or {}
                    save = prompt.get("70") if isinstance(prompt, Mapping) else None
                    prefix = "anime-studio/kf1/test/frame/v0001"
                    if isinstance(save, Mapping):
                        prefix = str(save.get("inputs", {}).get("filename_prefix") or prefix)
                    subfolder, _, base = prefix.rpartition("/")
                    output = {
                        "filename": f"{base}_00001_.png",
                        "subfolder": subfolder,
                        "type": "output",
                    }
                    extra = payload.get("extra_data", {})
                    if not outer.fixed_output:
                        scale = prompt.get("10") if isinstance(prompt, Mapping) else None
                        width, height = 704, 1280
                        if isinstance(scale, Mapping):
                            inputs = scale.get("inputs", {})
                            width = int(inputs.get("width", width) or width)
                            height = int(inputs.get("height", height) or height)
                        outer.output_map[output["filename"]] = _png(width, height)
                    if outer.mode == "terminal_failure":
                        entry = {
                            "prompt": [0, prompt_id, prompt, extra, []],
                            "status": {
                                "status_str": "error",
                                "completed": False,
                                "messages": [["execution_error", {"exception_message": "fake failure"}]],
                            },
                            "outputs": {},
                        }
                    else:
                        entry = {
                            "prompt": [0, prompt_id, prompt, extra, []],
                            "status": {"status_str": "success", "completed": True, "messages": []},
                            "outputs": {"70": {"images": [output]}},
                        }
                    if outer.mode != "ambiguous_unknown":
                        outer.history[prompt_id] = entry
                    if outer.mode in {"ambiguous", "ambiguous_unknown"}:
                        self._send(500, {"error": "accepted but response failed"})
                        return
                    self._send(200, {"prompt_id": prompt_id, "node_errors": {}})
                    return
                if parsed.path == "/free":
                    outer.free_count += 1
                    self._send(200, {})
                    return
                self._send(404, {})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

