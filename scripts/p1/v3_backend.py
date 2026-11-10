"""Small backend process/client composition for P1 v3 run-once.

The process is the existing ``python -m anime_studio serve`` application. The
client talks only to its loopback HTTP API; it never POSTs directly to ComfyUI
and never creates a second queue.

The helpers here are deliberately fail-closed:

* ``RunDeadline`` requires a complete ``v3_lifecycle.SessionBinding`` and clips
  every HTTP timeout to the remaining absolute session budget. A missing or
  partial binding is rejected instead of silently falling back to a fresh
  timeout.
* Backend endpoints are loopback HTTP only, with no userinfo, query or path.
* Requests use ``http.client`` directly, which never follows redirects or uses
  environment proxies.
* A per-request watchdog closes the socket at the absolute request deadline;
  response reads also check the monotonic deadline after every chunk. A late
  response is never returned as success.
* Request and response sizes are bounded, and idempotency keys are never
  retried after an ambiguous POST.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts.p1.v3_lifecycle import SessionBinding

DEFAULT_START_TIMEOUT = 30.0
MAX_RESPONSE_BYTES = 256 * 1024
MAX_PROMPT_CHARS = 4000
MAX_PROMPT_BYTES = 16 * 1024
MAX_TIMEOUT_SECONDS = 3600
MIN_IDEMPOTENCY_KEY = 8
MAX_IDEMPOTENCY_KEY = 128
EXPORT_RESERVE_SECONDS = 120
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class BackendError(Exception):
    """Backend composition failed closed."""


class BackendAmbiguous(BackendError):
    """POST /jobs outcome is unknown; callers must not repeat it."""


class BackendDeadlineExpired(BackendError):
    """The absolute session deadline expired; no request was started."""


@dataclass(frozen=True)
class RunDeadline:
    """Absolute run budget derived from the complete immutable session binding.

    ``planned_cancel_at`` reserves time for export/stop before the hard end.
    The effective deadline is the earlier of both limits. It can only shrink as
    wall-clock time passes; there is no extension method.
    """

    binding: SessionBinding
    export_reserve_seconds: int = EXPORT_RESERVE_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.binding, SessionBinding):
            raise BackendError(
                "run deadline requires a complete v3_lifecycle.SessionBinding"
            )
        if isinstance(self.export_reserve_seconds, bool) or not isinstance(
            self.export_reserve_seconds, int
        ):
            raise BackendError("export reserve must be a non-negative integer")
        if self.export_reserve_seconds < 0:
            raise BackendError("export reserve must be a non-negative integer")
        if self.binding.hard_end_at <= self.binding.created_at:
            raise BackendError("session hard end is not after creation")
        if self.binding.planned_cancel_at < self.binding.created_at:
            raise BackendError("planned cancel precedes session creation")
        if self.binding.planned_cancel_at > self.binding.hard_end_at:
            raise BackendError("planned cancel exceeds the session hard end")

    @property
    def effective_deadline_at(self) -> int:
        return min(
            self.binding.hard_end_at,
            self.binding.planned_cancel_at - self.export_reserve_seconds,
        )

    def remaining(self, *, now: float | None = None) -> float:
        current = time.time() if now is None else float(now)
        return float(self.effective_deadline_at) - current

    def require_remaining(self, *, now: float | None = None) -> float:
        remaining = self.remaining(now=now)
        if remaining <= 0:
            raise BackendDeadlineExpired(
                "absolute session deadline expired; no request was started"
            )
        return remaining

    def request_timeout(
        self, requested: float, *, now: float | None = None
    ) -> float:
        if isinstance(requested, bool) or not isinstance(requested, (int, float)):
            raise BackendError("requested timeout must be a number")
        if not math.isfinite(float(requested)) or float(requested) <= 0:
            raise BackendError("requested timeout must be finite and positive")
        remaining = self.require_remaining(now=now)
        return max(0.001, min(float(requested), remaining))

    def job_timeout_seconds(
        self, *, now: float | None = None, max_seconds: int = 1800
    ) -> int:
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, int):
            raise BackendError("max timeout seconds must be an integer")
        if max_seconds < 1:
            raise BackendError("max timeout seconds must be positive")
        remaining = self.require_remaining(now=now)
        # Keep five seconds for recording the response and reaching a terminal
        # local state. Never create a job timeout that extends past the session.
        seconds = int(math.floor(remaining)) - 5
        if seconds < 1:
            raise BackendDeadlineExpired(
                "not enough session deadline remains to submit one job"
            )
        return min(max_seconds, seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.binding.to_dict(),
            "export_reserve_seconds": self.export_reserve_seconds,
            "effective_deadline_at": self.effective_deadline_at,
        }


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _normalize_backend_url(value: str) -> str:
    """Return a loopback HTTP origin with an explicit port, or fail closed."""

    if not isinstance(value, str) or not value.strip():
        raise BackendError("backend URL is missing")
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError as exc:
        raise BackendError("backend URL is malformed") from exc
    if parts.scheme.lower() != "http":
        raise BackendError("backend URL must use plain loopback http")
    if parts.username or parts.password:
        raise BackendError("backend URL must not contain userinfo")
    if parts.query or parts.fragment:
        raise BackendError("backend URL must not contain query/fragment")
    if parts.path not in ("", "/"):
        raise BackendError("backend URL must not contain a path")
    host = parts.hostname or ""
    if not _is_loopback(host):
        raise BackendError("backend URL must be loopback only")
    try:
        port = parts.port
    except ValueError as exc:
        raise BackendError("backend URL has an invalid port") from exc
    if port is None or not 1 <= int(port) <= 65535:
        raise BackendError("backend URL must contain an explicit valid port")
    return f"http://{parts.netloc}"


def _validate_http_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendError("HTTP URL is missing")
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError as exc:
        raise BackendError("HTTP URL is malformed") from exc
    if parts.scheme.lower() not in {"http", "https"}:
        raise BackendError("HTTP URL must use http or https")
    if not parts.hostname:
        raise BackendError("HTTP URL must include a host")
    if parts.username or parts.password:
        raise BackendError("HTTP URL must not contain userinfo")
    if parts.query or parts.fragment:
        raise BackendError("HTTP URL must not contain query/fragment")
    return value.rstrip("/")


def _coerce_deadline(value: Any) -> RunDeadline | None:
    if value is None:
        return None
    if not isinstance(value, RunDeadline):
        raise BackendError(
            "deadline must be a RunDeadline built from a complete SessionBinding"
        )
    return value


def _validate_number(value: Any, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise BackendError(f"{name} must be finite and positive")
    if maximum is not None and number > maximum:
        raise BackendError(f"{name} exceeds the allowed maximum")
    return number


def _validate_job_id(job_id: Any) -> str:
    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        raise BackendError("job id is missing or contains unsafe characters")
    return job_id


def _validate_path(path: Any, name: str) -> str:
    text = str(path)
    if not text.startswith("/") or ".." in text or "\n" in text or "\r" in text:
        raise BackendError(f"{name} is not a safe absolute request path")
    return text


class _DeadlineAbort(Exception):
    """The independent socket watchdog closed the connection."""


class _SocketWatchdog:
    """Close one in-flight socket at an absolute monotonic deadline."""

    __slots__ = ("_deadline", "_sock", "_stop", "_aborted", "_thread")

    def __init__(self, deadline: float, sock: socket.socket) -> None:
        self._deadline = float(deadline)
        self._sock = sock
        self._stop = threading.Event()
        self._aborted = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="v3-backend-http-watchdog",
            daemon=True,
        )

    @property
    def aborted(self) -> bool:
        return self._aborted.is_set()

    def _close_socket(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                self._aborted.set()
                self._close_socket()
                return
            self._stop.wait(min(0.05, remaining))

    def start(self) -> None:
        self._thread.start()

    def stop_and_join(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()


def _http_origin(base_url: str) -> tuple[str, str, int, str]:
    parts = urllib.parse.urlsplit(base_url)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise BackendError("HTTP URL must use http or https")
    host = parts.hostname
    port = parts.port
    if not host:
        raise BackendError("HTTP URL must include a host")
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, host, int(port), parts.path.rstrip("/")


def _read_bounded_http(
    response: http.client.HTTPResponse,
    *,
    max_bytes: int,
    deadline: float,
    watchdog: _SocketWatchdog,
) -> bytes:
    declared = response.getheader("Content-Length")
    if declared is not None:
        try:
            declared_size = int(declared)
        except (TypeError, ValueError) as exc:
            raise BackendError("response has a malformed Content-Length") from exc
        if declared_size < 0 or declared_size > max_bytes:
            raise BackendError("response exceeds the bounded output limit")
    body = bytearray()
    while True:
        if watchdog.aborted or time.monotonic() >= deadline:
            raise _DeadlineAbort()
        chunk = response.read(min(65536, max_bytes + 1 - len(body)))
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > max_bytes:
            raise BackendError("response exceeds the bounded output limit")
        if watchdog.aborted or time.monotonic() >= deadline:
            raise _DeadlineAbort()
    if watchdog.aborted or time.monotonic() >= deadline:
        raise _DeadlineAbort()
    return bytes(body)


def _json_request(
    *,
    base_url: str,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float,
    deadline: RunDeadline | None = None,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> dict[str, Any]:
    """One bounded HTTP JSON request with no redirects, proxies or late success."""

    path = _validate_path(path, "request path")
    run_deadline = _coerce_deadline(deadline)
    deadline_clipped = False
    if run_deadline is not None:
        requested_timeout = _validate_number(timeout, "request timeout")
        remaining_before = run_deadline.require_remaining()
        deadline_clipped = remaining_before <= requested_timeout
        timeout = max(0.001, min(requested_timeout, remaining_before))
    else:
        timeout = _validate_number(timeout, "request timeout")
    absolute_deadline = time.monotonic() + timeout

    data = None
    request_headers = {"Accept": "application/json"}
    if payload is not None:
        encoded = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > max_bytes:
            raise BackendError("request payload exceeds the bounded input limit")
        data = encoded
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update({str(key): str(value) for key, value in headers.items()})

    scheme, host, port, base_path = _http_origin(base_url)
    request_path = _validate_path(
        (base_path + path) if base_path else path, "request path"
    )
    connection_class = (
        http.client.HTTPSConnection
        if scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_class(host, port, timeout=timeout)
    sock: socket.socket | None = None
    watchdog: _SocketWatchdog | None = None
    response: http.client.HTTPResponse | None = None
    sent = False
    cleanup_ok = True
    try:
        if time.monotonic() >= absolute_deadline:
            raise _DeadlineAbort()
        connection.connect()
        sock = connection.sock
        if sock is None:
            raise BackendError("backend HTTP connection has no socket")
        sock.settimeout(max(0.001, absolute_deadline - time.monotonic()))
        if time.monotonic() >= absolute_deadline:
            raise _DeadlineAbort()
        watchdog = _SocketWatchdog(absolute_deadline, sock)
        watchdog.start()
        # From this point a POST may have reached the server. Any failure is
        # ambiguous unless the deadline was already expired before this line.
        sent = True
        connection.request(
            method, request_path, body=data, headers=request_headers
        )
        response = connection.getresponse()
        status = int(response.status)
        try:
            body = _read_bounded_http(
                response,
                max_bytes=max_bytes,
                deadline=absolute_deadline,
                watchdog=watchdog,
            )
        except BackendError as exc:
            if method == "POST":
                raise BackendAmbiguous(str(exc)) from exc
            raise
        if watchdog.aborted or time.monotonic() >= absolute_deadline:
            raise _DeadlineAbort()
    except _DeadlineAbort as exc:
        session_expired = run_deadline is not None and (
            deadline_clipped or run_deadline.remaining() <= 0
        )
        if method == "POST":
            if sent:
                raise BackendAmbiguous(
                    "backend POST deadline expired after send; outcome unknown"
                ) from exc
            raise BackendDeadlineExpired(
                "backend POST deadline expired before send"
            ) from exc
        if session_expired:
            raise BackendDeadlineExpired(
                "backend GET deadline expired while waiting for a response"
            ) from exc
        raise BackendError("backend GET request timed out") from exc
    except (OSError, TimeoutError, socket.timeout, http.client.HTTPException) as exc:
        session_expired = run_deadline is not None and (
            deadline_clipped or run_deadline.remaining() <= 0
        )
        if method == "POST":
            if sent:
                raise BackendAmbiguous(
                    "backend POST failed after send; outcome unknown"
                ) from exc
            if session_expired:
                raise BackendDeadlineExpired(
                    "backend POST deadline expired before send"
                ) from exc
            raise BackendAmbiguous(
                f"backend POST outcome unknown: {type(exc).__name__}"
            ) from exc
        if session_expired:
            raise BackendDeadlineExpired(
                "backend GET deadline expired while waiting for a response"
            ) from exc
        raise BackendError(f"backend GET failed: {type(exc).__name__}") from exc
    finally:
        if watchdog is not None:
            cleanup_ok = watchdog.stop_and_join() and cleanup_ok
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
            if sock.fileno() != -1:
                cleanup_ok = False
        if not cleanup_ok:
            raise BackendError("backend HTTP cleanup was not confirmed")

    if status in _REDIRECT_CODES:
        if method == "POST":
            raise BackendAmbiguous("backend POST redirect is forbidden")
        raise BackendError("backend GET redirect is forbidden")
    if status < 200 or status >= 300:
        if method == "POST":
            raise BackendAmbiguous(f"backend POST failed with HTTP {status}")
        raise BackendError(f"backend GET failed with HTTP {status}")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if method == "POST":
            raise BackendAmbiguous("backend POST response is not JSON") from exc
        raise BackendError("backend response is not JSON") from exc
    if not isinstance(value, dict):
        if method == "POST":
            raise BackendAmbiguous("backend POST response must be an object")
        raise BackendError("backend response must be an object")
    if time.monotonic() > absolute_deadline:
        if method == "POST":
            raise BackendAmbiguous(
                "backend POST response arrived after the deadline"
            )
        session_expired = run_deadline is not None and (
            deadline_clipped or run_deadline.remaining() <= 0
        )
        if session_expired:
            raise BackendDeadlineExpired(
                "backend GET response arrived after the session deadline"
            )
        raise BackendError("backend GET response arrived after the timeout")
    return value


@dataclass(frozen=True)
class BackendProcess:
    argv: tuple[str, ...]
    base_url: str
    log_path: Path
    config_path: Path | None = None
    db_path: Path | None = None
    artifacts_dir: Path | None = None
    run_dir: Path | None = None
    _process: subprocess.Popen | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _normalize_backend_url(self.base_url))
        if "serve" not in self.argv or "anime_studio" not in self.argv:
            raise BackendError("backend argv must invoke the existing serve command")
        object.__setattr__(self, "log_path", Path(self.log_path))

    @classmethod
    def build(
        cls,
        *,
        config_path: Path,
        db_path: Path,
        artifacts_dir: Path,
        log_path: Path,
        port: int | None = None,
        run_dir: Path | None = None,
    ) -> "BackendProcess":
        if port is not None:
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise BackendError("backend port must be an integer in 1..65535")
        if run_dir is not None:
            root = Path(run_dir).resolve()
            for name, candidate in (
                ("db", db_path),
                ("artifacts", artifacts_dir),
                ("log", log_path),
            ):
                resolved = Path(candidate).resolve()
                if resolved != root and root not in resolved.parents:
                    raise BackendError(
                        f"backend {name} path must be inside the owned run directory"
                    )
        if Path(db_path).resolve() == Path(artifacts_dir).resolve():
            raise BackendError("backend db and artifacts paths must be distinct")
        chosen_port = port or _free_port()
        argv = (
            sys.executable,
            "-m",
            "anime_studio",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(chosen_port),
            "--db",
            str(db_path),
            "--artifacts",
            str(artifacts_dir),
            "--adapter",
            "comfy-http",
            "--allow-live-transport",
            "--comfy-config",
            str(config_path),
            "--log-level",
            "warning",
        )
        return cls(
            argv=argv,
            base_url=f"http://127.0.0.1:{chosen_port}",
            log_path=log_path,
            config_path=Path(config_path),
            db_path=Path(db_path),
            artifacts_dir=Path(artifacts_dir),
            run_dir=Path(run_dir) if run_dir is not None else None,
        )

    def start(
        self,
        *,
        timeout: float = DEFAULT_START_TIMEOUT,
        deadline: RunDeadline | None = None,
    ) -> "BackendProcess":
        if self._process is not None:
            raise BackendError("backend process is already started")
        run_deadline = _coerce_deadline(deadline)
        timeout = _validate_number(timeout, "startup timeout")
        if run_deadline is not None:
            timeout = min(timeout, run_deadline.require_remaining())
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log_path.open("ab")
        try:
            process = subprocess.Popen(
                list(self.argv),
                stdout=handle,
                stderr=handle,
                stdin=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parents[2]),
                start_new_session=(os.name != "nt"),
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
        finally:
            handle.close()
        object.__setattr__(self, "_process", process)
        startup_deadline = time.monotonic() + timeout
        while time.monotonic() < startup_deadline:
            if process.poll() is not None:
                if not self.stop():
                    raise BackendError(
                        "backend startup cleanup was not confirmed"
                    )
                raise BackendError(
                    f"backend exited before health check: {process.returncode}"
                )
            if run_deadline is not None and run_deadline.remaining() <= 0:
                self.stop()
                raise BackendDeadlineExpired(
                    "backend startup consumed the absolute session deadline"
                )
            try:
                health = self.health(deadline=run_deadline)
            except BackendDeadlineExpired:
                if not self.stop():
                    raise BackendError(
                        "backend startup cleanup was not confirmed"
                    )
                raise
            except BackendError:
                time.sleep(min(0.1, max(0.01, startup_deadline - time.monotonic())))
                continue
            if health.get("status") == "ok":
                return self
            time.sleep(min(0.1, max(0.01, startup_deadline - time.monotonic())))
        if not self.stop():
            raise BackendError("backend startup cleanup was not confirmed")
        if run_deadline is not None and run_deadline.remaining() <= 0:
            raise BackendDeadlineExpired(
                "backend startup consumed the absolute session deadline"
            )
        raise BackendError("backend health check timed out")

    def health(self, *, deadline: RunDeadline | None = None) -> dict[str, Any]:
        return self._get_json("/health", timeout=2.0, deadline=deadline)

    def _get_json(
        self,
        path: str,
        *,
        timeout: float,
        deadline: RunDeadline | None = None,
    ) -> dict[str, Any]:
        return _json_request(
            base_url=self.base_url,
            method="GET",
            path=path,
            timeout=timeout,
            deadline=deadline,
        )

    def stop(self, *, timeout: float = 5.0) -> bool:
        process = self._process
        if process is None:
            return True
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=timeout)
            except (OSError, subprocess.TimeoutExpired):
                return False
        object.__setattr__(self, "_process", None)
        return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class BackendClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        deadline: RunDeadline | None = None,
    ) -> None:
        self.base_url = _normalize_backend_url(base_url)
        self.timeout = _validate_number(timeout, "client timeout")
        self.deadline = _coerce_deadline(deadline)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        deadline: RunDeadline | None = None,
    ) -> dict[str, Any]:
        run_deadline = self.deadline if deadline is None else _coerce_deadline(deadline)
        return _json_request(
            base_url=self.base_url,
            method=method,
            path=path,
            payload=payload,
            headers=headers,
            timeout=self.timeout,
            deadline=run_deadline,
        )

    def health(self, *, deadline: RunDeadline | None = None) -> dict[str, Any]:
        return self._request("GET", "/health", deadline=deadline)

    def post_job(
        self,
        *,
        prompt: str,
        seed: int,
        timeout_seconds: int,
        idempotency_key: str,
        deadline: RunDeadline | None = None,
    ) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise BackendError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise BackendError("prompt exceeds the bounded input limit")
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise BackendError("prompt exceeds the bounded input limit")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise BackendError("seed must be an integer")
        if not 0 <= seed <= 2_147_483_647:
            raise BackendError("seed is outside the supported range")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
            raise BackendError("timeout_seconds must be an integer")
        if not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise BackendError("timeout_seconds is outside the supported range")
        if not isinstance(idempotency_key, str):
            raise BackendError("idempotency key must be a string")
        if not MIN_IDEMPOTENCY_KEY <= len(idempotency_key) <= MAX_IDEMPOTENCY_KEY:
            raise BackendError("idempotency key length is outside the allowed range")
        if idempotency_key.strip() != idempotency_key or any(
            ord(char) < 32 or ord(char) == 127 for char in idempotency_key
        ):
            raise BackendError("idempotency key contains unsafe characters")

        run_deadline = self.deadline if deadline is None else _coerce_deadline(deadline)
        if run_deadline is not None:
            # Do not silently rewrite a stored timeout: idempotency is defined
            # over the exact payload. Fail closed instead of extending the job
            # past the absolute session deadline.
            allowed = run_deadline.job_timeout_seconds(
                max_seconds=MAX_TIMEOUT_SECONDS
            )
            if timeout_seconds > allowed:
                raise BackendDeadlineExpired(
                    "job timeout would extend past the absolute session deadline"
                )
        payload = {
            "prompt": prompt,
            "seed": seed,
            "timeout_seconds": timeout_seconds,
        }
        return self._request(
            "POST",
            "/jobs",
            payload=payload,
            headers={"Idempotency-Key": idempotency_key},
            deadline=run_deadline,
        )

    def get_job(
        self, job_id: str, *, deadline: RunDeadline | None = None
    ) -> dict[str, Any]:
        job_id = _validate_job_id(job_id)
        return self._request("GET", f"/jobs/{job_id}", deadline=deadline)

    def get_events(
        self,
        job_id: str,
        *,
        after: int = 0,
        deadline: RunDeadline | None = None,
    ) -> dict[str, Any]:
        job_id = _validate_job_id(job_id)
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise BackendError("events cursor must be a non-negative integer")
        return self._request(
            "GET", f"/jobs/{job_id}/events?after={after}", deadline=deadline
        )

    def get_result(
        self, job_id: str, *, deadline: RunDeadline | None = None
    ) -> dict[str, Any]:
        job_id = _validate_job_id(job_id)
        return self._request("GET", f"/jobs/{job_id}/result", deadline=deadline)

    def wait_terminal(
        self,
        job_id: str,
        *,
        deadline: float,
        poll: float = 0.5,
        session_deadline: RunDeadline | None = None,
    ) -> dict[str, Any]:
        job_id = _validate_job_id(job_id)
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
        ):
            raise BackendError("terminal deadline must be a finite number")
        poll = _validate_number(poll, "poll interval")
        run_deadline = (
            self.deadline if session_deadline is None else _coerce_deadline(session_deadline)
        )
        if run_deadline is not None:
            deadline = min(deadline, time.monotonic() + run_deadline.require_remaining())
        terminal = {"succeeded", "failed", "cancelled", "timed_out"}
        while time.monotonic() < deadline:
            payload = self.get_job(job_id, deadline=run_deadline)
            job = payload.get("job")
            if isinstance(job, dict) and job.get("status") in terminal:
                return job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(max(0.05, poll), remaining))
        raise BackendError("job deadline expired before terminal state")


def check_comfy_queue(
    base_url: str,
    token: str | None = None,
    *,
    deadline: RunDeadline | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Read-only preflight: require empty ComfyUI running/pending queues."""

    base_url = _validate_http_url(base_url)
    if token is not None and not isinstance(token, str):
        raise BackendError("ComfyUI token must be a string when provided")
    parts = urllib.parse.urlsplit(base_url)
    host = parts.hostname or ""
    if not _is_loopback(host):
        if parts.scheme.lower() != "https":
            raise BackendError("non-loopback ComfyUI queue requires https")
        if not token:
            raise BackendError(
                "non-loopback ComfyUI queue requires an authenticated token"
            )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        payload = _json_request(
            base_url=base_url,
            method="GET",
            path="/queue",
            headers=headers,
            timeout=timeout,
            deadline=_coerce_deadline(deadline),
        )
    except BackendDeadlineExpired:
        raise
    except BackendError as exc:
        raise BackendError(f"ComfyUI queue preflight failed: {exc}") from exc
    running = payload.get("queue_running")
    pending = payload.get("queue_pending")
    if not isinstance(running, list) or not isinstance(pending, list):
        raise BackendError("ComfyUI queue response is malformed")
    if running or pending:
        raise BackendError("ComfyUI queue is not empty")
    return {"running": 0, "pending": 0}


__all__ = [
    "BackendAmbiguous",
    "BackendClient",
    "BackendDeadlineExpired",
    "BackendError",
    "BackendProcess",
    "DEFAULT_START_TIMEOUT",
    "MAX_RESPONSE_BYTES",
    "RunDeadline",
    "check_comfy_queue",
]
