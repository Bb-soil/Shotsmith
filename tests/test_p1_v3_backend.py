from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from types import SimpleNamespace

import pytest

from anime_studio.__main__ import _build_parser
from anime_studio.comfy_http import build_transport_from_config
from anime_studio.p1_config import check_readiness
from anime_studio.workflow import workflow_sha256
from scripts.p1 import clore_session as cs
from scripts.p1 import v3_backend, v3_lifecycle, v3_operator
from tests.comfy_stub import run_stub
from tests.helpers import db_row, db_scalar
from tests.p1_helpers import (
    PNG_FIXTURE,
    make_manifest,
    write_config,
    write_manifest,
    write_workflow,
)

ROOT = Path(__file__).resolve().parents[1]


def _binding(
    tmp_path: Path,
    *,
    name: str = "session-state.json",
    created_offset: int = -5,
    planned_offset: int = 600,
    hard_offset: int = 1800,
) -> v3_lifecycle.SessionBinding:
    quote = cs.CloreQuote.from_dict(
        json.loads(
            (ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json").read_text(
                encoding="utf-8"
            )
        )
    )
    now = datetime.now(timezone.utc)
    now_ts = int(now.timestamp())
    state = cs.initialize_state(
        tmp_path / name,
        order_id=97001,
        created_at=now_ts + created_offset,
        quote=quote,
        planned_cancel_at=now_ts + planned_offset,
        hard_end_at=now_ts + hard_offset,
        now=now,
    )
    return v3_lifecycle.SessionBinding.from_state(state)


def _operator_binding(*, planned_cancel_at: int, hard_end_at: int) -> v3_lifecycle.SessionBinding:
    created = min(int(time.time()) - 5, int(planned_cancel_at) - 1)
    return v3_lifecycle.SessionBinding(
        order_id=97001,
        server_id=112213,
        created_at=created,
        currency="USD",
        cap_usd="1.00",
        duration_seconds=3600,
        planned_cancel_at=int(planned_cancel_at),
        hard_end_at=int(hard_end_at),
        quote_hash="a" * 64,
    )


def _verified_config(tmp_path: Path, base_url: str) -> Path:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "manifest.json"
    sha = workflow_sha256(workflow_path)
    write_manifest(manifest_path, make_manifest(workflow_sha256=sha))
    input_path = tmp_path / "input.png"
    input_path.write_bytes(PNG_FIXTURE)
    return write_config(
        tmp_path,
        base_url=base_url,
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        input_image_path=input_path,
        workflow_sha256=sha,
    )


def _process(
    *,
    config_path: Path,
    run_dir: Path,
) -> v3_backend.BackendProcess:
    return v3_backend.BackendProcess.build(
        config_path=config_path,
        db_path=run_dir / "jobs.sqlite3",
        artifacts_dir=run_dir / "artifacts",
        log_path=run_dir / "backend.log",
        run_dir=run_dir,
    )


def _wait_until(predicate, *, timeout: float = 10.0, message: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {message}")


class _PolicyHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self) -> None:
        server = self.server
        server.hits.append(self.path)  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        if body:
            server.bodies.append(body)  # type: ignore[attr-defined]
        mode = server.mode  # type: ignore[attr-defined]
        if mode == "drop":
            self.close_connection = True
            return
        if mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.close_connection = True
            return
        if mode == "slow":
            time.sleep(3.0)
            self._json(200, {"status": "ok", "queue_running": [], "queue_pending": []})
            return
        if mode == "oversized":
            data = b"x" * (v3_backend.MAX_RESPONSE_BYTES + 10)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if mode == "error500":
            self._json(500, {"error": "synthetic failure"})
            return
        self._json(200, {"status": "ok", "queue_running": [], "queue_pending": []})

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()


def _watchdog_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "v3-backend-http-watchdog"
    ]


class _TrickleHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        return

    def _handle(self, *, post: bool) -> None:
        server = self.server
        server.hits.append(self.path)  # type: ignore[attr-defined]
        if post:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length) if length else b""
        status = 201 if post else 200
        payload = (
            b'{"job":{"job_id":"job-trickle","status":"queued"}}'
            if post
            else b'{"status":"ok"}'
        )
        body = payload + b" " * (200 - len(payload))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            for byte in body:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.05)
            server.completed = True  # type: ignore[attr-defined]
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            pass
        finally:
            server.finished.set()  # type: ignore[attr-defined]
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        self._handle(post=False)

    def do_POST(self) -> None:  # noqa: N802
        self._handle(post=True)


@contextmanager
def _trickle_probe() -> Iterator[tuple[str, ThreadingHTTPServer]]:
    """A local server that trickles a response past the client deadline."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), _TrickleHandler)
    server.hits = []  # type: ignore[attr-defined]
    server.completed = False  # type: ignore[attr-defined]
    server.finished = threading.Event()  # type: ignore[attr-defined]
    server.handle_error = lambda *args: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _policy_probe(
    mode: str,
) -> Iterator[tuple[str, ThreadingHTTPServer]]:
    """Small transport-policy probe; it is not the job-flow backend."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), _PolicyHandler)
    server.mode = mode  # type: ignore[attr-defined]
    server.hits = []  # type: ignore[attr-defined]
    server.bodies = []  # type: ignore[attr-defined]
    server.handle_error = lambda *args: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_backend_process_build_matches_existing_serve_parser_and_owned_paths(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    process = _process(config_path=tmp_path / "config.json", run_dir=run_dir)
    assert process.argv[3] == "serve"
    parsed = _build_parser().parse_args(list(process.argv[3:]))
    assert parsed.command == "serve"
    assert parsed.host == "127.0.0.1"
    assert parsed.db == str(run_dir / "jobs.sqlite3")
    assert parsed.artifacts == str(run_dir / "artifacts")
    assert parsed.adapter == "comfy-http"
    assert parsed.allow_live_transport is True
    assert parsed.comfy_config == str(tmp_path / "config.json")
    assert process.base_url.startswith("http://127.0.0.1:")
    assert process.db_path == run_dir / "jobs.sqlite3"
    assert process.artifacts_dir == run_dir / "artifacts"
    assert process.run_dir == run_dir

    with pytest.raises(v3_backend.BackendError):
        v3_backend.BackendProcess.build(
            config_path=tmp_path / "config.json",
            db_path=tmp_path / "outside.sqlite3",
            artifacts_dir=run_dir / "artifacts",
            log_path=run_dir / "backend.log",
            run_dir=run_dir,
        )
    with pytest.raises(v3_backend.BackendError):
        v3_backend.BackendProcess.build(
            config_path=tmp_path / "config.json",
            db_path=run_dir / "jobs.sqlite3",
            artifacts_dir=run_dir / "artifacts",
            log_path=tmp_path / "outside.log",
            run_dir=run_dir,
        )
    with pytest.raises(v3_backend.BackendError):
        v3_backend.BackendProcess.build(
            config_path=tmp_path / "config.json",
            db_path=run_dir / "jobs.sqlite3",
            artifacts_dir=run_dir / "artifacts",
            log_path=run_dir / "backend.log",
            port=0,
        )


def test_backend_client_rejects_non_loopback_and_partial_deadline(tmp_path: Path) -> None:
    for url in (
        "http://example.invalid:8000",
        "http://user:pass@127.0.0.1:8000",
        "http://127.0.0.1:8000/path",
        "https://127.0.0.1:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:0",
    ):
        with pytest.raises(v3_backend.BackendError):
            v3_backend.BackendClient(url)
    with pytest.raises(v3_backend.BackendError):
        v3_backend.BackendClient("http://127.0.0.1:8000", deadline=object())
    with pytest.raises(v3_backend.BackendError):
        v3_backend.RunDeadline(object())
    with pytest.raises(v3_backend.BackendError):
        v3_backend.check_comfy_queue("http://192.0.2.1:8188", None)
    with pytest.raises(v3_backend.BackendError):
        v3_backend.check_comfy_queue(
            "http://192.0.2.1:8188", token="SYNTHETIC_TOKEN"
        )
    with pytest.raises(v3_backend.BackendError):
        v3_backend.check_comfy_queue("http://127.0.0.1:1", deadline=object())


def test_backend_client_payload_bounds_and_safe_job_ids(tmp_path: Path) -> None:
    client = v3_backend.BackendClient("http://127.0.0.1:1", timeout=0.2)
    bad_calls = [
        dict(prompt="", seed=1, timeout_seconds=10, idempotency_key="v3-bounds-0001"),
        dict(prompt="   ", seed=1, timeout_seconds=10, idempotency_key="v3-bounds-0002"),
        dict(prompt="x" * 4001, seed=1, timeout_seconds=10, idempotency_key="v3-bounds-0003"),
        dict(prompt="x", seed=True, timeout_seconds=10, idempotency_key="v3-bounds-0004"),
        dict(prompt="x", seed=-1, timeout_seconds=10, idempotency_key="v3-bounds-0005"),
        dict(prompt="x", seed=2_147_483_648, timeout_seconds=10, idempotency_key="v3-bounds-0006"),
        dict(prompt="x", seed=1, timeout_seconds=0, idempotency_key="v3-bounds-0007"),
        dict(prompt="x", seed=1, timeout_seconds=3601, idempotency_key="v3-bounds-0008"),
        dict(prompt="x", seed=1, timeout_seconds=10, idempotency_key="short"),
        dict(prompt="x", seed=1, timeout_seconds=10, idempotency_key="x" * 129),
        dict(prompt="x", seed=1, timeout_seconds=10, idempotency_key=" leading-space"),
        dict(prompt="x", seed=1, timeout_seconds=10, idempotency_key="new\nline-key"),
    ]
    for kwargs in bad_calls:
        with pytest.raises(v3_backend.BackendError) as excinfo:
            client.post_job(**kwargs)
        assert not isinstance(excinfo.value, v3_backend.BackendAmbiguous)
    for job_id in ("", "../jobs", "bad/id", "bad id", "x" * 129):
        with pytest.raises(v3_backend.BackendError):
            client.get_job(job_id)
    with pytest.raises(v3_backend.BackendError):
        client.get_events("job-1", after=-1)
    with pytest.raises(v3_backend.BackendAmbiguous):
        client.post_job(
            prompt="x",
            seed=1,
            timeout_seconds=10,
            idempotency_key="v3-bounds-valid",
        )


def test_run_deadline_requires_full_session_binding_and_never_extends(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)
    deadline = v3_backend.RunDeadline(binding)
    assert deadline.effective_deadline_at == min(
        binding.hard_end_at, binding.planned_cancel_at - 120
    )
    assert deadline.to_dict()["session"] == binding.to_dict()
    assert deadline.request_timeout(10_000) <= deadline.remaining()
    first = deadline.job_timeout_seconds()
    time.sleep(0.05)
    second = deadline.job_timeout_seconds()
    assert second <= first
    with pytest.raises(v3_backend.BackendError):
        deadline.request_timeout(0)
    with pytest.raises(v3_backend.BackendError):
        deadline.job_timeout_seconds(max_seconds=0)

    expired_binding = _binding(
        tmp_path,
        name="expired-session.json",
        created_offset=-5,
        planned_offset=60,
        hard_offset=1800,
    )
    expired = v3_backend.RunDeadline(expired_binding)
    assert expired.remaining() <= 0
    with pytest.raises(v3_backend.BackendDeadlineExpired):
        expired.require_remaining()


def test_actual_app_fake_comfy_one_post_idempotency_prompt_persistence_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, state):
        config_path = _verified_config(tmp_path, base_url)
        readiness = check_readiness(config_path)
        assert readiness.live_ready is True
        assert readiness.manifest["status"] == "VERIFIED"

        binding = _binding(tmp_path)
        deadline = v3_backend.RunDeadline(binding)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Keep the child app free of ambient proxy variables.
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(name, raising=False)
        process = _process(config_path=config_path, run_dir=run_dir)
        process.start(deadline=deadline)
        try:
            # The client must also ignore proxies. Set them only after the
            # child app has inherited its clean environment.
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
            monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
            client = v3_backend.BackendClient(
                process.base_url, timeout=5.0, deadline=deadline
            )
            health = client.health()
            assert health["status"] == "ok"
            assert health["mode"] == "COMFY_HTTP"
            assert health["mock"] is False

            key = "v3-integration-0001"
            prompt = "INTEGRATION v3 prompt"
            response = client.post_job(
                prompt=prompt,
                seed=424242,
                timeout_seconds=30,
                idempotency_key=key,
            )
            assert response["idempotent_replay"] is False
            job_id = response["job"]["job_id"]

            _wait_until(lambda: state.prompt_requests == 1, message="one Comfy POST")
            _wait_until(
                lambda: (client.get_job(job_id).get("job") or {}).get("prompt_id")
                == "prompt-1",
                message="persisted prompt id",
            )
            assert state.prompt_requests == 1
            assert len(state.prompt_bodies) == 1
            assert state.prompt_bodies[0]["prompt"]["2"]["inputs"]["text"] == prompt

            db_path = run_dir / "jobs.sqlite3"
            row = db_row(db_path, "SELECT * FROM jobs WHERE job_id=?", (job_id,))
            assert row is not None
            assert row["idempotency_key"] == key
            assert row["prompt_id"] == "prompt-1"
            assert prompt in row["payload_canonical"]
            assert state.prompt_bodies[0]["extra_data"]["correlation_key"] == row[
                "correlation_key"
            ]
            deadline_before = row["deadline_ts"]

            replay = client.post_job(
                prompt=prompt,
                seed=424242,
                timeout_seconds=30,
                idempotency_key=key,
            )
            assert replay["idempotent_replay"] is True
            assert replay["job"]["job_id"] == job_id
            time.sleep(0.2)
            assert state.prompt_requests == 1

            # The restarted child must not inherit the synthetic proxy vars.
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                monkeypatch.delenv(name, raising=False)
            assert process.stop() is True
            time.sleep(0.1)
            process = _process(config_path=config_path, run_dir=run_dir)
            process.start(deadline=deadline)
            client2 = v3_backend.BackendClient(
                process.base_url, timeout=5.0, deadline=deadline
            )
            recovered = client2.get_job(job_id)["job"]
            assert recovered["prompt_id"] == "prompt-1"
            assert (
                db_scalar(
                    db_path,
                    "SELECT deadline_ts FROM jobs WHERE job_id=?",
                    (job_id,),
                )
                == deadline_before
            )

            state.set_success()
            final = client2.wait_terminal(
                job_id, deadline=time.monotonic() + 10
            )
            assert final["status"] == "succeeded"
            result = client2.get_result(job_id)
            assert result["mock"] is False
            artifacts = result["result"]["artifacts"]
            assert artifacts
            assert Path(artifacts[0]["path"]).is_file()
            assert state.prompt_requests == 1
            assert (
                db_scalar(
                    db_path,
                    "SELECT COUNT(*) FROM job_events WHERE job_id=? AND type='remote_submitted'",
                    (job_id,),
                )
                == 1
            )
        finally:
            process.stop()


def test_actual_app_ambiguous_comfy_responses_are_not_retried(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        config_path = _verified_config(tmp_path, base_url)
        deadline = v3_backend.RunDeadline(_binding(tmp_path))
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        process = _process(config_path=config_path, run_dir=run_dir)
        process.start(deadline=deadline)
        try:
            client = v3_backend.BackendClient(
                process.base_url, timeout=5.0, deadline=deadline
            )
            db_path = run_dir / "jobs.sqlite3"

            state.prompt_mode = "ambiguous_500"
            first = client.post_job(
                prompt="REAL ambiguous first",
                seed=1,
                timeout_seconds=30,
                idempotency_key="v3-ambiguous-0001",
            )
            first_id = first["job"]["job_id"]
            _wait_until(lambda: state.prompt_requests == 1, message="first Comfy POST")
            first_final = client.wait_terminal(
                first_id, deadline=time.monotonic() + 10
            )
            assert first_final["status"] == "failed"
            assert first_final["error"]["code"] == "AMBIGUOUS_DISPATCH"
            assert (
                db_scalar(
                    db_path, "SELECT prompt_id FROM jobs WHERE job_id=?", (first_id,)
                )
                is None
            )
            time.sleep(0.3)
            assert state.prompt_requests == 1

            state.prompt_mode = "malformed_200"
            second = client.post_job(
                prompt="REAL ambiguous second",
                seed=2,
                timeout_seconds=30,
                idempotency_key="v3-ambiguous-0002",
            )
            second_id = second["job"]["job_id"]
            _wait_until(lambda: state.prompt_requests == 2, message="second Comfy POST")
            second_final = client.wait_terminal(
                second_id, deadline=time.monotonic() + 10
            )
            assert second_final["status"] == "failed"
            assert second_final["error"]["code"] == "AMBIGUOUS_DISPATCH"
            time.sleep(0.3)
            assert state.prompt_requests == 2
            second_row = db_row(
                db_path, "SELECT * FROM jobs WHERE job_id=?", (second_id,)
            )
            assert second_row is not None
            assert "REAL ambiguous second" in second_row["payload_canonical"]
            assert second_row["prompt_id"] is None
        finally:
            process.stop()


def test_deadline_consumed_before_start_and_before_post(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        config_path = _verified_config(tmp_path, base_url)
        expired_binding = _binding(
            tmp_path,
            name="expired-session.json",
            created_offset=-5,
            planned_offset=60,
            hard_offset=1800,
        )
        expired = v3_backend.RunDeadline(expired_binding)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        process = _process(config_path=config_path, run_dir=run_dir)
        with pytest.raises(v3_backend.BackendDeadlineExpired):
            process.start(deadline=expired)
        assert not (run_dir / "backend.log").exists()

        valid = v3_backend.RunDeadline(_binding(tmp_path, name="valid-session.json"))
        process = _process(config_path=config_path, run_dir=run_dir)
        process.start(deadline=valid)
        try:
            client = v3_backend.BackendClient(
                process.base_url, timeout=5.0, deadline=expired
            )
            with pytest.raises(v3_backend.BackendDeadlineExpired):
                client.health()
            with pytest.raises(v3_backend.BackendDeadlineExpired):
                client.get_job("job-anything")
            with pytest.raises(v3_backend.BackendDeadlineExpired):
                client.post_job(
                    prompt="must not post",
                    seed=1,
                    timeout_seconds=10,
                    idempotency_key="v3-expired-0001",
                )
            with pytest.raises(v3_backend.BackendDeadlineExpired):
                v3_backend.check_comfy_queue(base_url, deadline=expired)
            time.sleep(0.5)
            assert state.prompt_requests == 0
            assert (
                db_scalar(run_dir / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 0
            )
        finally:
            process.stop()


def test_http_queue_and_health_timeouts_are_clipped_to_absolute_deadline(
    tmp_path: Path,
) -> None:
    with _policy_probe("slow") as (queue_url, queue_server):
        queue_binding = _binding(
            tmp_path,
            name="queue-short.json",
            created_offset=-5,
            planned_offset=2,
            hard_offset=1800,
        )
        queue_deadline = v3_backend.RunDeadline(
            queue_binding, export_reserve_seconds=0
        )
        started = time.monotonic()
        with pytest.raises(v3_backend.BackendDeadlineExpired):
            v3_backend.check_comfy_queue(
                queue_url, deadline=queue_deadline, timeout=10.0
            )
        elapsed = time.monotonic() - started
        assert elapsed < 2.8, elapsed
        assert queue_server.hits == ["/queue"]  # type: ignore[attr-defined]

    with _policy_probe("slow") as (health_url, health_server):
        health_binding = _binding(
            tmp_path,
            name="health-short.json",
            created_offset=-5,
            planned_offset=2,
            hard_offset=1800,
        )
        health_deadline = v3_backend.RunDeadline(
            health_binding, export_reserve_seconds=0
        )
        client = v3_backend.BackendClient(
            health_url, timeout=10.0, deadline=health_deadline
        )
        started = time.monotonic()
        with pytest.raises(v3_backend.BackendDeadlineExpired):
            client.health()
        elapsed = time.monotonic() - started
        assert elapsed < 2.8, elapsed
        assert health_server.hits == ["/health"]  # type: ignore[attr-defined]


def test_trickle_get_aborts_at_absolute_deadline_without_late_success(
    tmp_path: Path,
) -> None:
    with _trickle_probe() as (url, server):
        binding = _binding(
            tmp_path,
            name="trickle-get.json",
            created_offset=-5,
            planned_offset=1,
            hard_offset=1800,
        )
        deadline = v3_backend.RunDeadline(binding, export_reserve_seconds=0)
        client = v3_backend.BackendClient(url, timeout=30.0, deadline=deadline)
        started = time.monotonic()
        with pytest.raises(v3_backend.BackendDeadlineExpired):
            client.health()
        assert time.monotonic() - started < 2.5
        assert server.finished.wait(3.0)  # type: ignore[attr-defined]
        assert server.completed is False  # type: ignore[attr-defined]
        assert server.hits == ["/health"]  # type: ignore[attr-defined]
        assert not _watchdog_threads()


def test_trickle_queue_aborts_at_absolute_deadline(tmp_path: Path) -> None:
    with _trickle_probe() as (url, server):
        binding = _binding(
            tmp_path,
            name="trickle-queue.json",
            created_offset=-5,
            planned_offset=1,
            hard_offset=1800,
        )
        deadline = v3_backend.RunDeadline(binding, export_reserve_seconds=0)
        started = time.monotonic()
        with pytest.raises(v3_backend.BackendDeadlineExpired):
            v3_backend.check_comfy_queue(url, deadline=deadline, timeout=30.0)
        assert time.monotonic() - started < 2.5
        assert server.finished.wait(3.0)  # type: ignore[attr-defined]
        assert server.completed is False  # type: ignore[attr-defined]
        assert server.hits == ["/queue"]  # type: ignore[attr-defined]
        assert not _watchdog_threads()


def test_backend_process_health_uses_clipped_deadline(tmp_path: Path) -> None:
    with _trickle_probe() as (url, server):
        binding = _binding(
            tmp_path,
            name="trickle-startup.json",
            created_offset=-5,
            planned_offset=1,
            hard_offset=1800,
        )
        deadline = v3_backend.RunDeadline(binding, export_reserve_seconds=0)
        process = v3_backend.BackendProcess(
            argv=(sys.executable, "-m", "anime_studio", "serve"),
            base_url=url,
            log_path=tmp_path / "backend.log",
        )
        started = time.monotonic()
        with pytest.raises(v3_backend.BackendDeadlineExpired):
            process.health(deadline=deadline)
        assert time.monotonic() - started < 2.5
        assert server.finished.wait(3.0)  # type: ignore[attr-defined]
        assert server.completed is False  # type: ignore[attr-defined]
        assert not _watchdog_threads()


def test_trickle_post_after_send_is_ambiguous_not_presend(tmp_path: Path) -> None:
    with _trickle_probe() as (url, server):
        binding = _binding(
            tmp_path,
            name="trickle-post.json",
            created_offset=-5,
            planned_offset=8,
            hard_offset=1800,
        )
        deadline = v3_backend.RunDeadline(binding, export_reserve_seconds=0)
        client = v3_backend.BackendClient(url, timeout=30.0, deadline=deadline)
        started = time.monotonic()
        with pytest.raises(v3_backend.BackendAmbiguous, match="after send"):
            client.post_job(
                prompt="trickle post",
                seed=1,
                timeout_seconds=1,
                idempotency_key="v3-trickle-0001",
            )
        assert time.monotonic() - started < 10.0
        assert server.finished.wait(3.0)  # type: ignore[attr-defined]
        assert server.completed is False  # type: ignore[attr-defined]
        assert server.hits == ["/jobs"]  # type: ignore[attr-defined]
        assert not _watchdog_threads()


def test_timeout_is_stored_and_recovery_does_not_extend_it(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        config_path = _verified_config(tmp_path, base_url)
        deadline = v3_backend.RunDeadline(
            _binding(
                tmp_path,
                name="timeout-session.json",
                created_offset=-5,
                planned_offset=60,
                hard_offset=1800,
            ),
            export_reserve_seconds=30,
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        process = _process(config_path=config_path, run_dir=run_dir)
        process.start(deadline=deadline)
        try:
            client = v3_backend.BackendClient(
                process.base_url, timeout=5.0, deadline=deadline
            )
            with pytest.raises(v3_backend.BackendDeadlineExpired):
                client.post_job(
                    prompt="REAL too-long timeout",
                    seed=1,
                    timeout_seconds=3600,
                    idempotency_key="v3-timeout-0001",
                )
            assert state.prompt_requests == 0
            db_path = run_dir / "jobs.sqlite3"
            assert db_scalar(db_path, "SELECT COUNT(*) FROM jobs") == 0

            response = client.post_job(
                prompt="REAL stored timeout",
                seed=1,
                timeout_seconds=10,
                idempotency_key="v3-timeout-0002",
            )
            job_id = response["job"]["job_id"]
            _wait_until(lambda: state.prompt_requests == 1, message="Comfy POST")
            _wait_until(
                lambda: (client.get_job(job_id).get("job") or {}).get("prompt_id")
                == "prompt-1",
                message="persisted prompt id",
            )
            row = db_row(db_path, "SELECT * FROM jobs WHERE job_id=?", (job_id,))
            assert row is not None
            assert row["timeout_seconds"] == 10
            stored_deadline = row["deadline_ts"]

            assert process.stop() is True
            time.sleep(0.1)
            process = _process(config_path=config_path, run_dir=run_dir)
            process.start(deadline=deadline)
            client2 = v3_backend.BackendClient(
                process.base_url, timeout=5.0, deadline=deadline
            )
            recovered = client2.get_job(job_id)["job"]
            assert recovered["prompt_id"] == "prompt-1"
            recovered_row = db_row(
                db_path, "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            )
            assert recovered_row is not None
            assert recovered_row["timeout_seconds"] == 10
            assert recovered_row["deadline_ts"] == stored_deadline
            state.set_success()
            final = client2.wait_terminal(
                job_id, deadline=time.monotonic() + 10
            )
            assert final["status"] == "succeeded"
            assert state.prompt_requests == 1
        finally:
            process.stop()


def test_check_comfy_queue_nonempty_and_no_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, state):
        state.queue_pending = ["foreign-prompt"]
        with pytest.raises(v3_backend.BackendError):
            v3_backend.check_comfy_queue(base_url)
        state.queue_pending = []
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
        assert v3_backend.check_comfy_queue(base_url) == {
            "running": 0,
            "pending": 0,
        }


def test_http_transport_rejects_redirects_oversize_and_lost_post_response(
    tmp_path: Path,
) -> None:
    with _policy_probe("redirect") as (url, server):
        client = v3_backend.BackendClient(url, timeout=2.0)
        with pytest.raises(v3_backend.BackendError) as excinfo:
            client.health()
        assert "redirect" in str(excinfo.value).lower()
        with pytest.raises(v3_backend.BackendAmbiguous):
            client.post_job(
                prompt="redirect",
                seed=1,
                timeout_seconds=10,
                idempotency_key="v3-redirect-0001",
            )
        assert all(hit in {"/health", "/jobs"} for hit in server.hits)  # type: ignore[attr-defined]

    with _policy_probe("oversized") as (url, _server):
        client = v3_backend.BackendClient(url, timeout=2.0)
        with pytest.raises(v3_backend.BackendError) as excinfo:
            client.health()
        assert "bounded output" in str(excinfo.value).lower()

    with _policy_probe("error500") as (url, _server):
        token = "SYNTHETIC_SECRET_TOKEN_XYZ"
        with pytest.raises(v3_backend.BackendError) as excinfo:
            v3_backend.check_comfy_queue(url, token=token)
        assert token not in str(excinfo.value)

    with _policy_probe("drop") as (url, server):
        client = v3_backend.BackendClient(url, timeout=2.0)
        with pytest.raises(v3_backend.BackendAmbiguous):
            client.post_job(
                prompt="lost response",
                seed=1,
                timeout_seconds=10,
                idempotency_key="v3-lost-0001",
            )
        assert server.hits == ["/jobs"]  # type: ignore[attr-defined]
        assert len(server.bodies) == 1  # type: ignore[attr-defined]


def test_operator_run_backend_once_actual_app_and_fake_comfy_durable_intent(
    tmp_path: Path,
) -> None:
    with run_stub() as (base_url, state):
        config_path = _verified_config(tmp_path, base_url)
        target = tmp_path / "run"
        target.mkdir()
        v3_operator._ensure_run_document(
            target, v3_operator._bundle_hashes(v3_operator.load_bundle())
        )
        now = int(time.time())
        bundle = {"job": {"prompt": "REAL operator prompt", "seed": 424242}}

        def complete_after_prompt() -> None:
            _wait_until(lambda: state.prompt_requests == 1, message="Comfy POST")
            state.set_success()

        completer = threading.Thread(target=complete_after_prompt, daemon=True)
        completer.start()
        result = v3_operator._run_backend_once(
            bundle=bundle,
            target=target,
            config_path=config_path,
            binding=_operator_binding(planned_cancel_at=now + 300, hard_end_at=now + 600),
            backend_runner=None,
        )
        completer.join(timeout=10)
        assert not completer.is_alive()
        assert result["status"] == "succeeded"
        assert state.prompt_requests == 1
        job_id = result["job_id"]
        intent = json.loads((target / "submit-intent.json").read_text(encoding="utf-8"))
        assert intent["status"] == "JOB_RECORDED"
        assert intent["job_id"] == job_id
        assert intent["idempotency_key"]
        assert intent["prompt_sha256"]
        assert (target / "job-id.json").is_file()
        result_file = json.loads(
            (target / "backend-result.json").read_text(encoding="utf-8")
        )
        assert result_file["status"] == "succeeded"
        db_path = target / "jobs.sqlite3"
        row = db_row(db_path, "SELECT * FROM jobs WHERE job_id=?", (job_id,))
        assert row is not None
        assert row["prompt_id"] == "prompt-1"
        assert "REAL operator prompt" in row["payload_canonical"]
        assert db_scalar(db_path, "SELECT COUNT(*) FROM jobs") == 1

        collected = v3_operator.collect(run_dir=target)
        assert collected["status"] == "COLLECTED_DECODE_NOT_VERIFIED"
        assert collected["completed"][0]["target"] == "artifact.mp4"
        assert Path(collected["manifest_path"]).parent.joinpath("artifact.mp4").is_file()

        # Known-job recovery must reconnect through the same DB, not re-POST.
        again = v3_operator._run_backend_once(
            bundle=bundle,
            target=target,
            config_path=config_path,
            binding=_operator_binding(planned_cancel_at=now + 300, hard_end_at=now + 600),
            backend_runner=None,
        )
        assert again["status"] == "succeeded"
        assert state.prompt_requests == 1
        assert db_scalar(db_path, "SELECT COUNT(*) FROM jobs") == 1
        assert json.loads(
            (target / "submit-intent.json").read_text(encoding="utf-8")
        )["job_id"] == job_id


def test_operator_ambiguous_and_nonempty_queue_do_not_post(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        config_path = _verified_config(tmp_path, base_url)
        now = int(time.time())
        bundle = {"job": {"prompt": "REAL operator ambiguous", "seed": 1}}
        state.prompt_mode = "ambiguous_500"
        target = tmp_path / "ambiguous"
        target.mkdir()
        first = v3_operator._run_backend_once(
            bundle=bundle,
            target=target,
            config_path=config_path,
            binding=_operator_binding(planned_cancel_at=now + 300, hard_end_at=now + 600),
            backend_runner=None,
        )
        assert first["status"] == "failed"
        assert first["job"]["error"]["code"] == "AMBIGUOUS_DISPATCH"
        assert state.prompt_requests == 1
        intent = json.loads((target / "submit-intent.json").read_text(encoding="utf-8"))
        assert intent["status"] == "JOB_RECORDED"
        assert db_scalar(
            target / "jobs.sqlite3",
            "SELECT prompt_id FROM jobs WHERE job_id=?",
            (first["job_id"],),
        ) is None

        again = v3_operator._run_backend_once(
            bundle=bundle,
            target=target,
            config_path=config_path,
            binding=_operator_binding(planned_cancel_at=now + 300, hard_end_at=now + 600),
            backend_runner=None,
        )
        assert again["status"] == "failed"
        assert again["job"]["error"]["code"] == "AMBIGUOUS_DISPATCH"
        assert state.prompt_requests == 1

        state.prompt_mode = "accepted"
        state.queue_pending = ["foreign-prompt"]
        queue_target = tmp_path / "queue"
        queue_target.mkdir()
        with pytest.raises(v3_backend.BackendError):
            v3_operator._run_backend_once(
                bundle={"job": {"prompt": "REAL queue blocked", "seed": 2}},
                target=queue_target,
                config_path=config_path,
                binding=_operator_binding(planned_cancel_at=now + 300, hard_end_at=now + 600),
                backend_runner=None,
            )
        assert state.prompt_requests == 1
        assert (queue_target / "submit-intent.json").is_file()
        assert (
            db_scalar(queue_target / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 0
        )


def test_operator_deadline_before_backend_start_does_not_post(tmp_path: Path) -> None:
    with run_stub() as (base_url, state):
        config_path = _verified_config(tmp_path, base_url)
        target = tmp_path / "deadline"
        target.mkdir()
        now = int(time.time())
        result = v3_operator._run_backend_once(
            bundle={"job": {"prompt": "REAL deadline", "seed": 3}},
            target=target,
            config_path=config_path,
            binding=_operator_binding(
                planned_cancel_at=now + 100, hard_end_at=now + 600
            ),
            backend_runner=None,
        )
        assert result["status"] == "DEADLINE_BEFORE_POST"
        assert result["no_post"] is True
        assert state.prompt_requests == 0
        assert not (target / "submit-intent.json").exists()


def test_production_v3_draft_gate_is_unchanged() -> None:
    readiness = check_readiness(
        ROOT / "config" / "p1.wan22-5b.shot01-v3.draft.json"
    )
    assert readiness.live_ready is False
    assert any("manifest.status is not VERIFIED" in issue for issue in readiness.issues)

def test_actual_comfy_history_producer_exact_prefix_then_remote_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_stub() as (base_url, state):
        config_path = _verified_config(tmp_path, base_url)
        state.set_success("shot-01-v3_00001_.mp4")
        transport = build_transport_from_config(config_path)
        target = tmp_path / "run"
        v3_operator.prepare(run_dir=target)
        (target / "job-id.json").write_text(
            json.dumps({"job_id": "job-prefix", "idempotency_key": "key"}),
            encoding="utf-8",
        )
        (target / "backend-result.json").write_text(
            json.dumps(
                {
                    "status": "succeeded",
                    "job_id": "job-prefix",
                    "result": {
                        "job_id": "job-prefix",
                        "result": {"prompt_id": "prompt-1", "artifacts": []},
                    },
                }
            ),
            encoding="utf-8",
        )
        identity_file = tmp_path / "id"
        known_hosts = tmp_path / "known_hosts"
        identity_file.write_text("x", encoding="utf-8")
        known_hosts.write_text("[example.test]:22 ssh-ed25519 AAAA\n", encoding="utf-8")
        identity = SimpleNamespace(
            host="example.test",
            user="root",
            port=22,
            known_hosts=known_hosts,
            identity_file=identity_file,
        )
        now = int(time.time())
        lifecycle = v3_lifecycle.new_lifecycle(
            run_id=target.name,
            session=v3_lifecycle.SessionBinding(
                order_id=99001,
                server_id=112213,
                created_at=now - 5,
                currency="USD",
                cap_usd="1.00",
                duration_seconds=3600,
                planned_cancel_at=now + 300,
                hard_end_at=now + 600,
                quote_hash="a" * 64,
            ),
            bundle_hashes={"config": "a" * 64},
            phase="READY",
        )
        lifecycle = v3_lifecycle.update_phase(
            target / "lifecycle.json",
            lifecycle,
            phase="READY",
            pipeline={
                "ssh_identity": {
                    "host": identity.host,
                    "user": identity.user,
                    "port": identity.port,
                    "known_hosts": str(identity.known_hosts),
                    "fingerprint": "SHA256:test",
                }
            },
        )
        remote_root = tmp_path / "remote-root"
        local_artifact = remote_root / "ComfyUI" / "output" / "shot-01-v3_00001_.mp4"
        local_artifact.parent.mkdir(parents=True)
        local_artifact.write_bytes(b"structural-prefix-fixture")
        digest = hashlib.sha256(local_artifact.read_bytes()).hexdigest()

        class Runner:
            ssh_keygen_path = __import__("shutil").which("ssh-keygen")

            def run_remote(self, identity, args, timeout=None, **kwargs):
                assert args[0] == "python3" and args[2] == "metadata"
                remote_path = args[3]
                local = remote_root / remote_path.removeprefix("/workspace/")
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "p1" / "v3_remote_fetch_helper.py"),
                        "metadata",
                        str(local),
                        str(remote_root),
                    ],
                    capture_output=True,
                    check=False,
                )
                assert completed.returncode == 0, completed.stderr
                payload = json.loads(completed.stdout)
                payload["path"] = remote_path
                return SimpleNamespace(returncode=0, stdout=json.dumps(payload), timed_out=False)

            def put_file(self, *args, **kwargs):
                return None

        runner = Runner()
        export = v3_operator._produce_remote_export(
            target=target,
            remote=identity,
            workspace="/workspace",
            config_path=config_path,
            lifecycle=lifecycle,
            transport=transport,
            runner=runner,
            deadline_monotonic=time.monotonic() + 10,
        )
        assert export["filename"] == "shot-01-v3_00001_.mp4"
        assert export["remote_artifact_path"].endswith("/shot-01-v3_00001_.mp4")
        assert export["sha256"] == digest

        monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)

        def fake_fetch(**kwargs):
            Path(kwargs["local_path"]).write_bytes(local_artifact.read_bytes())
            return {"sha256": digest}

        monkeypatch.setattr(v3_lifecycle, "fetch_exact_stream", fake_fetch)
        collected = v3_operator.collect(
            run_dir=target,
            remote=identity,
            remote_artifact_path=export["remote_artifact_path"],
            remote_helper_path=export["remote_helper_path"],
            remote_expected_size=export["size"],
            remote_expected_sha256=export["sha256"],
            expected_fingerprint="SHA256:test",
            remote_runner=runner,
        )
        assert collected["completed"][0]["target"] == "artifact.mp4"
