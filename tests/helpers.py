from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

from anime_studio.artifacts import LocalArtifactStore
from anime_studio.comfy import MockComfyTransport
from anime_studio.db import Database
from anime_studio.models import JobRequest, MockOptions
from anime_studio.orchestrator import Orchestrator
from anime_studio.worker import Worker


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServerProcess:
    def __init__(
        self,
        *,
        db_path: Path,
        artifacts_dir: Path,
        log_path: Path,
        no_worker: bool = False,
        env: dict[str, str] | None = None,
        poll_interval: float = 0.02,
        extra_args: list[str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        self.log_path = log_path
        self.no_worker = no_worker
        self.extra_env = env or {}
        self.poll_interval = poll_interval
        self.extra_args = list(extra_args or [])
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self.client: httpx.Client | None = None
        self._log_handle = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.port = free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self.log_path, "ab", buffering=0)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.update(self.extra_env)
        args = [
            sys.executable,
            "-m",
            "anime_studio",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--db",
            str(self.db_path),
            "--artifacts",
            str(self.artifacts_dir),
            "--poll-interval",
            str(self.poll_interval),
            "--log-level",
            "warning",
        ]
        if self.no_worker:
            args.append("--no-worker")
        args.extend(self.extra_args)
        self.proc = subprocess.Popen(
            args,
            cwd=str(ROOT),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self.client = httpx.Client(base_url=self.url, timeout=10.0)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.proc.returncode}:\n{self.log_tail()}"
                )
            try:
                response = self.client.get("/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.03)
        raise RuntimeError(f"server did not become ready:\n{self.log_tail()}")

    def log_tail(self, lines: int = 60) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            return "\n".join(text.splitlines()[-lines:])
        except OSError:
            return "(no log)"

    def wait_exit(self, timeout: float = 10.0) -> int:
        assert self.proc is not None
        code = self.proc.wait(timeout=timeout)
        if self.client is not None:
            self.client.close()
            self.client = None
        return code

    def stop(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            self.proc = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


@contextmanager
def running_server(
    tmp_path: Path,
    *,
    name: str = "server",
    no_worker: bool = False,
    env: dict[str, str] | None = None,
    db_path: Path | None = None,
    artifacts_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> Iterator[ServerProcess]:
    server = ServerProcess(
        db_path=db_path or (tmp_path / "p0.sqlite3"),
        artifacts_dir=artifacts_dir or (tmp_path / "artifacts"),
        log_path=tmp_path / f"{name}.log",
        no_worker=no_worker,
        env=env,
        extra_args=extra_args,
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


@dataclass
class Stack:
    db: Database
    adapter: MockComfyTransport
    artifacts: LocalArtifactStore
    orchestrator: Orchestrator
    worker: Worker


def make_stack(
    tmp_path: Path, *, poll_interval: float = 0.02, lease_seconds: float = 0.5
) -> Stack:
    db = Database(tmp_path / "p0.sqlite3")
    db.init()
    adapter = MockComfyTransport(db)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    orchestrator = Orchestrator(
        db, adapter, artifacts, lease_seconds=lease_seconds
    )
    worker = Worker(
        orchestrator, poll_interval=poll_interval, lease_seconds=lease_seconds
    )
    return Stack(db, adapter, artifacts, orchestrator, worker)


def make_request(
    *,
    prompt: str = "MOCK test job",
    timeout_seconds: int = 30,
    duration_seconds: float = 0.5,
    fail: bool = False,
    fail_at_progress: int = 50,
    seed: int | None = 7,
) -> JobRequest:
    return JobRequest(
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        seed=seed,
        mock_options=MockOptions(
            duration_seconds=duration_seconds,
            fail=fail,
            fail_at_progress=fail_at_progress,
        ),
    )


def post_job(
    client: httpx.Client,
    key: str,
    *,
    prompt: str = "MOCK HTTP test",
    timeout_seconds: int = 30,
    duration_seconds: float = 0.5,
    fail: bool = False,
    fail_at_progress: int = 50,
    seed: int = 11,
) -> httpx.Response:
    return client.post(
        "/jobs",
        headers={"Idempotency-Key": key},
        json={
            "prompt": prompt,
            "timeout_seconds": timeout_seconds,
            "seed": seed,
            "mock_options": {
                "duration_seconds": duration_seconds,
                "fail": fail,
                "fail_at_progress": fail_at_progress,
            },
        },
    )


# A crash test may lose the HTTP response when the fault-injected server process
# exits after accepting the request. These are the only transport errors that
# can represent that response loss; connect errors, timeouts, broad HTTPError,
# arbitrary exceptions and unexpected statuses must fail the test.
CRASH_RESPONSE_ERRORS = (
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


def post_job_crash_tolerant(
    client: httpx.Client, key: str, **kwargs: Any
) -> httpx.Response | None:
    """Send exactly one POST and tolerate only an allowlisted response loss."""
    try:
        return post_job(client, key, **kwargs)
    except CRASH_RESPONSE_ERRORS:
        return None


def persisted_job_for_crash_response(
    db_path: Path,
    idempotency_key: str,
    response: httpx.Response | None,
) -> dict[str, Any]:
    """Recover one durable job and reconcile any received 201 payload with it."""
    count = db_scalar(
        db_path,
        "SELECT COUNT(*) FROM jobs WHERE idempotency_key=?",
        (idempotency_key,),
    )
    assert count == 1, (
        f"expected exactly one persisted job for {idempotency_key!r}, got {count}"
    )
    row = db_row(
        db_path,
        "SELECT * FROM jobs WHERE idempotency_key=?",
        (idempotency_key,),
    )
    assert row is not None, f"no persisted job for {idempotency_key!r}"
    if response is not None:
        assert response.status_code == 201, response.text
        payload = response.json()["job"]
        assert payload["job_id"] == row["job_id"]
        assert payload["correlation_key"] == row["correlation_key"]
        assert payload["payload_sha256"] == row["payload_sha256"]
    return row


def wait_job(
    client: httpx.Client,
    job_id: str,
    *,
    until: Callable[[dict[str, Any]], bool] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()["job"]
        if until is not None and until(job):
            return job
        if job["status"] in {"succeeded", "failed", "cancelled", "timed_out"}:
            return job
        time.sleep(0.03)
    raise TimeoutError(f"job {job_id} did not finish")


def wait_direct(
    worker: Worker,
    *,
    job_id: str,
    orchestrator: Orchestrator,
    until: Callable[[dict[str, Any]], bool] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        worker.tick()
        last = orchestrator.get_job(job_id)
        if last is not None:
            if until is not None and until(last):
                return last
            if last["status"] in {"succeeded", "failed", "cancelled", "timed_out"}:
                return last
        time.sleep(0.02)
    raise TimeoutError(f"direct job {job_id} did not finish; last={last}")


def db_scalar(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> Any:
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None


def db_row(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
