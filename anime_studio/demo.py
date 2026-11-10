"""Offline end-to-end demonstration and evidence writer.

The demo starts real uvicorn subprocesses on 127.0.0.1, uses the real HTTP API,
and shows progress, result, errors, cancellation, timeout and a restart. Every
result is MOCK JSON; no GPU, RunPod, model or network provider is involved.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx

from .timeutils import now_iso


ROOT = Path(__file__).resolve().parent.parent
MOCK_NOTE = "MOCK RESULT - NOT A REAL VIDEO"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _base_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def prepare_run_dir(data_dir: Path) -> Path:
    """Create a unique run subdirectory without deleting existing data.

    ``data_dir`` is a base directory owned by the caller. The demo never
    removes it, never removes the repository root or a drive root, and preserves
    unrelated files and sentinels. This makes repeated demos reproducible
    without destructive cleanup.
    """

    base = Path(data_dir).expanduser().resolve()
    repo = ROOT.resolve()
    if base == repo:
        raise ValueError("refusing to use the repository root as demo data dir")
    if base == Path(base.anchor):
        raise ValueError("refusing to use a drive/filesystem root as demo data dir")
    if base.exists() and not base.is_dir():
        raise ValueError(f"demo data dir exists but is not a directory: {base}")

    base.mkdir(parents=True, exist_ok=True)
    run_dir = None
    for _ in range(5):
        candidate = base / (
            f"run-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            run_dir = candidate
            break
        except FileExistsError:
            continue
    if run_dir is None:
        raise RuntimeError(f"could not create a unique demo run dir under {base}")

    sentinel = run_dir / "RUN_SENTINEL.json"
    sentinel.write_text(
        json.dumps(
            {
                "mock": True,
                "note": "Anime Video Studio P0 demo run. Do not delete sibling runs.",
                "created_at": now_iso(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


class DemoServer:
    def __init__(self, data_dir: Path, generation: int) -> None:
        self.data_dir = data_dir
        self.generation = generation
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self.client: httpx.Client | None = None
        self.log_path = data_dir / f"server-{generation}.log"
        self._log_handle = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.port = _free_port()
        self._log_handle = open(self.log_path, "ab", buffering=0)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "anime_studio",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--db",
                str(self.data_dir / "p0.sqlite3"),
                "--artifacts",
                str(self.data_dir / "artifacts"),
                "--poll-interval",
                "0.02",
                "--log-level",
                "warning",
            ],
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
                    f"demo server exited with {self.proc.returncode}:\n{self.log_tail()}"
                )
            try:
                response = self.client.get("/health")
                if response.status_code == 200:
                    print(f"[demo] server {self.generation} ready at {self.url}")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise RuntimeError(f"demo server did not become ready:\n{self.log_tail()}")

    def log_tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            return "\n".join(text.splitlines()[-lines:])
        except OSError:
            return "(no server log)"

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


def _post_job(
    client: httpx.Client,
    key: str,
    *,
    prompt: str,
    timeout_seconds: int,
    duration: float,
    fail: bool = False,
    fail_at: int = 50,
    seed: int = 424242,
) -> httpx.Response:
    return client.post(
        "/jobs",
        headers={"Idempotency-Key": key},
        json={
            "prompt": prompt,
            "timeout_seconds": timeout_seconds,
            "seed": seed,
            "mock_options": {
                "duration_seconds": duration,
                "fail": fail,
                "fail_at_progress": fail_at,
            },
        },
    )


def _wait_job(
    client: httpx.Client,
    job_id: str,
    *,
    cursor: int = 0,
    until: Callable[[dict[str, Any]], bool] | None = None,
    timeout: float = 45.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    deadline = time.monotonic() + timeout
    events: list[dict[str, Any]] = []
    last_status = None
    while time.monotonic() < deadline:
        response = client.get(
            f"/jobs/{job_id}/events", params={"after": cursor, "limit": 100}
        )
        response.raise_for_status()
        batch = response.json()
        cursor = int(batch["next_cursor"])
        for event in batch["events"]:
            events.append(event)
            print(
                f"[demo]   event {event['cursor']}: {event['type']} "
                f"{json.dumps(event['data'], ensure_ascii=False)}"
            )
        response = client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()["job"]
        if job["status"] != last_status:
            print(
                f"[demo]   status={job['status']} progress={job['progress']} "
                f"prompt_id={job['prompt_id']}"
            )
            last_status = job["status"]
        terminal = job["status"] in {"succeeded", "failed", "cancelled", "timed_out"}
        if until is not None and until(job):
            if not terminal:
                return job, events, cursor
        if terminal:
            # The status update and its terminal event are committed in one
            # transaction, but the status read above may follow the last event
            # fetch. A late-event guard can append one more event shortly after
            # cancellation. Drain until two consecutive polls are empty.
            empty_polls = 0
            for _ in range(10):
                response = client.get(
                    f"/jobs/{job_id}/events",
                    params={"after": cursor, "limit": 100},
                )
                response.raise_for_status()
                batch = response.json()
                cursor = int(batch["next_cursor"])
                if batch["events"]:
                    empty_polls = 0
                    for event in batch["events"]:
                        events.append(event)
                        print(
                            f"[demo]   event {event['cursor']}: {event['type']} "
                            f"{json.dumps(event['data'], ensure_ascii=False)}"
                        )
                else:
                    empty_polls += 1
                    if empty_polls >= 2:
                        break
                time.sleep(0.03)
            return job, events, cursor
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not reach the expected state")


def run_demo(*, evidence_path: Path, data_dir: Path) -> int:
    base_dir = Path(data_dir)
    evidence_path = Path(evidence_path)
    try:
        run_dir = prepare_run_dir(base_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[demo] refusing unsafe/unusable data dir: {exc}", file=sys.stderr)
        return 2
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = now_iso()
    evidence: dict[str, Any] = {
        "schema": "anime-video-studio/p0-demo/v1",
        "mode": "MOCK",
        "offline": True,
        "base_commit": _base_commit(),
        "worktree": "P0 changes are uncommitted by instruction; base commit is HEAD",
        "data_dir": str(base_dir.resolve()),
        "run_dir": str(run_dir),
        "started_at": started_at,
        "cloud_spend_rub": 0,
        "gpu": "not used and not checked",
        "ffmpeg": "not required for P0; not found on this machine",
        "scenarios": [],
        "server_logs": [],
        "notes": [
            "All results/artifacts are MOCK JSON, not real video.",
            "No RunPod, GPU, model download, LoRA training or external provider call.",
            "Persistence is SQLite; restart recovery uses the stable correlation key.",
        ],
    }
    servers: list[DemoServer] = []
    try:
        print(f"[demo] unique run dir: {run_dir}")
        server = DemoServer(run_dir, 1)
        servers.append(server)
        server.start()
        client = server.client
        assert client is not None

        # 1. Success, progress, result, idempotent replay and conflict.
        response = _post_job(
            client,
            "demo-ok-0001",
            prompt="MOCK: anime storyboard, four vertical shots, one visual style",
            timeout_seconds=30,
            duration=1.0,
        )
        response.raise_for_status()
        created = response.json()
        job_id = created["job"]["job_id"]
        print(f"[demo] submitted success job {job_id}")
        job, events, cursor = _wait_job(client, job_id)
        if job["status"] != "succeeded":
            raise RuntimeError(f"expected succeeded, got {job['status']}: {job}")
        result_response = client.get(f"/jobs/{job_id}/result")
        result_response.raise_for_status()
        result = result_response.json()
        artifact_path = Path(result["result"]["artifacts"][0]["path"])
        if not artifact_path.exists():
            raise RuntimeError(f"mock artifact missing: {artifact_path}")
        if artifact_path.suffix == ".mp4":
            raise RuntimeError("P0 must not produce MP4 artifacts")
        replay_response = _post_job(
            client,
            "demo-ok-0001",
            prompt="MOCK: anime storyboard, four vertical shots, one visual style",
            timeout_seconds=30,
            duration=1.0,
        )
        replay = replay_response.json()
        if replay_response.status_code != 200 or replay["job"]["job_id"] != job_id:
            raise RuntimeError("idempotent replay did not return the same job")
        conflict_response = _post_job(
            client,
            "demo-ok-0001",
            prompt="MOCK: different payload with the same key",
            timeout_seconds=30,
            duration=1.0,
        )
        if conflict_response.status_code != 409:
            raise RuntimeError("same key with a different payload must return 409")
        evidence["scenarios"].append(
            {
                "name": "success_and_idempotency",
                "job_id": job_id,
                "status": job["status"],
                "events": events,
                "next_cursor": cursor,
                "result": result,
                "artifact_exists": True,
                "artifact_note": MOCK_NOTE,
                "replay_status_code": replay_response.status_code,
                "replay_job_id": replay["job"]["job_id"],
                "conflict_status_code": conflict_response.status_code,
                "conflict_detail": conflict_response.json(),
            }
        )

        # 2. Execution error.
        response = _post_job(
            client,
            "demo-fail-0001",
            prompt="MOCK: intentional execution failure",
            timeout_seconds=30,
            duration=0.5,
            fail=True,
            fail_at=30,
        )
        response.raise_for_status()
        fail_job_id = response.json()["job"]["job_id"]
        fail_job, fail_events, fail_cursor = _wait_job(client, fail_job_id)
        if fail_job["status"] != "failed":
            raise RuntimeError(f"expected failed, got {fail_job['status']}")
        evidence["scenarios"].append(
            {
                "name": "execution_error",
                "job_id": fail_job_id,
                "status": fail_job["status"],
                "error": fail_job["error"],
                "events": fail_events,
                "next_cursor": fail_cursor,
            }
        )

        # 3. Cancellation after dispatch.
        response = _post_job(
            client,
            "demo-cancel-0001",
            prompt="MOCK: cancellation after dispatch",
            timeout_seconds=30,
            duration=5.0,
        )
        response.raise_for_status()
        cancel_job_id = response.json()["job"]["job_id"]
        cancel_job, cancel_events, cancel_cursor = _wait_job(
            client,
            cancel_job_id,
            until=lambda current: current["prompt_id"] is not None
            and current["status"] in {"submitted", "running"},
        )
        cancel_response = client.post(f"/jobs/{cancel_job_id}/cancel")
        cancel_response.raise_for_status()
        cancelled_job, late_events, cancel_cursor = _wait_job(
            client,
            cancel_job_id,
            cursor=cancel_cursor,
        )
        if cancelled_job["status"] != "cancelled":
            raise RuntimeError(
                f"expected cancelled, got {cancelled_job['status']}"
            )
        evidence["scenarios"].append(
            {
                "name": "cancellation_after_dispatch",
                "job_id": cancel_job_id,
                "status_before_cancel": cancel_job["status"],
                "cancel_response": cancel_response.json(),
                "status": cancelled_job["status"],
                "events": cancel_events + late_events,
                "next_cursor": cancel_cursor,
            }
        )

        # 4. Persistent deadline timeout.
        response = _post_job(
            client,
            "demo-timeout-0001",
            prompt="MOCK: persistent deadline timeout",
            timeout_seconds=1,
            duration=10.0,
        )
        response.raise_for_status()
        timeout_job_id = response.json()["job"]["job_id"]
        timeout_job, timeout_events, timeout_cursor = _wait_job(
            client, timeout_job_id, timeout=10.0
        )
        if timeout_job["status"] != "timed_out":
            raise RuntimeError(
                f"expected timed_out, got {timeout_job['status']}"
            )
        evidence["scenarios"].append(
            {
                "name": "persistent_timeout",
                "job_id": timeout_job_id,
                "status": timeout_job["status"],
                "error": timeout_job["error"],
                "events": timeout_events,
                "next_cursor": timeout_cursor,
            }
        )

        # 5. Restart while running; reconnect with the saved cursor.
        response = _post_job(
            client,
            "demo-restart-0001",
            prompt="MOCK: restart recovery and cursor replay",
            timeout_seconds=30,
            duration=4.0,
        )
        response.raise_for_status()
        restart_job_id = response.json()["job"]["job_id"]
        before_job, before_events, before_cursor = _wait_job(
            client,
            restart_job_id,
            until=lambda current: current["prompt_id"] is not None
            and current["status"] in {"submitted", "running"},
        )
        print("[demo] stopping server while the job is still running")
        server.stop()
        time.sleep(0.3)
        server2 = DemoServer(run_dir, 2)
        servers.append(server2)
        server2.start()
        client2 = server2.client
        assert client2 is not None
        after_job, after_events, after_cursor = _wait_job(
            client2,
            restart_job_id,
            cursor=before_cursor,
        )
        if after_job["status"] != "succeeded":
            raise RuntimeError(
                f"restart job ended as {after_job['status']}: {after_job}"
            )
        all_events = before_events + after_events
        remote_submitted_count = sum(
            1 for event in all_events if event["type"] == "remote_submitted"
        )
        dispatch_started_count = sum(
            1 for event in all_events if event["type"] == "dispatch_started"
        )
        if remote_submitted_count != 1:
            raise RuntimeError(
                "restart caused duplicate remote_submitted events: "
                f"{remote_submitted_count}"
            )
        evidence["scenarios"].append(
            {
                "name": "restart_and_cursor_replay",
                "job_id": restart_job_id,
                "status_before_restart": before_job["status"],
                "status_after_restart": after_job["status"],
                "cursor_before_restart": before_cursor,
                "cursor_after_restart": after_cursor,
                "remote_submitted_count": remote_submitted_count,
                "dispatch_started_count": dispatch_started_count,
                "events": all_events,
            }
        )

        evidence["finished_at"] = now_iso()
        evidence["server_logs"] = [str(item.log_path) for item in servers]
        evidence["ok"] = True
        evidence_path.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[demo] MOCK demo completed successfully; evidence: {evidence_path}")
        print(f"[demo] base dir: {base_dir.resolve()}")
        print(f"[demo] unique run dir: {run_dir}")
        return 0
    except Exception:
        evidence["finished_at"] = now_iso()
        evidence["ok"] = False
        evidence["error"] = traceback.format_exc()
        evidence["server_logs"] = [str(item.log_path) for item in servers]
        evidence_path.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[demo] FAILED; evidence: {evidence_path}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        for item in servers:
            try:
                item.stop()
            except Exception:
                pass
