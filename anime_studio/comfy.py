"""ComfyUI adapter boundary and a durable, fully offline mock transport.

The orchestrator only knows this interface. The mock implementation persists
its remote-side state in SQLite so a worker crash between "remote accepted" and
"prompt_id saved locally" can be reconciled by the stable correlation key.

A real ComfyUI adapter is intentionally not implemented in P0. Real ComfyUI
does not expose a proven exactly-once submission primitive; a real adapter must
therefore set supports_reconcile_by_correlation_key=False unless it has its own
durable idempotency mechanism. Blind resubmission is forbidden.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .db import Database
from .errors import AdapterError
from .timeutils import ms_to_iso, now_ms


MOCK_NOTE = "MOCK ARTIFACT - NOT A REAL VIDEO"


@dataclass(frozen=True)
class RemoteStatus:
    prompt_id: str
    correlation_key: str
    state: str
    progress: int
    error_code: str | None = None
    error_message: str | None = None
    artifact: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    updated_at: str = ""


class ComfyAdapter(Protocol):
    mode: str
    supports_reconcile_by_correlation_key: bool
    supports_idempotent_submit: bool

    def submit(
        self,
        correlation_key: str,
        payload: dict[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        ...

    def get(self, prompt_id: str) -> RemoteStatus | None:
        ...

    def cancel(self, prompt_id: str) -> RemoteStatus | None:
        ...

    def reconcile(self, correlation_key: str) -> RemoteStatus | None:
        ...


class MockComfyTransport:
    mode = "MOCK"
    execution_fingerprint = "MOCK"
    supports_reconcile_by_correlation_key = True
    supports_idempotent_submit = True

    REMOTE_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})

    def __init__(self, db: Database) -> None:
        self.db = db
        self.db.init()

    def _row_to_status(self, row: sqlite3.Row) -> RemoteStatus:
        artifact = json.loads(row["artifact_json"]) if row["artifact_json"] else None
        return RemoteStatus(
            prompt_id=row["prompt_id"],
            correlation_key=row["correlation_key"],
            state=row["state"],
            progress=int(row["progress"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            artifact=artifact,
            updated_at=ms_to_iso(int(row["updated_ms"])),
        )

    def _artifact_for(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        return {
            "scheme": "mock",
            "mock": True,
            "prompt_id": row["prompt_id"],
            "content": {
                "mock": True,
                "note": MOCK_NOTE,
                "prompt_id": row["prompt_id"],
                "prompt": payload.get("prompt"),
                "seed": payload.get("seed"),
                "generated_at": ms_to_iso(int(row["updated_ms"])),
            },
        }

    def _advance(self, conn: sqlite3.Connection, row: sqlite3.Row) -> sqlite3.Row:
        if row["state"] in self.REMOTE_TERMINAL:
            return row

        current_ms = now_ms()
        current_ts = time.time()
        state = row["state"]
        progress = int(row["progress"])
        error_code = row["error_code"]
        error_message = row["error_message"]
        artifact_json = row["artifact_json"]

        if row["cancelled_at_ms"] is not None:
            state = "cancelled"
            progress = max(progress, 0)
        else:
            duration = max(0.1, float(row["duration_seconds"]))
            elapsed = max(0.0, current_ts - float(row["started_ts"]))
            computed = min(100, int((elapsed / duration) * 100))
            progress = max(progress, computed)
            if bool(row["fail"]) and progress >= int(row["fail_at_progress"]):
                state = "failed"
                progress = max(progress, int(row["fail_at_progress"]))
                error_code = "MOCK_EXECUTION_ERROR"
                error_message = (
                    "Mock execution failed at the configured progress threshold; "
                    "this is an intentional OFFLINE test error."
                )
            elif progress >= 100:
                state = "succeeded"
                progress = 100
                artifact_json = json.dumps(
                    self._artifact_for(row), sort_keys=True, ensure_ascii=False
                )
            elif progress > 0:
                state = "running"

        conn.execute(
            """
            UPDATE mock_remote_jobs
               SET state=?, progress=?, updated_ms=?, error_code=?,
                   error_message=?, artifact_json=?
             WHERE correlation_key=?
            """,
            (
                state,
                progress,
                current_ms,
                error_code,
                error_message,
                artifact_json,
                row["correlation_key"],
            ),
        )
        return conn.execute(
            "SELECT * FROM mock_remote_jobs WHERE correlation_key=?",
            (row["correlation_key"],),
        ).fetchone()

    def submit(
        self,
        correlation_key: str,
        payload: dict[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        # The mock transport has no remote upload and therefore no deadline
        # window between an upload and a prompt send. deadline_ms is accepted
        # for ComfyAdapter compatibility and intentionally ignored here.
        del deadline_ms
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        opts = payload.get("mock_options") or {}
        duration = float(opts.get("duration_seconds", 1.0))
        fail = 1 if opts.get("fail") else 0
        fail_at = int(opts.get("fail_at_progress", 50))
        prompt_id = f"mock-{uuid.uuid4().hex}"
        current_ms = now_ms()
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mock_remote_jobs WHERE correlation_key=?",
                (correlation_key,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO mock_remote_jobs (
                        correlation_key, prompt_id, payload_json, state, progress,
                        created_ms, updated_ms, started_ts, duration_seconds,
                        fail, fail_at_progress
                    ) VALUES (?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        correlation_key,
                        prompt_id,
                        payload_json,
                        current_ms,
                        current_ms,
                        time.time(),
                        max(0.1, duration),
                        fail,
                        min(100, max(1, fail_at)),
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM mock_remote_jobs WHERE correlation_key=?",
                    (correlation_key,),
                ).fetchone()
            row = self._advance(conn, row)
            return self._row_to_status(row)

    def get(self, prompt_id: str) -> RemoteStatus | None:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mock_remote_jobs WHERE prompt_id=?", (prompt_id,)
            ).fetchone()
            if row is None:
                return None
            row = self._advance(conn, row)
            return self._row_to_status(row)

    def cancel(self, prompt_id: str) -> RemoteStatus | None:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mock_remote_jobs WHERE prompt_id=?", (prompt_id,)
            ).fetchone()
            if row is None:
                return None
            if row["state"] in self.REMOTE_TERMINAL:
                return self._row_to_status(row)
            current_ms = now_ms()
            conn.execute(
                """
                UPDATE mock_remote_jobs
                   SET state='cancelled', cancelled_at_ms=?, updated_ms=?
                 WHERE prompt_id=?
                """,
                (current_ms, current_ms, prompt_id),
            )
            row = conn.execute(
                "SELECT * FROM mock_remote_jobs WHERE prompt_id=?", (prompt_id,)
            ).fetchone()
            return self._row_to_status(row)

    def reconcile(self, correlation_key: str) -> RemoteStatus | None:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM mock_remote_jobs WHERE correlation_key=?",
                (correlation_key,),
            ).fetchone()
            if row is None:
                return None
            row = self._advance(conn, row)
            return self._row_to_status(row)

    def count_remote_jobs(self) -> int:
        with self.db.read() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM mock_remote_jobs").fetchone()[0])


class RealComfyAdapterNotConfigured:
    """Explicit fail-fast placeholder. No network or credentials are touched."""

    mode = "REAL_NOT_CONFIGURED"
    supports_reconcile_by_correlation_key = False
    supports_idempotent_submit = False

    def _raise(self) -> None:
        raise AdapterError(
            "Real ComfyUI adapter is not implemented in P0 and must not be used "
            "until P1 approves a worker, credentials and budget.",
            code="REAL_COMFY_ADAPTER_NOT_CONFIGURED",
        )

    def submit(
        self,
        correlation_key: str,
        payload: dict[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> RemoteStatus:
        del deadline_ms
        self._raise()
        raise AssertionError("unreachable")

    def get(self, prompt_id: str) -> RemoteStatus | None:
        self._raise()
        raise AssertionError("unreachable")

    def cancel(self, prompt_id: str) -> RemoteStatus | None:
        self._raise()
        raise AssertionError("unreachable")

    def reconcile(self, correlation_key: str) -> RemoteStatus | None:
        return None
