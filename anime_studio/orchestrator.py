"""Durable job orchestration.

The orchestrator owns job/event state and the state machine. It delegates
remote ComfyUI work to a ComfyAdapter and file materialization to an
ArtifactStore. It never talks to ComfyUI or downloads files itself.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import Any

from .artifacts import ArtifactStore
from .comfy import ComfyAdapter, RemoteStatus
from .db import Database
from .errors import (
    ArtifactError,
    IdempotencyConflict,
    JobAlreadyTerminal,
    JobNotFound,
    RemoteCancelUnsupported,
    ResultNotReady,
)
from .models import JobRequest, canonical_payload
from .states import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    IllegalTransition,
    JobStatus,
    can_transition,
    is_terminal,
)
from .timeutils import ms_to_iso, now_iso, now_ms


MOCK_RESULT_NOTE = "MOCK RESULT - NOT A REAL VIDEO"


class Orchestrator:
    def __init__(
        self,
        db: Database,
        adapter: ComfyAdapter,
        artifacts: ArtifactStore,
        *,
        lease_seconds: float = 5.0,
    ) -> None:
        self.db = db
        self.adapter = adapter
        self.artifacts = artifacts
        self.lease_seconds = lease_seconds
        self.execution_mode = str(getattr(adapter, "mode", "UNKNOWN"))
        self.execution_fingerprint = str(
            getattr(adapter, "execution_fingerprint", self.execution_mode)
        )
        # Must happen before any worker starts or any network call is possible.
        self.db.bind_execution(self.execution_mode, self.execution_fingerprint)
        # Artifact materialization and terminal-result commit must be serialized
        # per job. The dict is intentionally small and process-local: P0 has one
        # API process and one worker.
        self._job_locks: dict[str, threading.RLock] = {}
        self._job_locks_guard = threading.Lock()

    def _job_lock(self, job_id: str) -> threading.RLock:
        with self._job_locks_guard:
            return self._job_locks.setdefault(job_id, threading.RLock())

    # ------------------------------------------------------------------ reads
    def _get_row(self, conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
        return conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()

    def _mode(self) -> str:
        return self.execution_mode

    def _is_mock(self) -> bool:
        return self._mode() == "MOCK"

    @staticmethod
    def _row_mode(row: sqlite3.Row) -> str:
        return str(row["execution_mode"] or "MOCK")

    @staticmethod
    def _row_fingerprint(row: sqlite3.Row) -> str:
        return str(row["execution_fingerprint"] or "MOCK")

    def _row_to_job(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_canonical"])
        result = json.loads(row["result_json"]) if row["result_json"] else None
        error = None
        if row["error_code"]:
            error = {"code": row["error_code"], "message": row["error_message"]}
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "progress": int(row["progress"]),
            "prompt_id": row["prompt_id"],
            "correlation_key": row["correlation_key"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deadline_at": row["deadline_at"],
            "deadline_ts": int(row["deadline_ts"]),
            "timeout_seconds": int(row["timeout_seconds"]),
            "cancel_requested": row["cancel_requested_at"] is not None,
            "cancel_requested_at": row["cancel_requested_at"],
            "dispatch_started_at": row["dispatch_started_at"],
            "terminal_at": row["terminal_at"],
            "error": error,
            "result": result,
            "prompt": payload.get("prompt"),
            "seed": payload.get("seed"),
            "mock_options": payload.get("mock_options"),
            "payload_sha256": row["payload_sha256"],
            "mock": self._row_mode(row) == "MOCK",
            "mode": self._row_mode(row),
            "execution_mode": self._row_mode(row),
            "execution_fingerprint": self._row_fingerprint(row),
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            row = self._get_row(conn, job_id)
            return self._row_to_job(row) if row else None

    def get_events(
        self, job_id: str, *, after: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        with self.db.snapshot() as conn:
            row = self._get_row(conn, job_id)
            if row is None:
                raise JobNotFound(f"Unknown job: {job_id}")
            rows = conn.execute(
                """
                SELECT * FROM job_events
                 WHERE job_id=? AND event_id>?
                 ORDER BY event_id ASC
                 LIMIT ?
                """,
                (job_id, after, limit + 1),
            ).fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            events = [
                {
                    "cursor": int(event["event_id"]),
                    "seq": int(event["seq"]),
                    "type": event["type"],
                    "at": event["at"],
                    "data": json.loads(event["data_json"]),
                }
                for event in rows
            ]
            next_cursor = events[-1]["cursor"] if events else after
            return {
                "job_id": job_id,
                "status": row["status"],
                "events": events,
                "next_cursor": next_cursor,
                "has_more": has_more,
            }

    def get_result(self, job_id: str) -> dict[str, Any]:
        row = self.get_job(job_id)
        if row is None:
            raise JobNotFound(f"Unknown job: {job_id}")
        if row["status"] != JobStatus.SUCCEEDED.value:
            raise ResultNotReady(row["status"])
        return row["result"] or {}

    # --------------------------------------------------------------- creation
    def create_job(
        self, request: JobRequest, idempotency_key: str
    ) -> tuple[dict[str, Any], bool]:
        canonical, digest = canonical_payload(request)
        current_ms = now_ms()
        deadline_ts = current_ms + request.timeout_seconds * 1000
        deadline_at = ms_to_iso(deadline_ts)
        job_id = str(uuid.uuid4())
        correlation_key = f"corr-{uuid.uuid4().hex}"

        with self.db.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                if existing["payload_sha256"] != digest:
                    raise IdempotencyConflict(existing["job_id"])
                return self._row_to_job(existing), False

            try:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, idempotency_key, payload_canonical, payload_sha256,
                        status, progress, execution_mode, execution_fingerprint,
                        correlation_key, created_at, created_ms,
                        updated_at, updated_ms, deadline_at, deadline_ts,
                        timeout_seconds
                    ) VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        idempotency_key,
                        canonical,
                        digest,
                        self.execution_mode,
                        self.execution_fingerprint,
                        correlation_key,
                        now_iso(),
                        current_ms,
                        now_iso(),
                        current_ms,
                        deadline_at,
                        deadline_ts,
                        request.timeout_seconds,
                    ),
                )
            except sqlite3.IntegrityError:
                existing = conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if existing is None:
                    raise
                if existing["payload_sha256"] != digest:
                    raise IdempotencyConflict(existing["job_id"])
                return self._row_to_job(existing), False

            self._record_event(
                conn,
                job_id,
                "job_submitted",
                {
                    "payload_sha256": digest,
                    "timeout_seconds": request.timeout_seconds,
                    "deadline_at": deadline_at,
                    "mode": self._mode(),
                    "mock": self._is_mock(),
                    "execution_fingerprint": self.execution_fingerprint,
                },
            )
            row = self._get_row(conn, job_id)
            assert row is not None
            return self._row_to_job(row), True

    # --------------------------------------------------------------- helpers
    def _record_event(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> int:
        seq = int(
            conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM job_events WHERE job_id=?",
                (job_id,),
            ).fetchone()[0]
        )
        current_ms = now_ms()
        conn.execute(
            """
            INSERT INTO job_events (job_id, seq, type, at, at_ms, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                seq,
                event_type,
                ms_to_iso(current_ms),
                current_ms,
                json.dumps(data or {}, sort_keys=True, ensure_ascii=False, default=str),
            ),
        )
        return seq

    def _apply_updates(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        updates: dict[str, Any],
        *,
        touch: bool = True,
    ) -> None:
        if not updates and not touch:
            return
        fields = dict(updates)
        if touch:
            fields["updated_at"] = now_iso()
            fields["updated_ms"] = now_ms()
        columns = list(fields)
        sql = "UPDATE jobs SET " + ", ".join(f"{column}=?" for column in columns)
        sql += " WHERE job_id=?"
        conn.execute(sql, [*(fields[column] for column in columns), job_id])

    def _transition(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        target: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        updates: dict[str, Any] | None = None,
    ) -> sqlite3.Row | None:
        row = self._get_row(conn, job_id)
        if row is None:
            return None
        current = row["status"]
        if is_terminal(current):
            return None
        if not can_transition(current, target):
            raise IllegalTransition(f"{current} -> {target} is not allowed")
        fields = dict(updates or {})
        fields["status"] = target
        if is_terminal(target):
            fields["terminal_at"] = now_iso()
        self._apply_updates(conn, job_id, fields)
        self._record_event(conn, job_id, event_type, data)
        return self._get_row(conn, job_id)

    def _record_late_event(self, job_id: str, remote: RemoteStatus) -> None:
        with self.db.transaction() as conn:
            row = self._get_row(conn, job_id)
            if row is None:
                return
            self._record_event(
                conn,
                job_id,
                "late_event_ignored",
                {
                    "prompt_id": remote.prompt_id,
                    "remote_state": remote.state,
                    "remote_progress": remote.progress,
                    "job_status": row["status"],
                    "reason": "terminal states are immutable",
                },
            )

    # -------------------------------------------------------------- dispatch
    def link_remote(
        self,
        job_id: str,
        remote: RemoteStatus,
        *,
        event_type: str = "remote_submitted",
        apply_result: bool = True,
    ) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            row = self._get_row(conn, job_id)
            if row is None:
                return None
            if is_terminal(row["status"]):
                self._record_event(
                    conn,
                    job_id,
                    "late_event_ignored",
                    {
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "job_status": row["status"],
                    },
                )
                return self._row_to_job(row)
            if row["prompt_id"] and row["prompt_id"] != remote.prompt_id:
                self._record_event(
                    conn,
                    job_id,
                    "prompt_mismatch_ignored",
                    {
                        "stored_prompt_id": row["prompt_id"],
                        "received_prompt_id": remote.prompt_id,
                    },
                )
                return self._row_to_job(row)

            # The deadline is checked again at the final commit boundary. If it
            # expired after the caller's decision, preserve identity only.
            effective_apply = apply_result and int(row["deadline_ts"]) > now_ms()
            progress = (
                max(int(row["progress"]), int(remote.progress))
                if effective_apply
                else int(row["progress"])
            )
            target = row["status"]
            if remote.state == "running":
                target = JobStatus.RUNNING.value
            elif row["status"] == JobStatus.DISPATCHING.value:
                # A queued/succeeded/failed/cancelled remote response means the
                # prompt_id was accepted and saved; dispatching is finished.
                target = JobStatus.SUBMITTED.value
            updates = {
                "prompt_id": remote.prompt_id,
                "remote_state": remote.state,
                "progress": progress,
            }
            if target != row["status"] and can_transition(row["status"], target):
                fields = dict(updates)
                fields["status"] = target
                self._apply_updates(conn, job_id, fields)
                self._record_event(
                    conn,
                    job_id,
                    event_type,
                    {
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "progress": progress,
                        "correlation_key": row["correlation_key"],
                        "result_applied": effective_apply,
                        "diagnostics": remote.diagnostics,
                    },
                )
            else:
                self._apply_updates(conn, job_id, updates)
                self._record_event(
                    conn,
                    job_id,
                    event_type,
                    {
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "progress": progress,
                        "correlation_key": row["correlation_key"],
                        "result_applied": effective_apply,
                        "diagnostics": remote.diagnostics,
                    },
                )
            updated = self._get_row(conn, job_id)

        # Reconcile identity only: the caller is on the deadline path and will
        # perform best-effort cancellation and mark timed_out. A remote result
        # must never be applied merely because prompt_id was discovered.
        if not effective_apply:
            return self.get_job(job_id)
        if is_terminal(remote.state):
            return self.apply_remote_state(job_id, remote)
        if updated is not None and updated["cancel_requested_at"] is not None:
            return self.cancel_remote(job_id)
        return self.get_job(job_id)

    def apply_remote_state(
        self, job_id: str, remote: RemoteStatus
    ) -> dict[str, Any] | None:
        # Completion and cancellation can arrive concurrently. Serialize the
        # per-job finalization so only one caller materializes the artifact and
        # commits a terminal result. Artifact IO happens outside SQLite
        # transactions, under this lock.
        with self._job_lock(job_id):
            return self._apply_remote_state_locked(job_id, remote)

    def _apply_remote_state_locked(
        self, job_id: str, remote: RemoteStatus
    ) -> dict[str, Any] | None:
        row = self.get_job(job_id)
        if row is None:
            return None
        if row["status"] in TERMINAL_STATUSES:
            self._record_late_event(job_id, remote)
            return row
        if int(row["deadline_ts"]) <= now_ms():
            # A remote success/failure/cancel that is observed after the
            # persisted deadline must not be committed as a successful result.
            # Re-check immediately before any artifact IO and terminal commit.
            prompt_id = row["prompt_id"]
            cancel_attempted = False
            remote_state = remote.state
            if prompt_id:
                cancel_attempted = True
                try:
                    cancelled = self.adapter.cancel(prompt_id)
                    if cancelled is not None:
                        remote_state = cancelled.state
                except Exception:
                    pass
            return self.mark_timeout(
                job_id,
                deadline_at=row["deadline_at"],
                had_prompt_id=prompt_id is not None,
                cancel_attempted=cancel_attempted,
                remote_state=remote_state,
            )
        if row["prompt_id"] and row["prompt_id"] != remote.prompt_id:
            with self.db.transaction() as conn:
                current = self._get_row(conn, job_id)
                if current is not None:
                    self._record_event(
                        conn,
                        job_id,
                        "prompt_mismatch_ignored",
                        {
                            "stored_prompt_id": current["prompt_id"],
                            "received_prompt_id": remote.prompt_id,
                        },
                    )
            return self.get_job(job_id)

        if remote.state == "succeeded":
            if not remote.artifact:
                return self.fail_job(
                    job_id,
                    "RESULT_MISSING_ARTIFACT",
                    "Remote success did not include a materializable artifact reference; "
                    "empty artifacts are not success.",
                    event_type="job_failed",
                    data={"stage": "artifact", "remote_state": remote.state},
                    prompt_id=remote.prompt_id,
                )
            try:
                artifact_record = self.artifacts.fetch_result(remote.artifact, job_id)
            except ArtifactError as exc:
                return self.fail_job(
                    job_id,
                    exc.code,
                    str(exc),
                    event_type="job_failed",
                    data={"stage": "artifact"},
                )
            except OSError as exc:
                return self.fail_job(
                    job_id,
                    "ARTIFACT_IO_ERROR",
                    f"Artifact IO failed: {exc}",
                    event_type="job_failed",
                    data={"stage": "artifact"},
                )
            if not artifact_record:
                return self.fail_job(
                    job_id,
                    "RESULT_MISSING_ARTIFACT",
                    "Artifact store returned no materialized result.",
                    event_type="job_failed",
                    data={"stage": "artifact"},
                    prompt_id=remote.prompt_id,
                )
            row_mode = self._row_mode(row)
            is_mock = row_mode == "MOCK"
            result = {
                "mock": is_mock,
                "mode": row_mode,
                "execution_fingerprint": self._row_fingerprint(row),
                "prompt_id": remote.prompt_id,
                "correlation_key": row["correlation_key"],
                "artifacts": [artifact_record],
                "remote_state": remote.state,
            }
            if is_mock:
                result["note"] = MOCK_RESULT_NOTE
            else:
                result["note"] = (
                    "Real ComfyUI result materialized offline/structurally. "
                    "ffprobe decode is a live check and is NOT_RUN until P1 smoke."
                )
            expired_at_commit = False
            current = None
            with self.db.transaction() as conn:
                current = self._get_row(conn, job_id)
                if current is None:
                    return None
                if is_terminal(current["status"]):
                    self._record_event(
                        conn,
                        job_id,
                        "late_event_ignored",
                        {
                            "prompt_id": remote.prompt_id,
                            "remote_state": remote.state,
                            "job_status": current["status"],
                        },
                    )
                    return self._row_to_job(current)
                if int(current["deadline_ts"]) <= now_ms():
                    # Artifact IO may have crossed the deadline. Do not commit
                    # a success result after the persisted deadline.
                    expired_at_commit = True
                else:
                    updates = {
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "progress": 100,
                        "result_json": json.dumps(
                            result, sort_keys=True, ensure_ascii=False
                        ),
                    }
                    event_data = {
                        "prompt_id": remote.prompt_id,
                        "mock": self._row_mode(current) == "MOCK",
                        "mode": self._row_mode(current),
                        "execution_fingerprint": self._row_fingerprint(current),
                        "artifacts": [artifact_record],
                        "cancel_too_late": current["cancel_requested_at"] is not None,
                    }
                    updated = self._transition(
                        conn,
                        job_id,
                        JobStatus.SUCCEEDED.value,
                        "job_succeeded",
                        event_data,
                        updates=updates,
                    )
                    if (
                        updated is not None
                        and current["cancel_requested_at"] is not None
                    ):
                        self._record_event(
                            conn,
                            job_id,
                            "cancel_too_late",
                            {
                                "prompt_id": remote.prompt_id,
                                "status": updated["status"],
                            },
                        )
                    return (
                        self._row_to_job(updated)
                        if updated
                        else self.get_job(job_id)
                    )
            if expired_at_commit and current is not None:
                cancel_attempted = False
                if current["prompt_id"]:
                    cancel_attempted = True
                    try:
                        self.adapter.cancel(current["prompt_id"])
                    except Exception:
                        pass
                return self.mark_timeout(
                    job_id,
                    deadline_at=current["deadline_at"],
                    had_prompt_id=current["prompt_id"] is not None,
                    cancel_attempted=cancel_attempted,
                    remote_state=remote.state,
                )
            return self.get_job(job_id)

        if remote.state == "failed":
            return self.fail_job(
                job_id,
                remote.error_code or "REMOTE_EXECUTION_ERROR",
                remote.error_message or "Remote ComfyUI execution failed.",
                event_type="job_failed",
                data={
                    "prompt_id": remote.prompt_id,
                    "remote_state": remote.state,
                    "progress": remote.progress,
                },
                progress=remote.progress,
                prompt_id=remote.prompt_id,
            )

        if remote.state == "cancelled":
            with self.db.transaction() as conn:
                current = self._get_row(conn, job_id)
                if current is None:
                    return None
                if is_terminal(current["status"]):
                    self._record_event(
                        conn,
                        job_id,
                        "late_event_ignored",
                        {
                            "prompt_id": remote.prompt_id,
                            "remote_state": remote.state,
                            "job_status": current["status"],
                        },
                    )
                    return self._row_to_job(current)
                updated = self._transition(
                    conn,
                    job_id,
                    JobStatus.CANCELLED.value,
                    "job_cancelled",
                    {
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "reason": "remote_cancelled",
                    },
                    updates={
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "progress": max(int(current["progress"]), int(remote.progress)),
                    },
                )
                return self._row_to_job(updated) if updated else self.get_job(job_id)

        # queued or running
        with self.db.transaction() as conn:
            current = self._get_row(conn, job_id)
            if current is None:
                return None
            if is_terminal(current["status"]):
                self._record_event(
                    conn,
                    job_id,
                    "late_event_ignored",
                    {
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "job_status": current["status"],
                    },
                )
                return self._row_to_job(current)
            progress = max(int(current["progress"]), min(100, int(remote.progress)))
            updates = {
                "prompt_id": remote.prompt_id,
                "remote_state": remote.state,
                "progress": progress,
            }
            target = current["status"]
            if current["status"] in {
                JobStatus.DISPATCHING.value,
                JobStatus.SUBMITTED.value,
            }:
                if remote.state == "running" or progress > 0:
                    target = JobStatus.RUNNING.value
                else:
                    target = JobStatus.SUBMITTED.value
            changed = False
            if target != current["status"] and can_transition(
                current["status"], target
            ):
                fields = dict(updates)
                fields["status"] = target
                self._apply_updates(conn, job_id, fields)
                self._record_event(
                    conn,
                    job_id,
                    "job_running" if target == JobStatus.RUNNING.value else "job_submitted_remote",
                    {
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "progress": progress,
                    },
                )
                changed = True
            elif progress != int(current["progress"]):
                self._apply_updates(conn, job_id, updates)
                self._record_event(
                    conn,
                    job_id,
                    "progress",
                    {
                        "prompt_id": remote.prompt_id,
                        "remote_state": remote.state,
                        "progress": progress,
                    },
                )
                changed = True
            if not changed:
                self._apply_updates(conn, job_id, updates)
            return self.get_job(job_id)

    def fail_job(
        self,
        job_id: str,
        code: str,
        message: str,
        *,
        event_type: str = "job_failed",
        data: dict[str, Any] | None = None,
        progress: int | None = None,
        prompt_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            current = self._get_row(conn, job_id)
            if current is None:
                return None
            if is_terminal(current["status"]):
                return self._row_to_job(current)
            updates: dict[str, Any] = {
                "error_code": code,
                "error_message": message,
            }
            if progress is not None:
                updates["progress"] = max(int(current["progress"]), int(progress))
            if prompt_id is not None:
                updates["prompt_id"] = prompt_id
                updates["remote_state"] = "failed"
            updated = self._transition(
                conn,
                job_id,
                JobStatus.FAILED.value,
                event_type,
                {
                    "code": code,
                    "message": message,
                    **(data or {}),
                },
                updates=updates,
            )
            return self._row_to_job(updated) if updated else self.get_job(job_id)

    def mark_ambiguous(self, job_id: str, reason: str) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            current = self._get_row(conn, job_id)
            if current is None:
                return None
            if is_terminal(current["status"]):
                return self._row_to_job(current)
            message = (
                "Dispatch outcome is ambiguous: the remote side may have accepted "
                "the job while the local prompt_id was not saved. The real ComfyUI "
                "adapter cannot prove exactly-once, so remote absence/cancellation "
                "is NOT confirmed and the job is not blindly resubmitted. "
                "Manual reconciliation is required before any retry. Reason: "
                + reason
            )
            updated = self._transition(
                conn,
                job_id,
                JobStatus.FAILED.value,
                "dispatch_ambiguous",
                {
                    "reason": reason,
                    "correlation_key": current["correlation_key"],
                    "do_not_blindly_resubmit": True,
                    "message": message,
                },
                updates={
                    "error_code": "AMBIGUOUS_DISPATCH",
                    "error_message": message,
                    "ambiguity_reason": reason,
                },
            )
            return self._row_to_job(updated) if updated else self.get_job(job_id)

    def mark_timeout(
        self,
        job_id: str,
        *,
        deadline_at: str,
        had_prompt_id: bool,
        cancel_attempted: bool,
        remote_state: str | None,
        extra_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            current = self._get_row(conn, job_id)
            if current is None:
                return None
            if is_terminal(current["status"]):
                return self._row_to_job(current)
            updated = self._transition(
                conn,
                job_id,
                JobStatus.TIMED_OUT.value,
                "job_timed_out",
                {
                    "deadline_at": deadline_at,
                    "had_prompt_id": had_prompt_id,
                    "cancel_attempted": cancel_attempted,
                    "remote_state": remote_state,
                    **(extra_data or {}),
                },
                updates={
                    "error_code": "DEADLINE_EXCEEDED",
                    "error_message": "The persisted absolute deadline was exceeded.",
                    "remote_state": remote_state,
                },
            )
            return self._row_to_job(updated) if updated else self.get_job(job_id)

    # -------------------------------------------------------------- cancel
    def request_cancel(self, job_id: str) -> dict[str, Any]:
        prompt_id: str | None = None
        with self.db.transaction() as conn:
            row = self._get_row(conn, job_id)
            if row is None:
                raise JobNotFound(f"Unknown job: {job_id}")
            status = row["status"]
            if is_terminal(status):
                if status == JobStatus.CANCELLED.value:
                    return self._row_to_job(row)
                raise JobAlreadyTerminal(status)

            if status == JobStatus.QUEUED.value:
                updated = self._transition(
                    conn,
                    job_id,
                    JobStatus.CANCELLED.value,
                    "job_cancelled",
                    {"reason": "cancel_before_dispatch"},
                    updates={"cancel_requested_at": now_iso()},
                )
                return self._row_to_job(updated) if updated else self.get_job(job_id)

            if row["cancel_requested_at"] is None:
                self._apply_updates(
                    conn,
                    job_id,
                    {"cancel_requested_at": now_iso()},
                )
                self._record_event(
                    conn,
                    job_id,
                    "cancel_requested",
                    {"status": status, "has_prompt_id": row["prompt_id"] is not None},
                )
            prompt_id = row["prompt_id"]

        if prompt_id is None:
            # dispatching before prompt_id: the worker must decide whether it
            # can still prevent submission or must cancel the remote job.
            return self.get_job(job_id) or {}

        try:
            remote = self.adapter.cancel(prompt_id)
        except RemoteCancelUnsupported as exc:
            with self.db.transaction() as conn:
                current = self._get_row(conn, job_id)
                if current is not None and not is_terminal(current["status"]):
                    self._record_event(
                        conn,
                        job_id,
                        "cancel_unconfirmed",
                        {
                            "prompt_id": prompt_id,
                            "error_code": exc.code,
                            "message": exc.message,
                            "remote_absent_confirmed": False,
                        },
                    )
            return self.get_job(job_id) or {}
        except Exception as exc:  # adapter errors are explicit, not hidden
            with self.db.transaction() as conn:
                current = self._get_row(conn, job_id)
                if current is not None and not is_terminal(current["status"]):
                    self._record_event(
                        conn,
                        job_id,
                        "cancel_failed",
                        {"prompt_id": prompt_id, "error": str(exc)},
                    )
            return self.get_job(job_id) or {}

        if remote is None:
            with self.db.transaction() as conn:
                current = self._get_row(conn, job_id)
                if current is not None and not is_terminal(current["status"]):
                    self._record_event(
                        conn,
                        job_id,
                        "cancel_remote_missing",
                        {"prompt_id": prompt_id},
                    )
            return self.get_job(job_id) or {}

        return self.apply_remote_state(job_id, remote) or self.get_job(job_id) or {}

    def request_cancel_after_dispatch(
        self, job_id: str
    ) -> dict[str, Any] | None:
        """Cancel a job that the worker has claimed but may not have submitted.

        If prompt_id is already present, use the normal remote cancellation path.
        Otherwise mark the job cancelled before any remote submit. A race where
        prompt_id appears between reads falls back to request_cancel().
        """

        row = self.get_job(job_id)
        if row is None or row["status"] in TERMINAL_STATUSES:
            return row
        if row["prompt_id"]:
            return self.request_cancel(job_id)

        need_remote_cancel = False
        with self.db.transaction() as conn:
            current = self._get_row(conn, job_id)
            if current is None:
                return None
            if is_terminal(current["status"]):
                return self._row_to_job(current)
            if current["prompt_id"] is not None:
                need_remote_cancel = True
            else:
                updated = self._transition(
                    conn,
                    job_id,
                    JobStatus.CANCELLED.value,
                    "job_cancelled",
                    {"reason": "cancel_before_remote_submit"},
                )
                return self._row_to_job(updated) if updated else self.get_job(job_id)
        if need_remote_cancel:
            return self.request_cancel(job_id)
        return self.get_job(job_id)

    def cancel_remote(self, job_id: str) -> dict[str, Any] | None:
        row = self.get_job(job_id)
        if row is None or row["status"] in TERMINAL_STATUSES:
            return row
        prompt_id = row["prompt_id"]
        if not prompt_id:
            return row
        try:
            remote = self.adapter.cancel(prompt_id)
        except RemoteCancelUnsupported as exc:
            with self.db.transaction() as conn:
                current = self._get_row(conn, job_id)
                if current is not None and not is_terminal(current["status"]):
                    self._record_event(
                        conn,
                        job_id,
                        "cancel_unconfirmed",
                        {
                            "prompt_id": prompt_id,
                            "error_code": exc.code,
                            "message": exc.message,
                            "remote_absent_confirmed": False,
                        },
                    )
            return self.get_job(job_id)
        except Exception as exc:
            with self.db.transaction() as conn:
                current = self._get_row(conn, job_id)
                if current is not None and not is_terminal(current["status"]):
                    self._record_event(
                        conn,
                        job_id,
                        "cancel_failed",
                        {"prompt_id": prompt_id, "error": str(exc)},
                    )
            return self.get_job(job_id)
        if remote is None:
            return self.get_job(job_id)
        return self.apply_remote_state(job_id, remote)


__all__ = ["Orchestrator", "ALLOWED_TRANSITIONS"]
