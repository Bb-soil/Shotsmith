"""SQLite schema and transaction helpers. Uses stdlib sqlite3 only."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import ExecutionBindingMismatch
from .timeutils import now_iso


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_canonical TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    execution_mode TEXT NOT NULL DEFAULT 'MOCK',
    execution_fingerprint TEXT NOT NULL DEFAULT 'MOCK',
    correlation_key TEXT NOT NULL UNIQUE,
    prompt_id TEXT,
    remote_state TEXT,
    result_json TEXT,
    error_code TEXT,
    error_message TEXT,
    ambiguity_reason TEXT,
    created_at TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    updated_ms INTEGER NOT NULL,
    deadline_at TEXT NOT NULL,
    deadline_ts INTEGER NOT NULL,
    timeout_seconds INTEGER NOT NULL,
    cancel_requested_at TEXT,
    dispatch_started_at TEXT,
    lease_owner TEXT,
    lease_expires_at_ms INTEGER,
    terminal_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_created_idx
    ON jobs(status, created_ms);
CREATE INDEX IF NOT EXISTS jobs_deadline_idx
    ON jobs(status, deadline_ts);
CREATE INDEX IF NOT EXISTS jobs_dispatching_lease_idx
    ON jobs(status, lease_expires_at_ms);

CREATE TABLE IF NOT EXISTS db_binding (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    mode TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    at TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    UNIQUE(job_id, seq)
);
CREATE INDEX IF NOT EXISTS job_events_job_cursor_idx
    ON job_events(job_id, event_id);

CREATE TABLE IF NOT EXISTS mock_remote_jobs (
    correlation_key TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    progress INTEGER NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    started_ts REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    fail INTEGER NOT NULL DEFAULT 0,
    fail_at_progress INTEGER NOT NULL DEFAULT 100,
    error_code TEXT,
    error_message TEXT,
    cancelled_at_ms INTEGER,
    artifact_json TEXT
);
CREATE INDEX IF NOT EXISTS mock_remote_prompt_idx
    ON mock_remote_jobs(prompt_id);
"""


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def init(self) -> None:
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "execution_mode" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN execution_mode TEXT NOT NULL DEFAULT 'MOCK'"
                )
            if "execution_fingerprint" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN execution_fingerprint TEXT NOT NULL DEFAULT 'MOCK'"
                )
        finally:
            conn.close()

    def bind_execution(self, mode: str, fingerprint: str) -> None:
        """Bind this database to one execution identity, or fail fast.

        A legacy P0 database has no binding row. Its jobs were created with
        MOCK defaults, so it may only open in MOCK mode. A real binding is
        written only for a new/empty database. Existing bindings are immutable
        and are never silently updated.
        """

        with self.transaction() as conn:
            row = conn.execute(
                "SELECT mode, fingerprint FROM db_binding WHERE singleton=1"
            ).fetchone()
            if row is None:
                job_count = int(
                    conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                )
                if job_count and mode != "MOCK":
                    raise ExecutionBindingMismatch(
                        "Legacy P0 database has jobs without execution binding; "
                        "it is treated as MOCK and cannot be opened with a real "
                        "adapter."
                    )
                now = now_iso()
                conn.execute(
                    """
                    INSERT INTO db_binding (
                        singleton, mode, fingerprint, created_at, updated_at
                    ) VALUES (1, ?, ?, ?, ?)
                    """,
                    (mode, fingerprint, now, now),
                )
                row = {"mode": mode, "fingerprint": fingerprint}
            if row["mode"] != mode or row["fingerprint"] != fingerprint:
                raise ExecutionBindingMismatch(
                    "Database execution binding is immutable: stored "
                    f"{row['mode']}/{row['fingerprint'][:16]}..., requested "
                    f"{mode}/{fingerprint[:16]}... . Refusing to rebind."
                )
            mismatched = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM jobs
                     WHERE execution_mode != ? OR execution_fingerprint != ?
                    """,
                    (mode, fingerprint),
                ).fetchone()[0]
            )
            if mismatched:
                raise ExecutionBindingMismatch(
                    f"{mismatched} job(s) belong to a different execution identity; "
                    "refusing to dispatch or cancel them."
                )

    def get_binding(self) -> dict[str, str] | None:
        with self.read() as conn:
            row = conn.execute(
                "SELECT mode, fingerprint FROM db_binding WHERE singleton=1"
            ).fetchone()
            return dict(row) if row is not None else None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def snapshot(self) -> Iterator[sqlite3.Connection]:
        """Consistent read snapshot for a status/events response."""
        conn = self.connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
