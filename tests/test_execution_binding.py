from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from anime_studio.artifacts import LocalArtifactStore
from anime_studio.comfy import MockComfyTransport
from anime_studio.comfy_http import ComfyHttpTransport
from anime_studio.db import Database
from anime_studio.errors import ExecutionBindingMismatch
from anime_studio.orchestrator import Orchestrator
from anime_studio.workflow import load_workflow
from tests.comfy_stub import run_stub
from tests.helpers import make_request, make_stack
from tests.p1_helpers import (
    make_real_stack,
    make_transport,
    workflow_nodes,
)


def test_mock_queued_job_cannot_be_opened_by_real_adapter(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_stack(tmp_path)
        stack.orchestrator.create_job(
            make_request(prompt="MOCK queued before real startup"),
            "binding-mock-to-real",
        )
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(ExecutionBindingMismatch):
            Orchestrator(stack.db, transport, stack.artifacts)
        assert state.prompt_requests == 0
        assert state.delete_calls == []
        assert state.interrupt_calls == 0


def test_real_database_cannot_be_opened_by_mock_adapter(tmp_path) -> None:
    with run_stub() as (base_url, _state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        with pytest.raises(ExecutionBindingMismatch):
            Orchestrator(
                stack.db,
                MockComfyTransport(stack.db),
                LocalArtifactStore(tmp_path / "mock-artifacts"),
            )
        assert stack.orchestrator.get_job  # original real orchestrator still intact


def test_changed_endpoint_on_same_db_is_rejected_before_network(tmp_path) -> None:
    with run_stub() as (base_url, first_state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        with run_stub() as (other_url, second_state):
            transport = make_transport(tmp_path, base_url=other_url)
            with pytest.raises(ExecutionBindingMismatch):
                Orchestrator(stack.db, transport, stack.artifacts)
            assert first_state.prompt_requests == 0
            assert second_state.prompt_requests == 0


def test_changed_workflow_on_same_db_is_rejected_before_network(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_real_stack(tmp_path, base_url=base_url)
        workflow_path = stack.config.workflow_path
        changed = workflow_nodes()
        changed["4"]["inputs"]["ckpt_name"] = "different-fixture-model"
        workflow_path.write_text(json.dumps(changed), encoding="utf-8")
        workflow, _ = load_workflow(workflow_path)
        transport = ComfyHttpTransport(stack.config, workflow)
        with pytest.raises(ExecutionBindingMismatch):
            Orchestrator(stack.db, transport, stack.artifacts)
        assert state.prompt_requests == 0


def test_legacy_db_without_binding_is_mock_only(tmp_path) -> None:
    with run_stub() as (base_url, state):
        stack = make_stack(tmp_path)
        job, _ = stack.orchestrator.create_job(
            make_request(prompt="legacy P0 job"), "binding-legacy"
        )
        with stack.db.transaction() as conn:
            conn.execute("DELETE FROM db_binding WHERE singleton=1")
        legacy_mock = Orchestrator(
            stack.db,
            MockComfyTransport(stack.db),
            LocalArtifactStore(tmp_path / "legacy-artifacts"),
        )
        labelled = legacy_mock.get_job(job["job_id"])
        assert labelled is not None
        assert labelled["mode"] == "MOCK"
        assert labelled["mock"] is True
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(ExecutionBindingMismatch):
            Orchestrator(stack.db, transport, stack.artifacts)
        assert state.prompt_requests == 0


def test_actual_old_p0_schema_is_migrated_as_mock_only(tmp_path) -> None:
    db_path = tmp_path / "old-p0.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload_canonical TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
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
            """
        )
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, idempotency_key, payload_canonical, payload_sha256,
                status, progress, correlation_key, created_at, created_ms,
                updated_at, updated_ms, deadline_at, deadline_ts, timeout_seconds
            ) VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-job",
                "old-key",
                '{"payload_version":1}',
                "0" * 64,
                "old-corr",
                "2026-10-20T00:00:00.000Z",
                1,
                "2026-10-20T00:00:00.000Z",
                1,
                "2026-10-20T00:01:00.000Z",
                60001,
                60,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    db = Database(db_path)
    db.init()
    with db.read() as check:
        row = check.execute(
            "SELECT execution_mode, execution_fingerprint FROM jobs WHERE job_id='old-job'"
        ).fetchone()
    assert row["execution_mode"] == "MOCK"
    assert row["execution_fingerprint"] == "MOCK"
    db.bind_execution("MOCK", "MOCK")

    with run_stub() as (base_url, state):
        transport = make_transport(tmp_path, base_url=base_url)
        with pytest.raises(ExecutionBindingMismatch):
            Orchestrator(db, transport, LocalArtifactStore(tmp_path / "old-artifacts"))
        assert state.prompt_requests == 0
