from __future__ import annotations

import time
from pathlib import Path

import httpx

from tests.helpers import (
    ServerProcess,
    db_row,
    db_scalar,
    persisted_job_for_crash_response,
    post_job,
    post_job_crash_tolerant,
    wait_job,
)


def _start_server(
    tmp_path: Path,
    *,
    name: str,
    env: dict[str, str] | None = None,
    no_worker: bool = False,
) -> ServerProcess:
    server = ServerProcess(
        db_path=tmp_path / "p0.sqlite3",
        artifacts_dir=tmp_path / "artifacts",
        log_path=tmp_path / f"{name}.log",
        env=env,
        no_worker=no_worker,
    )
    server.start()
    return server


def test_crash_after_remote_submit_reconciles_without_duplicate(tmp_path) -> None:
    first = _start_server(
        tmp_path,
        name="crash-after-submit",
        env={"ANIME_STUDIO_FAULT": "crash_after_remote_submit"},
    )
    try:
        response = post_job_crash_tolerant(
            first.client,
            "restart-crash-0001",
            prompt="MOCK crash after remote submit",
            timeout_seconds=30,
            duration_seconds=5.0,
        )
        assert first.wait_exit(timeout=20.0) == 17, first.log_tail()
    finally:
        first.stop()

    job_before = persisted_job_for_crash_response(
        tmp_path / "p0.sqlite3", "restart-crash-0001", response
    )
    job_id = job_before["job_id"]
    remote_before = db_row(
        tmp_path / "p0.sqlite3",
        "SELECT * FROM mock_remote_jobs WHERE correlation_key=?",
        (job_before["correlation_key"],),
    )
    assert job_before["status"] == "dispatching", job_before
    assert job_before["prompt_id"] is None
    assert remote_before is not None
    remote_prompt_id = remote_before["prompt_id"]

    second = _start_server(tmp_path, name="restart-after-submit")
    try:
        final = wait_job(second.client, job_id, timeout=30.0)
        assert final["status"] == "succeeded", final
        assert final["prompt_id"] == remote_prompt_id
        assert (
            db_scalar(
                tmp_path / "p0.sqlite3",
                "SELECT COUNT(*) FROM mock_remote_jobs",
            )
            == 1
        )
        events = second.client.get(
            f"/jobs/{job_id}/events", params={"after": 0, "limit": 200}
        ).json()["events"]
        types = [event["type"] for event in events]
        assert types.count("dispatch_reconciled") == 1
        assert "dispatch_ambiguous" not in types
        assert types.count("remote_submitted") == 0
    finally:
        second.stop()


def _run_crash_after_claim_case(
    tmp_path: Path, *, key: str
) -> httpx.Response | None:
    first = _start_server(
        tmp_path,
        name=f"crash-after-claim-{key}",
        env={"ANIME_STUDIO_FAULT": "crash_after_claim"},
    )
    try:
        response = post_job_crash_tolerant(
            first.client,
            key,
            prompt="MOCK crash after claim",
            timeout_seconds=30,
            duration_seconds=1.0,
        )
        assert first.wait_exit(timeout=20.0) == 17, first.log_tail()
    finally:
        first.stop()

    job_before = persisted_job_for_crash_response(
        tmp_path / "p0.sqlite3", key, response
    )
    job_id = job_before["job_id"]
    assert job_before["status"] == "dispatching", job_before
    assert job_before["prompt_id"] is None
    assert (
        db_scalar(
            tmp_path / "p0.sqlite3",
            "SELECT COUNT(*) FROM mock_remote_jobs",
        )
        == 0
    )

    second = _start_server(tmp_path, name=f"restart-after-claim-{key}")
    try:
        final = wait_job(second.client, job_id, timeout=30.0)
        assert final["status"] == "succeeded", final
        assert final["prompt_id"] is not None
        remote_after = db_row(
            tmp_path / "p0.sqlite3",
            "SELECT * FROM mock_remote_jobs WHERE correlation_key=?",
            (job_before["correlation_key"],),
        )
        assert remote_after is not None
        assert final["prompt_id"] == remote_after["prompt_id"]
        assert (
            db_scalar(
                tmp_path / "p0.sqlite3",
                "SELECT COUNT(*) FROM mock_remote_jobs",
            )
            == 1
        )
        events = second.client.get(
            f"/jobs/{job_id}/events", params={"after": 0, "limit": 200}
        ).json()["events"]
        types = [event["type"] for event in events]
        assert types.count("dispatch_retried_idempotent") == 1
        assert "dispatch_ambiguous" not in types
        assert types.count("remote_submitted") == 0
    finally:
        second.stop()
    return response


def test_crash_after_claim_resubmits_safely(tmp_path) -> None:
    _run_crash_after_claim_case(tmp_path, key="restart-crash-0002")


def test_crash_after_claim_lost_http_response_recovers_by_idempotency_key(
    tmp_path, monkeypatch
) -> None:
    real_post = post_job

    def _lost_response(*args, **kwargs):
        response = real_post(*args, **kwargs)
        assert response.status_code == 201, response.text
        raise httpx.ReadError("simulated lost HTTP response after job persistence")

    monkeypatch.setattr("tests.helpers.post_job", _lost_response)
    response = _run_crash_after_claim_case(
        tmp_path, key="restart-crash-0003-lost-response"
    )
    assert response is None


def test_crash_after_claim_downtime_beyond_deadline_never_submits(tmp_path) -> None:
    first = _start_server(
        tmp_path,
        name="deadline-after-claim",
        env={"ANIME_STUDIO_FAULT": "crash_after_claim"},
    )
    try:
        response = post_job_crash_tolerant(
            first.client,
            "deadline-claim-0001",
            prompt="MOCK crash after claim beyond deadline",
            timeout_seconds=2,
            duration_seconds=10.0,
        )
        assert first.wait_exit(timeout=20.0) == 17, first.log_tail()
    finally:
        first.stop()

    persisted = persisted_job_for_crash_response(
        tmp_path / "p0.sqlite3", "deadline-claim-0001", response
    )
    job_id = persisted["job_id"]
    assert persisted["status"] == "dispatching"
    assert persisted["prompt_id"] is None
    assert (
        db_scalar(
            tmp_path / "p0.sqlite3",
            "SELECT COUNT(*) FROM mock_remote_jobs",
        )
        == 0
    )

    time.sleep(2.2)
    second = _start_server(tmp_path, name="deadline-after-claim-restart")
    try:
        final = wait_job(second.client, job_id, timeout=15.0)
        assert final["status"] == "timed_out", final
        assert (
            db_scalar(
                tmp_path / "p0.sqlite3",
                "SELECT COUNT(*) FROM mock_remote_jobs",
            )
            == 0
        )
        events = second.client.get(
            f"/jobs/{job_id}/events", params={"after": 0, "limit": 200}
        ).json()["events"]
        types = [event["type"] for event in events]
        assert "job_timed_out" in types
        assert "job_succeeded" not in types
        assert "remote_submitted" not in types
        assert "dispatch_retried_idempotent" not in types
    finally:
        second.stop()


def test_crash_after_remote_submit_downtime_beyond_deadline_times_out(tmp_path) -> None:
    first = _start_server(
        tmp_path,
        name="deadline-after-submit",
        env={"ANIME_STUDIO_FAULT": "crash_after_remote_submit"},
    )
    try:
        response = post_job_crash_tolerant(
            first.client,
            "deadline-submit-0001",
            prompt="MOCK crash after remote submit beyond deadline",
            timeout_seconds=2,
            duration_seconds=0.2,
        )
        assert first.wait_exit(timeout=20.0) == 17, first.log_tail()
    finally:
        first.stop()

    persisted = persisted_job_for_crash_response(
        tmp_path / "p0.sqlite3", "deadline-submit-0001", response
    )
    job_id = persisted["job_id"]
    remote_before = db_row(
        tmp_path / "p0.sqlite3",
        "SELECT * FROM mock_remote_jobs WHERE correlation_key=?",
        (persisted["correlation_key"],),
    )
    assert persisted["status"] == "dispatching", persisted
    assert persisted["prompt_id"] is None
    assert remote_before is not None
    remote_prompt_id = remote_before["prompt_id"]

    time.sleep(2.2)
    second = _start_server(tmp_path, name="deadline-after-submit-restart")
    try:
        final = wait_job(second.client, job_id, timeout=15.0)
        assert final["status"] == "timed_out", final
        assert final["prompt_id"] == remote_prompt_id
        assert (
            db_scalar(
                tmp_path / "p0.sqlite3",
                "SELECT COUNT(*) FROM mock_remote_jobs",
            )
            == 1
        )
        events = second.client.get(
            f"/jobs/{job_id}/events", params={"after": 0, "limit": 200}
        ).json()["events"]
        types = [event["type"] for event in events]
        assert types.count("job_timed_out") == 1
        assert "job_succeeded" not in types
        assert types.count("dispatch_reconciled_at_deadline") == 1
    finally:
        second.stop()


def test_timeout_survives_downtime(tmp_path) -> None:
    first = _start_server(tmp_path, name="timeout-down", no_worker=True)
    try:
        response = post_job(
            first.client,
            "restart-timeout-0001",
            prompt="MOCK timeout across restart",
            timeout_seconds=1,
            duration_seconds=10.0,
        )
        assert response.status_code == 201
        job_id = response.json()["job"]["job_id"]
    finally:
        first.stop()

    time.sleep(1.2)
    second = _start_server(tmp_path, name="timeout-up")
    try:
        final = wait_job(second.client, job_id, timeout=15.0)
        assert final["status"] == "timed_out", final
        assert final["error"]["code"] == "DEADLINE_EXCEEDED"
        events = second.client.get(
            f"/jobs/{job_id}/events", params={"after": 0, "limit": 100}
        ).json()["events"]
        assert any(event["type"] == "job_timed_out" for event in events)
    finally:
        second.stop()


def test_normal_restart_reconnects_without_resubmit(tmp_path) -> None:
    first = _start_server(tmp_path, name="normal-restart-1")
    try:
        response = post_job(
            first.client,
            "restart-normal-0001",
            prompt="MOCK normal restart",
            timeout_seconds=30,
            duration_seconds=3.0,
        )
        assert response.status_code == 201
        job_id = response.json()["job"]["job_id"]
        running = wait_job(
            first.client,
            job_id,
            until=lambda job: job["prompt_id"] is not None
            and job["status"] in {"submitted", "running"},
            timeout=15.0,
        )
        assert running["status"] in {"submitted", "running"}
        events_before = first.client.get(
            f"/jobs/{job_id}/events", params={"after": 0, "limit": 100}
        ).json()
        cursor = events_before["next_cursor"]
        before_types = [event["type"] for event in events_before["events"]]
        assert before_types.count("remote_submitted") == 1
    finally:
        first.stop()

    second = _start_server(tmp_path, name="normal-restart-2")
    try:
        final = wait_job(second.client, job_id, timeout=30.0)
        assert final["status"] == "succeeded", final
        after = second.client.get(
            f"/jobs/{job_id}/events",
            params={"after": cursor, "limit": 100},
        ).json()
        after_types = [event["type"] for event in after["events"]]
        assert "job_succeeded" in after_types
        # Reconnect continues from the cursor; no second submit is emitted.
        all_events = events_before["events"] + after["events"]
        assert sum(1 for event in all_events if event["type"] == "remote_submitted") == 1
        assert sum(1 for event in all_events if event["type"] == "dispatch_started") == 1
    finally:
        second.stop()
