from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import (
    db_scalar,
    post_job,
    running_server,
    wait_job,
)


def test_success_progress_events_result_replay_and_conflict(tmp_path) -> None:
    with running_server(tmp_path) as server:
        response = post_job(
            server.client,
            "flow-success-0001",
            prompt="MOCK flow success",
            duration_seconds=0.6,
        )
        assert response.status_code == 201
        created = response.json()
        assert created["idempotent_replay"] is False
        job_id = created["job"]["job_id"]

        final = wait_job(server.client, job_id, timeout=30.0)
        assert final["status"] == "succeeded", final
        assert final["progress"] == 100
        assert final["prompt_id"].startswith("mock-")
        assert final["mode"] == "MOCK"

        # Cursor pagination and replay: no event is duplicated or skipped.
        cursor = 0
        events = []
        while True:
            page = server.client.get(
                f"/jobs/{job_id}/events",
                params={"after": cursor, "limit": 2},
            )
            page.raise_for_status()
            payload = page.json()
            for event in payload["events"]:
                assert event["cursor"] > cursor
                events.append(event)
            cursor = payload["next_cursor"]
            if not payload["has_more"]:
                break

        types = [event["type"] for event in events]
        assert types[0] == "job_submitted"
        assert "dispatch_started" in types
        assert "remote_submitted" in types
        assert "job_succeeded" in types
        assert any(
            event["type"] in {"progress", "job_running"} for event in events
        )
        assert len({event["cursor"] for event in events}) == len(events)
        empty_page = server.client.get(
            f"/jobs/{job_id}/events", params={"after": cursor, "limit": 2}
        ).json()
        assert empty_page["events"] == []
        assert empty_page["next_cursor"] == cursor

        result_response = server.client.get(f"/jobs/{job_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()["result"]
        assert result["mock"] is True
        assert "MOCK" in result["note"]
        artifact = result["artifacts"][0]
        artifact_path = Path(artifact["path"])
        assert artifact_path.exists()
        assert artifact_path.suffix == ".json"
        assert artifact_path.name.endswith(".mock.json")
        assert artifact["mock"] is True
        artifact_body = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact_body["mock"] is True
        assert "NOT A REAL VIDEO" in artifact_body["note"]
        assert not list(Path(tmp_path / "artifacts").rglob("*.mp4"))

        replay = post_job(
            server.client,
            "flow-success-0001",
            prompt="MOCK flow success",
            duration_seconds=0.6,
        )
        assert replay.status_code == 200
        assert replay.json()["idempotent_replay"] is True
        assert replay.json()["job"]["job_id"] == job_id

        conflict = post_job(
            server.client,
            "flow-success-0001",
            prompt="MOCK different flow payload",
            duration_seconds=0.6,
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["existing_job_id"] == job_id
        assert (
            db_scalar(
                tmp_path / "p0.sqlite3",
                "SELECT COUNT(*) FROM mock_remote_jobs",
            )
            == 1
        )


def test_execution_error_is_terminal_and_result_is_not_available(tmp_path) -> None:
    with running_server(tmp_path) as server:
        response = post_job(
            server.client,
            "flow-error-0001",
            prompt="MOCK deliberate execution error",
            duration_seconds=0.5,
            fail=True,
            fail_at_progress=40,
        )
        assert response.status_code == 201
        job_id = response.json()["job"]["job_id"]
        final = wait_job(server.client, job_id, timeout=30.0)
        assert final["status"] == "failed", final
        assert final["error"]["code"] == "MOCK_EXECUTION_ERROR"
        assert "Mock execution failed" in final["error"]["message"]
        result = server.client.get(f"/jobs/{job_id}/result")
        assert result.status_code == 409
        assert result.json()["detail"]["code"] == "RESULT_NOT_READY"


def test_timeout_uses_persisted_deadline(tmp_path) -> None:
    with running_server(tmp_path) as server:
        response = post_job(
            server.client,
            "flow-timeout-0001",
            prompt="MOCK timeout",
            timeout_seconds=1,
            duration_seconds=10.0,
        )
        assert response.status_code == 201
        job_id = response.json()["job"]["job_id"]
        final = wait_job(server.client, job_id, timeout=15.0)
        assert final["status"] == "timed_out", final
        assert final["error"]["code"] == "DEADLINE_EXCEEDED"


def test_cancel_after_dispatch(tmp_path) -> None:
    with running_server(tmp_path) as server:
        response = post_job(
            server.client,
            "flow-cancel-0001",
            prompt="MOCK cancel after dispatch",
            duration_seconds=5.0,
        )
        assert response.status_code == 201
        job_id = response.json()["job"]["job_id"]
        dispatched = wait_job(
            server.client,
            job_id,
            until=lambda job: job["prompt_id"] is not None
            and job["status"] in {"submitted", "running"},
            timeout=15.0,
        )
        assert dispatched["prompt_id"] is not None
        cancel = server.client.post(f"/jobs/{job_id}/cancel")
        assert cancel.status_code == 200, cancel.text
        assert cancel.json()["job"]["status"] in {"cancelled", "running", "submitted"}
        final = wait_job(server.client, job_id, timeout=15.0)
        assert final["status"] == "cancelled", final

        events = server.client.get(
            f"/jobs/{job_id}/events", params={"after": 0, "limit": 100}
        ).json()["events"]
        types = [event["type"] for event in events]
        assert "cancel_requested" in types
        assert "job_cancelled" in types
        assert "job_succeeded" not in types


def test_cancel_before_dispatch_without_worker(tmp_path) -> None:
    with running_server(tmp_path, no_worker=True) as server:
        response = post_job(server.client, "flow-cancel-0002")
        assert response.status_code == 201
        job_id = response.json()["job"]["job_id"]
        cancel = server.client.post(f"/jobs/{job_id}/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["job"]["status"] == "cancelled"
        assert (
            db_scalar(
                tmp_path / "p0.sqlite3",
                "SELECT COUNT(*) FROM mock_remote_jobs",
            )
            == 0
        )
        events = server.client.get(
            f"/jobs/{job_id}/events", params={"after": 0, "limit": 100}
        ).json()["events"]
        cancelled_events = [
            event for event in events if event["type"] == "job_cancelled"
        ]
        assert cancelled_events
        assert cancelled_events[-1]["data"]["reason"] == "cancel_before_dispatch"


def test_cancel_terminal_job_conflicts(tmp_path) -> None:
    with running_server(tmp_path) as server:
        response = post_job(
            server.client,
            "flow-cancel-0003",
            prompt="MOCK cancel too late",
            duration_seconds=0.3,
        )
        assert response.status_code == 201
        job_id = response.json()["job"]["job_id"]
        final = wait_job(server.client, job_id, timeout=20.0)
        assert final["status"] == "succeeded"
        cancel = server.client.post(f"/jobs/{job_id}/cancel")
        assert cancel.status_code == 409
        assert cancel.json()["detail"]["code"] == "JOB_ALREADY_TERMINAL"


def test_result_for_unknown_job_is_404(tmp_path) -> None:
    with running_server(tmp_path, no_worker=True) as server:
        response = server.client.get("/jobs/unknown-job/result")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "JOB_NOT_FOUND"
