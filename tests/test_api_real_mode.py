from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from anime_studio.api import create_app
from anime_studio.settings import Settings
from anime_studio.workflow import workflow_sha256
from tests.comfy_stub import run_stub
from tests.helpers import db_scalar, wait_job
from tests.p1_helpers import (
    PNG_FIXTURE,
    make_manifest,
    write_config,
    write_manifest,
    write_workflow,
)


def _real_config(tmp_path: Path, base_url: str) -> Path:
    workflow_path = tmp_path / "workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        make_manifest(workflow_sha256=workflow_sha256(workflow_path)),
    )
    input_path = tmp_path / "input.png"
    input_path.write_bytes(PNG_FIXTURE)
    return write_config(
        tmp_path,
        base_url=base_url,
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        input_image_path=input_path,
        workflow_sha256=workflow_sha256(workflow_path),
    )


def test_real_api_health_and_result_are_not_hardcoded_mock(tmp_path) -> None:
    with run_stub() as (base_url, state):
        settings = Settings(
            db_path=tmp_path / "api.sqlite3",
            artifacts_dir=tmp_path / "artifacts",
            adapter_mode="comfy-http",
            allow_live_transport=True,
            comfy_config_path=_real_config(tmp_path, base_url),
            poll_interval=0.01,
        )
        app = create_app(settings)
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["mode"] == "COMFY_HTTP"
            assert health.json()["mock"] is False

            response = client.post(
                "/jobs",
                headers={"Idempotency-Key": "real-api-0001"},
                json={"prompt": "REAL API fixture", "timeout_seconds": 30, "seed": 5},
            )
            assert response.status_code == 201, response.text
            job_id = response.json()["job"]["job_id"]
            deadline = time.monotonic() + 10
            while state.prompt_requests == 0 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert state.prompt_requests == 1
            state.set_success()

            deadline = time.monotonic() + 10
            job = None
            while time.monotonic() < deadline:
                job = client.get(f"/jobs/{job_id}").json()["job"]
                if job["status"] in {"succeeded", "failed", "cancelled", "timed_out"}:
                    break
                time.sleep(0.02)
            assert job is not None and job["status"] == "succeeded", job
            result = client.get(f"/jobs/{job_id}/result").json()
            assert result["mock"] is False
            assert result["mode"] == "COMFY_HTTP"
            assert result["result"]["mock"] is False
            assert result["result"]["artifacts"][0]["media_type"] == "video/mp4"


def test_partial_acceptance_diagnostics_are_redacted_in_api_and_sqlite(
    tmp_path, monkeypatch
) -> None:
    token = "SYNTHETIC_TRANSPORT_TOKEN_123"
    query_secret = "SYNTHETIC_QUERY_SECRET_456"
    nested_secret = "SYNTHETIC_NESTED_PASSWORD_789"
    monkeypatch.setenv("ANIME_STUDIO_COMFY_TOKEN", token)
    with run_stub() as (base_url, state):
        state.prompt_mode = "node_errors"
        state.node_errors_payload = {
            "5": {
                "errors": [f"Authorization: Bearer {token}"],
                "token": token,
                "nested": {
                    "password": nested_secret,
                    "url": f"https://example.invalid/?api_key={query_secret}",
                },
                "deep": {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}},
            }
        }
        settings = Settings(
            db_path=tmp_path / "api-diagnostics.sqlite3",
            artifacts_dir=tmp_path / "artifacts-diagnostics",
            adapter_mode="comfy-http",
            allow_live_transport=True,
            comfy_config_path=_real_config(tmp_path, base_url),
            poll_interval=0.01,
        )
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.post(
                "/jobs",
                headers={"Idempotency-Key": "real-diagnostics-0001"},
                json={"prompt": "REAL partial acceptance", "timeout_seconds": 30, "seed": 5},
            )
            assert response.status_code == 201, response.text
            job_id = response.json()["job"]["job_id"]
            job = wait_job(
                client,
                job_id,
                until=lambda current: current["prompt_id"] is not None,
                timeout=10.0,
            )
            assert job["status"] in {"submitted", "running"}
            assert job["prompt_id"] == "prompt-1"
            assert job["mock"] is False
            events = client.get(
                f"/jobs/{job_id}/events", params={"after": 0, "limit": 200}
            ).json()
            remote_events = [
                event for event in events["events"] if event["type"] == "remote_submitted"
            ]
            assert remote_events
            diagnostics = remote_events[-1]["data"].get("diagnostics")
            assert diagnostics and diagnostics["partial_acceptance"] is True
            api_text = __import__("json").dumps(events, ensure_ascii=False)
            for secret in (token, query_secret, nested_secret):
                assert secret not in api_text
            db_text = db_scalar(
                settings.db_path,
                "SELECT GROUP_CONCAT(data_json) FROM job_events",
            )
            assert db_text
            for secret in (token, query_secret, nested_secret):
                assert secret not in db_text
            time.sleep(0.1)
            assert state.prompt_requests == 1
            cancel = client.post(f"/jobs/{job_id}/cancel")
            assert cancel.status_code == 200
            final = wait_job(client, job_id, timeout=10.0)
            assert final["status"] == "cancelled", final
            final_events = client.get(
                f"/jobs/{job_id}/events", params={"after": 0, "limit": 200}
            ).json()
            final_text = __import__("json").dumps(final_events, ensure_ascii=False)
            for secret in (token, query_secret, nested_secret):
                assert secret not in final_text
            assert state.prompt_requests == 1
