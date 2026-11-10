from __future__ import annotations

from typing import Any

import httpx

from tests.helpers import db_scalar, post_job, running_server, wait_job


VALID_BODY: dict[str, Any] = {
    "prompt": "MOCK validation",
    "timeout_seconds": 30,
    "seed": 1,
    "mock_options": {
        "duration_seconds": 0.5,
        "fail": False,
        "fail_at_progress": 50,
    },
}


def _assert_validation_error(response: httpx.Response) -> None:
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "VALIDATION_ERROR", detail


def test_invalid_http_inputs_are_rejected(tmp_path) -> None:
    with running_server(tmp_path) as server:
        client = server.client

        _assert_validation_error(client.post("/jobs", json=VALID_BODY))

        _assert_validation_error(
            client.post(
                "/jobs",
                headers={"Idempotency-Key": "short"},
                json=VALID_BODY,
            )
        )
        bad_bodies = [
            {**VALID_BODY, "prompt": "   "},
            {**VALID_BODY, "prompt": ""},
            {**VALID_BODY, "prompt": "x" * 4001},
            {**VALID_BODY, "timeout_seconds": 0},
            {**VALID_BODY, "timeout_seconds": 3601},
            {**VALID_BODY, "timeout_seconds": 1.5},
            {**VALID_BODY, "seed": -1},
            {**VALID_BODY, "seed": 2_147_483_648},
            {**VALID_BODY, "unknown_field": 1},
            {
                **VALID_BODY,
                "mock_options": {**VALID_BODY["mock_options"], "duration_seconds": 0},
            },
            {
                **VALID_BODY,
                "mock_options": {
                    **VALID_BODY["mock_options"],
                    "duration_seconds": 60.1,
                },
            },
            {
                **VALID_BODY,
                "mock_options": {**VALID_BODY["mock_options"], "fail_at_progress": 0},
            },
            {
                **VALID_BODY,
                "mock_options": {
                    **VALID_BODY["mock_options"],
                    "fail_at_progress": 101,
                },
            },
            {
                **VALID_BODY,
                "mock_options": {
                    **VALID_BODY["mock_options"],
                    "mode": "real",
                },
            },
        ]
        for index, body in enumerate(bad_bodies):
            response = client.post(
                "/jobs",
                headers={"Idempotency-Key": f"bad-body-key-{index:04d}"},
                json=body,
            )
            _assert_validation_error(response)

        _assert_validation_error(
            client.post(
                "/jobs",
                headers={
                    "Idempotency-Key": "malformed-key-0001",
                    "Content-Type": "application/json",
                },
                content=b"{",
            )
        )

        _assert_validation_error(
            client.get("/jobs/unknown/events", params={"after": -1})
        )
        _assert_validation_error(
            client.get("/jobs/unknown/events", params={"limit": 0})
        )
        _assert_validation_error(
            client.get("/jobs/unknown/events", params={"limit": 501})
        )
        _assert_validation_error(
            client.get(
                "/jobs/unknown/events",
                params={"after": 9_223_372_036_854_775_808},
            )
        )

        missing = client.get("/jobs/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "JOB_NOT_FOUND"

        missing_cancel = client.post("/jobs/does-not-exist/cancel")
        assert missing_cancel.status_code == 404

        missing_events = client.get("/jobs/does-not-exist/events")
        assert missing_events.status_code == 404


def test_non_finite_raw_json_is_validation_error_without_side_effects(tmp_path) -> None:
    with running_server(tmp_path) as server:
        raw_bodies = [
            '{"prompt":"MOCK nonfinite","mock_options":{"duration_seconds":1e309}}',
            '{"prompt":"MOCK nonfinite","mock_options":{"duration_seconds":NaN}}',
            '{"prompt":"MOCK nonfinite","mock_options":{"duration_seconds":Infinity}}',
        ]
        for index, raw in enumerate(raw_bodies):
            response = server.client.post(
                "/jobs",
                headers={
                    "Idempotency-Key": f"nonfinite-key-{index:04d}",
                    "Content-Type": "application/json",
                },
                content=raw.encode("utf-8"),
            )
            assert response.status_code == 422, response.text
            detail = response.json()["detail"]
            assert detail["code"] == "VALIDATION_ERROR"
            assert isinstance(detail["errors"], list)
            # The raw input (including inf/NaN) must not be echoed back.
            assert "1e309" not in response.text
            assert "Infinity" not in response.text
            assert "NaN" not in response.text

        assert (
            db_scalar(
                tmp_path / "p0.sqlite3",
                "SELECT COUNT(*) FROM jobs",
            )
            == 0
        )

        valid = post_job(server.client, "nonfinite-valid-0001")
        assert valid.status_code == 201
        valid_job_id = valid.json()["job"]["job_id"]
        final = wait_job(server.client, valid_job_id, timeout=20.0)
        assert final["status"] == "succeeded"


def test_result_not_ready_is_explicit(tmp_path) -> None:
    with running_server(tmp_path, no_worker=True) as server:
        response = post_job(server.client, "not-ready-key-0001")
        assert response.status_code == 201
        job_id = response.json()["job"]["job_id"]
        result = server.client.get(f"/jobs/{job_id}/result")
        assert result.status_code == 409
        assert result.json()["detail"]["code"] == "RESULT_NOT_READY"
