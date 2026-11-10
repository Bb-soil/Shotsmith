from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from anime_studio.errors import IdempotencyConflict
from anime_studio.models import JobRequest, MockOptions, canonical_payload
from tests.helpers import (
    db_scalar,
    make_request,
    make_stack,
    post_job,
    running_server,
    wait_job,
)


def test_canonical_payload_is_stable() -> None:
    first = JobRequest(
        prompt="MOCK canonical",
        timeout_seconds=10,
        seed=1,
        mock_options=MockOptions(duration_seconds=0.5),
    )
    second = JobRequest(
        prompt="MOCK canonical",
        timeout_seconds=10,
        seed=1,
        mock_options=MockOptions(duration_seconds=0.5, fail=False),
    )
    text1, digest1 = canonical_payload(first)
    text2, digest2 = canonical_payload(second)
    assert text1 == text2
    assert digest1 == digest2

    different = first.model_copy(update={"prompt": "MOCK different"})
    _, digest3 = canonical_payload(different)
    assert digest3 != digest1


def test_same_key_same_payload_returns_existing_job(tmp_path) -> None:
    stack = make_stack(tmp_path)
    request = make_request()
    first, created_first = stack.orchestrator.create_job(
        request, "idem-key-0001"
    )
    second, created_second = stack.orchestrator.create_job(
        request, "idem-key-0001"
    )
    assert created_first is True
    assert created_second is False
    assert first["job_id"] == second["job_id"]
    events = stack.orchestrator.get_events(first["job_id"], after=0, limit=100)
    assert sum(1 for event in events["events"] if event["type"] == "job_submitted") == 1


def test_same_key_different_payload_conflicts(tmp_path) -> None:
    stack = make_stack(tmp_path)
    first, _ = stack.orchestrator.create_job(
        make_request(prompt="MOCK A"), "idem-key-0002"
    )
    with pytest.raises(IdempotencyConflict) as exc_info:
        stack.orchestrator.create_job(
            make_request(prompt="MOCK B"), "idem-key-0002"
        )
    assert exc_info.value.existing_job_id == first["job_id"]


def test_concurrent_http_idempotency_creates_one_remote_job(tmp_path) -> None:
    with running_server(tmp_path) as server:
        def submit(_: int) -> httpx.Response:
            with httpx.Client(base_url=server.url, timeout=20.0) as client:
                return post_job(
                    client,
                    "concurrent-key-0001",
                    prompt="MOCK concurrent idempotency",
                    duration_seconds=0.8,
                )

        with ThreadPoolExecutor(max_workers=12) as pool:
            responses = list(pool.map(submit, range(12)))

        statuses = [response.status_code for response in responses]
        assert statuses.count(201) == 1, statuses
        assert statuses.count(200) == 11, statuses
        job_ids = {response.json()["job"]["job_id"] for response in responses}
        assert len(job_ids) == 1
        job_id = next(iter(job_ids))

        final = wait_job(server.client, job_id, timeout=30.0)
        assert final["status"] == "succeeded"
        assert (
            db_scalar(
                tmp_path / "p0.sqlite3",
                "SELECT COUNT(*) FROM mock_remote_jobs",
            )
            == 1
        )

        conflict = post_job(
            server.client,
            "concurrent-key-0001",
            prompt="MOCK different payload",
            duration_seconds=0.8,
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "IDEMPOTENCY_CONFLICT"
