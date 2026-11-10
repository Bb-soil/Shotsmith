from __future__ import annotations

import httpx
import pytest

from tests.helpers import (
    make_request,
    make_stack,
    persisted_job_for_crash_response,
    post_job_crash_tolerant,
)


class _ReadFailingClient:
    def post(self, *args, **kwargs):
        raise httpx.ReadError("response lost after the request was accepted")


class _ConnectFailingClient:
    def post(self, *args, **kwargs):
        raise httpx.ConnectError("request was not accepted")


class _RuntimeFailingClient:
    def post(self, *args, **kwargs):
        raise RuntimeError("unexpected transport failure")


def test_crash_helper_allows_only_response_loss_errors() -> None:
    assert post_job_crash_tolerant(_ReadFailingClient(), "helper-read-0001") is None

    with pytest.raises(httpx.ConnectError):
        post_job_crash_tolerant(_ConnectFailingClient(), "helper-connect-0001")

    with pytest.raises(RuntimeError):
        post_job_crash_tolerant(_RuntimeFailingClient(), "helper-runtime-0001")


@pytest.mark.parametrize("status_code", [200, 400, 500])
def test_crash_helper_rejects_non_201_response(tmp_path, status_code: int) -> None:
    stack = make_stack(tmp_path)
    job, _ = stack.orchestrator.create_job(make_request(), "helper-status-0001")
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://test/jobs"),
        json={"job": job},
    )

    with pytest.raises(AssertionError):
        persisted_job_for_crash_response(
            tmp_path / "p0.sqlite3", "helper-status-0001", response
        )
