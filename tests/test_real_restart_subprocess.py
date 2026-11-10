from __future__ import annotations

from pathlib import Path

from tests.comfy_stub import run_stub
from tests.helpers import (
    ServerProcess,
    persisted_job_for_crash_response,
    post_job_crash_tolerant,
    wait_job,
)
from tests.p1_helpers import (
    PNG_FIXTURE,
    make_manifest,
    write_config,
    write_manifest,
    write_workflow,
)
from anime_studio.workflow import workflow_sha256


def _real_server_assets(tmp_path: Path, base_url: str) -> Path:
    workflow_path = tmp_path / "real-workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "real-manifest.json"
    write_manifest(
        manifest_path,
        make_manifest(workflow_sha256=workflow_sha256(workflow_path)),
    )
    input_path = tmp_path / "real-input.png"
    input_path.write_bytes(PNG_FIXTURE)
    return write_config(
        tmp_path,
        base_url=base_url,
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        input_image_path=input_path,
        workflow_sha256=workflow_sha256(workflow_path),
    )


def _server(
    tmp_path: Path,
    config_path: Path,
    name: str,
    *,
    env: dict[str, str] | None = None,
) -> ServerProcess:
    return ServerProcess(
        db_path=tmp_path / "real.sqlite3",
        artifacts_dir=tmp_path / "real-artifacts",
        log_path=tmp_path / f"{name}.log",
        env=env,
        extra_args=[
            "--adapter",
            "comfy-http",
            "--allow-live-transport",
            "--comfy-config",
            str(config_path),
        ],
    )


def test_real_subprocess_crash_after_accepted_prompt_restart_is_ambiguous_no_resubmit(
    tmp_path,
) -> None:
    with run_stub() as (base_url, state):
        config_path = _real_server_assets(tmp_path, base_url)
        first = _server(
            tmp_path,
            config_path,
            "real-crash-first",
            env={"ANIME_STUDIO_FAULT": "crash_after_remote_submit"},
        )
        first.start()
        try:
            response = post_job_crash_tolerant(
                first.client,
                "real-restart-0001",
                prompt="REAL restart fixture",
                timeout_seconds=30,
                duration_seconds=1.0,
            )
            assert first.wait_exit(timeout=20.0) == 17, first.log_tail()
        finally:
            first.stop()

        before = persisted_job_for_crash_response(
            tmp_path / "real.sqlite3", "real-restart-0001", response
        )
        job_id = before["job_id"]
        assert before["status"] == "dispatching"
        assert before["prompt_id"] is None
        assert before["execution_mode"] == "COMFY_HTTP"
        assert state.prompt_requests == 1
        assert state.delete_calls == []
        assert state.interrupt_calls == 0

        second = _server(tmp_path, config_path, "real-crash-second")
        second.start()
        try:
            final = wait_job(second.client, job_id, timeout=20.0)
            assert final["status"] == "failed", final
            assert final["error"]["code"] == "AMBIGUOUS_DISPATCH"
            events = second.client.get(
                f"/jobs/{job_id}/events", params={"after": 0, "limit": 200}
            ).json()["events"]
            types = [event["type"] for event in events]
            assert "dispatch_ambiguous" in types
            assert "remote_submitted" not in types
            assert "job_cancelled" not in types
            assert state.prompt_requests == 1
            assert state.delete_calls == []
            assert state.interrupt_calls == 0
        finally:
            second.stop()
