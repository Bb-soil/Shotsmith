from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anime_studio.api import create_app
from anime_studio.errors import InstanceAlreadyRunning
from anime_studio.lock import InstanceLock
from anime_studio.settings import Settings
from anime_studio.workflow import workflow_sha256
from tests.comfy_stub import run_stub
from tests.helpers import post_job
from tests.p1_helpers import (
    PNG_FIXTURE,
    make_manifest,
    write_config,
    write_manifest,
    write_workflow,
)


def _config(tmp_path: Path, base_url: str, *, timeout: float = 2.0) -> Path:
    workflow_path = tmp_path / "lifecycle-workflow.json"
    write_workflow(workflow_path)
    manifest_path = tmp_path / "lifecycle-manifest.json"
    write_manifest(
        manifest_path,
        make_manifest(workflow_sha256=workflow_sha256(workflow_path)),
    )
    input_path = tmp_path / "lifecycle-input.png"
    input_path.write_bytes(PNG_FIXTURE)
    config_path = write_config(
        tmp_path,
        base_url=base_url,
        workflow_path=workflow_path,
        manifest_path=manifest_path,
        input_image_path=input_path,
        workflow_sha256=workflow_sha256(workflow_path),
        upload_timeout_seconds=timeout,
        read_timeout_seconds=timeout,
    )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["runtime"]["upload_timeout_seconds"] = timeout
    data["runtime"]["read_timeout_seconds"] = timeout
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def _settings(tmp_path: Path, config_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "lifecycle.sqlite3",
        artifacts_dir=tmp_path / "lifecycle-artifacts",
        adapter_mode="comfy-http",
        allow_live_transport=True,
        comfy_config_path=config_path,
        poll_interval=0.01,
    )


def test_successful_api_shutdown_releases_lock(tmp_path) -> None:
    with run_stub() as (base_url, _state):
        settings = _settings(tmp_path, _config(tmp_path, base_url))
        app = create_app(settings)
        with TestClient(app) as client:
            assert client.get("/health").json()["mode"] == "COMFY_HTTP"
        lock = InstanceLock(settings.lock_path).acquire()
        lock.release()


def test_slow_worker_shutdown_keeps_lock_until_thread_stops(tmp_path) -> None:
    with run_stub() as (base_url, state):
        state.upload_response_delay_seconds = 7.0
        settings = _settings(tmp_path, _config(tmp_path, base_url, timeout=30.0))
        app = create_app(settings)
        with TestClient(app) as client:
            response = post_job(
                client,
                "lifespan-slow-0001",
                prompt="API slow shutdown",
                timeout_seconds=60,
                duration_seconds=1.0,
            )
            assert response.status_code == 201, response.text
            deadline = time.monotonic() + 5
            while state.upload_requests == 0 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert state.upload_requests == 1
        # The worker exceeded the 5s stop budget; API must retain the lock.
        with pytest.raises(InstanceAlreadyRunning):
            InstanceLock(settings.lock_path).acquire()
        deadline = time.monotonic() + 15
        while app.state.worker.alive and time.monotonic() < deadline:
            time.sleep(0.1)
        assert app.state.worker.stop(timeout=5) is True
        app.state.instance_lock.release()
        lock = InstanceLock(settings.lock_path).acquire()
        lock.release()
