from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.i1 import ops, readiness, state as state_mod
from scripts.i1.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]


class FakeRunner:
    def run_remote(self, *args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)


def _record(profile, workspace: str, key: str) -> dict:
    return {
        "schema": state_mod.RECORD_SCHEMA,
        "name": "fake",
        "identity": {
            "host": "example.test",
            "user": "root",
            "port": 22,
            "fingerprint": "SHA256:fake",
            "workspace": workspace,
            "profile_id": profile.profile_id,
            "profile_sha256": profile.sha256,
        },
        "auth": {"kind": "agent"},
        "known_hosts": "unused",
        "created_at": state_mod.now_iso(),
        "updated_at": state_mod.now_iso(),
    }


def _context(tmp_path: Path) -> ops.Context:
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    workspace = "/workspace"
    key = state_mod.identity_key(
        host="example.test",
        user="root",
        port=22,
        fingerprint="SHA256:fake",
        workspace=workspace,
        profile_sha256=profile.sha256,
    )
    value = state_mod.new_state(
        identity_key_value=key,
        host="example.test",
        user="root",
        port=22,
        fingerprint="SHA256:fake",
        workspace=str(workspace),
        profile_id=profile.profile_id,
        profile_sha256=profile.sha256,
    )
    value = state_mod.transition(value, "INSPECTED")
    value = state_mod.transition(value, "PLANNED")
    value = state_mod.transition(value, "INSTALLING")
    value["remote_facts"] = {"home": "/workspace"}
    record = _record(profile, str(workspace), key)
    return ops.Context(
        state_dir=tmp_path,
        repo_root=ROOT,
        profile_path=ROOT / "config" / "i1" / "wan22-14b-ab.profile.json",
        profile=profile,
        name="fake",
        record=record,
        runner=FakeRunner(),
        identity=object(),
        state=value,
        run_dir=tmp_path / "run",
    )


def _probe(profile) -> dict:
    return {
        "schema": "anime-video-studio/i1-runtime-evidence/v1",
        "run_id": "r",
        "nonce": "n",
        "profile_sha256": profile.sha256,
        "workflow_sha256": profile.workflow["api_sha256"],
        "torch": {"version": "2.9.1", "cuda": "12.8", "available": True},
        "comfyui": {"commit": profile.comfy["commit"], "import_ok": True},
        "pip": {"check_exit": 0},
        "models": [
            {"target": item["target"], "actual_size_bytes": item["size_bytes"], "actual_sha256": item["sha256"]}
            for item in profile.models
        ],
        "object_info": {
            "classes": list(profile.required_classes),
            "input_checks": [
                {"class": name, "input": input_name, "present": True}
                for name, names in profile.required_inputs.items()
                for input_name in names
            ],
            "link_checks": [],
            "savevideo_dynamic": {"selected": {"format": "mp4", "codec": "h264", "encoding": "auto"}, "issues": []},
        },
        "system_stats": {"devices": [{"name": "fake", "vram_total": 1}]},
        "workflow": {"sha256": profile.workflow["api_sha256"], "classes": list(profile.required_classes)},
    }


def test_install_orchestration_reaches_ready_with_ui_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    ctx.state["run_id"] = "r"
    ctx.state["nonce"] = "n"
    profile = ctx.profile
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **kwargs: {})
    monkeypatch.setattr(ops, "_stage_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops, "_run_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *args, **kwargs: {
            "paths": {"attempt": "/workspace/.avs-i1/attempts/0001"},
            "job_sha256": "a" * 64,
            "launch_sha256": None,
        },
    )
    monkeypatch.setattr(
        ops.download,
        "inspect_attempt",
        lambda *args, **kwargs: {"state": "OK", "process": "ABSENT", "status": {"status": "COMPLETE"}, "progress": {}, "heartbeat": {}},
    )
    monkeypatch.setattr(ops.workflow, "register", lambda *args, **kwargs: {"ui_status": "DERIVED_FROM_API_NOT_LIVE_VERIFIED", "ui_filename": "x.json", "ui_path": "/ui", "graph_rendered": False})
    monkeypatch.setattr(ops.workflow, "list_userdata", lambda **kwargs: "UI_FILE_ONLY")
    monkeypatch.setattr(ops.comfy, "stage_launch", lambda *args, **kwargs: {})
    monkeypatch.setattr(ops.comfy, "start", lambda *args, **kwargs: {})
    monkeypatch.setattr(ops.comfy, "read_owner", lambda *args, **kwargs: {"pid": 1})
    monkeypatch.setattr(ops.tunnel, "ensure", lambda **kwargs: object())
    monkeypatch.setattr(ops.tunnel, "health", lambda **kwargs: {"system": {"comfyui_version": "fake"}})
    monkeypatch.setattr(ops, "_fetch_api", lambda *args, **kwargs: {"object_info": "/oi.json", "system_stats": "/ss.json"})
    monkeypatch.setattr(ops, "_run_probe", lambda *args, **kwargs: _probe(profile))
    result = ops._continue_install(ctx, deadline_epoch=9_999_999_999)
    assert result["readiness"]["ready"] is True
    assert ctx.state["phase"] == "READY"
    assert ctx.state["ui"]["ui_status"] == "DERIVED_FROM_API_NOT_LIVE_VERIFIED"
    assert result["readiness"]["ui_graph_rendered"] is False


def test_plan_is_idempotent_and_execute_from_planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    ctx.state["phase"] = "NEW"
    inspection = {"compatible": True, "facts": {"disk_free_bytes": 1}, "blockers": [], "warnings": []}
    monkeypatch.setattr(ops, "inspect_remote", lambda *args, **kwargs: inspection)
    first = ops.plan_install(ctx)
    second = ops.plan_install(ctx)
    assert first["plan"]["profile_id"] == second["plan"]["profile_id"]
    assert ctx.state["phase"] == "PLANNED"
    # Execute from PLANNED must not perform an illegal PLANNED->INSPECTED transition.
    profile = ctx.profile
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **kwargs: {})
    monkeypatch.setattr(ops, "_stage_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops, "_run_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *args, **kwargs: {
            "paths": {"attempt": "/workspace/.avs-i1/attempts/0001"},
            "job_sha256": "a" * 64,
            "launch_sha256": None,
        },
    )
    monkeypatch.setattr(
        ops.download,
        "inspect_attempt",
        lambda *args, **kwargs: {"state": "OK", "process": "ABSENT", "status": {"status": "COMPLETE"}, "progress": {}, "heartbeat": {}},
    )
    monkeypatch.setattr(ops.workflow, "register", lambda *args, **kwargs: {"ui_status": "UI_GRAPH_NOT_VERIFIED", "ui_filename": "x.json", "ui_path": "/ui", "graph_rendered": False})
    monkeypatch.setattr(ops.workflow, "list_userdata", lambda **kwargs: "UI_FILE_ONLY")
    monkeypatch.setattr(ops.comfy, "stage_launch", lambda *args, **kwargs: {})
    monkeypatch.setattr(ops.comfy, "start", lambda *args, **kwargs: {})
    monkeypatch.setattr(ops.comfy, "read_owner", lambda *args, **kwargs: {"pid": 1})
    monkeypatch.setattr(ops.tunnel, "ensure", lambda **kwargs: object())
    monkeypatch.setattr(ops.tunnel, "health", lambda **kwargs: {"system": {"comfyui_version": "fake"}})
    monkeypatch.setattr(ops, "_fetch_api", lambda *args, **kwargs: {"object_info": "/oi.json", "system_stats": "/ss.json"})
    monkeypatch.setattr(ops, "_run_probe", lambda *args, **kwargs: _probe(profile))
    result = ops.execute_install(ctx)
    assert result["readiness"]["ready"] is True
    assert ctx.state["phase"] == "READY"


def test_interrupted_download_is_resumable_and_open_disconnect_are_owned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    ctx.state["run_id"] = "r"
    ctx.state["nonce"] = "n"
    profile = ctx.profile
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **kwargs: {})
    monkeypatch.setattr(ops, "_stage_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops, "_run_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *args, **kwargs: {
            "paths": {"attempt": "/workspace/.avs-i1/attempts/0001"},
            "job_sha256": "a" * 64,
            "launch_sha256": None,
        },
    )
    monkeypatch.setattr(
        ops.download,
        "inspect_attempt",
        lambda *args, **kwargs: {"state": "OK", "process": "UNKNOWN", "status": {"status": "RUNNING"}, "progress": {}, "heartbeat": {}},
    )
    monkeypatch.setattr(ops, "inspect_remote", lambda *args, **kwargs: {"compatible": True, "facts": {}, "blockers": [], "warnings": []})
    monkeypatch.setattr(ops, "workflow", SimpleNamespace(register=lambda *a, **k: {}, list_userdata=lambda **k: "UI_FILE_ONLY"))
    with pytest.raises(Exception):
        ops.execute_install(ctx)
    assert ctx.state["phase"] == "INTERRUPTED"

    ctx.state["phase"] = "READY"
    monkeypatch.setattr(ops.comfy, "stage_launch", lambda *args, **kwargs: {})
    monkeypatch.setattr(ops.comfy, "start", lambda *args, **kwargs: {})
    monkeypatch.setattr(ops.tunnel, "ensure", lambda **kwargs: object())
    monkeypatch.setattr(ops.tunnel, "health", lambda **kwargs: {"system": {}})
    opened = ops.open_comfy(ctx, browser=False)
    assert opened["url"].startswith("http://127.0.0.1:")
    assert "не остановлена" in opened["billing_warning"]
    monkeypatch.setattr(ops.tunnel, "stop", lambda **kwargs: True)
    closed = ops.disconnect(ctx)
    assert closed["status"] == "DISCONNECTED"
    assert closed["owned_tunnel_only"] is True


def test_bounded_health_poll_waits_for_coldstart_and_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ops.RemoteFailure("not ready")
        return {"system": {"comfyui_version": "fake"}}

    monkeypatch.setattr(ops.tunnel, "health", flaky)
    health = ops._wait_health(8189, timeout=10.0)
    assert health["system"]["comfyui_version"] == "fake"
    monkeypatch.setattr(
        ops.tunnel, "health", lambda **kwargs: (_ for _ in ()).throw(ops.RemoteFailure("down"))
    )
    with pytest.raises(ops.RemoteFailure):
        ops._wait_health(8189, timeout=0.05)


def test_installed_resume_finalizes_without_setup_or_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    ctx.state["phase"] = "INSTALLED"
    ctx.state["artifacts"] = {
        "setup": "COMPLETE",
        "download": "COMPLETE",
        "workflow": {
            "ui_status": "UI_GRAPH_NOT_VERIFIED",
            "ui_filename": "x.json",
            "ui_path": "/ui",
            "graph_rendered": False,
        },
    }
    monkeypatch.setattr(ops.download, "stage", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download mutation")))
    monkeypatch.setattr(ops, "_run_setup", lambda *a, **k: (_ for _ in ()).throw(AssertionError("setup mutation")))
    monkeypatch.setattr(ops.comfy, "stage_launch", lambda *a, **k: {})
    monkeypatch.setattr(ops.comfy, "start", lambda *a, **k: {})
    monkeypatch.setattr(ops.comfy, "read_owner", lambda *a, **k: {"pid": 1})
    monkeypatch.setattr(ops.tunnel, "ensure", lambda **k: object())
    monkeypatch.setattr(ops.tunnel, "health", lambda **k: {"system": {"comfyui_version": "fake"}})
    monkeypatch.setattr(ops, "_fetch_api", lambda *a, **k: {"object_info": "/oi", "system_stats": "/ss"})
    monkeypatch.setattr(ops.workflow, "list_userdata", lambda **k: "UI_FILE_ONLY")
    monkeypatch.setattr(ops, "_run_probe", lambda *a, **k: _probe(ctx.profile))
    result = ops.resume(ctx)
    assert result["readiness"]["ready"] is True
    assert ctx.state["phase"] == "READY"


def test_online_status_marks_cached_ready_unconfirmed_when_tunnel_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    ctx.state["phase"] = "READY"
    ctx.state["readiness"] = {"status": "READY", "ready": True, "warnings": []}
    monkeypatch.setattr(ops.download, "read_optional", lambda *a, **k: None)
    monkeypatch.setattr(ops.comfy, "read_owner", lambda *a, **k: None)
    monkeypatch.setattr(ops.tunnel, "health", lambda **k: (_ for _ in ()).throw(ops.RemoteFailure("down")))
    payload = ops.status(ctx, offline=False)
    assert payload["phase"] == "READY"
    assert payload["readiness_current"] is False
    assert payload["readiness"]["status"] == "UNCONFIRMED"
    assert payload["readiness"]["ready"] is False


def test_download_attempt_is_registered_before_upload_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **kwargs: {})
    monkeypatch.setattr(ops, "_stage_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops, "_run_setup", lambda *args, **kwargs: None)
    calls = []

    def failing_stage(*args, **kwargs):
        calls.append(kwargs)
        raise ops.RemoteFailure("injected after attempt registration")

    monkeypatch.setattr(ops.download, "stage", failing_stage)
    with pytest.raises(ops.RemoteFailure):
        ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    first = dict(ctx.state["artifacts"]["download_attempt"])
    assert first["id"] == "0001"
    assert first["deadline_epoch"] > time.time()
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *args, **kwargs: {
            "paths": {"attempt": first["dir"]},
            "job_sha256": "a" * 64,
            "launch_sha256": None,
        },
    )
    monkeypatch.setattr(
        ops.download,
        "inspect_attempt",
        lambda *args, **kwargs: {
            "state": "OK",
            "process": "ABSENT",
            "status": {"status": "COMPLETE"},
            "progress": {},
            "heartbeat": {},
        },
    )
    monkeypatch.setattr(ops.workflow, "register", lambda *a, **k: {})
    monkeypatch.setattr(ops.comfy, "stage_launch", lambda *a, **k: {})
    monkeypatch.setattr(ops.comfy, "start", lambda *a, **k: {})
    monkeypatch.setattr(ops.comfy, "read_owner", lambda *a, **k: {})
    monkeypatch.setattr(ops.tunnel, "ensure", lambda **k: object())
    monkeypatch.setattr(ops.tunnel, "health", lambda **k: {"system": {"comfyui_version": "fake"}})
    monkeypatch.setattr(ops, "_fetch_api", lambda *a, **k: {"object_info": "/oi", "system_stats": "/ss"})
    monkeypatch.setattr(ops, "_run_probe", lambda *a, **k: (_ for _ in ()).throw(ops.RemoteFailure("probe down")))
    with pytest.raises(ops.NotReady):
        ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    assert ctx.state["artifacts"]["download_attempt"]["id"] == first["id"]
    assert ctx.state["artifacts"]["download_attempt"]["deadline_epoch"] == first["deadline_epoch"]

def _attempt_context(tmp_path: Path) -> ops.Context:
    ctx = _context(tmp_path)
    ctx.state["phase"] = "INSTALLING"
    ctx.state["artifacts"] = {
        "setup": "COMPLETE",
        "download_attempt": {
            "id": "0001",
            "created_at": time.time() - 10,
            "deadline_epoch": time.time() + 3600,
            "dir": "/workspace/.avs-i1/attempts/0001",
        },
    }
    return ctx


def _patch_stage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **kwargs: {})
    monkeypatch.setattr(ops.download, "ensure_guard", lambda *a, **k: None)
    monkeypatch.setattr(ops, "_stage_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops, "_run_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(ops.RemoteFailure("stop after registration")),
    )


@pytest.mark.parametrize(
    "decision",
    [
        {"state": "OK", "process": "DEAD", "status": {"status": "RUNNING"}},
        {"state": "MISSING"},
    ],
)
def test_new_attempt_only_after_proved_dead_or_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: dict
) -> None:
    ctx = _attempt_context(tmp_path)
    _patch_stage_failure(monkeypatch)
    monkeypatch.setattr(ops.download, "inspect_attempt", lambda *a, **k: decision)
    with pytest.raises(ops.RemoteFailure):
        ops._continue_install(ctx, deadline_epoch=time.time() + 3600, new_attempt=True)
    assert ctx.state["artifacts"]["download_attempt"]["id"] == "0002"


def test_completed_old_attempt_finalizes_without_second_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    monkeypatch.setattr(
        ops.download,
        "inspect_attempt",
        lambda *a, **k: {
            "state": "OK",
            "process": "ABSENT",
            "status": {"status": "COMPLETE"},
            "progress": {},
            "heartbeat": {},
        },
    )
    monkeypatch.setattr(
        ops,
        "_finalize_after_download",
        lambda ctx, **kwargs: {"readiness": {"ready": True}, "state": ctx.state},
    )
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no restage")),
    )
    monkeypatch.setattr(
        ops.download,
        "start",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no second launch")),
    )
    result = ops._continue_install(
        ctx, deadline_epoch=time.time() + 3600, new_attempt=True
    )
    assert result["readiness"]["ready"] is True


@pytest.mark.parametrize(
    "decision",
    [
        {"state": "UNREADABLE", "reason": "ssh read failed"},
        {"state": "OK", "process": "LIVE", "status": {"status": "FAILED"}},
        {"state": "OK", "process": "UNKNOWN", "status": {"status": "RUNNING"}},
    ],
)
def test_new_attempt_refuses_unreadable_or_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: dict
) -> None:
    ctx = _attempt_context(tmp_path)
    _patch_stage_failure(monkeypatch)
    monkeypatch.setattr(ops.download, "inspect_attempt", lambda *a, **k: decision)
    with pytest.raises(ops.Interrupted):
        ops._continue_install(ctx, deadline_epoch=time.time() + 3600, new_attempt=True)
    assert ctx.state["artifacts"]["download_attempt"]["id"] == "0001"


def test_new_attempt_refuses_when_inspection_read_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    _patch_stage_failure(monkeypatch)
    monkeypatch.setattr(
        ops.download,
        "inspect_attempt",
        lambda *a, **k: (_ for _ in ()).throw(ops.RemoteFailure("ssh unavailable")),
    )
    with pytest.raises((ops.Interrupted, ops.RemoteFailure)):
        ops._continue_install(ctx, deadline_epoch=time.time() + 3600, new_attempt=True)
    assert ctx.state["artifacts"]["download_attempt"]["id"] == "0001"


def _ready_ctx(tmp_path: Path, *, with_workflow: bool) -> ops.Context:
    ctx = _context(tmp_path)
    ctx.state["phase"] = "READY"
    ctx.state["readiness"] = {"status": "READY", "ready": True, "warnings": [], "checks": []}
    artifacts = {"download": "COMPLETE"}
    if with_workflow:
        artifacts["workflow"] = {
            "ui_status": "UI_GRAPH_NOT_VERIFIED",
            "ui_filename": "x.json",
            "ui_path": "/ui",
            "graph_rendered": False,
        }
    ctx.state["artifacts"] = artifacts
    return ctx


@pytest.mark.parametrize(
    "stage",
    ["workflow", "comfy", "tunnel", "health", "api"],
)
def test_pre_probe_failure_invalidates_cached_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    ctx = _ready_ctx(tmp_path, with_workflow=stage != "workflow")
    if stage == "workflow":
        monkeypatch.setattr(
            ops.workflow,
            "register",
            lambda *a, **k: (_ for _ in ()).throw(ops.RemoteFailure("workflow down")),
        )
    else:
        monkeypatch.setattr(ops.workflow, "register", lambda *a, **k: {})
    if stage == "comfy":
        monkeypatch.setattr(ops.comfy, "stage_launch", lambda *a, **k: {})
        monkeypatch.setattr(
            ops.comfy, "start", lambda *a, **k: (_ for _ in ()).throw(ops.RemoteFailure("comfy down"))
        )
    else:
        monkeypatch.setattr(ops.comfy, "stage_launch", lambda *a, **k: {})
        monkeypatch.setattr(ops.comfy, "start", lambda *a, **k: {})
    if stage == "tunnel":
        monkeypatch.setattr(
            ops.tunnel, "ensure", lambda **k: (_ for _ in ()).throw(ops.RemoteFailure("tunnel down"))
        )
    else:
        monkeypatch.setattr(ops.tunnel, "ensure", lambda **k: object())
    if stage == "health":
        monkeypatch.setattr(ops, "_wait_health", lambda *a, **k: (_ for _ in ()).throw(ops.RemoteFailure("health down")))
    else:
        monkeypatch.setattr(ops, "_wait_health", lambda *a, **k: {"system": {"comfyui_version": "fake"}})
    if stage == "api":
        monkeypatch.setattr(ops, "_fetch_api", lambda *a, **k: (_ for _ in ()).throw(ops.RemoteFailure("api down")))
    else:
        monkeypatch.setattr(ops, "_fetch_api", lambda *a, **k: {"object_info": "/oi", "system_stats": "/ss"})
    monkeypatch.setattr(ops.workflow, "list_userdata", lambda **k: "UI_FILE_ONLY")
    monkeypatch.setattr(ops, "_run_probe", lambda *a, **k: _probe(ctx.profile))
    with pytest.raises(Exception):
        ops._finalize_after_download(
            ctx,
            root="/workspace/.avs-i1",
            workspace="/workspace",
            profile=ctx.profile,
            state=ctx.state,
        )
    assert ctx.state["readiness"]["ready"] is False
    assert ctx.state["readiness"]["status"] != "READY"
    assert ctx.state["readiness_current"] is False
    assert any(
        item.get("report", {}).get("status") == "READY"
        for item in ctx.state.get("readiness_history", [])
    )


def test_ready_resume_refuses_when_instance_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anime_studio.lock import InstanceLock

    ctx = _ready_ctx(tmp_path, with_workflow=True)
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    lock = InstanceLock(ctx.run_dir / ".install.lock")
    lock.acquire()
    calls: list[str] = []
    monkeypatch.setattr(
        ops, "_observe_attempt", lambda *a, **k: calls.append("observe")
    )
    monkeypatch.setattr(
        ops, "_finalize_after_download", lambda *a, **k: calls.append("finalize")
    )
    try:
        with pytest.raises(ops.BlockedError):
            ops.resume(ctx)
        assert calls == []
    finally:
        lock.release()


def test_status_never_surfaces_cached_ready_when_phase_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    ctx.state["phase"] = "VERIFYING"
    ctx.state["readiness"] = {"status": "READY", "ready": True, "warnings": []}
    monkeypatch.setattr(ops.download, "read_optional", lambda *a, **k: None)
    monkeypatch.setattr(ops.comfy, "read_owner", lambda *a, **k: None)
    payload = ops.status(ctx, offline=True)
    assert payload["readiness"]["status"] == "UNCONFIRMED"
    assert payload["readiness"]["ready"] is False
    assert payload["readiness_current"] is False

def _download_loop_context(tmp_path: Path) -> ops.Context:
    ctx = _context(tmp_path)
    ctx.state["phase"] = "INSTALLING"
    ctx.state["artifacts"] = {
        "setup": "COMPLETE",
        "download_attempt": {
            "id": "0001",
            "created_at": time.time() - 10,
            "deadline_epoch": time.time() + 3600,
            "dir": "/workspace/.avs-i1/attempts/0001",
        },
    }
    return ctx


def _patch_download_loop(monkeypatch: pytest.MonkeyPatch, decisions: list) -> None:
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **kwargs: {})
    monkeypatch.setattr(ops, "_stage_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops, "_run_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *args, **kwargs: {
            "paths": {"attempt": "/workspace/.avs-i1/attempts/0001"},
            "job_sha256": "a" * 64,
            "launch_sha256": None,
        },
    )
    iterator = iter(decisions)
    monkeypatch.setattr(
        ops.download, "inspect_attempt", lambda *args, **kwargs: next(iterator)
    )
    monkeypatch.setattr(
        ops,
        "_finalize_after_download",
        lambda ctx, **kwargs: {"readiness": {"ready": False}, "state": ctx.state},
    )


def test_running_live_status_is_verified_and_not_duplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _download_loop_context(tmp_path)
    decisions = [
        {
            "state": "OK",
            "process": "LIVE",
            "status": {"status": "RUNNING"},
            "progress": {"stage": "DOWNLOADING"},
            "heartbeat": {},
        },
        {
            "state": "OK",
            "process": "ABSENT",
            "status": {"status": "COMPLETE"},
            "progress": {},
            "heartbeat": {},
        },
    ]
    _patch_download_loop(monkeypatch, decisions)
    starts = []
    monkeypatch.setattr(ops.download, "start", lambda *a, **k: starts.append(k))
    ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    assert starts == []


def test_old_live_worker_is_attached_without_restaging_changed_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _download_loop_context(tmp_path)
    ctx.state["artifacts"]["download_attempt"]["launch_sha256"] = "a" * 64
    decisions = [
        {
            "state": "OK",
            "process": "LIVE",
            "status": {"status": "RUNNING"},
            "progress": {"stage": "DOWNLOADING"},
            "heartbeat": {},
        },
        {
            "state": "OK",
            "process": "ABSENT",
            "status": {"status": "COMPLETE"},
            "progress": {},
            "heartbeat": {},
        },
    ]
    iterator = iter(decisions)
    stages: list[dict] = []
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **kwargs: {})
    monkeypatch.setattr(ops, "_stage_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops, "_run_setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops.download, "ensure_guard", lambda *a, **k: None)
    monkeypatch.setattr(
        ops.download, "inspect_attempt", lambda *a, **k: next(iterator)
    )
    monkeypatch.setattr(
        ops.download, "stage", lambda *a, **k: stages.append(k)
    )
    monkeypatch.setattr(
        ops,
        "_finalize_after_download",
        lambda ctx, **kwargs: {"readiness": {"ready": False}, "state": ctx.state},
    )
    ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    assert stages == []


def test_dead_attempt_resumes_without_restage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _download_loop_context(tmp_path)
    ctx.state["artifacts"]["download_attempt"]["launch_sha256"] = "a" * 64
    decisions = [
        {
            "state": "OK",
            "process": "DEAD",
            "status": {"status": "RUNNING"},
            "progress": {},
            "heartbeat": {"heartbeat_at": time.time()},
            "manifest_sha256": "b" * 64,
        },
        {
            "state": "OK",
            "process": "ABSENT",
            "status": {"status": "COMPLETE"},
            "progress": {},
            "heartbeat": {},
        },
    ]
    iterator = iter(decisions)
    monkeypatch.setattr(
        ops.download, "inspect_attempt", lambda *a, **k: next(iterator)
    )
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no restage")),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        ops.download,
        "recover",
        lambda *a, **k: calls.append("recover") or {"status": "RECOVERED_INTERRUPTED"},
    )
    monkeypatch.setattr(
        ops.download,
        "start",
        lambda *a, **k: calls.append("start") or {"status": "STARTED"},
    )
    monkeypatch.setattr(
        ops,
        "_finalize_after_download",
        lambda ctx, **kwargs: {"readiness": {"ready": True}, "state": ctx.state},
    )
    monkeypatch.setattr(ops.time, "sleep", lambda seconds: None)
    result = ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    assert result["readiness"]["ready"] is True
    assert calls == ["recover", "start"]


def test_running_dead_status_is_recovered_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _download_loop_context(tmp_path)
    decisions = [
        {
            "state": "OK",
            "process": "DEAD",
            "status": {"status": "RUNNING"},
            "progress": {},
            "heartbeat": {"heartbeat_at": time.time()},
        },
        {
            "state": "OK",
            "process": "ABSENT",
            "status": {"status": "COMPLETE"},
            "progress": {},
            "heartbeat": {},
        },
    ]
    _patch_download_loop(monkeypatch, decisions)
    calls = []
    monkeypatch.setattr(ops.download, "recover", lambda *a, **k: calls.append(("recover", k.get("stale_seconds"))) or {"status": "RECOVERED_INTERRUPTED"})
    monkeypatch.setattr(ops.download, "start", lambda *a, **k: calls.append(("start", k.get("resume_dead"))) or {"status": "STARTED"})
    ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    assert calls == [("recover", 0.0), ("start", True)]


def test_failed_but_live_worker_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _download_loop_context(tmp_path)
    _patch_download_loop(
        monkeypatch,
        [
            {
                "state": "OK",
                "process": "LIVE",
                "status": {"status": "FAILED"},
                "progress": {},
                "heartbeat": {},
            }
        ],
    )
    monkeypatch.setattr(
        ops.download,
        "start",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resume a live worker")),
    )
    with pytest.raises(ops.RemoteFailure):
        ops._continue_install(ctx, deadline_epoch=time.time() + 3600)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_ops_disconnect_resume_uses_real_owned_tunnel_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    ctx.state["run_id"] = "r"
    ctx.state["nonce"] = "n"
    ctx.state = state_mod.transition(ctx.state, "INSTALLED", stage="INSTALLED")
    ctx.state["artifacts"] = {
        "setup": "COMPLETE",
        "download": "COMPLETE",
        "workflow": {
            "ui_status": "UI_GRAPH_NOT_VERIFIED",
            "ui_filename": "x.json",
            "ui_path": "/ui",
            "graph_rendered": False,
        },
    }
    identity_file = tmp_path / "id"
    known_hosts = tmp_path / "known"
    identity_file.write_text("x", encoding="utf-8")
    known_hosts.write_text("x", encoding="utf-8")
    ctx.identity = SimpleNamespace(
        host="example.test",
        user="root",
        port=22,
        identity_file=identity_file,
        known_hosts=known_hosts,
        destination="root@example.test",
    )
    ctx.profile.data["runtime"]["local_tunnel_port"] = _free_port()
    ctx.profile.data["runtime"]["comfy_remote_port"] = _free_port()
    fake_ssh = tmp_path / "fake_ssh.py"
    fake_ssh.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    real_ensure = ops.tunnel.ensure
    real_stop = ops.tunnel.stop
    monkeypatch.setattr(
        ops.tunnel,
        "ensure",
        lambda **kwargs: real_ensure(
            **kwargs, ssh_executable=sys.executable, ssh_args_prefix=(str(fake_ssh),)
        ),
    )
    monkeypatch.setattr(
        ops.tunnel,
        "stop",
        lambda **kwargs: real_stop(
            **kwargs, ssh_executable=sys.executable, ssh_args_prefix=(str(fake_ssh),)
        ),
    )
    monkeypatch.setattr(ops.comfy, "stage_launch", lambda *a, **k: {})
    monkeypatch.setattr(ops.comfy, "start", lambda *a, **k: {})
    monkeypatch.setattr(ops.comfy, "read_owner", lambda *a, **k: {"pid": 1})
    monkeypatch.setattr(ops, "_wait_health", lambda *a, **k: {"system": {"comfyui_version": "fake"}})
    monkeypatch.setattr(ops, "_fetch_api", lambda *a, **k: {"object_info": "/oi", "system_stats": "/ss"})
    monkeypatch.setattr(ops.workflow, "list_userdata", lambda **k: "UI_FILE_ONLY")
    monkeypatch.setattr(ops, "_run_probe", lambda *a, **k: _probe(ctx.profile))
    try:
        first = ops.resume(ctx)
        assert first["readiness"]["ready"] is True
        first_owner = json.loads(
            (ctx.run_dir / "tunnel-owner.json").read_text(encoding="utf-8")
        )
        assert ops.disconnect(ctx)["status"] == "DISCONNECTED"
        assert (ctx.run_dir / "tunnel-state.json").is_file()
        ctx.state = state_mod.transition(ctx.state, "VERIFYING", stage="RESUME")
        second = ops.resume(ctx)
        assert second["readiness"]["ready"] is True
        second_owner = json.loads(
            (ctx.run_dir / "tunnel-owner.json").read_text(encoding="utf-8")
        )
        assert first_owner["monitor_pid"] != second_owner["monitor_pid"]
        archives = list((ctx.run_dir / "archive").glob("owned-tunnel-*"))
        assert len(archives) == 1
        assert (archives[0] / "tunnel-state.json").is_file()
    finally:
        try:
            ops.disconnect(ctx)
        except Exception:
            pass

