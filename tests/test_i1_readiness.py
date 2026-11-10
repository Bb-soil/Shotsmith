from __future__ import annotations

import json
from pathlib import Path

from scripts.i1 import readiness
from scripts.i1.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]


def _profile():
    return load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")


def _probe() -> dict:
    profile = _profile()
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
            {
                "target": item["target"],
                "actual_size_bytes": item["size_bytes"],
                "actual_sha256": item["sha256"],
            }
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
            "savevideo_dynamic": {
                "selected": {"format": "mp4", "codec": "h264", "encoding": "auto"},
                "issues": [],
            },
        },
        "system_stats": {"devices": [{"name": "fake", "vram_total": 1}]},
        "workflow": {
            "sha256": profile.workflow["api_sha256"],
            "classes": list(profile.required_classes),
        },
    }


def test_ready_requires_actual_probe_not_just_files() -> None:
    report = readiness.evaluate(probe=None, profile=_profile(), tunnel_health=None, ui={})
    assert report["status"] == readiness.INSTALLED
    assert report["ready"] is False


def test_ready_with_complete_actual_facts_and_ui_warning() -> None:
    report = readiness.evaluate(
        probe=_probe(),
        profile=_profile(),
        tunnel_health={"system": {"comfyui_version": "x"}},
        ui={"ui_status": "DERIVED_FROM_API_NOT_LIVE_VERIFIED", "ui_graph_rendered": False},
    )
    assert report["status"] == readiness.READY
    assert report["ready"] is True
    assert report["ui_graph_rendered"] is False
    assert any("UI graph" in item for item in report["warnings"])


def test_missing_model_or_input_blocks_ready() -> None:
    probe = _probe()
    probe["models"] = probe["models"][:-1]
    report = readiness.evaluate(
        probe=probe,
        profile=_profile(),
        tunnel_health={"system": {}},
        ui={"ui_status": "UI_GRAPH_NOT_VERIFIED"},
    )
    assert report["status"] == readiness.INSTALLED
    assert report["ready"] is False
    assert any("four model" in item for item in report["blockers"])
    probe2 = _probe()
    probe2["object_info"]["input_checks"][0]["present"] = False
    report2 = readiness.evaluate(
        probe=probe2,
        profile=_profile(),
        tunnel_health={"system": {}},
        ui={},
    )
    assert report2["ready"] is False


def test_offline_verified_is_not_ready() -> None:
    report = readiness.offline_report()
    assert report["status"] == readiness.OFFLINE_VERIFIED
    assert report["ready"] is False
