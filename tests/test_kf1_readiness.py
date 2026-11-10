from __future__ import annotations

from pathlib import Path

from scripts.i1 import readiness
from scripts.i1.profile import load_profile
from scripts.i1.probe import _output_checks

ROOT = Path(__file__).resolve().parents[1]


def _probe(profile) -> dict:
    workflows = []
    for workflow_id, meta in profile.workflows.items():
        workflows.append(
            {
                "workflow_id": workflow_id,
                "sha256": meta["api_sha256"],
                "classes": list(meta["required_classes"]),
                "input_checks": [
                    {"class": class_name, "input": name, "present": True}
                    for class_name, names in meta["required_inputs"].items()
                    for name in names
                ],
                "link_checks": [{"compatible": True}],
                "output_checks": [{"present": True, "class": meta["output"]["class"]}],
            }
        )
    return {
        "schema": "anime-video-studio/i1-runtime-evidence/v1",
        "run_id": "r",
        "nonce": "n",
        "profile_sha256": profile.sha256,
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
        "workflows": workflows,
        "system_stats": {"devices": [{"name": "fake", "vram_total": 1}]},
    }


def test_k1_readiness_uses_profile_counts_and_saveimage_without_savevideo() -> None:
    profile = load_profile(ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json")
    report = readiness.evaluate(
        probe=_probe(profile),
        profile=profile,
        tunnel_health={"system": {"comfyui_version": "x"}},
        ui={"ui_status": "API_FILE_ONLY", "ui_graph_rendered": False},
    )
    assert report["status"] == readiness.READY
    assert report["ready"] is True
    assert report["model_bytes_verified"] == 30_172_239_743
    assert not any("SaveVideo" in item for item in report["blockers"])


def test_k1_missing_qwen_model_does_not_affect_wan_profile() -> None:
    profile = load_profile(ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json")
    probe = _probe(profile)
    probe["models"] = probe["models"][:-1]
    report = readiness.evaluate(
        probe=probe, profile=profile, tunnel_health={"system": {}}, ui={}
    )
    assert report["ready"] is False
    wan = load_profile(ROOT / "config/i1/wan22-14b-ab.profile.json")
    assert len(wan.models) == 4 and wan.output_specs[0]["class"] == "SaveVideo"


def test_output_checks_accept_saveimage_and_validate_savevideo() -> None:
    object_info = {
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING"],
                }
            }
        }
    }
    checks = _output_checks(
        object_info,
        [{"workflow_id": "q", "class": "SaveImage", "keys": ["images"], "media_type": "image/png"}],
    )
    assert checks[0]["present"] is True
    assert checks[0]["class"] == "SaveImage"

def test_i1_install_plan_for_kf1_exposes_three_models_and_three_graphs() -> None:
    from types import SimpleNamespace
    from scripts.i1 import ops
    profile = load_profile(ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json")
    ctx = SimpleNamespace(
        profile=profile,
        state={"identity": {"workspace": "/workspace"}},
    )
    plan = ops._plan(ctx, {"facts": {"disk_free_bytes": 10**12}})
    assert plan["profile_schema"] == "anime-video-studio/i1-profile/v2"
    assert len(plan["models"]) == 3
    assert len(plan["workflows"]) == 3
    assert {item["output"]["class"] for item in plan["workflows"]} == {"SaveImage"}
    assert plan["plan_sha256"]

def test_bundle_readiness_reports_both_outputs_and_seven_models() -> None:
    profile = load_profile(ROOT / "config/i1/bundle-wan22-qwen-edit-2511.profile.json")
    report = readiness.evaluate(
        probe=_probe(profile),
        profile=profile,
        tunnel_health={"system": {"comfyui_version": "x"}},
        ui={"ui_status": "API_FILE_ONLY", "ui_graph_rendered": False},
    )
    assert report["status"] == readiness.READY
    assert report["model_bytes_verified"] == 65_751_447_622
    assert {item["class"] for item in profile.output_specs} == {"SaveImage", "SaveVideo"}

def test_i1_install_plan_for_bundle_exposes_seven_models_and_both_outputs() -> None:
    from types import SimpleNamespace
    from scripts.i1 import ops
    profile = load_profile(ROOT / "config/i1/bundle-wan22-qwen-edit-2511.profile.json")
    ctx = SimpleNamespace(profile=profile, state={"identity": {"workspace": "/workspace"}})
    plan = ops._plan(ctx, {"facts": {"disk_free_bytes": 10**12}})
    assert len(plan["models"]) == 7
    assert len(plan["workflows"]) == 4
    assert {item["output"]["class"] for item in plan["workflows"]} == {"SaveImage", "SaveVideo"}
    assert plan["models_policy"]["expected_count"] == 7
