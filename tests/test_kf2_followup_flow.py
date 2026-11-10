from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from scripts.kf2 import cli
from scripts.kf2.store import ProjectStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROJECT = ROOT / "data/projects/forest-river"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_fixture(tmp_path: Path):
    target = tmp_path / "projects" / "forest-river"
    shutil.copytree(SOURCE_PROJECT, target)
    store = ProjectStore(tmp_path / "projects")
    return store, store.load("forest-river"), target


def _fill_clone(store: ProjectStore, project: dict) -> dict:
    library = list(project.get("library") or [])
    river = next(item["id"] for item in library if item["role"] == "provided_keyframe")
    girl = [item["id"] for item in library if item.get("subject_id") == "girl"]
    monster = next(item["id"] for item in library if item.get("subject_id") == "monster")
    env = next(item["id"] for item in library if item["role"] == "environment")
    scenes = project["scenes"]
    scenes[0]["initial"]["asset_id"] = river
    visible = {
        "shot-01": [],
        "shot-02": ["girl"],
        "shot-03": ["monster"],
        "shot-04": ["girl", "monster"],
        "shot-05": ["girl"],
        "shot-06": ["girl", "monster"],
    }
    for scene in scenes:
        scene["action"] = scene.get("action") or "Explicit action."
        scene["camera"] = scene.get("camera") or "Explicit camera."
        scene["shared_state"] = scene.get("shared_state") or "Explicit common state."
        scene["subjects_visible"] = visible[scene["id"]]
        scene["initial"]["state_a"] = "Explicit A start state."
        scene["final"]["state_b"] = "Explicit B end state."
        scene["initial"]["instruction_mode"] = "states"
        scene["final"]["instruction_mode"] = "states"
    scenes[1]["initial"]["references"] = [{"asset_id": girl[0], "role": "identity"}]
    scenes[1]["final"]["references"] = [
        {"asset_id": girl[0], "role": "identity"},
        {"asset_id": girl[1], "role": "identity"},
    ]
    scenes[2]["initial"]["references"] = [{"asset_id": monster, "role": "identity"}]
    scenes[2]["final"]["references"] = [{"asset_id": env, "role": "environment"}]
    scenes[3]["initial"]["references"] = [{"asset_id": env, "role": "environment"}]
    scenes[3]["final"]["references"] = [
        {"asset_id": girl[0], "role": "identity"},
        {"asset_id": monster, "role": "identity"},
    ]
    scenes[4]["initial"]["references"] = [{"asset_id": env, "role": "environment"}]
    scenes[4]["final"]["references"] = [{"asset_id": girl[0], "role": "identity"}]
    scenes[5]["initial"]["references"] = [{"asset_id": env, "role": "environment"}]
    scenes[5]["final"]["references"] = [{"asset_id": girl[1], "role": "identity"}]
    project["generation_limit"] = 11
    return store.save(
        "forest-river", project, expected_revision=int(project["revision"])
    )


def test_forest_river_fixture_clone_full_offline_dryrun(tmp_path: Path) -> None:
    original_project = SOURCE_PROJECT / "project.json"
    before = _sha(original_project)
    original_plan = SOURCE_PROJECT / "plan.json"
    # Owner may already have prepared the real project. Both states must be
    # preserved; never require deleting their plan to make a test pass.
    plan_before = _sha(original_plan) if original_plan.exists() else None
    store, project, target = _project_fixture(tmp_path)
    project = _fill_clone(store, project)
    preview = cli._summary(project, target)
    assert preview["status"] in {"READY_TO_PREPARE", "STALE_PREPARE"}
    assert preview["can_prepare"] is True
    assert preview["next_fix"] is None

    code, prepared = cli._cmd_prepare(
        SimpleNamespace(
            id="forest-river",
            projects_dir=str(tmp_path / "projects"),
            kf1_state_dir=str(tmp_path / "state"),
            json=True,
        )
    )
    assert code == 0
    assert prepared["status"] == "PREPARED"
    assert prepared["dryrun"]["status"] == "PLAN_ONLY"
    plan = json.loads((target / "plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == "anime-studio/keyframe-plan/v2"
    refs = plan.get("references") or []
    assert any(item.get("role") == "identity" for item in refs)
    assert plan.get("background_reference") or any(
        item.get("role") == "environment" for item in refs
    )
    scene_b = plan["scenes"][1]["b"]
    assert scene_b["reference_roles"][0] == "anchor"
    assert len(scene_b["reference_roles"]) == 3
    assert _sha(original_project) == before
    assert (_sha(original_plan) if original_plan.exists() else None) == plan_before
