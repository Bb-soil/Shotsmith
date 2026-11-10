from __future__ import annotations

import json
from pathlib import Path

from scripts.kf2.cli import _add_image_to_project, _save
from scripts.kf2.compile import validate_and_compile
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import _png


def _base(tmp_path: Path, *, orientation: str = "portrait"):
    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("p1", title="Мой проект", orientation=orientation)
    first = _add_image_to_project(
        project,
        project_dir,
        role="provided_keyframe",
        path=str(tmp_path / "first.png"),
    )
    identity = _add_image_to_project(
        project,
        project_dir,
        role="object",
        path=str(tmp_path / "object.png"),
        subject_id="car",
        subject_label="car",
        view="front",
    )
    project = _save(store, project)
    return store, project, project_dir, first, identity


def _write_images(tmp_path: Path) -> None:
    (tmp_path / "first.png").write_bytes(_png(704, 1280))
    (tmp_path / "object.png").write_bytes(_png(64, 64, (20, 30, 40)))


def _scene(first_id: str, identity_id: str, *, final_kind: str = "continue") -> dict:
    return {
        "id": "shot-01",
        "description": "A car on a road.",
        "action": "The car approaches.",
        "camera": "Fixed camera.",
        "subjects_visible": ["car"],
        "subjects_absent": [],
        "shared_invariants": ["identity", "style"],
        "shared_allowed_changes": ["position", "apparent_scale"],
        "initial": {
            "kind": "first_frame",
            "asset_id": first_id,
            "instruction": "",
            "references": [],
            "invariants": [],
            "allowed_changes": [],
            "use_previous_anchor": False,
            "require_anchor_approval": True,
        },
        "final": {
            "kind": final_kind,
            "asset_id": None,
            "instruction": "The same car approaches the fixed camera.",
            "references": [{"asset_id": identity_id, "role": "identity"}],
            "invariants": [],
            "allowed_changes": [],
            "require_anchor_approval": False,
        },
    }


def test_compile_exact_v2_portrait_and_shared_context(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _store, project, project_dir, first, identity = _base(tmp_path)
    project["scenes"] = [_scene(first["id"], identity["id"])]
    project = _save(_store, project)
    result = validate_and_compile(project, project_dir)
    assert result["blockers"] == []
    plan = result["plan"]
    assert plan["schema"] == "anime-studio/keyframe-plan/v2"
    assert plan["title"] == "Мой проект"
    assert plan["output_policy"]["candidate_geometry"] == [704, 1280]
    assert plan["output_policy"]["video_geometry"] == [448, 816]
    assert plan["scenes"][0]["b"]["shot_mode"] == "continue_shot"
    assert plan["scenes"][0]["b"]["reference_roles"] == ["anchor", "character_reference"]
    assert "camera" in plan["scenes"][0]["b"]["invariants"]
    assert plan["scenes"][0]["b"]["require_anchor_approval"] is False
    instruction = plan["scenes"][0]["b"]["edit_instruction"]
    assert instruction.startswith("The same car approaches the fixed camera.")
    assert "Action: The car approaches." in instruction
    assert "Camera: Fixed camera." in instruction
    assert "Visible subjects: car" in instruction
    assert "heroine" not in json.dumps(plan, ensure_ascii=False).lower()


def test_compile_landscape_uses_exact_mandated_dims(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _store, project, project_dir, first, identity = _base(tmp_path, orientation="landscape")
    project["scenes"] = [_scene(first["id"], identity["id"])]
    project = _save(_store, project)
    result = validate_and_compile(project, project_dir)
    assert result["blockers"] == []
    assert result["plan"]["output_policy"]["candidate_geometry"] == [1280, 704]
    assert result["plan"]["output_policy"]["video_geometry"] == [816, 448]


def test_b_camera_change_is_actionable_blocker_not_silent_new_shot(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _store, project, project_dir, first, identity = _base(tmp_path)
    scene = _scene(first["id"], identity["id"])
    scene["final"]["allowed_changes"] = ["camera"]
    project["scenes"] = [scene]
    result = validate_and_compile(project, project_dir)
    assert result["plan"] is None
    assert any(item["code"] == "B_CAMERA_CHANGE_UNSUPPORTED" for item in result["blockers"])


def test_a_continue_and_b_new_shot_are_rejected(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _store, project, project_dir, first, identity = _base(tmp_path)
    scene = _scene(first["id"], identity["id"])
    scene["initial"] = {
        **scene["initial"],
        "kind": "continue",
        "instruction": "continue",
    }
    project["scenes"] = [scene]
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] in {"A_NEW_SHOT_ONLY", "A1_SOURCE"} for item in result["blockers"])

    scene = _scene(first["id"], identity["id"], final_kind="new_shot")
    project["scenes"] = [scene]
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "B_CONTINUE_ONLY" for item in result["blockers"])


def test_frame_reference_limit_and_duplicate_role_are_not_silently_dropped(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _store, project, project_dir, first, identity = _base(tmp_path)
    (tmp_path / "env.png").write_bytes(_png(32, 32, (1, 2, 3)))
    (tmp_path / "style.png").write_bytes(_png(32, 32, (4, 5, 6)))
    env = _add_image_to_project(project, project_dir, role="environment", path=str(tmp_path / "env.png"))
    style = _add_image_to_project(project, project_dir, role="style", path=str(tmp_path / "style.png"))
    scene = _scene(first["id"], identity["id"])
    scene["final"]["references"] = [
        {"asset_id": identity["id"], "role": "identity"},
        {"asset_id": env["id"], "role": "environment"},
        {"asset_id": style["id"], "role": "style"},
    ]
    project["scenes"] = [scene]
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "REF_LIMIT" for item in result["blockers"])
    scene["final"]["references"] = [
        {"asset_id": identity["id"], "role": "identity"},
        {"asset_id": identity["id"], "role": "identity"},
    ]
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "DUPLICATE_ASSET_REF" for item in result["blockers"])


def test_different_identity_assets_across_scenes_are_allowed(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _store, project, project_dir, first, identity = _base(tmp_path)
    (tmp_path / "other.png").write_bytes(_png(64, 64, (90, 90, 90)))
    other = _add_image_to_project(
        project,
        project_dir,
        role="object",
        path=str(tmp_path / "other.png"),
        subject_id="other",
        subject_label="other",
    )
    scene1 = _scene(first["id"], identity["id"])
    scene2 = _scene(first["id"], other["id"])
    scene2["id"] = "shot-02"
    scene2["subjects_visible"] = ["other"]
    scene2["initial"]["kind"] = "new_shot"
    scene2["initial"]["asset_id"] = None
    scene2["initial"]["instruction"] = "New camera."
    scene2["initial"]["use_previous_anchor"] = False
    scene2["initial"]["references"] = [{"asset_id": other["id"], "role": "identity"}]
    project["scenes"] = [scene1, scene2]
    before = [item["id"] for item in project["library"]]
    result = validate_and_compile(project, project_dir)
    assert result["plan"] is not None
    assert not any(item["code"] == "GLOBAL_ROLE_SELECTION_CONFLICT" for item in result["blockers"])
    assert [item["id"] for item in project["library"]] == before
    assert {identity["id"], other["id"]}.issubset(set(before))


def test_environment_only_project_is_valid(tmp_path: Path) -> None:
    _store = ProjectStore(tmp_path / "projects")
    project, project_dir = _store.create("p1", title="Environment", orientation="portrait")
    (tmp_path / "env-first.png").write_bytes(_png(704, 1280, (30, 40, 50)))
    (tmp_path / "env-ref.png").write_bytes(_png(64, 64, (60, 70, 80)))
    first = _add_image_to_project(
        project, project_dir, role="provided_keyframe", path=str(tmp_path / "env-first.png")
    )
    env = _add_image_to_project(
        project, project_dir, role="environment", path=str(tmp_path / "env-ref.png")
    )
    project["scenes"] = [
        {
            "id": "shot-01",
            "description": "Empty street.",
            "action": "",
            "camera": "Fixed.",
            "subjects_visible": [],
            "subjects_absent": [],
            "shared_invariants": ["style"],
            "shared_allowed_changes": ["lighting"],
            "initial": {
                "kind": "first_frame",
                "asset_id": first["id"],
                "instruction": "",
                "references": [],
                "invariants": [],
                "allowed_changes": [],
                "use_previous_anchor": False,
                "require_anchor_approval": True,
            },
            "final": {
                "kind": "continue",
                "asset_id": None,
                "instruction": "Keep the camera; change only lighting.",
                "references": [{"asset_id": env["id"], "role": "environment"}],
                "invariants": [],
                "allowed_changes": [],
                "require_anchor_approval": False,
            },
        }
    ]
    project = _save(_store, project)
    result = validate_and_compile(project, project_dir)
    assert result["blockers"] == []
    assert result["plan"]["scenes"][0]["b"]["reference_roles"] == ["anchor", "background_reference"]


def test_provided_later_frame_requires_provenance(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _store, project, project_dir, first, identity = _base(tmp_path)
    provided = _add_image_to_project(
        project,
        project_dir,
        role="provided_keyframe",
        path=str(tmp_path / "object.png"),
    )
    scene = _scene(first["id"], identity["id"])
    scene["id"] = "shot-01"
    scene["final"] = {
        "kind": "provided_keyframe",
        "asset_id": provided["id"],
        "instruction": "",
        "references": [],
        "invariants": [],
        "allowed_changes": [],
        "require_anchor_approval": False,
    }
    project["scenes"] = [scene]
    result = validate_and_compile(project, project_dir)
    assert any(
        item["code"] in {"PROVIDED_PROVENANCE_REQUIRED", "PROVIDED_PROVENANCE_FRAME"}
        for item in result["blockers"]
    )


def test_first_frame_role_and_missing_instruction_are_blocked(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _store, project, project_dir, first, identity = _base(tmp_path)
    scene = _scene(first["id"], identity["id"])
    scene["initial"]["asset_id"] = identity["id"]  # wrong role
    project["scenes"] = [scene]
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "FIRST_FRAME_ROLE" for item in result["blockers"])

    scene = _scene(first["id"], identity["id"])
    scene["final"]["instruction"] = ""
    project["scenes"] = [scene]
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "MISSING_INSTRUCTION" for item in result["blockers"])


def test_explicit_approval_on_provided_anchor_is_actionable_blocker(tmp_path: Path) -> None:
    _write_images(tmp_path)
    store, project, project_dir, first, identity = _base(tmp_path)
    scene = _scene(first["id"], identity["id"])
    scene["final"]["require_anchor_approval"] = True
    project["scenes"] = [scene]
    result = validate_and_compile(project, project_dir)
    assert result["plan"] is None
    assert any(item["code"] == "ANCHOR_APPROVAL_UNSUPPORTED" for item in result["blockers"])
