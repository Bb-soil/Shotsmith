from __future__ import annotations

from pathlib import Path

from scripts.kf2.cli import _add_image_to_project, _save
from scripts.kf2.compile import validate_and_compile
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import _png


def _write(tmp_path: Path, name: str, color: tuple[int, int, int]) -> Path:
    path = tmp_path / name
    path.write_bytes(_png(64, 64, color))
    return path


def _base(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("p1", title="refs", orientation="portrait")
    first = _add_image_to_project(
        project, project_dir, role="provided_keyframe", path=str(_write(tmp_path, "first.png", (1, 2, 3)))
    )
    girl = _add_image_to_project(
        project,
        project_dir,
        role="character",
        path=str(_write(tmp_path, "girl.png", (10, 20, 30))),
        subject_id="girl",
        subject_label="girl",
        view="front",
    )
    girl_side = _add_image_to_project(
        project,
        project_dir,
        role="character",
        path=str(_write(tmp_path, "girl-side.png", (10, 30, 20))),
        subject_id="girl",
        subject_label="girl",
        view="side",
    )
    monster = _add_image_to_project(
        project,
        project_dir,
        role="creature",
        path=str(_write(tmp_path, "monster.png", (40, 10, 10))),
        subject_id="monster",
        subject_label="monster",
        view="front",
    )
    env = _add_image_to_project(
        project, project_dir, role="environment", path=str(_write(tmp_path, "env.png", (5, 6, 7)))
    )
    project = _save(store, project)
    return store, project, project_dir, first, girl, girl_side, monster, env


def _scene(first_id: str, refs: list[dict], visible: list[str]) -> dict:
    return {
        "id": "shot-01",
        "description": "Explicit scene.",
        "action": "Explicit action.",
        "camera": "Explicit camera.",
        "subjects_visible": visible,
        "subjects_absent": [],
        "shared_state": "",
        "shared_invariants": [],
        "shared_allowed_changes": [],
        "initial": {
            "kind": "first_frame",
            "asset_id": first_id,
            "instruction": "",
            "references": [],
            "invariants": [],
            "allowed_changes": [],
            "use_previous_anchor": False,
            "require_anchor_approval": False,
        },
        "final": {
            "kind": "continue",
            "asset_id": None,
            "instruction": "Explicit B instruction.",
            "references": refs,
            "invariants": [],
            "allowed_changes": [],
            "require_anchor_approval": False,
        },
    }


def test_two_identity_views_same_frame_compile_with_distinct_picture_ids(tmp_path: Path) -> None:
    store, project, project_dir, first, girl, girl_side, _monster, _env = _base(tmp_path)
    project["scenes"] = [
        _scene(first["id"], [{"asset_id": girl["id"], "role": "identity"}, {"asset_id": girl_side["id"], "role": "identity"}], ["girl"])
    ]
    project = _save(store, project)
    result = validate_and_compile(project, project_dir)
    assert result["blockers"] == []
    plan = result["plan"]
    assert plan["scenes"][0]["b"]["reference_roles"] == [
        "anchor",
        "character_reference",
        f"ref-{girl_side['id']}",
    ]
    assert plan["references"][0]["id"] == f"ref-{girl_side['id']}"
    assert plan["references"][0]["role"] == "identity"
    assert result["reference_catalog_count"] == 2


def test_two_different_identity_subjects_same_frame_compile(tmp_path: Path) -> None:
    store, project, project_dir, first, girl, _girl_side, monster, _env = _base(tmp_path)
    project["scenes"] = [
        _scene(first["id"], [{"asset_id": girl["id"], "role": "identity"}, {"asset_id": monster["id"], "role": "identity"}], ["girl", "monster"])
    ]
    project = _save(store, project)
    result = validate_and_compile(project, project_dir)
    assert result["blockers"] == []
    assert result["plan"]["scenes"][0]["b"]["reference_roles"] == [
        "anchor",
        "character_reference",
        f"ref-{monster['id']}",
    ]


def test_same_asset_twice_and_same_content_are_actionable_blockers(tmp_path: Path) -> None:
    store, project, project_dir, first, girl, _girl_side, _monster, _env = _base(tmp_path)
    project["scenes"] = [
        _scene(first["id"], [{"asset_id": girl["id"], "role": "identity"}, {"asset_id": girl["id"], "role": "identity"}], ["girl"])
    ]
    project = _save(store, project)
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "DUPLICATE_ASSET_REF" for item in result["blockers"])

    same = _write(tmp_path, "same.png", (77, 77, 77))
    clone = _add_image_to_project(
        project,
        project_dir,
        role="character",
        path=str(same),
        subject_id="girl",
        subject_label="girl",
        view="front-copy",
    )
    project["scenes"][0]["final"]["references"] = [
        {"asset_id": girl["id"], "role": "identity"},
        {"asset_id": clone["id"], "role": "identity"},
    ]
    project = _save(store, project)
    # girl and clone have different bytes; make a true same-content association
    same2 = _add_image_to_project(
        project,
        project_dir,
        role="character",
        path=str(tmp_path / "girl.png"),
        subject_id="girl",
        subject_label="girl",
        view="front-again",
    )
    project["scenes"][0]["final"]["references"] = [
        {"asset_id": girl["id"], "role": "identity"},
        {"asset_id": same2["id"], "role": "identity"},
    ]
    project = _save(store, project)
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "DUPLICATE_REF_CONTENT" for item in result["blockers"])


def test_overflow_four_refs_is_blocked_without_dropping(tmp_path: Path) -> None:
    store, project, project_dir, first, girl, girl_side, _monster, env = _base(tmp_path)
    project["scenes"] = [
        _scene(
            first["id"],
            [
                {"asset_id": girl["id"], "role": "identity"},
                {"asset_id": girl_side["id"], "role": "identity"},
                {"asset_id": env["id"], "role": "environment"},
            ],
            ["girl"],
        )
    ]
    project = _save(store, project)
    result = validate_and_compile(project, project_dir)
    assert result["plan"] is None
    assert any(item["code"] == "REF_LIMIT" for item in result["blockers"])
    assert [item["asset_id"] for item in project["scenes"][0]["final"]["references"]] == [
        girl["id"],
        girl_side["id"],
        env["id"],
    ]


def test_a_new_shot_two_identity_and_environment_without_anchor_is_allowed(tmp_path: Path) -> None:
    store, project, project_dir, first, girl, girl_side, _monster, env = _base(tmp_path)
    first_scene = _scene(first["id"], [{"asset_id": girl["id"], "role": "identity"}], ["girl"])
    second_scene = _scene(
        first["id"],
        [{"asset_id": girl["id"], "role": "identity"}],
        ["girl"],
    )
    second_scene["id"] = "shot-02"
    second_scene["initial"] = {
        **second_scene["initial"],
        "kind": "new_shot",
        "asset_id": None,
        "instruction": "Explicit A instruction.",
        "references": [
            {"asset_id": girl["id"], "role": "identity"},
            {"asset_id": girl_side["id"], "role": "identity"},
            {"asset_id": env["id"], "role": "environment"},
        ],
    }
    project["scenes"] = [first_scene, second_scene]
    project = _save(store, project)
    result = validate_and_compile(project, project_dir)
    assert result["blockers"] == []
    assert len(result["plan"]["scenes"][1]["a"]["reference_roles"]) == 3
