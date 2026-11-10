from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.kf1.errors import PlanError
from scripts.kf1.plan import PLAN_SCHEMA_V2, resolve_plan
from tests.kf1_fake_comfy import _png


def _write(path: Path, color: tuple[int, int, int]) -> Path:
    path.write_bytes(_png(64, 64, color))
    return path


def _plan(
    tmp_path: Path,
    *,
    refs: list[dict],
    scene_b_roles: list[str],
    scene_a_roles: list[str] | None = None,
) -> Path:
    first = _write(tmp_path / "first.png", (1, 1, 1))
    data = {
        "schema": PLAN_SCHEMA_V2,
        "title": "followup refs",
        "language": "ru",
        "style": "",
        "first_frame": str(first),
        "references": refs,
        "scenes": [
            {
                "id": "shot-01",
                "description": "Explicit scene context.",
                "a": {"source": "first_frame"},
                "b": {
                    "source": "generate_from_scene_a",
                    "shot_mode": "continue_shot",
                    "reference_roles": scene_b_roles,
                    "invariants": ["camera"],
                    "allowed_changes": ["pose"],
                    "edit_instruction": "Explicit B instruction.",
                },
            }
        ],
        "output_policy": {
            "candidate_geometry": [704, 1280],
            "video_geometry": [448, 816],
            "overwrite_inputs": False,
            "automatic_video_submit": False,
        },
        "generation_limit": 2,
        "seed": 1,
    }
    if scene_a_roles is not None:
        data["scenes"].append(
            {
                "id": "shot-02",
                "description": "Second explicit scene.",
                "a": {
                    "source": "generate_from_previous_b",
                    "shot_mode": "new_shot",
                    "reference_roles": scene_a_roles,
                    "invariants": ["identity"],
                    "allowed_changes": ["camera", "environment", "position"],
                    "edit_instruction": "Explicit A instruction.",
                },
                "b": {
                    "source": "generate_from_scene_a",
                    "shot_mode": "continue_shot",
                    "reference_roles": ["anchor", refs[0]["id"]],
                    "invariants": ["camera"],
                    "allowed_changes": ["pose"],
                    "edit_instruction": "Explicit B2 instruction.",
                },
            }
        )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_two_identity_refs_same_frame_with_distinct_content(tmp_path: Path) -> None:
    front = _write(tmp_path / "front.png", (10, 20, 30))
    side = _write(tmp_path / "side.png", (30, 20, 10))
    plan = _plan(
        tmp_path,
        refs=[
            {"id": "id_front", "role": "identity", "path": str(front)},
            {"id": "id_side", "role": "identity", "path": str(side)},
        ],
        scene_b_roles=["anchor", "id_front", "id_side"],
    )
    resolved = resolve_plan(plan)
    assert resolved.refs["id_front"].role == "identity"
    assert resolved.refs["id_side"].role == "identity"
    b = next(frame for frame in resolved.frames if frame.frame_id == "shot-01/B")
    assert b.reference_roles == ("anchor", "id_front", "id_side")


def test_same_content_for_two_identity_refs_is_blocked(tmp_path: Path) -> None:
    same = _write(tmp_path / "same.png", (10, 20, 30))
    plan = _plan(
        tmp_path,
        refs=[
            {"id": "id_a", "role": "identity", "path": str(same)},
            {"id": "id_b", "role": "identity", "path": str(same)},
        ],
        scene_b_roles=["anchor", "id_a", "id_b"],
    )
    with pytest.raises(PlanError, match="content SHA256"):
        resolve_plan(plan)


def test_anchor_and_identity_same_image_remains_valid_for_old_shape(tmp_path: Path) -> None:
    same = _write(tmp_path / "same.png", (10, 20, 30))
    first = _write(tmp_path / "first.png", (10, 20, 30))
    data = {
        "schema": PLAN_SCHEMA_V2,
        "title": "legacy anchor-identity same bytes",
        "language": "ru",
        "style": "",
        "first_frame": str(first),
        "character_reference": str(same),
        "scenes": [
            {
                "id": "shot-01",
                "description": "Explicit scene.",
                "a": {"source": "first_frame"},
                "b": {
                    "source": "generate_from_scene_a",
                    "shot_mode": "continue_shot",
                    "reference_roles": ["anchor", "character_reference"],
                    "invariants": ["camera"],
                    "allowed_changes": ["pose"],
                    "edit_instruction": "Explicit B.",
                },
            }
        ],
        "output_policy": {
            "candidate_geometry": [704, 1280],
            "video_geometry": [448, 816],
            "overwrite_inputs": False,
            "automatic_video_submit": False,
        },
        "generation_limit": 1,
        "seed": 1,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    resolved = resolve_plan(path)
    assert resolved.refs["character_reference"].sha256 == resolved.refs["first_frame"].sha256


def test_three_identity_refs_in_one_frame_are_blocked(tmp_path: Path) -> None:
    refs = []
    for index in range(3):
        path = _write(tmp_path / f"id{index}.png", (index + 1, 2, 3))
        refs.append({"id": f"id_{index}", "role": "identity", "path": str(path)})
    plan = _plan(
        tmp_path,
        refs=refs,
        scene_b_roles=["anchor", "id_0"],
        scene_a_roles=["id_0", "id_1", "id_2"],
    )
    with pytest.raises(PlanError, match="identity refs"):
        resolve_plan(plan)


def test_duplicate_environment_and_style_per_frame_are_blocked(tmp_path: Path) -> None:
    env1 = _write(tmp_path / "env1.png", (1, 1, 2))
    env2 = _write(tmp_path / "env2.png", (2, 1, 1))
    style1 = _write(tmp_path / "style1.png", (1, 2, 1))
    style2 = _write(tmp_path / "style2.png", (2, 2, 1))
    refs = [
        {"id": "env1", "role": "environment", "path": str(env1)},
        {"id": "env2", "role": "environment", "path": str(env2)},
        {"id": "style1", "role": "style", "path": str(style1)},
        {"id": "style2", "role": "style", "path": str(style2)},
    ]
    env_plan = _plan(tmp_path, refs=refs[:2], scene_b_roles=["anchor", "env1", "env2"])
    with pytest.raises(PlanError, match="environment refs"):
        resolve_plan(env_plan)
    style_plan = _plan(tmp_path, refs=refs[2:], scene_b_roles=["anchor", "style1", "style2"])
    with pytest.raises(PlanError, match="style refs"):
        resolve_plan(style_plan)


def test_used_reference_catalog_is_bounded_at_64(tmp_path: Path) -> None:
    refs = []
    for index in range(65):
        refs.append(
            {
                "id": f"id_{index}",
                "role": "identity",
                "path": str(_write(tmp_path / f"id{index}.png", (index, 1, 1))),
            }
        )
    plan = _plan(
        tmp_path,
        refs=refs,
        scene_b_roles=["anchor", "id_0"],
    )
    with pytest.raises(PlanError, match="64"):
        resolve_plan(plan)


def test_public_historic_plan_hashes_are_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads(
        (root / "docs/k2-evidence/kf2-followup-20261104/historical-plans.json").read_text(
            encoding="utf-8"
        )
    )
    for item in baseline["config_plans"]:
        assert resolve_plan(root / item["path"]).plan_sha256 == item["plan_sha256"]
    for item in baseline["data_revisions"]:
        source = item.get("plan_source_path")
        if source:
            assert resolve_plan(root / source).plan_sha256 == item["resolved_plan_sha256"]


def test_manual_v2_unknown_roles_ids_duplicates_and_total_overflow_fail(tmp_path: Path) -> None:
    front = _write(tmp_path / "front.png", (10, 20, 30))
    plan = _plan(
        tmp_path,
        refs=[{"id": "id_front", "role": "identity", "path": str(front)}],
        scene_b_roles=["anchor", "missing_ref"],
    )
    with pytest.raises(PlanError, match="reference id"):
        resolve_plan(plan)

    plan = _plan(
        tmp_path,
        refs=[{"id": "id_front", "role": "weapon", "path": str(front)}],
        scene_b_roles=["anchor", "id_front"],
    )
    with pytest.raises(PlanError, match="identity/style/environment"):
        resolve_plan(plan)

    plan = _plan(
        tmp_path,
        refs=[
            {"id": "id_front", "role": "identity", "path": str(front)},
            {"id": "id_front", "role": "identity", "path": str(front)},
        ],
        scene_b_roles=["anchor", "id_front"],
    )
    with pytest.raises(PlanError, match="reference id"):
        resolve_plan(plan)

    plan = _plan(
        tmp_path,
        refs=[{"id": "id_front", "role": "identity", "path": str(front)}],
        scene_b_roles=["anchor", "id_front", "id_front"],
    )
    with pytest.raises(PlanError, match="reference_roles"):
        resolve_plan(plan)

    plan = _plan(
        tmp_path,
        refs=[{"id": "id_front", "role": "identity", "path": str(front)}],
        scene_b_roles=["anchor", "id_front", "id_front", "id_front"],
    )
    with pytest.raises(PlanError, match="1..3"):
        resolve_plan(plan)
