from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.kf1.binding import build_prompt, bind
from scripts.kf1.errors import BindingError
from scripts.kf1.plan import resolve_plan

ROOT = Path(__file__).resolve().parents[1]


def _frame():
    plan = resolve_plan(ROOT / "config/keyframes/two-scenes-v2.json")
    frame = next(item for item in plan.frames if item.frame_id == "shot-01/B")
    return plan, frame


def test_single_identity_prompt_and_hashes_are_byte_identical() -> None:
    plan, frame = _frame()
    baseline = json.loads(
        (ROOT / "docs/k2-evidence/kf2-followup-20261104/single-identity-builder-baseline.json").read_text(
            encoding="utf-8"
        )
    )["cases"][0]
    _bound, meta = bind(
        ref_count=2,
        image_names={"anchor": "a.png", "character_reference": "c.png"},
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=str(plan.data.get("style") or ""),
        roles=["anchor", "character_reference"],
        shot_mode=frame.shot_mode,
        invariants=frame.invariants,
        allowed_changes=frame.allowed_changes,
        subject_label=plan.subject_label,
        role_map={"anchor": "anchor", "character_reference": "identity"},
        seed=int(plan.settings["seed"]),
        version=1,
        plan_sha256=plan.plan_sha256,
        scene_id=frame.scene_id,
        role=frame.role,
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        parent_version=1,
        settings=plan.settings,
    )
    assert meta["positive_prompt"] == baseline["positive_prompt"]
    assert meta["negative_prompt"] == baseline["negative_prompt"]
    assert meta["stable_binding_sha256"] == baseline["stable_binding_sha256"]
    assert meta["binding_sha256"] == baseline["binding_sha256"]


def test_two_identity_refs_bind_with_deterministic_picture_captions() -> None:
    plan, frame = _frame()
    roles = ["anchor", "character_reference", "ref-asset-side"]
    role_map = {
        "anchor": "anchor",
        "character_reference": "identity",
        "ref-asset-side": "identity",
    }
    _bound, first = bind(
        ref_count=3,
        image_names={
            "anchor": "a.png",
            "character_reference": "front.png",
            "ref-asset-side": "side.png",
        },
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=str(plan.data.get("style") or ""),
        roles=roles,
        role_map=role_map,
        shot_mode=frame.shot_mode,
        invariants=frame.invariants,
        allowed_changes=frame.allowed_changes,
        subject_label=plan.subject_label,
        seed=int(plan.settings["seed"]),
        version=1,
        plan_sha256=plan.plan_sha256,
        scene_id=frame.scene_id,
        role=frame.role,
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        parent_version=1,
        settings=plan.settings,
    )
    _bound2, second = bind(
        ref_count=3,
        image_names={
            "anchor": "other-a.png",
            "character_reference": "other-front.png",
            "ref-asset-side": "other-side.png",
        },
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=str(plan.data.get("style") or ""),
        roles=roles,
        role_map=role_map,
        shot_mode=frame.shot_mode,
        invariants=frame.invariants,
        allowed_changes=frame.allowed_changes,
        subject_label=plan.subject_label,
        seed=int(plan.settings["seed"]),
        version=1,
        plan_sha256=plan.plan_sha256,
        scene_id=frame.scene_id,
        role=frame.role,
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        parent_version=1,
        settings=plan.settings,
    )
    prompt = first["positive_prompt"]
    assert "Picture 2 defines subject identity" in prompt
    assert "Picture 3 defines subject identity" in prompt
    assert "identity reference 1 of 2" in prompt
    assert "identity reference 2 of 2" in prompt
    assert first["roles"] == roles
    assert first["role_semantics"] == ["anchor", "identity", "identity"]
    assert first["stable_binding_sha256"] == second["stable_binding_sha256"]


def test_semantic_limits_block_three_identity_and_duplicate_env_style() -> None:
    with pytest.raises(BindingError, match="identity references"):
        build_prompt(
            edit_instruction="x",
            description="d",
            style="",
            roles=["a", "b", "c"],
            role_map={"a": "identity", "b": "identity", "c": "identity"},
            shot_mode="continue_shot",
        )
    with pytest.raises(BindingError, match="environment reference"):
        build_prompt(
            edit_instruction="x",
            description="d",
            style="",
            roles=["anchor", "e1", "e2"],
            role_map={"anchor": "anchor", "e1": "environment", "e2": "environment"},
            shot_mode="continue_shot",
        )
    with pytest.raises(BindingError, match="style reference"):
        build_prompt(
            edit_instruction="x",
            description="d",
            style="",
            roles=["anchor", "s1", "s2"],
            role_map={"anchor": "anchor", "s1": "style", "s2": "style"},
            shot_mode="continue_shot",
        )


def test_unknown_role_and_duplicate_reference_ids_are_fail_closed() -> None:
    with pytest.raises(BindingError, match="Неизвестная роль"):
        build_prompt(
            edit_instruction="x",
            description="d",
            style="",
            roles=["anchor", "weapon"],
            role_map={"anchor": "anchor", "weapon": "weapon"},
            shot_mode="continue_shot",
        )
    with pytest.raises(BindingError, match="Повторяющийся reference id"):
        build_prompt(
            edit_instruction="x",
            description="d",
            style="",
            roles=["anchor", "character_reference", "character_reference"],
            role_map={"anchor": "anchor", "character_reference": "identity"},
            shot_mode="continue_shot",
        )
