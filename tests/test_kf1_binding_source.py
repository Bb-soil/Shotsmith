from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.i1.profile import ProfileError, load_profile, validate_profile
from scripts.kf1.binding import bind, stable_binding_hash
from scripts.kf1.plan import resolve_plan
from scripts.kf1.source_contract import load_source_contract, source_hashes, validate_workflow

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_source_contract_extracts_all_kf1_classes() -> None:
    contract = load_source_contract()
    assert set(contract) == {
        "LoadImage",
        "ImageScale",
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "ModelSamplingAuraFlow",
        "CFGNorm",
        "TextEncodeQwenImageEditPlus",
        "VAEEncode",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
    assert contract["TextEncodeQwenImageEditPlus"]["optional"] == [
        "vae",
        "image1",
        "image2",
        "image3",
    ]
    assert contract["CFGNorm"]["optional"] == ["pre_cfg"]
    assert contract["SaveImage"]["outputs"] == ["IMAGE"]
    hashes = source_hashes()
    assert hashes["nodes_qwen.py"] == (
        "54ce02aa9e167c0b497ce026ac6b558f539427d2d1ed4fc1fcc2ec8e6bc529da"
    )
    assert hashes["image_qwen_image_edit_2511.json"] == (
        "b23103935f437e1fe3184247fa5bafb891a28113d0c63de7449b7cdfb0419111"
    )
    assert "license: apache-2.0" in (ROOT / "docs/k1-evidence/kf1-implementation-20261029/source/README.Qwen-Image-Edit.md").read_text(encoding="utf-8")


def test_derived_workflows_are_exact_source_contract_and_no_placeholder() -> None:
    for count in (1, 2, 3):
        path = ROOT / "workflows" / "K1" / f"qwen-edit-2511-{count}ref.api.json"
        workflow = json.loads(path.read_text(encoding="utf-8"))
        assert validate_workflow(workflow, allow_placeholder=True) == []
        assert "{{" not in path.read_text(encoding="utf-8")
        assert "KF1_BIND_REQUIRED" in path.read_text(encoding="utf-8")
        assert "SaveImage" in {node["class_type"] for node in workflow.values()}


def test_binding_1_2_3_and_environment_without_character() -> None:
    plan = resolve_plan(ROOT / "config/keyframes/two-scenes-v2.json")
    frame = [item for item in plan.frames if item.frame_id == "shot-01/B"][0]
    for count, names in (
        (1, {"anchor": "a.png"}),
        (2, {"anchor": "a.png", "character": "c.png"}),
        (3, {"anchor": "a.png", "character": "c.png", "environment": "e.png"}),
    ):
        bound, meta = bind(
            ref_count=count,
            image_names=names,
            edit_instruction=frame.edit_instruction or "",
            description=frame.description,
            style=plan.data["style"],
            mode="b",
            seed=plan.settings["seed"],
            version=1,
            plan_sha256=plan.plan_sha256,
            scene_id=frame.scene_id,
            role=frame.role,
            settings=plan.settings,
        )
        assert validate_workflow(bound) == []
        assert meta["ref_count"] == count
        assert "KF1_BIND_REQUIRED" not in json.dumps(bound, ensure_ascii=False)
        assert meta["output_keys"] == ["images"]
        assert meta["media_type"] == "image/png"

    env_only, env_meta = bind(
        ref_count=2,
        image_names={"anchor": "a.png", "environment": "e.png"},
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=plan.data["style"],
        mode="b",
        seed=plan.settings["seed"],
        version=1,
        plan_sha256=plan.plan_sha256,
        scene_id=frame.scene_id,
        role=frame.role,
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        parent_version=1,
        settings=plan.settings,
    )
    assert env_meta["roles"] == ["anchor", "environment"]
    assert "Picture 2 defines the environment/world only" in env_meta["positive_prompt"]
    assert "heroine" not in env_meta["positive_prompt"]


def test_stable_binding_hash_matches_bind_and_is_upload_name_independent() -> None:
    plan = resolve_plan(ROOT / "config/keyframes/two-scenes-v2.json")
    frame = [item for item in plan.frames if item.frame_id == "shot-01/B"][0]
    roles = ["anchor", "character"]
    digest, _prompt, _sha, _negative = stable_binding_hash(
        ref_count=2,
        roles=roles,
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=plan.data["style"],
        mode="b",
        seed=plan.settings["seed"],
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        plan_sha256=plan.plan_sha256,
        parent_version=1,
        settings=plan.settings,
    )
    _bound, first = bind(
        ref_count=2,
        image_names={"anchor": "a.png", "character": "c.png"},
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=plan.data["style"],
        mode="b",
        seed=plan.settings["seed"],
        version=1,
        plan_sha256=plan.plan_sha256,
        scene_id=frame.scene_id,
        role=frame.role,
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        parent_version=1,
        settings=plan.settings,
    )
    _bound2, second = bind(
        ref_count=2,
        image_names={"anchor": "different-anchor.png", "character": "different-char.png"},
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=plan.data["style"],
        mode="b",
        seed=plan.settings["seed"],
        version=1,
        plan_sha256=plan.plan_sha256,
        scene_id=frame.scene_id,
        role=frame.role,
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        parent_version=1,
        settings=plan.settings,
    )
    assert first["stable_binding_sha256"] == digest == second["stable_binding_sha256"]


def test_known_profiles_counts_and_backcompat() -> None:
    wan = load_profile(ROOT / "config/i1/wan22-14b-ab.profile.json")
    k1 = load_profile(ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json")
    bundle = load_profile(ROOT / "config/i1/bundle-wan22-qwen-edit-2511.profile.json")
    assert len(wan.models) == 4 and wan.model_bytes() == 35_579_207_879
    assert len(k1.models) == 3 and k1.model_bytes() == 30_172_239_743
    assert len(bundle.models) == 7 and bundle.model_bytes() == 65_751_447_622
    assert {item["class"] for item in k1.output_specs} == {"SaveImage"}
    assert {item["class"] for item in bundle.output_specs} == {"SaveImage", "SaveVideo"}
    bad = dict(k1.data)
    bad["models"] = list(bad["models"])[:-1]
    bad["models_policy"] = dict(bad["models_policy"], expected_count=2)
    with pytest.raises(ProfileError):
        validate_profile(bad)

def test_stdlib_png_fallback_normalizes_without_pillow(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image
    from scripts.kf1 import normalize

    source = tmp_path / "src.png"
    Image.new("RGB", (100, 50), (10, 20, 30)).save(source, format="PNG")
    before = source.read_bytes()
    monkeypatch.setattr(normalize, "Image", None)
    monkeypatch.setattr(normalize, "ImageOps", None)
    target = tmp_path / "out.png"
    record = normalize.normalize_image(
        source, target, policy="contain", target=[64, 64], pad=True, label="fixture"
    )
    assert source.read_bytes() == before
    assert record["stretch"] is False
    from scripts.kf1.png import read_png
    decoded = read_png(target.read_bytes())
    assert (decoded.width, decoded.height) == (64, 64)
    assert record["source"]["sha256"] == __import__("hashlib").sha256(before).hexdigest()

def test_committed_derived_profiles_match_builders() -> None:
    from scripts.kf1.profiles import build_bundle_profile, build_keyframe_profile
    assert build_keyframe_profile() == json.loads(
        (ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json").read_text(encoding="utf-8")
    )
    assert build_bundle_profile() == json.loads(
        (ROOT / "config/i1/bundle-wan22-qwen-edit-2511.profile.json").read_text(encoding="utf-8")
    )

def test_pins_cover_all_pinned_source_and_license_snapshots() -> None:
    import hashlib
    from scripts.kf1.source_contract import SOURCE_DIR, load_pins
    pins = load_pins()
    for entry in pins["comfyui"]["sources"].values():
        name = entry["url"].rsplit("/", 1)[-1]
        assert hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest() == entry["sha256"]
    for entry in pins["licenses"].values():
        name = "README.Qwen-Image-Edit.md" if "Qwen-Image-Edit" in entry["repo_id"] else "README.Qwen-Image.md"
        assert hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest() == entry["sha256"]
    assert pins["template"]["sha256"] == hashlib.sha256((SOURCE_DIR / "image_qwen_image_edit_2511.json").read_bytes()).hexdigest()
