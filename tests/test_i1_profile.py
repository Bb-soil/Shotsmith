from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.i1 import profile as p
from scripts.i1.ui_graph import (
    derive_ui_graph,
    derived_prompts_match_api,
    validate_ui_graph,
    write_ui_graph,
)

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "manifests" / "P1.wan22-14b-ab.sources.json"
API = ROOT / "workflows" / "P1" / "wan22-14b-shot02-alley-v1.api.json"


def test_derived_profile_exact_four_models_and_total() -> None:
    derived = p.build_profile_from_files(
        repo_root=ROOT,
        inventory_path=INV,
        api_workflow_path=API,
        ui_workflow_path=None,
    )
    assert len(derived.models) == 4
    assert derived.model_bytes() == 35_579_207_879
    assert derived.disk_required_bytes == 35_579_207_879 + 6442450944 + 5368709120
    assert derived.comfy["commit"] == p.COMFY_COMMIT
    assert set(derived.required_classes) == {
        "CLIPLoader",
        "CLIPTextEncode",
        "CreateVideo",
        "KSamplerAdvanced",
        "LoadImage",
        "ModelSamplingSD3",
        "SaveVideo",
        "UNETLoader",
        "VAEDecode",
        "VAELoader",
        "WanFirstLastFrameToVideo",
    }
    assert "width" in derived.required_inputs["WanFirstLastFrameToVideo"]
    assert "format" in derived.required_inputs["SaveVideo"]
    assert "format.codec" not in derived.required_inputs["SaveVideo"]
    assert derived.sha256 == p.validate_profile(derived.data).sha256


def test_profile_sources_are_pinned_and_original_untouched(tmp_path: Path) -> None:
    before = hashlib.sha256(INV.read_bytes()).hexdigest()
    derived = p.build_profile_from_files(
        repo_root=ROOT,
        inventory_path=INV,
        api_workflow_path=API,
        ui_workflow_path=None,
    )
    p.verify_profile_sources(derived, ROOT)
    assert hashlib.sha256(INV.read_bytes()).hexdigest() == before
    assert derived.data["source_inventory"]["sha256"] == before


def test_profile_rejects_wrong_inventory_or_unsafe_target(tmp_path: Path) -> None:
    raw = json.loads(INV.read_text(encoding="utf-8"))
    bad = dict(raw)
    bad["schema"] = "wrong"
    with pytest.raises(p.ProfileError):
        p._inventory_models(bad)  # type: ignore[attr-defined]
    raw2 = json.loads(INV.read_text(encoding="utf-8"))
    raw2["models"][0]["path"] = "../escape.safetensors"
    with pytest.raises(ValueError):
        p._inventory_models(raw2)  # type: ignore[attr-defined]


def test_ui_graph_is_derived_from_exact_api_prompts() -> None:
    api = json.loads(API.read_text(encoding="utf-8"))
    ui = derive_ui_graph(api)
    assert validate_ui_graph(ui) == 16
    assert derived_prompts_match_api(api, ui)
    save = [node for node in ui["nodes"] if node["type"] == "SaveVideo"][0]
    assert save["widgets_values"][1] == "mp4"
    assert save["widgets_values"][2] == "h264"
    assert save["widgets_values"][3] == "auto"


def test_ui_graph_written_is_stable(tmp_path: Path) -> None:
    api = json.loads(API.read_text(encoding="utf-8"))
    ui = derive_ui_graph(api, graph_id="fixture")
    first = write_ui_graph(tmp_path / "ui.json", ui)
    second = write_ui_graph(tmp_path / "ui.json", ui)
    assert first == second
    assert json.loads((tmp_path / "ui.json").read_text(encoding="utf-8"))["nodes"]


def test_checked_in_profile_matches_original_sources() -> None:
    checked = p.load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    p.verify_profile_sources(checked, ROOT)
    assert checked.model_bytes() == 35_579_207_879
