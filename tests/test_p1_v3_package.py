from __future__ import annotations

import json
from pathlib import Path

from anime_studio.p1_config import check_readiness, load_config, load_manifest
from anime_studio.workflow import load_workflow
from scripts.p1.preflight import load_object_info, validate_object_info

ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG = ROOT / "config" / "p1.wan22-5b.shot01-v3.draft.json"
V3_WORKFLOW = ROOT / "workflows" / "P1" / "wan22-5b-shot01-v3.api.draft.json"
V3_MANIFEST = ROOT / "manifests" / "P1.wan22-5b-shot01-v3.draft.json"
V3_JOB = ROOT / "config" / "p1.shot01-v3.job.json"
V3_SOURCE = ROOT / "assets" / "frames" / "shot-01-start-v3.png"
V3_DERIVED = ROOT / "assets" / "frames" / "derived" / "shot-01-start-v3-704x1280.png"
V3_POLICY = ROOT / "assets" / "frames" / "derived" / "shot-01-start-v3-704x1280.policy.json"
V2_WORKFLOW = ROOT / "workflows" / "P1" / "wan22-5b-shot01.api.draft.json"
V2_CONFIG = ROOT / "config" / "p1.wan22-5b.draft.json"
V2_MANIFEST = ROOT / "manifests" / "P1.wan22-5b.draft.json"
V2_JOB = ROOT / "config" / "p1.shot01.job.json"
V2_SOURCE = ROOT / "assets" / "frames" / "shot-01-start-v2.png"
OBJECT_INFO = ROOT / "tests" / "fixtures" / "p1" / "savevideo-object_info-7a283320.json"

EXPECTED = {
    V3_WORKFLOW: "4f5645d1a6af2c929a6260a609c0d3cf513db99bcd2522649bd02d4b40c52784",
    V3_CONFIG: "e53a873ec71372c1bd47d17a905f8e4e713e8ded077b552e7b063469a3cab66a",
    V3_MANIFEST: "f6504159494585286eb56c97cb8de112386f231a92b77fd9629413d6d6d39099",
    V3_JOB: "097da38d046a5192e4f746223dfeefd0ec0656ffd83b804fd3e3b03529116d87",
    V3_SOURCE: "5e3029aa7edb9207c1503635606beeeed9b012995b2a296616668c84fe68083a",
    V3_DERIVED: "925eccbe02a49b74bd2e8cd190fe26bd9557628bb996149f093b894136c53add",
    V2_SOURCE: "83f5a79b9acb3b576033b231d76a09065f20121f415ddc6ec08e137c5a801d20",
    V2_WORKFLOW: "abac4f35c05f22e0d8d517bdc569bee8f7419e48c81753c90ad8eedf310b2576",
    V2_CONFIG: "b5790bcaf36da862eda52cf3ff7606baf1790f6cfdced420a70dec034c8cd828",
    V2_MANIFEST: "454b0ad80c171a087d66b6fa1c02740a8610c6d29191568df60aebf280444697",
    V2_JOB: "58d81d03adb75619a9c0be0f2caa9503a661a9bf2bba3859283039b51f983ce9",
}


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompts() -> tuple[str, str]:
    text = (ROOT / "docs" / "P1-shot-01-v3.md").read_text(encoding="utf-8")
    fence = chr(96) * 3
    parts = text.split(fence)
    blocks = [part for part in parts[1::2] if part.startswith("text")]
    assert len(blocks) == 2
    return (
        blocks[0].split("\n", 1)[1].strip(),
        blocks[1].split("\n", 1)[1].strip(),
    )


def test_v3_files_are_separate_and_hashes_match() -> None:
    for path, digest in EXPECTED.items():
        assert path.is_file(), path
        assert _sha(path) == digest, path


def test_v3_prompts_are_exact_and_wind_is_preserved() -> None:
    positive, negative = _prompts()
    workflow, _ = load_workflow(V3_WORKFLOW)
    job = json.loads(V3_JOB.read_text(encoding="utf-8"))
    assert workflow["6"]["inputs"]["text"] == positive
    assert workflow["7"]["inputs"]["text"] == negative
    assert job["prompt"] == positive
    assert "gust" in positive.lower()
    assert "wind" in negative.lower()
    assert "hair" in positive.lower()
    assert "torso and shoulders stay almost still" in positive


def test_v3_generation_parameters_match_v2_except_bindings() -> None:
    v2, _ = load_workflow(V2_WORKFLOW)
    v3, _ = load_workflow(V3_WORKFLOW)
    assert v3["3"]["inputs"]["seed"] == 424242
    assert v3["3"]["inputs"]["steps"] == 20
    assert v3["3"]["inputs"]["cfg"] == 5
    assert v3["3"]["inputs"]["sampler_name"] == "uni_pc"
    assert v3["3"]["inputs"]["scheduler"] == "simple"
    assert v3["48"]["inputs"]["shift"] == 8
    assert v3["55"]["inputs"]["width"] == 704
    assert v3["55"]["inputs"]["height"] == 1280
    assert v3["55"]["inputs"]["length"] == 121
    assert v3["57"]["inputs"]["fps"] == 24
    assert v3["56"]["inputs"]["image"] == V3_DERIVED.name
    assert v3["58"]["inputs"]["filename_prefix"] == "anime-studio/P1/shot-01-v3"
    assert v3["58"]["inputs"]["format"] == "mp4"
    assert v3["58"]["inputs"]["format.codec"] == "h264"
    assert v3["58"]["inputs"]["format.codec.encoding"] == "auto"
    # Everything except the declared v3 bindings remains identical to v2.
    for node_id in ("3", "8", "37", "38", "39", "48", "55", "57"):
        left = dict(v3[node_id])
        right = dict(v2[node_id])
        left_inputs = dict(left["inputs"])
        right_inputs = dict(right["inputs"])
        for key in ("text",):
            left_inputs.pop(key, None)
            right_inputs.pop(key, None)
        left["inputs"] = left_inputs
        right["inputs"] = right_inputs
        assert left == right, node_id
    assert v3["58"]["inputs"]["video"] == v2["58"]["inputs"]["video"]


def test_v3_config_and_manifest_bind_but_stay_draft() -> None:
    config = load_config(V3_CONFIG)
    workflow, workflow_sha = load_workflow(V3_WORKFLOW)
    manifest = load_manifest(V3_MANIFEST)
    assert workflow_sha == config.workflow_sha256
    assert manifest["workflow"]["sha256"] == workflow_sha
    assert manifest["status"] == "DRAFT"
    assert config.binding.validate(workflow) == []
    bound = config.binding.bind(
        workflow,
        prompt=json.loads(V3_JOB.read_text(encoding="utf-8"))["prompt"],
        seed=424242,
        image_name=V3_DERIVED.name,
        width=704,
        height=1280,
        frames=121,
        fps=24,
    )
    assert len(bound) == 12
    assert "_meta" not in bound
    assert bound["58"]["class_type"] == "SaveVideo"
    assert bound["56"]["inputs"]["image"] == V3_DERIVED.name
    assert bound["3"]["inputs"]["positive"] == ["6", 0]
    assert bound["3"]["inputs"]["negative"] == ["7", 0]

    readiness = check_readiness(V3_CONFIG)
    assert readiness.live_ready is False
    assert readiness.issues == ("manifest.status is not VERIFIED",)


def test_v3_policy_has_exact_crop_and_no_anisotropy() -> None:
    source = _sha(V3_SOURCE)
    derived = _sha(V3_DERIVED)
    policy = json.loads(V3_POLICY.read_text(encoding="utf-8"))
    assert policy["source"]["sha256"] == source
    assert policy["target"]["sha256"] == derived
    assert policy["source"]["width"] == 941
    assert policy["source"]["height"] == 1672
    assert policy["target"]["width"] == 704
    assert policy["target"]["height"] == 1280
    assert policy["crop"]["box_left_top_right_bottom"] == [14, 6, 927, 1666]
    assert policy["crop"]["width"] == 913
    assert policy["crop"]["height"] == 1660
    assert policy["crop"]["top_crop_pixels"] == 6
    assert policy["resize"]["anisotropic"] is False


def test_v3_object_info_nested_gate_is_unchanged() -> None:
    issues = validate_object_info(load_object_info(OBJECT_INFO))
    assert issues == []
