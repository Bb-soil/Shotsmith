from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.kf1.comfy import ComfyClient, ComfyError, HistoryContractError, output_from_history
from scripts.kf1.connection import validate_kf1_context
from scripts.kf1.errors import StoreError, TerminalRemoteFailure
from scripts.kf1.gallery import export_storyboard
from scripts.kf1.plan import resolve_plan
from scripts.kf1.png import PNGError, read_png, write_png
from scripts.kf1.runner import Runner
from scripts.kf1.source_contract import source_contract_report, validate_workflow
from scripts.kf1.store import PlanStore
from tests.kf1_fake_comfy import FakeComfy, _png

ROOT = Path(__file__).resolve().parents[1]


def _plan():
    return resolve_plan(ROOT / "config/keyframes/two-scenes-v2.json")


def _run(tmp_path: Path, *, fixed_output: bytes | None = None):
    plan = _plan()
    fake = FakeComfy().start()
    if fixed_output is not None:
        fake.output_bytes = fixed_output
        fake.fixed_output = True
    client = ComfyClient(fake.base_url)
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    return plan, fake, client, store, Runner(plan, store, client)


def test_plan_edit_keeps_stable_path_state_ledger_and_history(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        (ROOT / "config/keyframes/two-scenes-v2.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plan = resolve_plan(plan_path)
    fake = FakeComfy().start()
    client = ComfyClient(fake.base_url)
    try:
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        assert store.ledger_count() == 3
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        data["title"] = "edited title"
        plan_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        edited = resolve_plan(plan_path)
        assert edited.plan_sha256 != plan.plan_sha256
        edited_store = PlanStore(tmp_path / "state", edited.plan_sha256, edited.path)
        state = edited_store.load(plan=edited)
        assert state["plan_sha256"] == edited.plan_sha256
        assert edited_store.ledger_count() == 3
        revisions = edited_store.iter_states(plan=edited)
        assert {item["plan_sha256"] for item in revisions} == {
            plan.plan_sha256,
            edited.plan_sha256,
        }
        assert edited_store.all_prompt_ids_all(plan=edited)
    finally:
        client.close()
        fake.stop()


def test_unresolved_old_revision_blocks_new_plan_edit(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        (ROOT / "config/keyframes/two-scenes-v2.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plan = resolve_plan(plan_path)
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    state = store.load(plan=plan)
    store.begin_attempt(
        state,
        frame_id="shot-01/B",
        version=1,
        intent={"nonce": "n", "correlation_key": "c", "frame_id": "shot-01/B"},
    )
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    data["seed"] = 999
    plan_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    edited = resolve_plan(plan_path)
    with pytest.raises(StoreError, match="unresolved intent"):
        PlanStore(tmp_path / "state", edited.plan_sha256, edited.path).load(plan=edited)


def test_crash_after_png_before_state_uses_receipt_and_no_second_post(tmp_path: Path) -> None:
    plan, fake, client, store, runner = _run(tmp_path)
    original_save = store.save
    raised = {"value": False}

    def save_once(state):
        if (
            not raised["value"]
            and any(
                item.get("status") == "COMPLETE" for item in state.get("attempts", [])
            )
        ):
            raised["value"] = True
            raise RuntimeError("simulated crash before state publication")
        return original_save(state)

    store.save = save_once  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError):
            runner.run(execute=True, deadline_seconds=15)
        assert fake.prompt_count == 1
        # Receipt is durable even though the state save failed.
        receipt = next((store.root.rglob("result.json")), None)
        assert receipt is not None and receipt.is_file()
        store.save = original_save  # type: ignore[method-assign]
        resumed_store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        result = Runner(plan, resumed_store, client).run(execute=True, deadline_seconds=15)
        assert result["status"] == "COMPLETE"
        state = resumed_store.load(plan=plan)
        assert len(resumed_store.attempts_for_frame(state, "shot-01/B")) == 1
        assert fake.prompt_count == 3
    finally:
        client.close()
        fake.stop()


def test_gallery_marks_invalidated_dependency_and_rejects_tamper(tmp_path: Path) -> None:
    plan, fake, client, store, runner = _run(tmp_path)
    try:
        runner.run(execute=True, deadline_seconds=15)
        runner.retry_frame("shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=15)
        state = store.load(plan=plan)
        manifest = export_storyboard(plan, state, tmp_path / "export")
        scene2 = manifest["scenes"][1]
        assert {item["role"]: item["status"] for item in scene2["roles"]} == {
            "A": "INVALIDATED",
            "B": "INVALIDATED",
        }
        assert all("path" not in item for item in scene2["roles"])
        # Tampering a current PNG must fail export, not silently rehash it.
        state = store.load(plan=plan)
        a1 = Path(state["frames"]["shot-01/A"]["versions"]["1"]["output"]["path"])
        a1.write_bytes(_png(32, 32))
        with pytest.raises(StoreError):
            export_storyboard(plan, store.load(plan=plan), tmp_path / "export2")
    finally:
        client.close()
        fake.stop()


def test_custom_geometry_binds_graph_and_wrong_dimension_never_completes(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    data = json.loads((ROOT / "config/keyframes/two-scenes-v2.json").read_text(encoding="utf-8"))
    data["output_policy"]["candidate_geometry"] = [512, 768]
    plan_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    plan = resolve_plan(plan_path)
    assert plan.output_policy["candidate_geometry"] == [512, 768]
    fake = FakeComfy().start()
    client = ComfyClient(fake.base_url)
    try:
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        result = Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        assert result["status"] == "COMPLETE"
        state = store.load(plan=plan)
        for frame in ("shot-01/B", "shot-02/A", "shot-02/B"):
            output = state["frames"][frame]["versions"]["1"]["output"]
            assert (output["width"], output["height"]) == (512, 768)
        assert fake.prompt_count == 3
    finally:
        client.close()
        fake.stop()

    bad_fake = FakeComfy().start()
    bad_fake.output_bytes = _png(64, 96)
    bad_fake.fixed_output = True
    bad_client = ComfyClient(bad_fake.base_url)
    try:
        bad_store = PlanStore(tmp_path / "bad-state", plan.plan_sha256, plan.path)
        with pytest.raises(TerminalRemoteFailure):
            Runner(plan, bad_store, bad_client).run(execute=True, deadline_seconds=15)
        state = bad_store.load(plan=plan)
        assert state["attempts"][0]["status"] == "FAILED_TERMINAL"
    finally:
        bad_client.close()
        bad_fake.stop()


def test_source_contract_rejects_wrong_literal_types_and_reports_dynamic() -> None:
    workflow = json.loads(
        (ROOT / "workflows/K1/qwen-edit-2511-1ref.api.json").read_text(encoding="utf-8")
    )
    workflow["50"]["inputs"]["steps"] = "wrong"
    assert any("must be INT" in item for item in validate_workflow(workflow, allow_placeholder=True))
    workflow = json.loads(
        (ROOT / "workflows/K1/qwen-edit-2511-1ref.api.json").read_text(encoding="utf-8")
    )
    workflow["50"]["inputs"]["cfg"] = "wrong"
    assert any("must be FLOAT" in item for item in validate_workflow(workflow, allow_placeholder=True))
    workflow = json.loads(
        (ROOT / "workflows/K1/qwen-edit-2511-1ref.api.json").read_text(encoding="utf-8")
    )
    workflow["50"]["inputs"]["seed"] = True
    assert any("must be INT" in item for item in validate_workflow(workflow, allow_placeholder=True))
    report = source_contract_report()
    assert report["actual_object_info"] == "NOT_RUN"
    assert report["source_dynamic_not_verified"]


def test_connection_gate_rejects_wan_installed_cached_and_tampered_evidence() -> None:
    class Ctx:
        pass

    from scripts.i1.profile import load_profile

    profile = load_profile(ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json")
    report = {
        "ready": True,
        "profile_sha256": profile.sha256,
        "model_bytes_verified": profile.model_bytes(),
        "workflow_hashes": {key: meta["api_sha256"] for key, meta in profile.workflows.items()},
        "checks": [
            {"name": name, "passed": True}
            for name in (
                "probe_schema", "profile_hash", "torch_cuda", "comfy_commit", "pip_check",
                "models", "object_info", "workflow_links", "output_schema", "system_stats",
                "workflow_schema", "tunnel_health",
            )
        ],
    }
    ctx = Ctx()
    ctx.profile = profile
    ctx.state = {"phase": "READY", "readiness_current": True, "readiness": report}
    validate_kf1_context(ctx)
    wan = load_profile(ROOT / "config/i1/wan22-14b-ab.profile.json")
    ctx.profile = wan
    with pytest.raises(Exception):
        validate_kf1_context(ctx)
    ctx.profile = profile
    ctx.state = {"phase": "INSTALLED", "readiness_current": True, "readiness": report}
    with pytest.raises(Exception):
        validate_kf1_context(ctx)
    ctx.state = {"phase": "READY", "readiness_current": False, "readiness": report}
    with pytest.raises(Exception):
        validate_kf1_context(ctx)
    ctx.state = {"phase": "READY", "readiness_current": True, "readiness": dict(report, model_bytes_verified=1)}
    with pytest.raises(Exception):
        validate_kf1_context(ctx)


def test_png_strict_rejects_missing_iend_trailing_crc_bomb_and_truncation() -> None:
    valid = _png(4, 4)
    assert read_png(valid).width == 4
    with pytest.raises(PNGError):
        read_png(valid[:-12])  # missing IEND
    with pytest.raises(PNGError):
        read_png(valid + b"junk")
    broken = bytearray(valid)
    broken[-5] ^= 0xFF
    with pytest.raises(PNGError):
        read_png(bytes(broken))
    huge_header = bytearray(valid)
    huge_header[16:20] = (100_000).to_bytes(4, "big")
    huge_header[20:24] = (100_000).to_bytes(4, "big")
    with pytest.raises(PNGError):
        read_png(bytes(huge_header))
    with pytest.raises(PNGError):
        read_png(valid[:40])

def test_multiprocess_ledger_reservation_is_serialized(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        (ROOT / "config/keyframes/two-scenes-v2.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plan = resolve_plan(plan_path)
    code = (
        "import sys; sys.path.insert(0, r'"
        + str(ROOT)
        + "'); import os; from scripts.kf1.store import PlanStore; "
        "s=PlanStore(r'"
        + str(tmp_path / "state")
        + "', '"
        + plan.plan_sha256
        + "', r'"
        + str(plan.path)
        + "'); s._ledger_reserve(str(os.getpid()), '"
        + plan.plan_sha256
        + "')"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(2)]
    for proc in procs:
        assert proc.wait(timeout=30) == 0
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    assert store.ledger_count() == 2

def test_history_contract_rejects_unkeyed_wrong_id_nonterminal_graph_and_multiples() -> None:
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        # Unkeyed top-level history is never accepted as a prompt entry.
        fake.history["prompt-1"] = {"outputs": {}, "status": {"status_str": "success", "completed": True}}
        # history key exists, but prompt tuple has another id -> exact mismatch.
        entry = {
            "prompt": [0, "other-id", {"70": {"class_type": "SaveImage"}}, {"correlation_key": "c", "nonce": "n"}, []],
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {"70": {"images": [{"filename": "v0001_00001_.png", "subfolder": "", "type": "output"}]}},
        }
        with pytest.raises(HistoryContractError):
            output_from_history(
                entry,
                node_id="70",
                keys=["images"],
                expected_output={"subfolder": "", "filename_prefix": "v0001"},
                expected_prompt_id="prompt-1",
            )
        with pytest.raises(Exception):
            output_from_history(entry, node_id="70", keys=["images"], expected_prompt_id="prompt-1")
        nonterminal = dict(entry)
        nonterminal["prompt"] = [0, "prompt-1", {}, {}, []]
        nonterminal["status"] = {"status_str": "running", "completed": False, "messages": []}
        with pytest.raises(Exception):
            output_from_history(nonterminal, node_id="70", keys=["images"], expected_prompt_id="prompt-1")
        foreign = dict(entry)
        foreign["prompt"] = [0, "prompt-1", {"70": {"class_type": "SaveImage"}}, {}, []]
        with pytest.raises(Exception):
            output_from_history(
                foreign,
                node_id="70",
                keys=["images"],
                expected_output={"subfolder": "", "filename_prefix": "v0001"},
                expected_graph_sha256="0" * 64,
                expected_prompt_id="prompt-1",
            )
        multi = dict(entry)
        multi["prompt"] = [0, "prompt-1", {}, {}, []]
        multi["outputs"] = {"70": {"images": [
            {"filename": "v0001_00001_.png", "subfolder": "", "type": "output"},
            {"filename": "v0001_00002_.png", "subfolder": "", "type": "output"},
        ]}}
        with pytest.raises(Exception):
            output_from_history(multi, node_id="70", keys=["images"], expected_prompt_id="prompt-1")
        # correlation is exact field equality, never a JSON substring.
        fake.history = {
            "prompt-2": {
                "prompt": [0, "prompt-2", {}, {"note": "correlation_key c nonce n"}, []],
                "status": {"status_str": "success", "completed": True, "messages": []},
                "outputs": {},
            }
        }
        assert client.find_prompt_by_correlation("c", "n") is None
        client.close()
    finally:
        fake.stop()

def test_text_plan_geometry_fields_reach_output_policy(tmp_path: Path) -> None:
    from scripts.kf1.plan import parse_text_plan
    text = "\n".join([
        "title: text geometry",
        "first_frame: assets/frames/derived/shot-01-start-v3-704x1280.png",
        "character_reference: assets/references/character-sheet.png",
        "candidate_geometry: 512x768",
        "video_geometry: 448x816",
        "scene: shot-01",
        "description: сцена",
        "a_source: first_frame",
        "b_source: generate_from_scene_a",
        "b_edit: Keep the exact camera.",
    ])
    data = parse_text_plan(text)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    plan = resolve_plan(path)
    assert plan.output_policy["candidate_geometry"] == [512, 768]
    assert plan.output_policy["video_geometry"] == [448, 816]

def test_source_contract_rejects_unsupported_clip_type_and_keeps_dynamic_report() -> None:
    workflow = json.loads((ROOT / "workflows/K1/qwen-edit-2511-1ref.api.json").read_text(encoding="utf-8"))
    workflow["21"]["inputs"]["type"] = "not-a-comfy-clip-type"
    issues = validate_workflow(workflow, allow_placeholder=True)
    assert any("not in pinned enum" in item for item in issues)
    report = source_contract_report()
    assert "KSampler.sampler_name" in report["source_dynamic_not_verified"]

def test_runtime_object_info_gate_rejects_enum_mismatch_and_missing_class() -> None:
    from scripts.i1.profile import load_profile
    from scripts.kf1.runtime_gate import validate_runtime_object_info
    from tests.kf1_fake_comfy import _generic_object_info
    profile = load_profile(ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json")
    good = _generic_object_info()
    assert validate_runtime_object_info(profile, good)["status"] == "OBJECT_INFO_OK"
    bad = _generic_object_info()
    bad["CLIPLoader"]["input"]["required"]["type"] = [["stable_diffusion"], {}]
    with pytest.raises(Exception):
        validate_runtime_object_info(profile, bad)
    missing = _generic_object_info()
    missing.pop("TextEncodeQwenImageEditPlus")
    with pytest.raises(Exception):
        validate_runtime_object_info(profile, missing)
