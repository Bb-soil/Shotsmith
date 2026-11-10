from __future__ import annotations

import json
import struct
import sys
import threading
import time
import tracemalloc
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.i1.profile import load_profile
from scripts.kf1 import cli
from scripts.kf1.binding import bind
from scripts.kf1.comfy import ComfyClient, HistoryContractError, output_from_history
from scripts.kf1.connection import open_kf1_connection, verify_kf1_runtime
from scripts.kf1.errors import AmbiguousDispatch, KF1Error, StoreError, TerminalRemoteFailure
from scripts.kf1.gallery import export_storyboard
from scripts.kf1.plan import resolve_plan
from scripts.kf1.png import PNGError, read_png
from scripts.kf1.runtime_gate import (
    validate_bound_workflow,
    validate_runtime_object_info,
    validate_upload_receipt,
)
from scripts.kf1.runner import Runner
from scripts.kf1.store import PlanStore
from tests.kf1_fake_comfy import FakeComfy, _png
from tests.kf1_object_info import build_object_info

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config/keyframes/two-scenes-v2.json"


def _plan():
    return resolve_plan(EXAMPLE)


def _one_scene_plan(tmp_path: Path, mutate=None):
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["scenes"] = data["scenes"][:1]
    if mutate is not None:
        mutate(data)
    path = tmp_path / "one-scene-plan.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path, resolve_plan(path)


def _profile():
    return load_profile(ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json")


def _receipt(name: str, *, type_: str = "input", subfolder: str = "") -> dict:
    path = f"input/{subfolder}/{name}" if subfolder else f"input/{name}"
    return {
        "name": name,
        "requested_name": name,
        "type": type_,
        "subfolder": subfolder,
        "path": path,
        "source_sha256": "a" * 64,
    }


def _bound_for_gate(image_names: dict[str, str]):
    plan = _plan()
    frame = next(item for item in plan.frames if item.frame_id == "shot-01/B")
    return bind(
        ref_count=len(image_names),
        image_names=image_names,
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=str(plan.data.get("style") or ""),
        mode="b",
        seed=plan.settings["seed"],
        version=1,
        plan_sha256=plan.plan_sha256,
        scene_id=frame.scene_id,
        role=frame.role,
        settings=plan.settings,
    )


# ---------------------------------------------------------------- F1 / F5
def test_installation_capability_gate_accepts_empty_enum_but_models_stay_strict() -> None:
    profile = _profile()
    good = build_object_info(load_image_files=[])
    result = validate_runtime_object_info(profile, good)
    assert result["status"] == "OBJECT_INFO_OK"
    assert result["job_binding_required"]
    assert all(
        item["status"] == "JOB_BINDING_REQUIRED"
        for item in result["job_binding_required"]
    )
    bad = build_object_info(load_image_files=[])
    bad["UNETLoader"]["input"]["required"]["unet_name"][0] = [
        "some-other-model.safetensors"
    ]
    with pytest.raises(KF1Error, match="enum mismatch"):
        validate_runtime_object_info(profile, bad)


def test_bound_gate_checks_literals_ranges_links_required_and_receipts() -> None:
    object_info = build_object_info(load_image_files=[])
    bound, meta = _bound_for_gate({"anchor": "anchor.png", "character": "char.png"})
    receipts = {
        "anchor.png": _receipt("anchor.png"),
        "char.png": _receipt("char.png"),
    }
    gate = validate_bound_workflow(
        bound, object_info, upload_receipts=receipts, expected_output=meta["expected_output"]
    )
    assert gate["status"] == "BOUND_GRAPH_OK"
    assert len(gate["job_binding_required"]) == 2
    assert gate["ranges_checked"] >= 4

    with pytest.raises(KF1Error, match="trusted upload receipt"):
        validate_bound_workflow(bound, object_info, expected_output=meta["expected_output"])
    with pytest.raises(KF1Error, match="type"):
        validate_bound_workflow(
            bound,
            object_info,
            upload_receipts={"anchor.png": _receipt("anchor.png", type_="output")},
            expected_output=meta["expected_output"],
        )

    missing_steps = json.loads(json.dumps(bound))
    missing_steps["50"]["inputs"].pop("steps")
    with pytest.raises(KF1Error, match="missing required input 'steps'"):
        validate_bound_workflow(
            missing_steps, object_info, upload_receipts=receipts,
            expected_output=meta["expected_output"],
        )

    extra = json.loads(json.dumps(bound))
    extra["50"]["inputs"]["bogus"] = 1
    with pytest.raises(KF1Error, match="отсутствует в actual object_info"):
        validate_bound_workflow(
            extra, object_info, upload_receipts=receipts,
            expected_output=meta["expected_output"],
        )

    out_of_range = json.loads(json.dumps(bound))
    out_of_range["50"]["inputs"]["steps"] = 0
    with pytest.raises(KF1Error, match="below min"):
        validate_bound_workflow(
            out_of_range, object_info, upload_receipts=receipts,
            expected_output=meta["expected_output"],
        )

    bad_link = json.loads(json.dumps(bound))
    bad_link["10"]["inputs"]["image"] = ["20", 0]  # UNETLoader MODEL -> IMAGE
    with pytest.raises(KF1Error, match="link"):
        validate_bound_workflow(
            bad_link, object_info, upload_receipts=receipts,
            expected_output=meta["expected_output"],
        )


def test_upload_receipt_exact_path_and_type() -> None:
    assert validate_upload_receipt(_receipt("x.png"), expected_name="x.png")
    with pytest.raises(KF1Error):
        validate_upload_receipt(_receipt("x.png", subfolder="nested"), expected_name="x.png")
    wrong = _receipt("x.png")
    wrong["path"] = "input/other.png"
    with pytest.raises(KF1Error):
        validate_upload_receipt(wrong, expected_name="x.png")


def test_fake_object_info_has_no_star_hiding_and_named_enum_after_upload(tmp_path: Path) -> None:
    profile = _profile()
    fixture = build_object_info(load_image_files=[])
    assert fixture["LoadImage"]["input"]["required"]["image"][0] == []
    assert fixture["LoadImage"]["input"]["required"]["image"][1]["image_upload"] is True
    assert "euler" in fixture["KSampler"]["input"]["required"]["sampler_name"][0]
    assert "qwen_image" in fixture["CLIPLoader"]["input"]["required"]["type"][0]
    assert "qwen_image_edit_2511_fp8mixed.safetensors" in fixture["UNETLoader"][
        "input"
    ]["required"]["unet_name"][0]
    # No wildcard type masks a real type mismatch.
    for spec in fixture.values():
        for section in ("required", "optional"):
            for value in spec.get("input", {}).get(section, {}).values():
                assert value[0] != "*"

    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", _plan().plan_sha256, _plan().path)
        plan = _plan()
        result = Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        assert result["status"] == "COMPLETE"
        assert fake.prompt_count == 3
        load_image = fake.object_info["LoadImage"]["input"]["required"]["image"]
        assert load_image[0] and set(load_image[0]) == set(fake.uploaded_image_names)
        state = store.load_revision_readonly(plan=plan)
        for attempt in state["attempts"]:
            assert attempt["runtime_gate"]["status"] == "BOUND_GRAPH_OK"
            for upload in attempt["uploads"]:
                assert upload["type"] == "input"
                assert upload["path"].startswith("input/")
        client.close()
    finally:
        fake.stop()


def test_runner_succeeds_when_upload_enum_is_not_listed_but_capability_receipt_is(
    tmp_path: Path,
) -> None:
    plan = _plan()
    fake = FakeComfy().start()
    fake.list_uploaded_images = False
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        result = Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        assert result["status"] == "COMPLETE"
        assert fake.prompt_count == 3
        state = store.load_revision_readonly(plan=plan)
        assert all(attempt["runtime_gate"]["job_binding_required"] for attempt in state["attempts"])
        client.close()
    finally:
        fake.stop()


def test_runner_refuses_bound_graph_mutation_before_submit(tmp_path: Path) -> None:
    import scripts.kf1.runner as runner_mod

    original = runner_mod.binding_mod.bind

    def missing_steps(*args, **kwargs):
        bound, meta = original(*args, **kwargs)
        bound["50"]["inputs"].pop("steps", None)
        return bound, meta

    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        plan = _plan()
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner_mod.binding_mod.bind = missing_steps
        try:
            with pytest.raises(KF1Error, match="missing required input 'steps'"):
                Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        finally:
            runner_mod.binding_mod.bind = original
        assert fake.prompt_count == 0
        client.close()
    finally:
        runner_mod.binding_mod.bind = original
        fake.stop()


# -------------------------------------------------------------------- F2
def test_complete_v1_then_v2_crash_before_receipt_resumes_exact_v2(tmp_path: Path) -> None:
    _path, plan = _one_scene_plan(tmp_path)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        assert fake.prompt_count == 1
        state = store.load_revision_readonly(plan=plan)
        v1_receipt = store.version_dir("shot-01/B", 1) / "result.json"
        assert v1_receipt.is_file()

        original_view = client.view

        def crash_before_receipt(output):
            raise RuntimeError("simulated crash after v2 POST before receipt")

        client.view = crash_before_receipt  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            Runner(plan, store, client).retry_frame(
                "shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=15
            )
        client.view = original_view  # type: ignore[method-assign]
        assert fake.prompt_count == 2
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-01/B"]["current_version"] == 1
        attempts = store.attempts_for_frame(state, "shot-01/B")
        assert [item["status"] for item in attempts] == ["COMPLETE", "SUBMITTED"]

        resumed_store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        result = Runner(plan, resumed_store, client).run(execute=True, deadline_seconds=15)
        assert result["status"] == "COMPLETE"
        assert fake.prompt_count == 2  # exact v2 poll, no new POST, v1 not masked
        state = resumed_store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-01/B"]["current_version"] == 2
        assert [item["status"] for item in state["attempts"]] == ["COMPLETE", "COMPLETE"]
        client.close()
    finally:
        fake.stop()


@pytest.mark.parametrize("mode", ["ambiguous_unknown", "terminal_failure"])
def test_new_ambiguous_or_failed_attempt_not_masked_by_old_receipt(
    tmp_path: Path, mode: str
) -> None:
    plan = _plan()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        fake.mode = mode
        with pytest.raises((AmbiguousDispatch, TerminalRemoteFailure)):
            Runner(plan, store, client).retry_frame(
                "shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=15
            )
        prompts = fake.prompt_count
        with pytest.raises((AmbiguousDispatch, TerminalRemoteFailure)):
            Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        assert fake.prompt_count == prompts
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-01/B"]["current_version"] == 1
        assert state["attempts"][-1]["status"] in {"AMBIGUOUS", "FAILED_TERMINAL"}
        client.close()
    finally:
        fake.stop()


# -------------------------------------------------------------------- F3
def test_export_is_readonly_and_new_revision_sees_all_unresolved(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["scenes"] = raw["scenes"][:1]
    original = json.dumps(raw, ensure_ascii=False)
    plan_path.write_text(original, encoding="utf-8")
    plan_a = resolve_plan(plan_path)
    state_dir = tmp_path / "state"
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store_a = PlanStore(state_dir, plan_a.plan_sha256, plan_a.path)
        Runner(plan_a, store_a, client).run(execute=True, deadline_seconds=15)

        data = json.loads(original)
        data["seed"] = 999
        plan_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        plan_b = resolve_plan(plan_path)
        fake.mode = "ambiguous_unknown"
        with pytest.raises(AmbiguousDispatch):
            Runner(plan_b, PlanStore(state_dir, plan_b.plan_sha256, plan_b.path), client).run(
                execute=True, deadline_seconds=15
            )
        current_files = list(Path(state_dir).glob("plans/*/current.json"))
        assert current_files
        current_before = json.loads(current_files[0].read_text(encoding="utf-8"))
        assert current_before["plan_sha256"] == plan_b.plan_sha256
        prompts_before_export = fake.prompt_count
        uploads_before_export = fake.upload_count

        plan_path.write_text(original, encoding="utf-8")
        plan_a2 = resolve_plan(plan_path)
        assert plan_a2.plan_sha256 == plan_a.plan_sha256
        export_dir = tmp_path / "export-a"
        code = cli.main(
            [
                "export",
                "--plan",
                str(plan_path),
                "--state-dir",
                str(state_dir),
                "--output",
                str(export_dir),
                "--json",
            ]
        )
        assert code == 0
        assert fake.prompt_count == prompts_before_export
        assert fake.upload_count == uploads_before_export
        current_after = json.loads(current_files[0].read_text(encoding="utf-8"))
        assert current_after == current_before

        data = json.loads(original)
        data["seed"] = 123
        plan_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        plan_c = resolve_plan(plan_path)
        with pytest.raises(StoreError, match="unresolved intent"):
            Runner(plan_c, PlanStore(state_dir, plan_c.plan_sha256, plan_c.path), client).run(
                execute=True, deadline_seconds=15
            )
        assert fake.prompt_count == prompts_before_export
        client.close()
    finally:
        fake.stop()


def test_export_of_unseen_plan_does_not_create_revision_state(tmp_path: Path) -> None:
    plan = _plan()
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    state = store.load_revision_readonly(plan=plan)
    assert state["plan_sha256"] == plan.plan_sha256
    assert not (tmp_path / "state" / "plans" / store.plan_key / "revisions" / plan.plan_sha256 / "state.json").exists()
    assert store.current_revision_sha256() is None
    export_storyboard(plan, state, tmp_path / "export")
    assert not (tmp_path / "state" / "plans" / store.plan_key / "revisions" / plan.plan_sha256 / "state.json").exists()


# -------------------------------------------------------------------- F4
def _valid_probe(profile, ctx_state) -> dict:
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
        "run_id": ctx_state["run_id"],
        "nonce": ctx_state["nonce"],
        "profile_sha256": profile.sha256,
        "observed_at": "2026-10-29T00:00:00Z",
        "object_info": {"sha256": "b" * 64},
        "models": [
            {
                "target": item["target"],
                "actual_size_bytes": item["size_bytes"],
                "actual_sha256": item["sha256"],
            }
            for item in profile.models
        ],
        "workflows": workflows,
    }


def _report(profile) -> dict:
    return {
        "ready": True,
        "status": "READY",
        "profile_sha256": profile.sha256,
        "model_bytes_verified": profile.model_bytes(),
        "workflow_hashes": {
            key: meta["api_sha256"] for key, meta in profile.workflows.items()
        },
        "checks": [
            {"name": name, "passed": True}
            for name in (
                "probe_schema",
                "profile_hash",
                "torch_cuda",
                "comfy_commit",
                "pip_check",
                "models",
                "object_info",
                "workflow_links",
                "output_schema",
                "system_stats",
                "workflow_schema",
                "tunnel_health",
            )
        ],
    }


def _ctx(profile=None):
    profile = profile or _profile()
    state = {
        "phase": "READY",
        "readiness_current": True,
        "readiness": _report(profile),
        "run_id": "run-1",
        "nonce": "nonce-1",
        "identity": {"workspace": "/workspace"},
    }
    return SimpleNamespace(
        profile=profile,
        state=state,
        profile_path=ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json",
        run_dir=Path("unused"),
        identity=object(),
    )


def _resume_factory(profile, *, tamper_model: bool = False, drop_probe: bool = False):
    def resume(ctx):
        probe = _valid_probe(profile, ctx.state)
        if tamper_model:
            probe["models"][0]["actual_sha256"] = "0" * 64
        state = dict(ctx.state)
        state["readiness"] = _report(profile)
        state["readiness_current"] = True
        return {
            "state": state,
            "readiness": state["readiness"],
            "probe": None if drop_probe else probe,
        }

    return resume


def test_fresh_runtime_verification_requires_actual_probe_and_model_hashes() -> None:
    ctx = _ctx()
    attestation = verify_kf1_runtime(ctx, resume_fn=_resume_factory(ctx.profile))
    assert attestation["i1_verified"] is True
    assert attestation["model_hashes"]
    assert attestation["profile_sha256"] == ctx.profile.sha256
    with pytest.raises(KF1Error, match="model bytes tampered"):
        verify_kf1_runtime(ctx, resume_fn=_resume_factory(ctx.profile, tamper_model=True))
    with pytest.raises(KF1Error, match="actual evidence"):
        verify_kf1_runtime(ctx, resume_fn=_resume_factory(ctx.profile, drop_probe=True))


def test_connection_probe_failure_causes_zero_upload_or_post(tmp_path: Path, monkeypatch) -> None:
    from scripts.i1 import ops

    fake = FakeComfy().start()
    ctx = _ctx()
    monkeypatch.setattr(ops, "load_context", lambda **kwargs: ctx)
    monkeypatch.setattr(
        ops, "resume", lambda context: (_ for _ in ()).throw(RuntimeError("probe tamper"))
    )
    with pytest.raises(KF1Error, match="actual I1 verification"):
        open_kf1_connection(
            base_url=None,
            connection="fake",
            state_dir=tmp_path,
            profile_path=ctx.profile_path,
        )
    assert fake.upload_count == 0
    assert fake.prompt_count == 0
    fake.stop()


def test_verify_rejects_probe_from_other_run() -> None:
    ctx = _ctx()
    def resume(other_ctx):
        probe = _valid_probe(ctx.profile, {"run_id": "other", "nonce": "other"})
        state = dict(ctx.state)
        state["readiness"] = _report(ctx.profile)
        return {"state": state, "readiness": state["readiness"], "probe": probe}

    with pytest.raises(KF1Error, match="run/nonce"):
        verify_kf1_runtime(ctx, resume_fn=resume)


# -------------------------------------------------------------------- F6
def _plan_with(tmp_path: Path, mutate) -> tuple[Path, object]:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path, resolve_plan(path)


def test_custom_geometry_export_pairs_complete_with_parent_derivative_identity(
    tmp_path: Path,
) -> None:
    def mutate(data):
        data["output_policy"]["candidate_geometry"] = [512, 768]

    plan_path, plan = _plan_with(tmp_path, mutate)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        state = store.load_revision_readonly(plan=plan)
        manifest = export_storyboard(plan, state, tmp_path / "export")
        for scene in manifest["scenes"]:
            assert {item["status"] for item in scene["roles"]} == {"COMPLETE"}
        b1 = manifest["scenes"][0]["roles"][1]
        assert b1["lineage"]["source_hashes"]["anchor_source_sha256"] == state["frames"][
            "shot-01/A"
        ]["versions"]["1"]["output"]["sha256"]
        assert b1["lineage"]["source_hashes"]["anchor_derivative_sha256"] != b1[
            "lineage"
        ]["source_hashes"]["anchor_source_sha256"]
        client.close()
    finally:
        fake.stop()


def test_other_first_frame_geometry_export_pairs_complete(tmp_path: Path) -> None:
    first = tmp_path / "first-720x1280.png"
    first.write_bytes(_png(720, 1280))

    def mutate(data):
        data["first_frame"] = str(first)

    plan_path, plan = _plan_with(tmp_path, mutate)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        Runner(plan, store, client).run(execute=True, deadline_seconds=15)
        manifest = export_storyboard(
            plan, store.load_revision_readonly(plan=plan), tmp_path / "export"
        )
        for scene in manifest["scenes"]:
            assert {item["status"] for item in scene["roles"]} == {"COMPLETE"}
        client.close()
    finally:
        fake.stop()


def test_gallery_unready_cards_never_render_none_uri(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=15)
        runner.retry_frame("shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=15)
        state = store.load_revision_readonly(plan=plan)
        manifest = export_storyboard(plan, state, tmp_path / "export")
        scene2 = manifest["scenes"][1]
        assert {item["status"] for item in scene2["roles"]} == {"INVALIDATED", "INVALIDATED"}
        html = (tmp_path / "export" / "index.html").read_text(encoding="utf-8")
        assert 'src="None"' not in html and 'href="None"' not in html
        assert "INVALIDATED" in html and "no usable image" in html
        client.close()
    finally:
        fake.stop()


# ----------------------------------------------------------------- extras
def test_source_contract_report_is_honest_about_regex_and_runtime_ranges() -> None:
    from scripts.kf1.source_contract import source_contract_report

    report = source_contract_report()
    assert report["actual_object_info"] == "NOT_RUN"
    assert report["source_extraction"] == "static_regex_from_pinned_snapshots"
    assert report["full_source_ast_proof"] == "NOT_IMPLEMENTED"
    assert report["numeric_ranges"] == "RUNTIME_OBJECT_INFO_ONLY"
    assert report["dynamic_enums"] == "RUNTIME_OBJECT_INFO_ONLY"


def test_history_foreign_extra_data_with_same_id_and_graph_is_rejected() -> None:
    graph = {"70": {"class_type": "SaveImage"}}
    sha = __import__("hashlib").sha256(
        json.dumps(graph, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    entry = {
        "prompt": [
            0,
            "prompt-1",
            graph,
            {"correlation_key": "foreign", "nonce": "n", "source": "foreign"},
            [],
        ],
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            "70": {
                "images": [
                    {"filename": "v_00001_.png", "subfolder": "", "type": "output"}
                ]
            }
        },
    }
    with pytest.raises(HistoryContractError, match="correlation_key"):
        output_from_history(
            entry,
            node_id="70",
            keys=["images"],
            expected_graph_sha256=sha,
            expected_prompt_id="prompt-1",
            expected_correlation_key="c",
            expected_nonce="n",
            expected_source="anime-video-studio-kf1",
        )


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png_with_ihdr(width: int, height: int, idat_raw: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(idat_raw, 9))
        + _chunk(b"IEND", b"")
    )


def _rewrite_ihdr_crc(data: bytes, width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return data[:8] + _chunk(b"IHDR", ihdr) + data[33:]


def test_png_valid_crc_oversized_header_and_decompressed_body_and_bomb_bounded() -> None:
    valid = _png(4, 4)
    with pytest.raises(PNGError, match="pixel count exceeds"):
        read_png(_rewrite_ihdr_crc(valid, 100_000, 100_000))

    # Valid CRCs and a small pixel count, but the decompressed body declares
    # more scanlines than the IHDR permits.
    extra = _png_with_ihdr(4, 4, b"\x00" * (4 * (1 + 12) + 2 * (1 + 12)))
    with pytest.raises(PNGError, match="length/terminal"):
        read_png(extra)

    bomb = _png_with_ihdr(4, 4, b"\x00" * (8 * 1024 * 1024))
    tracemalloc.start()
    try:
        with pytest.raises(PNGError, match="length/terminal"):
            read_png(bomb)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024


def test_progress_uses_scene_count_not_frame_count(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        lines: list[str] = []
        Runner(
            plan,
            PlanStore(tmp_path / "state", plan.plan_sha256, plan.path),
            client,
        ).run(execute=True, deadline_seconds=15, progress=lines.append)
        scene_lines = [line for line in lines if line.startswith("сцена ")]
        assert scene_lines
        assert any("2/2" in line for line in scene_lines)
        assert not any("4/4" in line for line in scene_lines)
        assert not any("3/2" in line for line in scene_lines)
        client.close()
    finally:
        fake.stop()


def test_evidence_capture_preserves_raw_utf8_bytes(tmp_path: Path) -> None:
    from scripts.kf1.evidence_capture import main as capture_main

    code = capture_main(
        [
            "--out",
            str(tmp_path / "evidence"),
            "--label",
            "utf8",
            "--",
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write('тест'.encode('utf-8')); "
            "sys.stderr.buffer.write(b'err'); raise SystemExit(7)",
        ]
    )
    assert code == 7
    base = tmp_path / "evidence" / "utf8"
    assert base.with_suffix(".stdout.bin").read_bytes() == "тест".encode("utf-8")
    assert base.with_suffix(".stdout.utf8.txt").read_text(encoding="utf-8") == "тест"
    meta = json.loads(base.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["exit_code"] == 7
    assert meta["argv"][0] == sys.executable
    assert meta["started_epoch"] <= meta["finished_epoch"]
