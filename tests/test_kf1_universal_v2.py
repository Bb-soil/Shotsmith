from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.kf1.approval import (
    approval_status,
    approve_frame,
    validate_approval,
    validate_saved_version,
)
from scripts.kf1.binding import bind, build_prompt, verify_saved_binding
from scripts.kf1.comfy import ComfyClient
from scripts.kf1.errors import (
    ApprovalRequired,
    BindingError,
    PlanError,
    StoreError,
    V1MigrationRequired,
)
from scripts.kf1.gallery import export_storyboard
from scripts.kf1.plan import resolve_plan
from scripts.kf1.runner import Runner
from scripts.kf1.source_contract import validate_workflow
from scripts.kf1.store import PlanStore
from tests.kf1_fake_comfy import FakeComfy, _png

ROOT = Path(__file__).resolve().parents[1]
OWNER_PLAN = ROOT / "config/keyframes/owner-feedback-20261030-v2.json"
GENERIC_PLAN = ROOT / "config/keyframes/two-scenes-v2.json"
CAR_PLAN = ROOT / "config/keyframes/car-approach-v2.json"
V1_PLAN = ROOT / "config/keyframes/two-scenes.example.json"


def _owner():
    return resolve_plan(OWNER_PLAN)


def _generic():
    return resolve_plan(GENERIC_PLAN)


def _car():
    return resolve_plan(CAR_PLAN)


def _run_to(plan, state_root: Path, stop_after: str):
    fake = FakeComfy().start()
    client = ComfyClient(fake.base_url)
    store = PlanStore(state_root, plan.plan_sha256, plan.path)
    runner = Runner(plan, store, client)
    result = runner.run(execute=True, deadline_seconds=10, stop_after=stop_after)
    return fake, client, store, runner, result


def _tree_hashes(root: Path) -> dict[str, str]:
    import hashlib

    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def test_v1_hash_frozen_and_fresh_generation_is_actionably_blocked(tmp_path: Path) -> None:
    plan = resolve_plan(V1_PLAN)
    assert plan.plan_sha256 == (
        "4305ed7833375b0c45f24c8b9ab4e021bff32d3d6ae24134df7230c86d795d99"
    )
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        with pytest.raises(V1MigrationRequired, match="v2"):
            Runner(plan, store, client).run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 0
        with pytest.raises(V1MigrationRequired, match="v2"):
            Runner(plan, store, client).retry_frame(
                "shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=10
            )
        assert fake.prompt_count == 0
        client.close()
    finally:
        fake.stop()


def test_v1_saved_binding_hash_and_readonly_export(tmp_path: Path) -> None:
    plan = resolve_plan(ROOT / "data/kf1/plans/session-2376920.json")
    store = PlanStore(ROOT / "data/kf1", plan.plan_sha256, plan.path)
    state = store.load_revision_readonly(plan=plan)
    frame = next(item for item in plan.frames if item.frame_id == "shot-02/B")
    version_info = state["frames"]["shot-02/B"]["versions"]["1"]
    verified = validate_saved_version(plan, store, state, frame, 1, version_info)
    assert verified["verified"]["stable_binding_sha256"] == (
        "b4172cf5a422319d21f089acb0533d858ce11e83ee6e502887f17d97410af977"
    )
    before = _tree_hashes(ROOT / "data/kf1/plans" / store.plan_key)
    manifest = export_storyboard(plan, state, tmp_path / "export", store=store)
    after = _tree_hashes(ROOT / "data/kf1/plans" / store.plan_key)
    assert before == after
    assert all(
        role.get("visual_status") == "NOT_REVIEWED"
        for scene in manifest["scenes"]
        for role in scene["roles"]
        if role.get("status") in {"COMPLETE", "PRESERVED"}
    )
    assert manifest["quality"] == "NOT_RUN"
    assert manifest["video_submitted"] is False


def test_v2_validation_disjoint_mode_camera_and_no_keyword_guess(tmp_path: Path) -> None:
    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    raw["scenes"][0]["b"]["invariants"].remove("camera")
    raw["scenes"][0]["b"]["allowed_changes"].append("camera")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PlanError, match="continue_shot"):
        resolve_plan(bad)

    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    raw["scenes"][0]["b"]["invariants"].append("position")
    raw["scenes"][0]["b"]["allowed_changes"].append("position")
    bad.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PlanError, match="пересекаются"):
        resolve_plan(bad)

    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    del raw["scenes"][0]["b"]["shot_mode"]
    bad.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PlanError, match="shot_mode"):
        resolve_plan(bad)


def test_v2_prompt_role_order_and_no_anime_human_defaults() -> None:
    prompt = build_prompt(
        shot_mode="continue_shot",
        edit_instruction="The car approaches the fixed camera.",
        description="A car on a road.",
        style="Clean automotive still.",
        roles=["anchor", "character_reference", "environment"],
        role_map={
            "anchor": "anchor",
            "character_reference": "identity",
            "environment": "environment",
        },
        invariants=["camera", "background", "identity"],
        allowed_changes=["position", "apparent_scale", "gaze", "pose"],
        subject_label="car",
    )
    for word in ("anime", "heroine", "hair", "outfit", "neck", "feet", "wind", "alley", "fear"):
        assert word not in prompt.lower()
    assert "Picture 1 is the current scene anchor" in prompt
    assert "Picture 2 defines subject identity" in prompt
    assert "Picture 3 defines the environment/world only" in prompt
    assert "Keep: camera position, background, subject identity" in prompt
    assert "Change only: subject position, apparent subject scale" in prompt


def test_binding_1_2_3_picture_order_positive_and_negative(tmp_path: Path) -> None:
    plan = _generic()
    frame = next(item for item in plan.frames if item.frame_id == "shot-01/B")
    roles = ["anchor", "character_reference", "environment"]
    for count in (1, 2, 3):
        image_names = {role: f"{role}.png" for role in roles[:count]}
        bound, meta = bind(
            ref_count=count,
            image_names=image_names,
            edit_instruction=frame.edit_instruction or "",
            description=frame.description,
            style=plan.data["style"],
            roles=roles[:count],
            role_map={
                "anchor": "anchor",
                "character_reference": "identity",
                "environment": "environment",
            },
            shot_mode=frame.shot_mode,
            invariants=frame.invariants,
            allowed_changes=frame.allowed_changes,
            subject_label=plan.subject_label,
            seed=plan.settings["seed"],
            version=1,
            plan_sha256=plan.plan_sha256,
            scene_id=frame.scene_id,
            role=frame.role,
            candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
            parent_version=1,
            settings=plan.settings,
        )
        assert validate_workflow(bound) == []
        assert meta["roles"] == roles[:count]
        assert bound["70"]["inputs"]["images"] == ["60", 0]
        for index, role in enumerate(roles[:count], start=1):
            assert bound[str(index)]["inputs"]["image"] == f"{role}.png"
            expected_link = ["10", 0] if index == 1 else [str(index), 0]
            assert bound["30"]["inputs"][f"image{index}"] == expected_link
            assert bound["31"]["inputs"][f"image{index}"] == expected_link
        assert bound["10"]["inputs"]["image"] == ["1", 0]
        assert bound["30"]["inputs"]["prompt"] == meta["positive_prompt"]
        assert bound["31"]["inputs"]["prompt"] == meta["negative_prompt"]
        assert "heroine" not in meta["positive_prompt"].lower()


def test_owner_stop_approve_resume_is_two_posts_and_exact(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        first = runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        assert first["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 1
        state = store.load_revision_readonly(plan=plan)
        a2 = state["frames"]["shot-02/A"]["versions"]["1"]
        assert state["frames"]["shot-01/B"]["versions"]["1"]["status"] == "PRESERVED"
        assert a2["status"] == "COMPLETE"
        assert not (store.version_dir("shot-02/A", 1) / "approval.json").exists()
        status = runner.status(state)
        a2_status = next(item for item in status["frames"] if item["frame_id"] == "shot-02/A")
        assert a2_status["reference_roles"] == ["character_reference"]
        assert a2_status["content_sha256"] == a2["output"]["sha256"]
        assert a2["output"]["sha256"] in a2_status["approval_command"]
        assert a2_status["approval_status"] == "MISSING"
        assert "shot-02/B" not in state["frames"] or state["frames"][
            "shot-02/B"
        ].get("current_version", 0) == 0
        waiting = runner.run(execute=True, deadline_seconds=10)
        assert waiting["status"] == "WAITING_APPROVAL"
        assert fake.prompt_count == 1
        approved = approve_frame(
            plan,
            store,
            frame_id="shot-02/A",
            version=1,
            sha256=a2["output"]["sha256"],
            owner_ack=True,
        )
        assert approved["status"] == "APPROVED"
        again = approve_frame(
            plan,
            store,
            frame_id="shot-02/A",
            version=1,
            sha256=a2["output"]["sha256"],
            owner_ack=True,
        )
        assert again["status"] == "ALREADY_APPROVED"
        final = runner.run(execute=True, deadline_seconds=10)
        assert final["status"] == "COMPLETE"
        assert fake.prompt_count == 2
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-02/B"]["versions"]["1"]["status"] == "COMPLETE"
        client.close()
    finally:
        fake.stop()


def test_approve_rejects_wrong_ack_hash_and_tamper_without_post(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        state = store.load_revision_readonly(plan=plan)
        info = state["frames"]["shot-02/A"]["versions"]["1"]
        sha = info["output"]["sha256"]
        with pytest.raises(StoreError, match="owner-ack"):
            approve_frame(
                plan, store, frame_id="shot-02/A", version=1, sha256=sha, owner_ack=False
            )
        with pytest.raises(StoreError, match="SHA256"):
            approve_frame(
                plan,
                store,
                frame_id="shot-02/A",
                version=1,
                sha256="0" * 64,
                owner_ack=True,
            )
        path = Path(info["output"]["path"])
        path.write_bytes(_png(704, 1280, (1, 2, 3)))
        with pytest.raises(StoreError, match="artifact"):
            approve_frame(
                plan,
                store,
                frame_id="shot-02/A",
                version=1,
                sha256=sha,
                owner_ack=True,
            )
        assert fake.prompt_count == 1
        client.close()
    finally:
        fake.stop()


def test_approval_does_not_transfer_to_new_version_or_revision(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        state = store.load_revision_readonly(plan=plan)
        a2 = state["frames"]["shot-02/A"]["versions"]["1"]
        approve_frame(
            plan,
            store,
            frame_id="shot-02/A",
            version=1,
            sha256=a2["output"]["sha256"],
            owner_ack=True,
        )
        runner.run(execute=True, deadline_seconds=10)
        prompts = fake.prompt_count
        runner.retry_frame("shot-02/A", owner_ack=True, raise_limit=True, deadline_seconds=10)
        assert fake.prompt_count == prompts + 1
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-02/A"]["current_version"] == 2
        assert state["frames"]["shot-02/B"].get("invalidated") is True
        assert approval_status(plan, store, state, "shot-02/A")["status"] == "MISSING"
        waiting = runner.run(execute=True, deadline_seconds=10)
        assert waiting["status"] == "WAITING_APPROVAL"
        assert fake.prompt_count == prompts + 1
        client.close()
    finally:
        fake.stop()


def test_unknown_until_frame_fails_before_network_and_complete_target_zero_post(
    tmp_path: Path,
) -> None:
    plan = _generic()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        with pytest.raises(StoreError, match="unknown --until-frame"):
            runner.run(execute=True, deadline_seconds=10, stop_after="shot-99/Z")
        assert fake.prompt_count == 0
        runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 3
        again = runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
        assert again["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 3
        client.close()
    finally:
        fake.stop()


def test_generic_stop_after_a2_does_not_upload_or_post_b2(tmp_path: Path) -> None:
    plan = _generic()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        assert result["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 2
        assert fake.upload_count == 3
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"].get("shot-02/B", {}).get("current_version", 0) == 0
        assert not (store.version_dir("shot-02/B", 1) / "intent.json").exists()
        runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 3
        assert fake.upload_count == 5
        client.close()
    finally:
        fake.stop()


def test_car_stop_approve_resume_has_no_human_template_words(tmp_path: Path) -> None:
    plan = _car()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        first = runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        assert first["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 2
        for payload in fake.prompts.values():
            prompt = payload["prompt"]
            text = " ".join(
                str(prompt[node]["inputs"]["prompt"])
                for node in ("30", "31")
                if node in prompt
            ).lower()
            for word in ("anime", "heroine", "hair", "outfit", "neck", "feet", "wind", "alley", "fear"):
                assert word not in text
        state = store.load_revision_readonly(plan=plan)
        a2 = state["frames"]["shot-02/A"]["versions"]["1"]
        approve_frame(
            plan,
            store,
            frame_id="shot-02/A",
            version=1,
            sha256=a2["output"]["sha256"],
            owner_ack=True,
        )
        final = runner.run(execute=True, deadline_seconds=10)
        assert final["status"] == "COMPLETE"
        assert fake.prompt_count == 3
        client.close()
    finally:
        fake.stop()


def test_provided_b1_source_tamper_fails_plan_resolution(tmp_path: Path) -> None:
    raw = json.loads(OWNER_PLAN.read_text(encoding="utf-8"))
    src = ROOT / raw["scenes"][0]["b"]["provided_frame"]["path"]
    copied = tmp_path / "b1.png"
    copied.write_bytes(src.read_bytes() + b"tamper")
    raw["scenes"][0]["b"]["provided_frame"]["path"] = str(copied)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PlanError, match="SHA256"):
        resolve_plan(plan_path)


def test_a2_without_composition_uses_lineage_only_and_custom_geometry(tmp_path: Path) -> None:
    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    raw["output_policy"]["candidate_geometry"] = [512, 768]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    plan = resolve_plan(plan_path)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        Runner(plan, store, client).run(execute=True, deadline_seconds=10)
        state = store.load_revision_readonly(plan=plan)
        a2 = state["frames"]["shot-02/A"]["versions"]["1"]
        source_hashes = a2["source_hashes"]
        assert source_hashes["anchor_policy"] == "lineage_only"
        assert source_hashes["anchor_uploaded"] is False
        assert source_hashes["anchor_source_sha256"] == state["frames"]["shot-01/B"][
            "versions"
        ]["1"]["output"]["sha256"]
        assert a2["output"]["width"] == 512
        assert a2["output"]["height"] == 768
        client.close()
    finally:
        fake.stop()


def test_ledger_retained_across_stop_and_plan_edit(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(GENERIC_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
    plan = resolve_plan(plan_path)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
        assert store.ledger_count() == 1
        runner.run(execute=True, deadline_seconds=10)
        assert store.ledger_count() == 3
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        data["title"] = "edited"
        plan_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        edited = resolve_plan(plan_path)
        assert edited.plan_sha256 != plan.plan_sha256
        edited_store = PlanStore(tmp_path / "state", edited.plan_sha256, edited.path)
        assert edited_store.ledger_count() == 3
        client.close()
    finally:
        fake.stop()


def test_v1_submitted_fixture_recovers_saved_binding_without_new_post(
    tmp_path: Path,
) -> None:
    plan = resolve_plan(V1_PLAN)
    old_root = ROOT / "data/kf1/plans/2eb40bcad7ea0192fb893c44cfadb3f4d550f0d81a4ab520db563e4100034dce/revisions/4305ed7833375b0c45f24c8b9ab4e021bff32d3d6ae24134df7230c86d795d99/frames/shot-01/B/v0001"
    old_intent = json.loads((old_root / "intent.json").read_text(encoding="utf-8"))
    old_binding = json.loads((old_root / "binding.json").read_text(encoding="utf-8"))
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    state = store.load(plan=plan)
    store.begin_attempt(
        state,
        frame_id="shot-01/B",
        version=1,
        intent=old_intent,
    )
    store.update_attempt(
        state,
        state["attempts"][-1]["attempt_id"],
        status="SUBMITTED",
        prompt_id=old_intent["expected_output"]["version"] and "prompt-old-1",
        submitted_at=0,
    )
    state["attempts"][-1]["prompt_id"] = "prompt-old-1"
    store.save(state)
    version_dir = store.ensure_version_dir("shot-01/B", 1)
    shutil.copy2(old_root / "binding.json", version_dir / "binding.json")
    shutil.copy2(old_root / "intent.json", version_dir / "intent.json")
    fake = FakeComfy().start()
    try:
        prompt_id = "prompt-old-1"
        output = {
            "filename": "v0001_00001_.png",
            "subfolder": old_intent["expected_output"]["subfolder"],
            "type": "output",
        }
        fake.output_map[output["filename"]] = _png(704, 1280)
        fake.history[prompt_id] = {
            "prompt": [
                0,
                prompt_id,
                old_binding["bound_workflow"],
                {
                    "correlation_key": old_intent["correlation_key"],
                    "nonce": old_intent["nonce"],
                    "source": "anime-video-studio-kf1",
                },
                [],
            ],
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {"70": {"images": [output]}},
        }
        client = ComfyClient(fake.base_url)
        result = Runner(plan, store, client).run(
            execute=True, deadline_seconds=10, stop_after="shot-01/B"
        )
        assert result["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 0
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-01/B"]["versions"]["1"]["status"] == "COMPLETE"
        client.close()
    finally:
        fake.stop()


def test_export_readonly_shows_approval_separately(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        state = store.load_revision_readonly(plan=plan)
        before = (store.state_path.read_bytes(), store.current_path.read_bytes())
        manifest = export_storyboard(plan, state, tmp_path / "export", store=store)
        after = (store.state_path.read_bytes(), store.current_path.read_bytes())
        assert before == after
        a2 = manifest["scenes"][1]["roles"][0]
        assert a2["approval_status"] == "MISSING"
        assert a2["visual_status"] == "NOT_REVIEWED"
        assert a2["quality"] == "NOT_RUN"
        assert manifest["quality"] == "NOT_RUN"
        assert manifest["video_submitted"] is False
        client.close()
    finally:
        fake.stop()

def test_provided_frame_approval_is_sidecar_only_no_png_copy(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
        assert result["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 0
        state = store.load_revision_readonly(plan=plan)
        b1 = state["frames"]["shot-01/B"]["versions"]["1"]
        before_pngs = sorted(
            path.relative_to(store.root).as_posix()
            for path in store.root.rglob("*.png")
        )
        approved = approve_frame(
            plan,
            store,
            frame_id="shot-01/B",
            version=1,
            sha256=b1["output"]["sha256"],
            owner_ack=True,
        )
        after_pngs = sorted(
            path.relative_to(store.root).as_posix()
            for path in store.root.rglob("*.png")
        )
        assert approved["status"] == "APPROVED"
        assert before_pngs == after_pngs
        assert not list(store.root.rglob("*.db"))
        assert not list(store.root.rglob("*.sqlite*"))
        assert (store.version_dir("shot-01/B", 1) / "approval.json").is_file()
        state = store.load_revision_readonly(plan=plan)
        assert approval_status(plan, store, state, "shot-01/B")["status"] == "APPROVED"
        client.close()
    finally:
        fake.stop()


def test_retry_dependent_gate_and_parent_tamper(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        with pytest.raises(ApprovalRequired):
            runner.retry_frame(
                "shot-02/B", owner_ack=True, raise_limit=True, deadline_seconds=10
            )
        assert fake.prompt_count == 1
        state = store.load_revision_readonly(plan=plan)
        a2 = state["frames"]["shot-02/A"]["versions"]["1"]
        approve_frame(
            plan,
            store,
            frame_id="shot-02/A",
            version=1,
            sha256=a2["output"]["sha256"],
            owner_ack=True,
        )
        approval_path = store.approval_path("shot-02/A", 1)
        record = json.loads(approval_path.read_text(encoding="utf-8"))
        record["parent_links"][0]["content_sha256"] = "0" * 64
        approval_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        state = store.load_revision_readonly(plan=plan)
        assert approval_status(plan, store, state, "shot-02/A")["status"] == "STALE"
        with pytest.raises(StoreError, match="parent_links"):
            runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 1
        client.close()
    finally:
        fake.stop()


def test_approval_does_not_transfer_to_new_plan_revision(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        state_root = tmp_path / "state"
        store = PlanStore(state_root, plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        state = store.load_revision_readonly(plan=plan)
        a2 = state["frames"]["shot-02/A"]["versions"]["1"]
        approve_frame(
            plan,
            store,
            frame_id="shot-02/A",
            version=1,
            sha256=a2["output"]["sha256"],
            owner_ack=True,
        )
        raw = json.loads(OWNER_PLAN.read_text(encoding="utf-8"))
        raw["title"] = "new authored v2 revision"
        new_path = tmp_path / "owner-feedback-new.json"
        new_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        new_plan = resolve_plan(new_path)
        assert new_plan.plan_sha256 != plan.plan_sha256
        new_store = PlanStore(state_root, new_plan.plan_sha256, new_plan.path)
        new_runner = Runner(new_plan, new_store, client)
        new_runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        assert fake.prompt_count == 2
        waiting = new_runner.run(execute=True, deadline_seconds=10)
        assert waiting["status"] == "WAITING_APPROVAL"
        assert fake.prompt_count == 2
        client.close()
    finally:
        fake.stop()


def test_generic_b_frames_keep_anchor_and_new_a_omits_it() -> None:
    plan = _generic()
    b1 = next(item for item in plan.frames if item.frame_id == "shot-01/B")
    a2 = next(item for item in plan.frames if item.frame_id == "shot-02/A")
    b2 = next(item for item in plan.frames if item.frame_id == "shot-02/B")
    assert b1.reference_roles[0] == "anchor"
    assert "character_reference" in b1.reference_roles
    assert a2.reference_roles == ("character_reference",)
    assert "anchor" not in a2.reference_roles
    assert b2.reference_roles[0] == "anchor"
    assert "character_reference" in b2.reference_roles


def test_runtime_profile_workflow_pins_unchanged() -> None:
    import hashlib

    from scripts.kf1.profiles import build_keyframe_profile
    from scripts.kf1.source_contract import source_hashes

    profile = json.loads(
        (ROOT / "config/i1/keyframes-qwen-edit-2511.profile.json").read_text(
            encoding="utf-8"
        )
    )
    assert build_keyframe_profile() == profile
    assert profile["models_policy"]["expected_count"] == 3
    assert {
        meta["output"]["class"] for meta in profile["workflows"].values()
    } == {"SaveImage"}
    hashes = source_hashes()
    assert hashes["nodes_qwen.py"] == (
        "54ce02aa9e167c0b497ce026ac6b558f539427d2d1ed4fc1fcc2ec8e6bc529da"
    )
    assert hashlib.sha256(
        (ROOT / "workflows/K1/qwen-edit-2511-2ref.api.json").read_bytes()
    ).hexdigest() == "2121e79ec4dd52d70fee80a7194a8f614f36973e8cdb3b722071e0982c9678d3"

def test_i1_namespace_hotfix_files_unchanged() -> None:
    import hashlib

    assert hashlib.sha256(
        (ROOT / "scripts/i1/probe.py").read_bytes()
    ).hexdigest() == "691954a0344d8715f62d9b0203ac3a0c1cc0308f18826b25ea64005c2354531b"
    assert hashlib.sha256(
        (ROOT / "tests/test_i1_probe_namespace.py").read_bytes()
    ).hexdigest() == "5693c71c561122f68fd751f7babbb4d05e257964125fa465329a0f11ef656bdc"

def test_cli_approve_exact_sha_offline(tmp_path: Path, capsys) -> None:
    from scripts.kf1 import cli

    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        state = store.load_revision_readonly(plan=plan)
        sha = state["frames"]["shot-02/A"]["versions"]["1"]["output"]["sha256"]
        code = cli.main(
            [
                "approve",
                "--plan",
                str(OWNER_PLAN),
                "--state-dir",
                str(tmp_path / "state"),
                "--frame",
                "shot-02/A",
                "--version",
                "1",
                "--sha256",
                sha,
                "--owner-ack",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "APPROVED"
        assert payload["approval"]["content_sha256"] == sha
        client.close()
    finally:
        fake.stop()

def test_v1_latest_submitted_retry_is_polled_not_masked_by_old_complete(
    tmp_path: Path,
) -> None:
    plan = resolve_plan(V1_PLAN)
    old_root = ROOT / "data/kf1/plans/2eb40bcad7ea0192fb893c44cfadb3f4d550f0d81a4ab520db563e4100034dce/revisions/4305ed7833375b0c45f24c8b9ab4e021bff32d3d6ae24134df7230c86d795d99"
    old_b1 = old_root / "frames/shot-01/B/v0001"
    old_intent = json.loads((old_b1 / "intent.json").read_text(encoding="utf-8"))
    old_binding = json.loads((old_b1 / "binding.json").read_text(encoding="utf-8"))
    old_state = json.loads((old_root / "state.json").read_text(encoding="utf-8"))
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    state = store.load(plan=plan)
    v1dir = store.ensure_version_dir("shot-01/B", 1)
    for name in ("frame.png", "anchor.png", "binding.json", "intent.json", "result.json"):
        shutil.copy2(old_b1 / name, v1dir / name)
    state["frames"]["shot-01/B"] = json.loads(
        json.dumps(old_state["frames"]["shot-01/B"])
    )
    state["frames"]["shot-01/B"]["current_version"] = 1
    intent2 = dict(old_intent)
    intent2["version"] = 2
    attempt = store.begin_attempt(
        state, frame_id="shot-01/B", version=2, intent=intent2
    )
    store.update_attempt(
        state,
        attempt["attempt_id"],
        status="SUBMITTED",
        prompt_id="prompt-v2",
        submitted_at=0,
    )
    store.save(state)
    v2dir = store.ensure_version_dir("shot-01/B", 2)
    shutil.copy2(old_b1 / "binding.json", v2dir / "binding.json")
    (v2dir / "intent.json").write_text(
        json.dumps(intent2, ensure_ascii=False), encoding="utf-8"
    )
    fake = FakeComfy().start()
    try:
        output = {
            "filename": "v0001_00001_.png",
            "subfolder": old_intent["expected_output"]["subfolder"],
            "type": "output",
        }
        fake.output_map[output["filename"]] = _png(704, 1280)
        fake.history["prompt-v2"] = {
            "prompt": [
                0,
                "prompt-v2",
                old_binding["bound_workflow"],
                {
                    "correlation_key": old_intent["correlation_key"],
                    "nonce": old_intent["nonce"],
                    "source": "anime-video-studio-kf1",
                },
                [],
            ],
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {"70": {"images": [output]}},
        }
        client = ComfyClient(fake.base_url)
        result = Runner(plan, store, client).run(
            execute=True, deadline_seconds=10, stop_after="shot-01/B"
        )
        assert result["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 0
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-01/B"]["current_version"] == 2
        assert state["frames"]["shot-01/B"]["versions"]["1"]["status"] == "COMPLETE"
        assert state["frames"]["shot-01/B"]["versions"]["2"]["status"] == "COMPLETE"
        client.close()
    finally:
        fake.stop()


def test_v1_latest_terminal_does_not_resurrect_old_complete(tmp_path: Path) -> None:
    plan = resolve_plan(V1_PLAN)
    old_root = ROOT / "data/kf1/plans/2eb40bcad7ea0192fb893c44cfadb3f4d550f0d81a4ab520db563e4100034dce/revisions/4305ed7833375b0c45f24c8b9ab4e021bff32d3d6ae24134df7230c86d795d99"
    old_b1 = old_root / "frames/shot-01/B/v0001"
    old_intent = json.loads((old_b1 / "intent.json").read_text(encoding="utf-8"))
    old_state = json.loads((old_root / "state.json").read_text(encoding="utf-8"))
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    state = store.load(plan=plan)
    v1dir = store.ensure_version_dir("shot-01/B", 1)
    for name in ("frame.png", "anchor.png", "binding.json", "intent.json", "result.json"):
        shutil.copy2(old_b1 / name, v1dir / name)
    state["frames"]["shot-01/B"] = json.loads(
        json.dumps(old_state["frames"]["shot-01/B"])
    )
    state["frames"]["shot-01/B"]["current_version"] = 1
    intent2 = dict(old_intent)
    intent2["version"] = 2
    attempt = store.begin_attempt(
        state, frame_id="shot-01/B", version=2, intent=intent2
    )
    store.update_attempt(
        state, attempt["attempt_id"], status="FAILED_TERMINAL", error="fake terminal"
    )
    store.save(state)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        with pytest.raises(Exception, match="terminal|migration"):
            Runner(plan, store, client).run(
                execute=True, deadline_seconds=10, stop_after="shot-01/B"
            )
        assert fake.prompt_count == 0
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-01/B"]["current_version"] == 1
        client.close()
    finally:
        fake.stop()


def test_global_scan_blocks_cross_plan_ambiguous_generate_and_retry(
    tmp_path: Path,
) -> None:
    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    path_a = tmp_path / "plan-a.json"
    path_b = tmp_path / "plan-b.json"
    raw["title"] = "plan A"
    path_a.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    raw["title"] = "plan B"
    path_b.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    plan_a = resolve_plan(path_a)
    plan_b = resolve_plan(path_b)
    fake = FakeComfy(mode="ambiguous_unknown").start()
    try:
        client = ComfyClient(fake.base_url)
        store_a = PlanStore(tmp_path / "state", plan_a.plan_sha256, plan_a.path)
        with pytest.raises(Exception):
            Runner(plan_a, store_a, client).run(execute=True, deadline_seconds=10)
        prompts = fake.prompt_count
        uploads = fake.upload_count
        store_b = PlanStore(tmp_path / "state", plan_b.plan_sha256, plan_b.path)
        with pytest.raises(StoreError, match="safety scan"):
            Runner(plan_b, store_b, client).run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == prompts
        assert fake.upload_count == uploads
        with pytest.raises(StoreError, match="safety scan"):
            Runner(plan_b, store_b, client).retry_frame(
                "shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=10
            )
        assert fake.prompt_count == prompts
        assert fake.upload_count == uploads
        client.close()
    finally:
        fake.stop()


def test_global_scan_allows_exact_submitted_recovery_while_other_plan_blocked(
    tmp_path: Path,
) -> None:
    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    path_a = tmp_path / "plan-a.json"
    path_b = tmp_path / "plan-b.json"
    raw["title"] = "plan A"
    path_a.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    raw["title"] = "plan B"
    path_b.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    plan_a = resolve_plan(path_a)
    plan_b = resolve_plan(path_b)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store_a = PlanStore(tmp_path / "state", plan_a.plan_sha256, plan_a.path)
        original_view = client.view
        client.view = lambda output: (_ for _ in ()).throw(RuntimeError("crash"))
        with pytest.raises(RuntimeError):
            Runner(plan_a, store_a, client).run(execute=True, deadline_seconds=10)
        client.view = original_view
        prompts = fake.prompt_count
        uploads = fake.upload_count
        store_b = PlanStore(tmp_path / "state", plan_b.plan_sha256, plan_b.path)
        with pytest.raises(StoreError, match="safety scan"):
            Runner(plan_b, store_b, client).run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == prompts
        assert fake.upload_count == uploads
        result = Runner(plan_a, store_a, client).run(
            execute=True, deadline_seconds=10, stop_after="shot-01/B"
        )
        assert result["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == prompts
        client.close()
    finally:
        fake.stop()


def test_global_scan_orphan_ledger_fails_without_deleting(tmp_path: Path) -> None:
    plan = _generic()
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    orphan = "a" * 64
    payload = {
        orphan: {
            "count": 1,
            "attempt_ids": ["orphan-attempt"],
            "plan_sha256": "b" * 64,
            "plan_path": str(tmp_path / "missing.json"),
        }
    }
    store.limits_path.write_text(
        json.dumps(payload), encoding="utf-8"
    )
    before = store.limits_path.read_bytes()
    with pytest.raises(StoreError, match="safety scan"):
        store.assert_global_clean_for_new_work()
    assert store.limits_path.read_bytes() == before


def test_generic_prompt_allows_multiple_subjects_and_no_identity_ref(
    tmp_path: Path,
) -> None:
    prompt = build_prompt(
        shot_mode="continue_shot",
        edit_instruction="Two cars approach the fixed camera.",
        description="Two cars approach on the road.",
        style="Clean automotive still.",
        roles=["anchor", "environment"],
        role_map={"anchor": "anchor", "environment": "environment"},
        invariants=["camera", "subject_count"],
        allowed_changes=["position", "apparent_scale"],
        subject_label="cars",
    )
    assert "Two cars approach" in prompt
    assert "subject count" in prompt
    assert "one coherent subject" not in prompt.lower()
    from scripts.kf1.binding import NEGATIVE_PROMPT

    assert "duplicate subject" not in NEGATIVE_PROMPT.lower()
    assert "extra subjects" not in NEGATIVE_PROMPT.lower()

    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    raw["character_reference"] = None
    raw["background_reference"] = "tests/fixtures/kf1/car-frame.png"
    raw["scenes"][0]["b"]["reference_roles"] = ["anchor", "background_reference"]
    raw["scenes"][1]["a"]["reference_roles"] = ["background_reference"]
    raw["scenes"][1]["b"]["reference_roles"] = ["anchor", "background_reference"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    plan = resolve_plan(path)
    assert "character_reference" not in plan.refs
    assert "background_reference" in plan.refs
    assert plan.refs["background_reference"].role == "environment"
    assert plan.frames[1].reference_roles == ("anchor", "background_reference")
    assert plan.frames[2].reference_roles == ("background_reference",)


def test_v2_canonical_includes_style_and_execution_fields(tmp_path: Path) -> None:
    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    base = resolve_plan(base_path)

    def changed(mutator) -> str:
        data = json.loads(json.dumps(raw))
        mutator(data)
        path = tmp_path / f"m-{len(list(tmp_path.glob('m-*')))}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return resolve_plan(path).plan_sha256

    assert changed(lambda d: d.__setitem__("style", "different style")) != base.plan_sha256
    assert changed(lambda d: d.__setitem__("seed", 99)) != base.plan_sha256
    assert changed(lambda d: d.__setitem__("generation_limit", 2)) != base.plan_sha256
    assert changed(lambda d: d["output_policy"].__setitem__("candidate_geometry", [512, 768])) != base.plan_sha256
    assert changed(lambda d: d["scenes"][0]["b"]["invariants"].append("lighting")) != base.plan_sha256
    assert changed(lambda d: d["scenes"][0]["b"]["allowed_changes"].append("gaze")) != base.plan_sha256
    assert changed(lambda d: d["scenes"][0]["b"].__setitem__("negative", "text")) != base.plan_sha256
    assert changed(lambda d: d["scenes"][1]["b"].__setitem__("require_anchor_approval", True)) != base.plan_sha256
    assert changed(lambda d: d.__setitem__("subject", {"label": "other", "kind": "object"})) != base.plan_sha256
    assert changed(lambda d: d.__setitem__("character_reference", "assets/frames/derived/shot-01-start-v3-704x1280.png")) != base.plan_sha256


def test_style_only_same_path_new_revision_keeps_old_approval(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(OWNER_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
    plan = resolve_plan(plan_path)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        state = store.load_revision_readonly(plan=plan)
        a2 = state["frames"]["shot-02/A"]["versions"]["1"]
        approve_frame(
            plan,
            store,
            frame_id="shot-02/A",
            version=1,
            sha256=a2["output"]["sha256"],
            owner_ack=True,
        )
        old_approval = store.approval_path("shot-02/A", 1).read_bytes()
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        raw["style"] = raw["style"] + " Different authored style only."
        plan_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        new_plan = resolve_plan(plan_path)
        assert new_plan.plan_sha256 != plan.plan_sha256
        new_store = PlanStore(tmp_path / "state", new_plan.plan_sha256, new_plan.path)
        new_store.load(plan=new_plan)
        assert new_store.approval_path("shot-02/A", 1).exists() is False
        assert store.approval_path("shot-02/A", 1).read_bytes() == old_approval
        client.close()
    finally:
        fake.stop()


def test_approval_each_mandatory_field_mutation_is_stale_and_blocks_post(
    tmp_path: Path,
) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-02/A")
        state = store.load_revision_readonly(plan=plan)
        a2 = state["frames"]["shot-02/A"]["versions"]["1"]
        approve_frame(
            plan,
            store,
            frame_id="shot-02/A",
            version=1,
            sha256=a2["output"]["sha256"],
            owner_ack=True,
        )
        approval_path = store.approval_path("shot-02/A", 1)
        good = json.loads(approval_path.read_text(encoding="utf-8"))
        prompts = fake.prompt_count

        def mutations(record):
            for value in (False, 1, "true"):
                item = json.loads(json.dumps(record))
                item["owner_ack"] = value
                yield f"owner_ack={value!r}", item
            for key, value in (
                ("approved_by", "system"),
                ("prompt_id", "wrong"),
                ("bound_workflow_sha256", "0" * 64),
                ("binding_sha256", "0" * 64),
                ("stable_binding_sha256", "0" * 64),
                ("attempt_id", "wrong"),
                ("frame_id", "shot-01/B"),
                ("revision_id", "0" * 64),
                ("plan_sha256", "0" * 64),
                ("content_sha256", "0" * 64),
                ("artifact_path", "wrong"),
                ("size_bytes", 1),
                ("width", 1),
                ("height", 1),
                ("approval_id", "bad"),
                ("approved_at", "NaN"),
                ("quality", "PASS"),
                ("schema", "bad"),
                ("note", 1),
                ("version", 2),
            ):
                item = json.loads(json.dumps(record))
                item[key] = value
                yield f"{key}={value!r}", item
            item = json.loads(json.dumps(record))
            item["parent_links"][0]["content_sha256"] = "0" * 64
            yield "parent_links", item
            item = json.loads(json.dumps(record))
            item["source_hashes"]["pins_sha256"] = "0" * 64
            yield "source_hashes", item

        for label, item in mutations(good):
            approval_path.write_text(
                json.dumps(item, ensure_ascii=False), encoding="utf-8"
            )
            state = store.load_revision_readonly(plan=plan)
            assert approval_status(plan, store, state, "shot-02/A")["status"] == "STALE", label
            with pytest.raises(StoreError):
                runner.run(execute=True, deadline_seconds=10)
            assert fake.prompt_count == prompts, label

        # Invalid existing sidecar must not be overwritten by approve.
        approval_path.write_text(json.dumps(good, ensure_ascii=False), encoding="utf-8")
        item = json.loads(json.dumps(good))
        item["prompt_id"] = "wrong"
        approval_path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        before = approval_path.read_bytes()
        with pytest.raises(StoreError, match="full validation|overwrite"):
            approve_frame(
                plan,
                store,
                frame_id="shot-02/A",
                version=1,
                sha256=a2["output"]["sha256"],
                owner_ack=True,
            )
        assert approval_path.read_bytes() == before

        approval_path.write_text(json.dumps(good, ensure_ascii=False), encoding="utf-8")
        state = store.load_revision_readonly(plan=plan)
        assert approval_status(plan, store, state, "shot-02/A")["status"] == "APPROVED"
        client.close()
    finally:
        fake.stop()


def test_provided_approval_records_explicit_none_generated_fields(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
        state = store.load_revision_readonly(plan=plan)
        b1 = state["frames"]["shot-01/B"]["versions"]["1"]
        approve_frame(
            plan,
            store,
            frame_id="shot-01/B",
            version=1,
            sha256=b1["output"]["sha256"],
            owner_ack=True,
        )
        record = json.loads(
            store.approval_path("shot-01/B", 1).read_text(encoding="utf-8")
        )
        assert record["prompt_id"] is None
        assert record["binding_sha256"] is None
        assert record["bound_workflow_sha256"] is None
        assert record["attempt_id"] is None
        state = store.load_revision_readonly(plan=plan)
        assert approval_status(plan, store, state, "shot-01/B")["status"] == "APPROVED"
        client.close()
    finally:
        fake.stop()

def _v1_latest_fixture(tmp_path: Path, status: str, prompt_id: str | None):
    plan = resolve_plan(V1_PLAN)
    old_root = ROOT / "data/kf1/plans/2eb40bcad7ea0192fb893c44cfadb3f4d550f0d81a4ab520db563e4100034dce/revisions/4305ed7833375b0c45f24c8b9ab4e021bff32d3d6ae24134df7230c86d795d99"
    old_b1 = old_root / "frames/shot-01/B/v0001"
    old_intent = json.loads((old_b1 / "intent.json").read_text(encoding="utf-8"))
    old_binding = json.loads((old_b1 / "binding.json").read_text(encoding="utf-8"))
    old_state = json.loads((old_root / "state.json").read_text(encoding="utf-8"))
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    state = store.load(plan=plan)
    v1dir = store.ensure_version_dir("shot-01/B", 1)
    for name in ("frame.png", "anchor.png", "binding.json", "intent.json", "result.json"):
        shutil.copy2(old_b1 / name, v1dir / name)
    state["frames"]["shot-01/B"] = json.loads(json.dumps(old_state["frames"]["shot-01/B"]))
    state["frames"]["shot-01/B"]["current_version"] = 1
    intent2 = dict(old_intent)
    intent2["version"] = 2
    attempt = store.begin_attempt(state, frame_id="shot-01/B", version=2, intent=intent2)
    store.update_attempt(
        state,
        attempt["attempt_id"],
        status=status,
        **({"prompt_id": prompt_id} if prompt_id is not None else {}),
    )
    store.save(state)
    v2dir = store.ensure_version_dir("shot-01/B", 2)
    shutil.copy2(old_b1 / "binding.json", v2dir / "binding.json")
    (v2dir / "intent.json").write_text(
        json.dumps(intent2, ensure_ascii=False), encoding="utf-8"
    )
    return plan, store, old_intent, old_binding


def test_v1_posting_without_prompt_marks_ambiguous_no_post(tmp_path: Path) -> None:
    plan, store, old_intent, old_binding = _v1_latest_fixture(
        tmp_path, "POSTING", None
    )
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        with pytest.raises(Exception, match="AMBIGUOUS|ambiguous|не восстановлен"):
            Runner(plan, store, client).run(
                execute=True, deadline_seconds=10, stop_after="shot-01/B"
            )
        assert fake.prompt_count == 0
        state = store.load_revision_readonly(plan=plan)
        assert state["attempts"][-1]["status"] == "AMBIGUOUS"
        assert state["frames"]["shot-01/B"]["current_version"] == 1
        client.close()
    finally:
        fake.stop()


def test_v1_known_posting_and_ambiguous_recover_exact_without_post(
    tmp_path: Path,
) -> None:
    for status in ("POSTING", "AMBIGUOUS"):
        case = tmp_path / status.lower()
        plan, store, old_intent, old_binding = _v1_latest_fixture(
            case, status, f"prompt-{status.lower()}"
        )
        fake = FakeComfy().start()
        try:
            output = {
                "filename": "v0001_00001_.png",
                "subfolder": old_intent["expected_output"]["subfolder"],
                "type": "output",
            }
            fake.output_map[output["filename"]] = _png(704, 1280)
            prompt_id = f"prompt-{status.lower()}"
            fake.history[prompt_id] = {
                "prompt": [
                    0,
                    prompt_id,
                    old_binding["bound_workflow"],
                    {
                        "correlation_key": old_intent["correlation_key"],
                        "nonce": old_intent["nonce"],
                        "source": "anime-video-studio-kf1",
                    },
                    [],
                ],
                "status": {"status_str": "success", "completed": True, "messages": []},
                "outputs": {"70": {"images": [output]}},
            }
            client = ComfyClient(fake.base_url)
            result = Runner(plan, store, client).run(
                execute=True, deadline_seconds=10, stop_after="shot-01/B"
            )
            assert result["status"] == "STOPPED_AFTER"
            assert fake.prompt_count == 0
            state = store.load_revision_readonly(plan=plan)
            assert state["frames"]["shot-01/B"]["current_version"] == 2
            client.close()
        finally:
            fake.stop()


def test_v1_bad_saved_binding_refuses_recovery_without_post(
    tmp_path: Path,
) -> None:
    plan, store, old_intent, old_binding = _v1_latest_fixture(
        tmp_path, "SUBMITTED", "prompt-bad"
    )
    binding_path = store.version_dir("shot-01/B", 2) / "binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["positive_prompt"] = "tampered prompt"
    binding_path.write_text(json.dumps(binding, ensure_ascii=False), encoding="utf-8")
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        with pytest.raises((StoreError, BindingError)):
            Runner(plan, store, client).run(
                execute=True, deadline_seconds=10, stop_after="shot-01/B"
            )
        assert fake.prompt_count == 0
        client.close()
    finally:
        fake.stop()


def test_cross_plan_global_os_lock_is_shared(tmp_path: Path) -> None:
    plan_a = _generic()
    raw = json.loads(GENERIC_PLAN.read_text(encoding="utf-8"))
    raw["title"] = "other"
    path_b = tmp_path / "plan-b.json"
    path_b.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    plan_b = resolve_plan(path_b)
    store_a = PlanStore(tmp_path / "state", plan_a.plan_sha256, plan_a.path)
    store_b = PlanStore(tmp_path / "state", plan_b.plan_sha256, plan_b.path)
    lock_a = store_a.lock()
    lock_a.acquire()
    try:
        with pytest.raises(Exception):
            store_b.lock().acquire()
    finally:
        lock_a.release()


def test_provided_provenance_is_in_canonical_hash(tmp_path: Path) -> None:
    # Use a valid copy with provenance mutation only.
    original = json.loads(OWNER_PLAN.read_text(encoding="utf-8"))
    original_path = tmp_path / "base.json"
    original_path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    base = resolve_plan(original_path)
    original["scenes"][0]["b"]["provided_frame"]["provenance"]["attempt_id"] = "x" * 32
    mutated_path = tmp_path / "mutated.json"
    mutated_path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    mutated = resolve_plan(mutated_path)
    assert mutated.plan_sha256 != base.plan_sha256

def _v1_seed_fixture(tmp_path: Path, *, seed: int = 0, attempt_seed: int | None = None):
    import hashlib

    plan = resolve_plan(V1_PLAN)
    frame = next(item for item in plan.frames if item.frame_id == "shot-01/B")
    bound, meta = bind(
        ref_count=2,
        image_names={"anchor": "a.png", "character": "c.png"},
        edit_instruction=frame.edit_instruction or "",
        description=frame.description,
        style=plan.data["style"],
        mode="b",
        seed=seed,
        version=1,
        plan_sha256=plan.plan_sha256,
        scene_id=frame.scene_id,
        role=frame.role,
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        parent_version=1,
        settings=plan.settings,
    )
    graph_sha = hashlib.sha256(
        json.dumps(bound, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    meta["bound_workflow"] = bound
    meta["bound_workflow_sha256"] = graph_sha
    source_hashes = {
        "pins_sha256": hashlib.sha256(
            json.dumps(
                plan.pins, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest(),
        "plan_sha256": plan.plan_sha256,
        "revision_id": plan.plan_sha256,
        "anchor": plan.refs["first_frame"].sha256,
        "anchor_version": 1,
        "character": plan.refs["character_reference"].sha256,
    }
    intent = {
        "frame_id": frame.frame_id,
        "scene_id": frame.scene_id,
        "role": frame.role,
        "version": 1,
        "binding_sha256": meta["stable_binding_sha256"],
        "bound_workflow_sha256": graph_sha,
        "expected_output": meta["expected_output"],
        "settings": meta["settings"],
        "prompt": meta["positive_prompt"],
        "negative_prompt": meta["negative_prompt"],
        "uploads": meta["uploads"],
        "source_hashes": source_hashes,
        "workflow_sha256": meta["workflow_sha256"],
        "seed": seed,
        "correlation_key": "corr-seed",
        "nonce": "nonce-seed",
    }
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    state = store.load(plan=plan)
    attempt = store.begin_attempt(
        state, frame_id="shot-01/B", version=1, intent=intent
    )
    update = {"status": "SUBMITTED", "prompt_id": "prompt-seed"}
    if attempt_seed is not None:
        update["seed"] = attempt_seed
    store.update_attempt(state, attempt["attempt_id"], **update)
    store.save(state)
    version_dir = store.ensure_version_dir("shot-01/B", 1)
    (version_dir / "binding.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    (version_dir / "intent.json").write_text(
        json.dumps(intent, ensure_ascii=False), encoding="utf-8"
    )
    return plan, store, bound, meta


def test_v1_seed_zero_exact_recovery_without_post(tmp_path: Path) -> None:
    plan, store, bound, meta = _v1_seed_fixture(tmp_path, seed=0)
    fake = FakeComfy().start()
    try:
        output = {
            "filename": "v0001_00001_.png",
            "subfolder": meta["expected_output"]["subfolder"],
            "type": "output",
        }
        fake.output_map[output["filename"]] = _png(704, 1280)
        fake.history["prompt-seed"] = {
            "prompt": [
                0,
                "prompt-seed",
                bound,
                {
                    "correlation_key": "corr-seed",
                    "nonce": "nonce-seed",
                    "source": "anime-video-studio-kf1",
                },
                [],
            ],
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {"70": {"images": [output]}},
        }
        client = ComfyClient(fake.base_url)
        result = Runner(plan, store, client).run(
            execute=True, deadline_seconds=10, stop_after="shot-01/B"
        )
        assert result["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 0
        state = store.load_revision_readonly(plan=plan)
        assert state["frames"]["shot-01/B"]["versions"]["1"]["status"] == "COMPLETE"
        client.close()
    finally:
        fake.stop()


def test_v1_seed_tamper_rejected_without_post(tmp_path: Path) -> None:
    plan, store, bound, meta = _v1_seed_fixture(
        tmp_path, seed=0, attempt_seed=1
    )
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        with pytest.raises(StoreError, match="seed"):
            Runner(plan, store, client).run(
                execute=True, deadline_seconds=10, stop_after="shot-01/B"
            )
        assert fake.prompt_count == 0
        client.close()
    finally:
        fake.stop()


def _completed_b1_plan(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(GENERIC_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
    plan = resolve_plan(plan_path)
    fake = FakeComfy().start()
    client = ComfyClient(fake.base_url)
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    runner = Runner(plan, store, client)
    runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
    return plan, store, runner, fake, client


def test_ledger_root_null_or_list_fails_closed_before_new_work(
    tmp_path: Path,
) -> None:
    for label, payload in (
        ("list", "[]"),
        ("null", "null"),
    ):
        case = tmp_path / label
        plan, store, runner, fake, client = _completed_b1_plan(case)
        try:
            prompts = fake.prompt_count
            uploads = fake.upload_count
            store.limits_path.write_text(payload, encoding="utf-8")
            before = store.limits_path.read_bytes()
            with pytest.raises(StoreError):
                runner.run(execute=True, deadline_seconds=10)
            assert fake.prompt_count == prompts
            assert fake.upload_count == uploads
            assert store.limits_path.read_bytes() == before
            assert not (store.version_dir("shot-02/A", 1) / "intent.json").exists()
            client.close()
        finally:
            fake.stop()


def test_ledger_missing_entry_or_file_after_completed_blocks_new_work(
    tmp_path: Path,
) -> None:
    for label in ("missing_entry", "missing_file"):
        case = tmp_path / label
        plan, store, runner, fake, client = _completed_b1_plan(case)
        try:
            prompts = fake.prompt_count
            uploads = fake.upload_count
            if label == "missing_entry":
                ledger = json.loads(store.limits_path.read_text(encoding="utf-8"))
                ledger.pop(store.plan_key)
                store.limits_path.write_text(
                    json.dumps(ledger), encoding="utf-8"
                )
                before = store.limits_path.read_bytes()
            else:
                store.limits_path.unlink()
                before = None
            with pytest.raises(StoreError, match="budget|ledger|reservation"):
                runner.run(execute=True, deadline_seconds=10)
            assert fake.prompt_count == prompts
            assert fake.upload_count == uploads
            if before is not None:
                assert store.limits_path.read_bytes() == before
            else:
                assert not store.limits_path.exists()
            assert not (store.version_dir("shot-02/A", 1) / "intent.json").exists()
            client.close()
        finally:
            fake.stop()


def test_preserved_only_plan_has_no_ledger_and_is_valid(tmp_path: Path) -> None:
    plan = _owner()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
        assert result["status"] == "STOPPED_AFTER"
        assert not store.limits_path.exists()
        store.assert_global_clean_for_new_work()
        client.close()
    finally:
        fake.stop()


def test_ledger_malformed_count_ids_mismatch_fail_closed(tmp_path: Path) -> None:
    plan = _generic()
    store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
    cases = [
        {"count": 1, "attempt_ids": ["a", "a"]},
        {"count": 2, "attempt_ids": ["a"]},
        {"count": True, "attempt_ids": []},
        {"count": 0, "attempt_ids": [1]},
        {"count": 0, "attempt_ids": []},
    ]
    for index, entry in enumerate(cases):
        payload = {store.plan_key: entry}
        store.limits_path.write_text(json.dumps(payload), encoding="utf-8")
        before = store.limits_path.read_bytes()
        if index == 4:
            store.assert_global_clean_for_new_work()
        else:
            with pytest.raises(StoreError):
                store.assert_global_clean_for_new_work()
        assert store.limits_path.read_bytes() == before

def test_provided_source_v2_imported_as_local_v1_requires_new_ack(
    tmp_path: Path,
) -> None:
    # Isolated source fixture: generate B1 v1, then explicit retry to v2.
    source_plan_path = tmp_path / "source-plan.json"
    source_plan_path.write_text(GENERIC_PLAN.read_text(encoding="utf-8"), encoding="utf-8")
    source_plan = resolve_plan(source_plan_path)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        source_store = PlanStore(tmp_path / "source-state", source_plan.plan_sha256, source_plan.path)
        source_runner = Runner(source_plan, source_store, client)
        source_runner.run(execute=True, deadline_seconds=10)
        source_runner.retry_frame(
            "shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=10
        )
        source_state = source_store.load_revision_readonly(plan=source_plan)
        source_info = source_state["frames"]["shot-01/B"]["versions"]["2"]
        source_result = json.loads(
            (source_store.version_dir("shot-01/B", 2) / "result.json").read_text(
                encoding="utf-8"
            )
        )
        source_path = Path(source_info["output"]["path"])
        source_sha = source_info["output"]["sha256"]

        raw = json.loads(OWNER_PLAN.read_text(encoding="utf-8"))
        raw["title"] = "local imported source v2"
        provided = raw["scenes"][0]["b"]["provided_frame"]
        provided["path"] = str(source_path)
        provided["sha256"] = source_sha
        provided["provenance"] = {
            "plan_sha256": source_plan.plan_sha256,
            "revision_id": source_plan.plan_sha256,
            "frame_id": "shot-01/B",
            "version": 2,
            "binding_sha256": source_info["binding_sha256"],
            "attempt_id": source_info["attempt_id"],
            "content_sha256": source_sha,
            "prompt_id": source_result["output"]["prompt_id"],
        }
        local_path = tmp_path / "local-plan.json"
        local_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        local_plan = resolve_plan(local_path)
        local_store = PlanStore(tmp_path / "local-state", local_plan.plan_sha256, local_plan.path)
        local_runner = Runner(local_plan, local_store, client)
        local_runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
        local_state = local_store.load_revision_readonly(plan=local_plan)
        local_info = local_state["frames"]["shot-01/B"]["versions"]["1"]
        assert local_info["status"] == "PRESERVED"
        approve_frame(
            local_plan,
            local_store,
            frame_id="shot-01/B",
            version=1,
            sha256=local_info["output"]["sha256"],
            owner_ack=True,
        )
        record = json.loads(
            local_store.approval_path("shot-01/B", 1).read_text(encoding="utf-8")
        )
        assert record["version"] == 1
        assert record["content_sha256"] == local_info["output"]["sha256"]
        assert record["source_provenance"]["version"] == 2
        assert record["source_provenance"]["frame_id"] == "shot-01/B"
        assert record["source_provenance"]["source_state_verified"] is True
        assert approval_status(
            local_plan, local_store, local_state, "shot-01/B"
        )["status"] == "APPROVED"

        # Origin mismatch tamper must be rejected, not approved.
        tampered = json.loads(json.dumps(raw))
        tampered["title"] = "tampered source origin"
        tampered["scenes"][0]["b"]["provided_frame"]["provenance"]["binding_sha256"] = "0" * 64
        tampered_path = tmp_path / "tampered-plan.json"
        tampered_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
        tampered_plan = resolve_plan(tampered_path)
        tampered_store = PlanStore(tmp_path / "tampered-state", tampered_plan.plan_sha256, tampered_plan.path)
        Runner(tampered_plan, tampered_store, client).run(
            execute=True, deadline_seconds=10, stop_after="shot-01/B"
        )
        tampered_state = tampered_store.load_revision_readonly(plan=tampered_plan)
        tampered_info = tampered_state["frames"]["shot-01/B"]["versions"]["1"]
        with pytest.raises(StoreError, match="source"):
            approve_frame(
                tampered_plan,
                tampered_store,
                frame_id="shot-01/B",
                version=1,
                sha256=tampered_info["output"]["sha256"],
                owner_ack=True,
            )

        # Local revision replacement does not inherit the old local approval.
        raw2 = json.loads(json.dumps(raw))
        raw2["style"] = raw2["style"] + " New local revision style."
        local_path.write_text(json.dumps(raw2, ensure_ascii=False), encoding="utf-8")
        local_plan2 = resolve_plan(local_path)
        local_store2 = PlanStore(
            tmp_path / "local-state", local_plan2.plan_sha256, local_plan2.path
        )
        Runner(local_plan2, local_store2, client).run(
            execute=True, deadline_seconds=10, stop_after="shot-01/B"
        )
        local_state2 = local_store2.load_revision_readonly(plan=local_plan2)
        assert approval_status(
            local_plan2, local_store2, local_state2, "shot-01/B"
        )["status"] != "APPROVED"
        client.close()
    finally:
        fake.stop()
