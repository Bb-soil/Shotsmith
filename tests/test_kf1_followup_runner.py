from __future__ import annotations

import json
from pathlib import Path

from scripts.kf1.comfy import ComfyClient
from scripts.kf1.plan import resolve_plan
from scripts.kf1.runner import Runner
from scripts.kf1.store import PlanStore
from tests.kf1_fake_comfy import FakeComfy, _png


def _two_identity_plan(tmp_path: Path) -> Path:
    source = json.loads(
        (Path(__file__).resolve().parents[1] / "config/keyframes/two-scenes-v2.json").read_text(
            encoding="utf-8"
        )
    )
    side = tmp_path / "girl-side.png"
    side.write_bytes(_png(32, 32, (90, 10, 20)))
    source["references"] = [
        {
            "id": "id_side",
            "role": "identity",
            "path": str(side),
            "sha256": __import__("hashlib").sha256(side.read_bytes()).hexdigest(),
        }
    ]
    source["scenes"][0]["b"]["reference_roles"] = [
        "anchor",
        "character_reference",
        "id_side",
    ]
    source["generation_limit"] = 1
    path = tmp_path / "two-identity-plan.json"
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    return path


def test_runner_uploads_two_identity_refs_in_picture_order(tmp_path: Path) -> None:
    plan = resolve_plan(_two_identity_plan(tmp_path))
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
        assert result["status"] == "STOPPED_AFTER"
        assert fake.prompt_count == 1
        state = store.load_revision_readonly(plan=plan)
        version_dir = store.version_dir("shot-01/B", 1)
        binding = json.loads((version_dir / "binding.json").read_text(encoding="utf-8"))
        assert binding["roles"] == ["anchor", "character_reference", "id_side"]
        assert binding["role_semantics"] == ["anchor", "identity", "identity"]
        assert len({item.get("name") or item.get("remote_name") for item in binding["uploads"]}) == 3
        assert "Picture 2 defines subject identity" in binding["positive_prompt"]
        assert "Picture 3 defines subject identity" in binding["positive_prompt"]
        assert "identity reference 1 of 2" in binding["positive_prompt"]
        assert "identity reference 2 of 2" in binding["positive_prompt"]
        names = [item.get("name") or item.get("remote_name") for item in binding["uploads"]]
        assert binding["bound_workflow"]["1"]["inputs"]["image"] == names[0]
        assert binding["bound_workflow"]["2"]["inputs"]["image"] == names[1]
        assert binding["bound_workflow"]["3"]["inputs"]["image"] == names[2]
        assert state["frames"]["shot-01/B"]["versions"]["1"]["status"] == "COMPLETE"
    finally:
        fake.stop()


def test_runner_uses_environment_policy_for_custom_environment_ref(tmp_path: Path) -> None:
    source = json.loads(
        (Path(__file__).resolve().parents[1] / "config/keyframes/two-scenes-v2.json").read_text(
            encoding="utf-8"
        )
    )
    env = tmp_path / "custom-env.png"
    env.write_bytes(_png(32, 32, (7, 9, 11)))
    source["references"] = [
        {
            "id": "custom_env",
            "role": "environment",
            "path": str(env),
            "sha256": __import__("hashlib").sha256(env.read_bytes()).hexdigest(),
        }
    ]
    source["scenes"][0]["b"]["reference_roles"] = ["anchor", "custom_env"]
    source["generation_limit"] = 1
    source["output_policy"]["character_policy"] = "contain"
    source["output_policy"]["environment_policy"] = "crop"
    path = tmp_path / "custom-env-plan.json"
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    plan = resolve_plan(path)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state-env", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10, stop_after="shot-01/B")
        assert result["status"] == "STOPPED_AFTER"
        refs = json.loads((store.root / "refs" / "refs.json").read_text(encoding="utf-8"))
        assert refs["custom_env"]["policy"] == "crop"
    finally:
        fake.stop()


def _base_source():
    return json.loads(
        (Path(__file__).resolve().parents[1] / "config/keyframes/two-scenes-v2.json").read_text(
            encoding="utf-8"
        )
    )


def _add_ref(source: dict, ref_id: str, role: str, path: Path) -> None:
    source.setdefault("references", []).append(
        {
            "id": ref_id,
            "role": role,
            "path": str(path),
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        }
    )


def _run_plan(tmp_path: Path, source: dict, stop_after: str):
    path = tmp_path / ("plan-" + source["title"].replace(" ", "-") + ".json")
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    plan = resolve_plan(path)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / ("state-" + source["title"].replace(" ", "-")), plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10, stop_after=stop_after)
        refs = json.loads((store.root / "refs" / "refs.json").read_text(encoding="utf-8"))
        return result, refs, store
    finally:
        fake.stop()


def test_materialized_ref_names_avoid_legacy_and_casefold_collisions(tmp_path: Path) -> None:
    source = _base_source()
    side = tmp_path / "side.png"
    side.write_bytes(_png(32, 32, (90, 10, 20)))
    _add_ref(source, "character", "identity", side)
    source["scenes"][0]["b"]["reference_roles"] = [
        "anchor",
        "character_reference",
        "character",
    ]
    source["generation_limit"] = 1
    result, refs, _store = _run_plan(tmp_path, source, "shot-01/B")
    assert result["status"] == "STOPPED_AFTER"
    char_name = Path(refs["character_reference"]["output"]["path"]).name
    custom_name = Path(refs["character"]["output"]["path"]).name
    assert char_name == "character.png"
    assert custom_name != char_name
    assert custom_name.casefold() != char_name.casefold()
    assert refs["character_reference"]["output"]["sha256"] != refs["character"]["output"]["sha256"]


def test_materialized_environment_names_avoid_legacy_collision_across_frames(tmp_path: Path) -> None:
    source = _base_source()
    env = tmp_path / "custom-env.png"
    env.write_bytes(_png(32, 32, (7, 9, 11)))
    legacy_env = tmp_path / "legacy-env.png"
    legacy_env.write_bytes(_png(32, 32, (11, 9, 7)))
    source["background_reference"] = str(legacy_env)
    _add_ref(source, "environment", "environment", env)
    source["scenes"][0]["b"]["reference_roles"] = ["anchor", "background_reference"]
    source["scenes"][1]["a"]["reference_roles"] = ["environment"]
    source["generation_limit"] = 2
    result, refs, _store = _run_plan(tmp_path, source, "shot-02/A")
    assert result["status"] == "STOPPED_AFTER"
    legacy = Path(refs["background_reference"]["output"]["path"]).name
    custom = Path(refs["environment"]["output"]["path"]).name
    assert legacy == "environment.png"
    assert custom.casefold() != legacy.casefold()
    assert refs["background_reference"]["output"]["sha256"] != refs["environment"]["output"]["sha256"]


def test_materialized_names_are_casefold_safe_for_custom_ids(tmp_path: Path) -> None:
    from scripts.kf1.runner import allocate_ref_names

    names = allocate_ref_names(["character_reference", "character", "Character"])
    assert names["character_reference"] == "character.png"
    assert len({value.casefold() for value in names.values()}) == len(names)
    assert names["character"] != names["Character"]


def test_upload_receipts_match_distinct_normalized_sources(tmp_path: Path) -> None:
    source = _base_source()
    side = tmp_path / "side.png"
    side.write_bytes(_png(32, 32, (90, 10, 20)))
    _add_ref(source, "character", "identity", side)
    source["scenes"][0]["b"]["reference_roles"] = ["anchor", "character_reference", "character"]
    source["generation_limit"] = 1
    result, refs, store = _run_plan(tmp_path, source, "shot-01/B")
    assert result["status"] == "STOPPED_AFTER"
    binding = json.loads(
        (store.version_dir("shot-01/B", 1) / "binding.json").read_text(encoding="utf-8")
    )
    uploads = binding["uploads"]
    assert len({item["source_sha256"] for item in uploads}) == 3
    assert uploads[1]["source_sha256"] == refs["character_reference"]["output"]["sha256"]
    assert uploads[2]["source_sha256"] == refs["character"]["output"]["sha256"]
    assert binding["bound_workflow"]["2"]["inputs"]["image"] == uploads[1]["name"]
    assert binding["bound_workflow"]["3"]["inputs"]["image"] == uploads[2]["name"]


def test_v1_legacy_background_role_uses_environment_policy(tmp_path: Path) -> None:
    source = json.loads(
        (Path(__file__).resolve().parents[1] / "config/keyframes/two-scenes.example.json").read_text(
            encoding="utf-8"
        )
    )
    legacy_env = tmp_path / "v1-background.png"
    legacy_env.write_bytes(_png(32, 32, (12, 13, 14)))
    source["background_reference"] = str(legacy_env)
    source["output_policy"]["character_policy"] = "contain"
    source["output_policy"]["environment_policy"] = "crop"
    path = tmp_path / "v1-policy.json"
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    plan = resolve_plan(path)
    store = PlanStore(tmp_path / "v1-state", plan.plan_sha256, plan.path)
    refs = Runner(plan, store, None)._prepare_refs()
    assert refs["background_reference"]["policy"] == "crop"
    assert refs["character_reference"]["policy"] == "contain"


def test_actual_runner_casefold_custom_ids_do_not_overwrite(tmp_path: Path) -> None:
    source = _base_source()
    source.pop("character_reference", None)
    upper = tmp_path / "upper.png"
    lower = tmp_path / "lower.png"
    upper.write_bytes(_png(32, 32, (20, 1, 2)))
    lower.write_bytes(_png(32, 32, (2, 1, 20)))
    _add_ref(source, "CaseA", "identity", upper)
    _add_ref(source, "casea", "identity", lower)
    source["scenes"][0]["b"]["reference_roles"] = ["anchor", "CaseA", "casea"]
    source["scenes"][1]["a"]["reference_roles"] = ["CaseA"]
    source["scenes"][1]["b"]["reference_roles"] = ["anchor", "CaseA"]
    source["generation_limit"] = 1
    result, refs, store = _run_plan(tmp_path, source, "shot-01/B")
    assert result["status"] == "STOPPED_AFTER"
    names = [Path(refs[key]["output"]["path"]).name for key in ("CaseA", "casea")]
    assert names[0].casefold() != names[1].casefold()
    assert refs["CaseA"]["output"]["sha256"] != refs["casea"]["output"]["sha256"]
    binding = json.loads((store.version_dir("shot-01/B", 1) / "binding.json").read_text(encoding="utf-8"))
    assert len({item["source_sha256"] for item in binding["uploads"]}) == 3


def test_allocator_literal_prefix_collision_remains_unique(tmp_path: Path) -> None:
    from hashlib import sha256
    from scripts.kf1.runner import allocate_ref_names

    hashed = "character-" + sha256(b"character").hexdigest()[:12]
    names = allocate_ref_names(["character_reference", "character", hashed])
    assert names["character_reference"] == "character.png"
    assert len({value.casefold() for value in names.values()}) == len(names)
    assert names[hashed] != names["character"]


def test_windows_reserved_ref_ids_get_safe_unique_names(tmp_path: Path) -> None:
    from pathlib import Path as _Path
    from scripts.kf1.runner import allocate_ref_names

    names = allocate_ref_names(["character_reference", "CON", "NUL", "COM1", "LPT1"])
    assert names["character_reference"] == "character.png"
    for ref_id in ("CON", "NUL", "COM1", "LPT1"):
        stem = _Path(names[ref_id]).stem.upper()
        assert stem not in {"CON", "NUL", "COM1", "LPT1"}
    assert len({value.casefold() for value in names.values()}) == len(names)


def test_actual_runner_reserved_custom_ids_produce_safe_paths(tmp_path: Path) -> None:
    source = _base_source()
    source.pop("character_reference", None)
    con = tmp_path / "con.png"
    nul = tmp_path / "nul.png"
    con.write_bytes(_png(32, 32, (31, 1, 2)))
    nul.write_bytes(_png(32, 32, (2, 1, 31)))
    _add_ref(source, "CON", "identity", con)
    _add_ref(source, "NUL", "identity", nul)
    source["scenes"][0]["b"]["reference_roles"] = ["anchor", "CON", "NUL"]
    source["scenes"][1]["a"]["reference_roles"] = ["CON"]
    source["scenes"][1]["b"]["reference_roles"] = ["anchor", "CON"]
    source["generation_limit"] = 1
    result, refs, store = _run_plan(tmp_path, source, "shot-01/B")
    assert result["status"] == "STOPPED_AFTER"
    for ref_id in ("CON", "NUL"):
        path = Path(refs[ref_id]["output"]["path"])
        assert path.is_file()
        assert path.stem.upper() not in {"CON", "NUL"}
    binding = json.loads((store.version_dir("shot-01/B", 1) / "binding.json").read_text(encoding="utf-8"))
    assert len({item["source_sha256"] for item in binding["uploads"]}) == 3
