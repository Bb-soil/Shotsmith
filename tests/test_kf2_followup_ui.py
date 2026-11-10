from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from scripts.kf2 import cli
from scripts.kf2.assets import change_asset_role
from scripts.kf2.compile import _compose_state_instruction, validate_and_compile
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import _png


def _feed(values):
    iterator = iter(values)

    def fake_input(prompt: str) -> str:
        return next(iterator)

    return fake_input


def _project(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("p1", title="ui", orientation="portrait")
    first = tmp_path / "first.png"
    first.write_bytes(_png(704, 1280, (1, 2, 3)))
    front = tmp_path / "front.png"
    front.write_bytes(_png(64, 64, (10, 20, 30)))
    side = tmp_path / "side.png"
    side.write_bytes(_png(64, 64, (30, 20, 10)))
    provided = cli._add_image_to_project(project, project_dir, role="provided_keyframe", path=str(first))
    girl = cli._add_image_to_project(
        project, project_dir, role="character", path=str(front),
        subject_id="girl", subject_label="girl", view="front",
    )
    girl_side = cli._add_image_to_project(
        project, project_dir, role="character", path=str(side),
        subject_id="girl", subject_label="girl", view="side",
    )
    project["scenes"] = [cli._blank_scene(0, "shot-01")]
    project["scenes"][0]["description"] = "Explicit scene description."
    project["scenes"][0]["initial"]["asset_id"] = provided["id"]
    project = cli._save(store, project)
    return store, project, project_dir, provided, girl, girl_side


def test_incomplete_preview_has_specific_next_fix_and_no_prepare(tmp_path: Path) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    project["scenes"][0]["initial"]["asset_id"] = None
    project = cli._save(store, project)
    payload = cli._summary(project, project_dir)
    assert payload["status"] == "INCOMPLETE"
    assert payload["can_prepare"] is False
    assert payload["next_command"] == '.\\avs.ps1 project resume "p1"'
    assert payload["next_fix"]["code"] == "FIRST_FRAME"
    assert "A1" in payload["next_fix"]["text"]
    assert "меню 11" in payload["next_fix"]["menu_path"]
    assert "prepare" not in payload["next_command"]


def test_a1_shortcut_uses_numbered_provided_asset(tmp_path: Path) -> None:
    store, project, _project_dir, provided, _girl, _girl_side = _project(tmp_path)
    project["scenes"][0]["initial"]["asset_id"] = None
    project = cli._save(store, project)
    cli._assign_a1_source(
        store,
        project,
        input_fn=_feed(["1"]),
        print_fn=lambda _line: None,
    )
    reloaded = store.load("p1")
    assert reloaded["scenes"][0]["initial"]["asset_id"] == provided["id"]
    assert reloaded["scenes"][0]["initial"]["kind"] == "first_frame"


def test_reference_picker_accepts_two_identity_numbers(tmp_path: Path) -> None:
    _store, project, _project_dir, _provided, girl, girl_side = _project(tmp_path)
    refs = cli._choose_refs(
        project,
        [],
        frame_id="shot-01/B",
        anchor_active=True,
        input_fn=_feed(["2,3"]),
        print_fn=lambda _line: None,
    )
    assert refs == [
        {"asset_id": girl["id"], "role": "identity"},
        {"asset_id": girl_side["id"], "role": "identity"},
    ]


def test_state_editor_composes_deterministic_a_and_b_targets(tmp_path: Path) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    scene = project["scenes"][0]
    answers = _feed(
        [
            "Action text", "---",
            "Camera text", "---",
            "Shared text", "---",
            "n",
            "Start state text", "---",
            "End state text", "---",
            "n",
            "n",
        ]
    )
    cli._edit_frame_states(
        store,
        project,
        scene,
        input_fn=answers,
        print_fn=lambda _line: None,
    )
    reloaded = store.load("p1")
    scene = reloaded["scenes"][0]
    assert scene["initial"]["state_a"] == "Start state text"
    assert scene["final"]["state_b"] == "End state text"
    assert scene["initial"]["instruction_mode"] == "manual"  # A1 is first_frame
    assert scene["final"]["instruction_mode"] == "states"
    assert scene["action"] == "Action text"
    assert scene["camera"] == "Camera text"
    assert scene["shared_state"] == "Shared text"
    composed = _compose_state_instruction(scene, scene["final"], "B", {"girl": "girl"})
    assert "Target B state: End state text" in composed
    assert "Common and unchanged: Shared text" in composed
    assert "Action/change: Action text" in composed
    assert "Camera: Camera text" in composed
    a_composed = _compose_state_instruction(scene, scene["initial"], "A", {"girl": "girl"})
    assert "Target A state: Start state text" in a_composed
    assert "End state text" not in a_composed


def test_change_asset_role_preserves_id_and_remaps_reference_semantic(tmp_path: Path) -> None:
    store, project, project_dir, _provided, girl, _girl_side = _project(tmp_path)
    scene = project["scenes"][0]
    scene["final"]["references"] = [{"asset_id": girl["id"], "role": "identity"}]
    project = cli._save(store, project)
    before_sha = next(item["sha256"] for item in project["library"] if item["id"] == girl["id"])
    result = change_asset_role(project, girl["id"], role="environment", view=None)
    assert result["role"] == "environment"
    assert result["usage"]["references"] == 1
    changed = next(item for item in project["library"] if item["id"] == girl["id"])
    assert changed["id"] == girl["id"]
    assert changed["role"] == "environment"
    assert changed["sha256"] == before_sha
    assert changed["subject_id"] is None
    assert project["scenes"][0]["final"]["references"][0] == {
        "asset_id": girl["id"],
        "role": "environment",
    }
    project = cli._save(store, project)
    assert store.load("p1")["library"][1]["role"] == "environment"


def test_state_composer_requires_nonempty_states(tmp_path: Path) -> None:
    store, project, project_dir, _provided, girl, _girl_side = _project(tmp_path)
    scene = project["scenes"][0]
    scene["action"] = "Action."
    scene["camera"] = "Camera."
    scene["final"]["instruction_mode"] = "states"
    scene["final"]["state_b"] = ""
    scene["final"]["references"] = [{"asset_id": girl["id"], "role": "identity"}]
    project = cli._save(store, project)
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "MISSING_STATE" for item in result["blockers"])
    assert result["plan"] is None


def test_blocked_human_preview_does_not_offer_prepare(tmp_path: Path, capsys) -> None:
    _store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    project["scenes"][0]["initial"]["asset_id"] = None
    project = cli._save(_store, project)
    payload = cli._summary(project, project_dir)
    cli._print_summary(payload)
    text = capsys.readouterr().out
    assert "Рекомендуемое следующее действие" in text
    assert "Prepare не предлагается" in text
    assert "Следующее: .\\avs.ps1 project prepare" not in text


def test_state_mode_instruction_has_no_duplicate_legacy_labels(tmp_path: Path) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    scene = project["scenes"][0]
    scene["action"] = "Action text."
    scene["camera"] = "Camera text."
    scene["shared_state"] = "Shared text."
    scene["final"]["instruction_mode"] = "states"
    scene["final"]["state_b"] = "End state text."
    project = cli._save(store, project)
    result = validate_and_compile(project, project_dir)
    assert result["blockers"] == []
    instruction = result["plan"]["scenes"][0]["b"]["edit_instruction"]
    assert instruction.count("Action/change: Action text.") == 1
    assert instruction.count("Camera: Camera text.") == 1
    assert instruction.count("Action: Action text.") == 0
    assert instruction.count("Visible subjects:") == 0


def test_role_change_to_provided_keeps_a1_assignment_and_shows_blocker(tmp_path: Path) -> None:
    store, project, project_dir, provided, _girl, _girl_side = _project(tmp_path)
    project["scenes"][0]["initial"]["asset_id"] = provided["id"]
    project = cli._save(store, project)
    result = change_asset_role(project, provided["id"], role="style", view=None)
    assert result["usage"]["a1_source"] == 1
    assert project["scenes"][0]["initial"]["asset_id"] == provided["id"]
    project = cli._save(store, project)
    compiled = validate_and_compile(project, project_dir)
    assert any(item["code"] == "FIRST_FRAME_ROLE" for item in compiled["blockers"])
    payload = cli._summary(project, project_dir)
    assert payload["next_fix"]["code"] == "FIRST_FRAME_ROLE"


def test_role_change_from_provided_to_identity_requires_subject(tmp_path: Path) -> None:
    _store, project, _project_dir, provided, _girl, _girl_side = _project(tmp_path)
    try:
        change_asset_role(project, provided["id"], role="character", subject_id=None)
    except Exception as exc:
        assert "subject" in str(exc)
    else:
        raise AssertionError("character role without subject must fail")


def test_invalid_instruction_mode_is_fail_closed(tmp_path: Path) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    project["scenes"][0]["final"]["instruction_mode"] = "magic"
    project["scenes"][0]["final"]["instruction"] = "Explicit B."
    project = cli._save(store, project)
    result = validate_and_compile(project, project_dir)
    assert any(item["code"] == "INVALID_INSTRUCTION_MODE" for item in result["blockers"])
    assert result["plan"] is None


def test_edit_asset_ui_changes_role_with_confirmation(tmp_path: Path) -> None:
    store, project, _project_dir, provided, girl, _girl_side = _project(tmp_path)
    project["scenes"][0]["final"]["references"] = [{"asset_id": girl["id"], "role": "identity"}]
    project = cli._save(store, project)
    cli._edit_asset(
        store,
        project,
        input_fn=_feed(["2", "environment", "y", "-", ""]),
        print_fn=lambda _line: None,
    )
    reloaded = store.load("p1")
    changed = next(item for item in reloaded["library"] if item["id"] == girl["id"])
    assert changed["role"] == "environment"
    assert changed["subject_id"] is None
    assert reloaded["scenes"][0]["final"]["references"][0]["role"] == "environment"


def test_user_edited_composed_instruction_uses_exact_mode_without_reappend(tmp_path: Path) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    scene = project["scenes"][0]
    answers = _feed(
        [
            "Action.", "---",
            "Camera.", "---",
            "Shared.", "---",
            "n",
            "A1 note", "---",
            "B state", "---",
            "y",
            "Edited exact instruction.", "---",
        ]
    )
    cli._edit_frame_states(
        store,
        project,
        scene,
        input_fn=answers,
        print_fn=lambda _line: None,
    )
    reloaded = store.load("p1")
    scene = reloaded["scenes"][0]
    assert scene["final"]["instruction_mode"] == "exact"
    assert scene["final"]["instruction"] == "Edited exact instruction."
    result = validate_and_compile(reloaded, project_dir)
    assert result["blockers"] == []
    assert result["plan"]["scenes"][0]["b"]["edit_instruction"] == "Edited exact instruction."


def test_human_preview_has_target_and_menu_path(tmp_path: Path, capsys) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    project["scenes"][0]["final"]["instruction"] = ""
    project = cli._save(store, project)
    before = json.dumps(project, sort_keys=True, ensure_ascii=False)
    payload = cli._summary(project, project_dir)
    assert payload["next_fix"]["target"] == "shot-01/B"
    assert payload["next_fix"]["menu_path"] == "меню 3 → сцена shot-01 → 12"
    cli._print_summary(payload)
    text = capsys.readouterr().out
    assert "shot-01/B" in text
    assert "меню 3 → сцена shot-01 → 12" in text
    assert json.dumps(project, sort_keys=True, ensure_ascii=False) == before
    assert payload["can_prepare"] is False


def test_ready_with_missing_action_camera_has_guided_advisory(tmp_path: Path, capsys) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    project["scenes"][0]["final"]["instruction"] = "Explicit B instruction."
    project = cli._save(store, project)
    payload = cli._summary(project, project_dir)
    assert payload["status"] == "READY_TO_PREPARE"
    assert payload["can_prepare"] is True
    assert payload["next_fix"]["code"] in {"MISSING_ACTION", "MISSING_CAMERA"}
    assert payload["next_fix"]["menu_path"] == "меню 3 → сцена shot-01 → 12"
    cli._print_summary(payload)
    text = capsys.readouterr().out
    assert "Рекомендуемое улучшение" in text
    assert "shot-01" in text


def test_generation_limit_fix_is_explicit_menu_9_4() -> None:
    fixes = cli._next_fixes(
        {"scenes": []},
        [{"code": "GENERATION_LIMIT", "message": "limit 3 меньше generated 11"}],
    )
    assert fixes[0]["target"] == "project"
    assert "меню 9 → 4" in fixes[0]["menu_path"]
    assert "расход" in fixes[0]["text"]


def test_anchor_overflow_reports_selected_total_not_drop(tmp_path: Path) -> None:
    store, project, project_dir, _provided, girl, girl_side = _project(tmp_path)
    env = tmp_path / "env.png"
    env.write_bytes(_png(8, 8, (1, 2, 3)))
    env_asset = cli._add_image_to_project(project, project_dir, role="environment", path=str(env))
    project = cli._save(store, project)
    refs = [
        {"asset_id": girl["id"], "role": "identity"},
        {"asset_id": girl_side["id"], "role": "identity"},
        {"asset_id": env_asset["id"], "role": "environment"},
    ]
    issues = cli._reference_choice_issues(project, refs, anchor_active=True)
    assert any("итого references включая anchor: 4" in issue for issue in issues)
    assert len(refs) == 3


@pytest.mark.parametrize("target", ["a", "b"])
def test_confirmed_import_compiled_generated_target_and_other_frame_mode(
    tmp_path: Path, target: str
) -> None:
    store, project, project_dir, _provided, girl, _girl_side = _project(tmp_path)
    first = project["scenes"][0]
    first["action"] = "Action."
    first["camera"] = "Camera."
    first["final"]["instruction"] = "First B."
    if target == "a":
        second = cli._blank_scene(1, "shot-02")
        second["description"] = "Second scene."
        second["action"] = "Second action."
        second["camera"] = "Second camera."
        second["subjects_visible"] = ["girl"]
        second["initial"].update(
            {
                "kind": "new_shot",
                "instruction_mode": "states",
                "state_a": "Old A state",
                "references": [{"asset_id": girl["id"], "role": "identity"}],
            }
        )
        second["final"].update(
            {
                "kind": "continue",
                "instruction_mode": "states",
                "state_b": "B state",
            }
        )
        project["scenes"].append(second)
        scene_id = "shot-02"
        requested = "a_instruction: Imported A.\n"
        old_text = "Old A state"
    else:
        first["final"].update(
            {
                "instruction_mode": "states",
                "state_b": "Old B state",
            }
        )
        scene_id = "shot-01"
        requested = "b_instruction: Imported B.\n"
        old_text = "Old B state"
    project = cli._save(store, project)
    text = tmp_path / "import.txt"
    text.write_text(f"scene: {scene_id}\n{requested}", encoding="utf-8")
    code, _payload = cli._cmd_import_text(
        __import__("types").SimpleNamespace(
            projects_dir=str(tmp_path / "projects"),
            json=True,
            id="p1",
            file=str(text),
            numbered=False,
            confirm=True,
        )
    )
    assert code == 0
    reloaded = store.load("p1")
    result = validate_and_compile(reloaded, project_dir)
    assert result["blockers"] == []
    if target == "a":
        scene = reloaded["scenes"][1]
        assert scene["initial"]["instruction_mode"] == "manual"
        assert scene["initial"]["instruction"] == "Imported A."
        assert scene["final"]["instruction_mode"] == "states"
        instruction = result["plan"]["scenes"][1]["a"]["edit_instruction"]
        other_instruction = result["plan"]["scenes"][1]["b"]["edit_instruction"]
        assert "Target B state: B state" in other_instruction
    else:
        scene = reloaded["scenes"][0]
        assert scene["final"]["instruction_mode"] == "manual"
        assert scene["final"]["instruction"] == "Imported B."
        assert scene["initial"]["instruction_mode"] == "manual"
        instruction = result["plan"]["scenes"][0]["b"]["edit_instruction"]
        other_instruction = result["plan"]["scenes"][0]["a"]["source"]
        assert other_instruction == "first_frame"
    assert "Imported " + target.upper() + "." in instruction
    assert old_text not in instruction


def test_unconfirmed_import_does_not_change_compiled_output(tmp_path: Path) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    scene = project["scenes"][0]
    scene["action"] = "Action."
    scene["camera"] = "Camera."
    scene["final"]["instruction"] = "Existing B."
    project = cli._save(store, project)
    before = validate_and_compile(project, project_dir)
    text = tmp_path / "proposal.txt"
    text.write_text("scene: shot-01\na_instruction: Proposed only.\n", encoding="utf-8")
    code, _payload = cli._cmd_import_text(
        __import__("types").SimpleNamespace(
            projects_dir=str(tmp_path / "projects"),
            json=True,
            id="p1",
            file=str(text),
            numbered=False,
            confirm=False,
        )
    )
    assert code == 0
    reloaded = store.load("p1")
    after = validate_and_compile(reloaded, project_dir)
    assert before["plan"] == after["plan"]
    assert after["blockers"] == []
    assert reloaded["scenes"][0]["initial"]["instruction_mode"] == "manual"
    assert reloaded["scenes"][0]["final"]["instruction_mode"] == "manual"


def test_description_only_import_preserves_states_mode(tmp_path: Path) -> None:
    from scripts.kf2.text_import import apply_proposal, parse_text_document

    _store, project, _project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    scene = project["scenes"][0]
    scene["initial"]["instruction_mode"] = "states"
    scene["final"]["instruction_mode"] = "states"
    proposal = parse_text_document(b"scene: shot-01\ndescription: New description.\n")
    updated = apply_proposal(project, proposal)
    assert updated["scenes"][0]["initial"]["instruction_mode"] == "states"
    assert updated["scenes"][0]["final"]["instruction_mode"] == "states"


def test_provenance_next_fix_points_to_asset_provenance_menu(tmp_path: Path, capsys) -> None:
    store, project, project_dir, _provided, _girl, _girl_side = _project(tmp_path)
    later = tmp_path / "later-provided.png"
    later.write_bytes(_png(64, 64, (99, 1, 2)))
    provided = cli._add_image_to_project(
        project, project_dir, role="provided_keyframe", path=str(later)
    )
    scene = project["scenes"][0]
    scene["final"].update(
        {
            "kind": "provided_keyframe",
            "asset_id": provided["id"],
            "instruction": "",
            "references": [],
        }
    )
    project = cli._save(store, project)
    payload = cli._summary(project, project_dir)
    assert payload["next_fix"]["code"] == "PROVIDED_PROVENANCE_REQUIRED"
    assert payload["next_fix"]["target"] == "shot-01/B"
    assert "меню 10" in payload["next_fix"]["menu_path"]
    assert "Provenance JSON" in payload["next_fix"]["menu_path"]
    cli._print_summary(payload)
    text = capsys.readouterr().out
    assert "shot-01/B" in text
    assert "PROVIDED_PROVENANCE_REQUIRED" in text
    assert "меню 10" in text
