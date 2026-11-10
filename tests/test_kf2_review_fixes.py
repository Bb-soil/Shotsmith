from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kf2 import assets as assets_mod
from scripts.kf2 import cli
from scripts.kf2.assets import decode_image_bytes, import_image
from scripts.kf2.errors import AssetError, ProjectError
from scripts.kf2.paths import resolve_local_source
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import _png


def _project(tmp_path: Path, *, title: str = "Проект"):
    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("p1", title=title, orientation="portrait")
    return store, project, project_dir


def _image(tmp_path: Path, name: str, size=(64, 64)) -> Path:
    path = tmp_path / name
    path.write_bytes(_png(*size))
    return path


def _feed(values):
    iterator = iter(values)

    def fake_input(prompt: str) -> str:
        return next(iterator)

    return fake_input


def test_imported_incomplete_scene_can_assign_a1_and_fix_b_approval(tmp_path: Path) -> None:
    store, project, project_dir = _project(tmp_path)
    first = _image(tmp_path, "start frame.png", (704, 1280))
    cli._cmd_add_image(
        SimpleNamespace(
            **vars(SimpleNamespace(projects_dir=str(tmp_path / "projects"), json=True, id="p1")),
            role="provided_keyframe",
            path=str(first),
            subject=None,
            subject_label=None,
            view=None,
            provenance=None,
        )
    )
    # Import a numbered proposal first: it has no A1 asset and no instructions.
    text = tmp_path / "story.txt"
    text.write_bytes("\ufeff1. Машина на дороге.\n".encode("utf-8"))
    cli._cmd_import_text(
        SimpleNamespace(
            projects_dir=str(tmp_path / "projects"), json=True, id="p1",
            file=str(text), numbered=True, confirm=True,
        )
    )
    project = store.load("p1")
    assert project["scenes"][0]["initial"]["asset_id"] is None
    # Restart: choose the existing A1 asset and fix B1 approval through the UI.
    asset_id = next(item["id"] for item in project["library"] if item["role"] == "provided_keyframe")
    answers = _feed(
        [
            "3",  # edit scene
            "shot-01",
            "8",  # A state
            asset_id,
            "",
            "9",  # B state
            "continue",
            "B instruction",
            "---",
            "",
            "",
            "y",  # unsupported approval must be forced false, not stuck
            "",
            "0",
            "6",
        ]
    )
    project = cli._menu(store, project, project_dir, input_fn=answers, print_fn=lambda _x: None)
    scene = project["scenes"][0]
    assert scene["initial"]["asset_id"] == asset_id
    assert scene["final"]["require_anchor_approval"] is False
    code, payload = cli._cmd_prepare(
        SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state"), json=True)
    )
    assert code == 0
    assert payload["prepared"]["approval_required_frames"] == []


def test_partial_scene_persists_after_eof_and_keyboard_interrupt(tmp_path: Path) -> None:
    store, project, project_dir = _project(tmp_path)
    menu_choices = iter(["2", "6"])

    def eof_after_description(prompt: str) -> str:
        if prompt.startswith("... "):
            raise EOFError
        if prompt == "Выбор: ":
            return next(menu_choices)
        values = {
            "ID сцены [shot-01]: ": "shot-01",
            "Поле сцены: ": "1",
            "Описание сцены (строка 1): ": "Первая строка",
        }
        return values.get(prompt, "")

    cli._menu(store, project, project_dir, input_fn=eof_after_description, print_fn=lambda _x: None)
    reloaded = ProjectStore(tmp_path / "projects").load("p1")
    assert reloaded["scenes"][0]["description"] == "Первая строка"

    menu_choices = iter(["3", "6"])

    def interrupt_on_action(prompt: str) -> str:
        if prompt.startswith("Действие (строка 1): "):
            raise KeyboardInterrupt
        if prompt == "Выбор: ":
            return next(menu_choices)
        values = {
            "ID сцены для редактирования: ": "shot-01",
            "Поле сцены: ": "2",
        }
        return values.get(prompt, "")

    project = ProjectStore(tmp_path / "projects").load("p1")
    cli._menu(store, project, project_dir, input_fn=interrupt_on_action, print_fn=lambda _x: None)
    assert ProjectStore(tmp_path / "projects").load("p1")["scenes"][0]["description"] == "Первая строка"


def test_edit_settings_clear_fields_and_orientation_are_atomic(tmp_path: Path) -> None:
    store, project, project_dir = _project(tmp_path)
    project["scenes"] = [cli._blank_scene(0, "shot-01")]
    project["scenes"][0]["description"] = "Старое"
    project = cli._save(store, project)
    answers = _feed(
        [
            "9",  # project settings
            "1", "Новое название",
            "2", "landscape",
            "3", "Стиль", "---",
            "4", "5",
            "0",
            "3", "shot-01", "1", "-", "0",  # clear description
            "6",
        ]
    )
    project = cli._menu(store, project, project_dir, input_fn=answers, print_fn=lambda _x: None)
    assert project["title"] == "Новое название"
    assert project["orientation"] == "landscape"
    assert project["candidate_geometry"] == [1280, 704]
    assert project["video_geometry"] == [816, 448]
    assert project["generation_limit"] == 5
    assert project["scenes"][0]["description"] == ""


def test_multiline_console_keeps_hash_and_literal_terminator_and_saves_per_line(tmp_path: Path) -> None:
    seen: list[str] = []
    value = cli._ask_multiline(
        "Текст",
        "",
        input_fn=_feed(["# literal", "\\---", "конец", "---"]),
        on_change=seen.append,
    )
    assert value == "# literal\n---\nконец"
    assert seen == ["# literal", "# literal\n---", "# literal\n---\nконец"]


def test_repeated_import_is_idempotent_and_distinct_association_is_separate(tmp_path: Path) -> None:
    _store, project, project_dir = _project(tmp_path)
    path = _image(tmp_path, "same.png")
    first = cli._add_image_to_project(project, project_dir, role="object", path=str(path), subject_id="car", view="front")
    cli._save(_store, project)
    project = _store.load("p1")
    second = cli._add_image_to_project(project, project_dir, role="object", path=str(path), subject_id="car", view="front")
    cli._save(_store, project)
    project = _store.load("p1")
    assert first["id"] == second["id"]
    assert len(project["library"]) == 1
    third = cli._add_image_to_project(project, project_dir, role="object", path=str(path), subject_id="car", view="side")
    cli._save(_store, project)
    assert third["id"] != first["id"]
    assert len(_store.load("p1")["library"]) == 2


def test_prepare_is_transactional_when_dryrun_fails(tmp_path: Path, monkeypatch) -> None:
    store, project, project_dir = _project(tmp_path)
    first = _image(tmp_path, "first.png", (704, 1280))
    identity = _image(tmp_path, "id.png")
    cli._add_image_to_project(project, project_dir, role="provided_keyframe", path=str(first))
    cli._add_image_to_project(project, project_dir, role="object", path=str(identity), subject_id="car", subject_label="car")
    project["scenes"] = [cli._blank_scene(0, "shot-01")]
    project["scenes"][0]["description"] = "Car"
    project["scenes"][0]["initial"]["asset_id"] = next(
        item["id"] for item in project["library"] if item["role"] == "provided_keyframe"
    )
    project["scenes"][0]["final"]["instruction"] = "Continue"
    project["scenes"][0]["final"]["references"] = [
        {"asset_id": next(item["id"] for item in project["library"] if item["role"] == "object"), "role": "identity"}
    ]
    project["scenes"][0]["final"]["require_anchor_approval"] = False
    project = cli._save(store, project)
    # First successful prepare gives an old pair.
    code, _ = cli._cmd_prepare(
        SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state1"), json=True)
    )
    assert code == 0
    old_project = (project_dir / "project.json").read_bytes()
    old_plan = (project_dir / "plan.json").read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("injected dryrun failure")

    monkeypatch.setattr("scripts.kf2.cli._dryrun", fail)
    with pytest.raises(RuntimeError):
        cli._cmd_prepare(
            SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state2"), json=True)
        )
    assert (project_dir / "project.json").read_bytes() == old_project
    assert (project_dir / "plan.json").read_bytes() == old_plan
    assert not (project_dir / "prepare.journal.json").exists()


def test_path_gate_rejects_ancestor_junction_and_provenance_link(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction test")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "junction"
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(real)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if made.returncode != 0:
        pytest.skip("junction privilege unavailable")
    with pytest.raises(ProjectError):
        ProjectStore(link / "projects")
    outside = tmp_path / "outside"
    outside.mkdir()
    provenance = outside / "p.json"
    provenance.write_text("{}", encoding="utf-8")
    link_file = tmp_path / "p-link.json"
    try:
        os.symlink(provenance, link_file)
    except OSError:
        pass
    else:
        with pytest.raises(AssetError):
            resolve_local_source(str(link_file), label="provenance", allowed_suffixes={".json"})


def test_image_header_cap_rejects_before_decoder(monkeypatch) -> None:
    import struct

    header = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", 10000, 6000)
        + bytes([8, 2, 0, 0, 0])
    )
    called = {"decode": False}

    def fail_decode(*args, **kwargs):
        called["decode"] = True
        raise AssertionError("decoder must not be called")

    monkeypatch.setattr(assets_mod, "validate_png_bytes", fail_decode)
    with pytest.raises(AssetError):
        decode_image_bytes(header, label="oversize")
    assert called["decode"] is False


def test_jpeg_oversize_header_does_not_load(monkeypatch) -> None:
    class FakeImage:
        format = "JPEG"
        is_animated = False
        n_frames = 1
        size = (10000, 6000)
        mode = "RGB"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getexif(self):
            return {}

        def load(self):
            raise AssertionError("load must not be called")

    class FakePIL:
        DecompressionBombWarning = Warning

        @staticmethod
        def open(*args, **kwargs):
            return FakeImage()

    monkeypatch.setattr(assets_mod, "Image", FakePIL)
    with pytest.raises(AssetError):
        decode_image_bytes(b"\xff\xd8not-a-real-header", label="jpeg")


def test_apply_proposal_sets_geometry_and_contextual_approval(tmp_path: Path) -> None:
    from scripts.kf2.text_import import apply_proposal, parse_text_document

    project = _project(tmp_path)[1]
    proposal = parse_text_document(
        "orientation: landscape\nscene: shot-01\ndescription: Текст\n".encode("utf-8")
    )
    project = apply_proposal(project, proposal)
    assert project["candidate_geometry"] == [1280, 704]
    assert project["video_geometry"] == [816, 448]
    assert project["scenes"][0]["final"]["require_anchor_approval"] is False
