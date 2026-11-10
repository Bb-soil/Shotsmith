from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kf2 import cli
from scripts.kf2.cli import _add_image_to_project, _save
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import _png


def _args(tmp_path: Path, **kwargs):
    values = {
        "projects_dir": str(tmp_path / "projects"),
        "json": True,
        "id": "p1",
        "title": "Проект",
        "orientation": "portrait",
        "from_text": None,
        "from_text_numbered": False,
        "no_interactive": True,
        "kf1_state_dir": str(tmp_path / "kf1-state"),
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _valid_project(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("p1", title="Проект", orientation="portrait")
    (tmp_path / "first.png").write_bytes(_png(704, 1280))
    (tmp_path / "object.png").write_bytes(_png(64, 64))
    first = _add_image_to_project(
        project, project_dir, role="provided_keyframe", path=str(tmp_path / "first.png")
    )
    identity = _add_image_to_project(
        project,
        project_dir,
        role="object",
        path=str(tmp_path / "object.png"),
        subject_id="car",
        subject_label="car",
        view="front",
    )
    project["scenes"] = [
        {
            "id": "shot-01",
            "description": "A car on a road.",
            "action": "The car approaches.",
            "camera": "Fixed camera.",
            "subjects_visible": ["car"],
            "subjects_absent": [],
            "shared_invariants": ["identity", "style"],
            "shared_allowed_changes": ["position", "apparent_scale"],
            "initial": {
                "kind": "first_frame",
                "asset_id": first["id"],
                "instruction": "",
                "references": [],
                "invariants": [],
                "allowed_changes": [],
                "use_previous_anchor": False,
                "require_anchor_approval": True,
            },
            "final": {
                "kind": "continue",
                "asset_id": None,
                "instruction": "The same car approaches the fixed camera.",
                "references": [{"asset_id": identity["id"], "role": "identity"}],
                "invariants": [],
                "allowed_changes": [],
                "require_anchor_approval": False,
            },
        }
    ]
    project = _save(store, project)
    return store, project, project_dir


def test_create_preview_list_and_prepare_offline(tmp_path: Path) -> None:
    code, payload = cli._cmd_create(_args(tmp_path))
    assert code == 0
    assert payload["status"] == "INCOMPLETE"
    code, listed = cli._cmd_list(_args(tmp_path))
    assert code == 0 and listed["projects"][0]["id"] == "p1"

    valid_root = tmp_path / "valid"
    _valid_project(valid_root)
    code, prepared = cli._cmd_prepare(_args(valid_root))
    assert code == 0
    assert prepared["status"] == "PREPARED"
    assert prepared["dryrun"]["status"] == "PLAN_ONLY"
    assert prepared["dryrun"]["post_plan"] == ["shot-01/B"]
    plan = valid_root / "projects" / "p1" / "plan.json"
    assert plan.is_file()
    assert not (valid_root / "kf1-state" / "limits.json").exists()
    assert (valid_root / "projects" / "p1" / "history").is_dir()


def test_prepare_never_calls_network_or_client(tmp_path: Path, monkeypatch) -> None:
    _valid_project(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("network/client must not be used")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr("scripts.kf1.cli._client", boom)
    code, payload = cli._cmd_prepare(_args(tmp_path))
    assert code == 0
    assert payload["status"] == "PREPARED"


def test_import_text_bom_proposal_then_confirm(tmp_path: Path) -> None:
    cli._cmd_create(_args(tmp_path))
    text = tmp_path / "story.txt"
    text.write_bytes(
        "\ufefftitle: Тест\r\nscene: shot-01\r\ndescription:\r\nРусский текст.\r\n---\r\n".encode("utf-8")
    )
    args = _args(tmp_path, confirm=False, file=str(text), numbered=False)
    code, payload = cli._cmd_import_text(args)
    assert code == 0
    assert payload["proposal"]["mode"] == "keyvalue"
    store = ProjectStore(tmp_path / "projects")
    project = store.load("p1")
    assert project["pending_proposal"]["needs_confirmation"] is True
    assert not project["scenes"]

    args.confirm = True
    code, payload = cli._cmd_import_text(args)
    assert code == 0
    project = ProjectStore(tmp_path / "projects").load("p1")
    assert project["scenes"][0]["description"] == "Русский текст."
    assert project["import"]["needs_confirmation"] is False


def test_create_interactive_wizard_can_exit_and_save(tmp_path: Path, monkeypatch) -> None:
    answers = iter(["p1", "Мой проект", "portrait", "6"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    code, payload = cli._cmd_create(_args(tmp_path, json=False, no_interactive=False))
    assert code == 0
    assert payload["id"] == "p1"
    assert (tmp_path / "projects" / "p1" / "project.json").is_file()


def test_resume_json_is_read_only(tmp_path: Path) -> None:
    cli._cmd_create(_args(tmp_path))
    before = (tmp_path / "projects" / "p1" / "project.json").read_bytes()
    code, payload = cli._cmd_resume(_args(tmp_path))
    assert code == 0 and payload["status"] == "INCOMPLETE"
    assert (tmp_path / "projects" / "p1" / "project.json").read_bytes() == before


def test_all_project_authoring_routes_are_offline(tmp_path: Path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("network must not be used by project authoring")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr("scripts.kf1.cli._client", boom)
    cli._cmd_create(_args(tmp_path))
    (tmp_path / "img.png").write_bytes(_png(8, 8))
    add_args = SimpleNamespace(
        **vars(_args(tmp_path)),
        role="provided_keyframe",
        path=str(tmp_path / "img.png"),
        subject=None,
        subject_label=None,
        view=None,
        provenance=None,
    )
    code, _ = cli._cmd_add_image(add_args)
    assert code == 0
    cli._cmd_preview(_args(tmp_path))
    cli._cmd_resume(_args(tmp_path))
    text = tmp_path / "story.txt"
    text.write_bytes("scene: shot-01\ndescription: текст\n".encode("utf-8"))
    cli._cmd_import_text(
        SimpleNamespace(**vars(_args(tmp_path)), file=str(text), numbered=False, confirm=False)
    )
