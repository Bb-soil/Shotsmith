from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.kf2.errors import ProjectConflict, ProjectError, ProjectExists, ProjectTamper
from scripts.kf2.store import ProjectStore, authoring_digest


def _store(tmp_path: Path) -> ProjectStore:
    return ProjectStore(tmp_path / "projects")


def test_create_is_exclusive_and_case_collision_safe(tmp_path: Path) -> None:
    store = _store(tmp_path)
    project, path = store.create("p1", title="Проект", orientation="portrait")
    assert project["schema"] == "anime-studio/kf2-project/v1"
    assert path.is_dir() and (path / "project.json").is_file()
    with pytest.raises(ProjectExists):
        store.create("p1")
    with pytest.raises(ProjectError):
        store.create("P1")
    with pytest.raises(ProjectError):
        store.create("../escape")
    with pytest.raises(ProjectError):
        store.create("con")
    with pytest.raises(ProjectError):
        store.create("p1.")


def test_save_is_optimistic_and_archives_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    project, path = store.create("p1", title="Первый", orientation="portrait")
    stale = store.load("p1")
    project["title"] = "Второй"
    saved = store.save("p1", project, expected_revision=project["revision"])
    assert saved["revision"] == 2
    assert (path / "history" / "project-v0001.json").is_file()
    stale["title"] = "Конфликт"
    with pytest.raises(ProjectConflict):
        store.save("p1", stale, expected_revision=stale["revision"])


def test_manual_tamper_is_rejected_without_overwrite(tmp_path: Path) -> None:
    store = _store(tmp_path)
    project, path = store.create("p1", title="Оригинал", orientation="portrait")
    raw = json.loads((path / "project.json").read_text(encoding="utf-8"))
    raw["title"] = "Подмена"
    (path / "project.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    before = (path / "project.json").read_bytes()
    with pytest.raises(ProjectTamper):
        store.load("p1")
    with pytest.raises(ProjectTamper):
        store.save("p1", project, expected_revision=project["revision"])
    assert (path / "project.json").read_bytes() == before


def test_project_json_symlink_and_directory_symlink_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    project, path = store.create("p1", title="x", orientation="portrait")
    other = tmp_path / "projects" / "linked"
    try:
        os.symlink(path, other, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink privilege unavailable")
    with pytest.raises(ProjectError):
        store.load("linked")
    linked_json = other / "project.json"
    try:
        os.unlink(linked_json)
    except OSError:
        pass


def test_authoring_digest_ignores_metadata_and_changes_on_content(tmp_path: Path) -> None:
    store = _store(tmp_path)
    project, _path = store.create("p1", title="A", orientation="portrait")
    digest = authoring_digest(project)
    project["revision"] = 99
    project["updated_utc"] = "later"
    project["status"] = "PREPARED"
    project["prepared"] = {"plan_sha256": "x" * 64}
    assert authoring_digest(project) == digest
    project["title"] = "B"
    assert authoring_digest(project) != digest
