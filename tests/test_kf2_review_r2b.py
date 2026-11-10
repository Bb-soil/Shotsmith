from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kf2 import cli
from scripts.kf2.errors import ProjectError
from tests.test_kf2_review_r2 import _prepare, _valid


def _tampered(tmp_path: Path, first_frame: str):
    store, project, project_dir = _valid(tmp_path)
    assert _prepare(store, project, project_dir, tmp_path, name="first")[0] == 0
    plan_path = project_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["title"] = "OLDPLAN-MARKER"
    plan["first_frame"] = first_frame
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    before_project = (project_dir / "project.json").read_bytes()
    before_plan = plan_path.read_bytes()
    history = project_dir / "history" / "plans"
    before_history = sorted(path.name for path in history.glob("*")) if history.exists() else []
    return store, project, project_dir, plan_path, before_project, before_plan, before_history


def _spy_old(monkeypatch, calls: list[str]):
    from scripts.kf1.plan import resolve_plan as real

    def spy(path):
        text = Path(path).read_text(encoding="utf-8")
        if "OLDPLAN-MARKER" in text:
            calls.append(str(path))
            raise AssertionError("old plan resolver called before bounded path validation")
        return real(path)

    monkeypatch.setattr("scripts.kf1.plan.resolve_plan", spy)


def _prepare_args(tmp_path: Path):
    return SimpleNamespace(
        id="p1", projects_dir=str(tmp_path / "projects"),
        kf1_state_dir=str(tmp_path / "second-state"), json=True,
    )


def test_prepare_rejects_old_plan_oversized_path_before_resolver(tmp_path: Path, monkeypatch) -> None:
    store, project, project_dir = _valid(tmp_path)
    assert _prepare(store, project, project_dir, tmp_path, name="first")[0] == 0
    huge = project_dir / "assets" / "old-oversized.png"
    huge.parent.mkdir(parents=True, exist_ok=True)
    with huge.open("wb") as handle:
        handle.truncate(32 * 1024 * 1024 + 1)
    plan_path = project_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["title"] = "OLDPLAN-MARKER"
    plan["first_frame"] = str(huge)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    before_project = (project_dir / "project.json").read_bytes()
    before_plan = plan_path.read_bytes()
    history = project_dir / "history" / "plans"
    before_history = sorted(path.name for path in history.glob("*")) if history.exists() else []
    calls: list[str] = []
    _spy_old(monkeypatch, calls)
    opened: list[str] = []
    real_open = open

    def guarded_open(file, *args, **kwargs):
        if os.path.normcase(os.path.abspath(str(file))) == os.path.normcase(os.path.abspath(str(huge))):
            opened.append(str(file))
            raise AssertionError("oversized old-plan asset opened before byte cap")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    with pytest.raises(ProjectError, match="33554432|размер"):
        cli._cmd_prepare(_prepare_args(tmp_path))
    assert calls == []
    assert opened == []
    assert (project_dir / "project.json").read_bytes() == before_project
    assert plan_path.read_bytes() == before_plan
    assert not (project_dir / "prepare.journal.json").exists()
    assert (sorted(path.name for path in history.glob("*")) if history.exists() else []) == before_history


def test_prepare_rejects_old_plan_unc_path_before_resolver(tmp_path: Path, monkeypatch) -> None:
    store, project, project_dir, plan_path, before_project, before_plan, _ = _tampered(
        tmp_path, r"\\server\share\old.png"
    )
    calls: list[str] = []
    _spy_old(monkeypatch, calls)
    with pytest.raises(ProjectError):
        cli._cmd_prepare(_prepare_args(tmp_path))
    assert calls == []
    assert (project_dir / "project.json").read_bytes() == before_project
    assert plan_path.read_bytes() == before_plan
    assert not (project_dir / "prepare.journal.json").exists()


def test_prepare_rejects_old_plan_parent_junction_before_resolver(tmp_path: Path, monkeypatch) -> None:
    junction = tmp_path / "junction"
    child = junction / "old.png"
    store, project, project_dir, plan_path, before_project, before_plan, _ = _tampered(
        tmp_path, str(child)
    )
    original_lstat = os.lstat
    attempts: list[str] = []

    def fake_lstat(value):
        text = os.path.normcase(os.path.abspath(str(value)))
        attempts.append(text)
        if text == os.path.normcase(os.path.abspath(str(junction))):
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        if text.startswith(os.path.normcase(os.path.abspath(str(junction)) + os.sep)):
            raise AssertionError("descendant lstat through old plan junction")
        return original_lstat(value)

    monkeypatch.setattr("scripts.kf2.paths.os.lstat", fake_lstat)
    calls: list[str] = []
    _spy_old(monkeypatch, calls)
    with pytest.raises(ProjectError):
        cli._cmd_prepare(_prepare_args(tmp_path))
    assert calls == []
    assert (project_dir / "project.json").read_bytes() == before_project
    assert plan_path.read_bytes() == before_plan
    assert not (project_dir / "prepare.journal.json").exists()
    descendant = os.path.normcase(os.path.abspath(str(junction)) + os.sep)
    assert not any(item.startswith(descendant) for item in attempts)


def test_prepare_rejects_semantically_changed_old_plan_without_archive(tmp_path: Path) -> None:
    store, project, project_dir = _valid(tmp_path)
    assert _prepare(store, project, project_dir, tmp_path, name="first")[0] == 0
    plan_path = project_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["title"] = "manually changed old plan"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    before_project = (project_dir / "project.json").read_bytes()
    before_plan = plan_path.read_bytes()
    with pytest.raises(ProjectError):
        cli._cmd_prepare(_prepare_args(tmp_path))
    assert (project_dir / "project.json").read_bytes() == before_project
    assert plan_path.read_bytes() == before_plan
    assert not (project_dir / "prepare.journal.json").exists()
