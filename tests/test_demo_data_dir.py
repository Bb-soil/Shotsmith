from __future__ import annotations

from pathlib import Path

import pytest

from anime_studio.demo import prepare_run_dir, run_demo
from tests.helpers import ROOT


def test_prepare_run_dir_is_unique_and_preserves_base(tmp_path) -> None:
    base = tmp_path / "demo-base"
    base.mkdir()
    keep = base / "keep.txt"
    keep.write_text("preserve me", encoding="utf-8")

    first = prepare_run_dir(base)
    second = prepare_run_dir(base)

    assert first != second
    assert first.parent == base
    assert second.parent == base
    assert first.is_dir() and second.is_dir()
    assert (first / "RUN_SENTINEL.json").is_file()
    assert (second / "RUN_SENTINEL.json").is_file()
    assert keep.read_text(encoding="utf-8") == "preserve me"


def test_prepare_run_dir_rejects_repo_and_drive_root() -> None:
    with pytest.raises(ValueError):
        prepare_run_dir(ROOT)
    with pytest.raises(ValueError):
        prepare_run_dir(Path(ROOT.anchor))


def test_prepare_run_dir_rejects_existing_file(tmp_path) -> None:
    path = tmp_path / "not-a-dir"
    path.write_text("file", encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_run_dir(path)


def test_run_demo_refuses_repo_root_without_mutation(tmp_path) -> None:
    before = sorted(path.name for path in ROOT.iterdir())
    evidence = tmp_path / "evidence.json"
    code = run_demo(evidence_path=evidence, data_dir=ROOT)
    assert code == 2
    assert not evidence.exists()
    assert sorted(path.name for path in ROOT.iterdir()) == before
