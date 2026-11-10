from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PS = shutil.which("powershell")
PWSH = shutil.which("pwsh")


def _parse(executable: str) -> str:
    script = (
        "& { $tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "(Resolve-Path 'avs.ps1'), [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors | ForEach-Object { $_.Message }; exit 1 } "
        "else { Write-Output 'PARSE_OK' } }"
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-Command", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_ps5_parses_project_dispatch_actual() -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 unavailable")
    assert _parse(PS) == "PARSE_OK"


def test_ps7_parses_if_available() -> None:
    if not PWSH:
        pytest.skip("PowerShell 7 unavailable (NOT_RUN)")
    assert _parse(PWSH) == "PARSE_OK"


def test_project_dispatch_uses_kf2_and_preserves_argv(tmp_path: Path) -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 unavailable")
    fake = tmp_path / "fake-python.cmd"
    fake.write_text(
        "@echo off\r\necho ARGS:%*\r\nexit /b 0\r\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["AVS_PYTHON"] = str(fake)
    result = subprocess.run(
        [
            PS,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "avs.ps1"),
            "project",
            "create",
            "--id",
            "p1",
            "--title",
            "ProjectOne",
            "--no-interactive",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0
    assert "-m scripts.kf2 create --id p1 --title ProjectOne --no-interactive --json" in result.stdout


def test_ps5_project_create_actual(tmp_path: Path) -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 unavailable")
    projects = tmp_path / "projects with spaces"
    result = subprocess.run(
        [
            PS,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "avs.ps1"),
            "project",
            "create",
            "--id",
            "ps1",
            "--title",
            "PS проект",
            "--orientation",
            "portrait",
            "--no-interactive",
            "--projects-dir",
            str(projects),
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (projects / "ps1" / "project.json").is_file()


def test_ps5_actual_wizard_reads_utf8_stdin(tmp_path: Path) -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 unavailable")
    projects = tmp_path / "projects"
    result = subprocess.run(
        [
            PS,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "avs.ps1"),
            "project",
            "create",
            "--projects-dir",
            str(projects),
        ],
        cwd=ROOT,
        input="p1\nPS проект\nportrait\n6\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    project = json.loads((projects / "p1" / "project.json").read_text(encoding="utf-8"))
    assert project["title"] == "PS проект"
    assert project["orientation"] == "portrait"


def test_ps5_actual_non_json_preview_and_prepare_have_russian_output(tmp_path: Path) -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 unavailable")
    from tests.test_kf2_review_fixes2 import _valid_project

    _valid_project(tmp_path)
    projects = tmp_path / "projects"
    preview = subprocess.run(
        [PS, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "avs.ps1"),
         "project", "preview", "p1", "--projects-dir", str(projects)],
        cwd=ROOT, capture_output=True, timeout=60,
    )
    assert preview.returncode == 0, preview.stderr
    text = preview.stdout.decode("utf-8")
    assert "Проект:" in text and "Библиотека:" in text and "Сцены:" in text

    prepare = subprocess.run(
        [PS, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "avs.ps1"),
         "project", "prepare", "p1", "--projects-dir", str(projects)],
        cwd=ROOT, capture_output=True, timeout=60,
    )
    assert prepare.returncode == 0, prepare.stderr
    text = prepare.stdout.decode("utf-8")
    assert "Подготовка:" in text and "Plan SHA256:" in text and "Dry-run: PLAN_ONLY" in text
    assert "--execute" not in text


def test_ps5_public_preview_recovers_journal_before_read(tmp_path: Path) -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 unavailable")
    from types import SimpleNamespace
    from tests.test_kf2_review_fixes2 import _valid_project
    from scripts.kf2 import cli
    from scripts.kf1.plan import resolve_plan

    store, project, project_dir = _valid_project(tmp_path)
    code, _ = cli._cmd_prepare(
        SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state"), json=True)
    )
    assert code == 0
    old_project = (project_dir / "project.json").read_bytes()
    old_plan = (project_dir / "plan.json").read_bytes()
    lock = store.lock("p1")
    try:
        store._write_prepare_journal_locked(
            "p1", old_project_bytes=old_project, old_plan_bytes=old_plan,
            old_plan_canonical_sha256=resolve_plan(project_dir / "plan.json").plan_sha256,
        )
    finally:
        lock.release()
    result = subprocess.run(
        [PS, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "avs.ps1"),
         "project", "preview", "p1", "--projects-dir", str(tmp_path / "projects"), "--json"],
        cwd=ROOT, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert not (project_dir / "prepare.journal.json").exists()
    assert (project_dir / "project.json").read_bytes() == old_project
    assert (project_dir / "plan.json").read_bytes() == old_plan
