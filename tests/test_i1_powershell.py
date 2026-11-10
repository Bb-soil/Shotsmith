from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PS = shutil.which("powershell")
PWSH = shutil.which("pwsh")


def _parse_script(executable: str) -> str:
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


def test_powershell5_parser_actual() -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    assert _parse_script(PS) == "PARSE_OK"


def test_powershell7_parser_not_run_if_absent() -> None:
    if not PWSH:
        pytest.skip("PowerShell 7 is unavailable on this machine (NOT_RUN, no install)")
    assert _parse_script(PWSH) == "PARSE_OK"


def test_wrapper_executes_json_without_dialogue_and_preserves_args(tmp_path: Path) -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    result = subprocess.run(
        [
            PS,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "avs.ps1"),
            "worker",
            "status",
            "--state-dir",
            str(tmp_path),
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode in {2, 4}
    payload = json.loads(result.stdout.strip())
    assert payload["status"] == "ERROR"


def test_wrapper_rejects_injection_name_without_creating_file(tmp_path: Path) -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    marker = tmp_path / "pwned"
    result = subprocess.run(
        [
            PS,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "avs.ps1"),
            "worker",
            "status",
            "--state-dir",
            str(tmp_path),
            "--name",
            "a b; touch " + str(marker),
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode in {2, 4}
    assert not marker.exists()


def test_wrapper_passes_argv_without_real_python_calls(tmp_path: Path) -> None:
    if not PS:
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    fake = tmp_path / "fake-python.cmd"
    fake.write_text(
        "@echo off\r\n"
        "echo ARGS:%*\r\n"
        "exit /b 0\r\n",
        encoding="utf-8",
    )
    env = dict(__import__("os").environ)
    env["AVS_PYTHON"] = str(fake)
    result = subprocess.run(
        [
            PS,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "avs.ps1"),
            "worker",
            "status",
            "--state-dir",
            "C:\\state",
            "--name",
            "srv",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0
    assert "ARGS:-m scripts.i1 status --state-dir C:\\state --name srv --json" in result.stdout


def test_runbook_has_no_backslash_line_continuation() -> None:
    text = (ROOT / "docs" / "I1-runbook.md").read_text(encoding="utf-8")
    assert "\\\n" not in text
