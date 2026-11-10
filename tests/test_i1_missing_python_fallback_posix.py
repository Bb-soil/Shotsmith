from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _wsl_runner():
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return None
    repo = str(ROOT).replace("\\", "/")
    if len(repo) >= 3 and repo[1] == ":":
        repo = "/mnt/" + repo[0].lower() + repo[2:]
    return [
        wsl,
        "-e",
        "python3",
        f"{repo}/tests/posix/i1_missing_python_fallback_harness.py",
        repo,
    ]


def test_actual_sh_wrapper_missing_python_is_typed_not_transport() -> None:
    runner = _wsl_runner()
    if runner is None:
        pytest.skip("no WSL/POSIX Python available")
    completed = subprocess.run(runner, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["python3"] is False
    assert payload["tool_presence"] == {"sh": True}
    assert payload["commands"] == []
    assert payload["attempt_state"] == "INCONCLUSIVE"
    assert payload["partial"] is False
    assert payload["budget_exhausted"] is False
