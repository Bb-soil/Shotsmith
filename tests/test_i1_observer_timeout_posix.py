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
        f"{repo}/tests/posix/i1_observer_timeout_harness.py",
        repo,
    ]


def test_actual_observer_child_timeout_kills_only_owned_child() -> None:
    runner = _wsl_runner()
    if runner is None:
        pytest.skip("no WSL/POSIX Python available")
    completed = subprocess.run(runner, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["partial"] is True
    assert payload["budget_exhausted"] is True
    assert payload["attempt_state"] == "INCONCLUSIVE"
    assert payload["attempt_reason"] == "observer_child_timeout"
    assert payload["pid_file_exists"] is True
    assert payload["child_gone"] is True
