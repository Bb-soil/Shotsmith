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
        f"{repo}/tests/posix/i1_fd_cleanup_harness.py",
        repo,
    ]


def test_fd_holding_descendants_are_killed_within_budget() -> None:
    runner = _wsl_runner()
    if runner is None:
        pytest.skip("no WSL/POSIX Python available")
    completed = subprocess.run(runner, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["env_partial"] is True
    assert payload["env_budget_exhausted"] is True
    assert payload["env_descendant_gone"] is True
    assert payload["env_worker_alive"] is True
    assert payload["env_elapsed"] < 3.0
    assert payload["env_nvidia_exit"] == 124
    assert payload["env_nvidia_timed_out"] is True
    assert payload["observer_partial"] is True
    assert payload["observer_budget_exhausted"] is True
    assert payload["observer_attempt_state"] == "INCONCLUSIVE"
    assert payload["observer_attempt_reason"] == "observer_child_timeout"
    assert payload["observer_descendant_gone"] is True
    assert payload["observer_worker_alive"] is True
    assert payload["observer_elapsed"] < 3.0
