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
    return [wsl, "-e", "python3", f"{repo}/tests/posix/i1_observe_download_harness.py", "--repo", repo]


def test_actual_posix_observe_download_is_read_only() -> None:
    runner = _wsl_runner()
    if runner is None:
        pytest.skip("no WSL/POSIX Python available")
    completed = subprocess.run(runner, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["no_proof_state"] == "INCONCLUSIVE"
    assert payload["no_proof_tree_unchanged"] is True
    assert payload["no_proof_created_lock"] is False
    assert payload["with_proof_process"] == "DEAD"
    assert payload["missing_state"] == "MISSING"
    assert payload["symlink_state"] == "UNREADABLE"
    assert payload["permission_state"] == "UNREADABLE"
    assert payload["collector_before_attempt_state"] == "INCONCLUSIVE"
    assert payload["collector_after_proof_process"] == "DEAD"
    assert payload["collector_before_tree_unchanged"] is True
    assert payload["collector_after_proof_tree_unchanged"] is True
    assert payload["child_disappeared_state"] in {"UNREADABLE", "INCONCLUSIVE"}
    assert payload["proof_disappeared_state"] in {"UNREADABLE", "INCONCLUSIVE"}
    assert payload["owner_only_state"] == "UNREADABLE"
    assert payload["ack_only_state"] == "UNREADABLE"
    assert payload["both_absent_running_state"] == "UNREADABLE"
    assert payload["both_absent_complete_state"] == "OK"
    assert payload["both_absent_complete_process"] == "ABSENT"
    assert payload["both_absent_missing_state"] == "MISSING"
    assert payload["both_present_process"] == "DEAD"
    assert payload["fifo_status_state"] in {"UNREADABLE", "INCONCLUSIVE"}
    assert payload["fifo_status_elapsed"] < 2.0
    assert payload["fifo_proof_state"] in {"UNREADABLE", "INCONCLUSIVE"}
    assert payload["fifo_proof_elapsed"] < 2.0
    assert payload["live_state"] == "OK"
    assert payload["live_process"] == "LIVE"
    assert payload["collector_live_state"] == "OK"
    assert payload["collector_live_process"] == "LIVE"
    assert payload["live_child_gone"] is True
    assert payload["complete_worker_state"] == "OK"
    assert payload["complete_worker_process"] == "ABSENT"
