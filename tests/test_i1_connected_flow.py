from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "posix" / "i1_connected_harness.py"


def _to_wsl(path: Path) -> str:
    drive, rest = os.path.splitdrive(str(path))
    if not drive:
        return str(path).replace("\\", "/")
    return "/mnt/" + drive.rstrip(":").lower() + "/" + rest.replace("\\", "/").lstrip("/")


def _runner() -> tuple[list[str], list[str]] | None:
    if os.name != "nt":
        return [sys.executable], [str(HARNESS), "--repo", str(ROOT)]
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return None
    probe = subprocess.run(
        [wsl, "-e", "python3", "--version"], capture_output=True, text=True, timeout=30
    )
    if probe.returncode != 0:
        return None
    return [wsl, "-e", "python3"], [_to_wsl(HARNESS), "--repo", _to_wsl(ROOT)]


def _run(*extra: str) -> dict:
    runner = _runner()
    if runner is None:
        pytest.skip("no local POSIX Python (WSL or native) available")
    prefix, args = runner
    result = subprocess.run(
        [*prefix, *args, *extra], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_connected_live_resume_and_proof_recovery_same_pid() -> None:
    payload = _run()
    assert payload["status"] == "OFFLINE_VERIFIED"
    assert payload["production_flow"] is True
    assert payload["live_resume_same_pid"] is True
    assert payload["proof_recovered_same_pid"] is True
    assert payload["disconnect_open_same_pid"] is True
    assert payload["comfy_killed_and_resumed"] is True
    assert payload["old_owner_retained"] is True
    assert payload["repeat_disconnect_resume_cycles"] == 5
    assert payload["repeat_archive_count"] == 5


def test_connected_fast_worker_completes_before_capture_and_finalizes() -> None:
    payload = _run("--fast-download")
    assert payload["status"] == "OFFLINE_VERIFIED"
    assert payload["fast_download_completed"] is True
    assert payload["production_flow"] is True
    assert payload["comfy_killed_and_resumed"] is True
