from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "posix" / "i1_comfy_shim_harness.py"


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


def test_comfy_shim_uses_real_checkout_sys_path_and_birth_before_work() -> None:
    runner = _runner()
    if runner is None:
        pytest.skip("no local POSIX Python (WSL or native) available")
    prefix, args = runner
    result = subprocess.run([*prefix, *args], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "OFFLINE_VERIFIED"
    assert payload["checks"] == {
        "sys_path_0": True,
        "sibling_origin": True,
        "nested_origin": True,
        "main_origin": True,
        "birth_before_work": True,
    }
