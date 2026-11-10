from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WSL = shutil.which("wsl")


def _wsl_path(path: Path) -> str:
    text = str(path.resolve())
    if len(text) >= 3 and text[1:3] == ":\\":
        return f"/mnt/{text[0].lower()}/" + text[3:].replace("\\", "/")
    return text.replace("\\", "/")


@pytest.mark.skipif(WSL is None, reason="WSL unavailable")
def test_actual_staged_kf1_worker_probe_executes_3_and_7_models() -> None:
    completed = subprocess.run(
        [
            "wsl",
            "-d",
            "Ubuntu",
            "--",
            "python3",
            _wsl_path(ROOT / "tests" / "posix" / "kf1_profile_harness.py"),
            "--repo",
            _wsl_path(ROOT),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + "\n" + completed.stdout[-4000:]
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "OFFLINE_VERIFIED"
    assert payload["k1"]["models"] == 3
    assert payload["k1"]["workflows"] == 3
    assert payload["bundle"]["models"] == 7
    assert payload["bundle"]["workflows"] == 4
    assert payload["k1"]["outputs"] == ["SaveImage"]
    assert payload["bundle"]["outputs"] == ["SaveImage", "SaveVideo"]
    assert payload["missing_model_exit"] != 0
    assert payload["model_tamper_exit"] != 0
    assert payload["model_tamper_rejected"] is True
    for label in ("k1", "bundle"):
        worker = payload[label]["worker"]
        assert worker["exit"] == 0, worker
        assert worker["status"] == "COMPLETE"
        assert worker["files"] == (3 if label == "k1" else 7)
        assert worker["skipped"] == worker["files"]

