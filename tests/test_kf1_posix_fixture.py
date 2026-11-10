from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WSL = shutil.which("wsl")


def _wsl_path(path: Path) -> str:
    text = str(path.resolve())
    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        return f"/mnt/{drive}/" + text[3:].replace("\\", "/")
    return text.replace("\\", "/")


@pytest.mark.skipif(WSL is None, reason="WSL unavailable")
def test_wsl_tiny_kf1_profile_staging_and_imports() -> None:
    fixture = ROOT / "data" / f"kf1-wsl-fixture-{uuid.uuid4().hex[:8]}"
    try:
        fixture.mkdir(parents=True)
        shutil.copytree(ROOT / "scripts", fixture / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "config" / "i1", fixture / "config" / "i1")
        shutil.copytree(ROOT / "manifests", fixture / "manifests")
        shutil.copytree(ROOT / "workflows" / "K1", fixture / "workflows" / "K1")
        shutil.copytree(ROOT / "workflows" / "P1", fixture / "workflows" / "P1")
        shutil.copytree(
            ROOT / "docs" / "k1-evidence" / "kf1-implementation-20261029" / "source",
            fixture / "docs" / "k1-evidence" / "kf1-implementation-20261029" / "source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        staging = fixture / "_stage"
        staging.mkdir()
        copies = {
            "i1_profile.py": ROOT / "scripts" / "i1" / "profile.py",
            "i1_util.py": ROOT / "scripts" / "i1" / "util.py",
            "probe.py": ROOT / "scripts" / "i1" / "probe.py",
            "worker.py": ROOT / "scripts" / "i1" / "worker.py",
            "download_entry_shim.py": ROOT / "scripts" / "i1" / "download_entry_shim.py",
            "child_entry.py": ROOT / "scripts" / "i1" / "child_entry.py",
            "comfy_entry_shim.py": ROOT / "scripts" / "i1" / "comfy_entry_shim.py",
            "comfy_supervisor.py": ROOT / "scripts" / "i1" / "comfy_supervisor.py",
            "setup_plan.py": ROOT / "scripts" / "p1" / "setup_plan.py",
            "preflight.py": ROOT / "scripts" / "p1" / "preflight.py",
        }
        for name, source in copies.items():
            shutil.copy2(source, staging / name)
        # Actual production profiles are staged and read inside WSL.
        for name in (
            "wan22-14b-ab.profile.json",
            "keyframes-qwen-edit-2511.profile.json",
            "bundle-wan22-qwen-edit-2511.profile.json",
        ):
            shutil.copy2(ROOT / "config" / "i1" / name, staging / name)
        wsl_fixture = _wsl_path(fixture)
        script = (
            "set -euo pipefail\n"
            f"cd '{wsl_fixture}'\n"
            "python3 - <<'PY'\n"
            "import json, pathlib, sys\n"
            "sys.path.insert(0, '.')\n"
            "from scripts.i1.profile import load_profile, verify_profile_sources\n"
            "root = pathlib.Path('.').resolve()\n"
            "wan = load_profile('config/i1/wan22-14b-ab.profile.json')\n"
            "k1 = load_profile('config/i1/keyframes-qwen-edit-2511.profile.json')\n"
            "bundle = load_profile('config/i1/bundle-wan22-qwen-edit-2511.profile.json')\n"
            "assert len(wan.models) == 4\n"
            "assert len(k1.models) == 3 and k1.model_bytes() == 30172239743\n"
            "assert len(bundle.models) == 7 and bundle.model_bytes() == 65751447622\n"
            "assert {o['class'] for o in k1.output_specs} == {'SaveImage'}\n"
            "assert {o['class'] for o in bundle.output_specs} == {'SaveImage', 'SaveVideo'}\n"
            "for profile in (wan, k1, bundle):\n"
            "    verify_profile_sources(profile, root)\n"
            "print('WSL_PROFILES_OK', len(wan.models), len(k1.models), len(bundle.models))\n"
            "PY\n"
            f"cd '{wsl_fixture}/_stage'\n"
            "python3 -m py_compile probe.py worker.py i1_profile.py i1_util.py setup_plan.py preflight.py download_entry_shim.py child_entry.py comfy_entry_shim.py comfy_supervisor.py\n"
            "python3 - <<'PY'\n"
            "import pathlib, sys\n"
            "sys.path.insert(0, '.')\n"
            "import i1_profile, i1_util, setup_plan, preflight, probe, worker\n"
            "for name in ('wan22-14b-ab.profile.json', 'keyframes-qwen-edit-2511.profile.json', 'bundle-wan22-qwen-edit-2511.profile.json'):\n"
            "    assert pathlib.Path(name).is_file()\n"
            "k1 = i1_profile.load_profile('keyframes-qwen-edit-2511.profile.json')\n"
            "bundle = i1_profile.load_profile('bundle-wan22-qwen-edit-2511.profile.json')\n"
            "assert len(k1.models) == 3 and len(bundle.models) == 7\n"
            "assert {o['class'] for o in k1.output_specs} == {'SaveImage'}\n"
            "assert {o['class'] for o in bundle.output_specs} == {'SaveImage', 'SaveVideo'}\n"
            "print('WSL_IMPORTS_OK', probe.SCHEMA)\n"
            "print('WSL_STAGE_OK')\n"
            "PY\n"
        )
        completed = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "WSL_PROFILES_OK 4 3 7" in completed.stdout
        assert "WSL_STAGE_OK" in completed.stdout
    finally:
        shutil.rmtree(fixture, ignore_errors=True)
