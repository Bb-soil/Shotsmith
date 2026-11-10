"""Offline fixture helpers for executing the real fixed runtime probe."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OBJECT_INFO = ROOT / "tests" / "fixtures" / "p1" / "savevideo-object_info-7a283320.json"


def make_probe_workspace(
    tmp_path: Path,
    model_targets: list[str],
    *,
    extra_inputs: dict[str, list[str]] | None = None,
) -> tuple[Path, Path, str]:
    workspace = tmp_path / "probe-workspace"
    (workspace / "p1-evidence").mkdir(parents=True)
    for index, target in enumerate(model_targets):
        path = workspace / Path(*Path(target).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"probe-fixture-model-{index}".encode("ascii"))
    (workspace / "p1-evidence" / "object_info.json").write_text(
        OBJECT_INFO.read_text(encoding="utf-8"), encoding="utf-8"
    )
    if extra_inputs:
        object_info = json.loads(
            (workspace / "p1-evidence" / "object_info.json").read_text(encoding="utf-8")
        )
        for class_name, input_names in extra_inputs.items():
            descriptor = object_info.get(class_name)
            if not isinstance(descriptor, dict):
                continue
            inputs = descriptor.setdefault("input", {})
            required = inputs.setdefault("required", {})
            for input_name in input_names:
                required.setdefault(input_name, ["STRING", {}])
        (workspace / "p1-evidence" / "object_info.json").write_text(
            json.dumps(object_info), encoding="utf-8"
        )
    (workspace / "p1-evidence" / "system_stats.json").write_text(
        json.dumps(
            {
                "system": {"comfyui_version": "fixture"},
                "devices": [
                    {
                        "name": "NVIDIA GeForce RTX 4090",
                        "type": "cuda",
                        "index": 0,
                        "vram_total": 25757220864,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    comfy = workspace / "ComfyUI"
    (comfy / "comfy").mkdir(parents=True)
    (comfy / "main.py").write_text("# fixture\n", encoding="utf-8")
    (comfy / "comfy" / "__init__.py").write_text(
        '__version__="fixture"\n', encoding="utf-8"
    )
    (comfy / "nodes.py").write_text("# fixture nodes\n", encoding="utf-8")
    site_packages = comfy / "site-packages"
    site_packages.mkdir()
    (site_packages / "torch.py").write_text(
        '__version__="2.9.1+cu128"\n'
        'class version: cuda="12.8"\n'
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return True\n"
        "    @staticmethod\n"
        "    def device_count(): return 1\n"
        "    @staticmethod\n"
        "    def get_device_name(i): return 'NVIDIA GeForce RTX 4090'\n"
        "    @staticmethod\n"
        "    def get_device_capability(i): return (8, 9)\n"
        "    class _Props:\n"
        "        total_memory=25757220864\n"
        "    @staticmethod\n"
        "    def get_device_properties(i): return cuda._Props()\n",
        encoding="utf-8",
    )
    for argv in (
        ["git", "init"],
        ["git", "config", "user.email", "fixture@example.invalid"],
        ["git", "config", "user.name", "fixture"],
        ["git", "add", "."],
        ["git", "commit", "-m", "fixture"],
    ):
        subprocess.run(argv, cwd=comfy, check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "-C", str(comfy), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return workspace, site_packages, commit


def model_facts_for(workspace: Path, targets: list[str]) -> list[dict[str, Any]]:
    facts = []
    for index, target in enumerate(targets):
        path = workspace / Path(*Path(target).parts)
        data = path.read_bytes()
        facts.append(
            {
                "id": f"fixture-{index}",
                "target": target,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return facts


def run_probe_process(
    *,
    tmp_path: Path,
    plan: dict[str, Any],
    mods: Path,
    extra_pythonpath: list[Path] | None = None,
) -> subprocess.CompletedProcess[str]:
    plan_path = tmp_path / "probe-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    env = dict(os.environ)
    paths = [str(mods)]
    if extra_pythonpath:
        paths.extend(str(path) for path in extra_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p1" / "v3_runtime_probe.py"), "--plan", str(plan_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=120,
        check=False,
        cwd=str(tmp_path),
    )


def run_actual_probe(*, tmp_path: Path, plan: dict[str, Any], mods: Path) -> tuple[dict[str, Any], bytes]:
    result = run_probe_process(tmp_path=tmp_path, plan=plan, mods=mods)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    raw = result.stdout.encode("utf-8")
    payload = json.loads(result.stdout)
    assert payload["schema"] == "anime-video-studio/p1-runtime-evidence/v1"
    return payload, raw
