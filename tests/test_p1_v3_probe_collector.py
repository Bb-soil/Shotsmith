from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.p1 import v3_bootstrap, v3_readiness
from tests.p1_probe_helpers import make_probe_workspace, model_facts_for

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_collector_runs_fixed_probe_and_checks_plan_and_collector_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    target = "models/diffusion_models/fixture.safetensors"
    workspace, mods, commit = make_probe_workspace(tmp_path, [target])
    facts = model_facts_for(workspace, [target])
    plan = {
        "schema": v3_readiness.RUNTIME_PLAN_SCHEMA,
        "run_id": "run-fixture",
        "nonce": "nonce-fixture",
        "session_binding_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "source_approval_sha256": "c" * 64,
        "host_key": {
            "fingerprint": "SHA256:trusted",
            "known_hosts_sha256": "d" * 64,
            "host": "example.test",
            "user": "root",
            "port": 22,
            "checked_at": "2026-10-26T00:00:00+00:00",
        },
        "workspace": str(workspace),
        "owned_root": str(ROOT),
        "venv_python": sys.executable,
        "comfy_dir": str(workspace / "ComfyUI"),
        "object_info_path": str(workspace / "p1-evidence" / "object_info.json"),
        "preflight_path": str(ROOT / "scripts" / "p1" / "preflight.py"),
        "models": facts,
        "required_classes": [
            "KSampler",
            "CLIPTextEncode",
            "VAEDecode",
            "UNETLoader",
            "CLIPLoader",
            "VAELoader",
            "ModelSamplingSD3",
            "Wan22ImageToVideoLatent",
            "LoadImage",
            "CreateVideo",
            "SaveVideo",
        ],
        "required_inputs": {
            "Wan22ImageToVideoLatent": ["width", "height", "length", "start_image"],
            "CreateVideo": ["fps"],
            "CLIPLoader": ["type", "device"],
        },
        "node_classes": {"3": "KSampler", "58": "SaveVideo"},
        "expected_comfy_commit": commit,
        "deadline_epoch": time.time() + 120,
        "max_bytes": 256 * 1024,
    }
    staged: dict[str, bytes] = {}

    class FakeRunner:
        def put_file(self, identity, remote, local, **kwargs):
            staged[remote] = Path(local).read_bytes()
            return None

        def run_remote(self, identity, args, **kwargs):
            plan_remote = args[-1]
            plan_local = tmp_path / "staged-plan.json"
            plan_local.write_bytes(staged[plan_remote])
            env = dict(os.environ)
            env["PYTHONPATH"] = str(mods)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "p1" / "v3_runtime_probe.py"), "--plan", str(plan_local)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                raise AssertionError(result.stderr or result.stdout)
            return SimpleNamespace(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
            )

    identity = SimpleNamespace(host="example.test", user="root", port=22)
    collected = v3_bootstrap.collect_runtime_evidence(
        runner=FakeRunner(),
        identity=identity,
        remote_root="/workspace",
        plan=plan,
        timeout=120,
        deadline_monotonic=time.monotonic() + 120,
    )
    assert collected["facts"]["schema"] == v3_readiness.RUNTIME_EVIDENCE_SCHEMA
    assert collected["facts"]["collector"]["plan_sha256"] == collected["plan_sha256"]
    assert collected["facts"]["collector"]["sha256"] == collected["collector_sha256"]
    assert collected["raw_sha256"] == hashlib.sha256(collected["raw"]).hexdigest()
    assert collected["facts"]["models"][0]["actual_sha256"] == facts[0]["sha256"]
