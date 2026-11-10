#!/usr/bin/env python3
"""Actual collector budget/timeout harness with a strictly owned slow child."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repo))
    from scripts.i1 import observation, stock_nodes
    from scripts.i1.inspect import inspect_remote
    from scripts.i1.profile import load_profile

    root = Path(tempfile.mkdtemp(prefix="i1-budget-"))
    bin_dir = root / "bin"
    bin_dir.mkdir()
    pid_file = root / "child.pid"
    slow = bin_dir / "nvidia-smi"
    slow.write_text(
        "#!/bin/sh\n"
        f"echo $$ > {pid_file}\n"
        "sleep 30\n",
        encoding="utf-8",
        newline="\n",
    )
    slow.chmod(0o755)
    workspace = root / "workspace"
    workspace.mkdir()
    marker = "f" * 32
    argv = observation.build_snapshot_argv(
        workspace=str(workspace),
        comfy=str(workspace / "ComfyUI"),
        commit="7a28332047b6af638dc6c7f72ef3481a718769fd",
        root=str(workspace / ".avs-i1"),
        attempt="",
        run_id="",
        nonce="",
        guard_sha=observation.guard_sha256(),
        targets=[],
        validator_source=stock_nodes.VALIDATOR_SOURCE,
        mode="environment",
        marker=marker,
        collector_budget=2.0,
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    completed = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=20)
    payload = json.loads(completed.stdout)
    observation._validate_snapshot(
        payload,
        marker=marker,
        run_id="",
        nonce="",
        mode="environment",
        attempt_path="",
        expected_job_sha256="",
        expected_launch_sha256="",
    )
    profile = load_profile(repo / "config" / "i1" / "wan22-14b-ab.profile.json")
    inspection = inspect_remote(
        observation.SnapshotRunner(payload),
        object(),
        workspace=str(workspace),
        profile=profile,
    )
    blocker_codes = {item["code"] for item in inspection["blockers"]}
    warning_codes = {item["code"] for item in inspection["warnings"]}
    nvidia = None
    for item in payload.get("commands") or []:
        try:
            command = json.loads(item["key"])
        except Exception:
            continue
        if command and command[0] == "nvidia-smi":
            nvidia = item
            break
    time.sleep(0.2)
    child_gone = True
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            child_gone = False
        except ProcessLookupError:
            child_gone = True
        except OSError:
            child_gone = False
    summary = {
        "rc": completed.returncode,
        "schema": payload.get("schema"),
        "marker": payload.get("marker"),
        "partial": payload.get("partial"),
        "budget_exhausted": payload.get("budget_exhausted"),
        "nvidia_timed_out": bool(nvidia and nvidia.get("timed_out")),
        "nvidia_unrun": bool(nvidia and nvidia.get("unrun")),
        "pid_file_exists": pid_file.exists(),
        "child_gone": child_gone,
        "collector_budget_seconds": payload.get("collector_budget_seconds"),
        "schema_validated": True,
        "inspect_conclusive": inspection.get("conclusive"),
        "inspect_gpu_blocker": "gpu" in blocker_codes,
        "inspect_os_blocker": "os" in blocker_codes,
        "inspect_gpu_unknown": "gpu_unknown" in warning_codes,
    }
    summary["ok"] = (
        summary["rc"] == 0
        and summary["schema"] == "anime-video-studio/i1-observation-snapshot/v1"
        and summary["marker"] == marker
        and summary["partial"] is True
        and summary["budget_exhausted"] is True
        and summary["nvidia_timed_out"] is True
        and summary["pid_file_exists"] is True
        and summary["child_gone"] is True
        and summary["collector_budget_seconds"] == 2.0
        and summary["schema_validated"] is True
        and summary["inspect_conclusive"] is False
        and summary["inspect_gpu_blocker"] is False
        and summary["inspect_os_blocker"] is False
        and summary["inspect_gpu_unknown"] is True
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
