#!/usr/bin/env python3
"""Actual sh-wrapper fallback with python3 absent (controlled PATH, no SSH)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repo))
    from scripts.i1 import observation

    root = Path(tempfile.mkdtemp(prefix="i1-nopy-"))
    bin_dir = root / "bin"
    bin_dir.mkdir()
    os.symlink("/bin/sh", bin_dir / "sh")
    workspace = root / "workspace"
    workspace.mkdir()
    marker = "e" * 32
    argv = observation.build_snapshot_argv(
        workspace=str(workspace),
        comfy=str(workspace / "ComfyUI"),
        commit="x",
        root=str(workspace / ".avs-i1"),
        attempt="",
        run_id="",
        nonce="",
        guard_sha=observation.guard_sha256(),
        targets=[],
        validator_source="exec('')",
        mode="environment",
        marker=marker,
        collector_budget=1.0,
    )
    env = {"PATH": str(bin_dir), "LANG": "C", "HOME": str(root)}
    completed = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=10)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        print("RAW_STDOUT", repr(completed.stdout))
        print("RAW_STDERR", repr(completed.stderr))
        raise
    summary = {
        "rc": completed.returncode,
        "schema": payload.get("schema"),
        "marker": payload.get("marker"),
        "python3": payload.get("python3"),
        "tool_presence": payload.get("tool_presence"),
        "commands": payload.get("commands"),
        "attempt_state": (payload.get("attempt") or {}).get("state"),
        "partial": payload.get("partial"),
        "budget_exhausted": payload.get("budget_exhausted"),
        "collector_budget_seconds": payload.get("collector_budget_seconds"),
    }
    summary["ok"] = (
        summary["rc"] == 0
        and summary["schema"] == "anime-video-studio/i1-observation-snapshot/v1"
        and summary["marker"] == marker
        and summary["python3"] is False
        and summary["tool_presence"] == {"sh": True}
        and summary["commands"] == []
        and summary["attempt_state"] == "INCONCLUSIVE"
        and summary["partial"] is False
        and summary["budget_exhausted"] is False
        and summary["collector_budget_seconds"] == 1.0
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
