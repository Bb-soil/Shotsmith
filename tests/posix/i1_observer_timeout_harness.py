#!/usr/bin/env python3
"""Actual owned-child observer timeout/cleanup harness (no SSH)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repo))
    from scripts.i1 import observation

    root = Path(tempfile.mkdtemp(prefix="i1-observer-timeout-"))
    workspace = root / "workspace"
    workspace.mkdir()
    pid_file = root / "observer.pid"
    source = (
        "import os,time\n"
        "def observe_download(**kwargs):\n"
        f"    open({str(pid_file)!r},'w').write(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        "    return {'state':'OK','process':'LIVE'}\n"
    )
    raw = source.encode("utf-8")
    guard_b64 = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    guard_sha = hashlib.sha256(raw).hexdigest()
    marker = "1" * 32
    argv = observation.build_snapshot_argv(
        workspace=str(workspace),
        comfy=str(workspace / "ComfyUI"),
        commit="x",
        root=str(workspace / ".avs-i1"),
        attempt=str(workspace / ".avs-i1" / "attempts" / "0001"),
        run_id="a" * 32,
        nonce="b" * 32,
        guard_sha=guard_sha,
        targets=[],
        validator_source="",
        mode="attempt",
        marker=marker,
        guard_source_b64=guard_b64,
        expected_job_sha256="1" * 64,
        expected_launch_sha256="2" * 64,
        collector_budget=1.0,
    )
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    payload = json.loads(completed.stdout)
    time.sleep(0.2)
    child_gone = False
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            child_gone = True
        except OSError:
            child_gone = False
    attempt = payload.get("attempt") or {}
    summary = {
        "rc": completed.returncode,
        "schema": payload.get("schema"),
        "marker": payload.get("marker"),
        "partial": payload.get("partial"),
        "budget_exhausted": payload.get("budget_exhausted"),
        "attempt_state": attempt.get("state"),
        "attempt_reason": attempt.get("reason"),
        "pid_file_exists": pid_file.exists(),
        "child_gone": child_gone,
        "collector_budget_seconds": payload.get("collector_budget_seconds"),
    }
    summary["ok"] = (
        summary["rc"] == 0
        and summary["schema"] == "anime-video-studio/i1-observation-snapshot/v1"
        and summary["marker"] == marker
        and summary["partial"] is True
        and summary["budget_exhausted"] is True
        and summary["attempt_state"] == "INCONCLUSIVE"
        and summary["attempt_reason"] == "observer_child_timeout"
        and summary["pid_file_exists"] is True
        and summary["child_gone"] is True
        and summary["collector_budget_seconds"] == 1.0
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
