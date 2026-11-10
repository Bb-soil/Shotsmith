#!/usr/bin/env python3
"""Actual FD-holding descendant cleanup harness (env + observer)."""

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


def _dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repo))
    from scripts.i1 import observation, stock_nodes

    root = Path(tempfile.mkdtemp(prefix="i1-fd-cleanup-"))
    bin_dir = root / "bin"
    bin_dir.mkdir()
    env_desc = root / "env-desc.pid"
    observer_desc = root / "observer-desc.pid"
    slow = bin_dir / "nvidia-smi"
    slow.write_text(
        "#!/bin/sh\n"
        "sleep 30 &\n"
        f"echo $! > {env_desc}\n"
        "exit 0\n",
        encoding="utf-8",
        newline="\n",
    )
    slow.chmod(0o755)
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        start_new_session=True,
    )
    try:
        workspace = root / "workspace"
        workspace.mkdir()
        env_marker = "e" * 32
        env_argv = observation.build_snapshot_argv(
            workspace=str(workspace),
            comfy=str(workspace / "ComfyUI"),
            commit="x",
            root=str(workspace / ".avs-i1"),
            attempt="",
            run_id="",
            nonce="",
            guard_sha=observation.guard_sha256(),
            targets=[],
            validator_source=stock_nodes.VALIDATOR_SOURCE,
            mode="environment",
            marker=env_marker,
            collector_budget=2.0,
        )
        env = dict(os.environ)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env_started = time.monotonic()
        env_completed = subprocess.run(
            env_argv, env=env, capture_output=True, text=True, timeout=10
        )
        env_elapsed = time.monotonic() - env_started
        env_payload = json.loads(env_completed.stdout)
        observation._validate_snapshot(
            env_payload,
            marker=env_marker,
            run_id="",
            nonce="",
            mode="environment",
            attempt_path="",
            expected_job_sha256="",
            expected_launch_sha256="",
        )
        env_nvidia = None
        for item in env_payload.get("commands") or []:
            try:
                command = json.loads(item["key"])
            except Exception:
                continue
            if command and command[0] == "nvidia-smi":
                env_nvidia = item
                break
        time.sleep(0.2)
        env_pid = int(env_desc.read_text().strip()) if env_desc.exists() else None
        env_descendant_gone = env_pid is not None and _dead(env_pid)
        env_worker_alive = not _dead(worker.pid)

        source = (
            "import os,subprocess,sys\n"
            "def observe_download(**kwargs):\n"
            "    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
            f"    open({str(observer_desc)!r},'w').write(str(p.pid))\n"
            "    return {'state':'OK','process':'LIVE'}\n"
        )
        raw = source.encode("utf-8")
        observer_guard_b64 = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
        observer_guard_sha = hashlib.sha256(raw).hexdigest()
        observer_marker = "f" * 32
        observer_argv = observation.build_snapshot_argv(
            workspace=str(workspace),
            comfy=str(workspace / "ComfyUI"),
            commit="x",
            root=str(workspace / ".avs-i1"),
            attempt=str(workspace / ".avs-i1" / "attempts" / "0001"),
            run_id="a" * 32,
            nonce="b" * 32,
            guard_sha=observer_guard_sha,
            targets=[],
            validator_source="",
            mode="attempt",
            marker=observer_marker,
            guard_source_b64=observer_guard_b64,
            expected_job_sha256="1" * 64,
            expected_launch_sha256="2" * 64,
            collector_budget=2.0,
        )
        observer_started = time.monotonic()
        observer_completed = subprocess.run(
            observer_argv, capture_output=True, text=True, timeout=10
        )
        observer_elapsed = time.monotonic() - observer_started
        observer_payload = json.loads(observer_completed.stdout)
        observation._validate_snapshot(
            observer_payload,
            marker=observer_marker,
            run_id="a" * 32,
            nonce="b" * 32,
            mode="attempt",
            attempt_path=str(workspace / ".avs-i1" / "attempts" / "0001"),
            expected_job_sha256="1" * 64,
            expected_launch_sha256="2" * 64,
        )
        time.sleep(0.2)
        observer_pid = (
            int(observer_desc.read_text().strip()) if observer_desc.exists() else None
        )
        observer_descendant_gone = observer_pid is not None and _dead(observer_pid)
        observer_worker_alive = not _dead(worker.pid)
        attempt = observer_payload.get("attempt") or {}
        summary = {
            "env_rc": env_completed.returncode,
            "env_schema": env_payload.get("schema"),
            "env_marker": env_payload.get("marker"),
            "env_partial": env_payload.get("partial"),
            "env_budget_exhausted": env_payload.get("budget_exhausted"),
            "env_descendant_pid": env_pid,
            "env_descendant_gone": env_descendant_gone,
            "env_worker_alive": env_worker_alive,
            "env_elapsed": round(env_elapsed, 3),
            "env_nvidia_exit": env_nvidia.get("exit") if env_nvidia else None,
            "env_nvidia_timed_out": bool(env_nvidia and env_nvidia.get("timed_out")),
            "observer_rc": observer_completed.returncode,
            "observer_schema": observer_payload.get("schema"),
            "observer_marker": observer_payload.get("marker"),
            "observer_partial": observer_payload.get("partial"),
            "observer_budget_exhausted": observer_payload.get("budget_exhausted"),
            "observer_attempt_state": attempt.get("state"),
            "observer_attempt_reason": attempt.get("reason"),
            "observer_descendant_pid": observer_pid,
            "observer_descendant_gone": observer_descendant_gone,
            "observer_worker_alive": observer_worker_alive,
            "observer_elapsed": round(observer_elapsed, 3),
        }
        summary["ok"] = (
            summary["env_rc"] == 0
            and summary["env_schema"] == "anime-video-studio/i1-observation-snapshot/v1"
            and summary["env_marker"] == env_marker
            and summary["env_partial"] is True
            and summary["env_budget_exhausted"] is True
            and summary["env_descendant_gone"] is True
            and summary["env_worker_alive"] is True
            and summary["env_elapsed"] < 3.0
            and summary["env_nvidia_exit"] == 124
            and summary["env_nvidia_timed_out"] is True
            and summary["observer_rc"] == 0
            and summary["observer_schema"] == "anime-video-studio/i1-observation-snapshot/v1"
            and summary["observer_marker"] == observer_marker
            and summary["observer_partial"] is True
            and summary["observer_budget_exhausted"] is True
            and summary["observer_attempt_state"] == "INCONCLUSIVE"
            and summary["observer_attempt_reason"] == "observer_child_timeout"
            and summary["observer_descendant_gone"] is True
            and summary["observer_worker_alive"] is True
            and summary["observer_elapsed"] < 3.0
        )
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["ok"] else 1
    finally:
        if not _dead(worker.pid):
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
        for path in (env_desc, observer_desc):
            if path.exists():
                try:
                    pid = int(path.read_text().strip())
                except ValueError:
                    continue
                if not _dead(pid):
                    try:
                        os.kill(pid, 9)
                    except OSError:
                        pass


if __name__ == "__main__":
    raise SystemExit(main())
