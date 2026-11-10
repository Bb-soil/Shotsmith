"""Fail-closed normalization of a dead same-run downloader.

Frozen v3_remote_launcher only accepts ``--resume-dead`` for a same-run FAILED
worker.  This tiny staged helper proves that the recorded owner is the same
run/nonce, its PID is dead, and its heartbeat is stale; only then does it add a
recovery marker and rewrite the durable status as resumable FAILED.  It never
kills a PID or removes an owner/job record.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

try:
    import i1_util as util
except ImportError:  # pragma: no cover
    from scripts.i1 import util  # type: ignore

OWNER_SCHEMA = "anime-video-studio/p1-v3-process-owner/v1"
RECOVERY_SCHEMA = "anime-video-studio/i1-worker-recovery/v1"


class RecoveryError(Exception):
    pass


def _pid_alive(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == 259
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def recover(
    root: Path,
    attempt_dir: Path,
    target_lock_path: Path,
    run_id: str,
    nonce: str,
    stale_seconds: float,
) -> dict[str, Any]:
    if not root.is_absolute() or root.name != ".avs-i1":
        raise RecoveryError("owned root is invalid")
    if (
        attempt_dir.parent.name != "attempts"
        or attempt_dir.parent.parent != root
    ):
        raise RecoveryError("attempt directory is invalid")
    owner = util.read_json(attempt_dir / "worker-owner.json")
    ack = util.read_json(attempt_dir / "worker-ack.json")
    status_path = attempt_dir / "status.json"
    heartbeat_path = attempt_dir / "heartbeat.json"
    if owner.get("schema") != OWNER_SCHEMA or owner.get("kind") != "download_worker":
        raise RecoveryError("worker owner schema/kind is invalid")
    if owner.get("run_id") != run_id or owner.get("nonce") != nonce:
        raise RecoveryError("worker owner run/nonce mismatch")
    if ack.get("run_id") != run_id or ack.get("nonce") != nonce:
        raise RecoveryError("worker ack run/nonce mismatch")
    pid = owner.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RecoveryError("worker owner PID is invalid")
    if ack.get("pid") != pid:
        raise RecoveryError("worker ack PID mismatch")
    if _pid_alive(pid):
        raise RecoveryError("worker owner PID is still alive; no recovery")
    status = util.read_json(status_path)
    if status.get("run_id") != run_id or status.get("nonce") != nonce:
        raise RecoveryError("worker status run/nonce mismatch")
    current = status.get("status")
    if current == "FAILED":
        return {"status": "ALREADY_FAILED", "run_id": run_id, "nonce": nonce}
    if current != "RUNNING":
        raise RecoveryError(f"worker status {current!r} is not recoverable")
    heartbeat = util.read_json(heartbeat_path)
    at = heartbeat.get("heartbeat_at")
    if not isinstance(at, (int, float)) or isinstance(at, bool):
        raise RecoveryError("worker heartbeat timestamp is invalid")
    age = time.time() - float(at)
    if age < 0 or age <= stale_seconds:
        raise RecoveryError("worker heartbeat is still fresh; no recovery")
    lock_owner_path = target_lock_path.with_name(
        target_lock_path.name + ".owner.json"
    )
    if lock_owner_path.is_file():
        lock_owner = util.read_json(lock_owner_path)
        lock_pid = lock_owner.get("pid")
        if (
            lock_owner.get("run_id") != run_id
            or lock_owner.get("nonce") != nonce
            or isinstance(lock_pid, bool)
            or not isinstance(lock_pid, int)
            or _pid_alive(lock_pid)
        ):
            raise RecoveryError("target lock is foreign or still alive")
    payload = {
        "schema": "anime-video-studio/i1-download-status/v1",
        "run_id": run_id,
        "nonce": nonce,
        "status": "FAILED",
        "stage": "INTERRUPTED",
        "recovered_interrupted": True,
        "recovered_from": "RUNNING",
        "heartbeat_age_seconds": age,
        "error": "INTERRUPTED_STALE_HEARTBEAT",
    }
    util.atomic_json(status_path, payload)
    util.atomic_json(
        attempt_dir / "recovery.json",
        {
            "schema": RECOVERY_SCHEMA,
            "run_id": run_id,
            "nonce": nonce,
            "pid": pid,
            "heartbeat_age_seconds": age,
            "status": "RECOVERED_INTERRUPTED",
            "at": time.time(),
        },
    )
    return {
        "status": "RECOVERED_INTERRUPTED",
        "run_id": run_id,
        "nonce": nonce,
        "heartbeat_age_seconds": age,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 6:
        print(
            "usage: i1_recover_worker.py <root> <attempt_dir> <target_lock> <run_id> <nonce> <stale_seconds>",
            file=sys.stderr,
        )
        return 2
    try:
        result = recover(
            Path(args[0]),
            Path(args[1]),
            Path(args[2]),
            args[3],
            args[4],
            float(args[5]),
        )
    except Exception as exc:
        print(json.dumps({"status": "REFUSED", "error": type(exc).__name__}, sort_keys=True))
        return 3
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
