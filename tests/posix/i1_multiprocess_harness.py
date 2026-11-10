#!/usr/bin/env python3
"""Actual multi-process target lock test (POSIX).

Two independent processes use different workspaces/run/nonce against the same
target lock file.  Only one may hold it.  The holder is SIGKILLed; the OS must
release the lock and the already-waiting contender must acquire it afterwards.
A different target is not blocked.  A tiny guarded file proves the writes are
serialized (holder line before contender line).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _holder(args) -> int:
    sys.path.insert(0, args.repo)
    from scripts.i1 import worker

    lock = worker.TargetLock(
        Path(args.workspace),
        "holder-run",
        "holder-nonce",
        lock_dir=Path(args.lock),
        target_id_value=args.target,
    ).acquire()
    try:
        _append(Path(args.guarded), "holder")
        _append(Path(args.events), "HELD")
        Path(args.ready).write_text("ready", encoding="utf-8")
        while True:
            time.sleep(0.1)
    finally:
        lock.release()


def _try_acquire(args) -> int:
    sys.path.insert(0, args.repo)
    from scripts.i1 import worker

    wrote_blocked = False
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            lock = worker.TargetLock(
                Path(args.workspace),
                "contender-run",
                "contender-nonce",
                lock_dir=Path(args.lock),
                target_id_value=args.target,
            ).acquire()
        except worker.WorkerError:
            if not wrote_blocked:
                _append(Path(args.events), "BLOCKED")
                wrote_blocked = True
            time.sleep(0.05)
            continue
        try:
            _append(Path(args.events), f"ACQUIRED {time.time_ns()}")
            _append(Path(args.guarded), "contender")
        finally:
            lock.release()
        return 0
    _append(Path(args.events), "TIMEOUT")
    return 2


def _other(args) -> int:
    sys.path.insert(0, args.repo)
    from scripts.i1 import worker

    lock = worker.TargetLock(
        Path(args.workspace),
        "other-run",
        "other-nonce",
        lock_dir=Path(args.other_lock),
        target_id_value=args.other_target,
    ).acquire()
    try:
        _append(Path(args.events), "OTHER_ACQUIRED")
        _append(Path(args.other_guarded), "other")
    finally:
        lock.release()
    return 0


def _wait_for(path: Path, needle: str, timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if needle in path.read_text(encoding="utf-8"):
                return True
        except OSError:
            pass
        time.sleep(0.02)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--role", default="orchestrate")
    parser.add_argument("--lock", default="")
    parser.add_argument("--other-lock", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--other-target", default="")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--events", default="")
    parser.add_argument("--guarded", default="")
    parser.add_argument("--other-guarded", default="")
    parser.add_argument("--ready", default="")
    args = parser.parse_args()
    if args.role == "holder":
        return _holder(args)
    if args.role == "contender":
        return _try_acquire(args)
    if args.role == "other":
        return _other(args)
    return _orchestrate(args)


def _orchestrate(args) -> int:
    root = Path(tempfile.mkdtemp(prefix="i1-multiprocess-"))
    events = root / "events.txt"
    guarded = root / "guarded.txt"
    other_guarded = root / "other-guarded.txt"
    ready = root / "holder-ready"
    target = "a" * 64
    other_target = "b" * 64
    lock = root / "home" / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{target}.lock"
    other_lock = root / "home" / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{other_target}.lock"
    lock.parent.mkdir(parents=True)
    common = [sys.executable, os.path.abspath(__file__), "--repo", str(Path(args.repo).resolve())]
    processes = []
    holder = None
    try:
        holder = subprocess.Popen(
            [
                *common, "--role", "holder", "--lock", str(lock), "--target", target,
                "--workspace", str(root / "workspace-a"), "--events", str(events),
                "--guarded", str(guarded), "--ready", str(ready),
            ],
            cwd=str(Path(args.repo).resolve()),
        )
        processes.append(holder)
        if not _wait_for(ready, "ready"):
            print(json.dumps({"status": "HOLDER_NOT_READY"}, ensure_ascii=False))
            return 1
        other = subprocess.Popen(
            [
                *common, "--role", "other", "--other-lock", str(other_lock),
                "--other-target", other_target, "--workspace", str(root / "workspace-other"),
                "--events", str(events), "--other-guarded", str(other_guarded),
            ],
            cwd=str(Path(args.repo).resolve()),
        )
        processes.append(other)
        if other.wait(timeout=10) != 0:
            print(json.dumps({"status": "DIFFERENT_TARGET_BLOCKED"}, ensure_ascii=False))
            return 1
        contender = subprocess.Popen(
            [
                *common, "--role", "contender", "--lock", str(lock), "--target", target,
                "--workspace", str(root / "workspace-b"), "--events", str(events),
                "--guarded", str(guarded),
            ],
            cwd=str(Path(args.repo).resolve()),
        )
        processes.append(contender)
        if not _wait_for(events, "BLOCKED"):
            print(json.dumps({"status": "CONTENDER_NOT_BLOCKED"}, ensure_ascii=False))
            return 1
        text = events.read_text(encoding="utf-8")
        if any(line.startswith("ACQUIRED ") for line in text.splitlines()):
            print(json.dumps({"status": "LOCK_OVERLAP_BEFORE_KILL", "events": text.splitlines(), "holder_poll": holder.poll()}, ensure_ascii=False))
            return 1
        os.kill(holder.pid, signal.SIGKILL)
        kill_ns = time.time_ns()
        holder.wait(timeout=10)
        if contender.wait(timeout=20) != 0:
            print(json.dumps({"status": "CONTENDER_FAILED_AFTER_KILL", "events": events.read_text(encoding="utf-8")}, ensure_ascii=False))
            return 1
        lines = events.read_text(encoding="utf-8").splitlines()
        acquired = [int(line.split()[1]) for line in lines if line.startswith("ACQUIRED ")]
        if not acquired or acquired[0] < kill_ns:
            print(json.dumps({"status": "ACQUIRE_BEFORE_OS_RELEASE", "events": lines, "kill_ns": kill_ns}, ensure_ascii=False))
            return 1
        guarded_lines = guarded.read_text(encoding="utf-8").splitlines()
        if guarded_lines != ["holder", "contender"]:
            print(json.dumps({"status": "GUARDED_WRITE_OVERLAP", "guarded": guarded_lines}, ensure_ascii=False))
            return 1
        if not other_guarded.is_file():
            print(json.dumps({"status": "DIFFERENT_TARGET_NOT_WRITTEN"}, ensure_ascii=False))
            return 1
        print(
            json.dumps(
                {
                    "status": "OFFLINE_VERIFIED",
                    "same_target_processes": 2,
                    "one_holder_then_kill_release": True,
                    "no_overlap_guarded_file": True,
                    "different_target_not_blocked": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.wait(timeout=5)
        import shutil

        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
