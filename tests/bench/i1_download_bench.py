#!/usr/bin/env python3
"""Reproducible local I1 download benchmark, not a pytest timing threshold.

Modes:
  old   byte-exact pre-change worker.py + setup_plan.py, serial
  new1  current worker.py + setup_plan.py, concurrency=1 (persistence isolation)
  new2  current worker.py + setup_plan.py, concurrency=2 (production cap)

The HTTP fixture is local and deterministic.  The worker still runs the real
production `worker.run()` path; only `setup_plan._urllib_open` is mapped to the
local HTTP server because no external network or TLS setup is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
MIB = 1024 * 1024
DEFAULT_FILES = 4
DEFAULT_MIB = 32
DEFAULT_RATE_MIB = 8
DEFAULT_REPEATS = 3


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        from importlib.machinery import SourceFileLoader

        spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Fixture:
    def __init__(self, files: int, size: int, rate_bytes: float) -> None:
        self.files: dict[str, bytes] = {}
        self.size = size
        for index in range(files):
            name = f"file{index}.bin"
            seed = (f"i1-bench-{index}-".encode() * (size // 12 + 2))[:size]
            self.files[name] = seed
        self.rate_bytes = rate_bytes
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


def _handler(fixture: Fixture):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            return

        def do_GET(self) -> None:
            name = self.path.split("?", 1)[0].lstrip("/")
            data = fixture.files.get(name)
            if data is None:
                self.send_error(404)
                return
            fixture.enter()
            try:
                start = 0
                status = 200
                if self.headers.get("Range"):
                    try:
                        start = int(self.headers["Range"].split("=", 1)[1].split("-", 1)[0] or "0")
                        status = 206
                    except Exception:
                        self.send_error(416)
                        return
                self.send_response(status)
                self.send_header("Content-Length", str(len(data) - start))
                self.send_header("Accept-Ranges", "bytes")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
                self.end_headers()
                chunk_size = 64 * 1024
                for offset in range(start, len(data), chunk_size):
                    chunk = data[offset : min(offset + chunk_size, len(data))]
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if fixture.rate_bytes > 0:
                        time.sleep(len(chunk) / fixture.rate_bytes)
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                fixture.leave()

    return Handler


def _profile_and_payload(
    repo: Path,
    tmp: Path,
    fixture: Fixture,
    port: int,
    concurrency: int,
) -> tuple[Path, Any]:
    sys.path.insert(0, str(repo))
    from scripts.i1 import profile as profile_mod

    base = json.loads(
        (repo / "config" / "i1" / "wan22-14b-ab.profile.json").read_text(encoding="utf-8")
    )
    base["disk"] = {"install_overhead_bytes": 0, "reserve_bytes": 0}
    for index, item in enumerate(base["models"]):
        name = f"file{index}.bin"
        data = fixture.files[name]
        item["url"] = f"https://bench.invalid/{name}"
        item["target"] = f"models/bench/{name}"
        item["size_bytes"] = len(data)
        item["sha256"] = hashlib.sha256(data).hexdigest()
    profile = profile_mod.validate_profile(base)
    workspace = tmp / "workspace"
    owned = workspace / ".avs-i1"
    attempt = owned / "attempts" / "0001"
    comfy = workspace / "ComfyUI"
    attempt.mkdir(parents=True)
    comfy.mkdir(parents=True)
    profile_path = attempt / "profile.json"
    profile_path.write_text(
        json.dumps(profile.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    target_home = tmp / "home"
    target_home.mkdir()
    target_id = "t" * 64
    lock_path = target_home / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{target_id}.lock"
    job = {
        "schema": "anime-video-studio/i1-download-job/v1",
        "run_id": "bench-run",
        "nonce": f"bench-{concurrency}",
        "workspace": str(workspace),
        "owned_root": str(attempt),
        "attempt_dir": str(attempt),
        "comfy_root": str(comfy),
        "profile_path": str(profile_path),
        "profile_sha256": profile.sha256,
        "target_home": str(target_home),
        "target_id": target_id,
        "target_lock_path": str(lock_path),
        "deadline_epoch": time.time() + 3600,
        "log": str(attempt / "download.log"),
        "status_path": str(attempt / "status.json"),
        "progress_path": str(attempt / "progress.json"),
        "heartbeat_path": str(attempt / "heartbeat.json"),
        "required_bytes": profile.disk_required_bytes,
    }
    job_path = attempt / "job.json"
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return job_path, profile


def _run_child(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from scripts.i1 import profile as profile_mod
    from scripts.i1 import util as util_mod

    old = args.mode == "old"
    worker_path = Path(args.old_worker) if old else Path(args.new_worker)
    setup_path = Path(args.old_setup_plan) if old else Path(args.new_setup_plan)
    setup_plan = _load_module("setup_plan", setup_path)
    sys.modules["i1_profile"] = profile_mod
    sys.modules["i1_util"] = util_mod

    fixture = Fixture(args.files, args.mib * MIB, args.rate_mib * MIB)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fixture))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def local_open(url: str, headers, timeout: float = 60.0):
        if not url.startswith("https://bench.invalid/"):
            raise AssertionError(f"unexpected external URL: {url}")
        name = url.rsplit("/", 1)[-1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/{name}",
            headers=dict(headers),
            method="GET",
        )
        return urllib.request.urlopen(request, timeout=float(timeout))

    setup_plan._urllib_open = local_open
    if os.environ.get("I1_BENCH_DEBUG_CONTAIN"):
        original_contained = setup_plan._contained_path

        def debug_contained(root, target):
            try:
                return original_contained(root, target)
            except BaseException:
                root_resolved = Path(root).resolve()
                candidate = (root_resolved / target).resolve()
                print(
                    json.dumps(
                        {
                            "root_input": str(root),
                            "root_resolved": str(root_resolved),
                            "candidate": str(candidate),
                            "candidate_parents": [str(item) for item in candidate.parents],
                            "target": target,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
                raise

        setup_plan._contained_path = debug_contained
    counts: Counter[str] = Counter()
    bytes_written: Counter[str] = Counter()
    real_atomic = util_mod.atomic_json

    def counted_atomic(path, payload):
        name = Path(path).name
        counts[name] += 1
        bytes_written[name] += len(util_mod.json_text(payload).encode("utf-8"))
        return real_atomic(path, payload)

    util_mod.atomic_json = counted_atomic

    tmp = Path(tempfile.mkdtemp(prefix=f"i1-bench-{args.mode}-"))
    result: dict[str, object] = {"mode": args.mode, "run": args.run_index}
    try:
        job_path, profile = _profile_and_payload(
            repo, tmp, fixture, port, 1 if old else (1 if args.mode == "new1" else 2)
        )
        worker = _load_module("bench_worker", worker_path)
        started = time.monotonic()
        if old:
            code = worker.run(job_path)
        else:
            code = worker.run(job_path, concurrency=1 if args.mode == "new1" else 2)
        wall = time.monotonic() - started
        final_ok = True
        for item in profile.models:
            final = Path(job_path).parent.parent.parent.parent / "ComfyUI" / item["target"]
            if not final.is_file():
                final_ok = False
                break
            if hashlib.sha256(final.read_bytes()).hexdigest() != item["sha256"]:
                final_ok = False
                break
        leftover = [
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith(("i1-download-", "i1-progress-"))
        ]
        admin = {
            name: {
                "atomic_json": counts.get(name, 0),
                "bytes": bytes_written.get(name, 0),
            }
            for name in ("progress.json", "status.json", "heartbeat.json")
        }
        result.update(
            {
                "exit": code,
                "wall_seconds": wall,
                "admin": admin,
                "admin_atomic_json": sum(item["atomic_json"] for item in admin.values()),
                "admin_bytes": sum(item["bytes"] for item in admin.values()),
                "admin_fsync_estimate": sum(item["atomic_json"] for item in admin.values()),
                "admin_replace_estimate": sum(item["atomic_json"] for item in admin.values()),
                "owner_atomic_json": counts.get("lock.owner.json", 0)
                + counts.get(f"{'t'*64}.lock.owner.json", 0),
                "server_max_active": fixture.max_active,
                "final_sha_ok": final_ok,
                "leftover_worker_threads": leftover,
            }
        )
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(result, sort_keys=True))
    return 0 if "error" not in result else 1


def _parent(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    modes = ("old", "new1", "new2")
    rows: list[dict] = []
    for index in range(args.repeats):
        for mode in modes:
            command = [
                sys.executable,
                str(script),
                "--child",
                "--mode",
                mode,
                "--run-index",
                str(index),
                "--repo",
                str(Path(args.repo).resolve()),
                "--old-worker",
                str(Path(args.old_worker).resolve()),
                "--old-setup-plan",
                str(Path(args.old_setup_plan).resolve()),
                "--new-worker",
                str(Path(args.new_worker).resolve()),
                "--new-setup-plan",
                str(Path(args.new_setup_plan).resolve()),
                "--files",
                str(args.files),
                "--mib",
                str(args.mib),
                "--rate-mib",
                str(args.rate_mib),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
            line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {
                    "mode": mode,
                    "run": index,
                    "error": f"child exit {completed.returncode}",
                    "stdout": completed.stdout[-2000:],
                    "stderr": completed.stderr[-2000:],
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row.get("mode") == mode and "error" not in row]
        if not selected:
            summary[mode] = {"runs": 0, "errors": len([r for r in rows if r.get("mode") == mode])}
            continue
        walls = [float(row["wall_seconds"]) for row in selected]
        admins = [int(row["admin_atomic_json"]) for row in selected]
        max_active = [int(row["server_max_active"]) for row in selected]
        summary[mode] = {
            "runs": len(selected),
            "wall_median": statistics.median(walls),
            "wall_min": min(walls),
            "wall_max": max(walls),
            "admin_median": statistics.median(admins),
            "admin_min": min(admins),
            "admin_max": max(admins),
            "server_max_active": max(max_active),
            "final_sha_ok": all(row.get("final_sha_ok") for row in selected),
            "leftover_threads": sorted(
                {name for row in selected for name in row.get("leftover_worker_threads", [])}
            ),
        }
    payload = {
        "fixture": {
            "files": args.files,
            "mib_each": args.mib,
            "rate_mib_per_connection": args.rate_mib,
            "transport": "local HTTP via injected opener; no external network/TLS",
        },
        "repeats": args.repeats,
        "rows": rows,
        "summary": summary,
    }
    output = Path(args.output) if args.output else Path("docs/i1-evidence/i1-download-optimization-20261030/benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("old", "new1", "new2"))
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--repo", default=str(ROOT))
    parser.add_argument(
        "--old-worker",
        default=str(ROOT / "docs/i1-evidence/i1-download-optimization-20261030/baseline/worker.py.before"),
    )
    parser.add_argument(
        "--old-setup-plan",
        default=str(ROOT / "docs/i1-evidence/i1-download-optimization-20261030/baseline/setup_plan.py.before"),
    )
    parser.add_argument("--new-worker", default=str(ROOT / "scripts/i1/worker.py"))
    parser.add_argument("--new-setup-plan", default=str(ROOT / "scripts/p1/setup_plan.py"))
    parser.add_argument("--files", type=int, default=DEFAULT_FILES)
    parser.add_argument("--mib", type=int, default=DEFAULT_MIB)
    parser.add_argument("--rate-mib", type=float, default=DEFAULT_RATE_MIB)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.child:
        if not args.mode:
            parser.error("--child requires --mode")
        return _run_child(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
