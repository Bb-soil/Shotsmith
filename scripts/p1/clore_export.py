"""Isolated local staged export for Clore.

The supervisor only performs lexical plan validation, spawns a local Python
worker, writes a bounded job to the worker's stdin, and reads a small bounded
JSON result from stdout. The worker owns the filesystem: it creates a unique
run namespace, validates actual paths/symlinks, copies in chunks, hashes, fsyncs,
writes the manifest atomically, and cleans its own temporary parts. The parent
does no job-file write, result-file read, manifest write, or hash after spawn.

Remote SSH/fetch is not implemented. A caller must stage local inputs first in a
separately approved future step. On timeout/cancellation the parent kills and
reaps the worker; if reaping is confirmed it launches a separate bounded cleanup
worker for exactly that run namespace. If cleanup/reap is unconfirmed, it does
not claim success and leaves the owned namespace for honest loss reporting.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_ITEM_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_RESULT_BYTES = 1_000_000
DEFAULT_MAX_JOB_BYTES = 64 * 1024
EXPORT_SCHEMA = "anime-video-studio/p1-clore-export/v1"
RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_EXPORT_ITEMS = (
    ("input.png", "inputs/input.png", True),
    ("output.mp4", "artifact.mp4", True),
    ("seed.txt", "inputs/seed.txt", True),
    ("workflow.json", "inputs/workflow.json", True),
    ("model-hashes.json", "inputs/model-hashes.json", True),
    ("runtime.json", "runtime/runtime.json", True),
    ("logs.txt", "logs/runtime.log", False),
    ("cost-ledger.json", "cost-ledger.json", True),
)


class ExportError(Exception):
    """The export plan/process is unsafe or incomplete."""


@dataclass(frozen=True)
class ExportItem:
    source: str
    target: str
    required: bool = True
    expected_sha256: str | None = None


@dataclass(frozen=True)
class ExportPlan:
    local_dir: Path
    source_dir: Path
    items: tuple[ExportItem, ...]


@dataclass(frozen=True)
class ExportResult:
    status: str
    completed: tuple[dict[str, Any], ...]
    lost: tuple[dict[str, Any], ...]
    manifest_path: Path | None
    message: str
    run_dir: Path | None = None


def _validate_relative(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ExportError(f"{name} must be a non-empty string")
    if "\\" in value or ":" in value:
        raise ExportError(f"{name} must use normalized POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ExportError(f"{name} must be a contained relative path")
    return str(path)


def _validate_root_text(value: Path | str, name: str) -> Path:
    text = str(value)
    if not text or "\x00" in text:
        raise ExportError(f"{name} must be a non-empty path")
    return Path(text)


def build_export_plan(
    local_dir: Path | str,
    *,
    source_dir: Path | str | None = None,
    items: Sequence[ExportItem] | None = None,
) -> ExportPlan:
    """Lexical validation only; no filesystem access in the supervisor."""

    root = _validate_root_text(local_dir, "local_dir")
    source = _validate_root_text(
        source_dir if source_dir is not None else root / "staging", "source_dir"
    )
    if items is None:
        selected = tuple(
            ExportItem(source_name, target, required)
            for source_name, target, required in DEFAULT_EXPORT_ITEMS
        )
    else:
        selected = tuple(items)
    if not selected:
        raise ExportError("export plan must contain at least one item")
    targets: set[str] = set()
    normalized: list[ExportItem] = []
    for item in selected:
        if not isinstance(item, ExportItem):
            raise ExportError("export items must be ExportItem values")
        source_name = _validate_relative(item.source, "export source")
        target = _validate_relative(item.target, "export target")
        if target in targets:
            raise ExportError("export targets must be unique")
        targets.add(target)
        if item.expected_sha256 is not None:
            text = item.expected_sha256.lower()
            if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
                raise ExportError("expected_sha256 must be 64 lowercase hex")
        normalized.append(
            ExportItem(
                source=source_name,
                target=target,
                required=item.required,
                expected_sha256=item.expected_sha256.lower()
                if item.expected_sha256
                else None,
            )
        )
    return ExportPlan(root, source, tuple(normalized))


def _job_payload(
    plan: ExportPlan,
    *,
    deadline_at: datetime,
    max_item_bytes: int,
    delay_seconds: float,
    manifest_delay_seconds: float,
    result_padding_bytes: int,
    run_id: str,
) -> dict[str, Any]:
    if deadline_at.tzinfo is None:
        raise ExportError("deadline_at must be timezone-aware UTC")
    if (
        isinstance(max_item_bytes, bool)
        or not isinstance(max_item_bytes, int)
        or max_item_bytes <= 0
    ):
        raise ExportError("max_item_bytes must be a positive integer")
    if (
        isinstance(delay_seconds, bool)
        or not isinstance(delay_seconds, (int, float))
        or not (delay_seconds >= 0)
        or delay_seconds != delay_seconds
        or delay_seconds == float("inf")
    ):
        raise ExportError("delay_seconds must be finite and non-negative")
    if (
        isinstance(manifest_delay_seconds, bool)
        or not isinstance(manifest_delay_seconds, (int, float))
        or not (manifest_delay_seconds >= 0)
        or manifest_delay_seconds != manifest_delay_seconds
        or manifest_delay_seconds == float("inf")
    ):
        raise ExportError("manifest_delay_seconds must be finite and non-negative")
    if (
        isinstance(result_padding_bytes, bool)
        or not isinstance(result_padding_bytes, int)
        or not (0 <= result_padding_bytes <= 2 * 1024 * 1024)
    ):
        raise ExportError("result_padding_bytes must be bounded and non-negative")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ExportError("run_id must be 32 lowercase hex characters")
    return {
        "schema": EXPORT_SCHEMA,
        "run_id": run_id,
        "source_root": str(plan.source_dir),
        "local_dir": str(plan.local_dir),
        "deadline_epoch": deadline_at.astimezone(timezone.utc).timestamp(),
        "max_item_bytes": max_item_bytes,
        "delay_seconds": float(delay_seconds),
        "manifest_delay_seconds": float(manifest_delay_seconds),
        "result_padding_bytes": result_padding_bytes,
        "items": [
            {
                "source": item.source,
                "target": item.target,
                "required": item.required,
                "expected_sha256": item.expected_sha256,
            }
            for item in plan.items
        ],
    }


def _contained(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ExportError("export path escapes its allowed root")
    if candidate.exists() and candidate.is_symlink():
        raise ExportError("export path must not be a symlink")
    return candidate


def _run_dir(local_dir: Path, run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ExportError("invalid run id")
    if local_dir.exists() and local_dir.is_symlink():
        raise ExportError("export root must not be a symlink")
    local_dir.mkdir(parents=True, exist_ok=True)
    run_dir = local_dir / (".clore-export-" + run_id)
    if run_dir.exists():
        raise ExportError("export run namespace already exists")
    run_dir.mkdir(parents=False)
    return run_dir


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp_name = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            temp_name.unlink(missing_ok=True)
        except OSError:
            pass


def _write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    status: str,
    completed: list[dict[str, Any]],
    lost: list[dict[str, Any]],
) -> tuple[Path, str]:
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema": EXPORT_SCHEMA,
        "run_id": run_id,
        "status": status,
        "completed": completed,
        "lost": lost,
        "note": (
            "local staged archive only; no remote SSH/fetch is implemented. "
            "Clore container data is ephemeral. Failed export is recorded as loss."
        ),
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    digest = _hash_file(manifest_path)
    _atomic_write_text(
        run_dir / "manifest.sha256",
        digest + "  manifest.json\n",
    )
    return manifest_path, digest


def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _emit_result(payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(b"RESULT " + str(len(data)).encode("ascii") + b"\n")
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _worker_failure(message: str) -> None:
    _emit_result(
        {
            "schema": EXPORT_SCHEMA,
            "status": "FAILED",
            "completed": [],
            "lost": [{"reason": message}],
            "manifest_path": None,
            "run_dir": None,
            "message": message,
        }
    )


def internal_copy_main() -> int:
    """Worker entry: read one bounded job from stdin, own all filesystem work."""

    raw = sys.stdin.buffer.read(DEFAULT_MAX_JOB_BYTES + 1)
    if len(raw) > DEFAULT_MAX_JOB_BYTES:
        _worker_failure("job JSON exceeded the bound")
        return 2
    try:
        job = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _worker_failure("job JSON is invalid")
        return 2
    if not isinstance(job, Mapping) or job.get("schema") != EXPORT_SCHEMA:
        _worker_failure("job schema is invalid")
        return 2
    run_id = job.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        _worker_failure("job run id is invalid")
        return 2
    source_root = Path(str(job.get("source_root", "")))
    local_dir = Path(str(job.get("local_dir", "")))
    items = job.get("items")
    if not isinstance(items, list):
        _worker_failure("job items are invalid")
        return 2
    delay = job.get("delay_seconds", 0)
    manifest_delay = job.get("manifest_delay_seconds", 0)
    result_padding = job.get("result_padding_bytes", 0)
    deadline = job.get("deadline_epoch")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        deadline = 0.0
    max_item_bytes = job.get("max_item_bytes")
    if isinstance(max_item_bytes, bool) or not isinstance(max_item_bytes, int):
        max_item_bytes = DEFAULT_MAX_ITEM_BYTES

    completed: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    run_dir: Path | None = None
    manifest_path: Path | None = None
    manifest_sha: str | None = None
    try:
        if source_root.exists() and source_root.is_symlink():
            raise ExportError("staged source root is a symlink")
        run_dir = _run_dir(local_dir, run_id)
        _emit(
            {
                "schema": EXPORT_SCHEMA,
                "event": "STARTED",
                "run_id": run_id,
                "pid": os.getpid(),
            }
        )
        for raw_item in items:
            if not isinstance(raw_item, Mapping):
                lost.append({"reason": "invalid item"})
                continue
            source_name = raw_item.get("source")
            target_name = raw_item.get("target")
            expected = raw_item.get("expected_sha256")
            try:
                source_rel = _validate_relative(source_name, "source")
                target_rel = _validate_relative(target_name, "target")
                source = _contained(source_root, source_rel)
                target = _contained(run_dir, target_rel)
                if not source.is_file():
                    raise ExportError("staged source is not a regular file")
                if time.time() >= float(deadline):
                    raise TimeoutError("export deadline expired")
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + ".part-" + uuid.uuid4().hex)
                if isinstance(delay, (int, float)) and delay > 0:
                    time.sleep(float(delay))
                digest = hashlib.sha256()
                total = 0
                try:
                    with source.open("rb") as src, temp.open("xb") as dst:
                        while True:
                            chunk = src.read(65536)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_item_bytes:
                                raise ExportError("exported item exceeds the size bound")
                            if time.time() >= float(deadline):
                                raise TimeoutError("export deadline expired during copy")
                            digest.update(chunk)
                            dst.write(chunk)
                        dst.flush()
                        os.fsync(dst.fileno())
                    actual = digest.hexdigest()
                    if expected is not None and actual != expected:
                        raise ExportError("exported item hash does not match the plan")
                    os.replace(temp, target)
                    completed.append(
                        {
                            "target": target_rel,
                            "source": source_rel,
                            "size_bytes": total,
                            "sha256": actual,
                        }
                    )
                finally:
                    try:
                        temp.unlink(missing_ok=True)
                    except OSError:
                        pass
            except BaseException as exc:
                lost.append(
                    {
                        "target": target_name,
                        "source": source_name,
                        "reason": type(exc).__name__,
                    }
                )
        status = (
            "COMPLETE"
            if completed and not lost
            else ("PARTIAL" if completed else "FAILED")
        )
        if isinstance(manifest_delay, (int, float)) and manifest_delay > 0:
            time.sleep(float(manifest_delay))
        manifest_path, manifest_sha = _write_manifest(
            run_dir,
            run_id=run_id,
            status=status,
            completed=completed,
            lost=lost,
        )
    except BaseException as exc:
        if run_dir is not None:
            # The worker owns only its own namespace; remove temporary parts.
            for part in run_dir.rglob("*.part-*"):
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass
        _worker_failure(f"worker failed: {type(exc).__name__}")
        return 2

    result_payload: dict[str, Any] = {
        "schema": EXPORT_SCHEMA,
        "run_id": run_id,
        "status": "COMPLETE" if completed and not lost else (
            "PARTIAL" if completed else "FAILED"
        ),
        "completed": completed,
        "lost": lost,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": manifest_sha,
        "run_dir": str(run_dir),
        "message": "export complete" if not lost else "export incomplete; loss recorded",
    }
    if isinstance(result_padding, int) and result_padding > 0:
        result_payload["test_padding"] = "x" * result_padding
    _emit_result(result_payload)
    return 0


def cleanup_namespace_main(root_text: str, run_id: str) -> int:
    """Remove exactly one owned run namespace; never glob the export root."""

    if not RUN_ID_RE.fullmatch(run_id):
        _emit({"status": "FAILED_REAP_INCOMPLETE", "message": "invalid run id"})
        return 2
    root = Path(root_text)
    if root.exists() and root.is_symlink():
        _emit({"status": "FAILED_REAP_INCOMPLETE", "message": "root is a symlink"})
        return 2
    target = (root / (".clore-export-" + run_id)).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents:
        _emit({"status": "FAILED_REAP_INCOMPLETE", "message": "namespace escapes root"})
        return 2
    if target.exists():
        try:
            shutil.rmtree(target)
        except OSError:
            _emit({"status": "FAILED_REAP_INCOMPLETE", "message": "cleanup failed"})
            return 2
    _emit({"status": "CLEANED"})
    return 0


_LATE_SPAWN_TASKS: set[asyncio.Task[Any]] = set()
_LATE_REAP_TASKS: set[asyncio.Task[Any]] = set()
_STDOUT_FRAME_OVERHEAD = 4096


@dataclass(frozen=True)
class LateSpawnOutcome:
    """Fixed safe late-work outcome; never contains exception text."""

    run_id: str | None
    reaped_confirmed: bool
    cleanup_confirmed: bool | None
    error_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "reaped_confirmed": self.reaped_confirmed,
            "cleanup_confirmed": self.cleanup_confirmed,
            "error_code": self.error_code,
        }


_LATE_OUTCOMES: list[LateSpawnOutcome] = []


class _StdoutDrain:
    """Concurrent bounded stdout drain; not limited by readline's 64 KiB."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_RESULT_BYTES) -> None:
        self.max_bytes = max_bytes
        self.buffer = bytearray()
        self.first_payload: Mapping[str, Any] | None = None
        self.first_event = asyncio.Event()
        self.overflow_event = asyncio.Event()
        self.overflow = False
        self.done = False

    async def run(self, stream: asyncio.StreamReader) -> None:
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                room = self.max_bytes + _STDOUT_FRAME_OVERHEAD - len(self.buffer)
                if len(chunk) > room:
                    if room > 0:
                        self.buffer.extend(chunk[:room])
                    self.overflow = True
                    self.overflow_event.set()
                else:
                    self.buffer.extend(chunk)
                if not self.first_event.is_set() and b"\n" in self.buffer:
                    line, _sep, _rest = self.buffer.partition(b"\n")
                    if line.startswith(b"RESULT "):
                        payload: Any = {"event": "RESULT"}
                    else:
                        try:
                            payload = json.loads(line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            payload = None
                    self.first_payload = payload if isinstance(payload, Mapping) else None
                    self.first_event.set()
        finally:
            self.done = True
            self.first_event.set()


def _track_late_spawn(
    task: "asyncio.Task[Any]", *, root: Path, run_id: str | None
) -> None:
    _LATE_SPAWN_TASKS.add(task)

    def done(completed: "asyncio.Task[Any]") -> None:
        _LATE_SPAWN_TASKS.discard(completed)
        if completed.cancelled():
            _LATE_OUTCOMES.append(
                LateSpawnOutcome(run_id, False, None, "spawn_cancelled")
            )
            return
        try:
            process = completed.result()
        except BaseException:
            _LATE_OUTCOMES.append(
                LateSpawnOutcome(run_id, False, None, "spawn_failed")
            )
            return
        reap = asyncio.create_task(_late_reap(process, root, run_id))
        _LATE_REAP_TASKS.add(reap)
        reap.add_done_callback(_LATE_REAP_TASKS.discard)

    task.add_done_callback(done)


async def _late_reap(
    process: asyncio.subprocess.Process, root: Path, run_id: str | None
) -> None:
    reaped = await _kill_and_reap(process)
    if not reaped:
        _LATE_OUTCOMES.append(
            LateSpawnOutcome(run_id, False, None, "reap_unconfirmed")
        )
        return
    if run_id is None:
        _LATE_OUTCOMES.append(LateSpawnOutcome(None, True, None, None))
        return
    cleaned = await _cleanup_run(root, run_id)
    _LATE_OUTCOMES.append(
        LateSpawnOutcome(
            run_id,
            True,
            cleaned,
            None if cleaned else "cleanup_unconfirmed",
        )
    )


async def drain_owned_spawns(timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Best-effort async shutdown for retained late spawn/reap tasks.

    Returns explicit pending counts; does not claim complete cleanup when the
    supplied ownership deadline expires.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not (timeout_seconds > 0)
        or timeout_seconds != timeout_seconds
        or timeout_seconds == float("inf")
    ):
        raise ExportError("timeout_seconds must be finite and positive")
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline and (_LATE_SPAWN_TASKS or _LATE_REAP_TASKS):
        await asyncio.sleep(0.01)
    pending_spawns = len(_LATE_SPAWN_TASKS)
    pending_reaps = len(_LATE_REAP_TASKS)
    tasks_done = not (pending_spawns or pending_reaps)
    outcomes = list(_LATE_OUTCOMES)
    reap_incomplete = sum(1 for outcome in outcomes if not outcome.reaped_confirmed)
    cleanup_incomplete = sum(
        1 for outcome in outcomes if outcome.cleanup_confirmed is False
    )
    complete = tasks_done and reap_incomplete == 0 and cleanup_incomplete == 0
    return {
        "complete": complete,
        "tasks_done": tasks_done,
        "pending_spawns": pending_spawns,
        "pending_reaps": pending_reaps,
        "reap_incomplete": reap_incomplete,
        "cleanup_incomplete": cleanup_incomplete,
        "outcomes": [outcome.to_dict() for outcome in outcomes],
    }


async def _spawn_owned(
    coro: Any,
    *,
    timeout: float,
    root: Path,
    run_id: str | None,
) -> tuple[asyncio.subprocess.Process | None, str | None]:
    """Own a spawn coroutine until it returns or becomes retained late work."""

    task = asyncio.create_task(coro)
    try:
        process = await asyncio.wait_for(
            asyncio.shield(task), timeout=max(0.0, timeout)
        )
        return process, None
    except asyncio.TimeoutError:
        _track_late_spawn(task, root=root, run_id=run_id)
        return None, "pending"
    except asyncio.CancelledError:
        _track_late_spawn(task, root=root, run_id=run_id)
        raise
    except BaseException:
        if task.done():
            _LATE_SPAWN_TASKS.discard(task)
        return None, "failed"


def _parse_stdout_frames(
    drain: _StdoutDrain,
) -> tuple[Mapping[str, Any] | None, str | None]:
    if drain.overflow:
        return None, "worker result exceeded the bound"
    if not drain.buffer:
        return None, "worker produced no result"
    line, sep, rest = bytes(drain.buffer).partition(b"\n")
    if not sep:
        return None, "worker STARTED frame is incomplete"
    if len(line) > 1024:
        return None, "worker STARTED frame is too long"
    if line.startswith(b"RESULT "):
        return None, "worker did not start with the strict STARTED frame"
    try:
        first = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "worker STARTED frame is invalid"
    if not isinstance(first, Mapping) or first.get("event") != "STARTED":
        return None, "worker did not start with the strict STARTED frame"
    header, sep2, body = rest.partition(b"\n")
    if not sep2:
        return None, "worker result frame header is incomplete"
    if len(header) > 64:
        return None, "worker result frame header is too long"
    match = re.fullmatch(rb"RESULT (\d+)", header)
    if match is None:
        return None, "worker result frame header is invalid"
    digits = match.group(1)
    if len(digits) > len(str(DEFAULT_MAX_RESULT_BYTES)):
        return None, "worker result length exceeds the bound"
    length = int(digits)
    if length > DEFAULT_MAX_RESULT_BYTES:
        return None, "worker result exceeded the bound"
    if len(body) < length:
        return None, "worker result frame is truncated"
    if body[length:] != b"":
        return None, "worker result frame has trailing bytes"
    try:
        payload = json.loads(body[:length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "worker result JSON is invalid"
    if not isinstance(payload, Mapping):
        return None, "worker result is not an object"
    return payload, None


def _validate_result_payload(
    payload: Mapping[str, Any], plan: ExportPlan, run_id: str
) -> str | None:
    """Strict structural validation; no fuzzy recovery or legacy payloads."""

    if payload.get("schema") != EXPORT_SCHEMA:
        return "worker result schema mismatch"
    if payload.get("run_id") != run_id:
        return "worker result run id mismatch"
    status = payload.get("status")
    if status not in {"COMPLETE", "PARTIAL", "FAILED"}:
        return "worker result status is invalid"
    completed = payload.get("completed")
    lost = payload.get("lost")
    if not isinstance(completed, list) or not isinstance(lost, list):
        return "worker result lists are invalid"
    planned = {item.target: item.source for item in plan.items}
    seen: set[str] = set()
    completed_targets: set[str] = set()
    lost_targets: set[str] = set()
    for item in completed:
        if not isinstance(item, Mapping):
            return "worker completed item is invalid"
        target = item.get("target")
        source = item.get("source")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if not isinstance(target, str) or target not in planned:
            return "worker completed target is unplanned"
        if source != planned[target]:
            return "worker completed source mismatch"
        if target in seen:
            return "worker completed target is duplicate"
        seen.add(target)
        completed_targets.add(target)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return "worker completed size is invalid"
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return "worker completed hash is invalid"
    for item in lost:
        if not isinstance(item, Mapping):
            return "worker lost item is invalid"
        target = item.get("target")
        source = item.get("source")
        reason = item.get("reason")
        if not isinstance(target, str) or target not in planned:
            return "worker lost target is unplanned"
        if source != planned[target]:
            return "worker lost source mismatch"
        if target in seen:
            return "worker lost target is duplicate"
        seen.add(target)
        lost_targets.add(target)
        if not isinstance(reason, str) or not reason:
            return "worker lost reason is invalid"
    if seen != set(planned):
        return "worker result does not cover exactly the planned items"
    if status == "COMPLETE" and (lost_targets or completed_targets != set(planned)):
        return "COMPLETE result does not cover all items"
    if status == "FAILED" and (completed_targets or not lost_targets):
        return "FAILED result cannot claim unaccounted success"
    if status == "PARTIAL" and not (completed_targets and lost_targets):
        return "PARTIAL result must contain completed and lost items"
    run_text = payload.get("run_dir")
    manifest_text = payload.get("manifest_path")
    if not isinstance(run_text, str) or not isinstance(manifest_text, str):
        return "worker result paths are missing"
    run_path = Path(run_text)
    manifest_path = Path(manifest_text)
    if run_path.name != ".clore-export-" + run_id:
        return "worker run namespace does not match run id"
    if run_path.parent != plan.local_dir:
        return "worker run namespace is outside the export root"
    if manifest_path.name != "manifest.json" or manifest_path.parent != run_path:
        return "worker manifest path is outside its run namespace"
    manifest_sha = payload.get("manifest_sha256")
    if not isinstance(manifest_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
        return "worker manifest hash is invalid"
    return None


async def _wait_reaped(process: asyncio.subprocess.Process, timeout: float) -> bool:
    try:
        await asyncio.wait_for(process.wait(), timeout=max(0.0, timeout))
        return process.returncode is not None
    except (asyncio.TimeoutError, ProcessLookupError, OSError):
        return False


async def _kill_and_reap(process: asyncio.subprocess.Process) -> bool:
    """Return True only after final wait confirmation; no abandoned claim."""

    if process.returncode is not None:
        return await _wait_reaped(process, 0.5)
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass
    if await _wait_reaped(process, 1.0):
        return True
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass
    if await _wait_reaped(process, 1.0):
        return True
    return False


async def _cleanup_run(root: Path, run_id: str, *, timeout: float = 2.0) -> bool:
    process: asyncio.subprocess.Process | None = None
    try:
        process, pending = await _spawn_owned(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "scripts.p1.clore_export",
                "--cleanup-namespace",
                str(root),
                run_id,
                cwd=str(ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=64 * 1024,
            ),
            timeout=timeout,
            root=root,
            run_id=None,
        )
        if process is None:
            return False
        await asyncio.wait_for(process.wait(), timeout=timeout)
        if process.returncode != 0:
            return False
        output = await process.stdout.read(DEFAULT_MAX_RESULT_BYTES + 1)
        if len(output) > DEFAULT_MAX_RESULT_BYTES:
            return False
        payload = json.loads(output.decode("utf-8"))
        return payload.get("status") == "CLEANED"
    except (asyncio.TimeoutError, OSError, ValueError, json.JSONDecodeError):
        if process is not None:
            await _kill_and_reap(process)
        return False
    finally:
        if process is not None and process.returncode is None:
            await _kill_and_reap(process)


async def run_staged_export(
    plan: ExportPlan,
    *,
    deadline_at: datetime,
    now: Any = None,
    max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES,
    worker_delay_seconds: float = 0.0,
    manifest_delay_seconds: float = 0.0,
    result_padding_bytes: int = 0,
    on_started: Callable[[], None] | None = None,
) -> ExportResult:
    """Spawn one owned worker; parent does no filesystem work after start."""

    if not isinstance(plan, ExportPlan):
        raise ExportError("plan must be an ExportPlan")
    if deadline_at.tzinfo is None:
        raise ExportError("deadline_at must be timezone-aware UTC")
    deadline = deadline_at.astimezone(timezone.utc)
    clock = now or (lambda: datetime.now(timezone.utc))
    current = clock().astimezone(timezone.utc)
    remaining = (deadline - current).total_seconds()
    monotonic_deadline = time.monotonic() + max(0.0, remaining)
    if remaining <= 0:
        return ExportResult(
            "FAILED",
            (),
            tuple(
                {
                    "target": item.target,
                    "source": item.source,
                    "reason": "export deadline expired before worker start",
                }
                for item in plan.items
            ),
            None,
            "export deadline already expired",
        )

    run_id = uuid.uuid4().hex
    job = _job_payload(
        plan,
        deadline_at=deadline,
        max_item_bytes=max_item_bytes,
        delay_seconds=worker_delay_seconds,
        manifest_delay_seconds=manifest_delay_seconds,
        result_padding_bytes=result_padding_bytes,
        run_id=run_id,
    )
    job_bytes = json.dumps(job, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(job_bytes) > DEFAULT_MAX_JOB_BYTES:
        raise ExportError("job JSON exceeds the bound")

    def lost_items(reason: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"target": item.target, "source": item.source, "reason": reason}
            for item in plan.items
        )

    process: asyncio.subprocess.Process | None = None
    drain: _StdoutDrain | None = None
    tasks: set[asyncio.Task[Any]] = set()

    async def cancel_tasks() -> None:
        for task in list(tasks):
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                pass
        tasks.clear()

    async def stop_outcome(message: str) -> ExportResult:
        nonlocal process
        reaped = await _kill_and_reap(process) if process is not None else True
        await cancel_tasks()
        if not reaped:
            return ExportResult(
                "FAILED_REAP_INCOMPLETE",
                (),
                lost_items("export worker reap unconfirmed; namespace retained"),
                None,
                "worker reap unconfirmed; no success claim",
            )
        cleaned = await _cleanup_run(plan.local_dir, run_id)
        return ExportResult(
            "FAILED" if cleaned else "FAILED_CLEANUP_INCOMPLETE",
            (),
            lost_items(message),
            None,
            message if cleaned else message + "; namespace cleanup incomplete",
        )

    spawn_coro = asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "scripts.p1.clore_export",
        "--internal-copy",
        cwd=str(ROOT),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        limit=64 * 1024,
    )
    process, spawn_state = await _spawn_owned(
        spawn_coro,
        timeout=max(0.0, monotonic_deadline - time.monotonic()),
        root=plan.local_dir,
        run_id=run_id,
    )
    if process is None:
        if spawn_state == "pending":
            return ExportResult(
                "FAILED_SPAWN_PENDING",
                (),
                lost_items("worker spawn pending; late ownership registered"),
                None,
                "worker spawn pending; no success claim",
            )
        return ExportResult(
            "FAILED_SPAWN",
            (),
            lost_items("worker spawn failed"),
            None,
            "worker spawn failed; no success claim",
        )

    try:
        if process.stdin is None or process.stdout is None:
            return await stop_outcome("worker pipes are unavailable")
        drain = _StdoutDrain()
        drain_task = asyncio.create_task(drain.run(process.stdout))
        tasks.add(drain_task)
        proc_task = asyncio.create_task(process.wait())
        tasks.add(proc_task)
        overflow_task = asyncio.create_task(drain.overflow_event.wait())
        tasks.add(overflow_task)
        first_task = asyncio.create_task(drain.first_event.wait())
        tasks.add(first_task)

        try:
            process.stdin.write(job_bytes)
            await asyncio.wait_for(
                process.stdin.drain(),
                timeout=max(0.0, monotonic_deadline - time.monotonic()),
            )
            process.stdin.close()
        except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
            return await stop_outcome("worker stdin closed before completion")

        started = False
        while True:
            remaining = max(0.0, monotonic_deadline - time.monotonic())
            if remaining <= 0:
                return await stop_outcome("export worker deadline expired")
            wait_set: set[asyncio.Task[Any]] = {drain_task, proc_task, overflow_task}
            if not started:
                wait_set.add(first_task)
            done, _pending = await asyncio.wait(
                wait_set,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return await stop_outcome("export worker deadline expired")
            if overflow_task in done:
                return await stop_outcome("worker result exceeded the bound")
            if not started:
                first = drain.first_payload
                if not (
                    isinstance(first, Mapping)
                    and first.get("event") == "STARTED"
                    and first.get("schema") == EXPORT_SCHEMA
                    and first.get("run_id") == run_id
                ):
                    return await stop_outcome(
                        "worker did not start with the strict STARTED frame"
                    )
                started = True
                if on_started is not None:
                    on_started()
            if drain_task in done:
                if not proc_task.done():
                    remaining = max(0.0, monotonic_deadline - time.monotonic())
                    done2, _pending2 = await asyncio.wait(
                        {proc_task}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                    )
                    if not done2:
                        return await stop_outcome("export worker deadline expired")
                break
            if proc_task in done and not drain_task.done():
                # Keep draining; the next bounded loop iteration observes EOF.
                continue

        payload, frame_error = _parse_stdout_frames(drain)
        if frame_error is not None:
            return await stop_outcome(frame_error)
        assert payload is not None
        validation_error = _validate_result_payload(payload, plan, run_id)
        if validation_error is not None:
            return await stop_outcome(validation_error)
        status = payload["status"]
        completed = tuple(dict(item) for item in payload["completed"])
        lost = tuple(dict(item) for item in payload["lost"])
        manifest_path = Path(payload["manifest_path"])
        run_dir = Path(payload["run_dir"])
        return ExportResult(
            status,
            completed,
            lost,
            manifest_path,
            str(payload.get("message") or "worker result"),
            run_dir,
        )
    except asyncio.CancelledError:
        reaped = await _kill_and_reap(process)
        if reaped:
            await _cleanup_run(plan.local_dir, run_id)
        await cancel_tasks()
        raise
    except BaseException:
        return await stop_outcome("worker startup failed")
    finally:
        await cancel_tasks()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/p1/clore_export.py",
        description=(
            "Plan/render a local staged Clore export or run the internal local "
            "copy/cleanup worker. No SSH/remote fetch/provider credentials."
        ),
    )
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--internal-copy", action="store_true", default=False)
    parser.add_argument("--cleanup-namespace", nargs=2, default=None, metavar=("ROOT", "RUN_ID"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.internal_copy:
        return internal_copy_main()
    if args.cleanup_namespace is not None:
        return cleanup_namespace_main(*args.cleanup_namespace)
    try:
        if args.local_dir is None:
            raise ExportError("--local-dir is required for a plan")
        plan = build_export_plan(Path(args.local_dir), source_dir=args.source_dir)
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "CLORE_EXPORT_PLAN",
                        "local_dir": str(plan.local_dir),
                        "source_dir": str(plan.source_dir),
                        "items": [
                            {
                                "source": item.source,
                                "target": item.target,
                                "required": item.required,
                            }
                            for item in plan.items
                        ],
                        "note": "local staged plan only; no SSH/copy/secret read",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print("# Clore local staged export plan (not executed)")
            for item in plan.items:
                print(f"# {item.source} -> {item.target}")
        return 0
    except ExportError as exc:
        print(f"export plan refused: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
