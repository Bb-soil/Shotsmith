"""Owned I1 downloader, staged under the frozen launcher-compatible name.

The file is intentionally standard-library only (plus the staged setup_plan and
profile modules).  It downloads exactly the pinned profile files, keeps a
single target lock, writes heartbeat/progress and never creates a second copy.

The orchestrator remains the owner of attempts, deadlines, resume policy and
target identity.  This worker only executes one already-registered attempt.
Progress state is updated in memory by download threads; a dedicated writer
thread persists progress/status, and a separate heartbeat thread persists only
heartbeat.json.  Network callbacks never perform fsync or file IO.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping

try:  # staged module names
    import i1_profile as profile_module
    import i1_util as util
    import setup_plan
except ImportError:  # pragma: no cover
    from scripts.i1 import profile as profile_module  # type: ignore
    from scripts.i1 import util  # type: ignore
    from scripts.p1 import setup_plan  # type: ignore

JOB_SCHEMA = "anime-video-studio/i1-download-job/v1"
STATUS_SCHEMA = "anime-video-studio/i1-download-status/v1"
PROGRESS_SCHEMA = "anime-video-studio/i1-download-progress/v1"
HEARTBEAT_SCHEMA = "anime-video-studio/i1-download-heartbeat/v1"
LOCK_SCHEMA = "anime-video-studio/i1-target-lock/v1"

MAX_CONCURRENT_DOWNLOADS = 2
PROGRESS_INTERVAL_SECONDS = 1.0
HEARTBEAT_INTERVAL_SECONDS = 2.0
SPEED_WINDOW_SECONDS = 5.0
DOWNLOAD_IO_TIMEOUT_SECONDS = 5.0
TERMINAL_WAIT_SECONDS = 10.0


class WorkerError(Exception):
    pass


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    util.atomic_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return util.read_json(path)


def _sha_file(path: Path) -> str:
    return util.sha256_file(path)


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


def _assert_disk(workspace: Path, required: int) -> None:
    if required < 0:
        raise WorkerError("required bytes are invalid")
    if not workspace.is_dir():
        raise WorkerError("workspace is missing")
    import shutil

    if required and shutil.disk_usage(workspace).free < required:
        raise WorkerError("insufficient disk before download")


def _remaining_need(profile: Any, comfy_root: Path) -> int:
    """Future allocation only: existing verified-size finals are not copied again."""

    future = 0
    for item in profile.models:
        target = comfy_root / Path(str(item["target"]))
        final = target
        part = target.with_name(target.name + ".part")
        expected = int(item["size_bytes"])
        if final.is_file() and final.stat().st_size == expected:
            continue
        if part.is_file() and 0 <= part.stat().st_size <= expected:
            future += expected - part.stat().st_size
        else:
            future += expected
    return future + int(profile.data["disk"]["install_overhead_bytes"]) + int(
        profile.data["disk"]["reserve_bytes"]
    )


def _canonical_target(comfy_root: Path, target: str) -> str:
    root = comfy_root.resolve()
    candidate = (comfy_root / Path(target)).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise WorkerError(f"download target escapes ComfyUI root: {target!r}")
    return os.path.normcase(str(candidate)) if os.name == "nt" else str(candidate)


def _validate_specs(specs: list[Any], comfy_root: Path) -> None:
    ids: set[str] = set()
    targets: set[str] = set()
    canonical: set[str] = set()
    for spec in specs:
        model_id = str(spec.model_id)
        target = str(spec.target)
        if not model_id or model_id in ids:
            raise WorkerError("download model ids must be non-empty and unique")
        if not target or target in targets:
            raise WorkerError("download targets must be non-empty and unique")
        key = _canonical_target(comfy_root, target)
        if key in canonical:
            raise WorkerError("download canonical targets must be unique")
        ids.add(model_id)
        targets.add(target)
        canonical.add(key)


def _existing_offset(comfy_root: Path, target: str, expected: int) -> int:
    final = comfy_root / Path(target)
    part = final.with_name(final.name + ".part")
    try:
        if final.is_file() and final.stat().st_size == expected:
            return expected
        if part.is_file() and 0 <= part.stat().st_size <= expected:
            return int(part.stat().st_size)
    except OSError:
        return 0
    return 0


class _LockedLog:
    """Serialize append-only worker log lines from concurrent download threads."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            return self._handle.write(text)

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()


class Progress:
    """Thread-safe in-memory progress plus dedicated durable writers.

    Ordinary progress/status are coalesced to at most one snapshot per second.
    Stage, start, per-file completion, terminal error and complete events wake
    the writer immediately.  Heartbeat has its own thread and writes only
    heartbeat.json, so a slow progress fsync cannot make the worker look dead.
    """

    def __init__(
        self,
        *,
        progress_path: Path,
        status_path: Path,
        heartbeat_path: Path,
        run_id: str,
        nonce: str,
        files: int,
        total: int,
    ) -> None:
        self.progress_path = progress_path
        self.status_path = status_path
        self.heartbeat_path = heartbeat_path
        self.run_id = run_id
        self.nonce = nonce
        self.files = files
        self.total = total
        self.stage = "STARTING"
        self.current_file: str | None = None
        self.pid = os.getpid()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._progress_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._heartbeat_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._hb_wake = threading.Event()
        self._terminal_event = threading.Event()
        self._terminal: str | None = None
        self._terminal_error: str | None = None
        self._terminal_files: list[dict[str, Any]] | None = None
        self._terminal_version = 0
        self._terminal_done_version = 0
        self._events: deque[dict[str, Any]] = deque()
        self._history: list[dict[str, Any]] = []
        self._io_error: str | None = None
        self._started = False
        self._writer: threading.Thread | None = None
        self._heartbeat: threading.Thread | None = None
        self._writer_wait_count = 0
        self._writer_decision_count = 0
        self._files: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._samples: deque[tuple[float, int]] = deque()
        self._network_total = 0

    # -- in-memory updates -------------------------------------------------

    def add_file(
        self,
        model_id: str,
        target: str,
        size_bytes: int,
        *,
        initial_written: int = 0,
    ) -> None:
        with self._lock:
            if model_id in self._files or any(
                item["target"] == target for item in self._files.values()
            ):
                raise WorkerError("download model id/target is duplicated")
            if initial_written < 0 or initial_written > int(size_bytes):
                initial_written = 0
            self._files[model_id] = {
                "model_id": model_id,
                "target": target,
                "size_bytes": int(size_bytes),
                "written": int(initial_written),
                "network_written": int(initial_written),
                "network_started": False,
                "network_total": 0,
                "stage": "PENDING",
                "completed": False,
            }
            self._order.append(model_id)

    def _entry(self, model_id: str) -> dict[str, Any]:
        entry = self._files.get(model_id)
        if entry is None:
            raise WorkerError("progress callback for an unknown model")
        return entry

    def _aggregate_stage(self) -> str:
        stages = [item["stage"] for item in self._files.values() if not item["completed"]]
        if any(stage == "DOWNLOADING" for stage in stages):
            return "DOWNLOADING"
        if any(stage == "VERIFYING" for stage in stages):
            return "VERIFYING"
        if any(stage == "STARTING" for stage in stages):
            return "STARTING"
        if not stages:
            return "COMPLETE" if self._terminal == "COMPLETE" else "DONE"
        return str(stages[0])

    def _enqueue_event_locked(self, kind: str, status: str, *, version: int = 0) -> None:
        snapshot = self._snapshot(update_rate=False, status=status)
        self._events.append(
            {
                "kind": kind,
                "status": status,
                "version": version,
                "snapshot": snapshot,
            }
        )
        self._wake.set()
        self._condition.notify_all()

    def set_stage(self, stage: str, current_file: str | None = None) -> None:
        with self._lock:
            if self._terminal is not None:
                return
            if current_file is not None:
                entry = self._entry(current_file)
                entry["stage"] = str(stage)
                self.current_file = current_file
            else:
                self.stage = str(stage)
            self.stage = self._aggregate_stage()
            self._enqueue_event_locked("stage", "RUNNING")

    def callback(self, model_id: str, written: int, total: int) -> None:
        if written < 0 or total <= 0 or written > total:
            raise WorkerError("invalid progress callback")
        with self._lock:
            entry = self._entry(model_id)
            if int(total) != int(entry["size_bytes"]):
                raise WorkerError("progress total does not match the pinned size")
            entry["written"] = int(written)
            self.stage = self._aggregate_stage()
            self.current_file = model_id

    def network_progress(self, model_id: str, written: int, total: int) -> None:
        """Legacy compatibility hook; I1 uses exact network_bytes deltas."""

        if written < 0 or total <= 0 or written > total:
            raise WorkerError("invalid network progress callback")
        with self._lock:
            entry = self._entry(model_id)
            if int(total) != int(entry["size_bytes"]):
                raise WorkerError("network progress total does not match the pinned size")
            previous = int(entry["network_written"])
            if not entry["network_started"]:
                entry["network_started"] = True
                delta = int(written) - previous if int(written) >= previous else int(written)
            elif int(written) < previous:
                delta = int(written)
            else:
                delta = int(written) - previous
            entry["network_written"] = int(written)
            if delta > 0:
                entry["network_total"] = int(entry["network_total"]) + delta
                self._network_total += delta

    def network_bytes(self, model_id: str, delta: int) -> None:
        """Record exact bytes read from the network, independent of absolute offset."""

        if delta <= 0:
            return
        with self._lock:
            entry = self._entry(model_id)
            entry["network_total"] = int(entry["network_total"]) + int(delta)
            self._network_total += int(delta)

    def file_done(self, model_id: str | None = None) -> None:
        with self._lock:
            if self._terminal is not None:
                return
            if model_id is not None:
                entry = self._entry(model_id)
                entry["completed"] = True
                entry["written"] = int(entry["size_bytes"])
                entry["stage"] = "DONE"
                self.current_file = model_id
            self.stage = self._aggregate_stage()
            self._enqueue_event_locked("file_done", "RUNNING")

    def check_io(self) -> None:
        with self._lock:
            if self._io_error:
                raise WorkerError(f"state persistence failed: {self._io_error}")

    # -- snapshots ---------------------------------------------------------

    def _snapshot(self, *, update_rate: bool, status: str) -> dict[str, Any]:
        now_mono = time.monotonic()
        now_wall = time.time()
        completed_bytes = 0
        active_written = 0
        pending_written = 0
        completed_files = 0
        active: list[dict[str, Any]] = []
        active_ids: list[str] = []
        pending_ids: list[str] = []
        any_downloading = False
        for model_id in self._order:
            entry = self._files[model_id]
            if entry["completed"]:
                completed_bytes += int(entry["size_bytes"])
                completed_files += 1
                continue
            if entry["stage"] == "PENDING":
                pending_ids.append(model_id)
                pending_written += int(entry["written"])
                continue
            active_written += int(entry["written"])
            active_ids.append(model_id)
            if entry["stage"] == "DOWNLOADING":
                any_downloading = True
            active.append(
                {
                    "model_id": model_id,
                    "target": entry["target"],
                    "stage": entry["stage"],
                    "bytes_written": int(entry["written"]),
                    "size_bytes": int(entry["size_bytes"]),
                }
            )
        bytes_done = min(self.total, completed_bytes + active_written + pending_written)
        remaining = max(0, self.total - bytes_done)

        if update_rate:
            self._samples.append((now_mono, self._network_total))
            cutoff = now_mono - SPEED_WINDOW_SECONDS
            while len(self._samples) > 1 and self._samples[0][0] < cutoff:
                self._samples.popleft()

        speed: float | None = None
        eta: float | None = None
        eta_known = False
        if remaining == 0:
            speed = 0.0
            eta = 0.0
            eta_known = True
        elif any_downloading and len(self._samples) >= 2:
            first_at, first_bytes = self._samples[0]
            elapsed = now_mono - first_at
            moved = self._network_total - first_bytes
            if elapsed >= 1.0 and moved > 0:
                speed = moved / elapsed
                eta = remaining / speed
                eta_known = True
            else:
                speed = 0.0
        elif any_downloading:
            speed = 0.0

        current = (
            self.current_file
            if self.current_file in active_ids
            else (active_ids[0] if active_ids else self.current_file)
        )
        stage = "COMPLETE" if status == "COMPLETE" else self._aggregate_stage()
        return {
            "schema": PROGRESS_SCHEMA,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "status": status,
            "stage": stage,
            "current_file": current,
            "files_done": completed_files,
            "files_total": self.files,
            "bytes_done": bytes_done,
            "bytes_total": self.total,
            "bytes_remaining": remaining,
            "speed_bytes_per_sec": speed,
            "eta_seconds": eta,
            "eta_known": eta_known,
            "current_files": active_ids,
            "active_files": active,
            "files_active": len(active_ids),
            "pending_files": pending_ids,
            "files_pending": len(pending_ids),
            "heartbeat_at": now_wall,
            "pid": self.pid,
            "last_error": self._terminal_error,
        }

    def _progress_payload(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(snapshot)
        payload["schema"] = PROGRESS_SCHEMA
        payload.pop("files", None)
        return payload

    def _status_payload(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(snapshot)
        payload["schema"] = STATUS_SCHEMA
        if payload.get("status") == "COMPLETE" and self._terminal_files is not None:
            payload["files"] = list(self._terminal_files)
            payload["bytes"] = sum(int(item.get("size_bytes") or 0) for item in self._terminal_files)
        return payload

    def _heartbeat_payload(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": HEARTBEAT_SCHEMA,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "status": snapshot["status"],
            "stage": snapshot["stage"],
            "heartbeat_at": snapshot["heartbeat_at"],
            "pid": self.pid,
        }

    # -- durable writers ---------------------------------------------------

    def _note_io_error(self, exc: BaseException) -> None:
        with self._lock:
            if self._io_error is None:
                self._io_error = type(exc).__name__
            self._condition.notify_all()
        self._terminal_event.set()
        self._wake.set()
        self._hb_wake.set()

    def _event_allowed(self, event: Mapping[str, Any]) -> bool:
        with self._lock:
            if self._io_error:
                return False
            if event.get("kind") == "terminal":
                return int(event.get("version") or 0) == self._terminal_version
            # A normal snapshot may finish the already-queued sequence before
            # the terminal event, but never after terminal persistence.
            if self._terminal is not None and (
                self._terminal_done_version >= self._terminal_version
            ):
                return False
            return True

    def _write_progress_event(self, event: Mapping[str, Any]) -> None:
        if not self._event_allowed(event):
            return
        with self._progress_lock:
            if not self._event_allowed(event):
                return
            _atomic(self.progress_path, self._progress_payload(event["snapshot"]))

    def _write_status_event(self, event: Mapping[str, Any]) -> None:
        if not self._event_allowed(event):
            return
        with self._status_lock:
            if not self._event_allowed(event):
                return
            _atomic(self.status_path, self._status_payload(event["snapshot"]))

    def _write_heartbeat_event(self, event: Mapping[str, Any]) -> None:
        if not self._event_allowed(event):
            return
        with self._heartbeat_lock:
            if not self._event_allowed(event):
                return
            _atomic(self.heartbeat_path, self._heartbeat_payload(event["snapshot"]))

    def _write_event(self, event: Mapping[str, Any]) -> None:
        try:
            if event.get("kind") == "terminal":
                version = int(event.get("version") or 0)
                status = str(event.get("status") or "FAILED")
                snapshot = dict(event["snapshot"])
                snapshot["last_error"] = self._terminal_error
                terminal_event = {**event, "snapshot": snapshot}
                # Heartbeat is independent of a blocked progress fsync and is
                # allowed to publish FAILED as soon as the terminal gate is set.
                self._write_heartbeat_event(terminal_event)
                if status == "FAILED":
                    # Status is authoritative for an early visible failure;
                    # progress may still be blocked on its own file lock.
                    self._write_status_event(terminal_event)
                    self._write_progress_event(terminal_event)
                else:
                    # COMPLETE becomes visible only after progress persistence.
                    self._write_progress_event(terminal_event)
                    self._write_status_event(terminal_event)
                with self._lock:
                    self._terminal_done_version = max(
                        self._terminal_done_version, version
                    )
                self._terminal_event.set()
            else:
                # Status before progress keeps fast stage/file-done transitions
                # ordered without waiting for a slow progress write.
                self._write_status_event(event)
                self._write_progress_event(event)
            with self._lock:
                self._history.append(
                    {
                        "kind": event.get("kind"),
                        "status": event.get("status"),
                        "stage": event.get("snapshot", {}).get("stage"),
                        "version": event.get("version", 0),
                    }
                )
        except BaseException as exc:
            self._note_io_error(exc)

    def _write_ordinary(self, snapshot: Mapping[str, Any]) -> None:
        event = {"kind": "ordinary", "status": "RUNNING", "snapshot": snapshot}
        try:
            self._write_status_event(event)
            self._write_progress_event(event)
        except BaseException as exc:
            self._note_io_error(exc)

    def _queue_terminal_locked(
        self,
        status: str,
        error: str | None,
        *,
        files: list[dict[str, Any]] | None = None,
    ) -> int:
        self._terminal = status
        self._terminal_error = error
        self._terminal_files = files
        self._terminal_version += 1
        version = self._terminal_version
        self._terminal_event.clear()
        self._enqueue_event_locked("terminal", status, version=version)
        self._hb_wake.set()
        return version

    def signal_failure(self, error: str) -> int:
        """Publish FAILED intent without doing IO on the network thread."""

        with self._lock:
            if self._terminal is None:
                return self._queue_terminal_locked("FAILED", str(error))
            return self._terminal_version

    def _wait_terminal(self, version: int) -> bool:
        deadline = time.monotonic() + TERMINAL_WAIT_SECONDS
        while time.monotonic() < deadline:
            with self._lock:
                if self._terminal_done_version >= version:
                    return self._io_error is None
                if self._io_error:
                    return False
            self._terminal_event.wait(0.05)
        return False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            snapshot = self._snapshot(update_rate=False, status="RUNNING")
        try:
            with self._progress_lock:
                _atomic(self.progress_path, self._progress_payload(snapshot))
            with self._heartbeat_lock:
                _atomic(self.heartbeat_path, self._heartbeat_payload(snapshot))
            with self._status_lock:
                _atomic(self.status_path, self._status_payload(snapshot))
        except BaseException as exc:
            self._note_io_error(exc)
            raise WorkerError(f"cannot persist initial progress: {type(exc).__name__}") from exc
        self._writer = threading.Thread(
            target=self._writer_loop, name="i1-progress-writer", daemon=False
        )
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="i1-progress-heartbeat", daemon=False
        )
        self._writer.start()
        self._heartbeat.start()

    def _writer_loop(self) -> None:
        next_tick = time.monotonic() + PROGRESS_INTERVAL_SECONDS
        while True:
            events: list[dict[str, Any]] = []
            ordinary: dict[str, Any] | None = None
            with self._condition:
                self._writer_decision_count += 1
                if self._stop.is_set():
                    return
                if self._events:
                    events = list(self._events)
                    self._events.clear()
                elif (
                    self._terminal is not None
                    and self._terminal_done_version < self._terminal_version
                ):
                    # A terminal gate exists but its immutable event is not
                    # queued yet; create it under the same lock as the queue.
                    self._enqueue_event_locked(
                        "terminal", self._terminal, version=self._terminal_version
                    )
                    continue
                elif self._terminal is None and time.monotonic() >= next_tick:
                    # Capture an ordinary snapshot atomically before unlock.
                    # It is written only if no event was queued meanwhile.
                    ordinary = {
                        "kind": "ordinary",
                        "status": "RUNNING",
                        "snapshot": self._snapshot(update_rate=True, status="RUNNING"),
                    }
                    next_tick = time.monotonic() + PROGRESS_INTERVAL_SECONDS
                else:
                    # No events and no due ordinary write.  Wait for a new
                    # event/terminal/stop; after terminal persistence this is an
                    # indefinite condition wait, never a sleep(0) busy loop.
                    self._writer_wait_count += 1
                    if self._terminal is not None and (
                        self._terminal_done_version >= self._terminal_version
                    ):
                        self._condition.wait()
                    else:
                        self._condition.wait(
                            max(0.0, next_tick - time.monotonic())
                        )
                    continue
            for event in events:
                self._write_event(event)
            if ordinary is not None:
                with self._lock:
                    if self._events or self._terminal is not None:
                        ordinary = None
                if ordinary is not None:
                    self._write_ordinary(ordinary["snapshot"])
            if self._io_error:
                return

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            self._hb_wake.wait(HEARTBEAT_INTERVAL_SECONDS)
            self._hb_wake.clear()
            if self._stop.is_set():
                return
            try:
                with self._heartbeat_lock:
                    with self._lock:
                        status = self._terminal or "RUNNING"
                        snapshot = self._snapshot(update_rate=False, status=status)
                    _atomic(self.heartbeat_path, self._heartbeat_payload(snapshot))
            except BaseException as exc:
                self._note_io_error(exc)
                return
            if self._io_error:
                return

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._hb_wake.set()
        with self._lock:
            self._condition.notify_all()
        current = threading.current_thread()
        for thread in (self._writer, self._heartbeat):
            if thread is not None and thread is not current:
                thread.join()

    def _drain_without_writer(self) -> None:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        for event in events:
            self._write_event(event)

    def fail(self, error: str) -> bool:
        with self._lock:
            version = self._queue_terminal_locked("FAILED", str(error))
            writer = self._writer
        if writer is None or not writer.is_alive():
            self._drain_without_writer()
            with self._lock:
                return self._terminal_done_version >= version and self._io_error is None
        return self._wait_terminal(version)

    def complete(self, files: list[dict[str, Any]]) -> bool:
        with self._lock:
            self.stage = "COMPLETE"
            version = self._queue_terminal_locked("COMPLETE", None, files=files)
            writer = self._writer
        if writer is None or not writer.is_alive():
            self._drain_without_writer()
            with self._lock:
                return self._terminal_done_version >= version and self._io_error is None
        return self._wait_terminal(version)

    @property
    def persistence_failed(self) -> bool:
        with self._lock:
            return self._io_error is not None


class TargetLock:
    """Stable OS file lock held for the whole owned download.

    The guard file is never removed; OS flock/msvcrt releases on process death.
    The owner sidecar is retained as history and is removed only if it is ours.
    """

    def __init__(
        self,
        target: Path,
        run_id: str,
        nonce: str,
        *,
        resume_dead: bool = False,
        lock_dir: Path | None = None,
        target_id_value: str = "",
    ) -> None:
        self.target = target
        self.run_id = run_id
        self.nonce = nonce
        self.resume_dead = resume_dead
        self.target_id = target_id_value
        self.lock_dir = lock_dir or (target / ".avs-i1-download.lock")
        self.owner_path = self.lock_dir.with_name(self.lock_dir.name + ".owner.json")
        self._handle = None
        self.held = False

    def _owner(self) -> dict[str, Any] | None:
        try:
            return _read_json(self.owner_path)
        except Exception:
            return None

    def _lock_handle(self, handle) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_handle(self, handle) -> None:
        if handle.closed:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    def acquire(self) -> "TargetLock":
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_dir.is_symlink():
            raise WorkerError("target lock guard must not be a symlink")
        handle = open(self.lock_dir, "a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                self._lock_handle(handle)
            except OSError as exc:
                raise WorkerError("target downloader lock is already held") from exc
            old = self._owner()
            if old is not None:
                try:
                    os.replace(
                        self.owner_path,
                        self.owner_path.with_name(
                            self.owner_path.name + f".dead-{time.time_ns()}"
                        ),
                    )
                except OSError:
                    pass
            _atomic(
                self.owner_path,
                {
                    "schema": LOCK_SCHEMA,
                    "run_id": self.run_id,
                    "nonce": self.nonce,
                    "target_id": self.target_id,
                    "pid": os.getpid(),
                    "created_at": time.time(),
                },
            )
        except BaseException:
            self._unlock_handle(handle)
            handle.close()
            raise
        self._handle = handle
        self.held = True
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            self.held = False
            return
        try:
            owner = self._owner()
            if (
                owner is not None
                and owner.get("run_id") == self.run_id
                and owner.get("nonce") == self.nonce
                and owner.get("target_id") == self.target_id
            ):
                self.owner_path.unlink(missing_ok=True)
        finally:
            self._unlock_handle(handle)
            handle.close()
            self._handle = None
            self.held = False


def _validate_job(payload: Mapping[str, Any]) -> tuple[Path, Path, Path, Path, str, str, float, str]:
    if payload.get("schema") != JOB_SCHEMA:
        raise WorkerError("download job schema is invalid")
    run_id = payload.get("run_id")
    nonce = payload.get("nonce")
    if not isinstance(run_id, str) or not run_id or not isinstance(nonce, str) or not nonce:
        raise WorkerError("download run/nonce is missing")
    workspace = Path(str(payload.get("workspace", "")))
    owned_root = Path(str(payload.get("owned_root", "")))
    attempt_dir = Path(str(payload.get("attempt_dir", "")))
    comfy_root = Path(str(payload.get("comfy_root", "")))
    profile_path = Path(str(payload.get("profile_path", "")))
    for label, path in (
        ("workspace", workspace),
        ("owned_root", owned_root),
        ("attempt_dir", attempt_dir),
        ("comfy_root", comfy_root),
        ("profile_path", profile_path),
    ):
        if not path.is_absolute():
            raise WorkerError(f"download {label} must be absolute")
    if owned_root != attempt_dir:
        raise WorkerError("download owned root must be the launch attempt")
    if (
        attempt_dir.parent.name != "attempts"
        or attempt_dir.parent.parent.name != ".avs-i1"
        or workspace not in attempt_dir.parents
    ):
        raise WorkerError("download attempt is outside the owned workspace")
    if comfy_root.parent != workspace:
        raise WorkerError("download ComfyUI root is outside workspace")
    if profile_path.parent != attempt_dir:
        raise WorkerError("download profile is outside the attempt")
    if not profile_path.is_file():
        raise WorkerError("download profile is missing")
    deadline = payload.get("deadline_epoch")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise WorkerError("download deadline is invalid")
    profile_sha = payload.get("profile_sha256")
    if not isinstance(profile_sha, str) or len(profile_sha) != 64:
        raise WorkerError("download profile hash is invalid")
    for key in ("status_path", "progress_path", "heartbeat_path"):
        path = Path(str(payload.get(key, "")))
        if path.parent != attempt_dir or not path.is_absolute():
            raise WorkerError(f"download {key} is outside the attempt")
    target_home = Path(str(payload.get("target_home", "")))
    target_id = payload.get("target_id")
    target_lock = Path(str(payload.get("target_lock_path", "")))
    if not target_home.is_absolute() or not target_home.is_dir():
        raise WorkerError("target home is invalid")
    if not isinstance(target_id, str) or len(target_id) != 64:
        raise WorkerError("target identity is invalid")
    expected_lock = target_home / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{target_id}.lock"
    if target_lock != expected_lock:
        raise WorkerError("target lock path is outside the target namespace")
    return workspace, owned_root, comfy_root, profile_path, run_id, nonce, float(deadline), profile_sha


def _install_stop_handlers(cancel: threading.Event) -> list[tuple[int, Any]]:
    previous: list[tuple[int, Any]] = []

    def handler(signum, frame) -> None:  # noqa: ANN001
        cancel.set()

    for name in ("SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous.append((signum, signal.getsignal(signum)))
            signal.signal(signum, handler)
        except (OSError, ValueError):
            pass
    return previous


def _restore_stop_handlers(previous: list[tuple[int, Any]]) -> None:
    for signum, handler in previous:
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            pass


def _cancelled_reason(
    cancel: threading.Event,
    reason: dict[str, str],
    deadline: float,
    clock: Callable[[], float] = time.time,
) -> bool:
    if cancel.is_set():
        return True
    if clock() >= deadline:
        reason.setdefault("value", "download deadline expired")
        cancel.set()
        return True
    return False


def run(
    payload_path: Path,
    *,
    concurrency: int = MAX_CONCURRENT_DOWNLOADS,
    clock: Callable[[], float] | None = None,
) -> int:
    clock = clock or time.time
    if isinstance(concurrency, bool) or not isinstance(concurrency, int):
        raise WorkerError("download concurrency is invalid")
    if concurrency < 1 or concurrency > MAX_CONCURRENT_DOWNLOADS:
        raise WorkerError("download concurrency must be within 1..2")
    payload = _read_json(payload_path)
    (
        workspace,
        owned_root,
        comfy_root,
        profile_path,
        run_id,
        nonce,
        deadline,
        profile_sha,
    ) = _validate_job(payload)
    profile = profile_module.load_profile(profile_path)
    if profile.sha256 != profile_sha:
        raise WorkerError("download profile hash mismatch")
    specs = [setup_plan.DownloadSpec(**item) for item in profile_module.download_specs(profile)]
    if len(specs) != len(profile.models):
        raise WorkerError("profile download specs do not match the pinned models")
    _validate_specs(specs, comfy_root)
    if sum(item.size_bytes for item in specs) != int(profile.model_bytes()):
        raise WorkerError("profile model byte total is inconsistent")
    required = _remaining_need(profile, comfy_root)
    _assert_disk(workspace, required)
    if clock() >= deadline:
        raise WorkerError("download deadline has passed")
    recorder = Progress(
        progress_path=Path(str(payload["progress_path"])),
        status_path=Path(str(payload["status_path"])),
        heartbeat_path=Path(str(payload["heartbeat_path"])),
        run_id=run_id,
        nonce=nonce,
        files=len(specs),
        total=sum(item.size_bytes for item in specs),
    )
    for spec in specs:
        recorder.add_file(
            spec.model_id,
            spec.target,
            spec.size_bytes,
            initial_written=_existing_offset(comfy_root, spec.target, spec.size_bytes),
        )
    resume_dead = (
        os.environ.get("P1_RESUME_DEAD") == "1"
        or os.environ.get("I1_RESUME_DEAD") == "1"
    )
    lock = TargetLock(
        workspace,
        run_id,
        nonce,
        resume_dead=resume_dead,
        lock_dir=Path(str(payload["target_lock_path"])),
        target_id_value=str(payload["target_id"]),
    )
    log: _LockedLog | None = None
    threads: list[threading.Thread] = []
    cancel = threading.Event()
    reason: dict[str, str] = {}
    previous_handlers = _install_stop_handlers(cancel)
    try:
        lock.acquire()
        log_path = Path(str(payload.get("log", owned_root / "download.log")))
        if log_path.parent != owned_root:
            raise WorkerError("download log is outside the owned root")
        log = _LockedLog(log_path)
        # No RUNNING marker is published before ownership is proven.
        recorder.start()
        if _cancelled_reason(cancel, reason, deadline, clock):
            raise WorkerError(reason.get("value", "download cancelled"))
        _assert_disk(workspace, _remaining_need(profile, comfy_root))

        pending = list(range(len(specs)))
        pending_lock = threading.Lock()
        results: list[Any | None] = [None] * len(specs)
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def should_cancel() -> bool:
            return _cancelled_reason(cancel, reason, deadline, clock)

        def worker_loop() -> None:
            while True:
                if should_cancel():
                    return
                with pending_lock:
                    if not pending:
                        return
                    index = pending.pop(0)
                spec = specs[index]
                try:
                    recorder.set_stage("DOWNLOADING", spec.model_id)

                    def progress(written: int, total: int, model_id: str = spec.model_id) -> None:
                        recorder.check_io()
                        recorder.callback(model_id, written, total)
                        if should_cancel():
                            raise WorkerError(reason.get("value", "download cancelled"))

                    def network_bytes(
                        delta: int, model_id: str = spec.model_id
                    ) -> None:
                        recorder.network_bytes(model_id, delta)

                    def stage(name: str, model_id: str = spec.model_id) -> None:
                        recorder.check_io()
                        if should_cancel():
                            raise WorkerError(reason.get("value", "download cancelled"))
                        recorder.set_stage(name, model_id)

                    result = setup_plan.download_verified(
                        spec,
                        root=comfy_root,
                        progress=progress,
                        network_bytes=network_bytes,
                        stage=stage,
                        log=log,
                        cancel=should_cancel,
                        network_timeout=DOWNLOAD_IO_TIMEOUT_SECONDS,
                    )
                    recorder.file_done(spec.model_id)
                    with pending_lock:
                        results[index] = result
                except BaseException as exc:
                    with errors_lock:
                        if not errors:
                            errors.append(exc)
                    reason.setdefault("value", "download cancelled")
                    cancel.set()
                    # Queue the first failure immediately; the writer publishes
                    # FAILED while sibling threads are still being drained.  No
                    # state IO is started on this socket/network thread.
                    recorder.signal_failure(type(exc).__name__)
                    return

        for slot in range(min(concurrency, len(specs))):
            thread = threading.Thread(
                target=worker_loop, name=f"i1-download-{slot + 1}", daemon=False
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        threads.clear()

        if errors:
            exc = next(
                (
                    item
                    for item in errors
                    if not isinstance(item, setup_plan.DownloadCancelled)
                ),
                errors[0],
            )
            if reason.get("value") == "download deadline expired":
                raise WorkerError("download deadline expired") from exc
            raise exc
        if any(result is None for result in results):
            raise WorkerError("download tasks ended without a result")
        if _cancelled_reason(cancel, reason, deadline, clock):
            raise WorkerError(reason.get("value", "download cancelled"))
        final_results = [
            {
                "model_id": spec.model_id,
                "target": spec.target,
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
                "skipped": result.skipped,
            }
            for spec, result in zip(specs, results)
        ]
        if not recorder.complete(final_results):
            raise WorkerError(
                f"state persistence failed: {recorder._io_error or 'terminal flush'}"
            )
        return 0
    except BaseException as exc:
        reason.setdefault("value", "worker failed")
        cancel.set()
        # Drain every non-daemon download thread before releasing ownership.
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join()
        try:
            message = util.redact(str(exc) or type(exc).__name__, limit=300)
            recorder.fail(message)
        except Exception:
            pass
        raise
    finally:
        try:
            recorder.stop()
        finally:
            if log is not None:
                try:
                    log.flush()
                finally:
                    log.close()
            lock.release()
            _restore_stop_handlers(previous_handlers)


def _record_early_failure(payload_path: Path, exc: BaseException) -> None:
    try:
        payload = _read_json(payload_path)
        run_id = payload.get("run_id")
        nonce = payload.get("nonce")
        owned_root = Path(str(payload.get("owned_root", "")))
        status_path = Path(str(payload.get("status_path", "")))
        progress_path = Path(str(payload.get("progress_path", "")))
        heartbeat_path = Path(str(payload.get("heartbeat_path", "")))
        if (
            not isinstance(run_id, str)
            or not run_id
            or not isinstance(nonce, str)
            or not nonce
        ):
            return
        for path, parent in (
            (status_path, owned_root),
            (progress_path, owned_root),
            (heartbeat_path, owned_root),
        ):
            if not path.is_absolute() or path.parent != parent:
                return
        if status_path.exists():
            return
        now = time.time()
        base = {
            "run_id": run_id,
            "nonce": nonce,
            "status": "FAILED",
            "stage": "FAILED",
            "error": type(exc).__name__,
            "last_error": type(exc).__name__,
            "heartbeat_at": now,
            "pid": os.getpid(),
            "files_done": 0,
            "files_total": int(payload.get("files_total") or 0),
            "bytes_done": 0,
            "bytes_total": int(payload.get("required_bytes") or 0),
            "bytes_remaining": int(payload.get("required_bytes") or 0),
            "current_file": None,
            "speed_bytes_per_sec": None,
            "eta_seconds": None,
            "eta_known": False,
            "current_files": [],
            "active_files": [],
            "files_active": 0,
        }
        _atomic(status_path, {"schema": STATUS_SCHEMA, **base})
        _atomic(progress_path, {"schema": PROGRESS_SCHEMA, **base})
        _atomic(
            heartbeat_path,
            {
                "schema": HEARTBEAT_SCHEMA,
                "run_id": run_id,
                "nonce": nonce,
                "status": "FAILED",
                "stage": "FAILED",
                "heartbeat_at": now,
                "pid": os.getpid(),
            },
        )
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: v3_download_worker.py <payload.json>", file=sys.stderr)
        return 2
    payload_path = Path(args[0])
    try:
        return run(payload_path)
    except KeyboardInterrupt:
        _record_early_failure(payload_path, KeyboardInterrupt())
        print("download worker interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        _record_early_failure(payload_path, exc)
        print(f"download worker failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
