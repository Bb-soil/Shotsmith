"""Focused offline lifecycle helpers for P1 v3.

This module is intentionally small: immutable session binding, owner/nonce
records, target-level downloader lock, supervisor process record, host-key
fingerprint verification, deadline/budget/disk checks and a progress recorder.
It does not start any network/provider/SSH activity by itself; callers must pass
explicit fake runners/process launchers in tests and explicit live gates in
production.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from anime_studio.errors import InstanceAlreadyRunning
from anime_studio.lock import InstanceLock
from scripts.p1 import clore_session, owner_risk
from scripts.p1.remote import (
    RemoteError,
    RemoteIdentity,
    RemoteRunner,
    RemoteStreamPartialError,
    _validate_remote_absolute_path,
    run_bounded_process,
)

SCHEMA = "anime-video-studio/p1-v3-lifecycle/v1"
SUPERVISOR_SCHEMA = "anime-video-studio/p1-v3-supervisor/v1"
SUPERVISOR_ATTEMPT_SCHEMA = "anime-video-studio/p1-v3-supervisor-attempt/v1"
TARGET_LOCK_SCHEMA = "anime-video-studio/p1-v3-target-lock/v1"
PROGRESS_SCHEMA = "anime-video-studio/p1-v3-download-progress/v1"
LAUNCH_SCHEMA = "anime-video-studio/p1-v3-launch/v1"
PROCESS_OWNER_SCHEMA = "anime-video-studio/p1-v3-process-owner/v1"
READ_SCHEMA = "anime-video-studio/p1-v3-read/v1"
REMOTE_STATE_SCHEMA = "anime-video-studio/p1-v3-remote-state/v1"


class LifecycleError(Exception):
    """A lifecycle/budget/ownership check failed closed."""


class SupervisorPending(LifecycleError):
    """A supervisor STARTING attempt is alive but has not acked yet.

    The attempt is owned and must be resumed; the caller must not spawn a
    second child.
    """


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
        newline="\n",
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot read lifecycle JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LifecycleError("lifecycle JSON must be an object")
    return payload


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        checker = getattr(path, "is_junction", None)
        return bool(checker and checker())
    except OSError:
        return True


def assert_safe_local_path(
    path: Path,
    *,
    root: Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Reject symlink/junction/non-regular local paths before reading/writing."""

    candidate = Path(path)
    if root is not None:
        root_resolved = Path(root).resolve()
        resolved = candidate.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise LifecycleError(f"local path escapes the owned root: {candidate}")
    current = candidate
    while True:
        if current.exists() or current.is_symlink():
            if _is_link(current):
                raise LifecycleError(f"local path has a symlink/junction component: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if candidate.exists():
        if not candidate.is_file():
            raise LifecycleError(f"local path is not a regular file: {candidate}")
    elif must_exist:
        raise LifecycleError(f"local path is missing: {candidate}")
    return candidate


def _hash_prefix(path: Path, *, limit: int) -> tuple[int, str]:
    """Stream-hash at most ``limit`` bytes; never read the whole file."""

    if limit < 0:
        raise LifecycleError("prefix limit is invalid")
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while read < limit:
            chunk = handle.read(min(1024 * 1024, limit - read))
            if not chunk:
                break
            read += len(chunk)
            digest.update(chunk)
    return read, digest.hexdigest()


def read_remote_json(
    *,
    runner: Any,
    identity: RemoteIdentity,
    helper_path: str,
    remote_path: str,
    remote_root: str,
    max_bytes: int = 256 * 1024,
    timeout: float = 15.0,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Read one fresh remote JSON file through the staged bounded helper."""

    helper = _validate_remote_absolute_path(helper_path, "helper_path")
    remote = _validate_remote_absolute_path(remote_path, "remote_path")
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if PurePosixPath(helper).name != "v3_remote_fetch_helper.py":
        raise LifecycleError("remote read requires the pinned helper")
    if not hasattr(runner, "read_remote_exact_json"):
        raise LifecycleError("runner does not support bounded remote reads")
    return runner.read_remote_exact_json(
        identity,
        helper_path=helper,
        remote_path=remote,
        root=root,
        max_bytes=max_bytes,
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LifecycleError(f"{name} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        raise LifecycleError(f"{name} must be finite and non-negative")
    return number


def clip_remote_timeout(deadline_epoch: float, requested: float) -> float:
    """Clip one remote call to the absolute window; expiry refuses the call."""

    if isinstance(deadline_epoch, bool) or not isinstance(deadline_epoch, (int, float)):
        raise LifecycleError("remote deadline is invalid")
    remaining = float(deadline_epoch) - time.time()
    if remaining <= 0:
        raise LifecycleError("session deadline expired before a remote call")
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        raise LifecycleError("remote timeout is invalid")
    if float(requested) <= 0:
        raise LifecycleError("remote timeout must be positive")
    return max(0.05, min(float(requested), remaining))


@dataclass(frozen=True)
class SessionBinding:
    order_id: int
    server_id: int
    created_at: int
    currency: str
    cap_usd: str
    duration_seconds: int
    planned_cancel_at: int
    hard_end_at: int
    quote_hash: str
    owner_risk_admission_sha256: str | None = None

    @classmethod
    def from_state(cls, state: clore_session.CloreSessionState) -> "SessionBinding":
        if not isinstance(state, clore_session.CloreSessionState):
            raise LifecycleError("session state is not CloreSessionState")
        if state.cap_usd <= 0:
            raise LifecycleError("session cap must be positive")
        if state.duration_seconds <= 0 or state.duration_seconds > 3600:
            raise LifecycleError("session duration is outside the allowed range")
        if state.planned_cancel_at < state.created_at:
            raise LifecycleError("planned cancel precedes creation")
        if state.hard_end_at <= state.created_at:
            raise LifecycleError("hard end must be after creation")
        if state.planned_cancel_at > state.hard_end_at:
            raise LifecycleError("planned cancel must not exceed hard end")
        admission_sha = state.owner_risk_sha256
        if state.owner_risk is not None and not isinstance(admission_sha, str):
            raise LifecycleError("state owner_risk requires a digest")
        if admission_sha is not None and (
            len(admission_sha) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in admission_sha)
        ):
            raise LifecycleError("state owner_risk digest is invalid")
        payload = clore_session.binding_payload(state)
        return cls(
            order_id=payload["order_id"],
            server_id=payload["server_id"],
            created_at=payload["created_at"],
            currency=payload["currency"],
            cap_usd=payload["cap_usd"],
            duration_seconds=payload["duration_seconds"],
            planned_cancel_at=payload["planned_cancel_at"],
            hard_end_at=payload["hard_end_at"],
            quote_hash=payload["quote_hash"],
            owner_risk_admission_sha256=payload.get(
                "owner_risk_admission_sha256"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "order_id": self.order_id,
            "server_id": self.server_id,
            "created_at": self.created_at,
            "currency": self.currency,
            "cap_usd": self.cap_usd,
            "duration_seconds": self.duration_seconds,
            "planned_cancel_at": self.planned_cancel_at,
            "hard_end_at": self.hard_end_at,
            "quote_hash": self.quote_hash,
        }
        if self.owner_risk_admission_sha256 is not None:
            payload["owner_risk_admission_sha256"] = (
                self.owner_risk_admission_sha256
            )
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionBinding":
        required = {
            "order_id",
            "server_id",
            "created_at",
            "currency",
            "cap_usd",
            "duration_seconds",
            "planned_cancel_at",
            "hard_end_at",
            "quote_hash",
        }
        optional = {"owner_risk_admission_sha256"}
        if (
            not isinstance(data, Mapping)
            or not required.issubset(data)
            or set(data) - required - optional
        ):
            raise LifecycleError("session binding is incomplete/unknown")
        if any(isinstance(data[key], bool) or not isinstance(data[key], int) for key in ("order_id", "server_id", "created_at", "duration_seconds", "planned_cancel_at", "hard_end_at")):
            raise LifecycleError("session binding integer fields are invalid")
        if not isinstance(data["currency"], str) or not data["currency"]:
            raise LifecycleError("session currency is invalid")
        if not isinstance(data["cap_usd"], str) or not data["cap_usd"]:
            raise LifecycleError("session cap is invalid")
        if not isinstance(data["quote_hash"], str) or len(data["quote_hash"]) != 64:
            raise LifecycleError("session quote hash is invalid")
        owner_risk_sha = data.get("owner_risk_admission_sha256")
        if owner_risk_sha is not None and (
            not isinstance(owner_risk_sha, str)
            or len(owner_risk_sha) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in owner_risk_sha)
        ):
            raise LifecycleError("session owner_risk digest is invalid")
        return cls(
            order_id=data["order_id"],
            server_id=data["server_id"],
            created_at=data["created_at"],
            currency=data["currency"],
            cap_usd=data["cap_usd"],
            duration_seconds=data["duration_seconds"],
            planned_cancel_at=data["planned_cancel_at"],
            hard_end_at=data["hard_end_at"],
            quote_hash=data["quote_hash"],
            owner_risk_admission_sha256=(
                owner_risk_sha.lower() if isinstance(owner_risk_sha, str) else None
            ),
        )

    def assert_matches_state(self, state: clore_session.CloreSessionState) -> None:
        current = SessionBinding.from_state(state)
        if current != self:
            raise LifecycleError("session identity/budget/deadline mismatch")


def load_session_binding(
    state_path: Path,
    *,
    expected_order_id: int,
    expected_created_at: int,
) -> tuple[clore_session.CloreSessionState, SessionBinding]:
    try:
        state = clore_session.load_state(state_path)
    except Exception as exc:
        raise LifecycleError(f"cannot load Clore session state: {type(exc).__name__}") from exc
    if state.order_id != expected_order_id or state.created_at != expected_created_at:
        raise LifecycleError("session identity does not match exact order/created_at")
    binding = SessionBinding.from_state(state)
    return state, binding


def assert_deadline(binding: SessionBinding, *, now: float | None = None) -> None:
    current = time.time() if now is None else float(now)
    if current >= binding.hard_end_at:
        raise LifecycleError("absolute session deadline has passed")
    if current >= binding.planned_cancel_at:
        raise LifecycleError("planned cancellation deadline has passed")


def assert_budget(
    binding: SessionBinding,
    state: clore_session.CloreSessionState,
    *,
    now: datetime | None = None,
) -> None:
    cap = Decimal(binding.cap_usd)
    if cap <= 0:
        raise LifecycleError("session cap must be positive")
    try:
        if state.owner_risk is not None:
            issues = owner_risk.validate_admission(
                state.owner_risk, now=now, allow_expired=False
            )
            if issues:
                raise LifecycleError(
                    "owner-risk admission is invalid: " + "; ".join(issues)
                )
            projected = owner_risk.effective_projection(
                state.quote, state.owner_risk, minutes=60
            )
            if projected is None:
                raise LifecycleError("owner-risk budget projection is incomplete")
        else:
            projected = state.quote.project_total(60)
    except Exception as exc:
        if isinstance(exc, LifecycleError):
            raise
        raise LifecycleError(f"cannot project session budget: {type(exc).__name__}") from exc
    if projected is None or projected > cap:
        raise LifecycleError("projected session budget exceeds immutable cap")
    if state.cap_usd != cap:
        raise LifecycleError("state cap differs from immutable lifecycle cap")
    if state.owner_risk_sha256 is not None and binding.owner_risk_admission_sha256 != state.owner_risk_sha256:
        raise LifecycleError("session owner_risk digest differs from state")


def required_download_bytes(manifest: Mapping[str, Any]) -> int:
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise LifecycleError("manifest models are missing")
    total = 0
    for model in models:
        size = model.get("size_bytes") if isinstance(model, Mapping) else None
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise LifecycleError("manifest model size is invalid")
        total += size
    return total


def assert_disk(workspace: Path | str, required_bytes: int, *, reserve_bytes: int = 0) -> None:
    if required_bytes <= 0:
        raise LifecycleError("required download bytes must be positive")
    path = Path(workspace)
    if not path.exists():
        raise LifecycleError("download workspace is unavailable")
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        raise LifecycleError("cannot read workspace disk usage") from exc
    if usage.free < required_bytes + reserve_bytes:
        raise LifecycleError("insufficient disk for downloads")


@dataclass(frozen=True)
class LifecycleState:
    schema: str
    run_id: str
    nonce: str
    session: SessionBinding
    bundle_hashes: Mapping[str, str]
    phase: str
    created_at: str
    heartbeat_at: str
    supervisor: Mapping[str, Any] | None = None
    target_lock: Mapping[str, Any] | None = None
    downloader: Mapping[str, Any] | None = None
    pipeline: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "session": self.session.to_dict(),
            "bundle_hashes": dict(self.bundle_hashes),
            "phase": self.phase,
            "created_at": self.created_at,
            "heartbeat_at": self.heartbeat_at,
            "supervisor": dict(self.supervisor) if self.supervisor else None,
            "target_lock": dict(self.target_lock) if self.target_lock else None,
            "downloader": dict(self.downloader) if self.downloader else None,
            "pipeline": dict(self.pipeline) if self.pipeline else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LifecycleState":
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
            raise LifecycleError("lifecycle schema is invalid")
        for key in ("run_id", "nonce", "phase", "created_at", "heartbeat_at"):
            if not isinstance(data.get(key), str) or not data.get(key):
                raise LifecycleError(f"lifecycle {key} is invalid")
        bundle_hashes = data.get("bundle_hashes")
        if not isinstance(bundle_hashes, Mapping) or not bundle_hashes:
            raise LifecycleError("lifecycle bundle hashes are missing")
        normalized_bundle = {}
        for key, value in bundle_hashes.items():
            if not isinstance(key, str) or not isinstance(value, str) or len(value) != 64:
                raise LifecycleError("lifecycle bundle hash is invalid")
            normalized_bundle[key] = value
        session = SessionBinding.from_dict(data.get("session") or {})
        for key in ("supervisor", "target_lock", "downloader", "pipeline"):
            value = data.get(key)
            if value is not None and not isinstance(value, Mapping):
                raise LifecycleError(f"lifecycle {key} must be an object/null")
        return cls(
            schema=SCHEMA,
            run_id=data["run_id"],
            nonce=data["nonce"],
            session=session,
            bundle_hashes=normalized_bundle,
            phase=data["phase"],
            created_at=data["created_at"],
            heartbeat_at=data["heartbeat_at"],
            supervisor=data.get("supervisor"),
            target_lock=data.get("target_lock"),
            downloader=data.get("downloader"),
            pipeline=data.get("pipeline"),
        )


def new_lifecycle(
    *,
    run_id: str,
    session: SessionBinding,
    bundle_hashes: Mapping[str, str],
    phase: str = "BOUND_OFFLINE",
) -> LifecycleState:
    now = datetime.now(timezone.utc).isoformat()
    return LifecycleState(
        schema=SCHEMA,
        run_id=run_id,
        nonce=uuid.uuid4().hex,
        session=session,
        bundle_hashes=dict(bundle_hashes),
        phase=phase,
        created_at=now,
        heartbeat_at=now,
    )


def write_lifecycle(path: Path, state: LifecycleState) -> None:
    _atomic_write(path, _json_text(state.to_dict()))


def load_lifecycle(path: Path) -> LifecycleState:
    return LifecycleState.from_dict(_read_json(path))


def update_phase(
    path: Path,
    state: LifecycleState,
    *,
    phase: str,
    supervisor: Mapping[str, Any] | None = None,
    target_lock: Mapping[str, Any] | None = None,
    downloader: Mapping[str, Any] | None = None,
    pipeline: Mapping[str, Any] | None = None,
) -> LifecycleState:
    updated = LifecycleState(
        schema=state.schema,
        run_id=state.run_id,
        nonce=state.nonce,
        session=state.session,
        bundle_hashes=state.bundle_hashes,
        phase=phase,
        created_at=state.created_at,
        heartbeat_at=datetime.now(timezone.utc).isoformat(),
        supervisor=supervisor if supervisor is not None else state.supervisor,
        target_lock=target_lock if target_lock is not None else state.target_lock,
        downloader=downloader if downloader is not None else state.downloader,
        pipeline=pipeline if pipeline is not None else state.pipeline,
    )
    write_lifecycle(path, updated)
    return updated


@dataclass
class TargetLock:
    """Target-level local lock with an explicit owner record."""

    target: Path
    run_id: str
    nonce: str
    _instance_lock: InstanceLock | None = None

    @property
    def path(self) -> Path:
        return self.target / ".p1-download.lock"

    @property
    def owner_path(self) -> Path:
        return self.target / ".p1-download.owner.json"

    def acquire(self) -> "TargetLock":
        self.target.mkdir(parents=True, exist_ok=True)
        try:
            lock = InstanceLock(self.path).acquire()
        except InstanceAlreadyRunning as exc:
            raise LifecycleError("target downloader lock is already held") from exc
        try:
            if self.owner_path.exists():
                existing = _read_json(self.owner_path)
                if existing.get("run_id") != self.run_id or existing.get("nonce") != self.nonce:
                    raise LifecycleError("target downloader lock is owned by another run")
            _atomic_write(
                self.owner_path,
                _json_text(
                    {
                        "schema": TARGET_LOCK_SCHEMA,
                        "run_id": self.run_id,
                        "nonce": self.nonce,
                        "pid": os.getpid(),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
            )
        except BaseException:
            lock.release()
            raise
        self._instance_lock = lock
        return self

    def assert_owned(self) -> None:
        if self._instance_lock is None:
            raise LifecycleError("target lock is not held by this process")
        if not self.owner_path.is_file():
            raise LifecycleError("target lock owner record is missing")
        owner = _read_json(self.owner_path)
        if owner.get("run_id") != self.run_id or owner.get("nonce") != self.nonce:
            raise LifecycleError("target lock owner identity mismatch")

    def release(self) -> None:
        try:
            if self.owner_path.exists():
                owner = _read_json(self.owner_path)
                if owner.get("run_id") == self.run_id and owner.get("nonce") == self.nonce:
                    self.owner_path.unlink(missing_ok=True)
        finally:
            if self._instance_lock is not None:
                self._instance_lock.release()
                self._instance_lock = None


@dataclass
class ProgressRecorder:
    path: Path
    run_id: str
    nonce: str
    files_total: int
    bytes_total: int
    files_done: int = 0
    bytes_done: int = 0
    last_error: str | None = None

    def callback(self, written: int, total: int) -> None:
        if written < 0 or total <= 0 or written > total:
            raise LifecycleError("invalid download progress counters")
        self.bytes_done = written
        self._write("RUNNING")

    def file_complete(self) -> None:
        self.files_done += 1
        self._write("RUNNING")

    def fail(self, message: str) -> None:
        self.last_error = message
        self._write("FAILED")

    def complete(self) -> None:
        self._write("COMPLETE")

    def _write(self, status: str) -> None:
        _atomic_write(
            self.path,
            _json_text(
                {
                    "schema": PROGRESS_SCHEMA,
                    "run_id": self.run_id,
                    "nonce": self.nonce,
                    "status": status,
                    "files_done": self.files_done,
                    "files_total": self.files_total,
                    "bytes_done": self.bytes_done,
                    "bytes_total": self.bytes_total,
                    "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                    "last_error": self.last_error,
                }
            ),
        )


@dataclass(frozen=True)
class SupervisorRecord:
    run_id: str
    nonce: str
    pid: int
    state_path: str
    quote_path: str
    log_path: str
    started_at: str
    order_id: int
    created_at: int
    config_sha256: str | None = None
    binding_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SUPERVISOR_SCHEMA,
            "run_id": self.run_id,
            "nonce": self.nonce,
            "pid": self.pid,
            "state_path": self.state_path,
            "quote_path": self.quote_path,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "order_id": self.order_id,
            "created_at": self.created_at,
        }
        if self.config_sha256 is not None:
            payload["config_sha256"] = self.config_sha256
        if self.binding_sha256 is not None:
            payload["binding_sha256"] = self.binding_sha256
        return payload


def _require_deadline_remaining(
    deadline_monotonic: float | None, label: str
) -> float | None:
    if deadline_monotonic is None:
        return None
    if isinstance(deadline_monotonic, bool) or not isinstance(
        deadline_monotonic, (int, float)
    ):
        raise LifecycleError(f"{label} deadline is invalid")
    remaining = float(deadline_monotonic) - time.monotonic()
    if remaining <= 0:
        raise LifecycleError(f"{label} deadline expired before any work")
    return remaining


def _pid_live(pid: int) -> bool:
    """Windows-safe liveness check that also verifies the process is running."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(code)
                ):
                    return False
                return code.value == STILL_ACTIVE
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


def _binding_sha256(binding: SessionBinding) -> str:
    return _sha_text(
        json.dumps(binding.to_dict(), sort_keys=True, separators=(",", ":"))
    )


class CloreSupervisor:
    """Owned independent Clore watch subprocess with fresh ack/heartbeat.

    The STARTING attempt is written atomically before the child is spawned, and
    the whole decision (existing attempt, spawn, ack wait, ownership record) is
    made under one OS lock. A timeout with a live child retains the same
    STARTING attempt for a bounded resume instead of leaving an unowned child
    or spawning a duplicate. A dead attempt is never auto-replaced; recovery is
    explicit.
    """

    def __init__(
        self,
        *,
        session_state_path: Path,
        quote_path: Path,
        binding: SessionBinding,
        log_path: Path,
        record_path: Path,
        run_id: str,
        nonce: str,
        popen: Callable[..., Any] | None = None,
        child_argv: Sequence[str] | None = None,
    ) -> None:
        self.session_state_path = session_state_path
        self.quote_path = quote_path
        self.binding = binding
        self.log_path = log_path
        self.record_path = record_path
        self.run_id = run_id
        self.nonce = nonce
        self._popen = popen or subprocess.Popen
        self._child_argv = tuple(child_argv) if child_argv is not None else None
        self.config_path = record_path.with_name("supervisor-config.json")
        self.ack_path = record_path.with_name(f"supervisor-ack-{nonce}.json")
        self.heartbeat_path = record_path.with_name(f"supervisor-heartbeat-{nonce}.json")
        self.attempt_path = record_path.with_name("supervisor-attempt.json")
        self.operation_lock_path = record_path.with_name("supervisor-operation.lock")

    def build_argv(self) -> tuple[str, ...]:
        if self._child_argv is not None:
            return self._child_argv
        return (
            sys.executable,
            "scripts/p1/v3_supervisor_child.py",
            "--config",
            str(self.config_path),
        )

    def _config_payload(self) -> dict[str, Any]:
        return {
            "schema": "anime-video-studio/p1-v3-supervisor-config/v1",
            "run_id": self.run_id,
            "nonce": self.nonce,
            "state_path": str(self.session_state_path),
            "quote_path": str(self.quote_path),
            "ack_path": str(self.ack_path),
            "heartbeat_path": str(self.heartbeat_path),
            "session": self.binding.to_dict(),
        }

    def _write_attempt(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        value["schema"] = SUPERVISOR_ATTEMPT_SCHEMA
        _atomic_write(self.attempt_path, _json_text(value))
        return value

    def _read_attempt(self) -> dict[str, Any] | None:
        if not self.attempt_path.is_file():
            return None
        attempt = _read_json(self.attempt_path)
        if attempt.get("schema") != SUPERVISOR_ATTEMPT_SCHEMA:
            raise LifecycleError("supervisor attempt schema is invalid")
        if attempt.get("run_id") != self.run_id or attempt.get("nonce") != self.nonce:
            raise LifecycleError("supervisor attempt belongs to another run/nonce")
        if attempt.get("session") != self.binding.to_dict():
            raise LifecycleError("supervisor attempt session binding mismatch")
        return attempt

    def _ack_matches(
        self,
        ack: Mapping[str, Any],
        *,
        pid: int,
        started_at: float,
        config_sha256: str,
        binding_sha256: str,
    ) -> bool:
        ack_pid = ack.get("pid")
        return (
            ack.get("schema") == "anime-video-studio/p1-v3-supervisor-ack/v1"
            and ack.get("run_id") == self.run_id
            and ack.get("nonce") == self.nonce
            and isinstance(ack_pid, int)
            and not isinstance(ack_pid, bool)
            and ack_pid > 0
            and ack.get("order_id") == self.binding.order_id
            and ack.get("created_at") == self.binding.created_at
            and ack.get("config_sha256") == config_sha256
            and ack.get("binding_sha256") == binding_sha256
            and isinstance(ack.get("ack_at"), (int, float))
            and float(ack["ack_at"]) >= started_at - 1.0
        )

    def _heartbeat_matches(
        self,
        heartbeat: Mapping[str, Any],
        *,
        pid: int,
        started_at: float,
        config_sha256: str | None,
        binding_sha256: str | None,
    ) -> bool:
        if (
            heartbeat.get("schema")
            != "anime-video-studio/p1-v3-supervisor-heartbeat/v1"
            or heartbeat.get("run_id") != self.run_id
            or heartbeat.get("nonce") != self.nonce
            or heartbeat.get("pid") != pid
            or not isinstance(heartbeat.get("at"), (int, float))
            or float(heartbeat["at"]) < started_at - 1.0
        ):
            return False
        if config_sha256 is not None and heartbeat.get("config_sha256") != config_sha256:
            return False
        if binding_sha256 is not None and heartbeat.get("binding_sha256") != binding_sha256:
            return False
        return True

    def _ack_status(
        self,
        *,
        pid: int,
        started_at: float,
        config_sha256: str,
        binding_sha256: str,
        deadline: float,
    ) -> tuple[str, dict[str, Any] | None]:
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if self.ack_path.is_file():
                try:
                    candidate = _read_json(self.ack_path)
                except LifecycleError:
                    candidate = None
                if candidate is not None and self._ack_matches(
                    candidate,
                    pid=pid,
                    started_at=started_at,
                    config_sha256=config_sha256,
                    binding_sha256=binding_sha256,
                ):
                    last = candidate
                    status = candidate.get("status")
                    if status == "ARMED":
                        ack_pid = int(candidate["pid"])
                        if self.heartbeat_path.is_file():
                            try:
                                heartbeat = _read_json(self.heartbeat_path)
                            except LifecycleError:
                                heartbeat = None
                            if heartbeat is not None and self._heartbeat_matches(
                                heartbeat,
                                pid=ack_pid,
                                started_at=started_at,
                                config_sha256=config_sha256,
                                binding_sha256=binding_sha256,
                            ):
                                if _pid_live(ack_pid):
                                    return "ARMED", candidate
                    elif isinstance(status, str) and status:
                        return status, candidate
            if not _pid_live(pid):
                return "DIED", last
            time.sleep(0.05)
        if not _pid_live(pid):
            return "DIED", last
        return "PENDING", last

    def _record_from_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        pid: int,
        config_sha256: str,
        binding_sha256: str,
    ) -> SupervisorRecord:
        return SupervisorRecord(
            run_id=self.run_id,
            nonce=self.nonce,
            pid=pid,
            state_path=str(self.session_state_path),
            quote_path=str(self.quote_path),
            log_path=str(self.log_path),
            started_at=datetime.now(timezone.utc).isoformat(),
            order_id=self.binding.order_id,
            created_at=self.binding.created_at,
            config_sha256=config_sha256,
            binding_sha256=binding_sha256,
        )

    def _publish_record(
        self,
        attempt: Mapping[str, Any],
        *,
        pid: int,
        config_sha256: str,
        binding_sha256: str,
    ) -> SupervisorRecord:
        try:
            state = clore_session.load_state(self.session_state_path, safety_mode=True)
        except Exception as exc:
            raise LifecycleError("supervisor state readback failed") from exc
        self.binding.assert_matches_state(state)
        record = self._record_from_attempt(
            attempt,
            pid=pid,
            config_sha256=config_sha256,
            binding_sha256=binding_sha256,
        )
        _atomic_write(self.record_path, _json_text(record.to_dict()))
        self._write_attempt(
            {
                **attempt,
                "status": "ARMED",
                "pid": pid,
                "acked_at": time.time(),
            }
        )
        return record

    def _resume_attempt(
        self,
        attempt: dict[str, Any],
        *,
        config_sha256: str,
        binding_sha256: str,
        ack_timeout: float,
        deadline_monotonic: float | None,
    ) -> SupervisorRecord:
        status = attempt.get("status")
        pid = attempt.get("pid")
        if attempt.get("config_sha256") != config_sha256:
            raise LifecycleError("supervisor attempt config hash mismatch")
        if attempt.get("binding_sha256") != binding_sha256:
            raise LifecycleError("supervisor attempt binding hash mismatch")
        if status == "ARMED":
            ack = _read_json(self.ack_path) if self.ack_path.is_file() else None
            ack_pid = ack.get("pid") if isinstance(ack, dict) else None
            if (
                not isinstance(ack_pid, int)
                or not isinstance(ack, dict)
                or not self._ack_matches(
                    ack,
                    pid=pid,  # type: ignore[arg-type]
                    started_at=float(attempt.get("started_epoch") or 0),
                    config_sha256=config_sha256,
                    binding_sha256=binding_sha256,
                )
                or not _pid_live(ack_pid)
            ):
                raise LifecycleError(
                    "supervisor attempt is ARMED but its child is not live; manual recovery is required"
                )
            record = self._publish_record(
                attempt,
                pid=ack_pid,
                config_sha256=config_sha256,
                binding_sha256=binding_sha256,
            )
            return record
        if status != "STARTING":
            raise LifecycleError(
                f"supervisor attempt is terminal ({status}); manual recovery is required, "
                "no second child was started"
            )
        if not isinstance(pid, int) or pid <= 0:
            self._write_attempt(
                {**attempt, "status": "FAILED_BEFORE_SPAWN", "finished_at": time.time()}
            )
            raise LifecycleError(
                "supervisor STARTING attempt has no child pid; manual recovery is required"
            )
        started_at = attempt.get("started_epoch")
        if not isinstance(started_at, (int, float)):
            raise LifecycleError("supervisor attempt start time is invalid")
        end = time.monotonic() + ack_timeout
        if deadline_monotonic is not None:
            end = min(end, deadline_monotonic)
        outcome, ack = self._ack_status(
            pid=pid,
            started_at=float(started_at),
            config_sha256=config_sha256,
            binding_sha256=binding_sha256,
            deadline=end,
        )
        if outcome == "ARMED":
            assert ack is not None
            return self._publish_record(
                attempt,
                pid=int(ack["pid"]),
                config_sha256=config_sha256,
                binding_sha256=binding_sha256,
            )
        if outcome == "DIED":
            self._write_attempt(
                {**attempt, "status": "DIED_BEFORE_ACK", "finished_at": time.time()}
            )
            raise LifecycleError(
                "supervisor child died before ack; stable record requires manual recovery"
            )
        if outcome == "PENDING":
            raise SupervisorPending(
                "supervisor attempt is still STARTING; resume the same owned attempt "
                f"(pid {pid}); no second child was started"
            )
        self._write_attempt(
            {**attempt, "status": f"REFUSED_{outcome}", "finished_at": time.time()}
        )
        raise LifecycleError(
            f"supervisor refused activity: {outcome}; no ownership record written"
        )

    def start(
        self,
        *,
        ack_timeout: float = 10.0,
        deadline_monotonic: float | None = None,
    ) -> SupervisorRecord:
        if isinstance(ack_timeout, bool) or not isinstance(ack_timeout, (int, float)):
            raise LifecycleError("supervisor ack timeout is invalid")
        if float(ack_timeout) <= 0:
            raise LifecycleError("supervisor ack timeout must be positive")
        _require_deadline_remaining(deadline_monotonic, "supervisor")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock = InstanceLock(self.operation_lock_path).acquire()
        except InstanceAlreadyRunning as exc:
            raise LifecycleError(
                "another supervisor operation is in progress; no second child started"
            ) from exc
        try:
            return self._start_locked(
                ack_timeout=float(ack_timeout),
                deadline_monotonic=deadline_monotonic,
            )
        finally:
            lock.release()

    def _start_locked(
        self,
        *,
        ack_timeout: float,
        deadline_monotonic: float | None,
    ) -> SupervisorRecord:
        if self.record_path.exists():
            raise LifecycleError("supervisor record already exists; no second start")
        config_payload = self._config_payload()
        _atomic_write(self.config_path, _json_text(config_payload))
        config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        binding_sha256 = _binding_sha256(self.binding)
        attempt = self._read_attempt()
        if attempt is not None:
            return self._resume_attempt(
                attempt,
                config_sha256=config_sha256,
                binding_sha256=binding_sha256,
                ack_timeout=ack_timeout,
                deadline_monotonic=deadline_monotonic,
            )
        if self.ack_path.exists():
            raise LifecycleError("supervisor ack path already exists")
        attempt = self._write_attempt(
            {
                "schema": SUPERVISOR_ATTEMPT_SCHEMA,
                "status": "STARTING",
                "run_id": self.run_id,
                "nonce": self.nonce,
                "pid": None,
                "config_path": str(self.config_path),
                "config_sha256": config_sha256,
                "binding_sha256": binding_sha256,
                "session": self.binding.to_dict(),
                "created_epoch": time.time(),
                "started_epoch": None,
            }
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.time()
        try:
            handle = self.log_path.open("ab")
            try:
                process = self._popen(
                    list(self.build_argv()),
                    stdout=handle,
                    stderr=handle,
                    stdin=subprocess.DEVNULL,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    start_new_session=(os.name != "nt"),
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if os.name == "nt"
                        else 0
                    ),
                )
            finally:
                handle.close()
        except BaseException as exc:
            self._write_attempt(
                {
                    **attempt,
                    "status": "SPAWN_FAILED",
                    "error": type(exc).__name__,
                    "finished_at": time.time(),
                }
            )
            raise
        attempt = self._write_attempt(
            {
                **attempt,
                "status": "STARTING",
                "pid": int(process.pid),
                "spawned_at": time.time(),
                "started_epoch": started_at,
            }
        )
        end = time.monotonic() + ack_timeout
        if deadline_monotonic is not None:
            end = min(end, deadline_monotonic)
        outcome, ack = self._ack_status(
            pid=int(process.pid),
            started_at=started_at,
            config_sha256=config_sha256,
            binding_sha256=binding_sha256,
            deadline=end,
        )
        if outcome == "ARMED":
            assert ack is not None
            return self._publish_record(
                attempt,
                pid=int(ack["pid"]),
                config_sha256=config_sha256,
                binding_sha256=binding_sha256,
            )
        if outcome == "PENDING" and process.poll() is not None:
            outcome = "DIED"
        if outcome == "DIED":
            self._write_attempt(
                {
                    **attempt,
                    "status": "DIED_BEFORE_ACK",
                    "returncode": process.poll(),
                    "finished_at": time.time(),
                }
            )
            raise LifecycleError(
                "supervisor child died before ack; stable record requires manual recovery"
            )
        if outcome == "PENDING":
            # Keep the same STARTING attempt and its live child. A retry must
            # resume this attempt under the operation lock, never spawn another.
            self._write_attempt({**attempt, "status": "STARTING"})
            raise SupervisorPending(
                "supervisor attempt is still STARTING; resume the same owned attempt "
                f"(pid {process.pid}); no second child was started"
            )
        self._write_attempt(
            {
                **attempt,
                "status": f"REFUSED_{outcome}",
                "finished_at": time.time(),
            }
        )
        raise LifecycleError(
            f"supervisor refused activity: {outcome}; no ownership record written"
        )

    def readback(
        self,
        *,
        max_heartbeat_age: float = 120.0,
        deadline_monotonic: float | None = None,
    ) -> clore_session.CloreSessionState:
        _require_deadline_remaining(deadline_monotonic, "supervisor readback")
        if not self.record_path.is_file():
            raise LifecycleError("supervisor ownership record is missing")
        record = _read_json(self.record_path)
        if record.get("run_id") != self.run_id or record.get("nonce") != self.nonce:
            raise LifecycleError("supervisor ownership identity mismatch")
        if not self.ack_path.is_file():
            raise LifecycleError("supervisor ack is missing")
        ack = _read_json(self.ack_path)
        if ack.get("run_id") != self.run_id or ack.get("nonce") != self.nonce:
            raise LifecycleError("supervisor ack identity mismatch")
        if ack.get("status") != "ARMED":
            raise LifecycleError(
                f"supervisor terminal status {ack.get('status')} refuses activity"
            )
        pid = record.get("pid")
        if not isinstance(pid, int) or not _pid_live(pid):
            raise LifecycleError("supervisor process is not alive")
        if ack.get("pid") != pid:
            raise LifecycleError("supervisor ack pid does not match the owner record")
        if not self.heartbeat_path.is_file():
            raise LifecycleError("supervisor heartbeat is missing")
        heartbeat = _read_json(self.heartbeat_path)
        config_sha256 = record.get("config_sha256")
        binding_sha256 = record.get("binding_sha256")
        if (
            not isinstance(config_sha256, str)
            or len(config_sha256) != 64
            or not isinstance(binding_sha256, str)
            or len(binding_sha256) != 64
        ):
            raise LifecycleError("supervisor ownership hashes are invalid")
        if not self.config_path.is_file():
            raise LifecycleError("supervisor config is missing")
        if hashlib.sha256(self.config_path.read_bytes()).hexdigest() != config_sha256:
            raise LifecycleError("supervisor config hash mismatch")
        if _binding_sha256(self.binding) != binding_sha256:
            raise LifecycleError("supervisor binding hash mismatch")
        if (
            ack.get("config_sha256") != config_sha256
            or ack.get("binding_sha256") != binding_sha256
        ):
            raise LifecycleError("supervisor ack ownership hash mismatch")
        if not self._heartbeat_matches(
            heartbeat,
            pid=pid,
            started_at=float("-inf"),
            config_sha256=config_sha256,
            binding_sha256=binding_sha256,
        ):
            raise LifecycleError("supervisor heartbeat identity mismatch")
        if (
            heartbeat.get("run_id") != self.run_id
            or heartbeat.get("nonce") != self.nonce
            or heartbeat.get("pid") != pid
            or not isinstance(heartbeat.get("at"), (int, float))
        ):
            raise LifecycleError("supervisor heartbeat identity is invalid")
        age = time.time() - float(heartbeat["at"])
        if age < -30 or age > max_heartbeat_age:
            raise LifecycleError("supervisor heartbeat is stale")
        try:
            state = clore_session.load_state(self.session_state_path, safety_mode=True)
        except Exception as exc:
            raise LifecycleError(
                f"supervisor readback unavailable: {type(exc).__name__}"
            ) from exc
        self.binding.assert_matches_state(state)
        return state


def fetch_exact_stream(
    *,
    runner: Any,
    identity: RemoteIdentity,
    remote_helper_path: str,
    remote_path: str,
    expected_size: int,
    expected_sha256: str,
    local_path: Path,
    offset: int = 0,
    expected_prefix_sha256: str | None = None,
    timeout: float = 120.0,
    remote_root: str | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Fetch exact bytes via SSH stdout typed header, no SCP/whole-file copy."""

    remote = _validate_remote_absolute_path(remote_path, "remote_path")
    helper = _validate_remote_absolute_path(remote_helper_path, "remote_helper_path")
    if remote_root is None:
        remote_root = str(PurePosixPath(remote).parent)
    root = _validate_remote_absolute_path(remote_root, "remote_root")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise LifecycleError("expected size is invalid")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise LifecycleError("expected sha256 is invalid")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset > expected_size:
        raise LifecycleError("offset is invalid")
    if offset > 0 and (
        not isinstance(expected_prefix_sha256, str)
        or len(expected_prefix_sha256) != 64
    ):
        raise LifecycleError("resume requires expected prefix sha256")
    local_path = assert_safe_local_path(local_path)
    if offset > 0:
        if not local_path.is_file():
            raise LifecycleError("local partial is missing for resume")
        actual_offset, prefix_hash = _hash_prefix(local_path, limit=offset + 1)
        if actual_offset != offset or prefix_hash != expected_prefix_sha256:
            raise LifecycleError("local partial prefix identity mismatch")
    suffix_path = local_path.with_name(local_path.name + ".suffix")
    assert_safe_local_path(suffix_path)
    if suffix_path.exists():
        suffix_path.unlink()
    if deadline_monotonic is None:
        deadline_monotonic = time.monotonic() + float(timeout)
    if deadline_monotonic - time.monotonic() <= 0:
        raise LifecycleError("stream deadline expired before the remote helper")
    try:
        header_raw, payload_size = runner.stream_remote_to_file(
            identity,
            ("python3", helper, "stream", remote, str(offset), root),
            suffix_path,
            timeout=max(0.05, deadline_monotonic - time.monotonic()),
            max_payload_bytes=max(1, expected_size - offset),
            expected_path=remote,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            expected_offset=offset,
            deadline_monotonic=deadline_monotonic,
        )
    except RemoteStreamPartialError as exc:
        header = exc.header
        valid_header = (
            isinstance(header, dict)
            and header.get("path") == remote
            and header.get("size") == expected_size
            and header.get("sha256") == expected_sha256
            and header.get("offset") == offset
            and header.get("length") == expected_size - offset
        )
        try:
            file_size = suffix_path.stat().st_size
        except OSError:
            file_size = -1
        if (
            not valid_header
            or exc.target.resolve() != suffix_path.resolve()
            or exc.payload_size <= 0
            or exc.payload_size != file_size
        ):
            suffix_path.unlink(missing_ok=True)
            raise LifecycleError(
                "stream partial did not match the caller expected identity"
            ) from exc
        try:
            mode = "ab" if local_path.is_file() else "wb"
            with suffix_path.open("rb") as source, local_path.open(mode) as out:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
        except OSError as exc2:
            raise LifecycleError("validated stream partial could not be preserved") from exc2
        finally:
            suffix_path.unlink(missing_ok=True)
        raise
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError("stream header is invalid") from exc
    if (
        not isinstance(header, dict)
        or header.get("schema") != "anime-video-studio/p1-v3-stream/v1"
        or header.get("path") != remote
        or header.get("size") != expected_size
        or header.get("sha256") != expected_sha256
        or header.get("offset") != offset
        or header.get("length") != expected_size - offset
        or payload_size != expected_size - offset
    ):
        suffix_path.unlink(missing_ok=True)
        raise LifecycleError("stream header/identity/size/hash mismatch")
    temp = local_path.with_name(local_path.name + ".fetch.tmp")
    assert_safe_local_path(temp)
    try:
        digest = hashlib.sha256()
        written_prefix = 0
        with temp.open("wb") as out:
            if offset:
                if not local_path.is_file():
                    raise LifecycleError("local partial is missing for resume")
                actual_offset, prefix_hash = _hash_prefix(local_path, limit=offset + 1)
                if actual_offset != offset or prefix_hash != expected_prefix_sha256:
                    raise LifecycleError("local partial prefix identity mismatch")
                with local_path.open("rb") as prefix:
                    for chunk in iter(lambda: prefix.read(1024 * 1024), b""):
                        written_prefix += len(chunk)
                        digest.update(chunk)
                        out.write(chunk)
                if written_prefix != offset:
                    raise LifecycleError("local partial size changed during resume")
            with suffix_path.open("rb") as suffix:
                for chunk in iter(lambda: suffix.read(1024 * 1024), b""):
                    digest.update(chunk)
                    out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != expected_sha256:
            raise LifecycleError("stream final sha256 mismatch")
        os.replace(temp, local_path)
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        suffix_path.unlink(missing_ok=True)
    return {"path": str(local_path), "size": expected_size, "sha256": expected_sha256, "offset": offset}


class RemoteDownloadWorker:
    """Owned asynchronous download worker protocol on one remote root.

    The launcher is a staged, fixed Python file.  Parent and child share the
    exact run_id/nonce and the hash of the canonical launch manifest; an owner
    record without those identities is foreign and is never replaced.  A start
    returns ``ATTACHED`` for a live same-run worker and fails closed for an
    unknown/foreign state, so a second worker cannot be spawned implicitly.
    """

    def __init__(
        self,
        *,
        runner: Any,
        identity: RemoteIdentity,
        remote_root: str,
        run_id: str,
        nonce: str,
        local_worker: Path,
        local_setup_plan: Path,
        local_manifest: Path,
        local_launcher: Path | None = None,
    ) -> None:
        self.runner = runner
        self.identity = identity
        self.remote_root = _validate_remote_absolute_path(remote_root, "remote_root")
        self.run_id = run_id
        self.nonce = nonce
        self.local_worker = Path(local_worker)
        self.local_setup_plan = Path(local_setup_plan)
        self.local_manifest = Path(local_manifest)
        self.local_helper = Path(__file__).resolve().parent / "v3_remote_fetch_helper.py"
        self.local_launcher = (
            Path(local_launcher)
            if local_launcher is not None
            else Path(__file__).resolve().parent / "v3_remote_launcher.py"
        )
        posix = PurePosixPath(self.remote_root)
        self.worker_path = str(posix / "v3_download_worker.py")
        self.setup_plan_path = str(posix / "setup_plan.py")
        self.manifest_path = str(posix / "manifest.json")
        self.payload_path = str(posix / "job.json")
        self.launcher_path = str(posix / "v3_remote_launcher.py")
        self.launch_manifest_path = str(posix / "launch.json")
        self.helper_path = str(posix / "scripts" / "p1" / "v3_remote_fetch_helper.py")
        self.owner_path = str(posix / "worker-owner.json")
        self.ack_path = str(posix / "worker-ack.json")
        self.status_path = str(posix / "status.json")
        self.progress_path = str(posix / "progress.json")
        self.heartbeat_path = str(posix / "heartbeat.json")
        self.log_path = str(posix / "download.log")
        self.launch_manifest_hash: str | None = None

    def _put(self, remote: str, local: Path, *, timeout: float = 60.0, deadline_monotonic: float | None = None) -> None:
        kwargs = {
            "root": self.remote_root,
            "allow_same_hash": True,
            "timeout": timeout,
            "deadline_monotonic": deadline_monotonic,
        }
        try:
            self.runner.put_file(self.identity, remote, local, **kwargs)
        except TypeError as exc:
            raise LifecycleError("runner does not support bounded staging") from exc

    def _read_optional(self, path: str, *, timeout: float = 15.0, deadline_monotonic: float | None = None) -> dict[str, Any] | None:
        try:
            return read_remote_json(
                runner=self.runner,
                identity=self.identity,
                helper_path=self.helper_path,
                remote_path=path,
                remote_root=self.remote_root,
                timeout=timeout,
                deadline_monotonic=deadline_monotonic,
            )
        except Exception:
            return None

    def _read_required(self, path: str, name: str, *, timeout: float = 15.0, deadline_monotonic: float | None = None) -> dict[str, Any]:
        try:
            return read_remote_json(
                runner=self.runner,
                identity=self.identity,
                helper_path=self.helper_path,
                remote_path=path,
                remote_root=self.remote_root,
                timeout=timeout,
                deadline_monotonic=deadline_monotonic,
            )
        except Exception as exc:
            raise LifecycleError(f"remote worker {name} is unavailable/unknown") from exc

    def replace_known_dead(
        self,
        *,
        launch_manifest_path: str,
        timeout: float,
    ) -> dict[str, Any]:
        result = self.runner.run_remote(
            self.identity,
            ("python3", self.launcher_path, "--launch", launch_manifest_path, "--resume-dead"),
            timeout=timeout,
        )
        return self._parse_launch_result(result)

    def _parse_launch_result(self, result: Any) -> dict[str, Any]:
        if getattr(result, "timed_out", False) or result.returncode != 0:
            raise LifecycleError("remote launcher refused or failed")
        try:
            payload = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LifecycleError("remote launcher result is invalid") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "anime-video-studio/p1-v3-launch-result/v1"
            or payload.get("run_id") != self.run_id
            or payload.get("nonce") != self.nonce
            or payload.get("kind") != "download_worker"
            or payload.get("status") not in {"STARTED", "ATTACHED", "ATTACHED_COMPLETE"}
        ):
            raise LifecycleError("remote launcher identity/status mismatch")
        if (
            self.launch_manifest_hash is not None
            and payload.get("command_sha256") != self.launch_manifest_hash
        ):
            raise LifecycleError("remote launcher command hash mismatch")
        return payload

    def stage(
        self,
        *,
        payload: Mapping[str, Any],
        manifest_sha256: str,
        required_bytes: int,
        timeout: float,
        resume_dead: bool = False,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        if timeout <= 0:
            raise LifecycleError("download staging deadline expired")
        if payload.get("run_id") != self.run_id or payload.get("nonce") != self.nonce:
            raise LifecycleError("download payload identity mismatch")
        if payload.get("manifest_sha256") != manifest_sha256:
            raise LifecycleError("download payload manifest hash mismatch")
        if (
            not isinstance(manifest_sha256, str)
            or len(manifest_sha256) != 64
            or hashlib.sha256(self.local_manifest.read_bytes()).hexdigest() != manifest_sha256
        ):
            raise LifecycleError("local manifest does not match the pinned hash")
        mkdir = self.runner.run_remote(
            self.identity,
            ("mkdir", "-p", self.remote_root),
            timeout=min(30.0, timeout),
            deadline_monotonic=deadline_monotonic,
        )
        if mkdir.returncode != 0 or getattr(mkdir, "timed_out", False):
            raise LifecycleError("remote owned root creation failed")
        df = self.runner.run_remote(
            self.identity,
            ("df", "-Pk", self.remote_root),
            timeout=min(30.0, timeout),
            deadline_monotonic=deadline_monotonic,
        )
        if df.returncode != 0 or getattr(df, "timed_out", False):
            raise LifecycleError("remote disk check failed")
        try:
            available_kb = int(
                [line.split() for line in df.stdout.splitlines() if line.strip()][-1][3]
            )
        except (IndexError, ValueError) as exc:
            raise LifecycleError("remote disk output is invalid") from exc
        if isinstance(required_bytes, bool) or not isinstance(required_bytes, int) or required_bytes < 0:
            raise LifecycleError("remote required bytes are invalid")
        if available_kb * 1024 < required_bytes:
            raise LifecycleError("remote disk is insufficient")
        for remote, local in (
            (self.launcher_path, self.local_launcher),
            (self.worker_path, self.local_worker),
            (self.setup_plan_path, self.local_setup_plan),
            (self.manifest_path, self.local_manifest),
            (self.helper_path, self.local_helper),
        ):
            self._put(remote, local, timeout=min(60.0, timeout), deadline_monotonic=deadline_monotonic)
        remote_payload = dict(payload)
        remote_payload.update(
            {
                "workspace": self.remote_root,
                "root": str(PurePosixPath(self.remote_root) / "ComfyUI"),
                "manifest_path": self.manifest_path,
                "status_path": self.status_path,
                "progress_path": self.progress_path,
                "heartbeat_path": self.heartbeat_path,
            }
        )
        # Recovery permission is carried by the launcher command/env, not by
        # mutating the immutable job payload or its hash.
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False, newline="\n"
        )
        local_payload = Path(handle.name)
        try:
            with handle:
                handle.write(_json_text(remote_payload))
            self._put(self.payload_path, local_payload, timeout=min(60.0, timeout), deadline_monotonic=deadline_monotonic)
            launch_manifest = {
                "schema": LAUNCH_SCHEMA,
                "kind": "download_worker",
                "run_id": self.run_id,
                "nonce": self.nonce,
                "root": self.remote_root,
                "worker": self.worker_path,
                "worker_sha256": hashlib.sha256(self.local_worker.read_bytes()).hexdigest(),
                "payload": self.payload_path,
                "payload_sha256": hashlib.sha256(local_payload.read_bytes()).hexdigest(),
                "deadline_epoch": remote_payload.get("deadline_epoch"),
                "status": self.status_path,
                "progress": self.progress_path,
                "heartbeat": self.heartbeat_path,
                "log": self.log_path,
                "owner": self.owner_path,
                "ack": self.ack_path,
            }
            self.launch_manifest_hash = _sha_text(
                json.dumps(launch_manifest, sort_keys=True, separators=(",", ":"))
            )
            launch_handle = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", delete=False, newline="\n"
            )
            launch_local = Path(launch_handle.name)
            try:
                with launch_handle:
                    launch_handle.write(_json_text(launch_manifest))
                self._put(self.launch_manifest_path, launch_local, timeout=min(60.0, timeout), deadline_monotonic=deadline_monotonic)
            finally:
                launch_local.unlink(missing_ok=True)
        finally:
            local_payload.unlink(missing_ok=True)
        return {
            "root": self.remote_root,
            "launch_manifest": self.launch_manifest_path,
            "worker": self.worker_path,
            "payload": self.payload_path,
        }

    def read_owner_ack(self, *, timeout: float = 15.0, deadline_monotonic: float | None = None) -> dict[str, Any]:
        owner = self._read_required(self.owner_path, "owner", timeout=timeout, deadline_monotonic=deadline_monotonic)
        ack = self._read_required(self.ack_path, "ack", timeout=timeout, deadline_monotonic=deadline_monotonic)
        for name, value in (("owner", owner), ("ack", ack)):
            if (
                value.get("schema") != PROCESS_OWNER_SCHEMA
                or value.get("run_id") != self.run_id
                or value.get("nonce") != self.nonce
                or value.get("kind") != "download_worker"
                or not isinstance(value.get("command_sha256"), str)
                or len(value["command_sha256"]) != 64
            ):
                raise LifecycleError(f"remote worker {name} identity mismatch")
        if owner.get("command_sha256") != ack.get("command_sha256"):
            raise LifecycleError("remote worker owner/ack command mismatch")
        return {"owner": owner, "ack": ack}

    def readback(self, *, timeout: float = 15.0, deadline_monotonic: float | None = None) -> dict[str, Any]:
        owner_ack = self.read_owner_ack(timeout=timeout, deadline_monotonic=deadline_monotonic)
        owner = owner_ack["owner"]
        ack = owner_ack["ack"]
        status = self._read_required(self.status_path, "status", timeout=timeout, deadline_monotonic=deadline_monotonic)
        progress = self._read_required(self.progress_path, "progress", timeout=timeout, deadline_monotonic=deadline_monotonic)
        heartbeat = self._read_required(self.heartbeat_path, "heartbeat", timeout=timeout, deadline_monotonic=deadline_monotonic)
        for name, value in (("status", status), ("progress", progress), ("heartbeat", heartbeat)):
            if value.get("run_id") != self.run_id or value.get("nonce") != self.nonce:
                raise LifecycleError(f"remote worker {name} identity mismatch")
        for key in ("files_done", "files_total", "bytes_done", "bytes_total"):
            number = progress.get(key)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise LifecycleError(f"remote worker progress {key} is invalid")
        if progress["bytes_done"] > progress["bytes_total"]:
            raise LifecycleError("remote worker progress bytes exceed the total")
        if status.get("status") not in {"RUNNING", "COMPLETE", "FAILED"}:
            raise LifecycleError("remote worker status is unknown")
        return {
            "schema": REMOTE_STATE_SCHEMA,
            "owner": owner,
            "ack": ack,
            "status": status,
            "progress": progress,
            "heartbeat": heartbeat,
            "run_id": self.run_id,
            "nonce": self.nonce,
        }

    def start(
        self,
        *,
        launch_manifest: str,
        timeout: float,
        resume_dead: bool = False,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        try:
            owner = self._read_optional(self.owner_path)
        except Exception:
            owner = None
        if owner is not None:
            if (
                owner.get("schema") != PROCESS_OWNER_SCHEMA
                or owner.get("run_id") != self.run_id
                or owner.get("nonce") != self.nonce
                or owner.get("kind") != "download_worker"
            ):
                raise LifecycleError("remote worker ownership conflict")
            owner_ack = self.read_owner_ack(deadline_monotonic=deadline_monotonic)
            status = self._read_optional(self.status_path, deadline_monotonic=deadline_monotonic) or {"status": "RUNNING"}
            if status.get("run_id") not in (None, self.run_id) or status.get("nonce") not in (None, self.nonce):
                raise LifecycleError("remote worker status identity mismatch")
            state = {
                "owner": owner_ack["owner"],
                "ack": owner_ack["ack"],
                "status": status,
                "progress": self._read_optional(self.progress_path, deadline_monotonic=deadline_monotonic) or {},
                "heartbeat": self._read_optional(self.heartbeat_path, deadline_monotonic=deadline_monotonic) or {},
                "run_id": self.run_id,
                "nonce": self.nonce,
            }
            if state["status"].get("status") == "COMPLETE":
                return {"status": "ATTACHED_COMPLETE", "state": state}
            if state["status"].get("status") == "FAILED" and not resume_dead:
                raise LifecycleError(
                    "remote worker is known dead; explicit --resume-dead policy required"
                )
            if state["status"].get("status") != "FAILED":
                return {"status": "ATTACHED", "state": state}
        else:
            # No owner but durable worker state/ack exists: this is an unknown
            # or partially deleted ownership state.  Never start a second
            # worker implicitly.
            for path in (self.ack_path, self.status_path, self.progress_path, self.heartbeat_path):
                if self._read_optional(path) is not None:
                    raise LifecycleError(
                        "remote worker state exists without a valid owner; explicit recovery required"
                    )
        args = ["--launch", launch_manifest]
        if resume_dead:
            args.append("--resume-dead")
        result = self.runner.run_remote(
            self.identity,
            ("python3", self.launcher_path, *args),
            timeout=timeout,
            deadline_monotonic=deadline_monotonic,
        )
        parsed = self._parse_launch_result(result)
        owner_ack = self.read_owner_ack(deadline_monotonic=deadline_monotonic)
        if parsed.get("command_sha256") != owner_ack["owner"].get("command_sha256"):
            raise LifecycleError("remote launcher/owner command mismatch")
        parsed["state"] = owner_ack
        if parsed.get("status") == "ATTACHED_COMPLETE":
            parsed["status"] = "ATTACHED_COMPLETE"
        elif parsed.get("status") == "ATTACHED":
            parsed["status"] = "ATTACHED"
        else:
            parsed["status"] = "STARTED"
        return parsed

    def wait(
        self,
        *,
        deadline_epoch: float,
        poll_seconds: float = 1.0,
    ) -> dict[str, Any]:
        if deadline_epoch <= 0:
            raise LifecycleError("remote worker deadline is invalid")
        remaining_session = deadline_epoch - time.time()
        if remaining_session <= 0:
            raise LifecycleError("remote worker deadline expired before readback")
        monotonic_deadline = time.monotonic() + remaining_session
        last_error: Exception | None = None
        while time.time() < deadline_epoch:
            try:
                state = self.readback(
                    timeout=clip_remote_timeout(deadline_epoch, 10.0),
                    deadline_monotonic=monotonic_deadline,
                )
            except Exception as exc:
                last_error = exc
                time.sleep(min(poll_seconds, max(0.05, deadline_epoch - time.time())))
                continue
            status = state["status"].get("status")
            if status in {"COMPLETE", "FAILED"}:
                return state
            time.sleep(min(poll_seconds, max(0.05, deadline_epoch - time.time())))
        if last_error is not None:
            raise LifecycleError(
                "remote worker unreachable before deadline; attach is required, "
                "no second worker was started"
            ) from last_error
        raise LifecycleError("remote worker did not finish before the deadline")


def run_remote_download_worker(
    *,
    runner: Any,
    identity: RemoteIdentity,
    remote_dir: str,
    local_worker: Path,
    local_setup_plan: Path,
    local_manifest: Path,
    payload: Mapping[str, Any],
    timeout: float,
    resume_dead: bool = False,
) -> dict[str, Any]:
    """Synchronous compatibility wrapper around the owned async protocol."""

    worker = RemoteDownloadWorker(
        runner=runner,
        identity=identity,
        remote_root=remote_dir,
        run_id=str(payload.get("run_id", "")),
        nonce=str(payload.get("nonce", "")),
        local_worker=local_worker,
        local_setup_plan=local_setup_plan,
        local_manifest=local_manifest,
    )
    staged = worker.stage(
        payload=payload,
        manifest_sha256=str(payload.get("manifest_sha256", "")),
        required_bytes=int(payload.get("required_bytes", 0) or 0),
        timeout=timeout,
        resume_dead=resume_dead,
    )
    started = worker.start(
        launch_manifest=staged["launch_manifest"],
        timeout=timeout,
        resume_dead=resume_dead,
    )
    if started.get("status") == "ATTACHED_COMPLETE":
        state = started["state"]
    else:
        state = worker.wait(
            deadline_epoch=float(payload.get("deadline_epoch", time.time() + timeout)),
            poll_seconds=0.5,
        )
    return {
        "run_id": payload.get("run_id"),
        "nonce": payload.get("nonce"),
        "status": state["status"].get("status"),
        "progress": state["progress"],
        "heartbeat": state["heartbeat"],
        "owner": state["owner"],
    }


def verify_host_key(
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    expected_fingerprint: str,
    known_hosts: Path,
    timeout: float = 10.0,
    deadline_monotonic: float | None = None,
) -> None:
    """Require existing known_hosts entry and exact trusted fingerprint."""

    if not isinstance(expected_fingerprint, str) or not expected_fingerprint.startswith("SHA256:"):
        raise LifecycleError("expected host fingerprint is missing/invalid")
    if not runner.ssh_keygen_path:
        raise LifecycleError("ssh-keygen is unavailable")
    if deadline_monotonic is not None:
        if isinstance(deadline_monotonic, bool) or not isinstance(deadline_monotonic, (int, float)):
            raise LifecycleError("host-key deadline is invalid")
        remaining = float(deadline_monotonic) - time.monotonic()
        if remaining <= 0:
            raise LifecycleError("deadline expired before host-key check")
        timeout = min(timeout, remaining)
    if not (0 < timeout <= 60):
        raise LifecycleError("host-key timeout is invalid")
    result = run_bounded_process(
        [
            runner.ssh_keygen_path,
            "-F",
            identity.host_key_name,
            "-f",
            str(known_hosts),
        ],
        timeout=timeout,
        max_output_bytes=64 * 1024,
    )
    if result.returncode == 1 or not result.stdout.strip():
        raise LifecycleError("host key is absent from explicit known_hosts")
    if result.returncode != 0 or result.timed_out:
        raise LifecycleError("host key lookup failed")
    fingerprints: set[str] = set()
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[1].startswith(("ssh-", "ecdsa-", "sk-")):
            continue
        try:
            blob = base64.b64decode(fields[2], validate=True)
        except Exception:
            continue
        digest = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii").rstrip("=")
        fingerprints.add(f"SHA256:{digest}")
    if expected_fingerprint not in fingerprints:
        raise LifecycleError("host key fingerprint does not match trusted config")


__all__ = [
    "CloreSupervisor",
    "LifecycleError",
    "LifecycleState",
    "ProgressRecorder",
    "RemoteDownloadWorker",
    "SCHEMA",
    "SessionBinding",
    "SupervisorPending",
    "SupervisorRecord",
    "TargetLock",
    "assert_budget",
    "assert_deadline",
    "assert_disk",
    "assert_safe_local_path",
    "fetch_exact_stream",
    "read_remote_json",
    "run_remote_download_worker",
    "load_lifecycle",
    "load_session_binding",
    "new_lifecycle",
    "required_download_bytes",
    "update_phase",
    "verify_host_key",
    "write_lifecycle",
]
