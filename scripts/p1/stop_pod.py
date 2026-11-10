"""External, fail-closed P1 Pod stop controller.

This controller is intentionally standalone and separate from the web app. It
persists one absolute UTC deadline derived from an explicitly supplied,
provider-reported ``created_at`` (not local setup completion). The deadline may
be initialized at or below ``created_at + 60 minutes`` and can never be reset or
extended by a restart.

Dry-run is the default and performs no provider calls, environment inspection or
credential reads. ``--execute`` validates the pinned runpodctl version first,
watches the persisted deadline by default, then uses only
``runpodctl pod stop <POD_ID>`` followed by a bounded readback poll of
``runpodctl pod get <POD_ID> --output json``.  ``--once`` skips the deadline
watch but still performs the full bounded stop/readback sequence; it is not a
scheduled-stop guarantee.  Init and evaluate hold an exclusive OS/process lock.

A persisted stop intent is written BEFORE the external stop.  If the process
dies in that window, a restart only reads back and reports LIVE_BLOCKED for
manual intervention if the Pod is still running; it never blindly sends a
second stop.  The readback bound is stored as an absolute timestamp and is
never reset or extended by a restart.  It does not extend the 60-minute
session deadline.  Provider/network failures may still leave the Pod running:
no hard cap is promised.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

# Allow ``python scripts/p1/stop_pod.py`` as well as package/test imports.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from anime_studio.errors import InstanceAlreadyRunning
from anime_studio.lock import InstanceLock

from scripts.p1.runpodctl_bridge import (
    STATUS_ERROR,
    STATUS_INITIALIZING,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_TERMINATED,
    STATUS_UNKNOWN,
    ParsedTimestamp,
    Provider,
    ProviderError,
    ProviderPod,
    ProviderTransientError,
    RunpodctlProvider,
    parse_utc_timestamp,
)

STATE_SCHEMA = "anime-video-studio/p1-stop-state/v1"
MAX_SESSION_SECONDS = 60 * 60
DEFAULT_READBACK_TIMEOUT_SECONDS = 120.0
MAX_READBACK_SECONDS = 600.0
DEFAULT_READBACK_POLL_SECONDS = 5.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 30.0
MAX_READBACK_ATTEMPTS = 1000
POD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
STATUSES = {"ARMED", "STOP_REQUESTED", "STOPPED", "LIVE_BLOCKED"}

Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]


class StopStateError(Exception):
    """Persisted state is invalid, mismatched or would be extended."""


class StopControllerError(Exception):
    """The controller cannot safely perform the requested operation."""


@dataclass(frozen=True)
class StopState:
    pod_id: str
    created_at: datetime
    deadline_at: datetime
    status: str
    updated_at: datetime
    last_error: str | None = None
    readback_started_at: datetime | None = None
    readback_deadline_at: datetime | None = None
    created_at_ns: int | None = None


@dataclass(frozen=True)
class StopOutcome:
    status: str
    message: str
    state: StopState


_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


@contextmanager
def _state_lock(state_path: Path) -> Iterator[None]:
    """Exclusive lock for init/read/provider/write.

    The OS file lock protects separate processes; the local lock also protects
    threads in this process (Windows msvcrt locking can otherwise be reentrant
    for another handle in the same process). Both are released together.
    """

    lock_path = Path(str(state_path) + ".lock")
    if lock_path.is_symlink():
        raise StopStateError("lock path must not be a symlink")
    key = str(state_path.resolve())
    with _LOCAL_LOCKS_GUARD:
        local_lock = _LOCAL_LOCKS.setdefault(key, threading.Lock())
    local_lock.acquire()
    lock = None
    try:
        try:
            lock = InstanceLock(lock_path).acquire()
        except InstanceAlreadyRunning as exc:
            raise StopStateError(
                "another stop controller holds the state lock; refusing concurrent control"
            ) from exc
        yield
    finally:
        if lock is not None:
            lock.release()
        local_lock.release()


def validate_pod_id(value: str) -> str:
    if not isinstance(value, str) or not POD_ID_RE.fullmatch(value):
        raise StopStateError("pod id must be one exact, validated RunPod Pod ID")
    return value


def _coerce_utc_precise(value: datetime | str, name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise StopStateError(f"{name} must be timezone-aware UTC")
        return value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.strip():
        raise StopStateError(f"{name} must be an ISO-8601 UTC timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise StopStateError(f"{name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise StopStateError(f"{name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _coerce_utc(value: datetime | str, name: str) -> datetime:
    return _coerce_utc_precise(value, name).replace(microsecond=0)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _format_utc_precise(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return _format_utc(utc)


def _datetime_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        raise StopStateError("timestamp must be timezone-aware UTC")
    utc = value.astimezone(timezone.utc)
    seconds = calendar.timegm(utc.utctimetuple())
    return seconds * 1_000_000_000 + utc.microsecond * 1_000


def _parse_state_created_at(value: datetime | str) -> ParsedTimestamp:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise StopStateError("created_at must be timezone-aware UTC")
        utc = value.astimezone(timezone.utc)
        fraction_ns = utc.microsecond * 1_000
        canonical = utc.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")
        if fraction_ns:
            canonical += f".{fraction_ns:09d}"
        canonical += "Z"
        return ParsedTimestamp(
            datetime=utc.replace(microsecond=0),
            epoch_ns=_datetime_to_ns(value),
            text=canonical,
        )
    parsed = parse_utc_timestamp(value)
    if parsed is None:
        raise StopStateError("created_at must be an ISO-8601 UTC timestamp")
    return parsed


def _positive_finite(value: Any, name: str, *, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise StopStateError(f"{name} must be a finite positive number")
    number = float(value)
    if maximum is not None and number > maximum:
        raise StopStateError(f"{name} must not exceed {maximum:g}")
    return number


def _safe_message(value: Any, *, limit: int = 300) -> str:
    text = " ".join(str(value).split())
    if not text:
        text = "provider operation failed"
    return text[:limit]


def _validate_state(state: StopState) -> StopState:
    validate_pod_id(state.pod_id)
    if state.deadline_at < state.created_at:
        raise StopStateError("deadline must not precede Pod creation")
    if state.deadline_at > state.created_at + timedelta(seconds=MAX_SESSION_SECONDS):
        raise StopStateError("deadline exceeds the 60-minute maximum from Pod creation")
    if not isinstance(state.status, str) or state.status not in STATUSES:
        raise StopStateError("state status is not recognized")
    if state.status == "STOPPED" and state.last_error is not None:
        raise StopStateError("STOPPED state must not retain an error")
    if state.created_at_ns is not None:
        if (
            isinstance(state.created_at_ns, bool)
            or not isinstance(state.created_at_ns, int)
            or state.created_at_ns < 0
        ):
            raise StopStateError("created_at_ns must be a non-negative integer")
        seconds = calendar.timegm(state.created_at.utctimetuple())
        if state.created_at_ns // 1_000_000_000 != seconds:
            raise StopStateError("created_at_ns does not match created_at")
    if state.readback_started_at is not None or state.readback_deadline_at is not None:
        if state.status == "ARMED":
            raise StopStateError(
                "ARMED state must not carry a readback anchor or deadline"
            )
        if state.readback_deadline_at is not None and state.readback_deadline_at < state.created_at:
            raise StopStateError("readback deadline must not precede Pod creation")
        if state.readback_started_at is not None:
            if state.readback_deadline_at is None:
                raise StopStateError(
                    "readback_started_at requires readback_deadline_at"
                )
            if state.readback_started_at < state.created_at:
                raise StopStateError(
                    "readback_started_at must not precede Pod creation"
                )
            readback_seconds = (
                state.readback_deadline_at - state.readback_started_at
            ).total_seconds()
            if readback_seconds <= 0 or readback_seconds > MAX_READBACK_SECONDS:
                raise StopStateError(
                    "readback interval must be positive and at most "
                    f"{MAX_READBACK_SECONDS:g} seconds"
                )
    return state


def _state_to_dict(state: StopState) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "pod_id": state.pod_id,
        "created_at": _format_utc(state.created_at),
        "created_at_ns": (
            state.created_at_ns
            if state.created_at_ns is not None
            else _datetime_to_ns(state.created_at)
        ),
        "deadline_at": _format_utc(state.deadline_at),
        "status": state.status,
        "updated_at": _format_utc(state.updated_at),
        "last_error": state.last_error,
        "readback_started_at": (
            _format_utc_precise(state.readback_started_at)
            if state.readback_started_at is not None
            else None
        ),
        "readback_deadline_at": (
            _format_utc_precise(state.readback_deadline_at)
            if state.readback_deadline_at is not None
            else None
        ),
    }


def load_state(path: Path | str) -> StopState:
    state_path = Path(path)
    if state_path.is_symlink():
        raise StopStateError("state path must not be a symlink")
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StopStateError(f"cannot read stop state {state_path}: {exc}") from exc
    if not isinstance(data, Mapping) or data.get("schema") != STATE_SCHEMA:
        raise StopStateError("stop state schema is missing or invalid")
    readback_raw = data.get("readback_deadline_at")
    readback_started_raw = data.get("readback_started_at")
    created_ns_raw = data.get("created_at_ns")
    try:
        created = _parse_state_created_at(data.get("created_at"))
        if created_ns_raw is not None:
            if (
                isinstance(created_ns_raw, bool)
                or not isinstance(created_ns_raw, int)
                or created_ns_raw < 0
            ):
                raise StopStateError("created_at_ns must be a non-negative integer")
            created_ns = created_ns_raw
        else:
            # Legacy state carried seconds precision only.  This is intentional:
            # a provider with a non-zero fraction will fail identity rather than
            # being silently treated as equal.  If an edited legacy file does
            # contain a fraction, preserve it rather than truncating silently.
            created_ns = created.epoch_ns
        state = StopState(
            pod_id=validate_pod_id(data.get("pod_id")),
            created_at=created.datetime,
            deadline_at=_coerce_utc(data.get("deadline_at"), "deadline_at"),
            status=data.get("status"),
            updated_at=_coerce_utc(data.get("updated_at"), "updated_at"),
            last_error=data.get("last_error"),
            readback_started_at=(
                _coerce_utc_precise(readback_started_raw, "readback_started_at")
                if readback_started_raw is not None
                else None
            ),
            readback_deadline_at=(
                _coerce_utc_precise(readback_raw, "readback_deadline_at")
                if readback_raw is not None
                else None
            ),
            created_at_ns=created_ns,
        )
    except (TypeError, ValueError) as exc:
        raise StopStateError("stop state fields are invalid") from exc
    if state.last_error is not None and not isinstance(state.last_error, str):
        raise StopStateError("last_error must be a string or null")
    return _validate_state(state)


def save_state(path: Path | str, state: StopState) -> None:
    _validate_state(state)
    state_path = Path(path)
    if state_path.is_symlink():
        raise StopStateError("state path must not be a symlink")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_state_to_dict(state), indent=2, sort_keys=True) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=state_path.parent,
        prefix=state_path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp_name = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, state_path)
    finally:
        try:
            temp_name.unlink(missing_ok=True)
        except OSError:
            pass


def initialize_state(
    path: Path | str,
    *,
    pod_id: str,
    created_at: datetime | str,
    deadline_at: datetime | str | None = None,
    now: datetime | None = None,
) -> StopState:
    """Create state once, or return persisted state without extending it."""

    state_path = Path(path)
    with _state_lock(state_path):
        return _initialize_state_unlocked(
            state_path,
            pod_id=pod_id,
            created_at=created_at,
            deadline_at=deadline_at,
            now=now,
        )


def _initialize_state_unlocked(
    path: Path | str,
    *,
    pod_id: str,
    created_at: datetime | str,
    deadline_at: datetime | str | None = None,
    now: datetime | None = None,
) -> StopState:
    pod_id = validate_pod_id(pod_id)
    created = _parse_state_created_at(created_at)
    requested_deadline = (
        _coerce_utc(deadline_at, "deadline_at") if deadline_at is not None else None
    )
    clock = _coerce_utc(now, "now") if now is not None else datetime.now(timezone.utc)
    state_path = Path(path)
    if state_path.exists():
        existing = load_state(state_path)
        if existing.pod_id != pod_id:
            raise StopStateError("persisted state belongs to a different Pod ID")
        if (
            existing.created_at != created.datetime
            or existing.created_at_ns != created.epoch_ns
        ):
            raise StopStateError("persisted created_at cannot be changed")
        if requested_deadline is not None and existing.deadline_at != requested_deadline:
            raise StopStateError("persisted deadline cannot be reset or extended")
        return existing

    deadline = requested_deadline or (
        created.datetime + timedelta(seconds=MAX_SESSION_SECONDS)
    )
    state = _validate_state(
        StopState(
            pod_id=pod_id,
            created_at=created.datetime,
            deadline_at=deadline,
            status="ARMED",
            updated_at=clock,
            created_at_ns=created.epoch_ns,
        )
    )
    save_state(state_path, state)
    return state


class StopController:
    def __init__(
        self,
        state_path: Path | str,
        *,
        pod_id: str,
        provider: Provider,
        clock: Clock | None = None,
        monotonic: Callable[[], float] | None = None,
        readback_timeout_seconds: float = DEFAULT_READBACK_TIMEOUT_SECONDS,
        readback_poll_interval_seconds: float = DEFAULT_READBACK_POLL_SECONDS,
        command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        self.state_path = Path(state_path)
        self.pod_id = validate_pod_id(pod_id)
        self.provider = provider
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self.readback_timeout_seconds = _positive_finite(
            readback_timeout_seconds,
            "readback_timeout_seconds",
            maximum=MAX_READBACK_SECONDS,
        )
        self.readback_poll_interval_seconds = _positive_finite(
            readback_poll_interval_seconds,
            "readback_poll_interval_seconds",
        )
        self.command_timeout_seconds = _positive_finite(
            command_timeout_seconds,
            "command_timeout_seconds",
        )

    def _now(self, value: datetime | None) -> datetime:
        if value is None:
            value = self._clock()
        return _coerce_utc(value, "now")

    def _now_precise(self) -> datetime:
        """Current UTC time for readback budgets, preserving microseconds."""

        return _coerce_utc_precise(self._clock(), "now")

    def load(self) -> StopState:
        state = load_state(self.state_path)
        if state.pod_id != self.pod_id:
            raise StopStateError("state Pod ID does not match the selected Pod ID")
        return state

    def initialize(
        self,
        *,
        created_at: datetime | str,
        deadline_at: datetime | str | None = None,
    ) -> StopState:
        return initialize_state(
            self.state_path,
            pod_id=self.pod_id,
            created_at=created_at,
            deadline_at=deadline_at,
            now=self._now(None),
        )

    def _blocked(self, state: StopState, message: str) -> StopOutcome:
        safe = _safe_message(message)
        updated = replace(
            state,
            status="LIVE_BLOCKED",
            updated_at=self._now(None),
            last_error=safe,
        )
        save_state(self.state_path, updated)
        return StopOutcome("LIVE_BLOCKED", safe, updated)

    def _stopped(self, state: StopState, message: str) -> StopOutcome:
        updated = replace(
            state,
            status="STOPPED",
            updated_at=self._now(None),
            last_error=None,
        )
        save_state(self.state_path, updated)
        return StopOutcome("STOPPED", message, updated)

    def _identity_matches(self, pod: ProviderPod | None, state: StopState) -> str | None:
        if pod is None:
            return "provider readback is missing the selected Pod"
        if pod.id != self.pod_id:
            return (
                "provider readback returned a different Pod ID; "
                "refusing to claim STOPPED"
            )
        if pod.created_at_ns is None and pod.created_at is None:
            return (
                "provider readback did not confirm Pod createdAt identity; "
                "refusing to claim STOPPED"
            )
        try:
            provider_ns = (
                pod.created_at_ns
                if pod.created_at_ns is not None
                else _datetime_to_ns(pod.created_at)  # type: ignore[arg-type]
            )
            state_ns = (
                state.created_at_ns
                if state.created_at_ns is not None
                else _datetime_to_ns(state.created_at)
            )
        except StopStateError:
            return "provider readback createdAt is not a valid UTC timestamp"
        if (
            isinstance(provider_ns, bool)
            or not isinstance(provider_ns, int)
            or provider_ns < 0
        ):
            return "provider readback createdAt is not a valid UTC timestamp"
        if provider_ns != state_ns:
            return (
                "provider readback createdAt does not match the persisted Pod "
                "identity; refusing to claim STOPPED"
            )
        return None

    @staticmethod
    def _canonical_status(pod: ProviderPod) -> str:
        return str(pod.status).strip().upper()

    def _pre_read(self, state: StopState) -> StopOutcome | None:
        """Read and verify identity before the first stop request.

        A missing/wrong ID, missing/mismatched createdAt or an ineligible
        status must never issue a stop.  A provider-confirmed STOPPED Pod is
        accepted without a redundant stop only after identity verification.
        """

        try:
            pod = self.provider.get(
                self.pod_id,
                timeout_seconds=self.command_timeout_seconds,
            )
        except ProviderError as exc:
            return self._blocked(state, f"pre-stop readback failed: {exc}")
        except Exception as exc:
            return self._blocked(
                state,
                f"provider pre-stop readback error: {type(exc).__name__}",
            )

        identity_error = self._identity_matches(pod, state)
        if identity_error is not None:
            return self._blocked(state, identity_error)

        assert pod is not None  # narrowed by _identity_matches
        status = self._canonical_status(pod)
        if status == STATUS_STOPPED:
            return self._stopped(
                state,
                "provider readback confirmed STOPPED before any stop request",
            )
        if status not in (STATUS_RUNNING, STATUS_INITIALIZING, STATUS_UNKNOWN):
            return self._blocked(
                state,
                f"pre-stop readback status is {status}, not eligible for stop",
            )
        return None

    def _begin_stop(
        self, state: StopState, *, sleeper: Sleeper | None
    ) -> StopOutcome:
        """Persist intent before the external stop, then poll readback.

        If the process dies after the intent is persisted but before/while the
        stop is sent, a restart will only read back.  It will not blindly send
        a second stop: if the Pod is still running, the controller reports
        LIVE_BLOCKED for manual intervention.
        """

        now = self._now(None)
        readback_now = self._now_precise()
        requested = replace(
            state,
            status="STOP_REQUESTED",
            updated_at=now,
            last_error=None,
            readback_started_at=readback_now,
            readback_deadline_at=readback_now
            + timedelta(seconds=self.readback_timeout_seconds),
        )
        save_state(self.state_path, requested)

        try:
            self.provider.stop(
                self.pod_id,
                timeout_seconds=self.command_timeout_seconds,
            )
        except ProviderError as exc:
            return self._blocked(requested, f"provider stop failed: {exc}")
        except Exception as exc:
            return self._blocked(
                requested,
                f"provider stop error: {type(exc).__name__}",
            )

        return self._confirm_stopped(requested, sleeper=sleeper)

    def _confirm_stopped(
        self, state: StopState, *, sleeper: Sleeper | None
    ) -> StopOutcome:
        readback_deadline = state.readback_deadline_at
        if state.readback_started_at is None or readback_deadline is None:
            return self._blocked(
                state,
                "stop intent lacks a persisted precise readback anchor; "
                "manual intervention required",
            )

        sleep = sleeper or time.sleep
        now = self._now_precise()
        remaining_wall = max(
            0.0, (readback_deadline - now).total_seconds()
        )
        monotonic_deadline = self._monotonic() + remaining_wall
        max_attempts = min(
            MAX_READBACK_ATTEMPTS,
            max(
                2,
                int(
                    max(self.readback_timeout_seconds, remaining_wall)
                    / self.readback_poll_interval_seconds
                )
                + 2,
            ),
        )
        last_status: str | None = None
        last_error: str | None = None
        attempts = 0

        if remaining_wall <= 0:
            return self._blocked(
                state,
                self._readback_failure_message(
                    last_status, last_error, "persisted readback deadline"
                ),
            )

        while True:
            attempts += 1
            if attempts > max_attempts:
                return self._blocked(
                    state,
                    self._readback_failure_message(
                        last_status, last_error, "attempt limit"
                    ),
                )

            now = self._now_precise()
            remaining = min(
                monotonic_deadline - self._monotonic(),
                (readback_deadline - now).total_seconds(),
            )
            if remaining <= 0:
                return self._blocked(
                    state,
                    self._readback_failure_message(
                        last_status, last_error, "persisted readback deadline"
                    ),
                )

            command_timeout = min(remaining, self.command_timeout_seconds)
            try:
                pod = self.provider.get(
                    self.pod_id,
                    timeout_seconds=command_timeout,
                )
            except ProviderTransientError as exc:
                last_error = _safe_message(exc)
            except ProviderError as exc:
                return self._blocked(state, _safe_message(exc))
            except Exception as exc:
                return self._blocked(
                    state,
                    f"provider readback error: {type(exc).__name__}",
                )
            else:
                now_after = self._now_precise()
                remaining_after = min(
                    monotonic_deadline - self._monotonic(),
                    (readback_deadline - now_after).total_seconds(),
                )
                if remaining_after <= 0:
                    return self._blocked(
                        state,
                        self._readback_failure_message(
                            last_status,
                            last_error,
                            "persisted readback deadline",
                        ),
                    )
                identity_error = self._identity_matches(pod, state)
                if identity_error is not None:
                    return self._blocked(state, identity_error)
                assert pod is not None  # narrowed by _identity_matches
                status = self._canonical_status(pod)
                last_status = status
                if status == STATUS_STOPPED:
                    return self._stopped(
                        state, "provider readback confirmed STOPPED"
                    )
                if status in (STATUS_TERMINATED, STATUS_ERROR):
                    return self._blocked(
                        state,
                        f"provider readback status is {status}, not STOPPED",
                    )
                if status not in (
                    STATUS_INITIALIZING,
                    STATUS_RUNNING,
                    STATUS_UNKNOWN,
                ):
                    return self._blocked(
                        state,
                        f"provider readback status is {status}, not recognized",
                    )
                last_error = None

            now = self._now_precise()
            remaining = min(
                monotonic_deadline - self._monotonic(),
                (readback_deadline - now).total_seconds(),
            )
            if remaining <= 0:
                return self._blocked(
                    state,
                    self._readback_failure_message(
                        last_status, last_error, "persisted readback deadline"
                    ),
                )
            sleep(min(self.readback_poll_interval_seconds, remaining))

    @staticmethod
    def _readback_failure_message(
        last_status: str | None,
        last_error: str | None,
        reason: str,
    ) -> str:
        parts = [
            f"readback did not confirm STOPPED within the {reason}",
        ]
        if last_status is not None:
            parts.append(f"last status: {last_status}")
        if last_error is not None:
            parts.append(f"last read error: {last_error}")
        return "; ".join(parts)

    def evaluate(
        self,
        *,
        now: datetime | None = None,
        explicit: bool = False,
        sleeper: Sleeper | None = None,
    ) -> StopOutcome:
        with _state_lock(self.state_path):
            return self._evaluate_locked(
                now=now, explicit=explicit, sleeper=sleeper
            )

    def _evaluate_locked(
        self,
        *,
        now: datetime | None = None,
        explicit: bool = False,
        sleeper: Sleeper | None = None,
    ) -> StopOutcome:
        state = self.load()
        current = self._now(now)
        if state.status == "STOPPED":
            return StopOutcome("STOPPED", "provider-confirmed stopped already", state)
        if state.status == "LIVE_BLOCKED":
            return StopOutcome(
                "LIVE_BLOCKED",
                state.last_error
                or "previous stop action is not provider-confirmed; manual "
                "intervention required",
                state,
            )
        if state.status == "STOP_REQUESTED":
            return self._confirm_stopped(state, sleeper=sleeper)
        if current < state.deadline_at and not explicit:
            return StopOutcome(
                "ARMED",
                "before deadline; no provider operation performed",
                state,
            )

        pre_read = self._pre_read(state)
        if pre_read is not None:
            return pre_read
        return self._begin_stop(state, sleeper=sleeper)

    def watch(
        self,
        *,
        poll_interval_seconds: float = 30.0,
        sleeper: Sleeper | None = None,
    ) -> StopOutcome:
        """Wait until the persisted deadline, then stop and confirm by readback.

        The lock is not held while sleeping. ``sleeper`` is injectable so tests
        advance a fake clock without real delays. A restart with the same state
        deadline evaluates immediately when already overdue. A persisted stop
        intent is resumed as readback-only; it never causes a second stop.
        """

        poll_interval = _positive_finite(
            poll_interval_seconds, "poll_interval_seconds"
        )
        sleep = sleeper or time.sleep
        while True:
            state = self.load()
            current = self._now(None)
            if state.status == "STOPPED":
                return StopOutcome(
                    "STOPPED",
                    "provider-confirmed stopped already",
                    state,
                )
            if state.status == "LIVE_BLOCKED":
                return StopOutcome(
                    "LIVE_BLOCKED",
                    state.last_error
                    or "previous stop action is not provider-confirmed; manual "
                    "intervention required",
                    state,
                )
            if state.status == "STOP_REQUESTED":
                return self.evaluate(now=current, sleeper=sleeper)
            if current >= state.deadline_at:
                return self.evaluate(now=current, sleeper=sleeper)
            remaining = (state.deadline_at - current).total_seconds()
            sleep(min(poll_interval, remaining))

    def dry_run_view(
        self,
        *,
        created_at: datetime | str | None = None,
        deadline_at: datetime | str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if self.state_path.exists():
            state = self.load()
            if created_at is not None:
                created = _parse_state_created_at(created_at)
                if (
                    created.datetime != state.created_at
                    or created.epoch_ns != state.created_at_ns
                ):
                    raise StopStateError("persisted created_at cannot be changed")
            if deadline_at is not None and _coerce_utc(deadline_at, "deadline_at") != state.deadline_at:
                raise StopStateError("persisted deadline cannot be reset or extended")
            return _state_to_dict(state)
        if created_at is None:
            raise StopStateError("created_at is required when no state exists")
        created = _parse_state_created_at(created_at)
        deadline = (
            _coerce_utc(deadline_at, "deadline_at")
            if deadline_at is not None
            else created.datetime + timedelta(seconds=MAX_SESSION_SECONDS)
        )
        _validate_state(
            StopState(
                pod_id=self.pod_id,
                created_at=created.datetime,
                deadline_at=deadline,
                status="ARMED",
                updated_at=self._now(now),
                created_at_ns=created.epoch_ns,
            )
        )
        return {
            "schema": STATE_SCHEMA,
            "pod_id": self.pod_id,
            "created_at": _format_utc(created.datetime),
            "created_at_ns": created.epoch_ns,
            "deadline_at": _format_utc(deadline),
            "status": "ARMED",
            "updated_at": _format_utc(self._now(now)),
            "last_error": None,
            "readback_started_at": None,
            "readback_deadline_at": None,
            "note": "dry-run: no provider call, no state write, no credential read",
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/p1/stop_pod.py",
        description="Fail-closed external P1 Pod stop controller; dry-run by default.",
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--created-at", default=None, help="provider-reported UTC timestamp")
    parser.add_argument("--deadline", default=None, help="absolute UTC deadline, at most +60m")
    parser.add_argument("--explicit-stop", action="store_true")
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "single evaluation without deadline watch; still performs the full "
            "bounded stop/readback sequence"
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="watch sleep interval in seconds, capped by remaining time",
    )
    parser.add_argument(
        "--readback-timeout",
        type=float,
        default=DEFAULT_READBACK_TIMEOUT_SECONDS,
        help="absolute readback bound in seconds; persisted at the first attempt",
    )
    parser.add_argument(
        "--readback-poll-interval",
        type=float,
        default=DEFAULT_READBACK_POLL_SECONDS,
        help="readback poll interval in seconds, capped by remaining bound",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runpodctl", default="runpodctl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        provider = RunpodctlProvider(args.runpodctl) if args.execute else None
        controller_kwargs = {
            "readback_timeout_seconds": args.readback_timeout,
            "readback_poll_interval_seconds": args.readback_poll_interval,
        }
        if provider is None:
            # Dry-run must not instantiate a provider or inspect credentials.
            class _NoProvider:
                pass

            controller = StopController(
                args.state,
                pod_id=args.pod_id,
                provider=_NoProvider(),  # type: ignore[arg-type]
                **controller_kwargs,
            )
            view = controller.dry_run_view(
                created_at=args.created_at,
                deadline_at=args.deadline,
            )
            print(json.dumps(view, indent=2, sort_keys=True))
            return 0

        # Explicit execute validates the installed CLI version BEFORE creating
        # state or entering the deadline watch.  No account/Pod call is made.
        try:
            provider.verify_contract()
        except ProviderError as exc:
            print(
                json.dumps(
                    {
                        "status": "LIVE_BLOCKED",
                        "message": _safe_message(exc),
                        "state": None,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 3

        controller = StopController(
            args.state,
            pod_id=args.pod_id,
            provider=provider,
            **controller_kwargs,
        )
        if not Path(args.state).exists():
            if args.created_at is None:
                raise StopStateError("--created-at is required to initialize state")
            controller.initialize(
                created_at=args.created_at,
                deadline_at=args.deadline,
            )
        if args.explicit_stop or args.once:
            outcome = controller.evaluate(explicit=args.explicit_stop)
        else:
            outcome = controller.watch(poll_interval_seconds=args.poll_interval)
        print(
            json.dumps(
                {
                    "status": outcome.status,
                    "message": outcome.message,
                    "state": _state_to_dict(outcome.state),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if outcome.status == "STOPPED" else 3
    except (StopStateError, StopControllerError, ProviderError) as exc:
        print(f"stop controller refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
