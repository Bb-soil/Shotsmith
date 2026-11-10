"""Isolated provider bridge for the external P1 stop controller.

The readback contract is pinned to ``runpodctl`` v3.0.1
(commit ``551d6a4310767ad9671961bea9f64cb2c365dc1e``):

* version probe: ``runpodctl --version`` -> ``runpodctl v3.0.1``;
* readback: ``runpodctl pod get <POD_ID> --output json``;
* stop: ``runpodctl pod stop <POD_ID>``.

``pod get`` prints a JSON object with ``desiredStatus`` (what was asked for),
``runtimeStatus`` (what the provider actually reports), and an optional
``createdAt``.  A successful stop command is only an accepted request; it is
never treated as proof that the Pod stopped.

The bridge is deliberately small: no shell, no create/restart/delete/payment,
no provider endpoint calls of its own.  The version gate is not a mutable
boolean: ``get``/``stop`` refuse to run until ``verify_contract`` has actually
probed the installed executable in this process.
"""

from __future__ import annotations

import calendar
import json
import math
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

Runner = Callable[..., Any]

SUPPORTED_RUNPODCTL_VERSIONS = frozenset({"v3.0.1"})
MAX_OUTPUT_BYTES = 1_000_000
POD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
_VERSION_RE = re.compile(r"^runpodctl\s+(v?\d+\.\d+\.\d+)\s*$")
_ISO_TIME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ](?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?"
    r"(?P<offset>Z|[+-]\d{2}:?\d{2})$"
)
_GO_TIME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))? "
    r"(?P<offset>[+-]\d{4}) [A-Za-z]{2,5}$"
)
_SAFE_REASONS = frozenset({"awaiting_container", "runtime_unavailable"})

# Finite error vocabulary pinned from the v3.0.1 README/source.  Anything else
# is reported as the fixed token ``unknown_error`` and the CLI text is dropped.
_KNOWN_ERROR_CODES = frozenset(
    {
        "api_error",
        "bad_request",
        "cli_error",
        "conflict",
        "create_uncertain",
        "forbidden",
        "graphql_error",
        "job_failed",
        "network_error",
        "no_credentials",
        "not_found",
        "rate_limited",
        "server_error",
        "timeout",
        "unauthorized",
        "usage_error",
        "wait_interrupted",
        "wait_timeout",
    }
)

STATUS_INITIALIZING = "INITIALIZING"
STATUS_RUNNING = "RUNNING"
STATUS_STOPPED = "STOPPED"
STATUS_TERMINATED = "TERMINATED"
STATUS_ERROR = "ERROR"
STATUS_UNKNOWN = "UNKNOWN"

_DESIRED_STATUSES = frozenset({"RUNNING", "EXITED", "TERMINATED", "ERROR"})
_RUNTIME_STATUSES = frozenset(
    {"running", "initializing", "stopped", "terminated", "error", "unknown"}
)
_AUTH_CODES = frozenset({"no_credentials", "unauthorized", "forbidden"})
_TRANSIENT_CODES = frozenset(
    {"network_error", "rate_limited", "server_error", "timeout"}
)


class ProviderError(Exception):
    """The provider operation failed or could not be trusted."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class ProviderContractUnverified(ProviderError):
    """The CLI version/schema is not verified or is unsupported."""


class ProviderVersionUnsupported(ProviderContractUnverified):
    """The installed runpodctl version is outside the pinned allowlist."""


class ProviderProtocolError(ProviderContractUnverified):
    """The CLI output does not match the pinned readback contract."""


class ProviderAuthError(ProviderError):
    """Authentication failed; retrying without intervention is unsafe."""


class ProviderNotFound(ProviderError):
    """The requested provider resource does not exist."""


class ProviderTransientError(ProviderError):
    """A network/timeout/rate/server failure that may be retried in a bound."""


@dataclass(frozen=True)
class ProviderPod:
    """Normalized readback fields only; raw CLI JSON is never retained."""

    id: str
    status: str
    created_at: datetime | None = None
    reason: str | None = None
    created_at_ns: int | None = None
    created_at_text: str | None = None


@dataclass(frozen=True)
class ParsedTimestamp:
    """UTC timestamp with full source precision, without float arithmetic."""

    datetime: datetime
    epoch_ns: int
    text: str


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


class Provider(Protocol):
    def verify_contract(self) -> str:
        ...

    def get(
        self, pod_id: str, *, timeout_seconds: float | None = None
    ) -> ProviderPod | None:
        ...

    def stop(self, pod_id: str, *, timeout_seconds: float | None = None) -> None:
        ...


def _validate_pod_id(value: str) -> str:
    if not isinstance(value, str) or not POD_ID_RE.fullmatch(value):
        raise ProviderProtocolError("pod id is not a valid RunPod Pod ID")
    return value


def parse_utc_timestamp(value: str) -> ParsedTimestamp | None:
    """Parse source/RFC3339 timestamps while retaining nanosecond precision.

    ``pod get`` renders rest v2's RFC3339 ``createdAt`` through Go's
    ``time.String()`` (for example ``2026-10-02 05:28:38.366 +0000 UTC``).
    RFC3339 is also accepted.  Fractions are kept as an integer nanosecond
    epoch; no float arithmetic is used.  Unknown formats are rejected rather
    than guessed.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    match = _ISO_TIME_RE.fullmatch(text)
    if match is not None:
        date = match.group("date")
        clock = match.group("time")
    else:
        match = _GO_TIME_RE.fullmatch(text)
        if match is None:
            return None
        date = match.group("date")
        clock = date[11:]
        date = date[:10]

    fraction = match.group("fraction") or ""
    offset = match.group("offset")
    if offset == "Z":
        offset = "+00:00"
    elif offset and offset[3] != ":":
        offset = offset[:3] + ":" + offset[3:]
    try:
        parsed = datetime.strptime(f"{date} {clock} {offset}", "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return None

    utc = parsed.astimezone(timezone.utc).replace(microsecond=0)
    fraction_ns = int(fraction.ljust(9, "0")) if fraction else 0
    epoch_ns = calendar.timegm(utc.utctimetuple()) * 1_000_000_000 + fraction_ns
    canonical = utc.strftime("%Y-%m-%dT%H:%M:%S")
    if fraction_ns:
        canonical += f".{fraction_ns:09d}"
    canonical += "Z"
    return ParsedTimestamp(datetime=utc, epoch_ns=epoch_ns, text=canonical)


def _combine_status(desired: str, runtime: str, reason: str | None) -> str:
    if desired == "RUNNING":
        if runtime == "running":
            if reason is not None:
                raise ProviderProtocolError("runtimeStatusReason is not valid for running")
            return STATUS_RUNNING
        if runtime == "initializing":
            if reason not in (None, "awaiting_container"):
                raise ProviderProtocolError(
                    "runtimeStatusReason is not valid for initializing"
                )
            return STATUS_INITIALIZING
        if runtime == "unknown":
            if reason != "runtime_unavailable":
                raise ProviderProtocolError(
                    "unknown runtimeStatus requires runtime_unavailable"
                )
            return STATUS_UNKNOWN
    if desired == "EXITED" and runtime == "stopped" and reason is None:
        return STATUS_STOPPED
    if desired == "TERMINATED" and runtime == "terminated" and reason is None:
        return STATUS_TERMINATED
    if desired == "ERROR" and runtime == "error" and reason is None:
        return STATUS_ERROR
    raise ProviderProtocolError(
        "desiredStatus and runtimeStatus do not describe a known Pod state"
    )


def _safe_error_details(stderr: str) -> tuple[str | None, int | None]:
    """Return only a whitelisted code and numeric HTTP status.

    The CLI's human-readable ``error`` text is deliberately never returned:
    it may contain provider data and must not reach exceptions, state or logs.
    """

    text = stderr.strip()
    if not text:
        return None, None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    code = payload.get("code")
    status = payload.get("status")
    if not isinstance(code, str) or code not in _KNOWN_ERROR_CODES:
        code = "unknown_error"
    if isinstance(status, bool) or not isinstance(status, int):
        status = None
    if status is not None and not 100 <= status <= 599:
        status = None
    return code, status


def _failure_message(action: str, code: str | None, http_status: int | None) -> str:
    details: list[str] = []
    if code:
        details.append(f"code={code}")
    if http_status is not None:
        details.append(f"status={http_status}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"runpodctl {action} failed{suffix}"


def _close_stream(stream: Any) -> None:
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def _terminate_and_reap(process: Any) -> None:
    try:
        process.terminate()
    except (OSError, ValueError):
        pass
    try:
        process.wait(timeout=1.0)
        return
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    try:
        process.kill()
    except (OSError, ValueError):
        pass
    try:
        process.wait(timeout=2.0)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass


def _thread_alive(thread: Any) -> bool:
    try:
        return bool(thread.is_alive())
    except BaseException:
        return False


def _cleanup_process(process: Any, threads: list[Any]) -> None:
    """Kill/reap/close/join without masking a propagating BaseException.

    Idempotent: safe to call after an earlier partial cleanup.  Every resource
    is attempted independently, so one cleanup failure does not abandon the
    others.
    """

    if process is not None:
        try:
            _terminate_and_reap(process)
        except BaseException:
            pass
        try:
            _close_stream(process.stdout)
        except BaseException:
            pass
        try:
            _close_stream(process.stderr)
        except BaseException:
            pass
    for thread in threads:
        try:
            thread.join(timeout=2.0)
        except BaseException:
            pass
    if any(_thread_alive(thread) for thread in threads):
        if process is not None:
            try:
                _close_stream(process.stdout)
            except BaseException:
                pass
            try:
                _close_stream(process.stderr)
            except BaseException:
                pass
        for thread in threads:
            try:
                thread.join(timeout=1.0)
            except BaseException:
                pass


def _finish_readers(process: Any, threads: list[Any]) -> None:
    for thread in threads:
        thread.join(timeout=2.0)
    if any(_thread_alive(thread) for thread in threads):
        _close_stream(process.stdout)
        _close_stream(process.stderr)
        for thread in threads:
            thread.join(timeout=1.0)
    if any(_thread_alive(thread) for thread in threads):
        _cleanup_process(process, threads)
        raise ProviderError("runpodctl output readers did not stop")


def _run_bounded_process(
    args: list[str],
    *,
    timeout_seconds: float,
    max_output_bytes: int,
    popen_factory: Callable[..., Any] | None = None,
    thread_factory: Callable[..., Any] | None = None,
) -> ProcessResult:
    """Run one explicit-argv process with concurrent bounded stream reads.

    Both stdout and stderr are read concurrently.  A stream is capped before
    the next chunk is appended, so a chatty process cannot force unbounded
    accumulation.  On timeout/cap the child is terminated, killed if needed,
    reaped, streams are closed and reader threads are joined.  No temporary
    files are used.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ProviderError("runpodctl timeout must be a finite positive number")
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
    ):
        raise ProviderError("runpodctl output limit must be a positive integer")

    factory = popen_factory or subprocess.Popen
    started = time.monotonic()
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    overflow_event = threading.Event()
    overflow_labels: list[str] = []
    overflow_lock = threading.Lock()
    started_threads: list[Any] = []
    process: Any = None
    timed_out = False
    abnormal = False

    def reader(stream: Any, buffer: bytearray, label: str) -> None:
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                if not isinstance(chunk, (bytes, bytearray)):
                    raise ProviderProtocolError(
                        f"runpodctl {label} had an invalid type"
                    )
                room = max_output_bytes - len(buffer)
                if len(chunk) > room:
                    if room > 0:
                        buffer.extend(chunk[:room])
                    with overflow_lock:
                        if not overflow_labels:
                            overflow_labels.append(label)
                    overflow_event.set()
                    break
                buffer.extend(chunk)
        except (OSError, ValueError):
            pass
        except Exception:
            with overflow_lock:
                if not overflow_labels:
                    overflow_labels.append(label)
            overflow_event.set()
        finally:
            _close_stream(stream)

    try:
        process = factory(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
        )
        make_thread = thread_factory or threading.Thread
        stdout_thread = make_thread(
            target=reader,
            args=(process.stdout, stdout_buffer, "stdout"),
            name="runpodctl-stdout-reader",
            daemon=False,
        )
        stderr_thread = make_thread(
            target=reader,
            args=(process.stderr, stderr_buffer, "stderr"),
            name="runpodctl-stderr-reader",
            daemon=False,
        )
        stdout_thread.start()
        started_threads.append(stdout_thread)
        stderr_thread.start()
        started_threads.append(stderr_thread)

        deadline = started + float(timeout_seconds)
        while True:
            if overflow_event.is_set():
                abnormal = True
                break
            if process.poll() is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                abnormal = True
                break
            overflow_event.wait(min(0.02, remaining))

        if abnormal:
            _cleanup_process(process, started_threads)
        else:
            _finish_readers(process, started_threads)
    except BaseException as exc:
        # One ownership scope after Popen: thread factory/build/start, poll,
        # wait/join and cancellation all clean up the child, streams and only
        # started threads.  Cleanup itself is idempotent and never masks this.
        _cleanup_process(process, started_threads)
        if isinstance(exc, FileNotFoundError):
            raise ProviderContractUnverified(
                "runpodctl executable was not found"
            ) from None
        if isinstance(exc, (OSError, subprocess.SubprocessError)):
            raise ProviderError(
                f"runpodctl command could not start ({type(exc).__name__})"
            ) from None
        raise

    if overflow_event.is_set():
        label = overflow_labels[0] if overflow_labels else "output"
        raise ProviderProtocolError(
            f"runpodctl {label} exceeded the output limit"
        )
    if timed_out:
        raise ProviderTransientError("runpodctl command timed out")

    try:
        stdout = bytes(stdout_buffer).decode("utf-8", errors="strict")
        stderr = bytes(stderr_buffer).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ProviderProtocolError("runpodctl output was not valid UTF-8") from None

    returncode = getattr(process, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise ProviderProtocolError("runpodctl did not provide an exit status")
    return ProcessResult(returncode=returncode, stdout=stdout, stderr=stderr)


class RunpodctlProvider:
    """Minimal runpodctl bridge; no create/restart/delete/payment operations."""

    def __init__(
        self,
        executable: str = "runpodctl",
        *,
        runner: Runner | None = None,
        popen_factory: Callable[..., Any] | None = None,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes <= 0
        ):
            raise ValueError("max_output_bytes must be a positive integer")
        self.executable = executable
        self.runner = runner
        self._popen_factory = popen_factory
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self._verified_version: str | None = None

    def verify_contract(self) -> str:
        """Probe the installed CLI and accept only the pinned version."""

        result = self._run(
            [self.executable, "--version"],
            timeout_seconds=self.timeout_seconds,
        )
        returncode = self._returncode(result)
        if returncode != 0:
            code, status = _safe_error_details(self._stderr(result))
            raise ProviderVersionUnsupported(
                _failure_message("version", code, status),
                code=code,
                http_status=status,
            )
        output = self._stdout(result).strip()
        match = _VERSION_RE.fullmatch(output)
        if match is None:
            raise ProviderVersionUnsupported(
                "runpodctl version output is not recognized"
            )
        version = match.group(1)
        if not version.startswith("v"):
            version = "v" + version
        if version not in SUPPORTED_RUNPODCTL_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_RUNPODCTL_VERSIONS))
            raise ProviderVersionUnsupported(
                f"runpodctl {version} is not supported; expected {supported}"
            )
        self._verified_version = version
        return version

    def get(
        self, pod_id: str, *, timeout_seconds: float | None = None
    ) -> ProviderPod | None:
        validated_id = _validate_pod_id(pod_id)
        self._require_verified()
        result = self._run(
            [self.executable, "pod", "get", validated_id, "--output", "json"],
            timeout_seconds=timeout_seconds,
        )
        returncode = self._returncode(result)
        if returncode != 0:
            code, status = _safe_error_details(self._stderr(result))
            if code == "not_found":
                return None
            self._raise_failure("get", code, status)
        return self._parse_payload(self._stdout(result), validated_id)

    def stop(self, pod_id: str, *, timeout_seconds: float | None = None) -> None:
        validated_id = _validate_pod_id(pod_id)
        self._require_verified()
        result = self._run(
            [self.executable, "pod", "stop", validated_id],
            timeout_seconds=timeout_seconds,
        )
        returncode = self._returncode(result)
        if returncode != 0:
            code, status = _safe_error_details(self._stderr(result))
            self._raise_failure("stop", code, status)
        # A successful stop request is not proof of stopped.  Read stdout only
        # to enforce the output bound; never parse or retain it as state.
        self._stdout(result)

    def _require_verified(self) -> None:
        if self._verified_version is None:
            raise ProviderContractUnverified(
                "runpodctl contract is not verified; call verify_contract first"
            )

    def _run(self, args: list[str], *, timeout_seconds: float | None) -> ProcessResult:
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ProviderError("runpodctl timeout must be a finite positive number")

        if self.runner is None:
            return _run_bounded_process(
                args,
                timeout_seconds=float(timeout),
                max_output_bytes=self.max_output_bytes,
                popen_factory=self._popen_factory,
            )

        try:
            result = self.runner(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                check=False,
                timeout=float(timeout),
                shell=False,
            )
        except subprocess.TimeoutExpired:
            raise ProviderTransientError("runpodctl command timed out") from None
        except FileNotFoundError:
            raise ProviderContractUnverified(
                "runpodctl executable was not found"
            ) from None
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProviderError(
                f"runpodctl command could not run ({type(exc).__name__})"
            ) from None
        except Exception as exc:  # runner failures must not leak raw text
            raise ProviderError(
                f"runpodctl command could not run ({type(exc).__name__})"
            ) from None

        returncode = self._returncode(result)
        # Even an injected runner is postvalidated on BOTH streams before any
        # command-specific parsing, so an oversized/secret stderr is not missed.
        stdout = self._stdout(result)
        stderr = self._stderr(result)
        return ProcessResult(returncode=returncode, stdout=stdout, stderr=stderr)

    def _returncode(self, result: Any) -> int:
        returncode = getattr(result, "returncode", None)
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise ProviderProtocolError("runpodctl did not provide an exit status")
        return returncode

    def _text(self, value: Any, label: str) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                raise ProviderProtocolError(
                    f"runpodctl {label} was not valid UTF-8"
                ) from None
        if not isinstance(value, str):
            raise ProviderProtocolError(f"runpodctl {label} had an invalid type")
        if len(value.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ProviderProtocolError(
                f"runpodctl {label} exceeded the output limit"
            )
        return value

    def _stdout(self, result: Any) -> str:
        return self._text(getattr(result, "stdout", None), "stdout")

    def _stderr(self, result: Any) -> str:
        return self._text(getattr(result, "stderr", None), "stderr")

    def _raise_failure(
        self, action: str, code: str | None, http_status: int | None
    ) -> None:
        message = _failure_message(action, code, http_status)
        if code in _AUTH_CODES:
            raise ProviderAuthError(
                message, code=code, http_status=http_status
            )
        if code == "not_found":
            raise ProviderNotFound(message, code=code, http_status=http_status)
        if code in _TRANSIENT_CODES:
            raise ProviderTransientError(
                message, code=code, http_status=http_status
            )
        raise ProviderError(message, code=code, http_status=http_status)

    def _parse_payload(self, stdout: str, expected_id: str) -> ProviderPod:
        try:
            payload = json.loads(stdout)
        except (TypeError, ValueError):
            raise ProviderProtocolError(
                "runpodctl pod get returned malformed JSON"
            ) from None
        if not isinstance(payload, dict):
            raise ProviderProtocolError(
                "runpodctl pod get returned a non-object JSON payload"
            )

        raw_id = payload.get("id")
        if not isinstance(raw_id, str) or not raw_id:
            raise ProviderProtocolError(
                "runpodctl pod get did not return a non-empty Pod ID"
            )
        if POD_ID_RE.fullmatch(raw_id) is None:
            raise ProviderProtocolError(
                "runpodctl pod get returned an invalid Pod ID"
            )
        if raw_id != expected_id:
            raise ProviderProtocolError(
                "runpodctl pod get returned a different Pod ID"
            )

        desired = payload.get("desiredStatus")
        runtime = payload.get("runtimeStatus")
        if not isinstance(desired, str) or desired not in _DESIRED_STATUSES:
            raise ProviderProtocolError(
                "runpodctl pod get returned an unknown desiredStatus"
            )
        if not isinstance(runtime, str) or runtime not in _RUNTIME_STATUSES:
            raise ProviderProtocolError(
                "runpodctl pod get returned an unknown runtimeStatus"
            )

        reason = payload.get("runtimeStatusReason")
        if reason is not None:
            if not isinstance(reason, str) or reason not in _SAFE_REASONS:
                raise ProviderProtocolError(
                    "runpodctl pod get returned an unknown runtimeStatusReason"
                )
        status = _combine_status(desired, runtime, reason)

        created_at: datetime | None = None
        created_at_ns: int | None = None
        created_at_text: str | None = None
        raw_created = payload.get("createdAt")
        if raw_created is not None:
            if not isinstance(raw_created, str):
                raise ProviderProtocolError(
                    "runpodctl pod get returned an invalid createdAt type"
                )
            parsed = parse_utc_timestamp(raw_created)
            if parsed is None:
                raise ProviderProtocolError(
                    "runpodctl pod get returned an unrecognized createdAt"
                )
            created_at = parsed.datetime
            created_at_ns = parsed.epoch_ns
            created_at_text = parsed.text

        return ProviderPod(
            id=raw_id,
            status=status,
            created_at=created_at,
            reason=reason,
            created_at_ns=created_at_ns,
            created_at_text=created_at_text,
        )


__all__ = [
    "MAX_OUTPUT_BYTES",
    "POD_ID_RE",
    "ParsedTimestamp",
    "ProcessResult",
    "Provider",
    "ProviderAuthError",
    "ProviderContractUnverified",
    "ProviderError",
    "ProviderNotFound",
    "ProviderPod",
    "ProviderProtocolError",
    "ProviderTransientError",
    "ProviderVersionUnsupported",
    "RunpodctlProvider",
    "STATUS_ERROR",
    "STATUS_INITIALIZING",
    "STATUS_RUNNING",
    "STATUS_STOPPED",
    "STATUS_TERMINATED",
    "STATUS_UNKNOWN",
    "SUPPORTED_RUNPODCTL_VERSIONS",
    "parse_utc_timestamp",
]
