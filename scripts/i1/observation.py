"""Read-only I1 observation: typed results, bounded retry, one SSH snapshot.

This module is deliberately separate from the shared P1 transport.  It only
wraps read-only commands.  Mutating operations (mkdir/upload/setup/start/
recover/generate) must never use :func:`run_readonly` or the retry helper.

The snapshot is one authenticated SSH command.  A small POSIX shell wrapper
checks for ``python3`` without requiring it for the shell fallback, then runs a
fixed trusted Python literal.  The literal collects environment facts and the
exact existing-attempt records/process proof.  No remote file is uploaded and
no remote mutation is performed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from . import stock_nodes
from scripts.p1.remote import RemoteError, RemoteHostKeyError, RemoteRunner

SNAPSHOT_SCHEMA = "anime-video-studio/i1-observation-snapshot/v1"
SNAPSHOT_MARKER_RE = re.compile(r"^[0-9a-f]{32}$")
HEX32_RE = re.compile(r"^[0-9a-f]{32}$")

HOME_CODE = "import os;print(os.path.expanduser('~'))"
CONTAINMENT_CODE = (
    "import json,os,sys;"
    "ws=os.path.realpath(sys.argv[1]); root=os.path.join(ws,'.avs-i1');"
    "comfy=os.path.join(ws,'ComfyUI'); venv=os.path.join(ws,'p1-venv');"
    "paths=[root,comfy,venv,os.path.join(venv,'bin'),os.path.join(root,'logs'),os.path.join(root,'scripts'),os.path.join(ws,'p1-evidence'),os.path.join(ws,'p1-evidence','object_info.json'),os.path.join(ws,'p1-evidence','system_stats.json'),os.path.join(ws,'p1-torch-constraints.txt'),os.path.join(root,'logs','setup.log'),os.path.join(root,'logs','torch-before.json'),os.path.join(root,'logs','torch-after.json')];"
    "bad=[p for p in paths if os.path.islink(p)];"
    "esc=[];"
    "[(esc.append(p)) for p in (root,comfy,venv) if os.path.exists(p) and not (os.path.realpath(p)==ws or os.path.realpath(p).startswith(ws+os.sep))];"
    "foreign_venv=os.path.exists(venv) and not os.path.isfile(os.path.join(venv,'pyvenv.cfg'));"
    "print(json.dumps({'symlinks':bad,'outside':esc,'foreign_venv':foreign_venv}))"
)
MEM_CODE = (
    "import json;d={};"
    "[(d.__setitem__(l.split(':',1)[0], int(l.split()[1])*1024)) "
    "for l in open('/proc/meminfo') if l.startswith(('MemTotal:','MemAvailable:'))];"
    "print(json.dumps(d))"
)
TORCH_CODE = (
    "import json,torch;print(json.dumps({"
    "'version':torch.__version__,'cuda':torch.version.cuda,"
    "'available':torch.cuda.is_available()}))"
)

RETRYABLE = frozenset({"TIMEOUT", "TRANSPORT_ERROR", "CONNECTION_LOST"})
NON_RETRYABLE = frozenset(
    {
        "OK",
        "HOST_KEY_CHANGED",
        "AUTH_FAILED",
        "PROTOCOL_ERROR",
        "REMOTE_COMMAND_ERROR",
        "INCONCLUSIVE",
    }
)

HOST_KEY_PATTERNS = (
    "remote host identification has changed",
    "host key verification failed",
    "host key is absent",
    "host key changed",
    "host key has changed",
    "host key mismatch",
    "no matching host key",
    "hostkey verification failed",
)
AUTH_PATTERNS = (
    "permission denied",
    "too many authentication failures",
    "authentication failed",
    "no such identity",
    "load key",
    "invalid format",
)
TRANSPORT_PATTERNS = (
    "connection timed out",
    "banner exchange",
    "connection closed",
    "connection reset",
    "broken pipe",
    "kex_exchange_identification",
    "no route to host",
    "network is unreachable",
    "connection refused",
    "operation timed out",
    "connection attempt failed",
    "software caused connection abort",
)


def guard_sha256() -> str:
    path = Path(__file__).resolve().with_name("process_guard.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard_source_b64() -> str:
    path = Path(__file__).resolve().with_name("process_guard.py")
    return base64.b64encode(zlib.compress(path.read_bytes(), 9)).decode("ascii")


def canonical_argv(argv: Sequence[str]) -> str:
    return json.dumps(
        [str(item) for item in argv], ensure_ascii=False, separators=(",", ":")
    )


@dataclass
class ObservationResult:
    kind: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    attempts: int = 0
    elapsed_seconds: float = 0.0
    snapshot: dict[str, Any] | None = None
    diagnostic: str = ""
    decision: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.kind == "OK"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "attempts": self.attempts,
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
            "diagnostic": self.diagnostic,
            "snapshot_schema": (
                self.snapshot.get("schema") if isinstance(self.snapshot, Mapping) else None
            ),
        }


_SHELL_WRAPPER = (
    'if command -v python3 >/dev/null 2>&1; then '
    'exec python3 -c "$2" "$1" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" "${18}"; '
    "fi; "
    "printf '{\"schema\":\"anime-video-studio/i1-observation-snapshot/v1\","
    "\"marker\":\"%s\",\"run_id\":\"%s\",\"nonce\":\"%s\",\"python3\":false,"
    "\"tool_presence\":{\"sh\":true},"
    "\"commands\":[],\"attempt\":{\"state\":\"INCONCLUSIVE\","
    "\"reason\":\"python3_missing\"},\"partial\":false,"
    "\"budget_exhausted\":false,\"collector_budget_seconds\":%s}\\n' "
    '"$1" "$8" "$9" "${17}"'
)

_SNAPSHOT_SOURCE = (
    Path(__file__).resolve().with_name("_observation_snapshot.py").read_text(encoding="utf-8")
    .replace("__CONTAINMENT_CODE__", repr(CONTAINMENT_CODE))
    .replace("__MEM_CODE__", repr(MEM_CODE))
    .replace("__TORCH_CODE__", repr(TORCH_CODE))
)


def build_snapshot_argv(
    *,
    workspace: str,
    comfy: str,
    commit: str,
    root: str,
    attempt: str,
    run_id: str,
    nonce: str,
    guard_sha: str,
    targets: Sequence[str],
    validator_source: str = stock_nodes.VALIDATOR_SOURCE,
    mode: str = "environment",
    marker: str | None = None,
    guard_source_b64: str = "",
    expected_job_sha256: str = "",
    expected_launch_sha256: str = "",
    expected_deadline_epoch: float | None = None,
    collector_budget: float = 6.0,
) -> tuple[str, ...]:
    if mode not in {"environment", "attempt"}:
        raise ValueError("snapshot mode is invalid")
    if not isinstance(workspace, str) or not workspace.startswith("/"):
        raise ValueError("workspace must be absolute")
    if attempt and not isinstance(attempt, str):
        raise ValueError("attempt must be a string")
    if run_id and not HEX32_RE.fullmatch(run_id):
        raise ValueError("run_id must be 32 hex characters")
    if nonce and not HEX32_RE.fullmatch(nonce):
        raise ValueError("nonce must be 32 hex characters")
    if not isinstance(guard_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", guard_sha):
        raise ValueError("guard hash must be sha256")
    if mode == "environment" and (not isinstance(validator_source, str) or not validator_source):
        raise ValueError("validator source is missing")
    if mode == "attempt":
        if not isinstance(guard_source_b64, str) or not guard_source_b64:
            raise ValueError("guard source is missing")
        for value in (expected_job_sha256, expected_launch_sha256):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("expected job/launch hash is invalid")
    if not (0.1 <= float(collector_budget) <= 6.0):
        raise ValueError("collector budget is invalid")
    if expected_deadline_epoch is not None and not _is_number(expected_deadline_epoch):
        raise ValueError("expected deadline is invalid")
    validator_literal = ""
    if validator_source:
        validator_literal = (
            validator_source
            if validator_source.startswith("exec(")
            else "exec(" + repr(validator_source) + ")"
        )
    marker_value = marker or uuid.uuid4().hex
    if not SNAPSHOT_MARKER_RE.fullmatch(marker_value):
        raise ValueError("snapshot marker is invalid")
    source = "exec(" + repr(_SNAPSHOT_SOURCE) + ")"
    targets_json = json.dumps([str(item) for item in targets], separators=(",", ":"))
    return (
        "sh",
        "-c",
        _SHELL_WRAPPER,
        "i1obs",
        marker_value,
        source,
        workspace,
        comfy,
        commit,
        root,
        attempt,
        run_id,
        nonce,
        guard_sha,
        targets_json,
        validator_literal,
        mode,
        guard_source_b64,
        expected_job_sha256,
        expected_launch_sha256,
        str(float(collector_budget)),
        "" if expected_deadline_epoch is None else str(float(expected_deadline_epoch)),
    )


def _reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _loads_strict(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicates)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_command(item: Any) -> str:
    if not isinstance(item, Mapping):
        raise ValueError("command record is not an object")
    for key in ("key", "exit", "out", "err", "timed_out"):
        if key not in item:
            raise ValueError("command record is incomplete")
    if not isinstance(item["key"], str) or not item["key"]:
        raise ValueError("command key is invalid")
    parsed = _loads_strict(item["key"])
    if not isinstance(parsed, list) or not all(isinstance(value, str) for value in parsed):
        raise ValueError("command key is not an argv list")
    if not _is_int(item["exit"]) or not (0 <= item["exit"] <= 255):
        raise ValueError("command exit is invalid")
    if not isinstance(item["out"], str) or not isinstance(item["err"], str):
        raise ValueError("command output is invalid")
    if not isinstance(item["timed_out"], bool):
        raise ValueError("command timed_out is invalid")
    if "unrun" in item and not isinstance(item["unrun"], bool):
        raise ValueError("command unrun is invalid")
    return item["key"]


def _validate_proof(value: Any, *, run_id: str, nonce: str, command_sha: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("process proof is not an object")
    if value.get("schema") != "anime-video-studio/i1-process-proof/v1":
        raise ValueError("process proof schema mismatch")
    if value.get("run_id") != run_id or value.get("nonce") != nonce:
        raise ValueError("process proof run/nonce mismatch")
    if value.get("command_sha256") != command_sha:
        raise ValueError("process proof command mismatch")
    if not _is_int(value.get("pid")) or value.get("pid", 0) <= 0:
        raise ValueError("process proof pid is invalid")
    if not _is_int(value.get("start_ticks")):
        raise ValueError("process proof start_ticks is invalid")
    if not isinstance(value.get("exe"), str) or not value.get("exe"):
        raise ValueError("process proof exe is invalid")
    if not _is_digest(value.get("cmdline_sha256")):
        raise ValueError("process proof cmdline hash is invalid")
    if not isinstance(value.get("boot_id"), str) or not value.get("boot_id"):
        raise ValueError("process proof boot_id is invalid")
    if not _is_digest(value.get("birth_sha256")):
        raise ValueError("process proof birth hash is invalid")


def _validate_attempt_decision(
    decision: Any,
    *,
    run_id: str,
    nonce: str,
    attempt_path: str,
    expected_job_sha256: str,
    expected_launch_sha256: str,
) -> None:
    if not isinstance(decision, Mapping):
        raise ValueError("attempt decision is not an object")
    state = decision.get("state")
    if state not in {"OK", "MISSING", "UNREADABLE", "INCONCLUSIVE"}:
        raise ValueError("attempt decision state is invalid")
    if state != "OK":
        if decision.get("reason") is not None and not isinstance(decision.get("reason"), str):
            raise ValueError("attempt decision reason is invalid")
        return
    process = decision.get("process")
    if process not in {"LIVE", "DEAD", "ABSENT", "UNKNOWN", "MISMATCH"}:
        raise ValueError("attempt process is invalid")
    if decision.get("run_id") != run_id or decision.get("nonce") != nonce:
        raise ValueError("attempt decision run/nonce mismatch")
    if decision.get("attempt") != attempt_path:
        raise ValueError("attempt decision path mismatch")
    if decision.get("job_sha256") != expected_job_sha256:
        raise ValueError("attempt decision job hash mismatch")
    if decision.get("launch_sha256") != expected_launch_sha256:
        raise ValueError("attempt decision launch hash mismatch")
    command_sha = decision.get("manifest_sha256")
    if not _is_digest(command_sha):
        raise ValueError("attempt manifest hash is invalid")
    status = decision.get("status")
    if not isinstance(status, Mapping):
        raise ValueError("attempt status is not an object")
    if status.get("schema") != "anime-video-studio/i1-download-status/v1":
        raise ValueError("attempt status schema mismatch")
    if status.get("run_id") != run_id or status.get("nonce") != nonce:
        raise ValueError("attempt status run/nonce mismatch")
    if status.get("status") not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise ValueError("attempt status value is invalid")
    owner = decision.get("owner")
    ack = decision.get("ack")
    if process == "ABSENT" and status.get("status") == "COMPLETE" and owner is None and ack is None:
        pass
    else:
        for name, record in (("owner", owner), ("ack", ack)):
            if not isinstance(record, Mapping):
                raise ValueError(f"attempt {name} is not an object")
            if record.get("schema") != "anime-video-studio/p1-v3-process-owner/v1":
                raise ValueError(f"attempt {name} schema mismatch")
            if record.get("kind") != "download_worker":
                raise ValueError(f"attempt {name} kind mismatch")
            if record.get("run_id") != run_id or record.get("nonce") != nonce:
                raise ValueError(f"attempt {name} run/nonce mismatch")
            if record.get("command_sha256") != command_sha:
                raise ValueError(f"attempt {name} command mismatch")
            if not _is_int(record.get("pid")) or record.get("pid", 0) <= 0:
                raise ValueError(f"attempt {name} pid is invalid")
        if owner.get("pid") != ack.get("pid"):
            raise ValueError("attempt owner/ack pid mismatch")
    if process in {"LIVE", "DEAD"}:
        _validate_proof(
            decision.get("proof"), run_id=run_id, nonce=nonce, command_sha=command_sha
        )
    progress = decision.get("progress")
    if progress:
        if not isinstance(progress, Mapping):
            raise ValueError("attempt progress is not an object")
        if progress.get("schema") != "anime-video-studio/i1-download-progress/v1":
            raise ValueError("attempt progress schema mismatch")
        if progress.get("run_id") != run_id or progress.get("nonce") != nonce:
            raise ValueError("attempt progress run/nonce mismatch")
        for key in ("bytes_done", "bytes_total", "files_done", "files_total"):
            if key in progress and not _is_int(progress[key]):
                raise ValueError("attempt progress counter is invalid")
    heartbeat = decision.get("heartbeat")
    if heartbeat:
        if not isinstance(heartbeat, Mapping):
            raise ValueError("attempt heartbeat is not an object")
        if heartbeat.get("schema") not in (
            None,
            "anime-video-studio/i1-download-heartbeat/v1",
        ):
            raise ValueError("attempt heartbeat schema mismatch")
        if heartbeat.get("run_id") != run_id or heartbeat.get("nonce") != nonce:
            raise ValueError("attempt heartbeat run/nonce mismatch")
        if "heartbeat_at" in heartbeat and not _is_number(heartbeat["heartbeat_at"]):
            raise ValueError("attempt heartbeat timestamp is invalid")


def _validate_snapshot(
    snapshot: Any,
    *,
    marker: str,
    run_id: str,
    nonce: str,
    mode: str,
    attempt_path: str,
    expected_job_sha256: str,
    expected_launch_sha256: str,
) -> None:
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot is not an object")
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("snapshot schema mismatch")
    if snapshot.get("marker") != marker:
        raise ValueError("snapshot marker mismatch")
    if snapshot.get("run_id") != run_id or snapshot.get("nonce") != nonce:
        raise ValueError("snapshot run/nonce mismatch")
    if not isinstance(snapshot.get("python3"), bool):
        raise ValueError("snapshot python3 flag is invalid")
    presence = snapshot.get("tool_presence")
    if presence is not None:
        if not isinstance(presence, Mapping):
            raise ValueError("snapshot tool_presence is invalid")
        for key, value in presence.items():
            if not isinstance(key, str) or not (value is None or isinstance(value, bool)):
                raise ValueError("snapshot tool_presence entry is invalid")
    commands = snapshot.get("commands")
    if not isinstance(commands, list):
        raise ValueError("snapshot commands is not a list")
    seen = set()
    for item in commands:
        key = _validate_command(item)
        if key in seen:
            raise ValueError("duplicate command key")
        seen.add(key)
    if not isinstance(snapshot.get("partial"), bool) or not isinstance(
        snapshot.get("budget_exhausted"), bool
    ):
        raise ValueError("snapshot budget flags are invalid")
    if not _is_number(snapshot.get("collector_budget_seconds")):
        raise ValueError("snapshot collector budget is invalid")
    attempt = snapshot.get("attempt")
    if not isinstance(attempt, Mapping):
        raise ValueError("snapshot attempt is not an object")
    state = attempt.get("state")
    if state not in {"MISSING", "UNREADABLE", "INCONCLUSIVE", "OK"}:
        raise ValueError("snapshot attempt state is invalid")
    if mode == "attempt" and state == "OK":
        _validate_attempt_decision(
            attempt.get("decision"),
            run_id=run_id,
            nonce=nonce,
            attempt_path=attempt_path,
            expected_job_sha256=expected_job_sha256,
            expected_launch_sha256=expected_launch_sha256,
        )


def _coerce_result(value: Any) -> ObservationResult:
    return ObservationResult(
        kind="INCONCLUSIVE",
        returncode=int(getattr(value, "returncode", 1)),
        stdout=str(getattr(value, "stdout", "") or ""),
        stderr=str(getattr(value, "stderr", "") or ""),
        timed_out=bool(getattr(value, "timed_out", False)),
    )


def _exception_result(exc: BaseException) -> ObservationResult:
    if isinstance(exc, TimeoutError):
        return ObservationResult(kind="TIMEOUT", returncode=124, timed_out=True)
    if isinstance(exc, OSError):
        return ObservationResult(
            kind="TRANSPORT_ERROR", returncode=255, stderr=type(exc).__name__
        )
    raise exc


def classify_result(result: ObservationResult) -> str:
    if result.timed_out:
        return "TIMEOUT"
    text = f"{result.stderr}\n{result.stdout}".lower()
    returncode = int(result.returncode or 0)
    if returncode == 0:
        try:
            payload = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError):
            return "PROTOCOL_ERROR"
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != SNAPSHOT_SCHEMA
            or not isinstance(payload.get("marker"), str)
            or not payload.get("marker")
        ):
            return "PROTOCOL_ERROR"
        return "OK"
    if any(pattern in text for pattern in HOST_KEY_PATTERNS):
        return "HOST_KEY_CHANGED"
    if any(pattern in text for pattern in AUTH_PATTERNS):
        return "AUTH_FAILED"
    if any(pattern in text for pattern in TRANSPORT_PATTERNS):
        return "CONNECTION_LOST" if returncode == 255 else "TRANSPORT_ERROR"
    return "REMOTE_COMMAND_ERROR"


def run_readonly(
    runner: Any,
    identity: Any,
    argv: Sequence[str],
    *,
    max_attempts: int = 3,
    per_attempt_seconds: float = 8.0,
    total_seconds: float = 25.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ObservationResult:
    """Run one read-only command with bounded, monotonic retry.

    Only timeout/connection-lost transport failures are retried.  Auth, host
    key changes, protocol errors and ordinary remote-command errors are never
    retried.  The caller must never pass a mutating command here.
    """

    if max_attempts < 1 or max_attempts > 5:
        raise ValueError("max_attempts must be 1..5")
    if not (0 < per_attempt_seconds <= 60):
        raise ValueError("per_attempt_seconds must be bounded")
    if not (0 < total_seconds <= 300):
        raise ValueError("total_seconds must be bounded")
    start = clock()
    deadline = start + float(total_seconds)
    last: ObservationResult | None = None
    for index in range(max_attempts):
        remaining = deadline - clock()
        if remaining <= 0:
            break
        timeout = min(float(per_attempt_seconds), remaining)
        try:
            if isinstance(runner, RemoteRunner):
                # Keep the fixed snapshot source out of Windows CreateProcess argv.
                # stdin is replayable only here: this helper is read-only.
                payload = ("exec " + shlex.join(tuple(argv)) + "\n").encode("utf-8")
                raw = runner.run_remote_with_input(
                    identity, ("sh", "-s"), payload, timeout=timeout
                )
            else:
                raw = runner.run_remote(identity, tuple(argv), timeout=timeout)
            result = _coerce_result(raw)
        except RemoteHostKeyError as exc:
            result = ObservationResult(
                kind="HOST_KEY_CHANGED", returncode=255, stderr=str(exc)
            )
        except RemoteError as exc:
            result = ObservationResult(
                kind="PROTOCOL_ERROR", returncode=255, stderr=str(exc)
            )
        except (TimeoutError, OSError) as exc:
            result = _exception_result(exc)
        result.attempts = index + 1
        result.elapsed_seconds = max(0.0, clock() - start)
        result.kind = classify_result(result)
        result.diagnostic = _diagnostic(result)
        if result.kind == "OK":
            return result
        if result.kind not in RETRYABLE:
            return result
        last = result
        if index + 1 < max_attempts:
            backoff = 0.5 if index == 0 else 1.0
            pause = min(backoff, max(0.0, deadline - clock()))
            if pause > 0:
                sleep(pause)
    if last is None:
        last = ObservationResult(kind="TIMEOUT", returncode=124, timed_out=True)
    last.elapsed_seconds = max(last.elapsed_seconds, clock() - start)
    last.diagnostic = _diagnostic(last)
    return last


def _diagnostic(result: ObservationResult) -> str:
    text = (result.stderr or result.stdout or "").replace("\x00", "").strip()
    if not text:
        return result.kind
    return text[:400]


def capture_snapshot(
    runner: Any,
    identity: Any,
    *,
    workspace: str,
    profile: Any,
    root: str,
    attempt: str = "",
    run_id: str = "",
    nonce: str = "",
    guard_sha: str | None = None,
    mode: str = "environment",
    expected_job_sha256: str = "",
    expected_launch_sha256: str = "",
    expected_deadline_epoch: float | None = None,
    collector_budget: float = 6.0,
    **retry_kwargs: Any,
) -> ObservationResult:
    comfy = f"{workspace.rstrip('/')}/ComfyUI" if workspace.startswith("/") else ""
    marker = uuid.uuid4().hex
    argv = build_snapshot_argv(
        workspace=workspace,
        comfy=comfy,
        commit=str(profile.comfy["commit"]),
        root=root,
        attempt=attempt,
        run_id=run_id,
        nonce=nonce,
        guard_sha=guard_sha or guard_sha256(),
        targets=[str(item["target"]) for item in profile.models],
        validator_source=stock_nodes.VALIDATOR_SOURCE if mode == "environment" else "",
        mode=mode,
        marker=marker,
        guard_source_b64=guard_source_b64() if mode == "attempt" else "",
        expected_job_sha256=expected_job_sha256 if mode == "attempt" else "",
        expected_launch_sha256=expected_launch_sha256 if mode == "attempt" else "",
        expected_deadline_epoch=expected_deadline_epoch if mode == "attempt" else None,
        collector_budget=collector_budget,
    )
    result = run_readonly(runner, identity, argv, **retry_kwargs)
    if result.kind != "OK":
        return result
    try:
        snapshot = _loads_strict(result.stdout)
        _validate_snapshot(
            snapshot,
            marker=marker,
            run_id=run_id,
            nonce=nonce,
            mode=mode,
            attempt_path=attempt,
            expected_job_sha256=expected_job_sha256,
            expected_launch_sha256=expected_launch_sha256,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        result.kind = "PROTOCOL_ERROR"
        result.diagnostic = f"snapshot protocol error: {type(exc).__name__}"
        return result
    result.snapshot = snapshot
    return result


def decision_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return the guard-compatible attempt decision, never inventing death."""

    attempt = snapshot.get("attempt") if isinstance(snapshot, Mapping) else None
    if not isinstance(attempt, Mapping):
        return {"state": "INCONCLUSIVE", "reason": "attempt snapshot missing"}
    state = attempt.get("state")
    if state == "MISSING":
        return {"state": "MISSING", "process": "ABSENT"}
    if state in {"UNREADABLE", "INCONCLUSIVE"}:
        return {
            "state": state,
            "reason": str(attempt.get("reason") or "attempt snapshot inconclusive"),
        }
    if state != "OK":
        return {"state": "INCONCLUSIVE", "reason": "attempt snapshot state is invalid"}
    decision = attempt.get("decision")
    if not isinstance(decision, Mapping):
        return {"state": "INCONCLUSIVE", "reason": "attempt decision missing"}
    return dict(decision)


class SnapshotRunner:
    """Replay a captured snapshot for the existing inspect_remote logic."""

    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        self._commands: dict[str, Mapping[str, Any]] = {}
        commands = snapshot.get("commands") if isinstance(snapshot, Mapping) else None
        if not isinstance(commands, list):
            raise ValueError("snapshot commands is not a list")
        for item in commands:
            key = _validate_command(item)
            if key in self._commands:
                raise ValueError("duplicate command key")
            self._commands[key] = item

    def run_remote(self, identity: Any, argv: Sequence[str], timeout: float | None = None):
        del identity, timeout
        item = self._commands.get(canonical_argv(argv))
        if item is None:
            return SimpleNamespace(
                returncode=127,
                stdout="",
                stderr="snapshot_command_missing",
                timed_out=False,
                probe_kind="missing",
            )
        timed_out = item["timed_out"] or item["exit"] == 124
        probe_kind = "not_run" if item.get("unrun") else ("timeout" if timed_out else "ok")
        return SimpleNamespace(
            returncode=item["exit"],
            stdout=item["out"],
            stderr=item["err"],
            timed_out=timed_out,
            probe_kind=probe_kind,
        )


__all__ = [
    "AUTH_PATTERNS",
    "CONTAINMENT_CODE",
    "HOME_CODE",
    "HOST_KEY_PATTERNS",
    "MEM_CODE",
    "NON_RETRYABLE",
    "ObservationResult",
    "RETRYABLE",
    "SNAPSHOT_SCHEMA",
    "SnapshotRunner",
    "TORCH_CODE",
    "TRANSPORT_PATTERNS",
    "build_snapshot_argv",
    "canonical_argv",
    "capture_snapshot",
    "classify_result",
    "decision_from_snapshot",
    "guard_sha256",
    "guard_source_b64",
    "run_readonly",
]
