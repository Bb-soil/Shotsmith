"""I1 command orchestration.

The public CLI is intentionally thin; all state transitions and remote stages
live here so the same functions can be exercised with fake runners in tests.
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from scripts.p1.remote import RemoteRunner
from scripts.p1.v3_bootstrap import prepare_remote_root

from . import comfy, download, observation, readiness, remote as i1_remote, state as state_mod, tunnel, workflow
from .errors import BlockedError, HostTrustError, Interrupted, NotReady, RemoteFailure, StateError, UsageError
from .hosttrust import (
    ensure_known_hosts,
    known_host_fingerprints,
    scan_host,
    trust_scanned_key,
    verify_trusted,
)
from .inspect import inspect_remote
from .profile import Profile, load_profile, verify_profile_sources
from .setup import setup_text_for
from .util import format_bytes, format_duration, redact, validate_name, validate_workspace

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = ROOT / "data" / "i1"
DEFAULT_PROFILE = ROOT / "config" / "i1" / "wan22-14b-ab.profile.json"
STALE_SECONDS = 30.0
INSTALL_DEADLINE_SECONDS = 12 * 3600


@dataclass
class Context:
    state_dir: Path
    repo_root: Path
    profile_path: Path
    profile: Profile
    name: str
    record: dict[str, Any]
    runner: RemoteRunner
    identity: Any
    state: dict[str, Any]
    run_dir: Path
    progress: Any | None = None


@dataclass
class ConnectArgs:
    name: str
    host: str
    user: str
    port: int
    workspace: str
    identity: str | None = None
    agent: bool = False
    accept_fingerprint: str | None = None
    replace: bool = False
    json: bool = False


def make_runner() -> RemoteRunner:
    return RemoteRunner()


def load_profile_or_fail(profile_path: Path | str) -> Profile:
    try:
        profile = load_profile(profile_path)
    except Exception as exc:
        raise BlockedError(f"не удалось прочитать профиль: {type(exc).__name__}") from exc
    verify_profile_sources(profile, ROOT)
    return profile


def _auth_dict(args: ConnectArgs) -> dict[str, Any]:
    if args.agent:
        return {"kind": "agent"}
    if not args.identity:
        raise UsageError("нужен --identity PATH или --agent")
    path = Path(args.identity).expanduser()
    if not path.is_file():
        raise UsageError("identity file path does not exist")
    return {"kind": "key", "path": str(path.resolve())}


def _record_payload(
    *,
    args: ConnectArgs,
    profile: Profile,
    known_hosts: Path,
    global_known_hosts: Path,
    fingerprint: str,
    host_key_type: str,
) -> dict[str, Any]:
    return {
        "schema": state_mod.RECORD_SCHEMA,
        "name": args.name,
        "identity": {
            "host": args.host,
            "user": args.user,
            "port": int(args.port),
            "fingerprint": fingerprint,
            "workspace": args.workspace,
            "profile_id": profile.profile_id,
            "profile_sha256": profile.sha256,
            "target_id": state_mod.target_id(
                host=args.host,
                user=args.user,
                port=args.port,
                fingerprint=fingerprint,
            ),
        },
        "host_key_type": host_key_type,
        "auth": _auth_dict(args),
        "known_hosts": str(known_hosts),
        "global_known_hosts": str(global_known_hosts),
        "created_at": state_mod.now_iso(),
        "updated_at": state_mod.now_iso(),
    }


def connect(
    *,
    args: ConnectArgs,
    state_dir: Path | str = DEFAULT_STATE_DIR,
    profile_path: Path | str = DEFAULT_PROFILE,
    runner: RemoteRunner | None = None,
    trust_runner: Any | None = None,
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile_or_fail(profile_path)
    name = validate_name(args.name, "profile name")
    workspace = validate_workspace(args.workspace)
    existing: dict[str, Any] | None = None
    if state_mod.record_path(state_dir, name).is_file():
        existing = state_mod.load_record(state_dir, name)
    known_hosts = ensure_known_hosts(
        Path(existing["known_hosts"])
        if existing is not None and isinstance(existing.get("known_hosts"), str)
        else state_mod.known_hosts_path(state_dir, name)
    )
    global_known_hosts = ensure_known_hosts(
        Path(existing["global_known_hosts"])
        if existing is not None and isinstance(existing.get("global_known_hosts"), str)
        else state_mod.global_known_hosts_path(state_dir, name)
    )
    if existing is not None:
        identity = existing["identity"]
        if (
            identity.get("host") != args.host
            or int(identity.get("port", 0)) != int(args.port)
            or identity.get("user") != args.user
            or identity.get("workspace") != workspace
        ):
            if not args.replace:
                raise UsageError("профиль уже существует с другим target/workspace; выберите новое имя или --replace")
        if identity.get("profile_sha256") != profile.sha256:
            raise BlockedError("профиль установки изменился; создайте новое соединение, старое state не переиспользуется")
        expected = str(identity.get("fingerprint"))
        verify_trusted(
            known_hosts=known_hosts,
            host=args.host,
            port=int(args.port),
            expected_fingerprint=expected,
            runner=trust_runner,
        )
        live = scan_host(
            host=args.host,
            port=int(args.port),
            runner=trust_runner,
        )
        matched = [item for item in live if item.fingerprint == expected]
        if len(matched) != 1:
            raise HostTrustError("host key changed; соединение заблокировано, known_hosts не изменён")
        fingerprint = expected
        host_key_type = str(
            existing.get("host_key_type") or (matched[0].key_type if matched else "")
        )
        if not host_key_type:
            raise HostTrustError("trusted host key type is missing")
    else:
        scanned = scan_host(
            host=args.host,
            port=int(args.port),
            runner=trust_runner,
        )
        fingerprints = [item.fingerprint for item in scanned]
        accepted = args.accept_fingerprint
        if accepted is not None:
            if accepted not in fingerprints:
                raise HostTrustError("указанный fingerprint не совпал с host key")
        elif sys.stdin.isatty() and not args.json:
            print("Отпечатки host key:")
            for value in fingerprints:
                print(f"  {value}")
            answer = input("Доверять этому host key? [y/N] ").strip().lower()
            if answer not in {"y", "yes", "д", "да"}:
                raise HostTrustError("host key не принят")
            accepted = fingerprints[0]
        else:
            raise HostTrustError(
                "неизвестный host key; в неинтерактивном режиме нужен --accept-fingerprint SHA256:..."
            )
        try:
            prior_keys = known_host_fingerprints(
                known_hosts=known_hosts,
                host=args.host,
                port=int(args.port),
                runner=trust_runner,
            )
        except HostTrustError:
            prior_keys = []
        if prior_keys and accepted not in {item.fingerprint for item in prior_keys}:
            raise HostTrustError(
                "host key changed for this host; known_hosts не изменён, нужен owner-driven repair"
            )
        trusted_key = trust_scanned_key(
            known_hosts=known_hosts,
            host=args.host,
            port=int(args.port),
            expected_fingerprint=accepted,
            runner=trust_runner,
        )
        fingerprint = accepted
        host_key_type = trusted_key.key_type
    record = _record_payload(
        args=args,
        profile=profile,
        known_hosts=known_hosts,
        global_known_hosts=global_known_hosts,
        fingerprint=fingerprint,
        host_key_type=host_key_type,
    )
    record["identity"]["workspace"] = workspace
    identity_obj = i1_remote.make_identity(record)
    real_runner = runner or make_runner()
    result = real_runner.run_remote(identity_obj, ("true",), timeout=20.0)
    if getattr(result, "timed_out", False) or int(getattr(result, "returncode", 1)) != 0:
        raise HostTrustError("SSH/agent не подтвердил доступ; профиль не сохранён")
    state_mod.save_record(state_dir, name, record)
    saved = state_mod.load_record(state_dir, name)
    state = state_mod.state_for_record(state_dir, saved)
    state_mod.save_state_for_record(state_dir, saved, state)
    return {
        "status": "CONNECTED",
        "name": name,
        "host": args.host,
        "user": args.user,
        "port": int(args.port),
        "workspace": workspace,
        "fingerprint": fingerprint,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "identity_key": state["identity_key"],
    }


def load_context(
    *,
    name: str | None,
    state_dir: Path | str = DEFAULT_STATE_DIR,
    profile_path: Path | str = DEFAULT_PROFILE,
    runner: RemoteRunner | None = None,
    verify_key: bool = True,
) -> Context:
    state_dir = Path(state_dir)
    profile = load_profile_or_fail(profile_path)
    resolved, record = state_mod.default_record(state_dir, name)
    identity_data = record["identity"]
    if identity_data.get("profile_sha256") != profile.sha256:
        raise BlockedError("connection profile pinned another installation profile; no state reuse")
    if not isinstance(record.get("host_key_type"), str) or not record["host_key_type"]:
        raise BlockedError("connection profile lacks an accepted host key type; reconnect")
    if verify_key:
        verify_trusted(
            known_hosts=record["known_hosts"],
            host=identity_data["host"],
            port=int(identity_data["port"]),
            expected_fingerprint=identity_data["fingerprint"],
        )
    identity = i1_remote.make_identity(record)
    state = state_mod.state_for_record(state_dir, record)
    run_dir = state_mod.run_dir(state_dir, state["identity_key"])
    return Context(
        state_dir=state_dir,
        repo_root=ROOT,
        profile_path=Path(profile_path),
        profile=profile,
        name=resolved,
        record=record,
        runner=runner or make_runner(),
        identity=identity,
        state=state,
        run_dir=run_dir,
    )


def _save(ctx: Context, state: Mapping[str, Any]) -> None:
    ctx.state = state_mod.validate_state(state)
    state_mod.save_state_for_record(ctx.state_dir, ctx.record, ctx.state)
    if ctx.progress is not None:
        ctx.progress(ctx.state)


def _attempt_id(state: Mapping[str, Any]) -> str | None:
    attempt = (state.get("artifacts") or {}).get("download_attempt") or {}
    value = attempt.get("id") if isinstance(attempt, Mapping) else None
    return str(value) if value else None


def _observe_attempt(
    ctx: Context,
    *,
    attempt: str | None = None,
    run_id: str | None = None,
    nonce: str | None = None,
) -> observation.ObservationResult:
    workspace = ctx.state["identity"]["workspace"]
    root = i1_remote.owned_root(workspace)
    attempt_id = attempt or _attempt_id(ctx.state) or ""
    attempt_dir = download.remote_paths(root, attempt_id)["attempt"] if attempt_id else ""
    if not isinstance(ctx.runner, RemoteRunner):
        # Compatibility path for injected test runners.  Production always uses
        # the batched read-only snapshot; this path never mutates and never
        # invents death from a transport error.
        if not attempt_id:
            return observation.ObservationResult(
                kind="OK",
                snapshot={"attempt": {"state": "INCONCLUSIVE", "reason": "not_requested"}},
                decision={"state": "INCONCLUSIVE"},
            )
        try:
            decision = download.inspect_attempt(
                ctx.runner,
                ctx.identity,
                root=root,
                run_id=run_id or ctx.state["run_id"],
                nonce=nonce or ctx.state["nonce"],
                attempt=attempt_id,
                timeout=30.0,
            )
        except Exception as exc:
            return observation.ObservationResult(
                kind="REMOTE_COMMAND_ERROR", diagnostic=str(exc)
            )
        return observation.ObservationResult(
            kind="OK",
            snapshot={
                "attempt": {
                    "state": (
                        decision.get("state")
                        if decision.get("state") in {"MISSING", "UNREADABLE", "INCONCLUSIVE"}
                        else "OK"
                    ),
                    "reason": decision.get("reason"),
                    "decision": decision,
                }
            },
            decision=decision,
        )
    attempt_info = (ctx.state.get("artifacts") or {}).get("download_attempt") or {}
    job_sha256 = attempt_info.get("job_sha256") if isinstance(attempt_info, Mapping) else None
    launch_sha256 = (
        attempt_info.get("launch_sha256") if isinstance(attempt_info, Mapping) else None
    )
    if attempt_id:
        valid_pins = all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value.lower())
            for value in (job_sha256, launch_sha256)
        )
        if not valid_pins:
            return observation.ObservationResult(
                kind="INCONCLUSIVE",
                diagnostic="attempt job/launch pins are missing or invalid",
            )
        expected_dir = attempt_info.get("dir")
        if expected_dir and os.path.normpath(str(expected_dir)) != os.path.normpath(
            attempt_dir
        ):
            return observation.ObservationResult(
                kind="INCONCLUSIVE",
                diagnostic="attempt directory does not match local state",
            )
    result = observation.capture_snapshot(
        ctx.runner,
        ctx.identity,
        workspace=workspace,
        profile=ctx.profile,
        root=root,
        attempt=attempt_dir,
        run_id=run_id or ctx.state["run_id"],
        nonce=nonce or ctx.state["nonce"],
        guard_sha=observation.guard_sha256(),
        mode="attempt" if attempt_id else "environment",
        expected_job_sha256=job_sha256,
        expected_launch_sha256=launch_sha256,
        expected_deadline_epoch=(
            attempt_info.get("deadline_epoch") if isinstance(attempt_info, Mapping) else None
        ),
    )
    if result.snapshot is not None:
        result.decision = observation.decision_from_snapshot(result.snapshot)
    return result


def _heartbeat_facts(decision: Mapping[str, Any]) -> tuple[float | None, bool | None]:
    heartbeat = decision.get("heartbeat")
    if not isinstance(heartbeat, Mapping):
        return None, None
    at = heartbeat.get("heartbeat_at")
    if isinstance(at, bool) or not isinstance(at, (int, float)):
        return None, None
    age = time.time() - float(at)
    if age < 0:
        # A client clock skew makes the age unknown, not fresh.
        return None, None
    return age, age <= STALE_SECONDS


def _observation_payload(
    result: observation.ObservationResult,
    *,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    previous = dict(previous or {})
    now = state_mod.now_iso()
    current = result.kind == "OK"
    decision = dict(result.decision or {})
    worker_state = "UNKNOWN"
    proof_verified = False
    heartbeat_age = None
    heartbeat_fresh = None
    progress_fresh = False
    progress: dict[str, Any] | None = None
    if current:
        state_name = decision.get("state")
        process = decision.get("process")
        if state_name == "MISSING":
            worker_state = "ABSENT"
        elif state_name == "OK" and process in {"LIVE", "DEAD", "ABSENT", "MISMATCH", "UNKNOWN"}:
            worker_state = str(process)
            proof_verified = process in {"LIVE", "DEAD", "ABSENT"}
        elif state_name in {"INCONCLUSIVE", "UNREADABLE"}:
            worker_state = "UNKNOWN"
        if state_name == "OK":
            heartbeat_age, heartbeat_fresh = _heartbeat_facts(decision)
            candidate = decision.get("progress")
            if isinstance(candidate, Mapping):
                progress = dict(candidate)
                progress_fresh = process in {"LIVE", "DEAD", "ABSENT"}
    bytes_now = None
    if progress is not None:
        candidate = progress.get("bytes_done")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            bytes_now = candidate
    previous_bytes = previous.get("previous_bytes")
    if isinstance(previous_bytes, bool) or not isinstance(previous_bytes, int):
        previous_bytes = None
    was_current = previous.get("connection_state") == "CURRENT"
    rate_fresh = bool(
        current
        and progress_fresh
        and heartbeat_fresh is True
        and was_current
        and previous_bytes is not None
        and bytes_now is not None
        and bytes_now >= previous_bytes
    )
    worker_responsiveness = (
        "FRESH" if heartbeat_fresh is True else "STALE" if heartbeat_fresh is False else "UNKNOWN"
    )
    cached_progress = previous.get("cached_progress")
    cached_as_of = previous.get("cached_as_of") or previous.get("last_confirmed_at")
    if current and progress is not None:
        cached_progress = dict(progress)
        cached_as_of = now
    bytes_observed_at = (
        now if current and progress is not None else previous.get("bytes_observed_at")
    )
    diagnostic = {
        "code": result.kind,
        "message": redact(result.diagnostic, limit=400),
    }
    return {
        "connection_state": "CURRENT" if current else "CONNECTION_LOST",
        "worker_state": worker_state,
        "worker_responsiveness": worker_responsiveness,
        "proof_verified": proof_verified,
        "heartbeat_age_seconds": (
            round(float(heartbeat_age), 3) if heartbeat_age is not None else None
        ),
        "heartbeat_fresh": heartbeat_fresh,
        "hb_responsive": heartbeat_fresh,
        "last_confirmed_at": now if current else previous.get("last_confirmed_at"),
        "last_attempt_at": now,
        "source": "remote" if current else "cache",
        "diagnostic": diagnostic,
        "retry_count": result.attempts,
        "wall_time_seconds": round(float(result.elapsed_seconds), 3),
        "progress_fresh": progress_fresh,
        "rate_fresh": rate_fresh,
        "bytes_observed_at": bytes_observed_at,
        "previous_bytes": (
            bytes_now if bytes_now is not None else previous.get("previous_bytes")
        ),
        "previous_bytes_at": (
            bytes_observed_at if bytes_now is not None else previous.get("previous_bytes_at")
        ),
        "cached_as_of": cached_as_of,
        "cached_progress": (
            dict(cached_progress) if isinstance(cached_progress, Mapping) else None
        ),
    }


def _lose_readiness(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    report = dict(result.get("readiness") or {})
    if report.get("ready") or report.get("status") == "READY":
        report["status"] = "UNCONFIRMED"
        report["ready"] = False
        warnings = list(report.get("warnings") or [])
        warnings.append("remote state is not confirmed; cached readiness is historical")
        report["warnings"] = warnings
        result["readiness"] = report
    result["readiness_current"] = False
    return result


def _record_observation(
    ctx: Context,
    result: observation.ObservationResult,
    *,
    stage: str | None = None,
) -> dict[str, Any]:
    previous = ctx.state.get("observation") or {}
    payload = _observation_payload(result, previous=previous)
    state = {
        **ctx.state,
        "observation": {**dict(previous), **payload},
    }
    if result.kind == "OK":
        decision = dict(result.decision or {})
        if decision.get("state") == "OK":
            progress = decision.get("progress")
            if isinstance(progress, Mapping):
                state["progress"] = dict(progress)
            state["worker"] = {
                "owner": decision.get("owner"),
                "ack": decision.get("ack"),
                "status": decision.get("status") or {},
                "progress": decision.get("progress") or {},
                "heartbeat": decision.get("heartbeat") or {},
            }
    else:
        state = _lose_readiness(state)
    if stage is not None:
        state = {**state, "stage": stage}
    _save(ctx, state)
    return payload


def _plan(ctx: Context, inspection: Mapping[str, Any]) -> dict[str, Any]:
    profile = ctx.profile
    workflows = [
        {
            "workflow_id": workflow_id,
            "path": item["api_path"],
            "sha256": item["api_sha256"],
            "output": dict(item.get("output") or {}),
        }
        for workflow_id, item in profile.workflow_items()
    ]
    primary = profile.workflow
    plan = {
        "schema": "anime-video-studio/i1-install-plan/v1",
        "profile_id": profile.profile_id,
        "profile_schema": profile.data.get("schema"),
        "profile_sha256": profile.sha256,
        "model_set": profile.data.get("model_set", profile.profile_id),
        "models_policy": profile.model_policy,
        "workspace": ctx.state["identity"]["workspace"],
        "models": [
            {
                "id": item["id"],
                "role": item.get("role"),
                "target": item["target"],
                "size_bytes": item["size_bytes"],
            }
            for item in profile.models
        ],
        "model_bytes": profile.model_bytes(),
        "disk_install_overhead_bytes": profile.data["disk"]["install_overhead_bytes"],
        "disk_reserve_bytes": profile.data["disk"]["reserve_bytes"],
        "disk_required_bytes": profile.disk_required_bytes,
        "disk_additional_required_bytes": inspection.get("facts", {}).get(
            "disk_required_bytes", profile.disk_required_bytes
        ),
        "disk_free_bytes": inspection.get("facts", {}).get("disk_free_bytes"),
        "remote_home": inspection.get("facts", {}).get("home"),
        "comfyui_commit": profile.comfy["commit"],
        "workflow_api": {
            "path": primary["api_path"],
            "sha256": primary["api_sha256"],
        },
        "workflows": workflows,
        "output_specs": list(profile.output_specs),
        "workflow_ui": {
            "path": primary.get("ui_path"),
            "sha256": primary.get("ui_sha256"),
            "status": primary.get("ui_status"),
            "graph_rendered": False,
        },
        "billing_warning": "Аренда уже тарифицируется; установщик не останавливает billing.",
        "no_model_duplicate": True,
        "blockers": list(inspection.get("blockers", [])),
        "warnings": list(inspection.get("warnings", [])),
    }
    stable = {
        key: plan[key]
        for key in (
            "profile_id",
            "profile_schema",
            "profile_sha256",
            "model_set",
            "models_policy",
            "models",
            "model_bytes",
            "disk_install_overhead_bytes",
            "disk_reserve_bytes",
            "disk_required_bytes",
            "comfyui_commit",
            "workflows",
            "output_specs",
        )
    }
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return plan


def plan_install(
    ctx: Context,
    *,
    inspection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inspection = inspection or inspect_remote(
        ctx.runner,
        ctx.identity,
        workspace=ctx.state["identity"]["workspace"],
        profile=ctx.profile,
    )
    if ctx.state["phase"] == "NEW":
        ctx.state = state_mod.transition(ctx.state, "INSPECTED", stage="INSPECT")
    if ctx.state["phase"] in {"INSPECTED", "PLANNED"}:
        ctx.state = state_mod.transition(
            ctx.state,
            "PLANNED",
            stage="PLAN",
            remote_facts=dict(inspection.get("facts", {})),
        )
    else:
        ctx.state = {
            **ctx.state,
            "remote_facts": dict(inspection.get("facts", {})),
            "stage": "INSPECT",
        }
    _save(ctx, ctx.state)
    plan = _plan(ctx, inspection)
    ctx.state = {**ctx.state, "plan": plan, "plan_sha256": plan["plan_sha256"]}
    _save(ctx, ctx.state)
    return {"inspection": inspection, "plan": plan}


def inspect(ctx: Context) -> dict[str, Any]:
    inspection = inspect_remote(
        ctx.runner,
        ctx.identity,
        workspace=ctx.state["identity"]["workspace"],
        profile=ctx.profile,
    )
    return {"profile_id": ctx.profile.profile_id, "inspection": inspection}


def _stage_setup(ctx: Context, root: str) -> None:
    setup_text = setup_text_for(ctx.profile, workspace=ctx.state["identity"]["workspace"])
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".sh", delete=False, newline="\n"
    )
    local = Path(handle.name)
    try:
        with handle:
            handle.write(setup_text)
        i1_remote.stage_file(ctx.runner, ctx.identity, str(PurePosixPath(root) / "setup.sh"), local, root=root, timeout=60.0)
    finally:
        local.unlink(missing_ok=True)
    i1_remote.stage_file(
        ctx.runner,
        ctx.identity,
        str(PurePosixPath(root) / "setup_plan.py"),
        Path(__file__).resolve().parent.parent / "p1" / "setup_plan.py",
        root=root,
        timeout=90.0,
    )


def _run_setup(ctx: Context, root: str) -> None:
    code = "import os,sys;os.chdir(sys.argv[1]);os.execvp('bash',['bash','setup.sh'])"
    result = ctx.runner.run_remote(
        ctx.identity,
        ("python3", "-c", code, root),
        timeout=1800.0,
    )
    if getattr(result, "timed_out", False) or int(getattr(result, "returncode", 1)) != 0:
        log = ""
        try:
            log = i1_remote.tail_log(ctx.runner, ctx.identity, str(PurePosixPath(root) / "logs" / "setup.log"))
        except Exception:
            pass
        detail = redact(str(getattr(result, "stderr", "") or ""), limit=800)
        raise RemoteFailure(
            "setup failed; log: "
            + str(PurePosixPath(root) / "logs" / "setup.log")
            + ("\n" + detail if detail else "")
            + ("\n" + log if log else "")
        )
    if "SETUP_COMPLETE" not in str(getattr(result, "stdout", "") or ""):
        raise RemoteFailure("setup did not publish SETUP_COMPLETE; not marking complete")
    for path in (
        str(PurePosixPath(root) / "logs" / "torch-before.json"),
        str(PurePosixPath(root) / "logs" / "torch-after.json"),
        i1_remote.venv_python(ctx.state["identity"]["workspace"]),
    ):
        check = ctx.runner.run_remote(
            ctx.identity, ("test", "-s" if path.endswith(".json") else "-x", path), timeout=15.0
        )
        if getattr(check, "timed_out", False) or int(getattr(check, "returncode", 1)) != 0:
            raise RemoteFailure(f"setup readback failed: {path}")


def _fetch_api(ctx: Context, *, remote_port: int, workspace: str) -> dict[str, str]:
    evidence = str(PurePosixPath(validate_workspace(workspace)) / "p1-evidence")
    i1_remote.mkdir(ctx.runner, ctx.identity, evidence)
    code = (
        "import os,sys,urllib.request,pathlib,tempfile;"
        "root=pathlib.Path(sys.argv[3]).resolve();p=pathlib.Path(sys.argv[2]);"
        "assert root==p.parent or root in p.parents;"
        "[(_ for _ in ()).throw(SystemExit(20)) for x in p.parents if x!=root and x.is_symlink()];"
        "assert not p.is_symlink();"
        "p.parent.mkdir(parents=True,exist_ok=True);"
        "d=urllib.request.urlopen(sys.argv[1],timeout=30).read(16*1024*1024+1);"
        "fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=str(p.parent));"
        "os.write(fd,d);os.fsync(fd);os.close(fd);os.replace(tmp,p);print(len(d))"
    )
    paths = {
        "object_info": str(PurePosixPath(evidence) / "object_info.json"),
        "system_stats": str(PurePosixPath(evidence) / "system_stats.json"),
    }
    for name, path in paths.items():
        url = f"http://127.0.0.1:{int(remote_port)}/{name}"
        last_error = None
        for attempt in range(3):
            result = ctx.runner.run_remote(
                ctx.identity,
                ("python3", "-c", code, url, path, evidence),
                timeout=60.0,
            )
            if not getattr(result, "timed_out", False) and int(
                getattr(result, "returncode", 1)
            ) == 0:
                break
            last_error = result
            time.sleep(1.0 + attempt)
        else:
            detail = redact(str(getattr(last_error, "stderr", "") or ""), limit=300)
            raise RemoteFailure(f"ComfyUI {name} fetch failed after 3 bounded attempts: {detail}")
    return paths


def _stage_probe(ctx: Context, root: str) -> None:
    package = Path(__file__).resolve().parent
    for name in ("probe.py", "util.py", "profile.py"):
        i1_remote.stage_file(
            ctx.runner,
            ctx.identity,
            str(PurePosixPath(root) / ("i1_probe.py" if name == "probe.py" else "i1_" + name)),
            package / name,
            root=root,
            timeout=60.0,
        )
    i1_remote.stage_file(
        ctx.runner,
        ctx.identity,
        str(PurePosixPath(root) / "preflight.py"),
        Path(__file__).resolve().parent.parent / "p1" / "preflight.py",
        root=root,
        timeout=60.0,
    )


def _run_probe(
    ctx: Context,
    *,
    root: str,
    workspace: str,
    object_info_path: str,
    system_stats_path: str,
    deadline_epoch: float,
    attempt: str = "0001",
    verification: str = "v0001",
) -> dict[str, Any]:
    _stage_probe(ctx, root)
    profile = ctx.profile
    workflow_specs = []
    for workflow_id, meta in profile.workflow_items():
        output_specs = [
            {"workflow_id": workflow_id, **dict(profile.output_specs[0])}
        ] if profile.data.get("schema") != "anime-video-studio/i1-profile/v2" else [
            {"workflow_id": workflow_id, **dict(meta["output"])}
        ]
        workflow_specs.append(
            {
                "workflow_id": workflow_id,
                "path": workflow.api_remote_path_for(root, profile, workflow_id),
                "sha256": meta["api_sha256"],
                "required_classes": list(meta.get("required_classes") or profile.data["required_classes"]),
                "required_inputs": dict(meta.get("required_inputs") or profile.data["required_inputs"]),
                "node_classes": dict(meta.get("node_classes") or profile.data["node_classes"]),
                "graph_inputs": list(meta.get("graph_inputs") or profile.data["graph_inputs"]),
                "output_specs": output_specs,
            }
        )
    primary = profile.workflow
    plan = {
        "schema": "anime-video-studio/i1-runtime-plan/v1",
        "run_id": ctx.state["run_id"],
        "nonce": ctx.state["nonce"],
        "workspace": workspace,
        "owned_root": root,
        "comfy_dir": i1_remote.comfy_dir(workspace),
        "venv_python": i1_remote.venv_python(workspace),
        "profile_path": download.remote_paths(root, attempt)["profile"],
        "profile_sha256": profile.sha256,
        "workflow_api_path": str(PurePosixPath(root) / "workflow" / f"{profile.profile_id}.api.json"),
        "workflow_sha256": primary["api_sha256"],
        "object_info_path": object_info_path,
        "system_stats_path": system_stats_path,
        "preflight_path": str(PurePosixPath(root) / "preflight.py"),
        "expected_comfy_commit": profile.comfy["commit"],
        "expected_model_count": len(profile.models),
        "required_classes": list(profile.required_classes),
        "required_inputs": profile.required_inputs,
        "node_classes": dict(primary.get("node_classes") or profile.data["node_classes"]),
        "graph_inputs": list(primary.get("graph_inputs") or profile.data["graph_inputs"]),
        "workflows": workflow_specs,
        "output_specs": list(profile.output_specs),
        "models": [
            {
                "id": item["id"],
                "role": item.get("role"),
                "target": item["target"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in profile.models
        ],
        "deadline_epoch": float(deadline_epoch),
    }
    verification_dir = str(
        PurePosixPath(root) / "verifications" / verification
    )
    plan_remote = str(PurePosixPath(verification_dir) / "runtime-plan.json")
    i1_remote.mkdir(ctx.runner, ctx.identity, verification_dir)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False, newline="\n"
    )
    local = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        i1_remote.stage_file(ctx.runner, ctx.identity, plan_remote, local, root=root, timeout=30.0)
    finally:
        local.unlink(missing_ok=True)
    result = ctx.runner.run_remote(
        ctx.identity,
        (
            i1_remote.venv_python(workspace),
            str(PurePosixPath(root) / "i1_probe.py"),
            "--plan",
            plan_remote,
        ),
        timeout=max(60.0, min(1200.0, deadline_epoch - time.time())),
    )
    if getattr(result, "timed_out", False) or int(getattr(result, "returncode", 1)) != 0:
        detail = redact(str(getattr(result, "stderr", "") or ""), limit=1000)
        raise RemoteFailure(f"actual probe failed: {detail}")
    try:
        payload = json.loads(str(result.stdout))
    except json.JSONDecodeError as exc:
        raise RemoteFailure("actual probe output is not JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "anime-video-studio/i1-runtime-evidence/v1"
        or payload.get("run_id") != ctx.state["run_id"]
        or payload.get("nonce") != ctx.state["nonce"]
    ):
        raise RemoteFailure("actual probe identity/schema mismatch")
    return payload


def _continue_install(
    ctx: Context,
    *,
    deadline_epoch: float | None = None,
    progress: Any | None = None,
    new_attempt: bool = False,
) -> dict[str, Any]:
    workspace = ctx.state["identity"]["workspace"]
    root = i1_remote.owned_root(workspace)
    profile = ctx.profile
    state = ctx.state

    def emit() -> None:
        if progress is not None:
            progress(ctx.state)
    if state.get("plan_sha256"):
        current_plan = _plan(ctx, {"facts": state.get("remote_facts", {})})
        if current_plan["plan_sha256"] != state["plan_sha256"]:
            raise BlockedError(
                "install plan changed since it was persisted; refusing to resume with a "
                "different profile/model/workflow set"
            )
    artifacts = dict(state.get("artifacts", {}))
    attempt_info = artifacts.get("download_attempt")
    existing_attempt_id = (
        str(attempt_info["id"])
        if isinstance(attempt_info, Mapping) and attempt_info.get("id")
        else None
    )
    observed: observation.ObservationResult | None = None
    decision: dict[str, Any] = {}
    attached_existing = False
    if existing_attempt_id:
        observed = _observe_attempt(ctx, attempt=existing_attempt_id)
        _record_observation(ctx, observed, stage="OBSERVE")
        state = ctx.state
        artifacts = dict(state.get("artifacts", {}))
        if observed.kind != "OK":
            raise Interrupted(
                "remote observation unavailable; no mutation; cached attempt preserved"
            )
        decision = dict(observed.decision or {})
        if decision.get("state") == "MISSING":
            if not new_attempt:
                raise Interrupted(
                    "recorded attempt is absent; use resume --new-attempt only after confirming"
                )
        elif decision.get("state") != "OK":
            raise Interrupted(
                "download attempt observation is inconclusive; no mutation"
            )
        elif decision.get("process") in {"UNKNOWN", "MISMATCH"}:
            raise Interrupted("download process identity is unknown; no mutation")
        elif decision.get("process") == "LIVE":
            attached_existing = True
        elif (decision.get("status") or {}).get("status") == "COMPLETE":
            state = {**state, "artifacts": {**artifacts, "download": "COMPLETE"}}
            _save(ctx, state)
            return _finalize_after_download(
                ctx, root=root, workspace=workspace, profile=profile, state=state
            )
    if not existing_attempt_id and state["phase"] == "INSTALLING" and (
        artifacts.get("setup") != "COMPLETE"
    ):
        state = state_mod.update_stage(state, "ROOT")
        _save(ctx, state)
        prepare_remote_root(
            runner=ctx.runner,
            identity=ctx.identity,
            remote_root=root,
            run_id=state["run_id"],
            nonce=state["nonce"],
            timeout=30.0,
        )
        state = {**state, "artifacts": {**state["artifacts"], "setup": "STAGED"}}
        _save(ctx, state)
        _stage_setup(ctx, root)
        state = state_mod.update_stage(state, "SETUP")
        _save(ctx, state)
        _run_setup(ctx, root)
        state = {**state, "artifacts": {**state["artifacts"], "setup": "COMPLETE"}}
        _save(ctx, state)
    elif existing_attempt_id and artifacts.get("setup") != "COMPLETE":
        raise Interrupted(
            "existing download attempt was observed but setup is not complete; no mutation"
        )
    artifacts = dict(state.get("artifacts", {}))
    remote_facts = dict(state.get("remote_facts", {}))
    remote_home = remote_facts.get("home")
    if not isinstance(remote_home, str) or not remote_home.startswith("/"):
        inspection = inspect_remote(
            ctx.runner,
            ctx.identity,
            workspace=workspace,
            profile=profile,
        )
        remote_facts = dict(inspection.get("facts", {}))
        remote_home = remote_facts.get("home")
        state = {**state, "remote_facts": remote_facts}
        _save(ctx, state)
    if not isinstance(remote_home, str) or not remote_home.startswith("/"):
        raise BlockedError("remote home is unavailable; target-level lock cannot be proven")
    target_id_value = str(state["identity"].get("target_id") or "")
    if not target_id_value:
        target_id_value = state_mod.target_id(
            host=state["identity"]["host"],
            user=state["identity"]["user"],
            port=int(state["identity"]["port"]),
            fingerprint=state["identity"]["fingerprint"],
        )
        state = {
            **state,
            "identity": {**state["identity"], "target_id": target_id_value},
        }
        _save(ctx, state)
    target_lock_path = str(
        PurePosixPath(remote_home)
        / ".cache"
        / "anime-video-studio"
        / "i1"
        / "targets"
        / f"{target_id_value}.lock"
    )
    if state["phase"] in {"INSTALLED", "VERIFYING"} or artifacts.get("download") == "COMPLETE":
        state = {**state, "artifacts": {**artifacts, "download": "COMPLETE"}}
        return _finalize_after_download(
            ctx,
            root=root,
            workspace=workspace,
            profile=profile,
            state=state,
        )
    attempt_info = artifacts.get("download_attempt")
    if new_attempt and existing_attempt_id:
        # The old attempt was observed read-only above.  Only a proved
        # dead/absent or safely completed attempt may be superseded.
        if not download.new_attempt_allowed(decision):
            status = (decision.get("status") or {}).get("status")
            raise Interrupted(
                "old download attempt is not proven dead/completed "
                f"(state={decision.get('state')}, process={decision.get('process')}, "
                f"status={status}); start refused"
            )
        attempt_number = int(existing_attempt_id) + 1
        attempt_info = {"id": f"{attempt_number:04d}"}
    if not isinstance(attempt_info, Mapping) or not attempt_info.get("id"):
        attempt_info = {
            "id": "0001",
            "created_at": time.time(),
            "deadline_epoch": time.time() + INSTALL_DEADLINE_SECONDS,
        }
    attempt_id = str(attempt_info["id"])
    attempt_deadline = float(
        attempt_info.get("deadline_epoch")
        or deadline_epoch
        or (time.time() + INSTALL_DEADLINE_SECONDS)
    )
    if attempt_deadline <= time.time():
        raise Interrupted(
            "download attempt deadline expired; no blind new deadline. "
            "Use resume --new-attempt only after the old worker is proven dead."
        )
    # Register the immutable attempt/deadline locally before any remote upload.
    # A crash after any put resumes the same id/path/deadline; no collision.
    attempt_info = {
        **dict(attempt_info),
        "id": attempt_id,
        "deadline_epoch": attempt_deadline,
        "dir": str(PurePosixPath(root) / "attempts" / attempt_id),
    }
    state = {
        **state,
        "artifacts": {**artifacts, "download_attempt": attempt_info},
    }
    _save(ctx, state)
    recorded_launch = attempt_info.get("launch_sha256")
    needs_stage = not attached_existing and (
        not existing_attempt_id or new_attempt or attempt_id != existing_attempt_id
    )
    if needs_stage:
        staged = download.stage(
            ctx.runner,
            ctx.identity,
            root=root,
            workspace=workspace,
            profile=profile,
            run_id=state["run_id"],
            nonce=state["nonce"],
            deadline_epoch=attempt_deadline,
            target_home=remote_home,
            target_id=target_id_value,
            target_lock_path=target_lock_path,
            attempt=attempt_id,
            expected_launch_sha256=recorded_launch if not new_attempt else None,
        )
        attempt_info = {
            **dict(attempt_info),
            "id": attempt_id,
            "deadline_epoch": attempt_deadline,
            "job_sha256": staged.get("job_sha256"),
            "launch_sha256": staged.get("launch_sha256"),
            "dir": staged["paths"]["attempt"],
        }
        state = {
            **state,
            "artifacts": {**artifacts, "download_attempt": attempt_info},
        }
        observed = None
        decision = {}
    state = state_mod.update_stage(state, "DOWNLOAD")
    _save(ctx, state)
    start_at = time.time()
    retried = False
    current_decision: dict[str, Any] | None = (
        dict(decision)
        if observed is not None and observed.kind == "OK" and not new_attempt
        else None
    )
    while True:
        if current_decision is None:
            observed = _observe_attempt(ctx, attempt=attempt_id)
            _record_observation(ctx, observed, stage="DOWNLOAD")
            state = ctx.state
            if observed.kind != "OK":
                raise Interrupted(
                    "remote observation lost; no mutation; cached attempt preserved"
                )
            current_decision = dict(observed.decision or {})
        current = current_decision
        record_state = current.get("state")
        if record_state == "INCONCLUSIVE":
            raise Interrupted(
                "download attempt observation is inconclusive; no mutation"
            )
        if record_state == "UNREADABLE":
            # A read error is not "no worker"; refusing prevents a second
            # worker being started beside an unreadable live attempt.
            raise RemoteFailure(
                "download attempt is unreadable; refusing to trust status: "
                + str(current.get("reason", "unreadable"))
            )
        if record_state == "MISSING":
            started = download.start(
                ctx.runner,
                ctx.identity,
                root=root,
                run_id=state["run_id"],
                nonce=state["nonce"],
                attempt=attempt_id,
                timeout=30.0,
            )
            if started.get("status") in {"ATTACHED_COMPLETE", "COMPLETED"}:
                break
        else:
            status_payload = current.get("status") or {}
            status = status_payload.get("status")
            process = current.get("process")
            progress = current.get("progress") or {}
            progress_view = {
                "stage": progress.get("stage"),
                "current_file": progress.get("current_file"),
                "files_done": progress.get("files_done", 0),
                "files_total": progress.get("files_total", len(profile.models)),
                "bytes_done": progress.get("bytes_done", 0),
                "bytes_total": progress.get("bytes_total", profile.model_bytes()),
                "heartbeat_at": progress.get("heartbeat_at"),
            }
            for key in (
                "bytes_remaining",
                "speed_bytes_per_sec",
                "eta_seconds",
                "eta_known",
                "current_files",
                "active_files",
                "files_active",
            ):
                if key in progress:
                    progress_view[key] = progress[key]
            state = {
                **state,
                "progress": progress_view,
                "worker": {
                    "owner": current.get("owner"),
                    "ack": current.get("ack"),
                    "status": status_payload,
                    "progress": progress,
                    "heartbeat": current.get("heartbeat") or {},
                },
            }
            _save(ctx, state)
            if status == "COMPLETE":
                break
            if process in {"UNKNOWN", "MISMATCH"}:
                raise RemoteFailure(
                    f"download process identity is {process}; refusing attach/resume"
                )
            if status == "FAILED":
                if process == "LIVE":
                    raise RemoteFailure(
                        "download worker reported FAILED but its process is still live; refusing resume"
                    )
                if retried:
                    raise Interrupted(
                        "downloader повторно FAILED; причина в status/log, resume остановлен"
                    )
                retried = True
                started = download.start(
                    ctx.runner,
                    ctx.identity,
                    root=root,
                    run_id=state["run_id"],
                    nonce=state["nonce"],
                    attempt=attempt_id,
                    resume_dead=True,
                    timeout=30.0,
                )
                if started.get("status") in {"ATTACHED_COMPLETE", "COMPLETED"}:
                    break
            elif status in {None, "RUNNING"}:
                if process in {"DEAD", "ABSENT"}:
                    if status is None:
                        raise Interrupted(
                            "worker died before publishing a recoverable status; "
                            "use resume --new-attempt"
                        )
                    if retried:
                        raise Interrupted("worker не восстановился; resume остановлен")
                    retried = True
                    # The guard proved the recorded PID absent; a zero stale
                    # window is safe here because process death is the proof.
                    download.recover(
                        ctx.runner,
                        ctx.identity,
                        root=root,
                        run_id=state["run_id"],
                        nonce=state["nonce"],
                        target_lock_path=target_lock_path,
                        attempt=attempt_id,
                        stale_seconds=0.0,
                        timeout=30.0,
                    )
                    started = download.start(
                        ctx.runner,
                        ctx.identity,
                        root=root,
                        run_id=state["run_id"],
                        nonce=state["nonce"],
                        attempt=attempt_id,
                        resume_dead=True,
                        timeout=30.0,
                    )
                    if started.get("status") in {"ATTACHED_COMPLETE", "COMPLETED"}:
                        break
                elif process != "LIVE":
                    raise RemoteFailure(
                        f"download process state {process!r} cannot be attached safely"
                    )
            else:
                raise RemoteFailure(
                    f"download worker state {status!r} cannot be resumed safely"
                )
        current_decision = None
        if time.time() >= attempt_deadline:
            raise Interrupted("download attempt deadline expired; worker may still be running, use status/resume")
        time.sleep(1.0)
    state = {**state, "artifacts": {**state["artifacts"], "download": "COMPLETE"}}
    return _finalize_after_download(
        ctx,
        root=root,
        workspace=workspace,
        profile=profile,
        state=state,
    )


def _wait_health(local_port: int, *, timeout: float, recover=None) -> dict[str, Any]:
    end = time.monotonic() + timeout
    last = None
    recovered = False
    while time.monotonic() < end:
        try:
            return tunnel.health(local_port=local_port, timeout=3.0)
        except Exception as exc:
            last = exc
            if recover is not None and not recovered:
                recovered = True
                # ensure only attaches a proven owner or replaces a proven-dead
                # bundle. Unknown/live/foreign ownership is never overridden.
                recover()
            time.sleep(1.0)
    raise RemoteFailure(
        f"ComfyUI did not become healthy within {timeout:.0f}s; tunnel log: "
        f"{last if last is not None else 'unavailable'}"
    )


def _invalidate_readiness(state: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically retire a previous READY before a new verification begins.

    Any failure in workflow/Comfy/tunnel/health/API/probe stages must leave the
    current readiness UNCONFIRMED and false.  The prior report is kept only in a
    separate history list and is never surfaced as current.
    """

    result = dict(state)
    previous = result.get("readiness")
    history = list(result.get("readiness_history") or [])
    if isinstance(previous, Mapping) and previous:
        history.append(
            {
                "retired_at": state_mod.now_iso(),
                "phase": result.get("phase"),
                "stage": result.get("stage"),
                "report": dict(previous),
            }
        )
        history = history[-20:]
    result["readiness_history"] = history
    result["readiness"] = {
        "status": "UNCONFIRMED",
        "ready": False,
        "checks": [],
        "blockers": ["a new verification has started; previous readiness is historical"],
        "warnings": ["cached readiness is not current until the full probe passes"],
    }
    result["readiness_current"] = False
    return result


def _finalize_after_download(
    ctx: Context,
    *,
    root: str,
    workspace: str,
    profile: Profile,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Workflow/Comfy/tunnel/actual-probe finalization without reinstall."""

    state = _invalidate_readiness(dict(state))
    state = state_mod.update_stage(state, "FINALIZE")
    _save(ctx, state)
    ui = (state.get("artifacts") or {}).get("workflow")
    if not isinstance(ui, Mapping):
        state = state_mod.update_stage(state, "WORKFLOW")
        _save(ctx, state)
        ui = workflow.register(
            ctx.runner,
            ctx.identity,
            root=root,
            workspace=workspace,
            profile=profile,
            repo_root=ROOT,
        )
        state = {
            **state,
            "artifacts": {**state["artifacts"], "workflow": ui},
            "ui": ui,
        }
        _save(ctx, state)
    ui = dict(ui)
    if state["phase"] == "READY":
        state = state_mod.transition(state, "VERIFYING", stage="REVERIFY")
    else:
        state = state_mod.transition(state, "INSTALLED", stage="INSTALLED")
    _save(ctx, state)
    state = state_mod.update_stage(state, "COMFY")
    _save(ctx, state)
    comfy.stage_launch(
        ctx.runner,
        ctx.identity,
        root=workspace,
        workspace=workspace,
        profile=profile,
        run_id=state["run_id"],
        nonce=state["nonce"],
    )
    comfy.start(
        ctx.runner,
        ctx.identity,
        root=workspace,
        run_id=state["run_id"],
        nonce=state["nonce"],
    )
    state = state_mod.update_stage(state, "TUNNEL")
    _save(ctx, state)
    tunnel.ensure(
        identity=ctx.identity,
        run_dir=ctx.run_dir,
        run_id=state["run_id"],
        nonce=state["nonce"],
        fingerprint=ctx.state["identity"]["fingerprint"],
        local_port=int(profile.data["runtime"]["local_tunnel_port"]),
        remote_port=int(profile.data["runtime"]["comfy_remote_port"]),
    )
    try:
        health = _wait_health(
            int(profile.data["runtime"]["local_tunnel_port"]), timeout=90.0,
            recover=lambda: tunnel.ensure(
                identity=ctx.identity, run_dir=ctx.run_dir,
                run_id=state["run_id"], nonce=state["nonce"],
                fingerprint=ctx.state["identity"]["fingerprint"],
                local_port=int(profile.data["runtime"]["local_tunnel_port"]),
                remote_port=int(profile.data["runtime"]["comfy_remote_port"]),
            ),
        )
    except RemoteFailure as exc:
        log = ""
        try:
            log = i1_remote.tail_log(
                ctx.runner,
                ctx.identity,
                str(PurePosixPath(workspace) / "comfy.log"),
            )
        except Exception:
            pass
        raise RemoteFailure(
            f"{exc}; ComfyUI log: {PurePosixPath(workspace) / 'comfy.log'}"
            + (f"\n{redact(log, limit=2000)}" if log else "")
        ) from exc
    fetched = _fetch_api(
        ctx,
        remote_port=int(profile.data["runtime"]["comfy_remote_port"]),
        workspace=workspace,
    )
    state = state_mod.transition(state, "INSTALLED", stage="INSTALLED")
    _save(ctx, state)
    state = state_mod.transition(state, "VERIFYING", stage="PROBE")
    download_attempt = (state.get("artifacts", {}).get("download_attempt") or {}).get(
        "id", "0001"
    )
    verification_id = f"v{time.time_ns()}"
    verification_deadline = time.time() + 1800
    state = {
        **state,
        "artifacts": {
            **state.get("artifacts", {}),
            "verification_attempt": {
                "id": verification_id,
                "deadline_epoch": verification_deadline,
                "dir": str(PurePosixPath(root) / "verifications" / verification_id),
            },
        },
    }
    _save(ctx, state)
    try:
        probe = _run_probe(
            ctx,
            root=root,
            workspace=workspace,
            object_info_path=fetched["object_info"],
            system_stats_path=fetched["system_stats"],
            deadline_epoch=verification_deadline,
            attempt=str(download_attempt),
            verification=verification_id,
        )
    except Exception as exc:
        state = {
            **state,
            "readiness": {
                "status": "INSTALLED",
                "ready": False,
                "blockers": [type(exc).__name__],
                "warnings": ["actual probe did not complete; READY is not claimed"],
            },
            "last_error": redact(str(exc), limit=800),
        }
        _save(ctx, state)
        raise NotReady("установка завершена, но actual probe не прошёл; READY не подтверждён") from exc
    ui_listed = workflow.list_userdata(
        local_port=int(profile.data["runtime"]["local_tunnel_port"]),
        filename=ui.get("ui_filename"),
    )
    ui = {**ui, "ui_listing": ui_listed}
    report = readiness.evaluate(
        probe=probe,
        profile=profile,
        tunnel_health=health,
        ui=ui,
    )
    state = {
        **state,
        "readiness": report,
        "readiness_current": bool(report.get("ready")),
        "ui": ui,
        "comfy": {
            "owner": comfy.read_owner(
                ctx.runner,
                ctx.identity,
                root=workspace,
                run_id=state["run_id"],
                nonce=state["nonce"],
            )
        },
        "tunnel": {
            "local_port": int(profile.data["runtime"]["local_tunnel_port"]),
            "remote_port": int(profile.data["runtime"]["comfy_remote_port"]),
            "owner_path": str(tunnel.owner_path(ctx.run_dir)),
        },
    }
    state = state_mod.transition(
        state,
        "READY" if report.get("ready") else "INSTALLED",
        stage="READY" if report.get("ready") else "INSTALLED",
    )
    _save(ctx, state)
    return {
        "plan": _plan(ctx, {"facts": state.get("remote_facts", {})}),
        "state": state,
        "readiness": report,
        # Raw actual-probe evidence for a caller that must authorize a job in
        # this exact run (never a persisted readiness boolean).
        "probe": probe,
    }


def _resume_ready(ctx: Context) -> dict[str, Any]:
    attempt_id = _attempt_id(ctx.state)
    observed = _observe_attempt(ctx, attempt=attempt_id or None)
    _record_observation(ctx, observed, stage="RESUME")
    if observed.kind != "OK":
        raise Interrupted(
            "cached READY is not current; remote observation unavailable; no mutation"
        )
    decision = dict(observed.decision or {})
    if (decision.get("status") or {}).get("status") != "COMPLETE":
        raise Interrupted(
            "cached READY is not current; downloader is not confirmed COMPLETE"
        )
    workspace = ctx.state["identity"]["workspace"]
    return _finalize_after_download(
        ctx,
        root=i1_remote.owned_root(workspace),
        workspace=workspace,
        profile=ctx.profile,
        state=ctx.state,
    )


def execute_install(
    ctx: Context,
    *,
    callbacks: Any | None = None,
    progress: Any | None = None,
    new_attempt: bool = False,
) -> dict[str, Any]:
    ctx.progress = progress or callbacks
    workspace = ctx.state["identity"]["workspace"]
    deadline_epoch = time.time() + INSTALL_DEADLINE_SECONDS
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = ctx.run_dir / ".install.lock"
    from anime_studio.lock import InstanceLock

    lock = InstanceLock(lock_path)
    try:
        lock.acquire()
    except Exception as exc:
        raise BlockedError("другой локальный installer уже работает с этим target/profile") from exc
    try:
        state = ctx.state
        if state["phase"] in {"NEW", "INSPECTED", "PLANNED"}:
            inspection = inspect_remote(
                ctx.runner,
                ctx.identity,
                workspace=state["identity"]["workspace"],
                profile=ctx.profile,
            )
            if inspection.get("blockers"):
                blockers = "; ".join(item["message"] for item in inspection["blockers"])
                failed = state_mod.transition(
                    state, "FAILED", stage="INSPECT", error=blockers
                )
                _save(ctx, failed)
                raise BlockedError("install refused before mutation: " + blockers)
            if inspection.get("conclusive") is False:
                raise RemoteFailure(
                    "environment observation is inconclusive; no install mutation was performed"
                )
            if state["phase"] == "NEW":
                state = state_mod.transition(state, "INSPECTED", stage="INSPECT")
            planned = _plan(ctx, inspection)
            state = {
                **state,
                "plan": planned,
                "plan_sha256": planned["plan_sha256"],
            }
            state = state_mod.transition(
                state,
                "PLANNED",
                stage="PLAN",
                remote_facts=dict(inspection.get("facts", {})),
            )
            state = state_mod.transition(state, "INSTALLING", stage="INSTALL")
        elif state["phase"] in {"INTERRUPTED", "INSTALLING"}:
            state = state_mod.transition(state, "INSTALLING", stage="RESUME")
        elif state["phase"] == "INSTALLED":
            state = state_mod.transition(state, "VERIFYING", stage="RESUME")
        elif state["phase"] == "VERIFYING":
            state = state_mod.update_stage(state, "RESUME")
        elif state["phase"] == "READY":
            workspace = state["identity"]["workspace"]
            ctx.state = state
            return _resume_ready(ctx)
        else:
            raise StateError(f"нельзя продолжать из состояния {state['phase']}")
        ctx.state = state
        _save(ctx, state)
        try:
            return _continue_install(
                ctx, deadline_epoch=deadline_epoch, new_attempt=new_attempt
            )
        except KeyboardInterrupt as exc:
            interrupted = state_mod.transition(ctx.state, "INTERRUPTED", stage="INTERRUPTED", error="KeyboardInterrupt")
            _save(ctx, interrupted)
            raise Interrupted("установка прервана локально; remote downloader может продолжать, используйте resume") from exc
        except Interrupted as exc:
            if ctx.state.get("phase") == "INSTALLING":
                interrupted = state_mod.transition(
                    ctx.state, "INTERRUPTED", stage="INTERRUPTED", error=redact(str(exc), limit=500)
                )
                _save(ctx, interrupted)
            raise
        except RemoteFailure as exc:
            if ctx.state.get("phase") == "INSTALLING":
                interrupted = state_mod.transition(
                    ctx.state, "INTERRUPTED", stage="INTERRUPTED", error=redact(str(exc), limit=500)
                )
                _save(ctx, interrupted)
            raise
    finally:
        lock.release()


def resume(
    ctx: Context,
    *,
    callbacks: Any | None = None,
    progress: Any | None = None,
    new_attempt: bool = False,
) -> dict[str, Any]:
    return execute_install(
        ctx, callbacks=callbacks, progress=progress, new_attempt=new_attempt
    )


def _status_readiness(phase: str, raw: Any, *, offline: bool) -> dict[str, Any]:
    """Never surface a cached READY as the current state from `status`.

    Only phase READY with a fresh install/resume probe may claim READY in the
    future; this command always marks the cached report UNCONFIRMED.  When the
    state is not READY (VERIFYING/INSTALLED/FAILED/...), even a stale READY
    report is downgraded rather than shown as current.
    """

    report = dict(raw) if isinstance(raw, Mapping) else {}
    if report.get("status") == "READY" or report.get("ready"):
        report["status"] = "UNCONFIRMED"
        report["ready"] = False
        report["cached"] = True
        warnings = list(report.get("warnings", []))
        warnings.append(
            "status is cached/offline and does not run a fresh full probe"
            if offline
            else "cached READY is not current until install/resume runs a fresh full probe"
        )
        report["warnings"] = warnings
    elif phase != "READY" and not report:
        report = {
            "status": "UNCONFIRMED",
            "ready": False,
            "checks": [],
            "blockers": [],
            "warnings": [],
        }
    return report


def status(ctx: Context, *, offline: bool = False) -> dict[str, Any]:
    workspace = ctx.state["identity"]["workspace"]
    root = i1_remote.owned_root(workspace)
    payload: dict[str, Any] = {
        "status": ctx.state["phase"],
        "phase": ctx.state["phase"],
        "stage": ctx.state.get("stage"),
        "profile_id": ctx.profile.profile_id,
        "workspace": workspace,
        "run_id": ctx.state["run_id"],
        "nonce": ctx.state["nonce"],
        "readiness": _status_readiness(
            ctx.state["phase"], ctx.state.get("readiness", {}), offline=offline
        ),
        "readiness_current": False,
        "historical_readiness": list(ctx.state.get("readiness_history") or []),
        "local": {
            "phase": ctx.state["phase"],
            "stage": ctx.state.get("stage"),
            "progress": ctx.state.get("progress", {}),
        },
    }
    if offline:
        payload["offline"] = True
        payload["effective_connection_state"] = "UNCONFIRMED"
        payload["worker_state"] = "UNKNOWN"
        payload["last_confirmed_at"] = (
            (ctx.state.get("observation") or {}).get("last_confirmed_at")
        )
        payload["progress_fresh"] = False
        return payload
    attempt = (ctx.state.get("artifacts", {}).get("download_attempt") or {})
    attempt_id = str(attempt.get("id") or "")
    observed = _observe_attempt(ctx, attempt=attempt_id or None)
    _record_observation(ctx, observed, stage="STATUS")
    decision = dict(observed.decision or {})
    observation_state = dict(ctx.state.get("observation") or {})
    payload["observation"] = observation_state
    payload["effective_connection_state"] = observation_state.get("connection_state")
    payload["worker_state"] = observation_state.get("worker_state")
    payload["worker_responsiveness"] = observation_state.get("worker_responsiveness")
    payload["hb_responsive"] = observation_state.get("hb_responsive")
    payload["last_confirmed_at"] = observation_state.get("last_confirmed_at")
    payload["last_attempt_at"] = observation_state.get("last_attempt_at")
    payload["progress_fresh"] = bool(observation_state.get("progress_fresh"))
    payload["rate_fresh"] = bool(observation_state.get("rate_fresh"))
    payload["bytes_observed_at"] = observation_state.get("bytes_observed_at")
    payload["diagnostic"] = observation_state.get("diagnostic")
    if observed.kind == "OK":
        if decision.get("state") == "OK":
            progress = dict(decision.get("progress") or {})
            payload["worker"] = {
                "owner": decision.get("owner"),
                "ack": decision.get("ack"),
                "status": decision.get("status") or {},
                "progress": progress,
                "heartbeat": decision.get("heartbeat") or {},
            }
            payload["worker_status"] = (decision.get("status") or {}).get("status")
            if observation_state.get("progress_fresh"):
                progress_view = dict(progress)
                if not observation_state.get("rate_fresh"):
                    progress_view["speed_bytes_per_sec"] = None
                    progress_view["eta_seconds"] = None
                    progress_view["eta_known"] = False
                payload["progress"] = progress_view
        elif decision.get("state") == "MISSING":
            payload["worker"] = {"state": "MISSING", "process": "ABSENT"}
            payload["worker_status"] = None
        else:
            payload["worker"] = {
                "state": decision.get("state"),
                "process": decision.get("process"),
                "reason": decision.get("reason"),
            }
            payload["worker_status"] = None
        payload["comfy_owner"] = (ctx.state.get("comfy") or {}).get("owner")
    else:
        payload["worker"] = {"state": "UNKNOWN", "process": "UNKNOWN"}
        payload["worker_status"] = None
        diagnostic = observation_state.get("diagnostic")
        payload["worker_error"] = (
            diagnostic.get("message")
            if isinstance(diagnostic, Mapping)
            else str(diagnostic or "")
        )
        payload["comfy_owner"] = (ctx.state.get("comfy") or {}).get("owner")
    if not observation_state.get("progress_fresh") or observed.kind != "OK":
        cached = observation_state.get("cached_progress")
        if isinstance(cached, Mapping) and cached:
            cached_as_of = observation_state.get("cached_as_of")
        else:
            cached = ctx.state.get("progress") or {}
            cached_as_of = observation_state.get("cached_as_of") or observation_state.get(
                "last_confirmed_at"
            )
        display = dict(cached) if isinstance(cached, Mapping) else {}
        display["speed_bytes_per_sec"] = None
        display["eta_seconds"] = None
        display["eta_known"] = False
        display["fresh"] = False
        display["as_of"] = cached_as_of
        payload["progress"] = None
        payload["cached_progress"] = display
        payload["cached_as_of"] = cached_as_of
        payload["speed_bytes_per_sec"] = None
        payload["eta_seconds"] = None
        payload["eta_known"] = False
    else:
        progress = dict(payload.get("progress") or {})
        if observation_state.get("rate_fresh"):
            payload["speed_bytes_per_sec"] = progress.get("speed_bytes_per_sec")
            payload["eta_seconds"] = progress.get("eta_seconds")
            payload["eta_known"] = bool(progress.get("eta_known"))
        else:
            payload["speed_bytes_per_sec"] = None
            payload["eta_seconds"] = None
            payload["eta_known"] = False
        payload["cached_progress"] = None
        payload["cached_as_of"] = None
    if observed.kind != "OK":
        readiness = dict(payload.get("readiness") or {})
        readiness["status"] = "UNCONFIRMED"
        readiness["ready"] = False
        warnings = list(readiness.get("warnings") or [])
        warnings.append("remote state not confirmed; cached progress only")
        readiness["warnings"] = warnings
        payload["readiness"] = readiness
    payload["readiness_current"] = False
    if ctx.state["phase"] == "READY":
        # A cached probe cannot be made current by arbitrary localhost HTTP.
        # Only install/resume with a fresh full probe may claim current READY.
        try:
            health = tunnel.health(
                local_port=int(ctx.profile.data["runtime"]["local_tunnel_port"]),
                timeout=3.0,
            )
            payload["tunnel_health"] = {
                "reachable": bool(health.get("system")),
                "current_ready": False,
            }
        except Exception:
            payload["tunnel_health"] = {"reachable": False, "current_ready": False}
    return payload


def open_comfy(ctx: Context, *, browser: bool = False) -> dict[str, Any]:
    if ctx.state["phase"] not in {"READY", "INSTALLED"}:
        raise BlockedError("сначала выполните install/resume; ComfyUI ещё не установлен")
    workspace = ctx.state["identity"]["workspace"]
    root = i1_remote.owned_root(workspace)
    profile = ctx.profile
    comfy.stage_launch(
        ctx.runner,
        ctx.identity,
        root=workspace,
        workspace=workspace,
        profile=profile,
        run_id=ctx.state["run_id"],
        nonce=ctx.state["nonce"],
    )
    comfy.start(
        ctx.runner,
        ctx.identity,
        root=workspace,
        run_id=ctx.state["run_id"],
        nonce=ctx.state["nonce"],
    )
    tunnel.ensure(
        identity=ctx.identity,
        run_dir=ctx.run_dir,
        run_id=ctx.state["run_id"],
        nonce=ctx.state["nonce"],
        fingerprint=ctx.state["identity"]["fingerprint"],
        local_port=int(profile.data["runtime"]["local_tunnel_port"]),
        remote_port=int(profile.data["runtime"]["comfy_remote_port"]),
    )
    local_port = int(profile.data["runtime"]["local_tunnel_port"])
    health = _wait_health(local_port, timeout=30.0)
    url = f"http://127.0.0.1:{local_port}"
    if browser:
        webbrowser.open(url)
    ui = ctx.state.get("ui", {})
    return {
        "status": "OPEN",
        "url": url,
        "health": health,
        "ui_status": ui.get("ui_status", "UI_GRAPH_NOT_VERIFIED"),
        "ui_graph_rendered": False,
        "local_api_file": str(ROOT / profile.workflow["api_path"]),
        "local_ui_file": str(ROOT / profile.workflow["ui_path"]) if profile.workflow.get("ui_path") else None,
        "billing_warning": "Аренда не остановлена; disconnect закрывает только owned tunnel.",
    }


def disconnect(ctx: Context) -> dict[str, Any]:
    stopped = tunnel.stop(
        identity=ctx.identity,
        run_dir=ctx.run_dir,
        run_id=ctx.state["run_id"],
        nonce=ctx.state["nonce"],
        fingerprint=ctx.state["identity"]["fingerprint"],
    )
    if stopped and ctx.state["phase"] == "READY":
        readiness = dict(ctx.state.get("readiness") or {})
        readiness["status"] = "UNCONFIRMED"
        readiness["ready"] = False
        readiness["warnings"] = list(readiness.get("warnings", [])) + [
            "owned tunnel disconnected; cached READY is not current"
        ]
        ctx.state = state_mod.transition(
            ctx.state, "READY", stage="DISCONNECTED", readiness=readiness
        )
        _save(ctx, ctx.state)
    return {
        "status": "DISCONNECTED" if stopped else "CLEANUP_UNCONFIRMED",
        "owned_tunnel_only": True,
        "billing_warning": "Аренда и ComfyUI не остановлены; disconnect не останавливает billing.",
    }


__all__ = [
    "ConnectArgs",
    "Context",
    "connect",
    "disconnect",
    "execute_install",
    "inspect",
    "load_context",
    "make_runner",
    "open_comfy",
    "plan_install",
    "resume",
    "status",
]
