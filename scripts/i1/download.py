"""Local client for the owned I1 downloader; uses the frozen P1 launcher."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from scripts.p1.remote import RemoteRunner

from . import remote as i1_remote
from .errors import RemoteFailure
from .profile import Profile
from .util import redact, sha256_file, validate_posix_absolute

LAUNCH_SCHEMA = "anime-video-studio/p1-v3-launch/v1"
LAUNCH_RESULT_SCHEMA = "anime-video-studio/p1-v3-launch-result/v1"
JOB_SCHEMA = "anime-video-studio/i1-download-job/v1"
OWNER_SCHEMA = "anime-video-studio/p1-v3-process-owner/v1"
STATUS_SCHEMA = "anime-video-studio/i1-download-status/v1"
PROGRESS_SCHEMA = "anime-video-studio/i1-download-progress/v1"
PROOF_SCHEMA = "anime-video-studio/i1-process-proof/v1"
INSPECT_SCHEMA = "anime-video-studio/i1-attempt-inspect/v1"
PACKAGE = Path(__file__).resolve().parent
P1 = PACKAGE.parent / "p1"


def remote_paths(root: str, attempt: str = "0001") -> dict[str, str]:
    base = PurePosixPath(validate_posix_absolute(root, "owned root"))
    if not isinstance(attempt, str) or not attempt.isalnum():
        raise ValueError("attempt id must be alphanumeric")
    attempt_base = base / "attempts" / attempt
    return {
        "root": str(base),
        "attempt": str(attempt_base),
        # A launch attempt is its own frozen-launcher root, so owner/ack cannot
        # conflict with a new attempt. Common read helper stays under owned root.
        "worker": str(attempt_base / "v3_download_worker.py"),
        "worker_impl": str(attempt_base / "i1_download_worker_impl.py"),
        "child_entry": str(attempt_base / "i1_child_entry.py"),
        "launcher": str(attempt_base / "v3_remote_launcher.py"),
        "helper": str(base / "scripts" / "p1" / "v3_remote_fetch_helper.py"),
        "guard": str(base / "scripts" / "p1" / "i1_process_guard.py"),
        "setup_plan": str(attempt_base / "setup_plan.py"),
        "util": str(attempt_base / "i1_util.py"),
        "profile_module": str(attempt_base / "i1_profile.py"),
        "recover": str(attempt_base / "i1_recover_worker.py"),
        "profile": str(attempt_base / "profile.json"),
        "job": str(attempt_base / "job.json"),
        "launch": str(attempt_base / "launch.json"),
        "proof_dir": str(attempt_base / "process-proofs"),
        "log": str(attempt_base / "download.log"),
        "owner": str(attempt_base / "worker-owner.json"),
        "ack": str(attempt_base / "worker-ack.json"),
        "status": str(attempt_base / "status.json"),
        "progress": str(attempt_base / "progress.json"),
        "heartbeat": str(attempt_base / "heartbeat.json"),
    }


def ensure_guard(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    timeout: float = 60.0,
) -> None:
    """Stage the generic process guard before inspecting an older attempt."""

    paths = remote_paths(root)
    i1_remote.mkdir(
        runner,
        identity,
        paths["root"],
        str(PurePosixPath(paths["root"]) / "scripts" / "p1"),
        timeout=timeout,
    )
    i1_remote.stage_file(
        runner,
        identity,
        paths["guard"],
        PACKAGE / "process_guard.py",
        root=paths["root"],
        timeout=timeout,
    )


def proof_path(root: str, attempt: str, pid: int, start_ticks: int) -> str:
    return str(
        PurePosixPath(remote_paths(root, attempt)["proof_dir"])
        / f"proof-{int(pid)}-{int(start_ticks)}.json"
    )


def proof_pointer(root: str, attempt: str, pid: int) -> str:
    return str(
        PurePosixPath(remote_paths(root, attempt)["proof_dir"])
        / f"current-{int(pid)}.json"
    )


def _sha(path: Path) -> str:
    return sha256_file(path)


def _json_bytes(payload: Mapping[str, Any], *, indent: int | None = 2) -> bytes:
    text = json.dumps(
        payload,
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
    )
    return (text + "\n").encode("utf-8")


def _stage_bytes(
    runner: RemoteRunner,
    identity: Any,
    *,
    remote_path: str,
    data: bytes,
    root: str,
    timeout: float,
) -> None:
    handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False)
    local = Path(handle.name)
    try:
        with handle:
            handle.write(data)
        i1_remote.stage_file(
            runner,
            identity,
            remote_path,
            local,
            root=root,
            timeout=timeout,
        )
    finally:
        local.unlink(missing_ok=True)


def _build_stage(
    *,
    root: str,
    workspace: str,
    profile: Profile,
    run_id: str,
    nonce: str,
    deadline_epoch: float,
    target_home: str,
    target_id: str,
    target_lock_path: str,
    attempt: str = "0001",
) -> dict[str, Any]:
    paths = remote_paths(root, attempt)
    files = {
        paths["launcher"]: P1 / "v3_remote_launcher.py",
        paths["helper"]: P1 / "v3_remote_fetch_helper.py",
        paths["guard"]: PACKAGE / "process_guard.py",
        paths["worker"]: PACKAGE / "download_entry_shim.py",
        paths["worker_impl"]: PACKAGE / "worker.py",
        paths["child_entry"]: PACKAGE / "child_entry.py",
        paths["setup_plan"]: P1 / "setup_plan.py",
        paths["util"]: PACKAGE / "util.py",
        paths["profile_module"]: PACKAGE / "profile.py",
        paths["recover"]: PACKAGE / "worker_recover.py",
    }
    profile_bytes = _json_bytes(profile.data)
    staged_files = {remote: _sha(local) for remote, local in files.items()}
    staged_files[paths["profile"]] = hashlib.sha256(profile_bytes).hexdigest()
    job = {
        "schema": JOB_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "workspace": workspace,
        "owned_root": paths["attempt"],
        "attempt_dir": paths["attempt"],
        "comfy_root": str(PurePosixPath(workspace) / "ComfyUI"),
        "profile_path": paths["profile"],
        "profile_sha256": profile.sha256,
        "worker_impl_sha256": _sha(PACKAGE / "worker.py"),
        "models": [
            {
                "id": item["id"],
                "target": item["target"],
                "size_bytes": int(item["size_bytes"]),
                "sha256": item["sha256"],
            }
            for item in profile.models
        ],
        "staged_files": staged_files,
        "target_home": target_home,
        "target_id": target_id,
        "target_lock_path": target_lock_path,
        "deadline_epoch": float(deadline_epoch),
        "log": paths["log"],
        "status_path": paths["status"],
        "progress_path": paths["progress"],
        "heartbeat_path": paths["heartbeat"],
        "required_bytes": profile.disk_required_bytes,
    }
    job_bytes = _json_bytes(job)
    job_sha256 = hashlib.sha256(job_bytes).hexdigest()
    launch = {
        "schema": LAUNCH_SCHEMA,
        "kind": "download_worker",
        "run_id": run_id,
        "nonce": nonce,
        "root": paths["attempt"],
        "worker": paths["worker"],
        "worker_sha256": _sha(PACKAGE / "download_entry_shim.py"),
        "payload": paths["job"],
        "payload_sha256": job_sha256,
        "deadline_epoch": float(deadline_epoch),
        "status": paths["status"],
        "progress": paths["progress"],
        "heartbeat": paths["heartbeat"],
        "log": paths["log"],
        "owner": paths["owner"],
        "ack": paths["ack"],
    }
    launch_bytes = _json_bytes(launch)
    return {
        "paths": paths,
        "files": files,
        "profile_bytes": profile_bytes,
        "job": job,
        "job_bytes": job_bytes,
        "job_sha256": job_sha256,
        "launch": launch,
        "launch_bytes": launch_bytes,
        "launch_sha256": hashlib.sha256(launch_bytes).hexdigest(),
    }


def stage(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    workspace: str,
    profile: Profile,
    run_id: str,
    nonce: str,
    deadline_epoch: float,
    target_home: str,
    target_id: str,
    target_lock_path: str,
    attempt: str = "0001",
    expected_launch_sha256: str | None = None,
) -> dict[str, Any]:
    plan = _build_stage(
        root=root,
        workspace=workspace,
        profile=profile,
        run_id=run_id,
        nonce=nonce,
        deadline_epoch=deadline_epoch,
        target_home=target_home,
        target_id=target_id,
        target_lock_path=target_lock_path,
        attempt=attempt,
    )
    if (
        expected_launch_sha256
        and plan["launch_sha256"] != expected_launch_sha256
    ):
        raise RemoteFailure(
            "download implementation or staged manifest changed since this attempt "
            "was registered; same-attempt resume is refused without overwriting the "
            "old attempt. Prove the old worker dead, then use resume --new-attempt; "
            "verified .part files are reused in the new attempt."
        )
    paths = plan["paths"]
    i1_remote.mkdir(
        runner,
        identity,
        paths["root"],
        str(PurePosixPath(paths["root"]) / "scripts" / "p1"),
        paths["attempt"],
        paths["proof_dir"],
    )
    for remote_path, local_path in plan["files"].items():
        i1_remote.stage_file(
            runner,
            identity,
            remote_path,
            local_path,
            root=paths["root"],
            timeout=90.0,
        )
    _stage_bytes(
        runner,
        identity,
        remote_path=paths["profile"],
        data=plan["profile_bytes"],
        root=paths["root"],
        timeout=30.0,
    )
    _stage_bytes(
        runner,
        identity,
        remote_path=paths["job"],
        data=plan["job_bytes"],
        root=paths["root"],
        timeout=30.0,
    )
    _stage_bytes(
        runner,
        identity,
        remote_path=paths["launch"],
        data=plan["launch_bytes"],
        root=paths["root"],
        timeout=30.0,
    )
    return {
        "paths": paths,
        "job": plan["job"],
        "launch": plan["launch"],
        "job_sha256": plan["job_sha256"],
        "launch_sha256": plan["launch_sha256"],
    }


def _parse_launch(result: Any, *, run_id: str, nonce: str) -> dict[str, Any]:
    if getattr(result, "timed_out", False):
        raise RemoteFailure("download launcher timed out")
    if int(getattr(result, "returncode", 1)) != 0:
        detail = redact(str(getattr(result, "stderr", "") or ""), limit=800)
        raise RemoteFailure(f"download launcher refused (exit {getattr(result, 'returncode', '?')}): {detail}")
    try:
        payload = json.loads(str(result.stdout))
    except json.JSONDecodeError as exc:
        raise RemoteFailure("download launcher result is not JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != LAUNCH_RESULT_SCHEMA
        or payload.get("kind") != "download_worker"
        or payload.get("run_id") != run_id
        or payload.get("nonce") != nonce
        or payload.get("status") not in {"STARTED", "ATTACHED", "ATTACHED_COMPLETE", "COMPLETED"}
    ):
        raise RemoteFailure("download launcher result identity mismatch")
    return payload


def start(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    run_id: str,
    nonce: str,
    attempt: str = "0001",
    resume_dead: bool = False,
    capture_delay: float = 0.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    paths = remote_paths(root, attempt)
    # The remote guard calls the frozen launcher and writes the durable proof
    # before its SSH command returns.  The local client never uploads proof
    # files, so closing the console cannot leave an unresumable live worker.
    args = ["python3", paths["guard"], "launch", paths["root"], paths["launch"]]
    if resume_dead:
        args.append("--resume-dead")
    if capture_delay and capture_delay > 0:
        args.extend(["--capture-delay", str(float(capture_delay))])
    result = runner.run_remote(identity, tuple(args), timeout=timeout)
    parsed = _parse_launch(result, run_id=run_id, nonce=nonce)
    if parsed.get("status") in {"ATTACHED_COMPLETE", "COMPLETED"}:
        return parsed
    pid = parsed.get("pid")
    command_sha256 = parsed.get("command_sha256")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise RemoteFailure("download launcher returned no usable PID")
    if not isinstance(command_sha256, str) or len(command_sha256) != 64:
        raise RemoteFailure("download launcher returned no command identity")
    proof = parsed.get("process_proof")
    if not isinstance(proof, Mapping):
        raise RemoteFailure("download launcher did not publish a remote process proof")
    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("run_id") != run_id
        or proof.get("nonce") != nonce
        or proof.get("command_sha256") != command_sha256
        or proof.get("pid") != pid
    ):
        raise RemoteFailure("download remote process proof identity mismatch")
    parsed["proof"] = dict(proof)
    parsed["proof_path"] = parsed.get("process_proof_path")
    return parsed


def inspect_attempt(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    run_id: str,
    nonce: str,
    attempt: str = "0001",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Distinguish confirmed-missing, unreadable and verified attempt state.

    SSH/remote failures are raised, never converted into ``None``.  The guard
    returns MISSING only when the attempt directory/record is confirmed absent,
    and OK only after owner/ack identity and process proof verification.
    """

    paths = remote_paths(root, attempt)
    result = runner.run_remote(
        identity,
        (
            "python3",
            paths["guard"],
            "inspect",
            paths["root"],
            paths["attempt"],
            run_id,
            nonce,
        ),
        timeout=timeout,
    )
    if getattr(result, "timed_out", False):
        raise RemoteFailure("download attempt inspection timed out")
    if int(getattr(result, "returncode", 1)) != 0:
        detail = redact(str(getattr(result, "stderr", "") or ""), limit=600)
        raise RemoteFailure(f"download attempt inspection failed: {detail}")
    try:
        payload = json.loads(str(result.stdout))
    except json.JSONDecodeError as exc:
        raise RemoteFailure("download attempt inspection is not JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != INSPECT_SCHEMA
        or payload.get("run_id") != run_id
        or payload.get("nonce") != nonce
        or payload.get("state") not in {"MISSING", "UNREADABLE", "OK"}
    ):
        raise RemoteFailure("download attempt inspection identity/schema mismatch")
    return payload


def new_attempt_allowed(state: Mapping[str, Any]) -> bool:
    """Only a proved-dead or safely-completed old attempt may be superseded."""

    record_state = state.get("state")
    if record_state == "MISSING":
        return True
    if record_state != "OK":
        return False
    process = state.get("process")
    if process == "LIVE":
        return False
    if process not in {"DEAD", "ABSENT"}:
        return False
    status = (state.get("status") or {}).get("status")
    if status in {"COMPLETE", "FAILED", "RUNNING"}:
        return True
    return status is None


def recover(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    run_id: str,
    nonce: str,
    target_lock_path: str,
    attempt: str = "0001",
    stale_seconds: float = 30.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    paths = remote_paths(root, attempt)
    result = runner.run_remote(
        identity,
        (
            "python3",
            paths["recover"],
            paths["root"],
            paths["attempt"],
            target_lock_path,
            run_id,
            nonce,
            str(float(stale_seconds)),
        ),
        timeout=timeout,
    )
    if getattr(result, "timed_out", False):
        raise RemoteFailure("worker recovery timed out")
    if int(getattr(result, "returncode", 1)) != 0:
        raise RemoteFailure("worker recovery refused; no launcher spawn")
    try:
        payload = json.loads(str(result.stdout))
    except json.JSONDecodeError as exc:
        raise RemoteFailure("worker recovery result is not JSON") from exc
    if payload.get("status") not in {"RECOVERED_INTERRUPTED", "ALREADY_FAILED"}:
        raise RemoteFailure("worker recovery status is invalid")
    return payload


def read(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    run_id: str,
    nonce: str,
    attempt: str = "0001",
    timeout: float = 15.0,
) -> dict[str, Any]:
    paths = remote_paths(root, attempt)
    owner = i1_remote.read_json(
        runner, identity, root=root, remote_path=paths["owner"], timeout=timeout
    )
    ack = i1_remote.read_json(
        runner, identity, root=root, remote_path=paths["ack"], timeout=timeout
    )
    status = i1_remote.read_json(
        runner, identity, root=root, remote_path=paths["status"], timeout=timeout
    )
    progress = i1_remote.read_json(
        runner, identity, root=root, remote_path=paths["progress"], timeout=timeout
    )
    heartbeat = i1_remote.read_json(
        runner, identity, root=root, remote_path=paths["heartbeat"], timeout=timeout
    )
    for name, value in (("owner", owner), ("ack", ack), ("status", status), ("progress", progress), ("heartbeat", heartbeat)):
        if value.get("run_id") != run_id or value.get("nonce") != nonce:
            raise RemoteFailure(f"download {name} identity mismatch")
    if owner.get("schema") != OWNER_SCHEMA or owner.get("kind") != "download_worker":
        raise RemoteFailure("download owner schema/kind is invalid")
    if status.get("schema") != STATUS_SCHEMA:
        raise RemoteFailure("download status schema is invalid")
    if progress.get("schema") != PROGRESS_SCHEMA:
        raise RemoteFailure("download progress schema is invalid")
    for key in ("files_done", "files_total", "bytes_done", "bytes_total"):
        value = progress.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RemoteFailure(f"download progress {key} is invalid")
    if progress["bytes_done"] > progress["bytes_total"]:
        raise RemoteFailure("download progress exceeds total")
    return {
        "owner": owner,
        "ack": ack,
        "status": status,
        "progress": progress,
        "heartbeat": heartbeat,
    }


def read_optional(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    run_id: str,
    nonce: str,
    attempt: str = "0001",
    timeout: float = 15.0,
) -> dict[str, Any] | None:
    try:
        return read(
            runner,
            identity,
            root=root,
            run_id=run_id,
            nonce=nonce,
            attempt=attempt,
            timeout=timeout,
        )
    except Exception:
        pass
    # An early worker failure may have owner/ack/status but no progress file.
    # Keep that same-run FAILED record visible so resume can use --resume-dead.
    try:
        paths = remote_paths(root, attempt)
        owner = i1_remote.read_json_optional(
            runner, identity, root=root, remote_path=paths["owner"], timeout=timeout
        )
        ack = i1_remote.read_json_optional(
            runner, identity, root=root, remote_path=paths["ack"], timeout=timeout
        )
        status = i1_remote.read_json_optional(
            runner, identity, root=root, remote_path=paths["status"], timeout=timeout
        )
        if not owner or not ack or not status:
            return None
        for value in (owner, ack, status):
            if value.get("run_id") != run_id or value.get("nonce") != nonce:
                return None
        if status.get("status") not in {"RUNNING", "COMPLETE", "FAILED"}:
            return None
        return {
            "owner": owner,
            "ack": ack,
            "status": status,
            "progress": {},
            "heartbeat": {},
        }
    except Exception:
        return None


def heartbeat_age(state: Mapping[str, Any]) -> float | None:
    heartbeat = state.get("heartbeat")
    if not isinstance(heartbeat, Mapping):
        return None
    at = heartbeat.get("heartbeat_at")
    if not isinstance(at, (int, float)) or isinstance(at, bool):
        return None
    return time.time() - float(at)


def resume_or_attach(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    run_id: str,
    nonce: str,
    attempt: str = "0001",
    target_lock_path: str = "",
    stale_seconds: float = 30.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    state = read_optional(
        runner,
        identity,
        root=root,
        run_id=run_id,
        nonce=nonce,
        attempt=attempt,
        timeout=timeout,
    )
    if state is None:
        return start(
            runner,
            identity,
            root=root,
            run_id=run_id,
            nonce=nonce,
            attempt=attempt,
            timeout=timeout,
        )
    status = str(state["status"].get("status"))
    if status == "COMPLETE":
        return {"status": "ATTACHED_COMPLETE", "state": state}
    age = heartbeat_age(state)
    if status == "RUNNING" and age is not None and age <= stale_seconds:
        return {"status": "ATTACHED", "state": state}
    if status == "FAILED" or (status == "RUNNING" and (age is None or age > stale_seconds)):
        if status == "RUNNING":
            recover(
                runner,
                identity,
                root=root,
                run_id=run_id,
                nonce=nonce,
                target_lock_path=target_lock_path,
                attempt=attempt,
                stale_seconds=stale_seconds,
                timeout=timeout,
            )
        result = start(
            runner,
            identity,
            root=root,
            run_id=run_id,
            nonce=nonce,
            attempt=attempt,
            resume_dead=True,
            timeout=timeout,
        )
        return result
    raise RemoteFailure(f"download worker state {status!r} cannot be resumed safely")


__all__ = [
    "JOB_SCHEMA",
    "ensure_guard",
    "heartbeat_age",
    "inspect_attempt",
    "new_attempt_allowed",
    "read",
    "read_optional",
    "recover",
    "remote_paths",
    "resume_or_attach",
    "stage",
    "start",
]
