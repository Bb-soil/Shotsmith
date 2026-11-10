"""Owned ComfyUI start/readback without importing Clore/session policy."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from scripts.p1.remote import RemoteRunner
from scripts.p1.v3_bootstrap import comfy_launch_manifest

from . import remote as i1_remote
from .errors import RemoteFailure
from .profile import Profile
from .util import redact, validate_workspace

P1 = Path(__file__).resolve().parent.parent / "p1"

LAUNCH_SCHEMA = "anime-video-studio/p1-v3-launch/v1"
LAUNCH_RESULT_SCHEMA = "anime-video-studio/p1-v3-launch-result/v1"
OWNER_SCHEMA = "anime-video-studio/p1-v3-process-owner/v1"
PROOF_SCHEMA = "anime-video-studio/i1-process-proof/v1"


def launch_path(root: str) -> str:
    return str(PurePosixPath(root) / "comfy-launch.json")


def owner_path(root: str) -> str:
    return str(PurePosixPath(root) / "comfy-owner.json")


def proof_path(root: str, pid: int, start_ticks: int) -> str:
    return str(
        PurePosixPath(root)
        / "comfy-process-proofs"
        / f"proof-{int(pid)}-{int(start_ticks)}.json"
    )


def proof_pointer(root: str, pid: int) -> str:
    return str(PurePosixPath(root) / "comfy-process-proofs" / f"current-{int(pid)}.json")


def stage_launch(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    workspace: str,
    profile: Profile,
    run_id: str,
    nonce: str,
    remote_port: int | None = None,
) -> dict[str, Any]:
    workspace = validate_workspace(workspace)
    port = int(remote_port or profile.data["runtime"]["comfy_remote_port"])
    manifest = comfy_launch_manifest(
        remote_root=root,
        workspace=workspace,
        run_id=run_id,
        nonce=nonce,
        remote_port=port,
    )
    entry_dir = str(PurePosixPath(root) / ".avs-i1" / "comfy-entry")
    # The frozen launcher accepts a safe staged main.py under the workspace while
    # the real ComfyUI checkout stays untouched.  The shim writes the child
    # birth record before runpy executes the actual ComfyUI/main.py.
    manifest["main"] = str(PurePosixPath(entry_dir) / "main.py")
    # A cold owned root must contain the frozen launcher and read helper before
    # comfy.start; download.stage only stages them inside its own attempt root.
    i1_remote.mkdir(
        runner,
        identity,
        root,
        str(PurePosixPath(root) / "scripts" / "p1"),
        str(PurePosixPath(root) / "comfy-process-proofs"),
        entry_dir,
    )
    i1_remote.stage_file(
        runner,
        identity,
        str(PurePosixPath(root) / "v3_remote_launcher.py"),
        P1 / "v3_remote_launcher.py",
        root=root,
        timeout=90.0,
    )
    i1_remote.stage_file(
        runner,
        identity,
        str(PurePosixPath(root) / "scripts" / "p1" / "v3_remote_fetch_helper.py"),
        P1 / "v3_remote_fetch_helper.py",
        root=root,
        timeout=90.0,
    )
    i1_remote.stage_file(
        runner,
        identity,
        str(PurePosixPath(root) / "i1_comfy_supervisor.py"),
        Path(__file__).resolve().parent / "comfy_supervisor.py",
        root=root,
        timeout=90.0,
    )
    i1_remote.stage_file(
        runner,
        identity,
        str(PurePosixPath(root) / "i1_process_guard.py"),
        Path(__file__).resolve().parent / "process_guard.py",
        root=root,
        timeout=90.0,
    )
    i1_remote.stage_file(
        runner,
        identity,
        str(PurePosixPath(entry_dir) / "main.py"),
        Path(__file__).resolve().parent / "comfy_entry_shim.py",
        root=root,
        timeout=90.0,
    )
    i1_remote.stage_file(
        runner,
        identity,
        str(PurePosixPath(entry_dir) / "i1_child_entry.py"),
        Path(__file__).resolve().parent / "child_entry.py",
        root=root,
        timeout=90.0,
    )
    i1_remote.stage_file(
        runner,
        identity,
        str(PurePosixPath(root) / "i1_util.py"),
        Path(__file__).resolve().parent / "util.py",
        root=root,
        timeout=90.0,
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False, newline="\n"
    )
    local = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        i1_remote.stage_file(
            runner,
            identity,
            launch_path(root),
            local,
            root=root,
            timeout=30.0,
        )
    finally:
        local.unlink(missing_ok=True)
    return manifest


def start(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    run_id: str,
    nonce: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    supervisor = str(PurePosixPath(root) / "i1_comfy_supervisor.py")
    result = runner.run_remote(
        identity,
        ("python3", supervisor, root, launch_path(root)),
        timeout=timeout,
    )
    if getattr(result, "timed_out", False):
        raise RemoteFailure("ComfyUI launcher timed out")
    if int(getattr(result, "returncode", 1)) != 0:
        detail = redact(str(getattr(result, "stderr", "") or ""), limit=800)
        raise RemoteFailure(f"ComfyUI launcher refused: {detail}")
    try:
        payload = json.loads(str(result.stdout))
    except json.JSONDecodeError as exc:
        raise RemoteFailure("ComfyUI launcher result is not JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != LAUNCH_RESULT_SCHEMA
        or payload.get("kind") != "comfy"
        or payload.get("run_id") != run_id
        or payload.get("nonce") != nonce
        or payload.get("status") not in {"STARTED", "ATTACHED"}
    ):
        raise RemoteFailure("ComfyUI launch identity mismatch")
    owner = i1_remote.read_json(
        runner,
        identity,
        root=root,
        remote_path=owner_path(root),
        timeout=timeout,
    )
    if (
        owner.get("schema") != OWNER_SCHEMA
        or owner.get("kind") != "comfy"
        or owner.get("run_id") != run_id
        or owner.get("nonce") != nonce
    ):
        raise RemoteFailure("ComfyUI owner identity mismatch")
    pid = payload.get("pid")
    command_sha256 = payload.get("command_sha256")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise RemoteFailure("ComfyUI launcher returned no usable PID")
    if not isinstance(command_sha256, str) or len(command_sha256) != 64:
        raise RemoteFailure("ComfyUI launcher returned no command identity")
    # The staged remote guard wrote/recovered the proof before the supervisor
    # returned.  The local client must never be the only place that uploads it:
    # a console/SSH loss after STARTED must still leave a resumable incarnation.
    proof = payload.get("process_proof")
    if not isinstance(proof, Mapping):
        raise RemoteFailure("ComfyUI launcher did not publish a remote process proof")
    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("run_id") != run_id
        or proof.get("nonce") != nonce
        or proof.get("command_sha256") != command_sha256
        or proof.get("pid") != pid
    ):
        raise RemoteFailure("ComfyUI remote process proof identity mismatch")
    return {"launch": payload, "owner": owner, "proof": dict(proof)}


def read_owner(
    runner: RemoteRunner,
    identity: Any,
    *,
    root: str,
    run_id: str,
    nonce: str,
    timeout: float = 15.0,
) -> dict[str, Any] | None:
    try:
        owner = i1_remote.read_json(
            runner,
            identity,
            root=root,
            remote_path=owner_path(root),
            timeout=timeout,
        )
    except RemoteFailure:
        return None
    if (
        owner.get("schema") != OWNER_SCHEMA
        or owner.get("kind") != "comfy"
        or owner.get("run_id") != run_id
        or owner.get("nonce") != nonce
    ):
        return None
    return owner


__all__ = ["launch_path", "owner_path", "read_owner", "stage_launch", "start"]
