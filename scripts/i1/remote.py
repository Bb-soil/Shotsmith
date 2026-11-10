"""Thin, bounded remote operations used by the I1 orchestrator."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scripts.p1.remote import RemoteCommandError, RemoteIdentity, RemoteRunner

from .errors import RemoteFailure
from .util import redact, validate_name, validate_posix_absolute, validate_workspace

HELPER_NAME = "v3_remote_fetch_helper.py"


def make_identity(record: Mapping[str, Any]) -> RemoteIdentity:
    identity = record["identity"]
    auth = record["auth"]
    key = Path(auth["path"]) if auth.get("kind") == "key" else None
    if not isinstance(record.get("global_known_hosts"), str):
        raise ValueError("connection profile lacks an owned global known-hosts file")
    return RemoteIdentity(
        host=str(identity["host"]),
        user=str(identity["user"]),
        port=int(identity["port"]),
        identity_file=key,
        known_hosts=Path(record["known_hosts"]),
        standard_known_hosts=True,
        global_known_hosts=Path(record["global_known_hosts"]),
        host_key_algorithms=record.get("host_key_type"),
    )


def owned_root(workspace: str) -> str:
    return str(PurePosixPath(validate_workspace(workspace)) / ".avs-i1")


def comfy_dir(workspace: str) -> str:
    return str(PurePosixPath(validate_workspace(workspace)) / "ComfyUI")


def venv_dir(workspace: str) -> str:
    return str(PurePosixPath(validate_workspace(workspace)) / "p1-venv")


def venv_python(workspace: str) -> str:
    return str(PurePosixPath(venv_dir(workspace)) / "bin" / "python")


def helper_path(root: str) -> str:
    return str(PurePosixPath(root) / "scripts" / "p1" / HELPER_NAME)


def _raise_result(what: str, result: Any) -> None:
    if getattr(result, "timed_out", False):
        raise RemoteFailure(f"{what} timed out")
    if int(getattr(result, "returncode", 1)) != 0:
        detail = redact(str(getattr(result, "stderr", "") or ""), limit=600)
        raise RemoteFailure(f"{what} failed (exit {getattr(result, 'returncode', '?')}): {detail}")


def run_checked(
    runner: RemoteRunner,
    identity: RemoteIdentity,
    argv: Sequence[str],
    *,
    timeout: float = 30.0,
    max_output_bytes: int = 64 * 1024,
) -> Any:
    result = runner.run_remote(
        identity,
        tuple(argv),
        timeout=timeout,
    )
    _raise_result(str(argv[0]) if argv else "remote command", result)
    return result


def run_text(
    runner: RemoteRunner,
    identity: RemoteIdentity,
    argv: Sequence[str],
    *,
    timeout: float = 30.0,
    max_output_bytes: int = 64 * 1024,
) -> str:
    result = runner.run_remote(
        identity,
        tuple(argv),
        timeout=timeout,
    )
    _raise_result(str(argv[0]) if argv else "remote command", result)
    return str(result.stdout)


def probe_result(
    runner: RemoteRunner,
    identity: RemoteIdentity,
    argv: Sequence[str],
    *,
    timeout: float = 30.0,
) -> Any:
    """Return the raw result for read-only inspection without raising."""

    return runner.run_remote(identity, tuple(argv), timeout=timeout)


def stage_file(
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_path: str,
    local_path: Path,
    *,
    root: str,
    timeout: float = 60.0,
    allow_same_hash: bool = True,
) -> None:
    runner.put_file(
        identity,
        remote_path,
        Path(local_path),
        root=root,
        allow_same_hash=allow_same_hash,
        timeout=timeout,
    )


def mkdir(
    runner: RemoteRunner,
    identity: RemoteIdentity,
    *paths: str,
    timeout: float = 30.0,
) -> None:
    for path in paths:
        validate_posix_absolute(path, "remote directory")
    result = runner.run_remote(identity, ("mkdir", "-p", *paths), timeout=timeout)
    _raise_result("remote mkdir", result)


def read_json(
    runner: RemoteRunner,
    identity: RemoteIdentity,
    *,
    root: str,
    remote_path: str,
    timeout: float = 15.0,
    helper_root: str | None = None,
) -> dict[str, Any]:
    try:
        return runner.read_remote_exact_json(
            identity,
            helper_path=helper_path(helper_root or root),
            remote_path=remote_path,
            root=root,
            max_bytes=256 * 1024,
            timeout=timeout,
        )
    except (RemoteCommandError, OSError, ValueError) as exc:
        raise RemoteFailure(f"remote JSON read failed: {remote_path}") from exc


_PROC_CODE = (
    "import hashlib,json,os,sys;"
    "pid=int(sys.argv[1]);"
    "raw=open(f'/proc/{pid}/stat','rb').read();"
    "end=raw.rfind(b')');"
    "stat=raw[end+2:].split();"
    "start_ticks=int(stat[19]);"
    "exe=os.readlink(f'/proc/{pid}/exe');"
    "cmd=open(f'/proc/{pid}/cmdline','rb').read();"
    "boot=open('/proc/sys/kernel/random/boot_id','r',encoding='ascii').read().strip();"
    "print(json.dumps({'pid':pid,'start_ticks':start_ticks,'exe':exe,"
    "'cmdline_sha256':hashlib.sha256(cmd).hexdigest(),'boot_id':boot},sort_keys=True))"
)


def capture_process(pid: int, *, runner: RemoteRunner, identity: RemoteIdentity, timeout: float = 15.0) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RemoteFailure("process PID is invalid")
    result = runner.run_remote(
        identity, ("python3", "-c", _PROC_CODE, str(pid)), timeout=timeout
    )
    if getattr(result, "timed_out", False) or int(getattr(result, "returncode", 1)) != 0:
        raise RemoteFailure("cannot read process ownership facts")
    try:
        import json

        value = json.loads(str(result.stdout))
    except Exception as exc:
        raise RemoteFailure("process ownership facts are not JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("pid") != pid
        or not isinstance(value.get("start_ticks"), int)
        or not isinstance(value.get("exe"), str)
        or not isinstance(value.get("cmdline_sha256"), str)
        or not isinstance(value.get("boot_id"), str)
    ):
        raise RemoteFailure("process ownership facts are invalid")
    return value


def make_process_proof(
    *,
    pid: int,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    run_id: str,
    nonce: str,
    command_sha256: str,
    timeout: float = 15.0,
) -> dict[str, Any]:
    facts = capture_process(pid, runner=runner, identity=identity, timeout=timeout)
    return {
        "schema": "anime-video-studio/i1-process-proof/v1",
        "run_id": run_id,
        "nonce": nonce,
        "command_sha256": command_sha256,
        **facts,
    }


def verify_process_proof(
    proof: Mapping[str, Any],
    *,
    runner: RemoteRunner,
    identity: RemoteIdentity,
    run_id: str,
    nonce: str,
    command_sha256: str,
    timeout: float = 15.0,
) -> None:
    if (
        not isinstance(proof, Mapping)
        or proof.get("schema") != "anime-video-studio/i1-process-proof/v1"
        or proof.get("run_id") != run_id
        or proof.get("nonce") != nonce
        or proof.get("command_sha256") != command_sha256
    ):
        raise RemoteFailure("process proof identity mismatch")
    pid = proof.get("pid")
    current = capture_process(
        int(pid), runner=runner, identity=identity, timeout=timeout
    )
    for key in ("pid", "start_ticks", "exe", "cmdline_sha256", "boot_id"):
        if current.get(key) != proof.get(key):
            raise RemoteFailure("process proof changed; PID reuse/foreign process refused")


def read_json_optional(
    runner: RemoteRunner,
    identity: RemoteIdentity,
    *,
    root: str,
    remote_path: str,
    timeout: float = 15.0,
) -> dict[str, Any] | None:
    try:
        return read_json(
            runner,
            identity,
            root=root,
            remote_path=remote_path,
            timeout=timeout,
        )
    except RemoteFailure:
        return None


def tail_log(
    runner: RemoteRunner,
    identity: RemoteIdentity,
    remote_path: str,
    *,
    lines: int = 60,
    timeout: float = 15.0,
) -> str:
    validate_posix_absolute(remote_path, "log path")
    result = runner.run_remote(
        identity,
        ("tail", "-n", str(max(1, min(int(lines), 200))), "--", remote_path),
        timeout=timeout,
    )
    if getattr(result, "timed_out", False) or int(getattr(result, "returncode", 1)) != 0:
        return ""
    return redact(str(result.stdout or ""), limit=8000)


__all__ = [
    "HELPER_NAME",
    "comfy_dir",
    "helper_path",
    "make_identity",
    "mkdir",
    "owned_root",
    "probe_result",
    "read_json",
    "read_json_optional",
    "run_checked",
    "run_text",
    "stage_file",
    "tail_log",
    "venv_dir",
    "venv_python",
]
