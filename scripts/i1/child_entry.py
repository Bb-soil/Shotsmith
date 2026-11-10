"""Child-side exact birth record for I1 processes.

The staged child entry shim calls :func:`write_birth` as its very first action,
before ``runpy`` executes the real worker/ComfyUI code.  The record is written
with ``fsync`` + atomic replace into the owned proof directory.  It is the only
accepted baseline for a later attach: the guard never invents a baseline from a
current PID, argv or a time window.

This file is staged next to the shims and uses the standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

BIRTH_SCHEMA = "anime-video-studio/i1-child-birth/v1"
LAUNCH_SCHEMA = "anime-video-studio/p1-v3-launch/v1"


class BirthError(Exception):
    pass


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(dict(manifest)))


def _cmdline_argv(raw: bytes) -> list[str]:
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    return [part.decode("utf-8", "surrogateescape") for part in parts]


def _proc_facts(pid: int) -> dict[str, Any]:
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    end = raw.rfind(b")")
    if end < 0:
        raise BirthError("process stat is malformed")
    stat = raw[end + 2 :].split()
    if len(stat) < 20:
        raise BirthError("process stat is truncated")
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    argv = _cmdline_argv(cmd)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    return {
        "pid": int(pid),
        "start_ticks": int(stat[19]),
        "exe": os.readlink(f"/proc/{pid}/exe"),
        "cmdline_sha256": sha256_bytes(cmd),
        "argv_sha256": sha256_bytes(canonical_bytes(argv)),
        "boot_id": boot_id,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BirthError("launch manifest is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema") != LAUNCH_SCHEMA:
        raise BirthError("launch manifest schema is invalid")
    return value


def proof_dir_for_manifest(manifest: Mapping[str, Any]) -> Path:
    root = Path(str(manifest.get("root", "")))
    if manifest.get("kind") == "comfy":
        return root / "comfy-process-proofs"
    if manifest.get("kind") == "download_worker":
        return root / "process-proofs"
    raise BirthError("launch kind is unsupported")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if target.is_symlink():
        raise BirthError("birth record must not be a symlink")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(target.parent), prefix=target.name + ".", suffix=".tmp", delete=False
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    finally:
        temp.unlink(missing_ok=True)


def write_birth(*, kind: str, manifest_path: Path | str) -> dict[str, Any]:
    """Write the exact child birth record before any real work starts."""

    manifest = _read_manifest(Path(manifest_path))
    if manifest.get("kind") != kind:
        raise BirthError("child kind does not match the manifest")
    if kind not in {"comfy", "download_worker"}:
        raise BirthError("child kind is unsupported")
    expected_root = Path(str(manifest.get("root", "")))
    if not expected_root.is_absolute():
        raise BirthError("manifest root is not absolute")
    facts = _proc_facts(os.getpid())
    record = {
        "schema": BIRTH_SCHEMA,
        "kind": kind,
        "run_id": manifest.get("run_id"),
        "nonce": manifest.get("nonce"),
        "manifest_sha256": manifest_hash(manifest),
        "created_at": time.time(),
        **facts,
    }
    directory = proof_dir_for_manifest(manifest)
    path = directory / f"birth-{facts['pid']}-{facts['start_ticks']}.json"
    _atomic_json(path, record)
    return manifest


__all__ = ["BIRTH_SCHEMA", "BirthError", "proof_dir_for_manifest", "write_birth"]
