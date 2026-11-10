"""Stable remote spawn/attach guard for I1 processes.

The frozen launcher starts the child, while the staged child entry shim writes an
exact birth record (PID/start_ticks/exe/cmdline/boot_id plus the launch manifest
hash) before executing any real work.  This guard only accepts that child-written
baseline; it never invents one from a current PID, argv or a time window.

The guard holds one OS lock, calls the frozen launcher unchanged, and turns the
child birth record into a durable per-incarnation proof before returning.  A
missing proof pointer is recoverable only from the already-written child birth
record.  If a live process has no birth baseline, every proof creation refuses.

Only standard-library imports are used because this file executes remotely.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

PROOF_SCHEMA = "anime-video-studio/i1-process-proof/v1"
BIRTH_SCHEMA = "anime-video-studio/i1-child-birth/v1"
INSPECT_SCHEMA = "anime-video-studio/i1-attempt-inspect/v1"
OWNER_SCHEMA = "anime-video-studio/p1-v3-process-owner/v1"
LAUNCH_SCHEMA = "anime-video-studio/p1-v3-launch/v1"
STATUS_SCHEMA = "anime-video-studio/i1-download-status/v1"
PROGRESS_SCHEMA = "anime-video-studio/i1-download-progress/v1"
HEARTBEAT_SCHEMA = "anime-video-studio/i1-download-heartbeat/v1"
GUARD_LOCK_NAME = ".i1-process-guard.lock"
BIRTH_WAIT_SECONDS = 10.0


class GuardError(Exception):
    pass


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(dict(manifest)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardError("record is unreadable") from exc
    if not isinstance(value, dict):
        raise GuardError("record must be an object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise GuardError("proof path must not be a symlink")
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


@contextmanager
def _guard_lock(root: Path) -> Iterator[None]:
    path = Path(root) / GUARD_LOCK_NAME
    if path.is_symlink():
        raise GuardError("guard lock must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
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
        handle.close()


def _cmdline_argv(raw: bytes) -> list[str]:
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    return [part.decode("utf-8", "surrogateescape") for part in parts]


def _proc_facts(pid: int) -> dict[str, Any]:
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    end = raw.rfind(b")")
    if end < 0:
        raise GuardError("process stat is malformed")
    stat = raw[end + 2 :].split()
    if len(stat) < 20:
        raise GuardError("process stat is truncated")
    state = stat[0].decode("ascii", "replace")
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    argv = _cmdline_argv(cmd)
    return {
        "pid": int(pid),
        "start_ticks": int(stat[19]),
        "state": state,
        "exe": os.readlink(f"/proc/{pid}/exe"),
        "cmdline_sha256": sha256_bytes(cmd),
        "argv_sha256": sha256_bytes(canonical_bytes(argv)),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
    }


def _confirmed_absent(pid: int) -> bool:
    """True only for a gone/exited/zombie PID, never for access errors."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        try:
            raw = Path(f"/proc/{pid}/stat").read_bytes()
            end = raw.rfind(b")")
            if end < 0:
                return False
            return raw[end + 2 :].split()[0] == b"Z"
        except (OSError, IndexError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False
    return False


def proof_dir_for_manifest(manifest: Mapping[str, Any]) -> Path:
    root = Path(str(manifest.get("root", "")))
    if manifest.get("kind") == "comfy":
        return root / "comfy-process-proofs"
    if manifest.get("kind") == "download_worker":
        return root / "process-proofs"
    raise GuardError("launch kind is not supported by the process guard")


def proof_path(proof_dir: Path, pid: int, start_ticks: int) -> Path:
    return Path(proof_dir) / f"proof-{int(pid)}-{int(start_ticks)}.json"


def proof_pointer(proof_dir: Path, pid: int) -> Path:
    return Path(proof_dir) / f"current-{int(pid)}.json"


def expected_owner(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": OWNER_SCHEMA,
        "kind": manifest.get("kind"),
        "run_id": manifest.get("run_id"),
        "nonce": manifest.get("nonce"),
        "command_sha256": manifest_hash(manifest),
    }


def expected_argv(manifest: Mapping[str, Any]) -> list[str]:
    kind = manifest.get("kind")
    if kind == "comfy":
        port = manifest.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
            raise GuardError("comfy port is invalid")
        values = [manifest.get("python"), manifest.get("main"), "--listen", "127.0.0.1", "--port", str(port)]
        if not all(isinstance(item, str) and item for item in values):
            raise GuardError("comfy argv fields are invalid")
        return [str(item) for item in values]
    if kind == "download_worker":
        worker = manifest.get("worker")
        payload = manifest.get("payload")
        if not isinstance(worker, str) or not isinstance(payload, str):
            raise GuardError("download argv fields are invalid")
        return [sys.executable, worker, payload]
    raise GuardError("launch kind is not supported")


def expected_argv_sha256(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(expected_argv(manifest)))


def _validate_record(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    pid: int | None = None,
) -> None:
    if not isinstance(record, Mapping):
        raise GuardError("owner record must be an object")
    for key, value in expected.items():
        if record.get(key) != value:
            raise GuardError("owner/ack identity mismatch")
    record_pid = record.get("pid")
    if isinstance(record_pid, bool) or not isinstance(record_pid, int) or record_pid <= 0:
        raise GuardError("owner/ack PID is invalid")
    if pid is not None and record_pid != pid:
        raise GuardError("owner/ack PID mismatch")


def _validate_owner_and_ack(
    owner: Mapping[str, Any],
    ack: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_argv_hash: str,
) -> None:
    _validate_record(owner, expected)
    if owner.get("argv_sha256") != expected_argv_hash:
        raise GuardError("owner argv does not match the current manifest")
    _validate_record(ack, expected, pid=int(owner["pid"]))


def _read_manifest_json(path: Path, root: Path) -> dict[str, Any]:
    launch_path = Path(path)
    root_path = Path(root)
    if not launch_path.is_absolute() or not root_path.is_absolute():
        raise GuardError("launch paths must be absolute")
    if launch_path.is_symlink() or not launch_path.is_file():
        raise GuardError("launch manifest must be a regular file")
    try:
        if os.path.commonpath([os.path.realpath(root_path), os.path.abspath(launch_path)]) != os.path.realpath(root_path):
            raise GuardError("launch manifest escapes the owned root")
    except ValueError as exc:
        raise GuardError("launch manifest escapes the owned root") from exc
    try:
        manifest = json.loads(launch_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardError("launch manifest is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != LAUNCH_SCHEMA:
        raise GuardError("launch manifest schema is invalid")
    if manifest.get("kind") not in {"comfy", "download_worker"}:
        raise GuardError("launch kind is not supported")
    for key in ("run_id", "nonce"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise GuardError(f"launch {key} is invalid")
    if not isinstance(manifest.get("root"), str) or not os.path.isabs(manifest["root"]):
        raise GuardError("launch root is invalid")
    for key in ("owner", "ack", "log"):
        value = manifest.get(key)
        if not isinstance(value, str) or not os.path.isabs(value):
            raise GuardError(f"launch {key} is invalid")
    return manifest


def _under_root(root: Path, value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise GuardError(f"{name} must be an absolute path")
    if os.path.normpath(value) != value:
        raise GuardError(f"{name} must be normalized")
    try:
        if os.path.commonpath([os.path.realpath(root), os.path.abspath(value)]) != os.path.realpath(root):
            raise GuardError(f"{name} escapes the owned root")
    except ValueError as exc:
        raise GuardError(f"{name} escapes the owned root") from exc
    return value


def validate_launch_manifest(manifest: Mapping[str, Any], root: Path | str) -> None:
    """Validate the immutable launch/job/payload identity used for a birth."""

    root_path = Path(root)
    manifest_root = Path(str(manifest.get("root")))
    # Comfy uses the workspace itself; a download attempt is its own
    # frozen-launcher root nested under the owned I1 root.
    _under_root(root_path, str(manifest_root), "manifest root")
    kind = manifest.get("kind")
    for key in ("owner", "ack", "log"):
        _under_root(manifest_root, manifest.get(key), key)
    if kind == "comfy":
        for key in ("python", "comfy_dir", "main"):
            _under_root(manifest_root, manifest.get(key), key)
        if os.path.basename(str(manifest.get("main"))) != "main.py":
            raise GuardError("comfy main filename is invalid")
        if os.path.basename(str(manifest.get("comfy_dir"))) != "ComfyUI":
            raise GuardError("comfy directory is unexpected")
        expected_argv(manifest)
        return
    if kind != "download_worker":
        raise GuardError("launch kind is not supported")
    for key in ("worker", "payload", "status", "progress", "heartbeat"):
        _under_root(manifest_root, manifest.get(key), key)
    worker = Path(str(manifest.get("worker")))
    payload = Path(str(manifest.get("payload")))
    if worker.name != "v3_download_worker.py":
        raise GuardError("download worker filename is invalid")
    if payload.name != "job.json":
        raise GuardError("download payload filename is invalid")
    if os.path.dirname(str(manifest.get("owner"))) != str(manifest_root) or os.path.dirname(str(manifest.get("ack"))) != str(manifest_root):
        raise GuardError("download owner/ack must be directly in the attempt root")
    for path, key in ((worker, "worker_sha256"), (payload, "payload_sha256")):
        if path.is_symlink() or not path.is_file():
            raise GuardError(f"download {key} file is missing")
        expected = manifest.get(key)
        if not isinstance(expected, str) or len(expected) != 64:
            raise GuardError(f"download {key} is invalid")
        if _sha256_file(path) != expected:
            raise GuardError(f"download {key} does not match the manifest")
    expected_argv(manifest)


def read_manifest(path: Path | str, root: Path | str) -> dict[str, Any]:
    manifest = _read_manifest_json(Path(path), Path(root))
    validate_launch_manifest(manifest, root)
    return manifest


def _birth_paths(proof_dir: Path, pid: int) -> list[Path]:
    directory = Path(proof_dir)
    if not directory.is_dir() or directory.is_symlink():
        return []
    return sorted(directory.glob(f"birth-{int(pid)}-*.json"))


def _validate_birth(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_argv_hash: str,
    facts: Mapping[str, Any] | None,
) -> None:
    if record.get("schema") != BIRTH_SCHEMA:
        raise GuardError("birth schema is invalid")
    for key in ("kind", "run_id", "nonce"):
        if record.get(key) != expected.get(key):
            raise GuardError("birth identity mismatch")
    if record.get("manifest_sha256") != expected.get("command_sha256"):
        raise GuardError("birth manifest hash mismatch")
    if record.get("argv_sha256") != expected_argv_hash:
        raise GuardError("birth argv does not match the current manifest")
    pid = record.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise GuardError("birth PID is invalid")
    if facts is not None:
        for key in ("pid", "start_ticks", "exe", "cmdline_sha256", "argv_sha256", "boot_id"):
            if record.get(key) != facts.get(key):
                raise GuardError("birth does not match the live process")


def _find_birth(
    proof_dir: Path,
    pid: int,
    expected: Mapping[str, Any],
    expected_argv_hash: str,
    facts: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    matches = []
    for path in _birth_paths(proof_dir, pid):
        if path.is_symlink():
            continue
        try:
            record = _read_json(path)
            _validate_birth(record, expected, expected_argv_hash, facts)
            if facts is not None:
                expected_name = f"birth-{int(facts['pid'])}-{int(facts['start_ticks'])}.json"
                if path.name != expected_name:
                    continue
            matches.append(record)
        except GuardError:
            continue
    if not matches:
        return None
    if len(matches) > 1:
        raise GuardError("multiple child birth baselines exist for this PID")
    return matches[0]


def _proof_payload(birth: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PROOF_SCHEMA,
        "run_id": expected.get("run_id"),
        "nonce": expected.get("nonce"),
        "command_sha256": expected.get("command_sha256"),
        "pid": int(birth["pid"]),
        "start_ticks": int(birth["start_ticks"]),
        "exe": birth["exe"],
        "cmdline_sha256": birth["cmdline_sha256"],
        "argv_sha256": birth["argv_sha256"],
        "boot_id": birth["boot_id"],
        "birth_sha256": sha256_bytes(canonical_bytes(dict(birth))),
    }


def _write_proof(proof_dir: Path, proof: Mapping[str, Any]) -> tuple[Path, Path]:
    directory = Path(proof_dir)
    if directory.is_symlink():
        raise GuardError("proof directory must not be a symlink")
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    pid = int(proof["pid"])
    ticks = int(proof["start_ticks"])
    per = proof_path(directory, pid, ticks)
    pointer = proof_pointer(directory, pid)
    _atomic_json(per, proof)
    _atomic_json(pointer, proof)
    return per, pointer


def _proof_matches(
    proof: Mapping[str, Any],
    expected: Mapping[str, Any],
    birth: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> bool:
    if not isinstance(proof, Mapping) or proof.get("schema") != PROOF_SCHEMA:
        return False
    for key in ("run_id", "nonce", "command_sha256"):
        if proof.get(key) != expected.get(key):
            return False
    for key in ("pid", "start_ticks", "exe", "cmdline_sha256", "argv_sha256", "boot_id"):
        if proof.get(key) != facts.get(key):
            return False
    if proof.get("birth_sha256") != sha256_bytes(canonical_bytes(dict(birth))):
        return False
    return True


def _process_state(
    owner: Mapping[str, Any],
    proof_dir: Path,
    *,
    expected: Mapping[str, Any],
    expected_argv_hash: str,
    write_recovery: bool = True,
    fresh: bool = False,
) -> dict[str, Any]:
    """Classify using an exact child birth baseline; never a PID/argv heuristic."""

    pid = owner.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return {"process": "MISMATCH", "reason": "owner PID is invalid"}
    pointer = proof_pointer(proof_dir, pid)
    existing = None
    if not fresh and pointer.is_file() and not pointer.is_symlink():
        try:
            existing = _read_json(pointer)
        except GuardError:
            existing = None
    try:
        facts = _proc_facts(pid)
    except (OSError, GuardError) as exc:
        if not _confirmed_absent(pid):
            return {"process": "UNKNOWN", "reason": f"process facts unavailable: {type(exc).__name__}"}
        birth = _find_birth(proof_dir, pid, expected, expected_argv_hash, None)
        if birth is None:
            return {"process": "UNKNOWN", "reason": "child birth baseline missing for dead owner"}
        return {"process": "DEAD", "reason": "recorded PID is absent", "birth": birth, "proof": existing}
    if facts.get("state") == "Z":
        birth = _find_birth(proof_dir, pid, expected, expected_argv_hash, facts)
        if birth is None:
            return {"process": "UNKNOWN", "reason": "child birth baseline missing for zombie"}
        return {"process": "DEAD", "reason": "recorded PID is a zombie", "birth": birth, "proof": existing}
    birth = _find_birth(proof_dir, pid, expected, expected_argv_hash, facts)
    if birth is None:
        return {"process": "MISSING_BIRTH", "reason": "child birth baseline is missing"}
    if existing is not None:
        if not _proof_matches(existing, expected, birth, facts):
            return {"process": "MISMATCH", "reason": "durable proof does not match the birth/live process"}
        return {"process": "LIVE", "proof": existing, "birth": birth, "recovered": False}
    proof = _proof_payload(birth, expected)
    if not write_recovery:
        return {"process": "LIVE", "proof": proof, "birth": birth, "recovered": True}
    per, point = _write_proof(proof_dir, proof)
    return {
        "process": "LIVE",
        "proof": proof,
        "birth": birth,
        "recovered": True,
        "proof_path": str(per),
        "proof_pointer": str(point),
    }


def assess_owner(
    *,
    root: Path | str,
    manifest: Mapping[str, Any],
    owner_path: Path | str,
    ack_path: Path | str,
    proof_dir: Path | str,
) -> dict[str, Any]:
    """Decide attach/archive/spawn before any mutation."""

    owner_file = Path(owner_path)
    ack_file = Path(ack_path)
    if owner_file.is_symlink() or ack_file.is_symlink():
        return {"action": "REFUSE", "reason": "owner/ack must not be symlinks"}
    if not owner_file.exists():
        if ack_file.exists():
            return {"action": "REFUSE", "reason": "ack exists without its owner"}
        return {"action": "SPAWN", "reason": "no owned process record"}
    try:
        owner = _read_json(owner_file)
    except GuardError as exc:
        return {"action": "REFUSE", "reason": f"owner is unreadable: {exc}"}
    expected = expected_owner(manifest)
    try:
        argv_hash = expected_argv_sha256(manifest)
    except GuardError as exc:
        return {"action": "REFUSE", "reason": f"manifest argv is invalid: {exc}"}
    if not ack_file.is_file():
        return {"action": "REFUSE", "reason": "ack is missing for the owned process record"}
    try:
        ack = _read_json(ack_file)
        _validate_owner_and_ack(owner, ack, expected, argv_hash)
    except GuardError as exc:
        return {"action": "REFUSE", "reason": f"owner/ack does not belong to this installation: {exc}"}
    state = _process_state(
        owner,
        Path(proof_dir),
        expected=expected,
        expected_argv_hash=argv_hash,
    )
    if state["process"] == "LIVE":
        return {"action": "ATTACH", "owner": owner, "ack": ack, **state}
    if state["process"] == "DEAD":
        return {"action": "ARCHIVE_DEAD", "owner": owner, "ack": ack, **state}
    return {"action": "REFUSE", "owner": owner, "ack": ack, **state}


def _wait_for_process_state(
    owner: Mapping[str, Any],
    proof_dir: Path,
    *,
    expected: Mapping[str, Any],
    expected_argv_hash: str,
    fresh: bool,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        state = _process_state(
            owner,
            proof_dir,
            expected=expected,
            expected_argv_hash=expected_argv_hash,
            fresh=fresh,
        )
        if state["process"] != "MISSING_BIRTH":
            return state
        if time.monotonic() >= deadline:
            return state
        time.sleep(0.05)


def _terminal_status(manifest: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(str(manifest.get("status", "")))
    if path.is_symlink() or not path.is_file():
        return None
    status = _read_json(path)
    if status.get("schema") != STATUS_SCHEMA:
        raise GuardError("terminal status schema is invalid")
    if status.get("run_id") != expected.get("run_id") or status.get("nonce") != expected.get("nonce"):
        raise GuardError("terminal status identity mismatch")
    if status.get("status") not in {"COMPLETE", "FAILED"}:
        return None
    return status


def launch_with_proof(
    root: Path | str,
    launch_path: Path | str,
    *,
    resume_dead: bool = False,
    capture_delay: float = 0.0,
    birth_timeout: float = BIRTH_WAIT_SECONDS,
) -> dict[str, Any]:
    root_path = Path(root)
    with _guard_lock(root_path):
        manifest = read_manifest(launch_path, root_path)
        expected = expected_owner(manifest)
        argv_hash = expected_argv_sha256(manifest)
        manifest_root = str(manifest["root"])
        if manifest_root not in sys.path:
            sys.path.insert(0, manifest_root)
        import v3_remote_launcher  # type: ignore

        result = v3_remote_launcher.launch(str(launch_path), resume_dead=resume_dead)
        if not isinstance(result, dict):
            raise GuardError("launcher result is not an object")
        if result.get("status") == "ATTACHED_COMPLETE":
            return {**result, "terminal_status": "COMPLETE", "manifest_sha256": expected["command_sha256"]}
        if result.get("status") not in {"STARTED", "ATTACHED"}:
            return result
        if result.get("command_sha256") != expected["command_sha256"]:
            raise GuardError("launcher command hash does not match the manifest")
        owner = _read_json(Path(str(manifest["owner"])))
        ack = _read_json(Path(str(manifest["ack"])))
        try:
            pid = int(result["pid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GuardError("launcher result lacks a usable PID") from exc
        _validate_owner_and_ack(owner, ack, expected, argv_hash)
        if capture_delay > 0:
            time.sleep(float(capture_delay))
        state = _wait_for_process_state(
            owner,
            proof_dir_for_manifest(manifest),
            expected=expected,
            expected_argv_hash=argv_hash,
            fresh=result.get("status") == "STARTED",
            timeout=birth_timeout,
        )
        process = state.get("process")
        if process == "LIVE":
            proof = state["proof"]
            path = proof_path(proof_dir_for_manifest(manifest), int(proof["pid"]), int(proof["start_ticks"]))
            pointer = proof_pointer(proof_dir_for_manifest(manifest), int(proof["pid"]))
            return {
                **result,
                "process_proof": proof,
                "process_proof_path": str(path),
                "process_proof_pointer": str(pointer),
                "process_proof_recovered": bool(state.get("recovered")),
                "birth": state.get("birth"),
            }
        if process == "DEAD" and manifest.get("kind") == "download_worker":
            status = _terminal_status(manifest, expected)
            if status is None:
                raise GuardError("download worker exited without a terminal status")
            if status.get("status") == "FAILED":
                raise GuardError(f"download worker failed: {status.get('error', 'FAILED')}")
            return {
                **result,
                "status": "COMPLETED",
                "terminal_status": "COMPLETE",
                "terminal": status,
                "birth": state.get("birth"),
                "manifest_sha256": expected["command_sha256"],
            }
        raise GuardError(f"launcher result is not a verified incarnation: {state.get('reason') or process}")


def _read_progress(attempt: Path, run_id: str, nonce: str) -> tuple[dict[str, Any], dict[str, Any]]:
    progress: dict[str, Any] = {}
    heartbeat: dict[str, Any] = {}
    progress_file = attempt / "progress.json"
    heartbeat_file = attempt / "heartbeat.json"
    if progress_file.exists():
        if progress_file.is_symlink():
            raise GuardError("progress is a symlink")
        progress = _read_json(progress_file)
        if progress.get("run_id") != run_id or progress.get("nonce") != nonce or progress.get("schema") != PROGRESS_SCHEMA:
            raise GuardError("progress identity/schema mismatch")
    if heartbeat_file.exists():
        if heartbeat_file.is_symlink():
            raise GuardError("heartbeat is a symlink")
        heartbeat = _read_json(heartbeat_file)
        if heartbeat.get("run_id") != run_id or heartbeat.get("nonce") != nonce:
            raise GuardError("heartbeat identity mismatch")
        if heartbeat.get("schema") not in (None, HEARTBEAT_SCHEMA):
            raise GuardError("heartbeat schema mismatch")
    return progress, heartbeat


def inspect_download(
    *,
    root: Path | str,
    attempt_dir: Path | str,
    run_id: str,
    nonce: str,
) -> dict[str, Any]:
    """Return a distinct MISSING/UNREADABLE/OK record for one attempt."""

    root_path = Path(root)
    attempt = Path(attempt_dir)
    if not attempt.is_absolute() or not root_path.is_absolute():
        raise GuardError("attempt paths must be absolute")
    if attempt.parent.name != "attempts" or attempt.parent.parent != root_path:
        raise GuardError("attempt directory is outside the owned root")
    base = {
        "schema": INSPECT_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "attempt": str(attempt),
    }
    with _guard_lock(root_path):
        if not attempt.is_dir() or attempt.is_symlink():
            return {**base, "state": "MISSING", "process": "ABSENT"}
        manifest_path = attempt / "launch.json"
        try:
            manifest = read_manifest(manifest_path, attempt)
        except GuardError as exc:
            # A registered attempt with a missing/unreadable current manifest is
            # UNREADABLE, not "no worker ever existed".
            return {**base, "state": "UNREADABLE", "reason": f"current attempt manifest: {exc}"}
        if manifest.get("run_id") != run_id or manifest.get("nonce") != nonce:
            return {**base, "state": "UNREADABLE", "reason": "current attempt manifest identity mismatch"}
        expected = expected_owner(manifest)
        try:
            argv_hash = expected_argv_sha256(manifest)
        except GuardError as exc:
            return {**base, "state": "UNREADABLE", "reason": f"manifest argv: {exc}"}
        status_file = Path(str(manifest["status"]))
        owner_file = Path(str(manifest["owner"]))
        ack_file = Path(str(manifest["ack"]))
        status: dict[str, Any] | None = None
        if status_file.exists():
            if status_file.is_symlink():
                return {**base, "state": "UNREADABLE", "reason": "status is a symlink"}
            try:
                status = _read_json(status_file)
            except GuardError as exc:
                return {**base, "state": "UNREADABLE", "reason": f"status: {exc}"}
            if status.get("run_id") != run_id or status.get("nonce") != nonce:
                return {**base, "state": "UNREADABLE", "reason": "status identity mismatch"}
            if status.get("schema") != STATUS_SCHEMA:
                return {**base, "state": "UNREADABLE", "reason": "status schema is invalid"}
        owner_exists = owner_file.is_file() and not owner_file.is_symlink()
        ack_exists = ack_file.is_file() and not ack_file.is_symlink()
        if not owner_exists and not ack_exists:
            if status is None:
                return {**base, "state": "MISSING", "process": "ABSENT"}
            if status.get("status") == "COMPLETE":
                return {
                    **base,
                    "state": "OK",
                    "process": "ABSENT",
                    "owner": None,
                    "ack": None,
                    "status": status,
                    "progress": {},
                    "heartbeat": {},
                    "manifest_sha256": expected["command_sha256"],
                }
            return {**base, "state": "UNREADABLE", "reason": "status exists without owner/ack"}
        if not owner_exists or not ack_exists:
            return {**base, "state": "UNREADABLE", "reason": "owner/ack pair is incomplete"}
        try:
            progress, heartbeat = _read_progress(attempt, run_id, nonce)
        except GuardError as exc:
            return {**base, "state": "UNREADABLE", "reason": str(exc)}
        # Reuse the same owner/ack/birth decision path as Comfy; the only extra
        # inputs here are the current attempt manifest and its status sidecars.
        decision = assess_owner(
            root=attempt,
            manifest=manifest,
            owner_path=owner_file,
            ack_path=ack_file,
            proof_dir=attempt / "process-proofs",
        )
        action = decision.get("action")
        if action == "REFUSE":
            return {
                **base,
                "state": "UNREADABLE",
                "reason": decision.get("reason") or decision.get("process"),
            }
        if action == "SPAWN":
            return {**base, "state": "MISSING", "process": "ABSENT"}
        return {
            **base,
            "state": "OK",
            "owner": decision.get("owner"),
            "ack": decision.get("ack"),
            "status": status,
            "progress": progress,
            "heartbeat": heartbeat,
            "manifest_sha256": expected["command_sha256"],
            **{key: value for key, value in decision.items() if key not in {"action", "owner", "ack"}},
        }


class _ObservationMissing(GuardError):
    pass


class _ObservationUnreadable(GuardError):
    pass


MAX_CONTROL_BYTES = 262144
MAX_PROOF_FILES = 64
MAX_TOTAL_BYTES = 2 * 1024 * 1024
CONTROL_NAMES = (
    "launch.json",
    "job.json",
    "worker-owner.json",
    "worker-ack.json",
    "status.json",
    "progress.json",
    "heartbeat.json",
    "recovery.json",
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _lstat_checked(path: str) -> os.stat_result:
    try:
        return os.lstat(path)
    except FileNotFoundError as exc:
        raise _ObservationMissing("path is missing") from exc
    except PermissionError as exc:
        raise _ObservationUnreadable("path is not readable") from exc
    except OSError as exc:
        raise _ObservationUnreadable("path lstat failed") from exc


def _safe_components(root: str | Path, target: str | Path, *, want_dir: bool = False) -> str:
    root_abs = os.path.abspath(str(root))
    target_abs = os.path.abspath(str(target))
    try:
        if os.path.commonpath([root_abs, target_abs]) != root_abs:
            raise _ObservationUnreadable("path escapes the owned root")
    except ValueError as exc:
        raise _ObservationUnreadable("path escapes the owned root") from exc
    root_info = _lstat_checked(root_abs)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise _ObservationUnreadable("owned root is not a regular directory")
    current = root_abs
    info = root_info
    relative = os.path.relpath(target_abs, root_abs)
    if relative == ".":
        return target_abs
    for part in relative.split(os.sep):
        if part in ("", "."):
            continue
        current = os.path.join(current, part)
        info = _lstat_checked(current)
        if stat.S_ISLNK(info.st_mode):
            raise _ObservationUnreadable("path contains a symlink")
        if current != target_abs and not stat.S_ISDIR(info.st_mode):
            raise _ObservationUnreadable("path parent is not a directory")
    if want_dir and not stat.S_ISDIR(info.st_mode):
        raise _ObservationUnreadable("path is not a directory")
    return target_abs


def _read_bytes_safe(
    root: str | Path,
    target: str | Path,
    max_bytes: int = MAX_CONTROL_BYTES,
    deadline: float | None = None,
) -> bytes:
    path = _safe_components(root, target)
    before = _lstat_checked(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise _ObservationUnreadable("file is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError as exc:
        raise _ObservationMissing("file is missing") from exc
    except PermissionError as exc:
        raise _ObservationUnreadable("file is not readable") from exc
    except OSError as exc:
        raise _ObservationUnreadable("file open failed") from exc
    try:
        after = os.fstat(fd)
        if not stat.S_ISREG(after.st_mode):
            raise _ObservationUnreadable("file is not regular")
        if after.st_size > max_bytes:
            raise _ObservationUnreadable("file is too large")
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            if deadline is not None and time.monotonic() >= deadline:
                raise _ObservationUnreadable("read deadline expired")
            chunk = os.read(fd, min(65536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > max_bytes:
            raise _ObservationUnreadable("file is too large")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise _ObservationUnreadable("duplicate JSON key")
        value[key] = item
    return value


def _read_json_safe(
    root: str | Path, target: str | Path, deadline: float | None = None
) -> dict[str, Any]:
    data = _read_bytes_safe(root, target, deadline=deadline)
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ObservationUnreadable("file is not valid JSON") from exc
    if not isinstance(value, dict):
        raise _ObservationUnreadable("JSON record is not an object")
    return value


def _read_optional_json(
    root: str | Path, target: str | Path, deadline: float | None = None
) -> tuple[bool, dict[str, Any] | None]:
    try:
        return True, _read_json_safe(root, target, deadline)
    except _ObservationMissing:
        return False, None


def _validate_status(status: Mapping[str, Any], run_id: str, nonce: str) -> None:
    if status.get("schema") != STATUS_SCHEMA:
        raise _ObservationUnreadable("status schema mismatch")
    if status.get("run_id") != run_id or status.get("nonce") != nonce:
        raise _ObservationUnreadable("status identity mismatch")
    if status.get("status") not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise _ObservationUnreadable("status value is invalid")


def _validate_progress(progress: Mapping[str, Any], run_id: str, nonce: str) -> None:
    if progress.get("schema") != PROGRESS_SCHEMA:
        raise _ObservationUnreadable("progress schema mismatch")
    if progress.get("run_id") != run_id or progress.get("nonce") != nonce:
        raise _ObservationUnreadable("progress identity mismatch")
    for key in ("bytes_done", "bytes_total", "files_done", "files_total"):
        if key in progress and (
            isinstance(progress[key], bool) or not isinstance(progress[key], int)
        ):
            raise _ObservationUnreadable("progress counter is invalid")


def _validate_heartbeat(heartbeat: Mapping[str, Any], run_id: str, nonce: str) -> None:
    if heartbeat.get("schema") not in (None, HEARTBEAT_SCHEMA):
        raise _ObservationUnreadable("heartbeat schema mismatch")
    if heartbeat.get("run_id") != run_id or heartbeat.get("nonce") != nonce:
        raise _ObservationUnreadable("heartbeat identity mismatch")
    if "heartbeat_at" in heartbeat and not _is_number(heartbeat["heartbeat_at"]):
        raise _ObservationUnreadable("heartbeat timestamp is invalid")


def _control_signature(
    root: str | Path, attempt: str | Path, deadline: float | None = None
) -> tuple[Any, ...]:
    _safe_components(root, attempt, want_dir=True)
    values: list[tuple[str, str]] = []
    total_bytes = 0
    for name in CONTROL_NAMES:
        target = os.path.join(str(attempt), name)
        try:
            data = _read_bytes_safe(root, target, deadline=deadline)
        except _ObservationMissing:
            values.append((name, "MISSING"))
            continue
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise _ObservationUnreadable("control signature byte limit exceeded")
        values.append((name, sha256_bytes(data)))
    proof_dir = os.path.join(str(attempt), "process-proofs")
    try:
        info = _lstat_checked(proof_dir)
    except _ObservationMissing:
        info = None
    if info is not None:
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise _ObservationUnreadable("proof directory is unsafe")
        try:
            names = sorted(os.listdir(proof_dir))
        except PermissionError as exc:
            raise _ObservationUnreadable("proof directory is not readable") from exc
        except OSError as exc:
            raise _ObservationUnreadable("proof directory list failed") from exc
        proof_names = [
            name for name in names if name.startswith("birth-") or name.startswith("current-")
        ]
        if len(proof_names) > MAX_PROOF_FILES:
            raise _ObservationUnreadable("proof file limit exceeded")
        for name in proof_names:
            target = os.path.join(proof_dir, name)
            try:
                data = _read_bytes_safe(root, target, deadline=deadline)
            except _ObservationMissing:
                values.append(("process-proofs/" + name, "MISSING"))
                continue
            total_bytes += len(data)
            if total_bytes > MAX_TOTAL_BYTES:
                raise _ObservationUnreadable("control signature byte limit exceeded")
            values.append(("process-proofs/" + name, sha256_bytes(data)))
    return tuple(sorted(values))


def _transient_read_failure(reason: Any) -> bool:
    text = str(reason or "").lower()
    return any(
        marker in text
        for marker in (
            "json",
            "changed",
            "truncated",
            "too large",
            "decode",
            "schema mismatch",
            "identity mismatch",
            "not an object",
            "disappeared",
        )
    )


def _validate_dead_proof(
    proof: Mapping[str, Any],
    birth: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if proof.get("schema") != PROOF_SCHEMA:
        raise GuardError("dead proof schema is invalid")
    for key in ("run_id", "nonce", "command_sha256"):
        if proof.get(key) != expected.get(key):
            raise GuardError("dead proof identity mismatch")
    for key in ("pid", "start_ticks", "exe", "cmdline_sha256", "argv_sha256", "boot_id"):
        if proof.get(key) != birth.get(key):
            raise GuardError("dead proof does not match birth")
    if proof.get("birth_sha256") != sha256_bytes(canonical_bytes(dict(birth))):
        raise GuardError("dead proof birth hash mismatch")


def _validate_live_proof(
    proof: Mapping[str, Any],
    birth: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    _validate_dead_proof(proof, birth, expected)


def _observe_once(
    *,
    root: str | Path,
    attempt_dir: str | Path,
    attempt_id: str,
    run_id: str,
    nonce: str,
    expected_job_sha256: str,
    expected_launch_sha256: str,
    expected_deadline_epoch: float | None,
    deadline: float | None,
) -> dict[str, Any]:
    expected_dir = os.path.normpath(os.path.join(str(root), "attempts", attempt_id))
    if os.path.normpath(str(attempt_dir)) != expected_dir:
        return {"state": "INCONCLUSIVE", "reason": "attempt path does not match expected id"}
    try:
        _safe_components(root, attempt_dir, want_dir=True)
    except _ObservationMissing:
        return {"state": "INCONCLUSIVE", "reason": "attempt disappeared during read"}
    except _ObservationUnreadable as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    launch_path = os.path.join(str(attempt_dir), "launch.json")
    job_path = os.path.join(str(attempt_dir), "job.json")
    try:
        launch_bytes = _read_bytes_safe(root, launch_path, deadline=deadline)
        job_bytes = _read_bytes_safe(root, job_path, deadline=deadline)
    except _ObservationMissing:
        return {"state": "INCONCLUSIVE", "reason": "attempt record disappeared during read"}
    except _ObservationUnreadable as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    if sha256_bytes(launch_bytes) != expected_launch_sha256:
        return {"state": "UNREADABLE", "reason": "expected launch hash mismatch"}
    if sha256_bytes(job_bytes) != expected_job_sha256:
        return {"state": "UNREADABLE", "reason": "expected job hash mismatch"}
    try:
        manifest = read_manifest(launch_path, attempt_dir)
    except GuardError as exc:
        return {"state": "UNREADABLE", "reason": f"launch manifest: {exc}"}
    if manifest.get("run_id") != run_id or manifest.get("nonce") != nonce:
        return {"state": "UNREADABLE", "reason": "launch identity mismatch"}
    if manifest.get("payload_sha256") != expected_job_sha256:
        return {"state": "UNREADABLE", "reason": "manifest payload hash mismatch"}
    manifest_deadline = manifest.get("deadline_epoch")
    if not _is_number(manifest_deadline):
        return {"state": "UNREADABLE", "reason": "manifest deadline is invalid"}
    if expected_deadline_epoch is not None and abs(
        float(manifest_deadline) - float(expected_deadline_epoch)
    ) > 1e-6:
        return {"state": "UNREADABLE", "reason": "manifest deadline mismatch"}
    expected = expected_owner(manifest)
    try:
        argv_hash = expected_argv_sha256(manifest)
    except GuardError as exc:
        return {"state": "UNREADABLE", "reason": f"manifest argv: {exc}"}
    owner_path = os.path.join(str(attempt_dir), "worker-owner.json")
    ack_path = os.path.join(str(attempt_dir), "worker-ack.json")
    status_path = os.path.join(str(attempt_dir), "status.json")
    progress_path = os.path.join(str(attempt_dir), "progress.json")
    heartbeat_path = os.path.join(str(attempt_dir), "heartbeat.json")
    try:
        owner_present, owner = _read_optional_json(root, owner_path, deadline)
        ack_present, ack = _read_optional_json(root, ack_path, deadline)
    except _ObservationUnreadable as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    if owner_present != ack_present:
        return {"state": "UNREADABLE", "reason": "owner/ack pair is incomplete"}
    if not owner_present:
        try:
            status = _read_json_safe(root, status_path, deadline)
        except _ObservationMissing:
            return {"state": "MISSING", "process": "ABSENT"}
        except _ObservationUnreadable as exc:
            return {"state": "UNREADABLE", "reason": str(exc)}
        try:
            _validate_status(status, run_id, nonce)
        except _ObservationUnreadable as exc:
            return {"state": "UNREADABLE", "reason": str(exc)}
        if status.get("status") != "COMPLETE":
            return {"state": "UNREADABLE", "reason": "owner/ack pair is incomplete"}
        progress: dict[str, Any] = {}
        heartbeat: dict[str, Any] = {}
        try:
            progress = _read_json_safe(root, progress_path, deadline)
            _validate_progress(progress, run_id, nonce)
        except _ObservationMissing:
            progress = {}
        except _ObservationUnreadable as exc:
            return {"state": "UNREADABLE", "reason": str(exc)}
        try:
            heartbeat = _read_json_safe(root, heartbeat_path, deadline)
            _validate_heartbeat(heartbeat, run_id, nonce)
        except _ObservationMissing:
            heartbeat = {}
        except _ObservationUnreadable as exc:
            return {"state": "UNREADABLE", "reason": str(exc)}
        return {
            "state": "OK",
            "process": "ABSENT",
            "owner": None,
            "ack": None,
            "status": status,
            "progress": progress,
            "heartbeat": heartbeat,
            "proof": None,
            "manifest_sha256": expected["command_sha256"],
            "job_sha256": expected_job_sha256,
            "launch_sha256": expected_launch_sha256,
            "attempt": str(attempt_dir),
            "run_id": run_id,
            "nonce": nonce,
        }
    try:
        _validate_owner_and_ack(owner, ack, expected, argv_hash)
    except GuardError as exc:
        return {"state": "UNREADABLE", "reason": f"owner/ack: {exc}"}
    try:
        status = _read_json_safe(root, status_path, deadline)
        _validate_status(status, run_id, nonce)
    except _ObservationMissing:
        return {"state": "UNREADABLE", "reason": "status is missing"}
    except _ObservationUnreadable as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    progress: dict[str, Any] = {}
    heartbeat: dict[str, Any] = {}
    try:
        progress = _read_json_safe(root, progress_path, deadline)
        _validate_progress(progress, run_id, nonce)
    except _ObservationMissing:
        progress = {}
    except _ObservationUnreadable as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    try:
        heartbeat = _read_json_safe(root, heartbeat_path, deadline)
        _validate_heartbeat(heartbeat, run_id, nonce)
    except _ObservationMissing:
        heartbeat = {}
    except _ObservationUnreadable as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    proof_dir = Path(str(attempt_dir)) / "process-proofs"
    try:
        if proof_dir.exists():
            info = _lstat_checked(str(proof_dir))
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                return {"state": "UNREADABLE", "reason": "proof directory is unsafe"}
            names = sorted(os.listdir(proof_dir))
            proof_names = [
                name for name in names if name.startswith("birth-") or name.startswith("current-")
            ]
            if len(proof_names) > MAX_PROOF_FILES:
                return {"state": "UNREADABLE", "reason": "proof file limit exceeded"}
            for name in proof_names:
                _read_bytes_safe(root, str(proof_dir / name), deadline=deadline)
    except _ObservationMissing:
        return {"state": "INCONCLUSIVE", "reason": "proof disappeared during read"}
    except _ObservationUnreadable as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    try:
        state = _process_state(
            owner,
            proof_dir,
            expected=expected,
            expected_argv_hash=argv_hash,
            write_recovery=False,
        )
    except GuardError as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    process = state.get("process")
    if process in {"MISSING_BIRTH", "UNKNOWN", "MISMATCH"}:
        return {
            "state": "INCONCLUSIVE",
            "reason": state.get("reason") or f"process {process}",
            "process": process,
        }
    if process == "DEAD":
        proof = state.get("proof")
        birth = state.get("birth")
        if proof is None:
            return {
                "state": "INCONCLUSIVE",
                "reason": "dead without existing proof pointer",
                "process": "UNKNOWN",
            }
        if not isinstance(proof, Mapping) or not isinstance(birth, Mapping):
            return {
                "state": "INCONCLUSIVE",
                "reason": "dead proof/birth is incomplete",
                "process": "UNKNOWN",
            }
        try:
            _validate_dead_proof(proof, birth, expected)
        except GuardError:
            return {
                "state": "INCONCLUSIVE",
                "reason": "dead proof does not match birth/identity",
                "process": "UNKNOWN",
            }
    if process == "LIVE":
        proof = state.get("proof")
        birth = state.get("birth")
        if isinstance(proof, Mapping) and isinstance(birth, Mapping):
            try:
                _validate_live_proof(proof, birth, expected)
            except GuardError:
                return {
                    "state": "INCONCLUSIVE",
                    "reason": "live proof does not match birth/identity",
                    "process": "UNKNOWN",
                }
    return {
        "state": "OK",
        "process": process,
        "owner": owner,
        "ack": ack,
        "status": status,
        "progress": progress,
        "heartbeat": heartbeat,
        "proof": state.get("proof"),
        "manifest_sha256": expected["command_sha256"],
        "job_sha256": expected_job_sha256,
        "launch_sha256": expected_launch_sha256,
        "attempt": str(attempt_dir),
        "run_id": run_id,
        "nonce": nonce,
    }


def observe_download(
    *,
    root: str | Path,
    attempt_dir: str | Path,
    attempt_id: str,
    run_id: str,
    nonce: str,
    expected_job_sha256: str,
    expected_launch_sha256: str,
    expected_deadline_epoch: float | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Pure read-only attempt observation; never locks, writes or recovers."""

    if not _is_sha256(expected_job_sha256) or not _is_sha256(expected_launch_sha256):
        return {"state": "INCONCLUSIVE", "reason": "expected job/launch hash missing"}
    if expected_deadline_epoch is not None and not _is_number(expected_deadline_epoch):
        return {"state": "INCONCLUSIVE", "reason": "expected deadline is invalid"}
    try:
        _safe_components(root, attempt_dir, want_dir=True)
    except _ObservationMissing:
        return {"state": "MISSING", "process": "ABSENT"}
    except _ObservationUnreadable as exc:
        return {"state": "UNREADABLE", "reason": str(exc)}
    last: dict[str, Any] | None = None
    for _ in range(3):
        if deadline is not None and time.monotonic() >= deadline:
            return {"state": "INCONCLUSIVE", "reason": "observer deadline expired"}
        try:
            before = _control_signature(root, attempt_dir, deadline)
        except _ObservationMissing:
            return {"state": "INCONCLUSIVE", "reason": "attempt disappeared during read"}
        except _ObservationUnreadable as exc:
            return {"state": "UNREADABLE", "reason": str(exc)}
        decision = _observe_once(
            root=root,
            attempt_dir=attempt_dir,
            attempt_id=attempt_id,
            run_id=run_id,
            nonce=nonce,
            expected_job_sha256=expected_job_sha256,
            expected_launch_sha256=expected_launch_sha256,
            expected_deadline_epoch=expected_deadline_epoch,
            deadline=deadline,
        )
        try:
            after = _control_signature(root, attempt_dir, deadline)
        except _ObservationMissing:
            return {"state": "INCONCLUSIVE", "reason": "attempt disappeared during read"}
        except _ObservationUnreadable as exc:
            return {"state": "UNREADABLE", "reason": str(exc)}
        last = decision
        if before == after:
            if decision.get("state") == "UNREADABLE" and _transient_read_failure(
                decision.get("reason")
            ):
                if deadline is not None and time.monotonic() >= deadline:
                    return {
                        "state": "INCONCLUSIVE",
                        "reason": "transient read failure during observation",
                    }
                time.sleep(0.05)
                continue
            return decision
        if deadline is not None and time.monotonic() >= deadline:
            return {"state": "INCONCLUSIVE", "reason": "attempt changed during read"}
        time.sleep(0.05)
    return {"state": "INCONCLUSIVE", "reason": "attempt changed during read"}


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: i1_process_guard.py <launch|inspect> ...", file=sys.stderr)
        return 2
    command = args[0]
    try:
        if command == "launch":
            if len(args) < 3:
                print(
                    "usage: i1_process_guard.py launch <root> <launch.json> [--resume-dead] [--capture-delay SECONDS]",
                    file=sys.stderr,
                )
                return 2
            resume_dead = "--resume-dead" in args
            capture_delay = 0.0
            if "--capture-delay" in args:
                index = args.index("--capture-delay")
                if index + 1 >= len(args):
                    return 2
                capture_delay = float(args[index + 1])
            result = launch_with_proof(
                args[1],
                args[2],
                resume_dead=resume_dead,
                capture_delay=capture_delay,
            )
        elif command == "inspect":
            if len(args) != 5:
                print("usage: i1_process_guard.py inspect <root> <attempt_dir> <run_id> <nonce>", file=sys.stderr)
                return 2
            result = inspect_download(
                root=args[1], attempt_dir=args[2], run_id=args[3], nonce=args[4]
            )
        else:
            print("unknown guard command", file=sys.stderr)
            return 2
    except Exception as exc:
        print(
            json.dumps(
                {"status": "REFUSED", "error": type(exc).__name__, "reason": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
