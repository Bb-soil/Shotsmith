"""Focused detached-process launcher for P1 v3 remote work.

This file is staged into the operator-owned remote directory and executed with
one absolute launch-manifest path.  It is deliberately not a generic command
runner: the manifest selects one of two fixed process shapes (the pinned
download worker or ComfyUI), every path is re-validated against the owned root,
and ownership is a no-clobber record bound to run_id + nonce + the manifest
hash.  A PID alone is never accepted as ownership.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import time

SCHEMA = "anime-video-studio/p1-v3-launch/v1"
RESULT_SCHEMA = "anime-video-studio/p1-v3-launch-result/v1"
OWNER_SCHEMA = "anime-video-studio/p1-v3-process-owner/v1"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_HASH_BYTES = 64 * 1024 * 1024
_COMMON_KEYS = {
    "schema",
    "kind",
    "run_id",
    "nonce",
    "root",
    "log",
    "owner",
    "ack",
}
_DOWNLOAD_KEYS = _COMMON_KEYS | {
    "worker",
    "worker_sha256",
    "payload",
    "payload_sha256",
    "deadline_epoch",
    "status",
    "progress",
    "heartbeat",
}
_COMFY_KEYS = _COMMON_KEYS | {"python", "comfy_dir", "main", "port"}


class LauncherError(Exception):
    pass


def _canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_hash(manifest) -> str:
    return hashlib.sha256(_canonical(manifest)).hexdigest()


def _validate_path(root: str, value, name: str, *, allow_missing: bool = False, allow_final_symlink: bool = False) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise LauncherError(f"{name} must be a non-empty string")
    if os.name == "nt":
        if not os.path.isabs(value):
            raise LauncherError(f"{name} must be an absolute path")
    else:
        if "\\" in value or not os.path.isabs(value):
            raise LauncherError(f"{name} must be an absolute POSIX path")
    normalized = os.path.normpath(value)
    if normalized != value or any(part == ".." for part in value.split(os.sep)):
        raise LauncherError(f"{name} must be normalized without traversal")
    root_real = os.path.realpath(root)
    lexical = os.path.abspath(value)
    try:
        if os.path.commonpath([root_real, lexical]) != root_real:
            raise LauncherError(f"{name} escapes the owned root")
    except ValueError as exc:
        raise LauncherError(f"{name} escapes the owned root") from exc
    if os.path.islink(value):
        if not allow_final_symlink:
            raise LauncherError(f"{name} must not be a symlink")
        real = os.path.realpath(value)
        if not os.path.isfile(real) or not os.path.basename(real).startswith("python"):
            raise LauncherError(f"{name} symlink target is not an expected Python interpreter")
        allowed_roots = [
            os.path.realpath(os.path.join(root, "p1-venv")),
            os.path.realpath(os.path.dirname(sys.executable)),
            "/usr/bin",
            "/usr/local/bin",
            "/bin",
        ]
        if not any(
            real == allowed or real.startswith(allowed.rstrip("/\\") + os.sep)
            for allowed in allowed_roots
        ):
            raise LauncherError(f"{name} symlink target is outside the expected interpreter roots")
        # The final Python link itself is allowed, but a symlinked/junctioned
        # parent chain could redirect the lexical owned path into a foreign
        # venv. Require the real parent chain to stay under the owned root and
        # to be canonical (no symlink/junction parents).
        parent = os.path.dirname(value)
        parent_real = os.path.realpath(parent)
        try:
            if os.path.commonpath([root_real, parent_real]) != root_real:
                raise LauncherError(f"{name} parent escapes the owned root")
            expected_parent = os.path.normpath(
                os.path.join(root_real, os.path.relpath(parent, root))
            )
        except ValueError as exc:
            raise LauncherError(f"{name} parent escapes the owned root") from exc
        if parent_real != expected_parent:
            raise LauncherError(f"{name} parent resolves through a symlink or junction")
        # Keep the lexical owned venv/bin/python path for exec: on POSIX it is a
        # symlink to the base interpreter, and executing the resolved target
        # would drop sys.prefix/site-packages. The target above was still
        # validated as a regular Python file inside the allowed roots, and the
        # lexical path is contained in the owned root.
        return lexical
    real = os.path.realpath(value)
    if real != value:
        raise LauncherError(f"{name} resolves through a symlink or junction")
    if not allow_missing:
        if not os.path.lexists(value):
            raise LauncherError(f"{name} does not exist")
        if os.path.islink(value):
            raise LauncherError(f"{name} must not be a symlink")
    return value


def _assert_regular(path: str, name: str, *, expected_sha: str | None = None) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise LauncherError(f"{name} is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or os.path.islink(path):
        raise LauncherError(f"{name} must be a regular, non-symlink file")
    if info.st_size <= 0 or info.st_size > MAX_HASH_BYTES:
        raise LauncherError(f"{name} size is outside the bound")
    if expected_sha is not None:
        digest = hashlib.sha256()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise LauncherError(f"{name} cannot be opened safely") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise LauncherError(f"{name} is not a regular file")
            if info.st_size != os.path.getsize(path):
                raise LauncherError(f"{name} changed during validation")
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(fd)
        if digest.hexdigest() != expected_sha:
            raise LauncherError(f"{name} hash does not match the launch manifest")


def _read_manifest(path: str) -> tuple[dict, str]:
    if not os.path.isabs(path):
        raise LauncherError("launch manifest path must be absolute")
    if os.path.islink(path) or not os.path.isfile(path):
        raise LauncherError("launch manifest must be a regular non-symlink file")
    size = os.path.getsize(path)
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise LauncherError("launch manifest size is outside the bound")
    with open(path, "rb") as handle:
        data = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise LauncherError("launch manifest exceeds the bound")
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherError("launch manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise LauncherError("launch manifest must be an object")
    return manifest, hashlib.sha256(_canonical(manifest)).hexdigest()


def _validate_manifest(manifest: dict, *, manifest_path: str) -> tuple[dict, str]:
    kind = manifest.get("kind")
    if kind == "download_worker":
        allowed = _DOWNLOAD_KEYS
    elif kind == "comfy":
        allowed = _COMFY_KEYS
    else:
        raise LauncherError("launch kind is not allowlisted")
    if set(manifest) != allowed:
        raise LauncherError("launch manifest has unknown/missing keys")
    if manifest.get("schema") != SCHEMA:
        raise LauncherError("launch manifest schema is invalid")
    for key in ("run_id", "nonce"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise LauncherError(f"{key} is invalid")
    root = manifest.get("root")
    if not isinstance(root, str):
        raise LauncherError("root is invalid")
    if not os.path.isabs(root) or os.path.normpath(root) != root:
        raise LauncherError("root must be a normalized absolute path")
    _validate_path(root, manifest_path, "launch manifest")
    log = _validate_path(root, manifest.get("log"), "log", allow_missing=True)
    owner = _validate_path(root, manifest.get("owner"), "owner", allow_missing=True)
    ack = _validate_path(root, manifest.get("ack"), "ack", allow_missing=True)
    if kind == "download_worker":
        worker = _validate_path(root, manifest.get("worker"), "worker")
        payload = _validate_path(root, manifest.get("payload"), "payload")
        if os.path.basename(worker) != "v3_download_worker.py":
            raise LauncherError("worker filename is not the pinned worker")
        if os.path.basename(payload) != "job.json":
            raise LauncherError("payload filename is not the pinned job")
        _assert_regular(worker, "worker", expected_sha=manifest.get("worker_sha256"))
        _assert_regular(payload, "payload", expected_sha=manifest.get("payload_sha256"))
        deadline = manifest.get("deadline_epoch")
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise LauncherError("deadline_epoch is invalid")
        if time.time() >= float(deadline):
            raise LauncherError("launch deadline has already passed")
        if os.path.dirname(log) != root or os.path.dirname(owner) != root or os.path.dirname(ack) != root:
            raise LauncherError("download owner/log/ack must be directly in the owned root")
        status_path = _validate_path(
            root, manifest.get("status"), "status", allow_missing=True
        )
        _validate_path(root, manifest.get("progress"), "progress", allow_missing=True)
        _validate_path(root, manifest.get("heartbeat"), "heartbeat", allow_missing=True)
        argv = [sys.executable, worker, payload]
        cwd = root
    else:
        python = _validate_path(
            root, manifest.get("python"), "python", allow_final_symlink=True
        )
        comfy_dir = _validate_path(root, manifest.get("comfy_dir"), "comfy_dir")
        main = _validate_path(root, manifest.get("main"), "main")
        port = manifest.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
            raise LauncherError("comfy port is invalid")
        if os.path.basename(main) != "main.py":
            raise LauncherError("comfy entrypoint filename is not main.py")
        if not os.path.isfile(python):
            raise LauncherError("comfy python must resolve to a regular file")
        if not os.path.islink(python):
            _assert_regular(python, "comfy python")
        _assert_regular(main, "comfy main.py")
        if os.path.dirname(log) != root or os.path.dirname(owner) != root or os.path.dirname(ack) != root:
            raise LauncherError("comfy owner/log/ack must be directly in the owned root")
        argv = [python, main, "--listen", "127.0.0.1", "--port", str(port)]
        cwd = comfy_dir
    return {
        "kind": kind,
        "root": root,
        "run_id": manifest["run_id"],
        "nonce": manifest["nonce"],
        "log": log,
        "owner": owner,
        "ack": ack,
        "argv": argv,
        "cwd": cwd,
        "command_sha256": manifest_hash(manifest),
    }, manifest_hash(manifest)


def _read_owner(path: str) -> dict | None:
    if not os.path.lexists(path):
        return None
    if os.path.islink(path) or not os.path.isfile(path):
        raise LauncherError("owner record is not a regular file")
    if os.path.getsize(path) > 64 * 1024:
        raise LauncherError("owner record exceeds the bound")
    try:
        value = json.loads(open(path, "rb").read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherError("owner record is unreadable") from exc
    if not isinstance(value, dict):
        raise LauncherError("owner record must be an object")
    return value


def _write_noclobber(path: str, payload) -> None:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _pid_alive(pid: int) -> bool:
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


def _terminate_child(process) -> None:
    pid = getattr(process, "pid", None)
    try:
        if os.name == "nt" and isinstance(pid, int) and pid > 0:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        else:
            process.kill()
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _read_status_identity(path: str) -> dict:
    if os.path.islink(path) or not os.path.isfile(path):
        raise LauncherError("worker status for dead-resume is missing/unsafe")
    if os.path.getsize(path) > 64 * 1024:
        raise LauncherError("worker status exceeds the bound")
    try:
        value = json.loads(open(path, "rb").read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherError("worker status for dead-resume is unreadable") from exc
    if not isinstance(value, dict):
        raise LauncherError("worker status must be an object")
    return value


LOCK_WAIT_SECONDS = 20.0


class _LaunchLock:
    """One OS-level lock held across the whole ownership decision.

    The OS releases the lock if the launcher crashes, so a stale lock file is
    never treated as live ownership.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._handle = None

    def acquire(self, wait_seconds: float = LOCK_WAIT_SECONDS) -> None:
        if os.path.islink(self.path):
            raise LauncherError("launch lock must not be a symlink")
        handle = open(self.path, "a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        deadline = time.monotonic() + max(0.05, float(wait_seconds))
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return
            except OSError as exc:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise LauncherError(
                        "another launcher owns the launch lock; no second process started"
                    ) from exc
                time.sleep(0.05)

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


def launch(manifest_path: str, *, resume_dead: bool = False) -> dict:
    manifest, raw_hash = _read_manifest(manifest_path)
    checked, command_hash = _validate_manifest(manifest, manifest_path=manifest_path)
    if checked["command_sha256"] != raw_hash:
        raise LauncherError("manifest hash changed during validation")
    expected_owner = {
        "schema": OWNER_SCHEMA,
        "kind": checked["kind"],
        "run_id": checked["run_id"],
        "nonce": checked["nonce"],
        "command_sha256": command_hash,
    }
    lock = _LaunchLock(checked["owner"] + ".launch.lock")
    lock.acquire()
    try:
        # The entire ownership decision happens under the lock: read current
        # owner, compare the same attempt, archive only a verified dead
        # attempt, then spawn and register exactly one child.
        owner = _read_owner(checked["owner"])
        if owner is not None:
            if any(owner.get(key) != value for key, value in expected_owner.items()):
                raise LauncherError(
                    "remote ownership conflict: another run/nonce/command owns this root"
                )
            pid = owner.get("pid")
            if isinstance(pid, int) and _pid_alive(pid):
                return {
                    "schema": RESULT_SCHEMA,
                    "status": "ATTACHED",
                    "kind": checked["kind"],
                    "run_id": checked["run_id"],
                    "nonce": checked["nonce"],
                    "command_sha256": command_hash,
                    "pid": pid,
                }
            status_path = (
                manifest.get("status")
                if checked["kind"] == "download_worker"
                else None
            )
            status = _read_status_identity(status_path) if status_path else None
            if status is not None and (
                status.get("run_id") == checked["run_id"]
                and status.get("nonce") == checked["nonce"]
                and status.get("status") == "COMPLETE"
            ):
                return {
                    "schema": RESULT_SCHEMA,
                    "status": "ATTACHED_COMPLETE",
                    "kind": checked["kind"],
                    "run_id": checked["run_id"],
                    "nonce": checked["nonce"],
                    "command_sha256": command_hash,
                    "pid": pid,
                }
            if not resume_dead or checked["kind"] != "download_worker":
                raise LauncherError(
                    "owned process is recorded but not alive; explicit dead-resume policy required"
                )
            if (
                status is None
                or status.get("run_id") != checked["run_id"]
                or status.get("nonce") != checked["nonce"]
                or status.get("status") != "FAILED"
            ):
                raise LauncherError(
                    "dead-resume refused: worker status is not a same-run FAILED record"
                )
            ack = _read_owner(checked["ack"])
            if (
                ack is None
                or any(ack.get(key) != value for key, value in expected_owner.items())
                or ack.get("pid") != pid
            ):
                raise LauncherError(
                    "dead-resume refused: ack does not match the same dead attempt"
                )
            if _pid_alive(pid):
                raise LauncherError(
                    "dead-resume refused: the recorded attempt became alive"
                )
            stamp = time.time_ns()
            try:
                os.replace(checked["owner"], f"{checked['owner']}.dead-{stamp}")
                if os.path.lexists(checked["ack"]):
                    os.replace(checked["ack"], f"{checked['ack']}.dead-{stamp}")
            except OSError as exc:
                raise LauncherError(
                    "dead-resume could not archive the old owner record"
                ) from exc
        if checked["kind"] == "comfy" and not _port_free(int(manifest["port"])):
            raise LauncherError("comfy port is occupied by an unknown process")
        if os.path.lexists(checked["ack"]):
            raise LauncherError("stale ack exists without the same owner; manual recovery required")
        log_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        log_flags |= getattr(os, "O_NOFOLLOW", 0)
        log_fd = os.open(checked["log"], log_flags, 0o600)
        process = None
        try:
            kwargs = {
                "stdin": subprocess.DEVNULL,
                "stdout": log_fd,
                "stderr": subprocess.STDOUT,
                "cwd": checked["cwd"],
                "close_fds": True,
            }
            if resume_dead and checked["kind"] == "download_worker":
                env = dict(os.environ)
                env["P1_RESUME_DEAD"] = "1"
                kwargs["env"] = env
            if os.name == "nt":
                kwargs["creationflags"] = (
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "DETACHED_PROCESS", 0)
                )
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(checked["argv"], **kwargs)
        finally:
            os.close(log_fd)
        record = dict(expected_owner)
        record.update(
            {
                "pid": int(process.pid),
                "argv_sha256": hashlib.sha256(_canonical(checked["argv"])).hexdigest(),
                "started_at": time.time(),
                "manifest_path": manifest_path,
            }
        )
        try:
            _write_noclobber(checked["owner"], record)
        except FileExistsError as exc:
            # A racing launcher wrote first; never overwrite it and never leave
            # an unregistered child running.
            _terminate_child(process)
            raise LauncherError("remote ownership record already exists") from exc
        except BaseException:
            _terminate_child(process)
            raise
        ack = dict(expected_owner)
        ack.update({"pid": int(process.pid), "ack_at": time.time()})
        try:
            _write_noclobber(checked["ack"], ack)
        except BaseException as exc:
            _terminate_child(process)
            # Remove only the owner record this launcher just wrote, so a
            # failed registration does not leave dead ownership behind.
            try:
                current = _read_owner(checked["owner"])
                if current is not None and all(
                    current.get(key) == value
                    for key, value in expected_owner.items()
                ) and current.get("pid") == int(process.pid):
                    os.unlink(checked["owner"])
            except (LauncherError, OSError):
                pass
            if isinstance(exc, FileExistsError):
                raise LauncherError("remote ack record already exists") from exc
            raise
    finally:
        lock.release()
    return {
        "schema": RESULT_SCHEMA,
        "status": "STARTED",
        "kind": checked["kind"],
        "run_id": checked["run_id"],
        "nonce": checked["nonce"],
        "command_sha256": command_hash,
        "pid": int(process.pid),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (2, 3) or args[0] != "--launch":
        print("usage: v3_remote_launcher.py --launch <manifest.json> [--resume-dead]", file=sys.stderr)
        return 2
    resume_dead = args[2:] == ["--resume-dead"]
    try:
        result = launch(args[1], resume_dead=resume_dead)
    except Exception as exc:
        print(
            json.dumps(
                {"schema": RESULT_SCHEMA, "status": "REFUSED", "error": type(exc).__name__, "reason": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
