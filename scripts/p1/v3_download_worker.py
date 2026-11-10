"""Focused P1 v3 remote download worker (self-contained except setup_plan).

The operator stages this file, ``setup_plan.py`` and one pinned JSON payload via
the bounded SSH channel, then runs this fixed script with one payload path. It
does not accept arbitrary commands, upload a whole filesystem or read secrets.
The worker owns a target-level atomic lock directory + run_id/nonce owner file,
writes durable progress/status and uses ``setup_plan.download_verified``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from scripts.p1 import setup_plan  # noqa: E402
except ImportError:
    import setup_plan  # noqa: E402  staged beside this worker on the worker host


class WorkerError(Exception):
    pass


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
        newline="\n",
    )
    name = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerError("JSON must be an object")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def assert_disk(workspace: Path, required: int) -> None:
    if required <= 0:
        raise WorkerError("required bytes must be positive")
    if not workspace.exists():
        raise WorkerError("workspace is missing")
    usage = shutil.disk_usage(workspace)
    if usage.free < required:
        raise WorkerError("insufficient disk")


class Progress:
    def __init__(
        self,
        path: Path,
        status_path: Path,
        heartbeat_path: Path,
        run_id: str,
        nonce: str,
        files: int,
        total: int,
    ):
        self.path = path
        self.status_path = status_path
        self.heartbeat_path = heartbeat_path
        self.run_id = run_id
        self.nonce = nonce
        self.files = files
        self.total = total
        self.done = 0
        self.bytes = 0
        self.completed_bytes = 0
        self.file_bytes = 0

    def _write(self, status: str, error: str | None = None) -> None:
        now = time.time()
        bytes_done = min(self.total, self.completed_bytes + self.file_bytes)
        _atomic_write(
            self.path,
            json.dumps(
                {
                    "schema": "anime-video-studio/p1-v3-download-progress/v1",
                    "run_id": self.run_id,
                    "nonce": self.nonce,
                    "status": status,
                    "files_done": self.done,
                    "files_total": self.files,
                    "bytes_done": bytes_done,
                    "bytes_total": self.total,
                    "heartbeat_at": now,
                    "last_error": error,
                },
                sort_keys=True,
            ) + "\n",
        )
        _atomic_write(
            self.status_path,
            json.dumps(
                {
                    "schema": "anime-video-studio/p1-v3-download-status/v1",
                    "run_id": self.run_id,
                    "nonce": self.nonce,
                    "status": status,
                    "files_done": self.done,
                    "files_total": self.files,
                    "bytes_done": bytes_done,
                    "bytes_total": self.total,
                    "heartbeat_at": now,
                    "last_error": error,
                },
                sort_keys=True,
            ) + "\n",
        )
        _atomic_write(
            self.heartbeat_path,
            json.dumps(
                {
                    "schema": "anime-video-studio/p1-v3-download-heartbeat/v1",
                    "run_id": self.run_id,
                    "nonce": self.nonce,
                    "status": status,
                    "heartbeat_at": now,
                    "pid": os.getpid(),
                },
                sort_keys=True,
            ) + "\n",
        )

    def start(self) -> None:
        self._write("RUNNING")

    def callback(self, written: int, total: int) -> None:
        if written < 0 or total <= 0 or written > total:
            raise WorkerError("invalid progress")
        self.file_bytes = written
        self.bytes = written
        self._write("RUNNING")

    def file_done(self) -> None:
        self.completed_bytes += self.file_bytes
        self.file_bytes = 0
        self.done += 1
        self._write("RUNNING")

    def fail(self, error: str) -> None:
        self._write("FAILED", error)

    def complete(self) -> None:
        self._write("COMPLETE")


class TargetLock:
    """Atomic target-level lock directory; owner identity is run_id+nonce."""

    def __init__(self, target: Path, run_id: str, nonce: str, *, resume_dead: bool = False):
        self.target = target
        self.run_id = run_id
        self.nonce = nonce
        self.resume_dead = resume_dead
        self.lock_dir = target / ".p1-download.lock"
        self.owner = self.lock_dir / "owner.json"
        self.held = False

    def _dead_owner_matches(self) -> bool:
        try:
            owner = _read_json(self.owner)
        except Exception:
            return False
        if owner.get("run_id") != self.run_id or owner.get("nonce") != self.nonce:
            return False
        pid = owner.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        if os.name == "nt":
            try:
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if not handle:
                    return True
                ctypes.windll.kernel32.CloseHandle(handle)
                return False
            except Exception:
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

    def acquire(self) -> "TargetLock":
        self.target.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(self.lock_dir)
        except FileExistsError as exc:
            if self.resume_dead and self._dead_owner_matches():
                # Explicit dead-resume only: same run/nonce and recorded PID is
                # gone.  Keep the old owner record as audit, never a blind reset.
                archived = self.target / f".p1-download.owner.dead-{int(time.time())}.json"
                try:
                    os.replace(self.owner, archived)
                    os.rmdir(self.lock_dir)
                    os.mkdir(self.lock_dir)
                except OSError:
                    raise WorkerError("dead target lock could not be reclaimed") from exc
            else:
                raise WorkerError("target downloader lock is already held") from exc
        try:
            _atomic_write(
                self.owner,
                json.dumps(
                    {
                        "schema": "anime-video-studio/p1-v3-target-lock/v1",
                        "run_id": self.run_id,
                        "nonce": self.nonce,
                        "pid": os.getpid(),
                        "created_at": time.time(),
                    },
                    sort_keys=True,
                ) + "\n",
            )
        except BaseException:
            self.lock_dir.rmdir()
            raise
        self.held = True
        return self

    def assert_owned(self) -> None:
        if not self.held:
            raise WorkerError("target lock is not held")
        owner = _read_json(self.owner)
        if owner.get("run_id") != self.run_id or owner.get("nonce") != self.nonce:
            raise WorkerError("target lock owner mismatch")

    def release(self) -> None:
        try:
            if self.held and self.owner.exists():
                self.owner.unlink()
            if self.held:
                self.lock_dir.rmdir()
        except OSError:
            pass
        self.held = False


def _write_result(path: Path, payload: dict) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(payload_path: Path) -> int:
    payload = _read_json(payload_path)
    if payload.get("schema") != "anime-video-studio/p1-v3-download-job/v1":
        raise WorkerError("download payload schema is invalid")
    run_id = payload.get("run_id")
    nonce = payload.get("nonce")
    workspace = Path(str(payload.get("workspace", "")))
    root = Path(str(payload.get("root", "")))
    deadline_epoch = payload.get("deadline_epoch")
    manifest_path = Path(str(payload.get("manifest_path", "")))
    expected_manifest_sha = payload.get("manifest_sha256")
    status_path = Path(str(payload.get("status_path", "")))
    progress_path = Path(str(payload.get("progress_path", "")))
    heartbeat_path = Path(str(payload.get("heartbeat_path", "")))
    resume_dead = os.environ.get("P1_RESUME_DEAD") == "1"
    if not isinstance(run_id, str) or not run_id:
        raise WorkerError("download run_id is missing")
    if not isinstance(nonce, str) or not nonce:
        raise WorkerError("download nonce is missing")
    if not isinstance(deadline_epoch, (int, float)) or isinstance(deadline_epoch, bool):
        raise WorkerError("download deadline is missing")
    for label, value in (
        ("status", status_path),
        ("progress", progress_path),
        ("heartbeat", heartbeat_path),
    ):
        if not value.is_absolute() or value.parent != workspace:
            raise WorkerError(f"download {label} path is outside the owned root")
    if not manifest_path.is_file():
        raise WorkerError("download manifest is missing")
    if _sha_file(manifest_path) != expected_manifest_sha:
        raise WorkerError("download manifest hash mismatch")
    manifest = setup_plan.load_manifest(manifest_path)
    specs = setup_plan.manifest_download_specs(manifest)
    required = sum(spec.size_bytes for spec in specs)
    assert_disk(workspace, required)
    if time.time() >= float(deadline_epoch):
        raise WorkerError("download deadline has passed")
    recorder = Progress(
        progress_path,
        status_path,
        heartbeat_path,
        run_id,
        nonce,
        len(specs),
        required,
    )
    recorder.start()
    lock = TargetLock(workspace, run_id, nonce, resume_dead=resume_dead)
    try:
        lock.acquire()
        lock.assert_owned()
        results = []
        for spec in specs:
            if time.time() >= float(deadline_epoch):
                raise WorkerError("download deadline expired")
            assert_disk(workspace, spec.size_bytes)

            def progress(written: int, total: int) -> None:
                recorder.callback(written, total)
                if time.time() >= float(deadline_epoch):
                    raise WorkerError("download deadline expired")

            with open(status_path.parent / "download.log", "a", encoding="utf-8") as log:
                result = setup_plan.download_verified(
                    spec,
                    root=root,
                    progress=progress,
                    log=log,
                )
            recorder.file_done()
            results.append(
                {
                    "model_id": spec.model_id,
                    "target": spec.target,
                    "size_bytes": result.size_bytes,
                    "sha256": result.sha256,
                    "skipped": result.skipped,
                }
            )
        recorder.complete()
        _write_result(
            status_path,
            {
                "schema": "anime-video-studio/p1-v3-download-status/v1",
                "run_id": run_id,
                "nonce": nonce,
                "status": "COMPLETE",
                "files": results,
                "bytes": sum(item["size_bytes"] for item in results),
            },
        )
        return 0
    except BaseException as exc:
        recorder.fail(type(exc).__name__)
        _write_result(
            status_path,
            {
                "schema": "anime-video-studio/p1-v3-download-status/v1",
                "run_id": run_id,
                "nonce": nonce,
                "status": "FAILED",
                "error": type(exc).__name__,
            },
        )
        raise
    finally:
        lock.release()


def _record_early_failure(payload_path: Path, exc: BaseException) -> None:
    """Persist a same-run FAILED status if the worker died before its recorder.

    This keeps explicit dead-resume honest: the launcher can later verify a
    same-run FAILED record instead of treating an owner-without-status state as
    automatically recoverable.
    """

    try:
        payload = _read_json(payload_path)
        run_id = payload.get("run_id")
        nonce = payload.get("nonce")
        workspace = Path(str(payload.get("workspace", "")))
        status_path = Path(str(payload.get("status_path", "")))
        progress_path = Path(str(payload.get("progress_path", "")))
        heartbeat_path = Path(str(payload.get("heartbeat_path", "")))
        if not isinstance(run_id, str) or not run_id or not isinstance(nonce, str) or not nonce:
            return
        for path in (status_path, progress_path, heartbeat_path):
            if not path.is_absolute() or path.parent != workspace:
                return
        if status_path.exists():
            return
        now = time.time()
        base = {
            "run_id": run_id,
            "nonce": nonce,
            "status": "FAILED",
            "error": type(exc).__name__,
        }
        _write_result(
            status_path,
            {
                "schema": "anime-video-studio/p1-v3-download-status/v1",
                **base,
            },
        )
        _atomic_write(
            progress_path,
            json.dumps(
                {
                    "schema": "anime-video-studio/p1-v3-download-progress/v1",
                    **base,
                    "files_done": 0,
                    "files_total": 0,
                    "bytes_done": 0,
                    "bytes_total": 0,
                    "heartbeat_at": now,
                },
                sort_keys=True,
            )
            + "\n",
        )
        _atomic_write(
            heartbeat_path,
            json.dumps(
                {
                    "schema": "anime-video-studio/p1-v3-download-heartbeat/v1",
                    **base,
                    "heartbeat_at": now,
                    "pid": os.getpid(),
                },
                sort_keys=True,
            )
            + "\n",
        )
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: v3_download_worker.py <payload.json>", file=sys.stderr)
        return 2
    payload_path = Path(args[0])
    try:
        return run(payload_path)
    except Exception as exc:
        _record_early_failure(payload_path, exc)
        print(f"download worker failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
