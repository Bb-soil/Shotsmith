"""I1-only Comfy spawn/attach supervisor.

The supervisor holds a stable local lock and delegates ownership decisions to
the staged ``i1_process_guard`` helper.  The helper validates the full owner/ack
identity against the launch manifest, proves the live incarnation through
``/proc`` and writes a durable remote proof before returning.  A proved-live
owner is attached directly; the bounded port wait is used only for a new spawn.
Foreign or partial owner/ack records are retained byte-for-byte.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

try:
    import i1_process_guard as guard  # type: ignore
except ImportError:  # pragma: no cover
    from scripts.i1 import process_guard as guard  # type: ignore


class SupervisorError(Exception):
    pass


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_port_free(port: int, timeout: float = 75.0) -> bool:
    # The frozen launcher's final bind check does not set SO_REUSEADDR, so a
    # just-killed Comfy can leave TIME_WAIT sockets that make an immediate
    # restart look like an unknown live listener.  Wait for the kernel to
    # release the port before invoking the frozen launcher; never kill or
    # bypass an actual foreign listener.  This path is only used for a new
    # spawn, never for attaching to a proved-live owned Comfy.
    deadline = time.monotonic() + timeout
    while True:
        if _port_free(port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


def _archive(directory: Path, names: tuple[str, ...]) -> None:
    archive = directory / "comfy-archive"
    archive.mkdir(mode=0o700, exist_ok=True)
    stamp = time.time_ns()
    for name in names:
        candidate = directory / name
        if candidate.exists():
            os.replace(candidate, archive / f"{name}.dead-{stamp}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: i1_comfy_supervisor.py <root> <launch.json>", file=sys.stderr)
        return 2
    root = Path(args[0])
    launch_path = Path(args[1])
    if not root.is_absolute() or launch_path.parent != root:
        print("supervisor paths are invalid", file=sys.stderr)
        return 2
    lock_path = root / "comfy-owner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
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
        try:
            manifest = guard.read_manifest(launch_path, root)
            decision = guard.assess_owner(
                root=root,
                manifest=manifest,
                owner_path=root / "comfy-owner.json",
                ack_path=root / "comfy-ack.json",
                proof_dir=root / "comfy-process-proofs",
            )
        except guard.GuardError as exc:
            print(f"comfy owner validation failed: {exc}", file=sys.stderr)
            return 3
        action = decision.get("action")
        if action == "REFUSE":
            print(
                "comfy owner refused without mutation: "
                + str(decision.get("reason") or decision.get("process")),
                file=sys.stderr,
            )
            return 3
        if action == "ARCHIVE_DEAD":
            # Validation of the full owner/ack identity and the absent process
            # happened before this single rename.  Nothing foreign is archived.
            _archive(root, ("comfy-owner.json", "comfy-ack.json"))
        if action != "ATTACH":
            # A new spawn (including after a proved-dead archive) waits for the
            # old listen socket to be released.  A proved-live attach never
            # waits on the port because its own Comfy owns it.
            try:
                port = int(manifest["port"])
            except (KeyError, TypeError, ValueError):
                print("comfy launch manifest has no usable port", file=sys.stderr)
                return 3
            if not _wait_port_free(port):
                print(
                    "comfy port did not become free; foreign listener preserved",
                    file=sys.stderr,
                )
                return 3
        try:
            result = guard.launch_with_proof(root, launch_path)
        except guard.GuardError as exc:
            print(f"frozen comfy launcher refused: {exc}", file=sys.stderr)
            return 3
        except Exception as exc:
            print(
                f"frozen comfy launcher refused: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 3
        print(json.dumps(result, sort_keys=True))
        return 0
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


if __name__ == "__main__":
    raise SystemExit(main())
