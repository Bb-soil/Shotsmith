"""Focused remote byte-stream/read helper for exact P1 v3 artifacts.

``stream`` prints one JSON header line followed by the requested file bytes from
``offset``.  ``read`` prints one bounded JSON envelope with the exact file
content so the operator can read fresh durable status files without a shell
``cat`` redirection.  Both modes reject absolute paths outside an explicitly
provided owned root, any symlink/junction component, and non-regular files
before hashing or writing anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys

HEADER_SCHEMA = "anime-video-studio/p1-v3-stream/v1"
READ_SCHEMA = "anime-video-studio/p1-v3-read/v1"
METADATA_SCHEMA = "anime-video-studio/p1-v3-metadata/v1"
MAX_READ_BYTES = 256 * 1024


def _lstat(path: str):
    try:
        return os.lstat(path)
    except OSError:
        return None


def _assert_safe_path(path: str, *, root: str | None) -> str:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise SystemExit("remote path must be a non-empty string")
    if os.name == "nt":
        if not os.path.isabs(path):
            raise SystemExit("remote path must be an absolute path")
    else:
        if "\\" in path or not os.path.isabs(path):
            raise SystemExit("remote path must be an absolute POSIX path")
    normalized = os.path.normpath(path)
    if normalized != path or any(part == ".." for part in path.split(os.sep)):
        raise SystemExit("remote path must be normalized without traversal")
    lowered = path.lower().replace("\\", "/")
    if any(part in lowered for part in ("/.env", "/.ssh", "/.aws", "/.config")):
        raise SystemExit("remote path is forbidden")
    if root is not None:
        root_real = os.path.realpath(root)
        real = os.path.realpath(path)
        try:
            if os.path.commonpath([root_real, real]) != root_real:
                raise SystemExit("remote path escapes the owned root")
        except ValueError:
            raise SystemExit("remote path escapes the owned root")
        if real != path:
            raise SystemExit("remote path resolves through a symlink/junction")
    # Reject any symlink/junction component before hashing or opening.
    current = path
    while True:
        info = _lstat(current)
        if info is not None and stat.S_ISLNK(info.st_mode):
            raise SystemExit("remote path has a symlink/junction component")
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    info = _lstat(path)
    if info is None or not stat.S_ISREG(info.st_mode):
        raise SystemExit("remote path is not a regular file")
    return path


def _open_regular(path: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SystemExit(f"remote file cannot be opened safely: {type(exc).__name__}") from exc
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise SystemExit("remote file is not a regular file")
    return fd


def _sha_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha(path: str) -> str:
    fd = _open_regular(path)
    try:
        return _sha_fd(fd)
    finally:
        os.close(fd)


def _stream(path: str, offset: int, root: str | None) -> int:
    target = _assert_safe_path(path, root=root)
    fd = _open_regular(target)
    try:
        size = os.fstat(fd).st_size
        if offset < 0 or offset > size:
            raise SystemExit("offset is invalid")
        header = {
            "schema": HEADER_SCHEMA,
            "path": target,
            "size": size,
            "sha256": _sha_fd(fd),
            "offset": offset,
            "length": size - offset,
        }
        os.lseek(fd, offset, os.SEEK_SET)
        sys.stdout.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    finally:
        os.close(fd)
    return 0


def _read(path: str, max_bytes: int, root: str | None) -> int:
    target = _assert_safe_path(path, root=root)
    if max_bytes <= 0 or max_bytes > MAX_READ_BYTES:
        raise SystemExit("read bound is invalid")
    fd = _open_regular(target)
    try:
        size = os.fstat(fd).st_size
        if size > max_bytes:
            raise SystemExit("remote file exceeds the read bound")
        sha = _sha_fd(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        data = bytearray()
        while True:
            chunk = os.read(fd, min(65536, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > max_bytes:
                raise SystemExit("remote file exceeds the read bound")
    finally:
        os.close(fd)
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit("remote file is not valid UTF-8") from exc
    envelope = {
        "schema": READ_SCHEMA,
        "path": target,
        "size": size,
        "sha256": sha,
        "content": text,
    }
    sys.stdout.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def _metadata(path: str, root: str | None) -> int:
    target = _assert_safe_path(path, root=root)
    fd = _open_regular(target)
    try:
        size = os.fstat(fd).st_size
        if size <= 0 or size > 4 * 1024 * 1024 * 1024:
            raise SystemExit("remote artifact size is outside the bound")
        sha = _sha_fd(fd)
    finally:
        os.close(fd)
    payload = {
        "schema": METADATA_SCHEMA,
        "path": target,
        "size": size,
        "sha256": sha,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: v3_remote_fetch_helper.py stream <absolute-path> <offset> <root> | "
            "read <absolute-path> <max-bytes> <root> | "
            "metadata <absolute-path> <root>",
            file=sys.stderr,
        )
        return 2
    mode = args[0]
    try:
        if mode == "stream" and len(args) == 4:
            return _stream(args[1], int(args[2]), args[3] or None)
        if mode == "read" and len(args) == 4:
            return _read(args[1], int(args[2]), args[3] or None)
        if mode == "metadata" and len(args) == 3:
            return _metadata(args[1], args[2] or None)
    except ValueError:
        return 2
    print("invalid helper arguments", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
