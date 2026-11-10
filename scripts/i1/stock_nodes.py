"""Single read-only stock custom_nodes validator for I1 inspect and setup.

The same source string is executed by the remote inspector and embedded in the
rendered POSIX setup script.  It never imports or executes node code and never
writes to the checkout.  It compares the working tree against the pinned HEAD
tree, so index flags such as assume-unchanged/skip-worktree cannot hide a
changed tracked file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

SCHEMA = "anime-video-studio/i1-stock-nodes/v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 4096
MAX_VIOLATIONS = 64
MAX_REPORTED = 200
GIT_TIMEOUT = 30.0

# The validator is a standalone Python program.  Keep it compatible with the
# system python3 used before the I1 venv is created.
VALIDATOR_SOURCE = r'''
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys

SCHEMA = "anime-video-studio/i1-stock-nodes/v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 4096
MAX_VIOLATIONS = 64
MAX_REPORTED = 200
GIT_TIMEOUT = 30.0
CACHE_RE = re.compile(r"^(.+)\.cpython-\d+(?:\.opt-\d+)?\.pyc$")

result = {
    "schema": SCHEMA,
    "ok": False,
    "head": None,
    "expected": None,
    "object_format": None,
    "tracked": [],
    "tracked_total": 0,
    "cache_unverified": [],
    "cache_dirs": [],
    "violations": [],
    "truncated": False,
}


class _Stop(Exception):
    pass


def fail(code, path, message):
    if len(result["violations"]) >= MAX_VIOLATIONS:
        result["truncated"] = True
        return
    result["violations"].append(
        {"code": str(code), "path": str(path), "message": str(message)}
    )


def safe_rel(rel):
    if not rel.startswith("custom_nodes/"):
        return False
    if "\\" in rel or "\x00" in rel:
        return False
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    return True


def cache_source(rel, tracked):
    parts = rel.split("/")
    if len(parts) < 3 or parts[-2] != "__pycache__":
        return None
    if parts.count("__pycache__") != 1:
        return None
    match = CACHE_RE.match(parts[-1])
    if match is None:
        return None
    source = "/".join(parts[:-2] + [match.group(1) + ".py"])
    return source if source in tracked else None


def cache_dir(rel, tracked):
    parts = rel.split("/")
    if not parts or parts[-1] != "__pycache__" or parts.count("__pycache__") != 1:
        return False
    parent = "/".join(parts[:-1])
    for source in tracked:
        if not source.endswith(".py"):
            continue
        if source.rsplit("/", 1)[0] == parent:
            return True
    return False


def hash_file(path, fmt):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not_regular")
        if info.st_size > MAX_FILE_BYTES:
            raise ValueError("file_too_large")
        digest = hashlib.new(fmt)
        digest.update(b"blob " + str(info.st_size).encode("ascii") + b"\0")
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise ValueError("file_too_large")
            digest.update(chunk)
        if total != info.st_size:
            raise ValueError("file_changed_during_read")
        return digest.hexdigest()
    finally:
        os.close(fd)


def validate():
    raw_comfy = sys.argv[1]
    expected = str(sys.argv[2]).strip().lower()
    result["expected"] = expected
    if re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", expected) is None:
        fail("revision", "", "expected revision is not a full object id")
        raise _Stop

    if os.path.islink(raw_comfy):
        fail("symlink", ".", "ComfyUI path is a symlink")
        raise _Stop
    if not os.path.isdir(raw_comfy):
        fail("missing", ".", "ComfyUI directory is missing")
        raise _Stop
    comfy = os.path.realpath(raw_comfy)

    dotgit = os.path.join(comfy, ".git")
    if os.path.islink(dotgit):
        fail("symlink", ".git", ".git is a symlink")
        raise _Stop
    if not os.path.isdir(dotgit):
        fail("git", ".git", ".git is not a regular directory")
        raise _Stop

    def git(*args):
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env["GIT_NO_REPLACE_OBJECTS"] = "1"
        proc = subprocess.run(
            ["git", "-C", comfy] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GIT_TIMEOUT,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError("git command failed")
        return proc.stdout

    top = git("rev-parse", "--show-toplevel").decode("utf-8", "replace").strip()
    if os.path.realpath(top) != comfy:
        fail("git", "", "git top-level is not the ComfyUI directory")
        raise _Stop

    head = git("rev-parse", "--verify", "HEAD").decode("ascii", "replace").strip().lower()
    result["head"] = head
    if head != expected:
        fail("revision", "", "HEAD does not match the pinned revision")
        raise _Stop

    fmt = git("rev-parse", "--show-object-format").decode("ascii", "replace").strip().lower()
    if fmt not in ("sha1", "sha256"):
        fail("git", "", "unsupported git object format")
        raise _Stop
    result["object_format"] = fmt

    raw_tree = git(
        "ls-tree", "-r", "-z", "--full-tree", expected, "--", "custom_nodes"
    )
    tracked = {}
    for record in raw_tree.split(b"\0"):
        if not record:
            continue
        try:
            meta, raw_path = record.split(b"\t", 1)
            mode, kind, oid = meta.split(b" ", 2)
        except ValueError:
            fail("git", "", "malformed ls-tree record")
            continue
        rel = raw_path.decode("utf-8", "replace")
        if not safe_rel(rel):
            fail("unsafe_path", rel, "tree path is outside custom_nodes")
            continue
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            fail("type", rel, "tracked entry is not a regular file blob")
            continue
        tracked[rel] = oid.decode("ascii", "replace").lower()

    result["tracked_total"] = len(tracked)
    result["tracked"] = sorted(tracked)[:MAX_REPORTED]
    if len(tracked) > MAX_REPORTED:
        result["truncated"] = True

    allowed_dirs = set()
    for rel in tracked:
        parts = rel.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            allowed_dirs.add("/".join(parts[:index]))

    for rel, expected_oid in tracked.items():
        parts = rel.split("/")
        current = comfy
        usable = True
        for index, part in enumerate(parts):
            current = os.path.join(current, part)
            try:
                info = os.lstat(current)
            except OSError:
                fail("missing", rel, "tracked path is missing")
                usable = False
                break
            if stat.S_ISLNK(info.st_mode):
                fail("symlink", rel, "tracked path contains a symlink")
                usable = False
                break
            if index < len(parts) - 1:
                if not stat.S_ISDIR(info.st_mode):
                    fail("type", rel, "tracked parent is not a directory")
                    usable = False
                    break
            elif not stat.S_ISREG(info.st_mode):
                fail("type", rel, "tracked entry is not a regular file")
                usable = False
                break
        if not usable:
            continue
        try:
            actual_oid = hash_file(current, fmt)
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.ELOOP:
                fail("symlink", rel, "tracked file is a symlink")
            else:
                fail("unreadable", rel, "tracked file cannot be read")
        except ValueError as exc:
            fail("file", rel, str(exc))
        else:
            if actual_oid != expected_oid:
                fail("content", rel, "tracked content differs from the pinned blob")

    nodes = os.path.join(comfy, "custom_nodes")
    if os.path.lexists(nodes):
        if os.path.islink(nodes):
            fail("symlink", "custom_nodes", "custom_nodes is a symlink")
            raise _Stop
        if not os.path.isdir(nodes):
            fail("type", "custom_nodes", "custom_nodes is not a directory")
            raise _Stop

        seen_dirs = set()

        def walk_error(exc):
            raise exc

        entries = 0
        stopped = False
        for dirpath, dirnames, filenames in os.walk(
            nodes, followlinks=False, onerror=walk_error
        ):
            rel_dir = os.path.relpath(dirpath, comfy).replace(os.sep, "/")
            for name in list(dirnames):
                entries += 1
                if entries > MAX_ENTRIES:
                    fail("limit", "custom_nodes", "custom_nodes entry limit exceeded")
                    stopped = True
                    break
                full = os.path.join(dirpath, name)
                rel = rel_dir + "/" + name
                try:
                    info = os.lstat(full)
                except OSError:
                    fail("unreadable", rel, "entry cannot be inspected")
                    dirnames.remove(name)
                    continue
                if stat.S_ISLNK(info.st_mode):
                    fail("symlink", rel, "untracked symlink directory")
                    dirnames.remove(name)
                elif not stat.S_ISDIR(info.st_mode):
                    fail("type", rel, "entry is not a directory")
                    dirnames.remove(name)
                else:
                    seen_dirs.add(rel)
            if stopped:
                break
            for name in filenames:
                entries += 1
                if entries > MAX_ENTRIES:
                    fail("limit", "custom_nodes", "custom_nodes entry limit exceeded")
                    stopped = True
                    break
                full = os.path.join(dirpath, name)
                rel = rel_dir + "/" + name
                try:
                    info = os.lstat(full)
                except OSError:
                    fail("unreadable", rel, "entry cannot be inspected")
                    continue
                if stat.S_ISLNK(info.st_mode):
                    fail("symlink", rel, "untracked symlink")
                elif not stat.S_ISREG(info.st_mode):
                    fail("type", rel, "untracked non-regular entry")
                elif rel in tracked:
                    continue
                elif cache_source(rel, tracked) is not None:
                    if len(result["cache_unverified"]) < MAX_REPORTED:
                        result["cache_unverified"].append(rel)
                    else:
                        result["truncated"] = True
                else:
                    fail("untracked", rel, "untracked entry under custom_nodes")
            if stopped:
                break

        for rel in sorted(seen_dirs):
            if rel in allowed_dirs:
                continue
            if cache_dir(rel, tracked):
                if len(result["cache_dirs"]) < MAX_REPORTED:
                    result["cache_dirs"].append(rel)
                else:
                    result["truncated"] = True
                continue
            fail("untracked", rel, "untracked directory under custom_nodes")


try:
    validate()
except _Stop:
    pass
except Exception as exc:
    fail("validator_error", "", type(exc).__name__)

result["ok"] = not result["violations"]
if result["violations"]:
    summary = "; ".join(
        (item.get("code", "error") + (":" + item["path"] if item.get("path") else ""))
        for item in result["violations"][:5]
    )
    sys.stderr.write("custom_nodes: " + summary + "\n")
if result["cache_unverified"] or result["cache_dirs"]:
    sys.stderr.write("custom_nodes: runtime cache present; contents not verified\n")
sys.stdout.write(json.dumps(result, sort_keys=True, ensure_ascii=True) + "\n")
raise SystemExit(0 if result["ok"] else 1)
'''


def _error(message: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "head": None,
        "expected": None,
        "object_format": None,
        "tracked": [],
        "tracked_total": 0,
        "cache_unverified": [],
        "cache_dirs": [],
        "violations": [
            {"code": "validator_error", "path": "", "message": message}
        ],
        "truncated": False,
    }


def parse_output(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return _error("validator returned no output")
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return _error("validator output is not valid JSON")
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return _error("validator schema is invalid")
    for key in ("tracked", "cache_unverified", "cache_dirs", "violations"):
        if not isinstance(data.get(key), list):
            return _error("validator field %s is invalid" % key)
    if not isinstance(data.get("ok"), bool):
        return _error("validator ok field is invalid")
    result = dict(data)
    result["ok"] = bool(result["ok"]) and not result["violations"]
    return result


def describe_violations(data: dict[str, Any], limit: int = 3) -> str:
    items = data.get("violations")
    if not isinstance(items, list) or not items:
        return "validator error"
    parts = []
    for item in items[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "error")
        path = str(item.get("path") or "")
        parts.append(code + (":" + path if path else ""))
    return "; ".join(parts) if parts else "validator error"


def remote_argv(comfy_dir: str, expected_commit: str) -> tuple[str, ...]:
    return (
        "python3",
        "-B",
        "-c",
        "exec(" + repr(VALIDATOR_SOURCE) + ")",
        str(comfy_dir),
        str(expected_commit),
    )


def run_local(
    comfy_dir: str,
    expected_commit: str,
    *,
    timeout: float = 60.0,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "exec(" + repr(VALIDATOR_SOURCE) + ")",
            str(comfy_dir),
            str(expected_commit),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc, parse_output(proc.stdout)


__all__ = [
    "MAX_ENTRIES",
    "MAX_FILE_BYTES",
    "MAX_REPORTED",
    "MAX_VIOLATIONS",
    "SCHEMA",
    "VALIDATOR_SOURCE",
    "describe_violations",
    "parse_output",
    "remote_argv",
    "run_local",
]
