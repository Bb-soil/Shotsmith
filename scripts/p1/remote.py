"""Thin, bounded OpenSSH command wrapper for P1 v3 offline orchestration.

This module does not implement a new transport. It builds explicit ``ssh`` /
``scp`` argv, requires strict host-key checking with an explicit known_hosts
file, and returns bounded output. No environment variable, secret value or
credential file is read. Actual SSH execution is never performed by the offline
test suite; tests inject a fake runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
MAX_TIMEOUT_SECONDS = 3600.0
MAX_OUTPUT_BYTES = 1024 * 1024
STREAM_HEADER_SCHEMA = "anime-video-studio/p1-v3-stream/v1"
_STREAM_HEADER_KEYS = {"schema", "path", "size", "sha256", "offset", "length"}

# Fixed remote receiver.  It never interpolates paths into a shell command and
# verifies the owned root, every path component, the existing-file identity and
# the payload hash before writing with O_EXCL/O_NOFOLLOW.
_REMOTE_RECEIVER = r"""
import hashlib, os, stat, sys
dest, root, expected, mode = sys.argv[1:5]
if "\x00" in dest or "\x00" in root:
    sys.exit(10)
root = os.path.realpath(root)
real = os.path.realpath(dest)
if real != dest:
    sys.exit(10)
try:
    if os.path.commonpath([root, real]) != root:
        sys.exit(10)
except ValueError:
    sys.exit(10)
cur = dest
while True:
    try:
        info = os.lstat(cur)
    except OSError:
        info = None
    if info is not None and stat.S_ISLNK(info.st_mode):
        sys.exit(11)
    parent = os.path.dirname(cur)
    if parent == cur:
        break
    cur = parent
data = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
if len(data) > 2 * 1024 * 1024:
    sys.exit(12)
digest = hashlib.sha256(data).hexdigest()
if expected and digest != expected:
    sys.exit(13)
if os.path.lexists(dest):
    info = os.lstat(dest)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        sys.exit(9)
    if mode == "same" and expected:
        h = hashlib.sha256()
        with open(dest, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() == expected:
            sys.exit(0)
    sys.exit(9)
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
flags |= getattr(os, "O_NOFOLLOW", 0)
fd = os.open(dest, flags, 0o600)
try:
    os.write(fd, data)
    os.fsync(fd)
finally:
    os.close(fd)
"""


class RemoteError(Exception):
    """Base class for bounded remote execution errors."""


class RemoteUnavailable(RemoteError):
    """The required OpenSSH executable is not available."""


class RemoteHostKeyError(RemoteError):
    """The host key is missing, changed or cannot be checked safely."""


class RemoteCommandError(RemoteError):
    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


class RemoteOutputTooLarge(RemoteError):
    """The command produced more output than the bounded allowance."""


class RemoteUnsupported(RemoteError):
    """A remote operation is intentionally disabled until it is safe."""


class RemoteCleanupError(RemoteError):
    """A process/reader could not be reaped and confirmed clean."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        timed_out: bool = False,
        reaped: bool = False,
        readers_done: bool = False,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.timed_out = timed_out
        self.reaped = reaped
        self.readers_done = readers_done


class RemoteStreamProtocolError(RemoteCommandError):
    """The typed stream header or payload protocol is invalid or mismatched."""


class RemoteStreamPartialError(RemoteCommandError):
    """A recoverable transport failure left a structurally validated partial.

    The target file exists only when the typed header was structurally valid
    (and matched any caller-provided expected path/size/hash/offset), the child
    was reaped, and its reader thread was confirmed stopped. Callers must still
    compare ``header`` with their exact expected artifact identity before
    treating the bytes as resumable.
    """

    def __init__(
        self,
        message: str,
        *,
        header: Mapping[str, Any],
        payload_size: int,
        target: Path,
        returncode: int | None,
        timed_out: bool,
    ) -> None:
        super().__init__(message, returncode=returncode)
        self.header = dict(header)
        self.payload_size = int(payload_size)
        self.target = Path(target)
        self.timed_out = bool(timed_out)


@dataclass(frozen=True)
class BoundedProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    reaped: bool
    readers_done: bool
    writer_done: bool = True


class RemoteStateError(RemoteError):
    """Durable worker state is malformed or unsafe."""


@dataclass(frozen=True)
class RemoteIdentity:
    host: str
    user: str
    port: int
    identity_file: Path | None
    known_hosts: Path
    standard_known_hosts: bool = False
    global_known_hosts: Path | None = None
    host_key_algorithms: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise RemoteError("remote host must be a non-empty string")
        if any(char.isspace() or ord(char) < 32 for char in self.host):
            raise RemoteError("remote host contains whitespace/control characters")
        if self.host.startswith("-"):
            raise RemoteError("remote host must not start with '-'")
        if not isinstance(self.user, str) or not self.user:
            raise RemoteError("remote user must be a non-empty string")
        if not all(char.isalnum() or char in "._-" for char in self.user):
            raise RemoteError("remote user contains unsupported characters")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise RemoteError("remote port must be an integer 1..65535")
        identity = Path(self.identity_file) if self.identity_file is not None else None
        known = Path(self.known_hosts)
        object.__setattr__(self, "identity_file", identity)
        object.__setattr__(self, "known_hosts", known)
        if identity is not None and not identity.is_file():
            raise RemoteError("identity file path does not exist")
        if not known.is_file():
            raise RemoteError("known_hosts path does not exist")

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def host_key_name(self) -> str:
        if self.standard_known_hosts and int(self.port) == 22:
            return self.host
        return f"[{self.host}]:{self.port}"


@dataclass(frozen=True)
class RemoteResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _bounded_text(data: bytes, limit: int) -> str:
    if len(data) > limit:
        data = data[:limit]
    return data.decode("utf-8", errors="replace")


def _validate_remote_path(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RemoteError(f"{name} must be a non-empty string")
    if value.startswith(("/", "\\")) or ".." in value.split("/") or "\\" in value:
        raise RemoteError(f"{name} must be a contained relative POSIX path")
    if any(ord(char) < 32 for char in value):
        raise RemoteError(f"{name} contains control characters")
    return value


def _validate_remote_args(args: Sequence[str]) -> tuple[str, ...]:
    if not args:
        raise RemoteError("remote command must not be empty")
    checked: list[str] = []
    for arg in args:
        if not isinstance(arg, str) or "\x00" in arg:
            raise RemoteError("remote argv entries must be NUL-free strings")
        if any(ord(char) < 32 and char not in "\t" for char in arg):
            raise RemoteError("remote argv contains control characters")
        checked.append(arg)
    return tuple(checked)


def _kill_tree(process: Any) -> None:
    """Kill the exact process and, when possible, its descendants."""

    pid = getattr(process, "pid", None)
    if os.name == "nt":
        if isinstance(pid, int) and pid > 0:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    check=False,
                )
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            process.kill()
        except OSError:
            pass
        return
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return
        except (OSError, ProcessLookupError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def run_bounded_process(
    argv: Sequence[str],
    *,
    timeout: float,
    max_output_bytes: int,
    input_bytes: bytes | None = None,
    popen: Any | None = None,
) -> BoundedProcessResult:
    """Run one process with a strict in-flight output cap and confirmed reap.

    This shared helper is used by SSH commands and local media decode. It never
    uses ``subprocess`` capture-all helpers. If reap/readers cannot be
    confirmed, it raises ``RemoteCleanupError`` instead of returning success.
    """

    if not (0 < timeout <= MAX_TIMEOUT_SECONDS):
        raise RemoteError("timeout must be bounded")
    if not (1024 <= max_output_bytes <= MAX_OUTPUT_BYTES):
        raise RemoteError("max_output_bytes must be bounded")
    if input_bytes is not None:
        if not isinstance(input_bytes, (bytes, bytearray)):
            raise RemoteError("input_bytes must be bytes")
        if len(input_bytes) > 2 * 1024 * 1024:
            raise RemoteError("input_bytes exceeds the bound")
    args = tuple(str(item) for item in argv)
    launcher = popen or subprocess.Popen
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        "bufsize": 0,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    process = launcher(args, **kwargs)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()
    readers: list[threading.Thread] = []

    def drain(name: str, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                target = buffers[name]
                if len(target) <= max_output_bytes:
                    target.extend(chunk[: max_output_bytes + 1 - len(target)])
                if len(target) > max_output_bytes:
                    overflow.set()
                    break
        finally:
            try:
                stream.close()
            except OSError:
                pass

    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None:
            continue
        thread = threading.Thread(target=drain, args=(name, stream), daemon=True)
        thread.start()
        readers.append(thread)

    writer_thread: threading.Thread | None = None
    if input_bytes is not None and process.stdin is not None:

        def write_input() -> None:
            try:
                process.stdin.write(bytes(input_bytes))
            except (OSError, ValueError, BrokenPipeError):
                pass
            finally:
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass

        writer_thread = threading.Thread(
            target=write_input,
            name="p1-remote-stdin-writer",
            daemon=True,
        )
        writer_thread.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    while process.poll() is None:
        if overflow.is_set():
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.02)
    if overflow.is_set() or timed_out:
        _kill_tree(process)
    # Closing stdin unblocks a writer that was waiting on a non-reading child.
    stdin = getattr(process, "stdin", None)
    if stdin is not None:
        try:
            stdin.close()
        except (OSError, ValueError):
            pass
    reaped = False
    try:
        process.wait(timeout=2.0)
        reaped = True
    except (subprocess.TimeoutExpired, OSError):
        _kill_tree(process)
        try:
            process.wait(timeout=2.0)
            reaped = True
        except (subprocess.TimeoutExpired, OSError):
            reaped = False
    for thread in readers:
        thread.join(timeout=2.0)
    readers_done = all(not thread.is_alive() for thread in readers)
    if not readers_done:
        _kill_tree(process)
        for thread in readers:
            thread.join(timeout=2.0)
        readers_done = all(not thread.is_alive() for thread in readers)
    writer_done = True
    if writer_thread is not None:
        writer_thread.join(timeout=2.0)
        writer_done = not writer_thread.is_alive()
        if not writer_done:
            _kill_tree(process)
            stdin = getattr(process, "stdin", None)
            if stdin is not None:
                try:
                    stdin.close()
                except (OSError, ValueError):
                    pass
            writer_thread.join(timeout=2.0)
            writer_done = not writer_thread.is_alive()
    returncode = 124 if timed_out else getattr(process, "returncode", None)
    if returncode is None:
        returncode = 124
    if not reaped or not readers_done or not writer_done:
        raise RemoteCleanupError(
            "process cleanup/reap/readers/writer were not confirmed clean",
            returncode=returncode,
            timed_out=timed_out,
            reaped=reaped,
            readers_done=readers_done,
        )
    if overflow.is_set():
        raise RemoteOutputTooLarge("process output exceeded the bound")
    return BoundedProcessResult(
        int(returncode),
        bytes(buffers["stdout"][: max_output_bytes + 1]),
        bytes(buffers["stderr"][: max_output_bytes + 1]),
        timed_out,
        reaped,
        readers_done,
        writer_done,
    )



def _validate_remote_absolute_path(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RemoteError(f"{name} must be a non-empty string")
    if not value.startswith("/") or ".." in value.split("/"):
        raise RemoteError(f"{name} must be a normalized absolute POSIX path")
    lowered = value.lower()
    if any(part in lowered for part in ("/.env", "/.ssh", "/.aws", "/.config")):
        raise RemoteError(f"{name} is a forbidden secret-like path")
    if any(ord(char) < 32 for char in value):
        raise RemoteError(f"{name} contains control characters")
    return value


def _validate_stream_header(raw: bytes, *, header_limit: int) -> dict[str, Any]:
    if len(raw) > header_limit:
        raise RemoteOutputTooLarge("stream header exceeds the bound")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RemoteStreamProtocolError("stream header is not UTF-8") from exc
    try:
        header = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RemoteStreamProtocolError("stream header is not valid JSON") from exc
    if not isinstance(header, dict) or set(header) != _STREAM_HEADER_KEYS:
        raise RemoteStreamProtocolError("stream header schema/fields are invalid")
    if header.get("schema") != STREAM_HEADER_SCHEMA:
        raise RemoteStreamProtocolError("stream header schema is unknown")
    path = header.get("path")
    size = header.get("size")
    sha = header.get("sha256")
    offset = header.get("offset")
    length = header.get("length")
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or "\x00" in path
        or ".." in path.split("/")
    ):
        raise RemoteStreamProtocolError("stream header path is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RemoteStreamProtocolError("stream header size is invalid")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in sha)
    ):
        raise RemoteStreamProtocolError("stream header sha256 is invalid")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset > size:
        raise RemoteStreamProtocolError("stream header offset is invalid")
    if isinstance(length, bool) or not isinstance(length, int) or length != size - offset:
        raise RemoteStreamProtocolError("stream header length is invalid")
    return header


def _validate_stream_expected(
    header: Mapping[str, Any],
    *,
    expected_path: str | None,
    expected_size: int | None,
    expected_sha256: str | None,
    expected_offset: int | None,
    payload_size: int | None = None,
) -> None:
    if (
        expected_path is None
        and expected_size is None
        and expected_sha256 is None
        and expected_offset is None
    ):
        return
    if (
        not isinstance(expected_path, str)
        or not expected_path
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in expected_sha256)
        or isinstance(expected_offset, bool)
        or not isinstance(expected_offset, int)
        or expected_offset < 0
        or expected_offset > expected_size
    ):
        raise RemoteStreamProtocolError("expected stream identity is incomplete")
    if (
        header.get("path") != expected_path
        or header.get("size") != expected_size
        or header.get("sha256") != expected_sha256
        or header.get("offset") != expected_offset
        or header.get("length") != expected_size - expected_offset
    ):
        raise RemoteStreamProtocolError("stream header identity mismatch")
    if payload_size is not None and payload_size > expected_size - expected_offset:
        raise RemoteStreamProtocolError("stream payload exceeds the expected length")


def _unlink_stream_target(target: Path) -> None:
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        raise RemoteCleanupError("stream partial could not be removed") from exc


def run_stream_to_file(
    argv: Sequence[str],
    target: Path,
    *,
    timeout: float,
    max_payload_bytes: int,
    header_limit: int = 4096,
    expected_path: str | None = None,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    expected_offset: int | None = None,
) -> tuple[bytes, int]:
    """Run one process and stream its typed header + payload into target.

    A stop-aware bounded queue reader and the main thread share one absolute
    timeout. A recoverable transport failure (timeout or child exit 255) keeps
    the target only when the typed header is structurally valid and any
    caller-provided path/size/hash/offset identity matches. The caller receives
    ``RemoteStreamPartialError`` with that metadata for an explicit resume
    decision. Malformed protocol, checksum mismatch and overflow delete the
    target. Cleanup must confirm process reap and reader stop or the operation
    fails closed; it never reports success while a reader/process remains.
    """

    import queue

    if not (0 < timeout <= MAX_TIMEOUT_SECONDS):
        raise RemoteError("timeout must be bounded")
    if not (1 <= max_payload_bytes <= 1024 * 1024 * 1024):
        raise RemoteError("max_payload_bytes must be bounded")
    if header_limit <= 0 or header_limit > MAX_OUTPUT_BYTES:
        raise RemoteError("header_limit must be bounded")
    expected_values = (
        expected_path,
        expected_size,
        expected_sha256,
        expected_offset,
    )
    if any(value is not None for value in expected_values) and any(
        value is None for value in expected_values
    ):
        raise RemoteStreamProtocolError("expected stream identity is incomplete")
    expected_supplied = all(value is not None for value in expected_values)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "bufsize": 0,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(tuple(str(item) for item in argv), **kwargs)
    assert process.stdout is not None
    chunks: queue.Queue[Any] = queue.Queue(maxsize=64)
    stop = threading.Event()

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                chunks.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def reader() -> None:
        try:
            while not stop.is_set():
                chunk = process.stdout.read(65536)
                if not chunk:
                    put(None)
                    return
                if not put(chunk):
                    return
        except BaseException as exc:  # surfaced to main; never swallowed
            put(exc)

    thread = threading.Thread(
        target=reader,
        name="p1-remote-stream-reader",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + timeout
    header_raw = bytearray()
    header: dict[str, Any] | None = None
    header_done = False
    payload = 0
    timed_out = False
    write_failed = False
    failure: BaseException | None = None
    reaped = False
    readers_done = False
    cleanup_ok = False
    try:
        try:
            with target.open("wb") as handle:
                try:
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            timed_out = True
                            failure = RemoteCommandError("stream deadline expired")
                            break
                        try:
                            chunk = chunks.get(timeout=remaining)
                        except queue.Empty:
                            timed_out = True
                            failure = RemoteCommandError("stream deadline expired")
                            break
                        if chunk is None:
                            break
                        if isinstance(chunk, BaseException):
                            failure = RemoteCommandError(
                                f"stream reader failed: {type(chunk).__name__}"
                            )
                            break
                        data = bytes(chunk)
                        if not header_done:
                            if b"\n" not in data:
                                header_raw.extend(data)
                                if len(header_raw) > header_limit:
                                    failure = RemoteOutputTooLarge(
                                        "stream header exceeds the bound"
                                    )
                                    break
                                continue
                            line, data = data.split(b"\n", 1)
                            header_raw.extend(line)
                            if len(header_raw) > header_limit:
                                failure = RemoteOutputTooLarge(
                                    "stream header exceeds the bound"
                                )
                                break
                            try:
                                header = _validate_stream_header(
                                    bytes(header_raw), header_limit=header_limit
                                )
                            except RemoteError as exc:
                                failure = exc
                                break
                            header_done = True
                        if data:
                            payload += len(data)
                            if payload > max_payload_bytes:
                                failure = RemoteOutputTooLarge(
                                    "stream payload exceeds the bound"
                                )
                                break
                            if header is not None and payload > header["length"]:
                                failure = RemoteStreamProtocolError(
                                    "stream payload exceeds the header length"
                                )
                                break
                            try:
                                handle.write(data)
                            except OSError:
                                write_failed = True
                                failure = RemoteCommandError(
                                    "stream local write failed"
                                )
                                break
                            if time.monotonic() >= deadline:
                                timed_out = True
                                failure = RemoteCommandError("stream deadline expired")
                                break
                    if failure is None and not header_done:
                        failure = RemoteCommandError(
                            "stream ended before complete header"
                        )
                finally:
                    try:
                        handle.flush()
                        os.fsync(handle.fileno())
                    except (OSError, ValueError):
                        pass
        except BaseException as exc:
            if failure is None:
                failure = exc
        if failure is None:
            try:
                process.wait(timeout=2.0)
                reaped = True
            except (subprocess.TimeoutExpired, OSError):
                failure = RemoteCommandError(
                    "stream helper did not exit after EOF"
                )
            if failure is None and process.returncode != 0:
                returncode = (
                    int(process.returncode)
                    if isinstance(process.returncode, int)
                    else 1
                )
                failure = RemoteCommandError(
                    f"stream helper exited {returncode}",
                    returncode=returncode,
                )
    finally:
        stop.set()
        if process.poll() is None:
            _kill_tree(process)
        try:
            process.wait(timeout=2.0)
            reaped = True
        except (subprocess.TimeoutExpired, OSError):
            _kill_tree(process)
            try:
                process.wait(timeout=2.0)
                reaped = True
            except (subprocess.TimeoutExpired, OSError):
                reaped = False
        try:
            process.stdout.close()
        except (OSError, ValueError):
            pass
        thread.join(timeout=2.0)
        if thread.is_alive():
            _kill_tree(process)
            try:
                process.stdout.close()
            except (OSError, ValueError):
                pass
            thread.join(timeout=2.0)
        readers_done = not thread.is_alive()
        cleanup_ok = reaped and readers_done

    if not cleanup_ok:
        _unlink_stream_target(target)
        raise RemoteCleanupError(
            "stream process/reader cleanup was not confirmed",
            returncode=(
                int(process.returncode)
                if isinstance(process.returncode, int)
                else None
            ),
            timed_out=timed_out,
            reaped=reaped,
            readers_done=readers_done,
        )

    returncode = (
        124
        if timed_out
        else (
            int(process.returncode)
            if isinstance(process.returncode, int)
            else 124
        )
    )
    if failure is not None:
        if isinstance(failure, (RemoteOutputTooLarge, RemoteStreamProtocolError)):
            _unlink_stream_target(target)
            raise failure
        recoverable = (timed_out or returncode == 255) and not write_failed
        if (
            recoverable
            and expected_supplied
            and header_done
            and header is not None
            and payload > 0
        ):
            try:
                _validate_stream_expected(
                    header,
                    expected_path=expected_path,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    expected_offset=expected_offset,
                    payload_size=payload,
                )
            except RemoteError:
                _unlink_stream_target(target)
                raise
            try:
                confirmed = target.is_file() and target.stat().st_size == payload
            except OSError as exc:
                _unlink_stream_target(target)
                raise RemoteCleanupError(
                    "stream partial size could not be confirmed"
                ) from exc
            if not confirmed:
                _unlink_stream_target(target)
                raise RemoteCleanupError("stream partial size was not confirmed")
            raise RemoteStreamPartialError(
                str(failure),
                header=header,
                payload_size=payload,
                target=target,
                returncode=returncode,
                timed_out=timed_out,
            ) from failure
        _unlink_stream_target(target)
        raise failure

    if not header_done or header is None:
        _unlink_stream_target(target)
        raise RemoteStreamProtocolError("stream header is missing")
    try:
        _validate_stream_expected(
            header,
            expected_path=expected_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            expected_offset=expected_offset,
            payload_size=payload,
        )
    except RemoteError:
        _unlink_stream_target(target)
        raise
    if payload != header["length"]:
        _unlink_stream_target(target)
        raise RemoteStreamProtocolError("stream payload length mismatch")
    return bytes(header_raw), payload


def _remaining_until(deadline: float | None, *, action: str) -> float | None:
    """Return positive remaining seconds or fail before starting work."""

    if deadline is None:
        return None
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
    ):
        raise RemoteError("deadline_monotonic must be a finite number")
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0:
        raise RemoteCommandError(f"deadline expired before {action}")
    return remaining


class RemoteRunner:
    """Build and optionally execute bounded OpenSSH argv."""

    def __init__(
        self,
        *,
        ssh_path: str | None = None,
        scp_path: str | None = None,
        ssh_keygen_path: str | None = None,
        runner: Any | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        self.ssh_path = ssh_path or self._which("ssh")
        self.scp_path = scp_path or self._which("scp")
        self.ssh_keygen_path = ssh_keygen_path or self._which("ssh-keygen")
        self._runner = runner  # test-only injection; production uses bounded Popen
        if not (0 < timeout_seconds <= MAX_TIMEOUT_SECONDS):
            raise RemoteError("timeout_seconds must be bounded")
        if not (1024 <= max_output_bytes <= MAX_OUTPUT_BYTES):
            raise RemoteError("max_output_bytes must be bounded")
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = int(max_output_bytes)

    @staticmethod
    def _which(name: str) -> str | None:
        import shutil

        return shutil.which(name)

    def require_tools(self) -> None:
        missing = [
            name
            for name, value in (
                ("ssh", self.ssh_path),
                ("scp", self.scp_path),
                ("ssh-keygen", self.ssh_keygen_path),
            )
            if not value
        ]
        if missing:
            raise RemoteUnavailable(f"missing OpenSSH tools: {', '.join(missing)}")

    def _ssh_options(self, identity: RemoteIdentity) -> list[str]:
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={identity.known_hosts}",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            "-p",
            str(identity.port),
        ]
        if identity.global_known_hosts is not None:
            options.extend(["-o", f"GlobalKnownHostsFile={identity.global_known_hosts}"])
        if identity.host_key_algorithms:
            options.extend(["-o", f"HostKeyAlgorithms={identity.host_key_algorithms}"])
        if identity.identity_file is not None:
            options.extend(["-i", str(identity.identity_file)])
        return options

    def host_key_argv(self, identity: RemoteIdentity) -> tuple[str, ...]:
        if not self.ssh_keygen_path:
            raise RemoteUnavailable("ssh-keygen is not available")
        return (
            self.ssh_keygen_path,
            "-F",
            identity.host_key_name,
            "-f",
            str(identity.known_hosts),
        )

    def check_host_key(
        self, identity: RemoteIdentity, *, timeout: float | None = None
    ) -> bool:
        if timeout is not None:
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
            ):
                raise RemoteError("host-key timeout must be finite")
            if float(timeout) <= 0:
                raise RemoteCommandError("deadline expired before host-key check")
            timeout = float(timeout)
        host_timeout = 10.0 if timeout is None else min(10.0, timeout)
        argv = self.host_key_argv(identity)
        result = self.run_argv(
            argv, timeout=host_timeout, max_output_bytes=16 * 1024
        )
        if result.timed_out:
            raise RemoteHostKeyError("host-key lookup timed out")
        if result.returncode == 0 and result.stdout.strip():
            return True
        if result.returncode in (0, 1) and not result.stdout.strip():
            raise RemoteHostKeyError(
                "host key is absent from the explicit known_hosts file"
            )
        raise RemoteHostKeyError(
            f"host-key lookup failed with exit {result.returncode}"
        )

    def ssh_argv(
        self,
        identity: RemoteIdentity,
        remote_args: Sequence[str],
        *,
        extra_options: Iterable[str] = (),
    ) -> tuple[str, ...]:
        if not self.ssh_path:
            raise RemoteUnavailable("ssh is not available")
        command = _validate_remote_args(remote_args)
        return (
            self.ssh_path,
            *self._ssh_options(identity),
            *tuple(extra_options),
            identity.destination,
            shlex.join(command),
        )

    def scp_argv(
        self,
        identity: RemoteIdentity,
        remote_path: str,
        local_path: Path,
    ) -> tuple[str, ...]:
        if not self.scp_path:
            raise RemoteUnavailable("scp is not available")
        if identity.identity_file is None:
            raise RemoteUnavailable("scp requires an explicit identity file")
        remote = _validate_remote_path(remote_path, "remote_path")
        local = Path(local_path)
        if local.exists() and local.is_symlink():
            raise RemoteError("local scp target must not be a symlink")
        return (
            self.scp_path,
            "-P",
            str(identity.port),
            "-i",
            str(identity.identity_file),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={identity.known_hosts}",
            "-o",
            "ConnectTimeout=10",
            f"{identity.destination}:{remote}",
            str(local),
        )

    def run_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        max_output_bytes: int | None = None,
    ) -> RemoteResult:
        argv_tuple = tuple(str(item) for item in argv)
        limit = self.max_output_bytes if max_output_bytes is None else int(max_output_bytes)
        if not (0 < limit <= MAX_OUTPUT_BYTES):
            raise RemoteError("max_output_bytes must be bounded")
        timeout_value = self.timeout_seconds if timeout is None else float(timeout)
        if not (0 < timeout_value <= MAX_TIMEOUT_SECONDS):
            raise RemoteError("timeout must be bounded")
        if self._runner is not None:
            return self._run_injected(argv_tuple, timeout_value, limit)
        return self._run_bounded_popen(argv_tuple, timeout_value, limit)

    def _run_injected(
        self, argv: tuple[str, ...], timeout: float, limit: int
    ) -> RemoteResult:
        """Test-only fake-runner path; no temp output files or unbounded capture."""

        timed_out = False
        try:
            completed = self._runner(argv, timeout=timeout, check=False)
            returncode = int(getattr(completed, "returncode", 1))
            stdout_raw = getattr(completed, "stdout", b"") or b""
            stderr_raw = getattr(completed, "stderr", b"") or b""
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            stdout_raw = b""
            stderr_raw = b""
        stdout_bytes = stdout_raw.encode() if isinstance(stdout_raw, str) else bytes(stdout_raw)
        stderr_bytes = stderr_raw.encode() if isinstance(stderr_raw, str) else bytes(stderr_raw)
        if len(stdout_bytes) > limit or len(stderr_bytes) > limit:
            raise RemoteOutputTooLarge("remote command output exceeded the bound")
        return RemoteResult(
            argv,
            returncode,
            _bounded_text(stdout_bytes, limit),
            _bounded_text(stderr_bytes, limit),
            timed_out,
        )

    def _run_bounded_popen(
        self, argv: tuple[str, ...], timeout: float, limit: int
    ) -> RemoteResult:
        result = run_bounded_process(
            argv,
            timeout=timeout,
            max_output_bytes=limit,
        )
        return RemoteResult(
            argv,
            result.returncode,
            _bounded_text(result.stdout, limit),
            _bounded_text(result.stderr, limit),
            result.timed_out,
        )

    def run_remote(
        self,
        identity: RemoteIdentity,
        remote_args: Sequence[str],
        *,
        extra_options: Iterable[str] = (),
        timeout: float | None = None,
        deadline_monotonic: float | None = None,
    ) -> RemoteResult:
        deadline = (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else time.monotonic() + (self.timeout_seconds if timeout is None else float(timeout))
        )
        host_remaining = _remaining_until(deadline, action="host-key check")
        self.check_host_key(identity, timeout=host_remaining)
        argv = self.ssh_argv(identity, remote_args, extra_options=extra_options)
        command_remaining = _remaining_until(deadline, action="remote command")
        return self.run_argv(argv, timeout=command_remaining)

    def run_remote_with_input(
        self,
        identity: RemoteIdentity,
        remote_args: Sequence[str],
        input_bytes: bytes,
        *,
        timeout: float | None = None,
        deadline_monotonic: float | None = None,
    ) -> RemoteResult:
        deadline = (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else time.monotonic() + (self.timeout_seconds if timeout is None else float(timeout))
        )
        host_remaining = _remaining_until(deadline, action="host-key check")
        self.check_host_key(identity, timeout=host_remaining)
        argv = self.ssh_argv(identity, remote_args)
        command_remaining = _remaining_until(deadline, action="remote command")
        result = run_bounded_process(
            argv,
            timeout=command_remaining,
            max_output_bytes=self.max_output_bytes,
            input_bytes=input_bytes,
        )
        return RemoteResult(
            argv,
            result.returncode,
            _bounded_text(result.stdout, self.max_output_bytes),
            _bounded_text(result.stderr, self.max_output_bytes),
            result.timed_out,
        )

    def put_file(
        self,
        identity: RemoteIdentity,
        remote_path: str,
        local_path: Path,
        *,
        timeout: float = 60.0,
        root: str | None = None,
        allow_same_hash: bool = False,
        deadline_monotonic: float | None = None,
    ) -> None:
        deadline = (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else time.monotonic() + float(timeout)
        )
        _remaining_until(deadline, action="file staging")
        remote = _validate_remote_absolute_path(remote_path, "remote_path")
        owned_root = root or str(__import__("pathlib").PurePosixPath(remote).parent)
        _validate_remote_absolute_path(owned_root, "remote_root")
        local = Path(local_path)
        if local.is_symlink() or not local.is_file():
            raise RemoteError("local file must be a regular file")
        size = local.stat().st_size
        if size <= 0 or size > 2 * 1024 * 1024:
            raise RemoteError("local file size is outside the staging bound")
        data = local.read_bytes()
        expected = __import__("hashlib").sha256(data).hexdigest()
        # Keep the fixed receiver as one NUL/control-free argv token. Passing
        # the multiline source directly would be rejected as control characters
        # by the generic remote argv validator.
        receiver = "exec(" + repr(_REMOTE_RECEIVER) + ")"
        result = self.run_remote_with_input(
            identity,
            (
                "python3",
                "-c",
                receiver,
                remote,
                owned_root,
                expected,
                "same" if allow_same_hash else "noclobber",
            ),
            data,
            timeout=timeout,
            deadline_monotonic=deadline,
        )
        if result.timed_out or result.returncode != 0:
            raise RemoteCommandError(
                f"staged file upload failed with exit {result.returncode}",
                returncode=result.returncode,
            )

    def read_remote_exact_json(
        self,
        identity: RemoteIdentity,
        *,
        helper_path: str,
        remote_path: str,
        root: str,
        max_bytes: int = 256 * 1024,
        timeout: float = 15.0,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """Read one small remote JSON file through the staged read helper."""

        deadline = (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else time.monotonic() + float(timeout)
        )
        _remaining_until(deadline, action="remote JSON read")
        helper = _validate_remote_absolute_path(helper_path, "helper_path")
        remote = _validate_remote_absolute_path(remote_path, "remote_path")
        owned_root = _validate_remote_absolute_path(root, "remote_root")
        if PurePosixPath(helper).name != "v3_remote_fetch_helper.py":
            raise RemoteError("helper path is not the pinned helper")
        if max_bytes <= 0 or max_bytes > 256 * 1024:
            raise RemoteError("read bound is invalid")
        result = self.run_remote(
            identity,
            ("python3", helper, "read", remote, str(max_bytes), owned_root),
            timeout=timeout,
            deadline_monotonic=deadline,
        )
        if result.timed_out or result.returncode != 0:
            raise RemoteCommandError(
                f"remote read failed with exit {result.returncode}",
                returncode=result.returncode,
            )
        try:
            envelope = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RemoteStateError("remote read envelope is not valid JSON") from exc
        if (
            not isinstance(envelope, dict)
            or envelope.get("schema") != "anime-video-studio/p1-v3-read/v1"
            or envelope.get("path") != remote
            or not isinstance(envelope.get("content"), str)
            or not isinstance(envelope.get("size"), int)
            or not isinstance(envelope.get("sha256"), str)
        ):
            raise RemoteStateError("remote read envelope identity is invalid")
        if len(envelope["content"].encode("utf-8")) != envelope["size"]:
            raise RemoteStateError("remote read envelope size is inconsistent")
        if hashlib.sha256(envelope["content"].encode("utf-8")).hexdigest() != envelope["sha256"]:
            raise RemoteStateError("remote read envelope hash is inconsistent")
        try:
            payload = json.loads(envelope["content"])
        except json.JSONDecodeError as exc:
            raise RemoteStateError("remote file is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RemoteStateError("remote file must be a JSON object")
        return payload

    def stream_remote_to_file(
        self,
        identity: RemoteIdentity,
        remote_args: Sequence[str],
        target: Path,
        *,
        timeout: float,
        max_payload_bytes: int,
        header_limit: int = 4096,
        expected_path: str | None = None,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
        expected_offset: int | None = None,
        deadline_monotonic: float | None = None,
    ) -> tuple[bytes, int]:
        deadline = (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else time.monotonic() + float(timeout)
        )
        host_remaining = _remaining_until(deadline, action="host-key check")
        self.check_host_key(identity, timeout=host_remaining)
        argv = self.ssh_argv(identity, remote_args)
        command_remaining = _remaining_until(deadline, action="stream command")
        return run_stream_to_file(
            argv,
            target,
            timeout=command_remaining,
            max_payload_bytes=max_payload_bytes,
            header_limit=header_limit,
            expected_path=expected_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            expected_offset=expected_offset,
        )

    def start_background(
        self,
        identity: RemoteIdentity,
        remote_args: Sequence[str],
        *,
        log_path: str,
        pid_path: str,
    ) -> str:
        """Disabled until a durable ownership/attach protocol exists.

        The previous implementation could overwrite an existing PID/log and had
        no atomic ownership handshake. It is intentionally unreachable now; no
        process, runner or network call is started.
        """

        del identity, remote_args, log_path, pid_path
        raise RemoteUnsupported(
            "BACKGROUND_OWNERSHIP_NOT_IMPLEMENTED: no process was started"
        )

    def read_worker_state(
        self,
        identity: RemoteIdentity,
        path: str,
        *,
        helper_path: str | None = None,
        root: str | None = None,
        max_bytes: int = 256 * 1024,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """Read durable worker state through the staged bounded helper.

        A shell ``cat`` redirection is intentionally not used: it cannot reject
        a symlinked parent directory and has no byte bound.  Without an explicit
        staged helper this method fails closed.
        """

        if helper_path is None or root is None:
            raise RemoteUnsupported(
                "REMOTE_STATE_REQUIRES_STAGED_HELPER: no shell cat fallback"
            )
        return self.read_remote_exact_json(
            identity,
            helper_path=helper_path,
            remote_path=path,
            root=root,
            max_bytes=max_bytes,
            deadline_monotonic=deadline_monotonic,
        )

    def scp_fetch(
        self,
        identity: RemoteIdentity,
        remote_path: str,
        local_path: Path,
        *,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Disabled until a streaming, byte-capped SCP/FTP protocol exists.

        Plain ``scp`` writes the whole remote file before the caller can enforce
        ``max_bytes``. No host-key check, process or network call is made here.
        """

        del identity, remote_path, local_path, max_bytes
        raise RemoteUnsupported(
            "SCP_STREAM_BOUNDED_PROTOCOL_NOT_IMPLEMENTED: no remote fetch was started"
        )


def pid_alive(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            return True
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        return False


def write_worker_state(path: Path | str, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=target.name + ".",
        suffix=".tmp",
        delete=False,
        newline="\n",
    )
    temp_name = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        try:
            temp_name.unlink(missing_ok=True)
        except OSError:
            pass


def read_worker_state(path: Path | str) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteStateError(f"worker state is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RemoteStateError("worker state must be an object")
    return payload


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "BoundedProcessResult",
    "RemoteCleanupError",
    "RemoteCommandError",
    "RemoteError",
    "RemoteHostKeyError",
    "RemoteIdentity",
    "RemoteOutputTooLarge",
    "RemoteResult",
    "RemoteRunner",
    "RemoteStateError",
    "RemoteStreamPartialError",
    "RemoteStreamProtocolError",
    "RemoteUnavailable",
    "RemoteUnsupported",
    "STREAM_HEADER_SCHEMA",
    "pid_alive",
    "read_worker_state",
    "run_bounded_process",
    "run_stream_to_file",
    "write_worker_state",
]
