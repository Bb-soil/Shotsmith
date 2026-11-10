from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts.p1 import remote as remote_module
from scripts.p1.remote import (
    RemoteCleanupError,
    RemoteCommandError,
    RemoteError,
    RemoteHostKeyError,
    RemoteIdentity,
    RemoteOutputTooLarge,
    RemoteRunner,
    RemoteStreamPartialError,
    RemoteStreamProtocolError,
    RemoteUnsupported,
    run_bounded_process,
    run_stream_to_file,
    pid_alive,
    read_worker_state,
    write_worker_state,
)


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _identity(tmp_path: Path) -> RemoteIdentity:
    identity = tmp_path / "id_ed25519"
    known = tmp_path / "known_hosts"
    identity.write_bytes(b"not-a-real-private-key\n")
    known.write_text("[example.test]:22 ssh-ed25519 AAAA\n", encoding="utf-8")
    return RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=identity,
        known_hosts=known,
    )


_STREAM_SCRIPT = r"""
import hashlib, json, subprocess, sys, time
mode = sys.argv[1]
data = bytes.fromhex(sys.argv[2])
offset = int(sys.argv[3])
send = int(sys.argv[4])
exit_code = int(sys.argv[5])
sleep_after = float(sys.argv[6])
path = sys.argv[7]
if mode == "malformed":
    sys.stdout.buffer.write(b"not-json\n")
    sys.stdout.buffer.flush()
else:
    header = {
        "schema": "anime-video-studio/p1-v3-stream/v1",
        "path": path,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "offset": offset,
        "length": len(data) - offset,
    }
    sys.stdout.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()
sys.stdout.buffer.write(data[offset:offset + send])
sys.stdout.buffer.flush()
if mode == "flood":
    chunk = b"x" * 65536
    while True:
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
if mode == "child":
    subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
if sleep_after:
    time.sleep(sleep_after)
if exit_code:
    raise SystemExit(exit_code)
"""


def _stream_argv(
    *,
    mode: str,
    data: bytes,
    offset: int,
    send: int,
    exit_code: int = 0,
    sleep_after: float = 0.0,
    path: str = "/remote/artifact.mp4",
) -> list[str]:
    return [
        sys.executable,
        "-c",
        _STREAM_SCRIPT,
        mode,
        data.hex(),
        str(offset),
        str(send),
        str(exit_code),
        str(sleep_after),
        path,
    ]


def _remote_reader_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "p1-remote-stream-reader"
    ]


def _runner(tmp_path: Path, *, stdout: bytes = b"", stderr: bytes = b"", code: int = 0, limit: int = 4096):
    out_data = stdout
    err_data = stderr

    def run(argv, *, timeout, check=False):
        return _FakeCompleted(code, out_data, err_data)

    return RemoteRunner(
        ssh_path="ssh",
        scp_path="scp",
        ssh_keygen_path="ssh-keygen",
        runner=run,
        max_output_bytes=limit,
    )


def test_identity_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(RemoteError):
        RemoteIdentity(
            host="example.test",
            user="root",
            port=22,
            identity_file=tmp_path / "missing",
            known_hosts=tmp_path / "missing-known",
        )


def test_host_key_check_is_strict_and_explicit(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    runner = _runner(tmp_path, stdout=b"example.test ssh-ed25519 AAAA\n")
    assert runner.check_host_key(identity) is True
    argv = runner.host_key_argv(identity)
    assert "-F" in argv
    assert "[example.test]:22" in argv
    assert str(identity.known_hosts) in argv


def test_missing_host_key_fails_closed(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    runner = _runner(tmp_path, stdout=b"", code=1)
    with pytest.raises(RemoteHostKeyError):
        runner.check_host_key(identity)


def test_ssh_argv_contains_strict_options_and_exact_destination(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    runner = _runner(tmp_path)
    argv = runner.ssh_argv(identity, ("true",))
    assert "StrictHostKeyChecking=yes" in argv
    assert "BatchMode=yes" in argv
    assert f"UserKnownHostsFile={identity.known_hosts}" in argv
    assert identity.destination in argv
    assert argv[-1] == "true"
    assert "not-a-real-private-key" not in " ".join(argv)


def test_scp_path_rejects_absolute_and_traversal(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    runner = _runner(tmp_path)
    with pytest.raises(RemoteError):
        runner.scp_argv(identity, "/etc/passwd", tmp_path / "x")
    with pytest.raises(RemoteError):
        runner.scp_argv(identity, "../secret", tmp_path / "x")


def test_put_file_receiver_token_is_safe_and_executes(tmp_path: Path) -> None:
    receiver = "exec(" + repr(remote_module._REMOTE_RECEIVER) + ")"
    assert "\n" not in receiver
    assert "\r" not in receiver
    assert "\x00" not in receiver
    data = b"staged-bytes"
    dest = tmp_path / "dest.bin"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            receiver,
            str(dest),
            str(tmp_path),
            hashlib.sha256(data).hexdigest(),
            "noclobber",
        ],
        input=data,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert dest.read_bytes() == data


def test_bounded_output_raises(tmp_path: Path) -> None:
    runner = _runner(tmp_path, stdout=b"x" * 4096, limit=1024)
    with pytest.raises(RemoteOutputTooLarge):
        runner.run_argv(("echo", "x"), max_output_bytes=1024)


def test_worker_state_roundtrip_and_pid(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    write_worker_state(path, {"run_id": "r1", "pid": 123, "phase": "RUNNING"})
    assert read_worker_state(path) == {"run_id": "r1", "pid": 123, "phase": "RUNNING"}
    assert pid_alive(99999999) is False
    assert pid_alive(0) is False


def test_start_background_is_disabled_before_runner(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    runner = _runner(tmp_path, stdout=b"4242\n")
    with pytest.raises(RemoteUnsupported):
        runner.start_background(
            identity,
            ("sleep", "1"),
            log_path="run/worker.log",
            pid_path="run/worker.pid",
        )


def test_scp_fetch_is_disabled_before_runner(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    runner = _runner(tmp_path)
    with pytest.raises(RemoteUnsupported):
        runner.scp_fetch(
            identity,
            "artifact.mp4",
            tmp_path / "artifact.mp4",
            max_bytes=1024,
        )


def test_real_popen_output_is_killed_at_bound() -> None:
    import sys

    runner = RemoteRunner(
        ssh_path="ssh",
        scp_path="scp",
        ssh_keygen_path="ssh-keygen",
        max_output_bytes=1024,
    )
    with pytest.raises(RemoteOutputTooLarge):
        runner.run_argv(
            [sys.executable, "-c", "print('x' * 200000)"],
            timeout=10.0,
            max_output_bytes=1024,
        )


class _FakePopenFailWait:
    def __init__(self, *args, **kwargs):
        import io

        self.pid = None
        self.returncode = 1
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"")
        self.killed = False

    def poll(self):
        return 1

    def wait(self, timeout=None):
        import subprocess

        raise subprocess.TimeoutExpired("fake", timeout)

    def kill(self):
        self.killed = True


def test_run_bounded_cleanup_failure_is_typed_and_not_success() -> None:
    import scripts.p1.remote as remote_module

    with pytest.raises(remote_module.RemoteCleanupError):
        run_bounded_process(
            ["fake"],
            timeout=1.0,
            max_output_bytes=1024,
            popen=_FakePopenFailWait,
        )


def test_run_bounded_kills_inherited_pipe_descendant() -> None:
    import sys

    script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], "
        "stdout=sys.stdout,stderr=sys.stderr); time.sleep(30)"
    )
    result = run_bounded_process(
        [sys.executable, "-c", script],
        timeout=0.6,
        max_output_bytes=4096,
    )
    assert result.timed_out is True
    assert result.returncode == 124


def test_run_bounded_input_writer_is_bounded_when_child_never_reads() -> None:
    result = run_bounded_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout=0.5,
        max_output_bytes=4096,
        input_bytes=b"x" * (2 * 1024 * 1024),
    )
    assert result.timed_out is True
    assert result.returncode == 124
    assert result.writer_done is True
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name == "p1-remote-stdin-writer"
    ]


def test_run_stream_deadline_covers_silent_header_and_payload(tmp_path: Path) -> None:
    cases = [
        "import time; time.sleep(5)",
        "import sys,time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(5)",
        "import sys,time; sys.stdout.write('{}\\n'); sys.stdout.flush(); time.sleep(5)",
    ]
    for index, script in enumerate(cases):
        target = tmp_path / f"stream-{index}.bin"
        started = time.monotonic()
        with pytest.raises(RemoteError):
            run_stream_to_file(
                [sys.executable, "-c", script],
                target,
                timeout=0.5,
                max_payload_bytes=1024,
            )
        assert time.monotonic() - started < 3.0
        assert not target.exists()
        assert not _remote_reader_threads()


def test_run_stream_exit255_keeps_validated_partial_and_resumes(tmp_path: Path) -> None:
    data = b"0123456789abcdef" * 256
    digest = hashlib.sha256(data).hexdigest()
    first = tmp_path / "first.part"
    with pytest.raises(RemoteStreamPartialError) as excinfo:
        run_stream_to_file(
            _stream_argv(mode="normal", data=data, offset=0, send=40, exit_code=255),
            first,
            timeout=5.0,
            max_payload_bytes=len(data),
            expected_path="/remote/artifact.mp4",
            expected_size=len(data),
            expected_sha256=digest,
            expected_offset=0,
        )
    error = excinfo.value
    assert error.returncode == 255
    assert error.payload_size == 40
    assert error.header["path"] == "/remote/artifact.mp4"
    assert error.header["offset"] == 0
    assert first.read_bytes() == data[:40]
    assert not _remote_reader_threads()

    second = tmp_path / "second.part"
    header_raw, payload_size = run_stream_to_file(
        _stream_argv(
            mode="normal",
            data=data,
            offset=40,
            send=len(data) - 40,
            exit_code=0,
        ),
        second,
        timeout=5.0,
        max_payload_bytes=len(data) - 40,
        expected_path="/remote/artifact.mp4",
        expected_size=len(data),
        expected_sha256=digest,
        expected_offset=40,
    )
    assert payload_size == len(data) - 40
    assert json.loads(header_raw.decode("utf-8"))["offset"] == 40
    combined = first.read_bytes() + second.read_bytes()
    assert combined == data
    assert hashlib.sha256(combined).hexdigest() == digest
    assert not _remote_reader_threads()


def test_run_stream_partial_without_expected_identity_is_not_kept(
    tmp_path: Path,
) -> None:
    data = b"unvalidated-partial" * 8
    target = tmp_path / "no-expected.part"
    with pytest.raises(RemoteCommandError):
        run_stream_to_file(
            _stream_argv(mode="normal", data=data, offset=0, send=40, exit_code=255),
            target,
            timeout=3.0,
            max_payload_bytes=len(data),
        )
    assert not target.exists()
    assert not _remote_reader_threads()


def test_run_stream_malformed_header_deletes_partial(tmp_path: Path) -> None:
    target = tmp_path / "malformed.part"
    with pytest.raises(RemoteStreamProtocolError):
        run_stream_to_file(
            _stream_argv(
                mode="malformed",
                data=b"x" * 40,
                offset=0,
                send=40,
                exit_code=255,
            ),
            target,
            timeout=3.0,
            max_payload_bytes=1024,
        )
    assert not target.exists()
    assert not _remote_reader_threads()


def test_run_stream_expected_checksum_mismatch_deletes_partial(tmp_path: Path) -> None:
    data = b"checksum-data" * 100
    target = tmp_path / "mismatch.part"
    with pytest.raises(RemoteStreamProtocolError):
        run_stream_to_file(
            _stream_argv(mode="normal", data=data, offset=0, send=40, exit_code=255),
            target,
            timeout=3.0,
            max_payload_bytes=len(data),
            expected_path="/remote/artifact.mp4",
            expected_size=len(data),
            expected_sha256="f" * 64,
            expected_offset=0,
        )
    assert not target.exists()
    assert not _remote_reader_threads()


def test_run_stream_timeout_keeps_validated_partial(tmp_path: Path) -> None:
    data = b"timeout-partial" * 10
    target = tmp_path / "timeout.part"
    with pytest.raises(RemoteStreamPartialError) as excinfo:
        run_stream_to_file(
            _stream_argv(
                mode="normal",
                data=data,
                offset=0,
                send=len(data),
                exit_code=0,
                sleep_after=5.0,
            ),
            target,
            timeout=0.6,
            max_payload_bytes=len(data),
            expected_path="/remote/artifact.mp4",
            expected_size=len(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_offset=0,
        )
    assert excinfo.value.timed_out is True
    assert target.read_bytes() == data
    assert not _remote_reader_threads()


def test_run_stream_payload_overflow_cleans_reader_and_target(tmp_path: Path) -> None:
    data = b"x" * 4096
    target = tmp_path / "overflow.part"
    with pytest.raises(RemoteOutputTooLarge):
        run_stream_to_file(
            _stream_argv(mode="normal", data=data, offset=0, send=len(data)),
            target,
            timeout=5.0,
            max_payload_bytes=1024,
        )
    assert not target.exists()
    assert not _remote_reader_threads()


def test_run_stream_overflow_with_flooded_queue_stops_reader(tmp_path: Path) -> None:
    data = b"x" * 4096
    target = tmp_path / "flood.part"
    with pytest.raises(RemoteOutputTooLarge):
        run_stream_to_file(
            _stream_argv(mode="flood", data=data, offset=0, send=0),
            target,
            timeout=5.0,
            max_payload_bytes=1024,
        )
    assert not target.exists()
    assert not _remote_reader_threads()


def test_run_stream_inherited_pipe_child_is_killed_and_cleanup_confirmed(
    tmp_path: Path,
) -> None:
    data = b"child-pipe" * 4
    target = tmp_path / "child.part"
    with pytest.raises((RemoteStreamPartialError, RemoteCleanupError)):
        run_stream_to_file(
            _stream_argv(
                mode="child",
                data=data,
                offset=0,
                send=len(data),
                exit_code=0,
                sleep_after=10.0,
            ),
            target,
            timeout=1.0,
            max_payload_bytes=len(data),
            expected_path="/remote/artifact.mp4",
            expected_size=len(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_offset=0,
        )
    assert not _remote_reader_threads()


class _FakeStreamCleanupFail:
    def __init__(self, payload: bytes) -> None:
        self.pid = None
        self.returncode = 1
        self.stdout = io.BytesIO(payload)
        self.killed = False

    def poll(self):
        return 1

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("fake", timeout)

    def kill(self):
        self.killed = True


def test_run_stream_cleanup_failure_is_typed_and_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = {
        "schema": "anime-video-studio/p1-v3-stream/v1",
        "path": "/remote/artifact.mp4",
        "size": 4,
        "sha256": hashlib.sha256(b"data").hexdigest(),
        "offset": 0,
        "length": 4,
    }
    payload = json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\ndata"
    fake = _FakeStreamCleanupFail(payload)
    monkeypatch.setattr(remote_module.subprocess, "Popen", lambda *args, **kwargs: fake)
    target = tmp_path / "cleanup.part"
    with pytest.raises(RemoteCleanupError):
        run_stream_to_file(
            ["fake"],
            target,
            timeout=1.0,
            max_payload_bytes=1024,
        )
    assert not target.exists()
    assert not _remote_reader_threads()


def test_remote_staging_methods_respect_absolute_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity(tmp_path)
    calls: list[tuple[tuple[str, ...], float]] = []

    def fake_run(argv, *, timeout, check=False):
        calls.append((tuple(argv), float(timeout)))
        return _FakeCompleted(0, b"example.test ssh-ed25519 AAAA\n")

    def fake_bounded(argv, *, timeout, max_output_bytes, input_bytes=None, popen=None):
        calls.append((tuple(argv), float(timeout)))
        return remote_module.BoundedProcessResult(0, b"", b"", False, True, True)

    monkeypatch.setattr(remote_module, "run_bounded_process", fake_bounded)
    runner = RemoteRunner(
        ssh_path="ssh",
        scp_path="scp",
        ssh_keygen_path="ssh-keygen",
        runner=fake_run,
    )
    expired = time.monotonic() - 1.0
    with pytest.raises(RemoteCommandError):
        runner.run_remote(identity, ("true",), deadline_monotonic=expired)
    with pytest.raises(RemoteCommandError):
        runner.run_remote_with_input(
            identity, ("true",), b"x", deadline_monotonic=expired
        )
    local = tmp_path / "staged.bin"
    local.write_bytes(b"payload")
    with pytest.raises(RemoteCommandError):
        runner.put_file(
            identity,
            "/remote/root/staged.bin",
            local,
            deadline_monotonic=expired,
        )
    with pytest.raises(RemoteCommandError):
        runner.read_remote_exact_json(
            identity,
            helper_path="/remote/root/v3_remote_fetch_helper.py",
            remote_path="/remote/root/state.json",
            root="/remote/root",
            deadline_monotonic=expired,
        )
    with pytest.raises(RemoteCommandError):
        runner.stream_remote_to_file(
            identity,
            ("true",),
            tmp_path / "never.bin",
            timeout=1.0,
            max_payload_bytes=1024,
            deadline_monotonic=expired,
        )
    assert calls == []

    calls.clear()
    deadline = time.monotonic() + 2.0
    runner.run_remote(identity, ("true",), deadline_monotonic=deadline)
    assert len(calls) == 2
    assert all(0 < timeout <= 2.0 for _argv, timeout in calls)
    calls.clear()
    deadline = time.monotonic() + 2.0
    runner.run_remote_with_input(
        identity, ("true",), b"x", deadline_monotonic=deadline
    )
    assert len(calls) == 2
    assert all(0 < timeout <= 2.0 for _argv, timeout in calls)
    calls.clear()
    deadline = time.monotonic() + 2.0
    runner.put_file(
        identity,
        "/remote/root/staged.bin",
        local,
        deadline_monotonic=deadline,
    )
    assert len(calls) == 2
    assert all(0 < timeout <= 2.0 for _argv, timeout in calls)
