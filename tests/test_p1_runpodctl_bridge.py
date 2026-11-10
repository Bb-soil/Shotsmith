from __future__ import annotations

import calendar
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.p1.runpodctl_bridge import (
    MAX_OUTPUT_BYTES,
    STATUS_ERROR,
    STATUS_INITIALIZING,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_TERMINATED,
    STATUS_UNKNOWN,
    ProviderAuthError,
    ProviderContractUnverified,
    ProviderError,
    ProviderProtocolError,
    ProviderTransientError,
    ProviderVersionUnsupported,
    RunpodctlProvider,
    _run_bounded_process,
    parse_utc_timestamp,
)


FIXTURES = Path(__file__).parent / "fixtures" / "p1" / "runpodctl-v3.0.1"
POD_ID = "pod-abc123"
VERSION_OK = "runpodctl v3.0.1\n"


@dataclass
class Completed:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class QueueRunner:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> Any:
        self.calls.append((list(args), dict(kwargs)))
        if not self.responses:
            raise AssertionError(f"unexpected runpodctl call: {args!r}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def payload(name: str) -> Completed:
    return Completed(0, fixture(name), "")


def error_payload(name: str, returncode: int = 1) -> Completed:
    return Completed(returncode, "", fixture(name))


def provider_for(*responses: Any, version: Any = None) -> tuple[RunpodctlProvider, QueueRunner]:
    version_response = Completed(0, VERSION_OK, "") if version is None else version
    runner = QueueRunner([version_response, *responses])
    return RunpodctlProvider(runner=runner), runner


def verified_provider(*responses: Any) -> tuple[RunpodctlProvider, QueueRunner]:
    provider, runner = provider_for(*responses)
    provider.verify_contract()
    return provider, runner


def assert_safe_call(args: list[str], kwargs: dict[str, Any]) -> None:
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert "token" not in " ".join(args).lower()
    assert "key" not in " ".join(args).lower()


def test_verify_contract_requires_exact_pinned_argv_and_version() -> None:
    provider, runner = provider_for()

    assert provider.verify_contract() == "v3.0.1"
    assert len(runner.calls) == 1
    args, kwargs = runner.calls[0]
    assert args == ["runpodctl", "--version"]
    assert_safe_call(args, kwargs)
    assert kwargs["timeout"] == 30.0


def test_get_and_stop_require_verification_and_have_exact_argv() -> None:
    provider, runner = verified_provider(
        payload("pod-get-stopped.json"),
        Completed(0, '{"id":"pod-abc123","desiredStatus":"RUNNING"}\n', ""),
    )

    pod = provider.get(POD_ID)
    provider.stop(POD_ID)

    assert pod is not None
    assert pod.id == POD_ID
    assert pod.status == STATUS_STOPPED
    assert [call[0] for call in runner.calls] == [
        ["runpodctl", "--version"],
        ["runpodctl", "pod", "get", POD_ID, "--output", "json"],
        ["runpodctl", "pod", "stop", POD_ID],
    ]
    for args, kwargs in runner.calls:
        assert_safe_call(args, kwargs)


def test_no_bool_bypass_can_enable_readback() -> None:
    provider = RunpodctlProvider(runner=QueueRunner([]))
    with pytest.raises(ProviderContractUnverified):
        provider.get(POD_ID)
    setattr(provider, "readback_contract_verified", True)
    with pytest.raises(ProviderContractUnverified):
        provider.get(POD_ID)
    setattr(provider, "readback_contract_verified", True)
    assert provider._verified_version is None
    # No fake runner response was consumed; the attribute has no effect.
    with pytest.raises(ProviderContractUnverified):
        provider.stop(POD_ID)


@pytest.mark.parametrize(
    "version_response",
    [
        Completed(0, "runpodctl v2.15.0\n", ""),
        Completed(0, "runpodctl 3.0.1-extra\n", ""),
        Completed(0, "not runpodctl\n", ""),
        Completed(1, "", '{"code":"network_error"}'),
    ],
)
def test_unsupported_or_unparseable_version_fails_closed(version_response: Any) -> None:
    provider, runner = provider_for(version=version_response)

    with pytest.raises(ProviderVersionUnsupported):
        provider.verify_contract()

    assert len(runner.calls) == 1
    assert runner.calls[0][0] == ["runpodctl", "--version"]


def test_missing_executable_and_version_timeout_are_sanitized() -> None:
    provider, runner = provider_for(version=FileNotFoundError("synthetic-secret-path"))
    with pytest.raises(ProviderContractUnverified) as missing:
        provider.verify_contract()
    assert "synthetic-secret-path" not in str(missing.value)
    assert missing.value.__cause__ is None
    assert len(runner.calls) == 1

    timeout = subprocess.TimeoutExpired(["runpodctl", "--version"], 30)
    provider, _ = provider_for(version=timeout)
    with pytest.raises(ProviderTransientError) as timed_out:
        provider.verify_contract()
    assert "timed out" in str(timed_out.value)
    assert timed_out.value.__cause__ is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pod-get-running.json", STATUS_RUNNING),
        ("pod-get-initializing.json", STATUS_INITIALIZING),
        ("pod-get-stopped.json", STATUS_STOPPED),
        ("pod-get-terminated.json", STATUS_TERMINATED),
        ("pod-get-error.json", STATUS_ERROR),
        ("pod-get-unknown.json", STATUS_UNKNOWN),
    ],
)
def test_known_statuses_are_normalized(name: str, expected: str) -> None:
    provider, _ = verified_provider(payload(name))
    pod = provider.get(POD_ID)
    assert pod is not None
    assert pod.status == expected


@pytest.mark.parametrize(
    "name",
    [
        "pod-get-malformed.json",
        "pod-get-wrong-type.json",
        "pod-get-wrong-id.json",
        "pod-get-missing-id.json",
        "pod-get-missing-fields.json",
        "pod-get-contradictory.json",
        "pod-get-null-runtime.json",
        "pod-get-missing-runtime.json",
        "pod-get-unknown-no-reason.json",
        "pod-get-createdat-bad.json",
    ],
)
def test_malformed_or_inconsistent_readback_fails_closed(name: str) -> None:
    provider, _ = verified_provider(payload(name))
    with pytest.raises(ProviderProtocolError):
        provider.get(POD_ID)


def test_created_at_supports_source_formats_and_missing_is_not_identity() -> None:
    provider, _ = verified_provider(
        payload("pod-get-stopped.json"),
        payload("pod-get-running.json"),
        payload("pod-get-unknown.json"),
        payload("pod-get-no-createdat.json"),
    )

    stopped = provider.get(POD_ID)
    running = provider.get(POD_ID)
    unknown = provider.get(POD_ID)
    missing = provider.get(POD_ID)

    expected = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    base_ns = calendar.timegm(expected.utctimetuple()) * 1_000_000_000
    assert stopped is not None and stopped.created_at == expected
    assert running is not None and running.created_at == expected
    assert unknown is not None and unknown.created_at == expected
    assert missing is not None and missing.created_at is None
    assert stopped.created_at_ns == base_ns + 123_456_789
    assert running.created_at_ns == base_ns + 987_654_321
    assert unknown.created_at_ns == base_ns
    assert missing.created_at_ns is None
    assert stopped.created_at_text == "2026-10-21T12:00:00.123456789Z"
    assert running.created_at_text == "2026-10-21T12:00:00.987654321Z"


def test_null_or_missing_runtime_is_never_stopped() -> None:
    for name in ("pod-get-null-runtime.json", "pod-get-missing-runtime.json"):
        provider, _ = verified_provider(payload(name))
        with pytest.raises(ProviderProtocolError):
            provider.get(POD_ID)


def test_terminal_and_error_states_are_not_stopped() -> None:
    provider, _ = verified_provider(
        payload("pod-get-terminated.json"),
        payload("pod-get-error.json"),
    )
    terminated = provider.get(POD_ID)
    error = provider.get(POD_ID)
    assert terminated is not None and terminated.status == STATUS_TERMINATED
    assert error is not None and error.status == STATUS_ERROR


def test_not_found_returns_none_without_claiming_stopped() -> None:
    provider, _ = verified_provider(error_payload("pod-get-notfound.stderr.json", 1))
    assert provider.get(POD_ID) is None


@pytest.mark.parametrize(
    ("fixture_name", "error_type"),
    [
        ("pod-get-unauthorized.stderr.json", ProviderAuthError),
        ("pod-get-network.stderr.json", ProviderTransientError),
        ("pod-get-server.stderr.json", ProviderTransientError),
    ],
)
def test_get_error_codes_are_classified_without_raw_output(
    fixture_name: str, error_type: type[ProviderError]
) -> None:
    provider, _ = verified_provider(error_payload(fixture_name, 1))
    with pytest.raises(error_type) as excinfo:
        provider.get(POD_ID)

    message = str(excinfo.value)
    assert "synthetic-" not in message
    assert "must-not-leak" not in message
    assert excinfo.value.__cause__ is None
    if fixture_name == "pod-get-network.stderr.json":
        assert excinfo.value.code == "network_error"
    if fixture_name == "pod-get-server.stderr.json":
        assert excinfo.value.http_status == 503


def test_stop_nonzero_is_sanitized_and_does_not_read_back() -> None:
    provider, runner = verified_provider(
        error_payload("pod-stop-conflict.stderr.json", 1)
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.stop(POD_ID)

    assert "synthetic-" not in str(excinfo.value)
    assert "must-not-leak" not in str(excinfo.value)
    assert excinfo.value.code == "conflict"
    assert [call[0] for call in runner.calls] == [
        ["runpodctl", "--version"],
        ["runpodctl", "pod", "stop", POD_ID],
    ]


def test_stop_timeout_is_transient_and_sanitized() -> None:
    timeout = subprocess.TimeoutExpired(["runpodctl", "pod", "stop", POD_ID], 30)
    provider, _ = verified_provider(timeout)
    with pytest.raises(ProviderTransientError) as excinfo:
        provider.stop(POD_ID)
    assert "timed out" in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_runner_exception_is_wrapped_without_raw_text_or_traceback() -> None:
    provider, _ = verified_provider(
        RuntimeError("synthetic-runner-secret-must-not-leak")
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.get(POD_ID)
    assert "synthetic-runner-secret" not in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_output_size_limit_is_finite() -> None:
    provider, _ = verified_provider(
        Completed(0, "x" * (MAX_OUTPUT_BYTES + 1), "")
    )
    with pytest.raises(ProviderProtocolError):
        provider.get(POD_ID)


def test_secret_fields_are_ignored_and_never_exposed() -> None:
    provider, _ = verified_provider(payload("pod-get-secret-fields.json"))
    pod = provider.get(POD_ID)
    assert pod is not None
    serialized = repr(pod)
    for secret in (
        "synthetic-secret-must-not-leak",
        "synthetic-key-must-not-leak",
        "synthetic-ssh-error-must-not-leak",
        "synthetic-volume-must-not-leak",
    ):
        assert secret not in serialized
    assert not hasattr(pod, "raw")
    assert set(pod.__dataclass_fields__) == {
        "id",
        "status",
        "created_at",
        "reason",
        "created_at_ns",
        "created_at_text",
    }


def test_get_timeout_uses_explicit_remaining_budget() -> None:
    provider, runner = verified_provider(payload("pod-get-running.json"))
    provider.get(POD_ID, timeout_seconds=1.25)
    assert runner.calls[-1][1]["timeout"] == 1.25


def test_json_payload_with_secret_in_error_never_leaks() -> None:
    response = Completed(
        1,
        "",
        json.dumps(
            {
                "error": "synthetic-error-secret-must-not-leak",
                "code": "unauthorized",
                "status": 401,
            }
        ),
    )
    provider, _ = verified_provider(response)
    with pytest.raises(ProviderAuthError) as excinfo:
        provider.get(POD_ID)
    assert "synthetic-error-secret" not in str(excinfo.value)
    assert excinfo.value.http_status == 401


class FakeStream:
    def __init__(self, data: bytes = b"", *, hold_open: bool = False) -> None:
        self.data = bytearray(data)
        self.hold_open = hold_open
        self.closed = False
        self.finished = threading.Event()
        self.unblock = threading.Event()

    def read(self, size: int) -> bytes:
        if self.closed:
            return b""
        if self.data:
            chunk = bytes(self.data[:size])
            del self.data[:size]
            return chunk
        if self.hold_open:
            self.unblock.wait(timeout=2.0)
            return b""
        self.finished.set()
        return b""

    def close(self) -> None:
        self.closed = True
        self.finished.set()
        self.unblock.set()


class FakePopen:
    def __init__(
        self,
        *,
        stdout_data: bytes = b"",
        stderr_data: bytes = b"",
        returncode: int = 0,
        hold_open: bool = False,
        ignore_terminate: bool = False,
    ) -> None:
        self.stdout = FakeStream(stdout_data, hold_open=hold_open)
        self.stderr = FakeStream(stderr_data, hold_open=hold_open)
        self._returncode = returncode
        self.returncode: int | None = None
        self.ignore_terminate = ignore_terminate
        self.terminated = False
        self.killed = False
        self.reaped = False

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self.stdout.finished.is_set() and self.stderr.finished.is_set():
            self.returncode = self._returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignore_terminate:
            self.returncode = -15
            self._close_streams()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._close_streams()

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None and self.ignore_terminate:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        self.reaped = True
        if self.returncode is None:
            self.returncode = self._returncode
        return self.returncode

    def _close_streams(self) -> None:
        self.stdout.close()
        self.stderr.close()


def _fake_popen_factory(fake: FakePopen):
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def factory(args: list[str], **kwargs: Any) -> FakePopen:
        calls.append((list(args), dict(kwargs)))
        return fake

    return factory, calls


def _assert_no_reader_threads() -> None:
    names = {"runpodctl-stdout-reader", "runpodctl-stderr-reader"}
    assert not any(
        thread.name in names and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_bounded_process_reads_both_streams_under_cap() -> None:
    fake = FakePopen(stdout_data=b"stdout-ok", stderr_data=b"stderr-ok")
    factory, calls = _fake_popen_factory(fake)

    result = _run_bounded_process(
        ["runpodctl", "pod", "get"],
        timeout_seconds=1.0,
        max_output_bytes=64,
        popen_factory=factory,
    )

    assert result.returncode == 0
    assert result.stdout == "stdout-ok"
    assert result.stderr == "stderr-ok"
    assert calls[0][0] == ["runpodctl", "pod", "get"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["stdout"] is subprocess.PIPE
    assert calls[0][1]["stderr"] is subprocess.PIPE
    assert calls[0][1]["bufsize"] == 0
    assert fake.stdout.closed and fake.stderr.closed
    _assert_no_reader_threads()


def test_bounded_process_caps_stderr_even_when_exit_is_zero() -> None:
    secret = b"synthetic-stream-secret-must-not-leak"
    fake = FakePopen(stdout_data=b"ok", stderr_data=secret + b"x" * 256)
    factory, _ = _fake_popen_factory(fake)

    with pytest.raises(ProviderProtocolError) as excinfo:
        _run_bounded_process(
            ["runpodctl", "pod", "get"],
            timeout_seconds=1.0,
            max_output_bytes=32,
            popen_factory=factory,
        )

    assert "synthetic-stream-secret" not in str(excinfo.value)
    assert "stderr" in str(excinfo.value)
    assert fake.terminated and fake.reaped
    assert fake.stdout.closed and fake.stderr.closed
    _assert_no_reader_threads()


def test_bounded_process_caps_stdout_when_exit_is_nonzero() -> None:
    secret = b"synthetic-stdout-secret-must-not-leak"
    fake = FakePopen(stdout_data=secret + b"x" * 256, stderr_data=b"err", returncode=1)
    factory, _ = _fake_popen_factory(fake)

    with pytest.raises(ProviderProtocolError) as excinfo:
        _run_bounded_process(
            ["runpodctl", "pod", "get"],
            timeout_seconds=1.0,
            max_output_bytes=32,
            popen_factory=factory,
        )

    assert "synthetic-stdout-secret" not in str(excinfo.value)
    assert "stdout" in str(excinfo.value)
    assert fake.terminated and fake.reaped
    assert fake.stdout.closed and fake.stderr.closed
    _assert_no_reader_threads()


def test_bounded_process_timeout_kills_reaps_and_closes() -> None:
    fake = FakePopen(hold_open=True, ignore_terminate=True)
    factory, _ = _fake_popen_factory(fake)

    with pytest.raises(ProviderTransientError) as excinfo:
        _run_bounded_process(
            ["runpodctl", "pod", "get"],
            timeout_seconds=0.01,
            max_output_bytes=32,
            popen_factory=factory,
        )

    assert "timed out" in str(excinfo.value)
    assert fake.terminated and fake.killed and fake.reaped
    assert fake.stdout.closed and fake.stderr.closed
    _assert_no_reader_threads()


def test_created_at_nanoseconds_and_canonical_equivalence() -> None:
    a = parse_utc_timestamp("2026-10-21 12:00:00.100 +0000 UTC")
    b = parse_utc_timestamp("2026-10-21T12:00:00.1Z")
    c = parse_utc_timestamp("2026-10-21T14:00:00.100+02:00")
    d = parse_utc_timestamp("2026-10-21T12:00:00.900Z")
    e = parse_utc_timestamp("2026-10-21T12:00:00.100000001Z")

    assert a is not None and b is not None and c is not None
    assert d is not None and e is not None
    assert a.epoch_ns == b.epoch_ns == c.epoch_ns
    assert a.text == b.text == c.text == "2026-10-21T12:00:00.100000000Z"
    assert d.epoch_ns != a.epoch_ns
    assert e.epoch_ns != a.epoch_ns
    assert a.datetime == b.datetime == c.datetime == d.datetime == e.datetime


def test_unknown_cli_error_code_is_never_echoed() -> None:
    response = Completed(
        1,
        "",
        json.dumps(
            {
                "error": "synthetic-error-must-not-leak",
                "code": "syntheticsecretcode",
                "status": 401,
            }
        ),
    )
    provider, _ = verified_provider(response)
    with pytest.raises(ProviderError) as excinfo:
        provider.get(POD_ID)
    assert excinfo.value.code == "unknown_error"
    assert "syntheticsecretcode" not in str(excinfo.value)
    assert "synthetic-error-must-not-leak" not in str(excinfo.value)


class NeverStartedThread:
    def start(self) -> None:
        raise RuntimeError("second reader start failure")

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_alive(self) -> bool:
        return False


class FailingSecondThreadFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.started: list[threading.Thread] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            thread = threading.Thread(**kwargs)
            self.started.append(thread)
            return thread
        return NeverStartedThread()


def test_second_thread_start_failure_cleans_up_started_thread_and_process() -> None:
    fake = FakePopen(hold_open=True)
    factory, _ = _fake_popen_factory(fake)
    threads = FailingSecondThreadFactory()

    with pytest.raises(RuntimeError, match="second reader start failure"):
        _run_bounded_process(
            ["runpodctl", "pod", "get"],
            timeout_seconds=1.0,
            max_output_bytes=32,
            popen_factory=factory,
            thread_factory=threads,
        )

    assert threads.calls == 2
    assert fake.terminated and fake.reaped
    assert fake.stdout.closed and fake.stderr.closed
    assert threads.started and not threads.started[0].is_alive()
    _assert_no_reader_threads()


def test_keyboard_interrupt_cancels_child_and_joins_readers() -> None:
    class CancellingPopen(FakePopen):
        def poll(self) -> int | None:
            raise KeyboardInterrupt

    fake = CancellingPopen(hold_open=True)
    factory, _ = _fake_popen_factory(fake)

    with pytest.raises(KeyboardInterrupt):
        _run_bounded_process(
            ["runpodctl", "pod", "get"],
            timeout_seconds=1.0,
            max_output_bytes=32,
            popen_factory=factory,
        )

    assert fake.terminated and fake.reaped
    assert fake.stdout.closed and fake.stderr.closed
    _assert_no_reader_threads()


class StartedFakeThread:
    def __init__(self) -> None:
        self.join_calls = 0

    def start(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1

    def is_alive(self) -> bool:
        return False


class RaisingSecondThreadFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.first = StartedFakeThread()

    def __call__(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return self.first
        raise RuntimeError("thread factory second call failure")


def test_thread_factory_failure_after_popen_cleans_child_and_streams() -> None:
    fake = FakePopen(hold_open=True)
    factory, _ = _fake_popen_factory(fake)
    threads = RaisingSecondThreadFactory()

    with pytest.raises(RuntimeError, match="thread factory second call failure"):
        _run_bounded_process(
            ["runpodctl", "pod", "get"],
            timeout_seconds=1.0,
            max_output_bytes=32,
            popen_factory=factory,
            thread_factory=threads,
        )

    assert threads.calls == 2
    assert fake.terminated and fake.reaped
    assert fake.stdout.closed and fake.stderr.closed
    assert threads.first.join_calls == 0  # never started, must not be joined
    _assert_no_reader_threads()


class JoinCancellingFactory:
    def __init__(self) -> None:
        self.first = StartedFakeThread()
        self.second = StartedFakeThread()
        self._cancel_once = True

    def __call__(self, **kwargs: Any) -> Any:
        if kwargs.get("name") == "runpodctl-stdout-reader":
            return JoinCancellingThread(self, self.first)
        return JoinCancellingThread(self, self.second)


class JoinCancellingThread(StartedFakeThread):
    def __init__(self, factory: JoinCancellingFactory, target: StartedFakeThread) -> None:
        super().__init__()
        self.factory = factory
        self.target = target

    def join(self, timeout: float | None = None) -> None:
        self.target.join(timeout)
        if self.factory._cancel_once:
            self.factory._cancel_once = False
            raise KeyboardInterrupt


class AlreadyExitedPopen(FakePopen):
    def poll(self) -> int | None:
        self.returncode = 0
        return 0


def test_keyboard_interrupt_during_normal_reader_join_cleans_child() -> None:
    fake = AlreadyExitedPopen(hold_open=True)
    factory, _ = _fake_popen_factory(fake)
    threads = JoinCancellingFactory()

    with pytest.raises(KeyboardInterrupt):
        _run_bounded_process(
            ["runpodctl", "pod", "get"],
            timeout_seconds=1.0,
            max_output_bytes=32,
            popen_factory=factory,
            thread_factory=threads,
        )

    assert fake.terminated and fake.reaped
    assert fake.stdout.closed and fake.stderr.closed
    assert threads.first.join_calls >= 1
    assert threads.second.join_calls >= 1
    _assert_no_reader_threads()
