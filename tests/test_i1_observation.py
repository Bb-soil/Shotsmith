from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.i1 import cli, observation, ops, state as state_mod
from scripts.i1.errors import Interrupted, RemoteFailure
from scripts.i1.profile import load_profile
from scripts.p1.remote import RemoteIdentity, RemoteRunner

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeProcess:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv, *, timeout, check=False):
        self.calls.append(tuple(argv))
        return self.responder(tuple(argv))


def _identity(tmp_path: Path) -> RemoteIdentity:
    known = tmp_path / "known_hosts"
    known.write_text("example.test ssh-ed25519 AAAA\n", encoding="utf-8")
    return RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=None,
        known_hosts=known,
    )


def _snapshot(marker: str, run_id: str = "", nonce: str = "", *, commands=None, attempt=None, tool_presence=None, python3=True):
    return {
        "schema": observation.SNAPSHOT_SCHEMA,
        "marker": marker,
        "run_id": run_id,
        "nonce": nonce,
        "python3": python3,
        "tool_presence": dict(tool_presence or {}),
        "commands": list(commands or []),
        "attempt": attempt
        or {"state": "INCONCLUSIVE", "reason": "not_requested"},
        "partial": False,
        "budget_exhausted": False,
        "collector_budget_seconds": 6.0,
    }


def _snapshot_responder(*, commands=None, attempt=None, tool_presence=None, python3=True):
    def respond(argv):
        if Path(argv[0]).name.startswith("ssh-keygen"):
            return SimpleNamespace(
                returncode=0,
                stdout="example.test ssh-ed25519 AAAA\n",
                stderr="",
                timed_out=False,
            )
        remote = shlex.split(argv[-1])
        marker = remote[4]
        run_id = remote[11]
        nonce = remote[12]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                _snapshot(
                    marker,
                    run_id,
                    nonce,
                    commands=commands,
                    attempt=attempt,
                    tool_presence=tool_presence,
                    python3=python3,
                )
            ),
            stderr="",
            timed_out=False,
        )

    return respond


def _valid_decision(
    *,
    run_id: str,
    nonce: str,
    attempt_path: str,
    job_sha: str,
    launch_sha: str,
    process: str = "LIVE",
    status: str = "RUNNING",
    progress=None,
    heartbeat=None,
):
    command_sha = "c" * 64
    owner = {
        "schema": "anime-video-studio/p1-v3-process-owner/v1",
        "kind": "download_worker",
        "run_id": run_id,
        "nonce": nonce,
        "command_sha256": command_sha,
        "pid": 123,
        "argv_sha256": "d" * 64,
    }
    ack = dict(owner)
    proof = {
        "schema": "anime-video-studio/i1-process-proof/v1",
        "run_id": run_id,
        "nonce": nonce,
        "command_sha256": command_sha,
        "pid": 123,
        "start_ticks": 1,
        "exe": "/usr/bin/python3",
        "cmdline_sha256": "e" * 64,
        "boot_id": "boot",
        "birth_sha256": "f" * 64,
    }
    return {
        "state": "OK",
        "process": process,
        "run_id": run_id,
        "nonce": nonce,
        "attempt": attempt_path,
        "job_sha256": job_sha,
        "launch_sha256": launch_sha,
        "manifest_sha256": command_sha,
        "owner": owner,
        "ack": ack,
        "status": {
            "schema": "anime-video-studio/i1-download-status/v1",
            "run_id": run_id,
            "nonce": nonce,
            "status": status,
        },
        "progress": progress or {},
        "heartbeat": heartbeat or {},
        "proof": proof if process in {"LIVE", "DEAD"} else None,
    }


def _runner(tmp_path: Path, responder) -> tuple[RemoteRunner, FakeProcess]:
    fake = FakeProcess(responder)
    runner = RemoteRunner(
        ssh_path="ssh",
        scp_path="scp",
        ssh_keygen_path="ssh-keygen",
        runner=fake,
    )
    def with_input(identity, remote_args, input_bytes, *, timeout=None):
        assert tuple(remote_args) == ("sh", "-s")
        decoded = shlex.split(input_bytes.decode("utf-8"))
        assert decoded.pop(0) == "exec"
        return runner.run_remote(identity, decoded, timeout=timeout)

    runner.run_remote_with_input = with_input
    return runner, fake


def _context(tmp_path: Path) -> ops.Context:
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    workspace = "/workspace"
    key = state_mod.identity_key(
        host="example.test",
        user="root",
        port=22,
        fingerprint="SHA256:fake",
        workspace=workspace,
        profile_sha256=profile.sha256,
    )
    state = state_mod.new_state(
        identity_key_value=key,
        host="example.test",
        user="root",
        port=22,
        fingerprint="SHA256:fake",
        workspace=workspace,
        profile_id=profile.profile_id,
        profile_sha256=profile.sha256,
    )
    state = state_mod.transition(state, "INSPECTED")
    state = state_mod.transition(state, "PLANNED")
    state = state_mod.transition(state, "INSTALLING")
    state["remote_facts"] = {"home": "/root"}
    state["progress"] = {
        "files_done": 1,
        "files_total": 7,
        "bytes_done": 123,
        "bytes_total": 456,
        "speed_bytes_per_sec": 99,
        "eta_seconds": 10,
        "eta_known": True,
    }
    record = {
        "schema": state_mod.RECORD_SCHEMA,
        "name": "fake",
        "identity": {
            "host": "example.test",
            "user": "root",
            "port": 22,
            "fingerprint": "SHA256:fake",
            "workspace": workspace,
            "profile_id": profile.profile_id,
            "profile_sha256": profile.sha256,
        },
        "auth": {"kind": "agent"},
        "known_hosts": "unused",
        "created_at": state_mod.now_iso(),
        "updated_at": state_mod.now_iso(),
    }
    return ops.Context(
        state_dir=tmp_path,
        repo_root=ROOT,
        profile_path=ROOT / "config" / "i1" / "wan22-14b-ab.profile.json",
        profile=profile,
        name="fake",
        record=record,
        runner=SimpleNamespace(run_remote=lambda *a, **k: None),
        identity=object(),
        state=state,
        run_dir=tmp_path / "run",
    )


def _attempt_context(tmp_path: Path) -> ops.Context:
    ctx = _context(tmp_path)
    ctx.state["artifacts"] = {
        "setup": "COMPLETE",
        "download_attempt": {
            "id": "0001",
            "created_at": time.time() - 10,
            "deadline_epoch": time.time() + 3600,
            "dir": "/workspace/.avs-i1/attempts/0001",
            "launch_sha256": "a" * 64,
        },
    }
    return ctx


def test_classify_transport_vs_protocol_vs_proven() -> None:
    assert observation.classify_result(
        observation.ObservationResult(kind="", timed_out=True)
    ) == "TIMEOUT"
    assert observation.classify_result(
        observation.ObservationResult(
            kind="", returncode=255, stderr="Connection timed out during banner exchange"
        )
    ) == "CONNECTION_LOST"
    assert observation.classify_result(
        observation.ObservationResult(
            kind="", returncode=255, stderr="REMOTE HOST IDENTIFICATION HAS CHANGED"
        )
    ) == "HOST_KEY_CHANGED"
    assert observation.classify_result(
        observation.ObservationResult(
            kind="", returncode=255, stderr="Permission denied (publickey)"
        )
    ) == "AUTH_FAILED"
    assert observation.classify_result(
        observation.ObservationResult(kind="", returncode=0, stdout="not-json")
    ) == "PROTOCOL_ERROR"


def test_run_readonly_three_attempts_total_25_seconds() -> None:
    clock = Clock()

    class TimeoutRunner:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def run_remote(self, identity, argv, timeout=None):
            self.calls.append(float(timeout))
            clock.now += float(timeout)
            return SimpleNamespace(
                returncode=255,
                stdout="",
                stderr="Connection timed out during banner exchange",
                timed_out=True,
            )

    runner = TimeoutRunner()
    result = observation.run_readonly(
        runner,
        object(),
        ("observe",),
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.kind == "TIMEOUT"
    assert result.attempts == 3
    assert runner.calls == [8.0, 8.0, 7.5]
    assert clock.sleeps == [0.5, 1.0]
    assert result.elapsed_seconds == pytest.approx(25.0)


def test_host_key_change_is_not_retried() -> None:
    clock = Clock()

    class HostKeyRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_remote(self, identity, argv, timeout=None):
            self.calls += 1
            return SimpleNamespace(
                returncode=255,
                stdout="",
                stderr="REMOTE HOST IDENTIFICATION HAS CHANGED",
                timed_out=False,
            )

    runner = HostKeyRunner()
    result = observation.run_readonly(
        runner, object(), ("observe",), clock=clock, sleep=clock.sleep
    )
    assert result.kind == "HOST_KEY_CHANGED"
    assert runner.calls == 1
    assert clock.sleeps == []


def test_remote_host_key_error_is_hard_block() -> None:
    from scripts.p1.remote import RemoteHostKeyError

    clock = Clock()

    class RaisingRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_remote(self, identity, argv, timeout=None):
            self.calls += 1
            raise RemoteHostKeyError("host key changed")

    runner = RaisingRunner()
    result = observation.run_readonly(
        runner, object(), ("observe",), clock=clock, sleep=clock.sleep
    )
    assert result.kind == "HOST_KEY_CHANGED"
    assert runner.calls == 1
    assert clock.sleeps == []


def test_status_uses_one_ssh_snapshot(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.state["artifacts"] = {}
    runner, fake = _runner(tmp_path, _snapshot_responder(commands=[]))
    ctx.runner = runner
    ctx.identity = _identity(tmp_path)
    payload = ops.status(ctx)
    assert payload["effective_connection_state"] == "CURRENT"
    assert payload["worker_state"] == "UNKNOWN"
    assert len([call for call in fake.calls if Path(call[0]).name == "ssh"]) == 1


def test_snapshot_attempt_mode_uses_one_network_command(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    run_id = "a" * 32
    nonce = "b" * 32
    attempt_path = "/workspace/.avs-i1/attempts/0001"
    job_sha = "1" * 64
    launch_sha = "2" * 64
    decision = _valid_decision(
        run_id=run_id,
        nonce=nonce,
        attempt_path=attempt_path,
        job_sha=job_sha,
        launch_sha=launch_sha,
    )
    runner, fake = _runner(
        tmp_path,
        _snapshot_responder(
            attempt={"state": "OK", "decision": decision},
        ),
    )
    result = observation.capture_snapshot(
        runner,
        _identity(tmp_path),
        workspace="/workspace",
        profile=profile,
        root="/workspace/.avs-i1",
        attempt=attempt_path,
        run_id=run_id,
        nonce=nonce,
        mode="attempt",
        expected_job_sha256=job_sha,
        expected_launch_sha256=launch_sha,
    )
    assert result.kind == "OK"
    assert result.snapshot["attempt"]["decision"]["process"] == "LIVE"
    ssh_calls = [call for call in fake.calls if Path(call[0]).name == "ssh"]
    assert len(ssh_calls) == 1


def test_status_connection_lost_never_claims_worker_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    ctx.state["observation"] = {"last_confirmed_at": "2026-10-31T10:00:00+00:00"}
    monkeypatch.setattr(
        ops,
        "_observe_attempt",
        lambda *a, **k: observation.ObservationResult(
            kind="TIMEOUT",
            attempts=3,
            elapsed_seconds=25.0,
            diagnostic="Connection timed out during banner exchange",
        ),
    )
    payload = ops.status(ctx)
    assert payload["effective_connection_state"] == "CONNECTION_LOST"
    assert payload["worker_state"] == "UNKNOWN"
    assert payload["worker_status"] is None
    assert payload["progress"] is None
    assert payload["cached_progress"]["bytes_done"] == 123
    assert payload["cached_as_of"] == "2026-10-31T10:00:00+00:00"
    assert payload["speed_bytes_per_sec"] is None
    assert payload["eta_seconds"] is None
    assert payload["eta_known"] is False
    assert payload["readiness"]["status"] == "UNCONFIRMED"
    assert payload["readiness_current"] is False


def test_status_live_process_with_stale_heartbeat_is_not_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    decision = {
        "state": "OK",
        "process": "LIVE",
        "status": {"status": "RUNNING"},
        "progress": {"bytes_done": 10, "bytes_total": 100},
        "heartbeat": {"heartbeat_at": time.time() - 1000},
    }
    monkeypatch.setattr(
        ops,
        "_observe_attempt",
        lambda *a, **k: observation.ObservationResult(
            kind="OK",
            snapshot={"attempt": {"state": "OK", "proof": {"state": "OK", "value": decision}}},
            decision=decision,
        ),
    )
    payload = ops.status(ctx)
    assert payload["effective_connection_state"] == "CURRENT"
    assert payload["worker_state"] == "LIVE"
    assert payload["observation"]["proof_verified"] is True
    assert payload["observation"]["heartbeat_fresh"] is False
    assert payload["progress_fresh"] is True


def test_status_coherent_cache_and_reconnect_rate_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    run_id = ctx.state["run_id"]
    nonce = ctx.state["nonce"]

    def current(bytes_done: int, speed: float = 12.0):
        return observation.ObservationResult(
            kind="OK",
            decision={
                "state": "OK",
                "process": "LIVE",
                "status": {"status": "RUNNING"},
                "progress": {
                    "bytes_done": bytes_done,
                    "bytes_total": 1000,
                    "files_done": 1,
                    "files_total": 7,
                    "speed_bytes_per_sec": speed,
                    "eta_seconds": 10,
                    "eta_known": True,
                },
                "heartbeat": {"heartbeat_at": time.time()},
            },
        )

    responses = iter(
        [
            current(100),
            current(200),
            observation.ObservationResult(
                kind="TIMEOUT", attempts=3, diagnostic="banner exchange"
            ),
            current(250),
            current(260),
        ]
    )
    monkeypatch.setattr(ops, "_observe_attempt", lambda *a, **k: next(responses))
    first = ops.status(ctx)
    second = ops.status(ctx)
    lost = ops.status(ctx)
    assert second["last_confirmed_at"] != first["last_confirmed_at"]
    assert lost["effective_connection_state"] == "CONNECTION_LOST"
    assert lost["worker_state"] == "UNKNOWN"
    assert lost["cached_progress"]["bytes_done"] == 200
    assert lost["cached_as_of"] == second["last_confirmed_at"]
    assert lost["last_confirmed_at"] == second["last_confirmed_at"]
    assert lost["speed_bytes_per_sec"] is None
    assert lost["eta_seconds"] is None
    assert lost["eta_known"] is False
    reconnect = ops.status(ctx)
    assert reconnect["effective_connection_state"] == "CURRENT"
    assert reconnect["rate_fresh"] is False
    assert reconnect["speed_bytes_per_sec"] is None
    assert reconnect["eta_seconds"] is None
    fresh = ops.status(ctx)
    assert fresh["rate_fresh"] is True
    assert fresh["speed_bytes_per_sec"] == 12.0
    assert fresh["eta_seconds"] == 10.0


def test_stale_heartbeat_live_process_has_null_rate_in_both_renderers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = ops._observation_payload(
        observation.ObservationResult(
            kind="OK",
            decision={
                "state": "OK",
                "process": "LIVE",
                "status": {"status": "RUNNING"},
                "progress": {
                    "bytes_done": 100,
                    "bytes_total": 1000,
                    "files_done": 1,
                    "files_total": 7,
                    "speed_bytes_per_sec": 99,
                    "eta_seconds": 10,
                    "eta_known": True,
                },
                "heartbeat": {"heartbeat_at": time.time() - 503},
            },
        )
    )
    assert payload["connection_state"] == "CURRENT"
    assert payload["worker_state"] == "LIVE"
    assert payload["worker_responsiveness"] == "STALE"
    assert payload["rate_fresh"] is False
    cli._human_status(
        {
            "phase": "INSTALLING",
            "workspace": "/workspace",
            "effective_connection_state": "CURRENT",
            "worker_state": "LIVE",
            "worker_responsiveness": "STALE",
            "progress_fresh": True,
            "rate_fresh": False,
            "progress": {
                "bytes_done": 100,
                "bytes_total": 1000,
                "files_done": 1,
                "files_total": 7,
                "speed_bytes_per_sec": 99,
                "eta_seconds": 10,
                "eta_known": True,
            },
        }
    )
    human = capsys.readouterr().out
    assert "Responsiveness: STALE" in human
    assert "Скорость: неизвестно" in human
    assert "99" not in human
    printer = cli.ProgressPrinter()
    printer.last_at = 0.0
    printer(
        {
            "phase": "INSTALLING",
            "stage": "DOWNLOAD",
            "progress": {
                "stage": "DOWNLOAD",
                "bytes_done": 100,
                "bytes_total": 1000,
                "files_done": 1,
                "files_total": 7,
                "speed_bytes_per_sec": 99,
                "eta_seconds": 10,
                "eta_known": True,
            },
            "observation": {
                "connection_state": "CURRENT",
                "worker_state": "LIVE",
                "worker_responsiveness": "STALE",
                "heartbeat_fresh": False,
                "progress_fresh": True,
                "rate_fresh": False,
            },
        }
    )
    rendered = capsys.readouterr().out
    assert "heartbeat STALE" in rendered
    assert "99" not in rendered


def test_unknown_process_keeps_cached_bytes_without_live_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    ctx.state["observation"] = {
        "connection_state": "CURRENT",
        "cached_progress": {"bytes_done": 77, "bytes_total": 1000},
        "cached_as_of": "2026-11-01T00:00:00+00:00",
        "last_confirmed_at": "2026-11-01T00:00:00+00:00",
    }
    monkeypatch.setattr(
        ops,
        "_observe_attempt",
        lambda *a, **k: observation.ObservationResult(
            kind="OK", decision={"state": "INCONCLUSIVE", "reason": "birth missing"}
        ),
    )
    payload = ops.status(ctx)
    assert payload["effective_connection_state"] == "CURRENT"
    assert payload["worker_state"] == "UNKNOWN"
    assert payload["cached_progress"]["bytes_done"] == 77
    assert payload["speed_bytes_per_sec"] is None
    assert payload["eta_seconds"] is None


def test_permanent_banner_transport_failure_three_attempts_no_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    attempt_path = "/workspace/.avs-i1/attempts/0001"
    ctx.state["artifacts"]["download_attempt"].update(
        {
            "dir": attempt_path,
            "job_sha256": "1" * 64,
            "launch_sha256": "2" * 64,
        }
    )
    ssh_calls = {"n": 0}

    def respond(argv):
        if Path(argv[0]).name.startswith("ssh-keygen"):
            return SimpleNamespace(
                returncode=0,
                stdout="example.test ssh-ed25519 AAAA\n",
                stderr="",
                timed_out=False,
            )
        ssh_calls["n"] += 1
        return SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="Connection timed out during banner exchange",
            timed_out=True,
        )

    runner, _ = _runner(tmp_path, respond)
    ctx.runner = runner
    ctx.identity = _identity(tmp_path)
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no stage")),
    )
    monkeypatch.setattr(
        ops.download,
        "start",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no start")),
    )
    monkeypatch.setattr(
        ops.download,
        "recover",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no recover")),
    )
    payload = ops.status(ctx)
    assert ssh_calls["n"] == 3
    assert payload["effective_connection_state"] == "CONNECTION_LOST"
    assert payload["worker_state"] == "UNKNOWN"
    assert payload["speed_bytes_per_sec"] is None
    assert payload["eta_seconds"] is None
    assert payload["cached_progress"]["bytes_done"] == 123


def test_temporary_banner_then_live_attach_no_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    run_id = ctx.state["run_id"]
    nonce = ctx.state["nonce"]
    attempt_path = "/workspace/.avs-i1/attempts/0001"
    job_sha = "1" * 64
    launch_sha = "2" * 64
    ctx.state["artifacts"]["download_attempt"].update(
        {
            "dir": attempt_path,
            "job_sha256": job_sha,
            "launch_sha256": launch_sha,
        }
    )
    live = _valid_decision(
        run_id=run_id,
        nonce=nonce,
        attempt_path=attempt_path,
        job_sha=job_sha,
        launch_sha=launch_sha,
        process="LIVE",
        status="RUNNING",
    )
    complete = _valid_decision(
        run_id=run_id,
        nonce=nonce,
        attempt_path=attempt_path,
        job_sha=job_sha,
        launch_sha=launch_sha,
        process="ABSENT",
        status="COMPLETE",
    )
    responses = iter([live, complete])
    ssh_calls = {"n": 0}

    def respond(argv):
        if Path(argv[0]).name.startswith("ssh-keygen"):
            return SimpleNamespace(
                returncode=0,
                stdout="example.test ssh-ed25519 AAAA\n",
                stderr="",
                timed_out=False,
            )
        ssh_calls["n"] += 1
        if ssh_calls["n"] == 1:
            return SimpleNamespace(
                returncode=255,
                stdout="",
                stderr="Connection timed out during banner exchange",
                timed_out=True,
            )
        remote = shlex.split(argv[-1])
        payload = _snapshot(
            remote[4],
            run_id,
            nonce,
            attempt={"state": "OK", "decision": next(responses)},
        )
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr="", timed_out=False
        )

    runner, _ = _runner(tmp_path, respond)
    ctx.runner = runner
    ctx.identity = _identity(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **k: calls.append("root"))
    monkeypatch.setattr(ops, "_stage_setup", lambda *a, **k: calls.append("stage_setup"))
    monkeypatch.setattr(ops, "_run_setup", lambda *a, **k: calls.append("setup"))
    monkeypatch.setattr(ops.download, "stage", lambda *a, **k: calls.append("stage"))
    monkeypatch.setattr(ops.download, "start", lambda *a, **k: calls.append("start"))
    monkeypatch.setattr(ops.download, "recover", lambda *a, **k: calls.append("recover"))
    monkeypatch.setattr(ops.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        ops,
        "_finalize_after_download",
        lambda ctx, **kwargs: {"readiness": {"ready": True}, "state": ctx.state},
    )
    result = ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    assert result["readiness"]["ready"] is True
    assert calls == []
    assert ssh_calls["n"] >= 2


def test_resume_observes_existing_attempt_before_any_setup_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    decision = {
        "state": "OK",
        "process": "LIVE",
        "status": {"status": "RUNNING"},
        "progress": {},
        "heartbeat": {"heartbeat_at": time.time()},
        "manifest_sha256": "a" * 64,
    }
    calls: list[str] = []

    def observe(*args, **kwargs):
        calls.append("observe")
        return observation.ObservationResult(
            kind="OK",
            snapshot={"attempt": {"state": "OK", "proof": {"state": "OK", "value": decision}}},
            decision=decision,
        )

    monkeypatch.setattr(ops, "_observe_attempt", observe)
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **k: calls.append("root"))
    monkeypatch.setattr(ops, "_stage_setup", lambda *a, **k: calls.append("stage_setup"))
    monkeypatch.setattr(ops, "_run_setup", lambda *a, **k: calls.append("setup"))
    monkeypatch.setattr(ops.download, "stage", lambda *a, **k: calls.append("download_stage"))
    monkeypatch.setattr(ops.download, "start", lambda *a, **k: calls.append("start"))
    monkeypatch.setattr(ops.download, "recover", lambda *a, **k: calls.append("recover"))
    with pytest.raises(Interrupted):
        ops._continue_install(ctx, deadline_epoch=time.time() + 3600, new_attempt=True)
    assert calls == ["observe"]


def test_complete_while_offline_finalizes_without_second_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _attempt_context(tmp_path)
    decision = {
        "state": "OK",
        "process": "ABSENT",
        "status": {"status": "COMPLETE"},
        "progress": {},
        "heartbeat": {},
        "manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        ops,
        "_observe_attempt",
        lambda *a, **k: observation.ObservationResult(
            kind="OK",
            snapshot={"attempt": {"state": "OK", "proof": {"state": "OK", "value": decision}}},
            decision=decision,
        ),
    )
    monkeypatch.setattr(
        ops,
        "_finalize_after_download",
        lambda ctx, **kwargs: {"readiness": {"ready": True}, "state": ctx.state},
    )
    monkeypatch.setattr(
        ops.download,
        "stage",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no restage")),
    )
    monkeypatch.setattr(
        ops.download,
        "start",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no second launch")),
    )
    result = ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    assert result["readiness"]["ready"] is True


def test_missing_tool_valid_marker_is_proven_not_transport(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    commands = [
        {
            "key": observation.canonical_argv(("bash", "--version")),
            "exit": 127,
            "out": "",
            "err": "FileNotFoundError",
            "timed_out": False,
        },
        {
            "key": observation.canonical_argv(("python3", "--version")),
            "exit": 127,
            "out": "",
            "err": "FileNotFoundError",
            "timed_out": False,
        },
    ]
    runner, fake = _runner(tmp_path, _snapshot_responder(commands=commands))
    inspection = __import__("scripts.i1.inspect", fromlist=["inspect_remote"]).inspect_remote(
        runner, _identity(tmp_path), workspace="/workspace", profile=profile
    )
    assert len([call for call in fake.calls if Path(call[0]).name == "ssh"]) == 1
    codes = {item["code"] for item in inspection["blockers"]}
    assert "bash" in codes
    assert "python3" in codes
    assert "os" not in codes
    assert inspection["facts"]["os"] is None


def test_missing_python_fallback_is_proven_not_transport(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    runner, fake = _runner(
        tmp_path,
        _snapshot_responder(commands=[], tool_presence={"sh": True}, python3=False),
    )
    inspection = __import__("scripts.i1.inspect", fromlist=["inspect_remote"]).inspect_remote(
        runner, _identity(tmp_path), workspace="/workspace", profile=profile
    )
    codes = {item["code"] for item in inspection["blockers"]}
    assert "python3" in codes
    assert "os" not in codes
    assert inspection["facts"]["os"] is None
    assert inspection["facts"]["bash"] is None
    assert inspection["conclusive"] is False
    assert len([call for call in fake.calls if Path(call[0]).name == "ssh"]) == 1


def _capture_payload(tmp_path: Path, payload: dict, *, marker_override: str | None = None):
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    run_id = "a" * 32
    nonce = "b" * 32
    attempt_path = "/workspace/.avs-i1/attempts/0001"

    def respond(argv):
        if Path(argv[0]).name.startswith("ssh-keygen"):
            return SimpleNamespace(
                returncode=0,
                stdout="example.test ssh-ed25519 AAAA\n",
                stderr="",
                timed_out=False,
            )
        remote = shlex.split(argv[-1])
        body = dict(payload)
        body["marker"] = marker_override or remote[4]
        body["run_id"] = run_id
        body["nonce"] = nonce
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(body), stderr="", timed_out=False
        )

    runner, _ = _runner(tmp_path, respond)
    return observation.capture_snapshot(
        runner,
        _identity(tmp_path),
        workspace="/workspace",
        profile=profile,
        root="/workspace/.avs-i1",
        attempt=attempt_path,
        run_id=run_id,
        nonce=nonce,
        mode="attempt",
        expected_job_sha256="1" * 64,
        expected_launch_sha256="2" * 64,
    )


def _valid_snapshot_payload() -> dict:
    run_id = "a" * 32
    nonce = "b" * 32
    attempt_path = "/workspace/.avs-i1/attempts/0001"
    decision = _valid_decision(
        run_id=run_id,
        nonce=nonce,
        attempt_path=attempt_path,
        job_sha="1" * 64,
        launch_sha="2" * 64,
    )
    return _snapshot(
        "0" * 32,
        run_id,
        nonce,
        attempt={"state": "OK", "decision": decision},
    )


def test_snapshot_wrong_marker_is_protocol_error(tmp_path: Path) -> None:
    result = _capture_payload(
        tmp_path, _valid_snapshot_payload(), marker_override="x" * 32
    )
    assert result.kind == "PROTOCOL_ERROR"
    assert result.snapshot is None


def test_snapshot_duplicate_fields_is_protocol_error(tmp_path: Path) -> None:
    payload = json.dumps(_valid_snapshot_payload())
    duplicate = payload[:-1] + ',"schema":"duplicate"}'

    def respond(argv):
        if Path(argv[0]).name.startswith("ssh-keygen"):
            return SimpleNamespace(
                returncode=0,
                stdout="example.test ssh-ed25519 AAAA\n",
                stderr="",
                timed_out=False,
            )
        return SimpleNamespace(returncode=0, stdout=duplicate, stderr="", timed_out=False)

    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    runner, _ = _runner(tmp_path, respond)
    result = observation.capture_snapshot(
        runner,
        _identity(tmp_path),
        workspace="/workspace",
        profile=profile,
        root="/workspace/.avs-i1",
        attempt="/workspace/.avs-i1/attempts/0001",
        run_id="a" * 32,
        nonce="b" * 32,
        mode="attempt",
        expected_job_sha256="1" * 64,
        expected_launch_sha256="2" * 64,
    )
    assert result.kind == "PROTOCOL_ERROR"


def test_snapshot_weird_boolean_is_protocol_error(tmp_path: Path) -> None:
    payload = _valid_snapshot_payload()
    payload["python3"] = 1
    result = _capture_payload(tmp_path, payload)
    assert result.kind == "PROTOCOL_ERROR"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("run_id", "c" * 32),
        lambda d: d.__setitem__("nonce", "d" * 32),
        lambda d: d.__setitem__("attempt", "/workspace/.avs-i1/attempts/9999"),
        lambda d: d.__setitem__("job_sha256", "3" * 64),
        lambda d: d.__setitem__("launch_sha256", "4" * 64),
        lambda d: d.__setitem__("proof", None),
        lambda d: d["proof"].__setitem__("pid", True),
    ],
)
def test_snapshot_proof_identity_mismatch_is_protocol_error(
    tmp_path: Path, mutate
) -> None:
    payload = _valid_snapshot_payload()
    mutate(payload["attempt"]["decision"])
    result = _capture_payload(tmp_path, payload)
    assert result.kind == "PROTOCOL_ERROR"


def test_malformed_snapshot_is_inconclusive_not_missing(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")

    def respond(argv):
        if Path(argv[0]).name.startswith("ssh-keygen"):
            return SimpleNamespace(
                returncode=0, stdout="example.test ssh-ed25519 AAAA\n", stderr="", timed_out=False
            )
        return SimpleNamespace(returncode=0, stdout="not-json", stderr="", timed_out=False)

    runner, _ = _runner(tmp_path, respond)
    inspection = __import__("scripts.i1.inspect", fromlist=["inspect_remote"]).inspect_remote(
        runner, _identity(tmp_path), workspace="/workspace", profile=profile
    )
    assert inspection["conclusive"] is False
    assert inspection["blockers"] == []


def test_observation_loss_does_not_mutate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _attempt_context(tmp_path)
    monkeypatch.setattr(
        ops,
        "_observe_attempt",
        lambda *a, **k: observation.ObservationResult(
            kind="CONNECTION_LOST", diagnostic="banner exchange", attempts=3
        ),
    )
    calls: list[str] = []
    monkeypatch.setattr(ops, "prepare_remote_root", lambda **k: calls.append("root"))
    monkeypatch.setattr(ops, "_stage_setup", lambda *a, **k: calls.append("stage_setup"))
    monkeypatch.setattr(ops, "_run_setup", lambda *a, **k: calls.append("setup"))
    monkeypatch.setattr(ops.download, "stage", lambda *a, **k: calls.append("download_stage"))
    monkeypatch.setattr(ops.download, "start", lambda *a, **k: calls.append("start"))
    monkeypatch.setattr(ops.download, "recover", lambda *a, **k: calls.append("recover"))
    with pytest.raises(Interrupted):
        ops._continue_install(ctx, deadline_epoch=time.time() + 3600)
    assert calls == []


def test_heartbeat_clock_skew_is_unknown_not_fresh() -> None:
    payload = ops._observation_payload(
        observation.ObservationResult(
            kind="OK",
            decision={
                "state": "OK",
                "process": "LIVE",
                "heartbeat": {"heartbeat_at": time.time() + 3600},
                "progress": {},
            },
        )
    )
    assert payload["heartbeat_age_seconds"] is None
    assert payload["heartbeat_fresh"] is None


def test_real_transport_long_observation_uses_short_argv_and_stdin(tmp_path, monkeypatch):
    runner = RemoteRunner(ssh_path="ssh", scp_path="scp", ssh_keygen_path="ssh-keygen")
    monkeypatch.setattr(runner, "check_host_key", lambda *a, **k: None)
    captured = {}
    def bounded(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"schema": observation.SNAPSHOT_SCHEMA, "marker": "a" * 32}).encode(), stderr=b"", timed_out=False)
    monkeypatch.setattr("scripts.p1.remote.run_bounded_process", bounded)
    original = ("sh", "-c", "printf '%s' \"$1\"", "probe", "x" * 80000 + " quote' \n unicode-кадр")
    result = observation.run_readonly(runner, _identity(tmp_path), original)
    assert result.kind == "OK", (result.stderr, result.diagnostic)
    assert sum(map(len, captured["argv"])) < 4096
    assert shlex.split(captured["input_bytes"].decode("utf-8")) == ["exec", *original]
