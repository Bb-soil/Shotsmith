from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from scripts.p1 import clore_export as ce
from scripts.p1 import clore_session as cs
from scripts.p1 import clore_startup as cst
from scripts.p1.clore_bridge import CloreCancelAck, CloreOrder

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "p1" / "clore-v1"
CREATED = 1761090000
START = datetime.fromtimestamp(CREATED, tz=timezone.utc)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _quote() -> cs.CloreQuote:
    data = json.loads((FIXTURE_DIR / "quote-ready-example.json").read_text(encoding="utf-8"))
    return cs.CloreQuote.from_dict(data)


def _order(**overrides: Any) -> CloreOrder:
    base = CloreOrder(
        order_id=7001,
        server_id=112213,
        created_at=CREATED,
        price_per_day=Decimal("13.2"),
        currency="USD-Blockchain",
        expired=False,
        fee_percent=Decimal("5"),
        creation_fee=Decimal("0.10"),
        spend=Decimal("0"),
        spot=False,
        mrl_seconds=518400,
    )
    return replace(base, **overrides)


class _FakeClock:
    def __init__(self, start: datetime = START) -> None:
        self.now = start
        self.mono = 0.0

    def clock(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.mono

    async def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)
        self.mono += seconds


class _Bridge:
    def __init__(self) -> None:
        self.cancel_calls = 0

    async def read_order(
        self, order_id: int, *, deadline_monotonic: float | None = None
    ) -> CloreOrder:
        if self.cancel_calls:
            return _order(expired=True)
        return _order()

    async def cancel_order(
        self, order_id: int, *, deadline_monotonic: float | None = None
    ) -> CloreCancelAck:
        self.cancel_calls += 1
        return CloreCancelAck(0, True, False, "ack")


# ---------------------------------------------------------------------------
# Startup plan
# ---------------------------------------------------------------------------


def test_startup_default_is_blocked_and_has_no_public_http_ports() -> None:
    readiness = cst.assess_startup()
    assert readiness.live_ready is False
    assert any("entrypoint" in blocker for blocker in readiness.blockers)
    assert any("digest" in blocker for blocker in readiness.blockers)
    assert any("UID/HOME" in blocker for blocker in readiness.blockers)
    assert any("onstart" in blocker for blocker in readiness.blockers)
    text = readiness.plan
    assert "127.0.0.1:8188" in text
    assert "22/tcp" in text
    assert "no public 8188/8888" in text
    assert "BEGIN OPENSSH PRIVATE KEY" not in text
    assert "ssh_password" not in text


def test_startup_ready_requires_every_observed_fact() -> None:
    evidence = cst.StartupEvidence(
        base_entrypoint_inspected=True,
        actual_image_digest=cst.EXPECTED_IMAGE_DIGEST,
        uid_home_python_torch_verified=True,
        onstart_script_verified=True,
        ssh_public_key_only=True,
        only_tcp22=True,
        comfyui_loopback=True,
        no_public_jupyter=True,
    )
    readiness = cst.assess_startup(evidence)
    assert readiness.live_ready is True
    assert readiness.blockers == ()


def test_startup_digest_mismatch_keeps_launch_blocked() -> None:
    evidence = cst.StartupEvidence(
        base_entrypoint_inspected=True,
        actual_image_digest="sha256:" + "0" * 64,
        uid_home_python_torch_verified=True,
        onstart_script_verified=True,
        ssh_public_key_only=True,
        only_tcp22=True,
        comfyui_loopback=True,
        no_public_jupyter=True,
    )
    readiness = cst.assess_startup(evidence)
    assert readiness.live_ready is False
    assert any("digest" in blocker for blocker in readiness.blockers)


def test_startup_cli_subprocess_prints_nonempty_expected_plan(tmp_path: Path) -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "scripts/p1/clore_startup.py", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip()
    payload = json.loads(completed.stdout)
    assert payload["status"] == "CLORE_STARTUP_PLAN"
    assert payload["live_ready"] is False
    assert payload["blockers"]
    assert "127.0.0.1:8188" in payload["plan"]


# ---------------------------------------------------------------------------
# Local staged export process
# ---------------------------------------------------------------------------


def _plan(tmp_path: Path, *, items: tuple[ce.ExportItem, ...] | None = None) -> ce.ExportPlan:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "output.mp4").write_bytes(b"fixture-video")
    (staging / "input.png").write_bytes(b"fixture-input")
    (staging / "seed.txt").write_bytes(b"424242")
    archive = tmp_path / "archive"
    return ce.build_export_plan(
        archive,
        source_dir=staging,
        items=items
        or (
            ce.ExportItem("input.png", "inputs/input.png"),
            ce.ExportItem("output.mp4", "artifact.mp4"),
            ce.ExportItem("seed.txt", "inputs/seed.txt"),
        ),
    )


def test_default_plan_includes_original_input_image() -> None:
    sources = {source for source, _target, _required in ce.DEFAULT_EXPORT_ITEMS}
    assert "input.png" in sources
    assert "output.mp4" in sources
    assert "cost-ledger.json" in sources


def test_local_export_complete_atomic_manifest_and_hashes(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fixed = datetime.now(timezone.utc)
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed + timedelta(seconds=20),
            now=lambda: fixed,
        )
    )
    assert result.status == "COMPLETE"
    assert result.lost == ()
    assert result.manifest_path is not None
    archive = result.run_dir
    assert archive is not None
    assert (archive / "artifact.mp4").read_bytes() == b"fixture-video"
    assert (archive / "inputs" / "input.png").read_bytes() == b"fixture-input"
    assert (archive / "inputs" / "seed.txt").read_bytes() == b"424242"
    assert not list(archive.rglob("*.part-*"))
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == ce.EXPORT_SCHEMA
    assert manifest["status"] == "COMPLETE"
    expected = hashlib.sha256(b"fixture-video").hexdigest()
    assert any(item["sha256"] == expected for item in manifest["completed"])
    manifest_hash = (archive / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    assert manifest_hash == hashlib.sha256(
        (archive / "manifest.json").read_bytes()
    ).hexdigest()


def test_export_path_traversal_and_duplicate_targets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ce.ExportError):
        ce.build_export_plan(
            tmp_path / "archive",
            source_dir=tmp_path / "staging",
            items=(ce.ExportItem("a", "../escape"),),
        )
    with pytest.raises(ce.ExportError):
        ce.build_export_plan(
            tmp_path / "archive",
            source_dir=tmp_path / "staging",
            items=(
                ce.ExportItem("a", "same"),
                ce.ExportItem("b", "same"),
            ),
        )


def test_export_failure_records_loss_and_never_finalizes(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        items=(
            ce.ExportItem("missing.mp4", "artifact.mp4"),
            ce.ExportItem("seed.txt", "inputs/seed.txt"),
        ),
    )
    fixed = datetime.now(timezone.utc)
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed + timedelta(seconds=20),
            now=lambda: fixed,
        )
    )
    assert result.status == "PARTIAL"
    assert [item["source"] for item in result.lost] == ["missing.mp4"]
    archive = result.run_dir
    assert archive is not None
    assert not (archive / "artifact.mp4").exists()
    assert (archive / "inputs" / "seed.txt").exists()
    assert not list(archive.rglob("*.part-*"))


def test_export_stream_cap_aborts_before_finalize(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        items=(ce.ExportItem("output.mp4", "artifact.mp4"),),
    )
    fixed = datetime.now(timezone.utc)
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed + timedelta(seconds=20),
            now=lambda: fixed,
            max_item_bytes=4,
        )
    )
    assert result.status == "FAILED"
    assert result.lost[0]["reason"] == "ExportError"
    archive = result.run_dir
    assert archive is not None
    assert not (archive / "artifact.mp4").exists()
    assert not list(archive.rglob("*.part-*"))


def test_export_hang_is_bounded_killed_and_cleaned(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    started = time.perf_counter()
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=datetime.now(timezone.utc) + timedelta(seconds=0.25),
            worker_delay_seconds=10.0,
        )
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0
    assert result.status == "FAILED"
    assert result.lost
    assert not list((tmp_path / "archive").rglob(".clore-export-*"))


def test_export_expected_hash_mismatch_records_loss(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        items=(
            ce.ExportItem(
                "output.mp4",
                "artifact.mp4",
                expected_sha256="0" * 64,
            ),
        ),
    )
    fixed = datetime.now(timezone.utc)
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed + timedelta(seconds=20),
            now=lambda: fixed,
        )
    )
    assert result.status == "FAILED"
    assert result.lost
    if result.run_dir is not None:
        assert not (result.run_dir / "artifact.mp4").exists()


def test_export_deadline_already_expired_makes_no_process(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fixed = datetime.now(timezone.utc)
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed - timedelta(seconds=1),
            now=lambda: fixed,
        )
    )
    assert result.status == "FAILED"
    assert result.manifest_path is None
    assert result.run_dir is None
    assert all("deadline" in item["reason"] for item in result.lost)


def test_export_does_not_delay_independent_cancel(tmp_path: Path) -> None:
    quote = _quote()
    state = cs.initialize_state(
        tmp_path / "state.json",
        order_id=7001,
        created_at=CREATED,
        quote=quote,
        now=START + timedelta(seconds=5),
    )
    assert state.status == "ARMED"
    bridge = _Bridge()
    clock = _FakeClock(START + timedelta(seconds=300))
    session = cs.CloreSession(
        tmp_path / "state.json",
        bridge=bridge,
        clock=clock.clock,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    plan = _plan(tmp_path)

    async def scenario() -> None:
        started = asyncio.Event()
        export_task = asyncio.create_task(
            ce.run_staged_export(
                plan,
                deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
                worker_delay_seconds=10.0,
                on_started=started.set,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5.0)
        heartbeat = 0
        heartbeat += 1
        outcome = await session.stop_now(reason="generation finished early")
        assert outcome.status == "ENDED_REPORTED"
        assert bridge.cancel_calls == 1
        assert heartbeat == 1
        export_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await export_task

    _run(scenario())
    assert not list((tmp_path / "archive").rglob("*.part-*"))
    assert not list((tmp_path / "archive").rglob(".clore-export-*"))


def test_concurrent_exports_use_distinct_owned_namespaces(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fixed = datetime.now(timezone.utc)

    async def scenario() -> list[ce.ExportResult]:
        return await asyncio.gather(
            ce.run_staged_export(
                plan,
                deadline_at=fixed + timedelta(seconds=30),
                now=lambda: fixed,
            ),
            ce.run_staged_export(
                plan,
                deadline_at=fixed + timedelta(seconds=30),
                now=lambda: fixed,
            ),
        )

    results = _run(scenario())
    assert all(result.status == "COMPLETE" for result in results)
    dirs = {result.run_dir for result in results}
    assert len(dirs) == 2
    for result in results:
        assert result.run_dir is not None
        assert (result.run_dir / "artifact.mp4").read_bytes() == b"fixture-video"


def test_parent_does_not_call_sync_fs_or_hash_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    fixed = datetime.now(timezone.utc)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("parent must not run sync filesystem/hash helpers")

    monkeypatch.setattr(ce, "_atomic_write_text", forbidden)
    monkeypatch.setattr(ce, "_hash_file", forbidden)
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed + timedelta(seconds=30),
            now=lambda: fixed,
        )
    )
    assert result.status == "COMPLETE"
    assert result.run_dir is not None

def test_reap_reports_success_only_after_final_wait_confirmation() -> None:
    class FakeProcess:
        def __init__(self, success: bool) -> None:
            self.success = success
            self.returncode = None
            self.kills = 0
            self.wait_calls = 0

        def kill(self) -> None:
            self.kills += 1

        async def wait(self) -> int:
            self.wait_calls += 1
            if self.success and self.wait_calls >= 2:
                self.returncode = 0
                return 0
            raise asyncio.TimeoutError()

    async def scenario() -> tuple[bool, bool, FakeProcess, FakeProcess]:
        good = FakeProcess(True)
        bad = FakeProcess(False)
        good_result = await ce._kill_and_reap(good)  # type: ignore[arg-type]
        bad_result = await ce._kill_and_reap(bad)  # type: ignore[arg-type]
        return good_result, bad_result, good, bad

    good_result, bad_result, good, bad = _run(scenario())
    assert good_result is True
    assert good.wait_calls >= 2
    assert good.kills == 2
    assert bad_result is False
    assert bad.wait_calls >= 2
    assert bad.kills == 2
def test_manifest_phase_slow_worker_is_bounded_and_cleaned(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    started = time.perf_counter()
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=datetime.now(timezone.utc) + timedelta(seconds=0.25),
            manifest_delay_seconds=10.0,
        )
    )
    assert time.perf_counter() - started < 3.0
    assert result.status == "FAILED"
    assert not list((tmp_path / "archive").rglob(".clore-export-*"))
def test_large_result_over_64k_many_items_is_success(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    items: list[ce.ExportItem] = []
    for index in range(500):
        source = f"f{index}.txt"
        target = f"many/f{index}.txt"
        (staging / source).write_bytes(b"x")
        items.append(ce.ExportItem(source, target))
    plan = ce.build_export_plan(
        tmp_path / "archive", source_dir=staging, items=tuple(items)
    )
    fixed = datetime.now(timezone.utc)
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed + timedelta(seconds=30),
            now=lambda: fixed,
        )
    )
    assert result.status == "COMPLETE"
    assert len(json.dumps(result.completed)) > 65536
    assert result.run_dir is not None
    assert (result.run_dir / "many" / "f499.txt").exists()


def test_oversized_result_is_structured_failure_and_cleaned(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fixed = datetime.now(timezone.utc)
    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed + timedelta(seconds=20),
            now=lambda: fixed,
            result_padding_bytes=1_100_000,
        )
    )
    assert result.status in {"FAILED", "FAILED_CLEANUP_INCOMPLETE"}
    assert "exceeded the bound" in result.message
    assert result.status != "COMPLETE"
    assert not list((tmp_path / "archive").rglob(".clore-export-*"))


def test_spawn_pending_returns_bounded_and_late_process_is_reaped(
    tmp_path: Path,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.kills = 0
            self.waits = 0

        def kill(self) -> None:
            self.kills += 1
            self.returncode = 0

        async def wait(self) -> int:
            self.waits += 1
            self.returncode = 0
            return 0

    release = asyncio.Event()
    fake = FakeProcess()

    async def slow_spawn() -> FakeProcess:
        await release.wait()
        return fake

    async def scenario() -> FakeProcess:
        process, state = await ce._spawn_owned(
            slow_spawn(), timeout=0.02, root=tmp_path, run_id=None
        )
        assert process is None
        assert state == "pending"
        release.set()
        for _ in range(100):
            if not ce._LATE_SPAWN_TASKS and not ce._LATE_REAP_TASKS:
                break
            await asyncio.sleep(0.01)
        return fake

    fake_result = _run(scenario())
    assert fake_result.kills >= 1
    assert fake_result.waits >= 1


def test_spawn_cancellation_retains_late_ownership(tmp_path: Path) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.kills = 0

        def kill(self) -> None:
            self.kills += 1
            self.returncode = 0

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    release = asyncio.Event()
    fake = FakeProcess()

    async def slow_spawn() -> FakeProcess:
        await release.wait()
        return fake

    async def scenario() -> FakeProcess:
        task = asyncio.create_task(
            ce._spawn_owned(slow_spawn(), timeout=10.0, root=tmp_path, run_id=None)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(100):
            if not ce._LATE_SPAWN_TASKS and not ce._LATE_REAP_TASKS:
                break
            await asyncio.sleep(0.01)
        return fake

    fake_result = _run(scenario())
    assert fake_result.kills >= 1
def _drain(buffer: bytes) -> ce._StdoutDrain:
    drain = ce._StdoutDrain()
    drain.buffer.extend(buffer)
    return drain


def _started_line(run_id: str = "a" * 32) -> bytes:
    return (
        b'{"event":"STARTED","schema":"anime-video-studio/p1-clore-export/v1",'
        + b'"run_id":"'
        + run_id.encode("ascii")
        + b'"}\n'
    )


def test_strict_protocol_rejects_plain_started_early_result_and_legacy_frames() -> None:
    plain = _drain(b'STARTED\nRESULT 2\n{}')
    assert ce._parse_stdout_frames(plain)[1] == "worker STARTED frame is invalid"
    early = _drain(b'RESULT 2\n{}')
    assert (
        ce._parse_stdout_frames(early)[1]
        == "worker did not start with the strict STARTED frame"
    )
    legacy = _drain(b'{"schema":"anime-video-studio/p1-clore-export/v1"}\n')
    assert (
        ce._parse_stdout_frames(legacy)[1]
        == "worker did not start with the strict STARTED frame"
    )


def test_strict_result_frame_length_limits_and_trailing_bytes() -> None:
    prefix = _started_line()
    huge = _drain(prefix + b'RESULT 999999999999\n')
    assert (
        ce._parse_stdout_frames(huge)[1]
        == "worker result length exceeds the bound"
    )
    truncated = _drain(prefix + b'RESULT 5\nabc')
    assert (
        ce._parse_stdout_frames(truncated)[1]
        == "worker result frame is truncated"
    )
    trailing = _drain(prefix + b'RESULT 2\n{}{}')
    assert (
        ce._parse_stdout_frames(trailing)[1]
        == "worker result frame has trailing bytes"
    )
    long_header = _drain(prefix + b"RESULT " + b"9" * 100 + b"\n")
    assert (
        ce._parse_stdout_frames(long_header)[1]
        == "worker result frame header is too long"
    )


def _valid_payload(tmp_path: Path, run_id: str) -> dict[str, Any]:
    run_dir = tmp_path / "archive" / (".clore-export-" + run_id)
    return {
        "schema": ce.EXPORT_SCHEMA,
        "run_id": run_id,
        "status": "COMPLETE",
        "completed": [
            {
                "target": "a.txt",
                "source": "a.txt",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        ],
        "lost": [],
        "manifest_path": str(run_dir / "manifest.json"),
        "manifest_sha256": "1" * 64,
        "run_dir": str(run_dir),
        "message": "ok",
    }


def test_strict_payload_validation_rejects_mismatch_and_coverage(tmp_path: Path) -> None:
    plan = ce.build_export_plan(
        tmp_path / "archive",
        source_dir=tmp_path / "staging",
        items=(ce.ExportItem("a.txt", "a.txt"),),
    )
    run_id = "a" * 32
    valid = _valid_payload(tmp_path, run_id)
    assert ce._validate_result_payload(valid, plan, run_id) is None
    wrong_run = dict(valid)
    wrong_run["run_id"] = "b" * 32
    assert ce._validate_result_payload(wrong_run, plan, run_id)
    wrong_schema = dict(valid)
    wrong_schema["schema"] = "other"
    assert ce._validate_result_payload(wrong_schema, plan, run_id)
    missing_manifest = dict(valid)
    missing_manifest["manifest_path"] = None
    assert ce._validate_result_payload(missing_manifest, plan, run_id)
    duplicate = dict(valid)
    duplicate["completed"] = valid["completed"] + valid["completed"]
    assert ce._validate_result_payload(duplicate, plan, run_id)
    unplanned = dict(valid)
    unplanned["completed"] = [
        {
            "target": "other.txt",
            "source": "other.txt",
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
    ]
    assert ce._validate_result_payload(unplanned, plan, run_id)
    missing = dict(valid)
    missing["completed"] = []
    assert ce._validate_result_payload(missing, plan, run_id)
    false_success = dict(valid)
    false_success["status"] = "FAILED"
    assert ce._validate_result_payload(false_success, plan, run_id)


def test_repeated_exports_leave_no_helper_tasks(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fixed = datetime.now(timezone.utc)

    async def scenario() -> list[Any]:
        before = set(asyncio.all_tasks())
        for _ in range(3):
            result = await ce.run_staged_export(
                plan,
                deadline_at=fixed + timedelta(seconds=30),
                now=lambda: fixed,
            )
            assert result.status == "COMPLETE"
        await asyncio.sleep(0.05)
        current = asyncio.current_task()
        return [task for task in asyncio.all_tasks() if task not in before and task is not current and not task.done()]

    assert _run(scenario()) == []


def test_drain_owned_spawns_reports_pending_then_completes(tmp_path: Path) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.kills = 0

        def kill(self) -> None:
            self.kills += 1
            self.returncode = 0

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    release = asyncio.Event()
    fake = FakeProcess()

    async def slow_spawn() -> FakeProcess:
        await release.wait()
        return fake

    async def scenario() -> tuple[dict[str, Any], dict[str, Any], FakeProcess]:
        process, state = await ce._spawn_owned(
            slow_spawn(), timeout=0.01, root=tmp_path, run_id=None
        )
        assert process is None and state == "pending"
        pending = await ce.drain_owned_spawns(0.02)
        release.set()
        complete = await ce.drain_owned_spawns(2.0)
        return pending, complete, fake

    pending, complete, fake_result = _run(scenario())
    assert pending["complete"] is False
    assert complete["complete"] is True
    assert fake_result.kills >= 1
def test_late_unreaped_outcome_is_retained_and_drain_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ce, "_LATE_OUTCOMES", [])

    class NeverReaped:
        returncode = None

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            raise asyncio.TimeoutError()

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        run_id = "a" * 32
        await ce._late_reap(NeverReaped(), tmp_path, run_id)  # type: ignore[arg-type]
        first = await ce.drain_owned_spawns(0.1)
        second = await ce.drain_owned_spawns(0.1)
        return first, second

    first, second = _run(scenario())
    assert first["tasks_done"] is True
    assert first["reap_incomplete"] == 1
    assert first["complete"] is False
    assert first["outcomes"][0]["reaped_confirmed"] is False
    assert first["outcomes"][0]["error_code"] == "reap_unconfirmed"
    assert second["outcomes"] == first["outcomes"]
    assert second["complete"] is False


def test_late_reaped_cleanup_false_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ce, "_LATE_OUTCOMES", [])

    class Reaped:
        returncode = None

        def kill(self) -> None:
            self.returncode = 0

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    async def cleanup_false(_root: Path, _run_id: str) -> bool:
        return False

    monkeypatch.setattr(ce, "_cleanup_run", cleanup_false)

    async def scenario() -> dict[str, Any]:
        run_id = "b" * 32
        await ce._late_reap(Reaped(), tmp_path, run_id)  # type: ignore[arg-type]
        return await ce.drain_owned_spawns(0.1)

    result = _run(scenario())
    assert result["tasks_done"] is True
    assert result["cleanup_incomplete"] == 1
    assert result["complete"] is False
    assert result["outcomes"][0]["cleanup_confirmed"] is False
    assert result["outcomes"][0]["error_code"] == "cleanup_unconfirmed"


def test_early_result_then_hang_fails_fast_without_waiting_for_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeStdin:
        def write(self, _data: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeProcess:
        def __init__(self, first: bytes) -> None:
            self.returncode = None
            self.stdin = FakeStdin()
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(first)
            self._release = asyncio.Event()

        def kill(self) -> None:
            self.returncode = 0
            self._release.set()

        async def wait(self) -> int:
            await self._release.wait()
            return self.returncode or 0

    async def fake_spawn(*_args: Any, **_kwargs: Any) -> FakeProcess:
        return FakeProcess(b'RESULT 2\n{}')

    monkeypatch.setattr(ce.asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setattr(ce, "_cleanup_run", lambda *_args, **_kwargs: _true_async())
    started = time.perf_counter()
    result = _run(
        ce.run_staged_export(
            _plan(tmp_path),
            deadline_at=datetime.now(timezone.utc) + timedelta(seconds=30),
        )
    )
    assert time.perf_counter() - started < 2.0
    assert result.status in {"FAILED", "FAILED_CLEANUP_INCOMPLETE"}
    assert "STARTED" in result.message


def _true_async():
    async def value() -> bool:
        return True

    return value()
def test_on_started_exception_is_sanitized_and_cleanup_runs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    fixed = datetime.now(timezone.utc)
    secret = "synthetic-secret-on-started"

    def raise_secret() -> None:
        raise RuntimeError(secret)

    result = _run(
        ce.run_staged_export(
            plan,
            deadline_at=fixed + timedelta(seconds=20),
            now=lambda: fixed,
            on_started=raise_secret,
        )
    )
    rendered = json.dumps(result.__dict__, default=str)
    assert result.status in {"FAILED", "FAILED_CLEANUP_INCOMPLETE"}
    assert secret not in rendered
    assert not list((tmp_path / "archive").rglob(".clore-export-*"))


def test_no_raw_traceback_debug_prints_in_clore_export() -> None:
    text = Path(ce.__file__).read_text(encoding="utf-8")
    assert "traceback.print_exc" not in text
    assert "print_exc" not in text