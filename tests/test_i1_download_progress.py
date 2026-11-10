from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from scripts.i1 import worker
from scripts.p1 import setup_plan


def _progress(tmp_path: Path, files: int = 2, total: int = 200) -> worker.Progress:
    return worker.Progress(
        progress_path=tmp_path / "progress.json",
        status_path=tmp_path / "status.json",
        heartbeat_path=tmp_path / "heartbeat.json",
        run_id="run-1",
        nonce="nonce-1",
        files=files,
        total=total,
    )


def _read(path: Path) -> dict:
    last: Exception | None = None
    for _ in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.01)
    raise AssertionError(f"cannot read {path}: {last}")


@pytest.mark.parametrize("count", [4, 7])
def test_active_slots_are_assigned_only_and_pending_is_separate(
    tmp_path: Path, count: int
) -> None:
    total = count * 100
    p = _progress(tmp_path, files=count, total=total)
    for index in range(count):
        p.add_file(
            f"m{index}",
            f"models/m{index}.bin",
            100,
            initial_written=10 if index >= 2 else 0,
        )
    p.start()
    try:
        p.set_stage("DOWNLOADING", "m0")
        p.set_stage("DOWNLOADING", "m1")
        time.sleep(1.1)
        payload = _read(tmp_path / "progress.json")
        assert payload["files_active"] == 2
        assert payload["current_files"] == ["m0", "m1"]
        assert payload["files_pending"] == count - 2
        assert payload["pending_files"] == [f"m{i}" for i in range(2, count)]
        assert payload["bytes_done"] == (count - 2) * 10
    finally:
        p.stop()


def test_two_active_files_aggregate_bytes_and_eta_unknown_explicit(tmp_path: Path) -> None:
    p = _progress(tmp_path, files=2, total=200)
    p.add_file("a", "models/a.bin", 100)
    p.add_file("b", "models/b.bin", 100)
    p.start()
    try:
        p.set_stage("DOWNLOADING", "a")
        p.set_stage("DOWNLOADING", "b")
        p.callback("a", 40, 100)
        p.callback("b", 30, 100)
        time.sleep(1.25)
        progress = _read(tmp_path / "progress.json")
        assert progress["bytes_done"] == 70
        assert progress["bytes_remaining"] == 130
        assert progress["files_active"] == 2
        assert set(progress["current_files"]) == {"a", "b"}
        assert isinstance(progress["speed_bytes_per_sec"], (int, float))
        assert progress["eta_seconds"] is None
        assert progress["eta_known"] is False
    finally:
        p.stop()


def test_completed_and_active_bytes_do_not_double_count(tmp_path: Path) -> None:
    p = _progress(tmp_path, files=2, total=200)
    p.add_file("a", "models/a.bin", 100)
    p.add_file("b", "models/b.bin", 100)
    p.start()
    try:
        p.set_stage("DOWNLOADING", "a")
        p.set_stage("DOWNLOADING", "b")
        p.callback("a", 100, 100)
        p.file_done("a")
        p.callback("b", 30, 100)
        time.sleep(1.25)
        progress = _read(tmp_path / "progress.json")
        assert progress["bytes_done"] == 130
        assert progress["files_done"] == 1
        assert progress["files_active"] == 1
        assert progress["current_files"] == ["b"]
    finally:
        p.stop()


def test_cached_skip_and_resume_offset_are_not_network_speed(tmp_path: Path) -> None:
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    p.start()
    try:
        p.set_stage("VERIFYING", "a")
        p.callback("a", 100, 100)  # cached verified final, not network traffic
        p.file_done("a")
        time.sleep(1.25)
        progress = _read(tmp_path / "progress.json")
        assert progress["bytes_done"] == 100
        assert progress["speed_bytes_per_sec"] in (None, 0.0)
        assert progress["eta_seconds"] is None or progress["eta_known"] is True
    finally:
        p.stop()


def test_fast_stage_and_finish_events_are_fifo_and_terminal_is_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "PROGRESS_INTERVAL_SECONDS", 0.05)
    real = worker._atomic
    entered = threading.Event()
    release = threading.Event()
    state = {"block": False}

    def blocked(path: Path, payload):  # noqa: ANN001
        if path.name == "progress.json" and state["block"]:
            entered.set()
            release.wait(10.0)
        return real(path, payload)

    monkeypatch.setattr(worker, "_atomic", blocked)
    p = _progress(tmp_path, files=2, total=200)
    p.add_file("a", "models/a.bin", 100)
    p.add_file("b", "models/b.bin", 100)
    p.start()
    try:
        state["block"] = True
        p.set_stage("DOWNLOADING", "a")
        p.set_stage("VERIFYING", "a")
        p.file_done("a")
        p.set_stage("DOWNLOADING", "b")
        assert entered.wait(3.0)
        release.set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(p._history) < 4:
            time.sleep(0.02)
        stages = [(item["kind"], item["stage"]) for item in p._history]
        assert ("stage", "DOWNLOADING") in stages
        assert ("stage", "VERIFYING") in stages
        assert any(kind == "file_done" for kind, _stage in stages)
        p.signal_failure("WorkerError")
        p.set_stage("DOWNLOADING", "a")
        p.file_done("b")
        assert p.fail("WorkerError") is True
        assert p._history[-1]["kind"] == "terminal"
        assert p._history[-1]["status"] == "FAILED"
        assert not any(
            item["kind"] == "stage" and item["status"] == "RUNNING"
            for item in p._history[p._history.index(p._history[-1]) :]
        )
    finally:
        release.set()
        p.stop()


def test_heartbeat_continues_and_publishes_failed_while_progress_io_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "PROGRESS_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(worker, "HEARTBEAT_INTERVAL_SECONDS", 0.05)
    real = worker._atomic
    entered = threading.Event()
    release = threading.Event()
    state = {"block": False}

    def blocked(path: Path, payload):  # noqa: ANN001
        if path.name == "progress.json" and state["block"]:
            entered.set()
            release.wait(10.0)
        return real(path, payload)

    monkeypatch.setattr(worker, "_atomic", blocked)
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    p.start()
    try:
        state["block"] = True
        p.set_stage("DOWNLOADING", "a")
        assert entered.wait(3.0)
        first_hb = _read(tmp_path / "heartbeat.json")["heartbeat_at"]
        started = time.monotonic()
        p.callback("a", 50, 100)
        p.network_bytes("a", 50)
        assert time.monotonic() - started < 1.0
        time.sleep(0.25)
        second_hb = _read(tmp_path / "heartbeat.json")["heartbeat_at"]
        assert second_hb > first_hb
        failure: dict[str, bool] = {}

        def fail() -> None:
            failure["ok"] = p.fail("WorkerError")

        thread = threading.Thread(target=fail)
        thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if _read(tmp_path / "heartbeat.json")["status"] == "FAILED":
                break
            time.sleep(0.02)
        assert _read(tmp_path / "heartbeat.json")["status"] == "FAILED"
        assert thread.is_alive(), "fail must still be waiting for progress IO, not deadlocking heartbeat"
        release.set()
        thread.join(5.0)
        assert thread.is_alive() is False
        assert failure["ok"] is True
        assert _read(tmp_path / "status.json")["status"] == "FAILED"
    finally:
        release.set()
        p.stop()


def test_heartbeat_is_written_by_independent_thread_during_slow_progress_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = worker._atomic

    def slow(path: Path, payload):  # noqa: ANN001
        if path.name == "progress.json":
            time.sleep(1.2)
        return real(path, payload)

    monkeypatch.setattr(worker, "_atomic", slow)
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    p.start()
    try:
        first = _read(tmp_path / "heartbeat.json")["heartbeat_at"]
        time.sleep(2.3)
        second = _read(tmp_path / "heartbeat.json")["heartbeat_at"]
        assert second > first
    finally:
        p.stop()


def test_terminal_failed_is_not_overwritten_by_heartbeat(tmp_path: Path) -> None:
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    p.start()
    try:
        assert p.fail("WorkerError") is True
        time.sleep(2.2)
        assert _read(tmp_path / "status.json")["status"] == "FAILED"
        assert _read(tmp_path / "progress.json")["status"] == "FAILED"
        assert _read(tmp_path / "heartbeat.json")["status"] == "FAILED"
    finally:
        p.stop()


def test_write_failure_propagates_and_never_claims_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path, payload):  # noqa: ANN001
        if path.name == "progress.json":
            raise OSError("injected state IO failure")
        return real(path, payload)

    real = worker._atomic
    monkeypatch.setattr(worker, "_atomic", broken)
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    with pytest.raises(worker.WorkerError):
        p.start()
    assert p.persistence_failed is True


def test_io_error_does_not_leave_fail_waiting_on_empty_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = worker._atomic
    state = {"fail": False}

    def broken(path: Path, payload):  # noqa: ANN001
        if path.name == "progress.json" and state["fail"]:
            raise OSError("injected state IO failure")
        return real(path, payload)

    monkeypatch.setattr(worker, "_atomic", broken)
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    p.start()
    try:
        state["fail"] = True
        p.callback("a", 10, 100)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not p.persistence_failed:
            time.sleep(0.02)
        assert p.persistence_failed is True
        started = time.monotonic()
        assert p.fail("WorkerError") is False
        assert time.monotonic() - started < 3.0
    finally:
        p.stop()


def test_duplicate_canonical_targets_are_rejected_before_io(tmp_path: Path) -> None:
    root = tmp_path / "ComfyUI"
    root.mkdir()
    spec_a = setup_plan.DownloadSpec("a", "https://example.invalid/a", "models/x.bin", 1, "a" * 64)
    spec_b = setup_plan.DownloadSpec("b", "https://example.invalid/b", "models/./x.bin", 1, "b" * 64)
    with pytest.raises(worker.WorkerError):
        worker._validate_specs([spec_a, spec_b], root)


def test_progress_keeps_old_v1_fields(tmp_path: Path) -> None:
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    p.start()
    try:
        p.callback("a", 10, 100)
        time.sleep(1.1)
        payload = _read(tmp_path / "progress.json")
        for key in (
            "schema",
            "run_id",
            "nonce",
            "status",
            "stage",
            "current_file",
            "files_done",
            "files_total",
            "bytes_done",
            "bytes_total",
            "heartbeat_at",
            "pid",
            "last_error",
        ):
            assert key in payload
        for key in ("bytes_remaining", "speed_bytes_per_sec", "eta_seconds", "eta_known"):
            assert key in payload
    finally:
        p.stop()


def test_immutable_event_has_priority_over_ordinary_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "PROGRESS_INTERVAL_SECONDS", 0.5)
    real = worker._atomic
    records: list[tuple[str, str, int]] = []
    lock = threading.Lock()

    def record(path: Path, payload):  # noqa: ANN001
        if path.name in {"progress.json", "status.json"}:
            with lock:
                records.append((path.name, str(payload.get("stage")), int(payload.get("bytes_done") or 0)))
        return real(path, payload)

    monkeypatch.setattr(worker, "_atomic", record)
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    p.start()
    try:
        with lock:
            records.clear()
        p.callback("a", 50, 100)
        p.set_stage("DOWNLOADING", "a")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                if any(item[1] == "DOWNLOADING" for item in records):
                    break
            time.sleep(0.01)
        with lock:
            progress_records = [item for item in records if item[0] == "progress.json"]
        assert progress_records
        assert progress_records[0][1] == "DOWNLOADING"
        assert progress_records[0][2] == 50
        # No ordinary write may slip before the queued event; the next due
        # ordinary snapshot may appear only after the interval.
        time.sleep(0.15)
        with lock:
            before_interval = list(progress_records)
        assert all(item[1] == "DOWNLOADING" for item in before_interval)
        time.sleep(0.55)
        with lock:
            after_interval = [item for item in records if item[0] == "progress.json"]
        assert len(after_interval) >= len(before_interval)
    finally:
        p.stop()


def test_terminal_done_writer_blocks_without_busy_spin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "PROGRESS_INTERVAL_SECONDS", 0.05)
    p = _progress(tmp_path, files=1, total=100)
    p.add_file("a", "models/a.bin", 100)
    p.start()
    try:
        assert p.fail("WorkerError") is True
        time.sleep(0.1)
        first = p._writer_wait_count
        time.sleep(0.3)
        second = p._writer_wait_count
        assert second - first <= 2
        assert p._writer_decision_count < 50
    finally:
        p.stop()
