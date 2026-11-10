from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.i1 import download
from scripts.i1.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]


class FakeRemote:
    def __init__(self, root: Path) -> None:
        self.base = root
        self.puts = []
        self.commands = []

    def _local(self, remote: str) -> Path:
        if remote.startswith("/workspace/.avs-i1"):
            return self.base / remote[len("/workspace/.avs-i1/") :]
        raise AssertionError(remote)

    def put_file(self, identity, remote_path, local_path, **kwargs):
        self.puts.append(remote_path)
        target = self._local(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local_path).read_bytes())

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout):
        return json.loads(self._local(remote_path).read_text(encoding="utf-8"))

    def run_remote(self, identity, argv, timeout=None, **kwargs):
        self.commands.append(tuple(argv))
        if argv[0] == "mkdir":
            return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)
        if argv[0] == "python3" and "-c" in argv and "start_ticks" in argv[2]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "pid": int(argv[3]),
                        "start_ticks": 777,
                        "exe": "/usr/bin/python3",
                        "cmdline_sha256": "b" * 64,
                        "boot_id": "boot-1",
                    }
                ),
                stderr="",
                timed_out=False,
            )
        if argv[0] == "python3" and "i1_process_guard.py" in argv[1] and argv[2] == "inspect":
            attempt = self.base / "attempts" / "0001"
            owner = json.loads((attempt / "worker-owner.json").read_text(encoding="utf-8"))
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "anime-video-studio/i1-attempt-inspect/v1",
                        "run_id": argv[4],
                        "nonce": argv[5],
                        "state": "OK",
                        "process": "DEAD",
                        "owner": owner,
                        "ack": owner,
                        "status": json.loads((attempt / "status.json").read_text(encoding="utf-8")),
                        "progress": {},
                        "heartbeat": {},
                    }
                ),
                stderr="",
                timed_out=False,
            )
        if argv[0] == "python3" and "i1_process_guard.py" in argv[1] and argv[2] == "launch":
            launch = json.loads(self._local(argv[4]).read_text(encoding="utf-8"))
            attempt = self.base / "attempts" / "0001"
            attempt.mkdir(parents=True, exist_ok=True)
            proof = {
                "schema": "anime-video-studio/i1-process-proof/v1",
                "run_id": launch["run_id"],
                "nonce": launch["nonce"],
                "command_sha256": "a" * 64,
                "pid": 12345,
                "start_ticks": 777,
                "exe": "/usr/bin/python3",
                "cmdline_sha256": "b" * 64,
                "argv_sha256": "c" * 64,
                "boot_id": "boot-1",
                "started_at": 1.0,
            }
            owner = {
                "schema": "anime-video-studio/p1-v3-process-owner/v1",
                "kind": "download_worker",
                "run_id": launch["run_id"],
                "nonce": launch["nonce"],
                "command_sha256": "a" * 64,
                "pid": 12345,
            }
            for name, value in (
                ("worker-owner.json", owner),
                ("worker-ack.json", owner),
            ):
                (attempt / name).write_text(json.dumps(value), encoding="utf-8")
            for name, value in (
                ("status.json", {"schema": "anime-video-studio/i1-download-status/v1", "run_id": launch["run_id"], "nonce": launch["nonce"], "status": "COMPLETE"}),
                ("progress.json", {"schema": "anime-video-studio/i1-download-progress/v1", "run_id": launch["run_id"], "nonce": launch["nonce"], "status": "COMPLETE", "files_done": 4, "files_total": 4, "bytes_done": 10, "bytes_total": 10}),
                ("heartbeat.json", {"schema": "anime-video-studio/i1-download-heartbeat/v1", "run_id": launch["run_id"], "nonce": launch["nonce"], "heartbeat_at": 1.0}),
            ):
                (attempt / name).write_text(json.dumps(value), encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"schema": "anime-video-studio/p1-v3-launch-result/v1", "kind": "download_worker", "run_id": launch["run_id"], "nonce": launch["nonce"], "status": "STARTED", "command_sha256": "a" * 64, "pid": 12345, "process_proof": proof}),
                stderr="",
                timed_out=False,
            )
        raise AssertionError(f"unexpected command: {argv}")


def test_download_stage_has_four_models_and_frozen_launch_contract(tmp_path: Path) -> None:
    runner = FakeRemote(tmp_path / "remote")
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    staged = download.stage(
        runner,
        object(),
        root="/workspace/.avs-i1",
        workspace="/workspace",
        profile=profile,
        run_id="r",
        nonce="n",
        deadline_epoch=9_999_999_999,
        target_home="/workspace",
        target_id="t" * 64,
        target_lock_path="/workspace/.cache/anime-video-studio/i1/targets/t.lock",
    )
    assert staged["launch"]["kind"] == "download_worker"
    assert staged["launch"]["worker"].endswith("v3_download_worker.py")
    assert staged["launch"]["payload"].endswith("job.json")
    assert staged["job"]["schema"] == download.JOB_SCHEMA
    assert staged["job"]["profile_sha256"] == profile.sha256
    assert len(staged["job"]["worker_impl_sha256"]) == 64
    for key in ("worker", "worker_impl", "child_entry", "guard"):
        assert runner._local(staged["paths"][key]).is_file(), key
    assert set(staged["launch"]) == {
        "schema",
        "kind",
        "run_id",
        "nonce",
        "root",
        "worker",
        "worker_sha256",
        "payload",
        "payload_sha256",
        "deadline_epoch",
        "status",
        "progress",
        "heartbeat",
        "log",
        "owner",
        "ack",
    }
    assert len(staged["job"]["models"]) == len(profile.models) == 4
    assert {
        (item["id"], item["target"], item["size_bytes"], item["sha256"])
        for item in staged["job"]["models"]
    } == {
        (item["id"], item["target"], int(item["size_bytes"]), item["sha256"])
        for item in profile.models
    }
    assert staged["job"]["required_bytes"] == profile.disk_required_bytes


def test_attempt_paths_and_deadline_are_immutable_across_resume(tmp_path: Path) -> None:
    runner = FakeRemote(tmp_path / "remote")
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    first = download.stage(
        runner,
        object(),
        root="/workspace/.avs-i1",
        workspace="/workspace",
        profile=profile,
        run_id="r",
        nonce="n",
        deadline_epoch=9_999_999_999,
        target_home="/workspace",
        target_id="t" * 64,
        target_lock_path="/workspace/.cache/anime-video-studio/i1/targets/t.lock",
        attempt="0001",
    )
    second = download.stage(
        runner,
        object(),
        root="/workspace/.avs-i1",
        workspace="/workspace",
        profile=profile,
        run_id="r",
        nonce="n",
        deadline_epoch=9_999_999_999,
        target_home="/workspace",
        target_id="t" * 64,
        target_lock_path="/workspace/.cache/anime-video-studio/i1/targets/t.lock",
        attempt="0001",
    )
    assert first["paths"]["attempt"] == "/workspace/.avs-i1/attempts/0001"
    assert second["paths"]["attempt"] == first["paths"]["attempt"]
    assert second["launch_sha256"] == first["launch_sha256"]
    changed = download.stage(
        runner,
        object(),
        root="/workspace/.avs-i1",
        workspace="/workspace",
        profile=profile,
        run_id="r",
        nonce="n",
        deadline_epoch=9_999_999_998,
        target_home="/workspace",
        target_id="t" * 64,
        target_lock_path="/workspace/.cache/anime-video-studio/i1/targets/t.lock",
        attempt="0001",
    )
    assert changed["launch_sha256"] != first["launch_sha256"]


def test_changed_launch_is_refused_before_any_remote_upload(tmp_path: Path) -> None:
    runner = FakeRemote(tmp_path / "remote")
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    first = download.stage(
        runner,
        object(),
        root="/workspace/.avs-i1",
        workspace="/workspace",
        profile=profile,
        run_id="r",
        nonce="n",
        deadline_epoch=9_999_999_999,
        target_home="/workspace",
        target_id="t" * 64,
        target_lock_path="/workspace/.cache/anime-video-studio/i1/targets/t.lock",
        attempt="0001",
    )
    puts = list(runner.puts)
    with pytest.raises(download.RemoteFailure, match="new-attempt"):
        download.stage(
            runner,
            object(),
            root="/workspace/.avs-i1",
            workspace="/workspace",
            profile=profile,
            run_id="r",
            nonce="n",
            deadline_epoch=9_999_999_998,
            target_home="/workspace",
            target_id="t" * 64,
            target_lock_path="/workspace/.cache/anime-video-studio/i1/targets/t.lock",
            attempt="0001",
            expected_launch_sha256=first["launch_sha256"],
        )
    assert runner.puts == puts


def test_download_start_and_read_roundtrip(tmp_path: Path) -> None:
    runner = FakeRemote(tmp_path / "remote")
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    download.stage(
        runner,
        object(),
        root="/workspace/.avs-i1",
        workspace="/workspace",
        profile=profile,
        run_id="r",
        nonce="n",
        deadline_epoch=9_999_999_999,
        target_home="/workspace",
        target_id="t" * 64,
        target_lock_path="/workspace/.cache/anime-video-studio/i1/targets/t.lock",
    )
    puts_before_start = list(runner.puts)
    started = download.start(runner, object(), root="/workspace/.avs-i1", run_id="r", nonce="n")
    assert started["status"] == "STARTED"
    assert runner.puts == puts_before_start, "local client must not upload proof files"
    state = download.read(runner, object(), root="/workspace/.avs-i1", run_id="r", nonce="n")
    assert state["status"]["status"] == "COMPLETE"
    assert state["progress"]["files_done"] == 4


def test_stale_running_is_recovered_before_dead_resume(monkeypatch) -> None:
    calls = []
    stale = {
        "status": {"status": "RUNNING"},
        "progress": {},
        "heartbeat": {"heartbeat_at": 0},
        "owner": {},
        "ack": {},
    }
    monkeypatch.setattr(download, "read_optional", lambda *a, **k: stale)
    monkeypatch.setattr(download, "heartbeat_age", lambda state: 999.0)
    monkeypatch.setattr(download, "recover", lambda *a, **k: calls.append("recover") or {"status": "RECOVERED_INTERRUPTED"})
    monkeypatch.setattr(download, "start", lambda *a, **k: calls.append(("start", k.get("resume_dead"))) or {"status": "ATTACHED"})
    result = download.resume_or_attach(object(), object(), root="/workspace/.avs-i1", run_id="r", nonce="n")
    assert result["status"] == "ATTACHED"
    assert calls == ["recover", ("start", True)]


def test_new_attempt_policy_distinguishes_missing_dead_unreadable_and_live() -> None:
    assert download.new_attempt_allowed({"state": "MISSING"}) is True
    assert (
        download.new_attempt_allowed(
            {"state": "OK", "process": "DEAD", "status": {"status": "RUNNING"}}
        )
        is True
    )
    assert (
        download.new_attempt_allowed(
            {"state": "OK", "process": "ABSENT", "status": {"status": "COMPLETE"}}
        )
        is True
    )
    assert (
        download.new_attempt_allowed(
            {"state": "OK", "process": "UNREADABLE", "status": {"status": "FAILED"}}
        )
        is False
    )
    assert (
        download.new_attempt_allowed(
            {"state": "OK", "process": "LIVE", "status": {"status": "FAILED"}}
        )
        is False
    )
    assert (
        download.new_attempt_allowed(
            {"state": "OK", "process": "LIVE", "status": {"status": "COMPLETE"}}
        )
        is False
    )
    assert download.new_attempt_allowed({"state": "UNREADABLE"}) is False


def test_inspect_attempt_raises_on_remote_read_error(tmp_path: Path) -> None:
    class BrokenRemote:
        def run_remote(self, identity, argv, timeout=None, **kwargs):
            return SimpleNamespace(returncode=3, stdout="", stderr="ssh broken", timed_out=False)

    with pytest.raises(download.RemoteFailure):
        download.inspect_attempt(
            BrokenRemote(), object(), root="/workspace/.avs-i1", run_id="r", nonce="n"
        )


def test_actual_receiver_refuses_overwrite_and_accepts_same_hash(tmp_path: Path) -> None:
    from scripts.p1.remote import _REMOTE_RECEIVER

    root = tmp_path / "root"
    root.mkdir()
    dest = root / "file.bin"
    dest.write_bytes(b"old")
    data = b"new"
    code = "exec(" + repr(_REMOTE_RECEIVER) + ")"
    expected = hashlib.sha256(data).hexdigest()
    result = subprocess.run(
        [sys.executable, "-c", code, str(dest), str(root), expected, "noclobber"],
        input=data,
        capture_output=True,
    )
    assert result.returncode == 9
    assert dest.read_bytes() == b"old"
    result2 = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(dest),
            str(root),
            hashlib.sha256(b"old").hexdigest(),
            "same",
        ],
        input=b"old",
        capture_output=True,
    )
    assert result2.returncode == 0
    assert dest.read_bytes() == b"old"

