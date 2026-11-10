#!/usr/bin/env python3
"""Actual POSIX read-only observation harness (no SSH, no network)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _json(path: Path, payload: dict) -> bytes:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write(path, data)
    return data


def _tree(root: Path) -> dict[str, str]:
    result = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + list(filenames):
            full = Path(dirpath) / name
            info = os.lstat(full)
            if os.path.islink(full):
                result[str(full.relative_to(root))] = "symlink"
            elif full.is_dir():
                result[str(full.relative_to(root))] = "dir"
            else:
                result[str(full.relative_to(root))] = _sha(full.read_bytes())
    return dict(sorted(result.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from scripts.i1 import observation, process_guard as guard

    root = Path(tempfile.mkdtemp(prefix="i1-observe-"))
    owned = root / ".avs-i1"
    attempt = owned / "attempts" / "0001"
    attempt.mkdir(parents=True)
    run_id = "a" * 32
    nonce = "b" * 32
    worker = attempt / "v3_download_worker.py"
    payload = attempt / "job.json"
    _write(worker, b"import time\ntime.sleep(60)\n")
    _write(
        payload,
        json.dumps(
            {
                "schema": "anime-video-studio/i1-download-job/v1",
                "run_id": run_id,
                "nonce": nonce,
            }
        ).encode("utf-8"),
    )
    launch = {
        "schema": guard.LAUNCH_SCHEMA,
        "kind": "download_worker",
        "run_id": run_id,
        "nonce": nonce,
        "root": str(attempt),
        "worker": str(worker),
        "worker_sha256": _sha(worker.read_bytes()),
        "payload": str(payload),
        "payload_sha256": _sha(payload.read_bytes()),
        "deadline_epoch": time.time() + 3600,
        "status": str(attempt / "status.json"),
        "progress": str(attempt / "progress.json"),
        "heartbeat": str(attempt / "heartbeat.json"),
        "log": str(attempt / "download.log"),
        "owner": str(attempt / "worker-owner.json"),
        "ack": str(attempt / "worker-ack.json"),
    }
    launch_bytes = _json(attempt / "launch.json", launch)
    expected = guard.expected_owner(launch)
    argv_hash = guard.expected_argv_sha256(launch)
    owner = {**expected, "pid": 999999, "argv_sha256": argv_hash}
    ack = dict(owner)
    _json(attempt / "worker-owner.json", owner)
    _json(attempt / "worker-ack.json", ack)
    _json(
        attempt / "status.json",
        {
            "schema": guard.STATUS_SCHEMA,
            "run_id": run_id,
            "nonce": nonce,
            "status": "RUNNING",
        },
    )
    _json(
        attempt / "progress.json",
        {
            "schema": guard.PROGRESS_SCHEMA,
            "run_id": run_id,
            "nonce": nonce,
            "bytes_done": 10,
            "bytes_total": 100,
            "files_done": 0,
            "files_total": 1,
        },
    )
    _json(
        attempt / "heartbeat.json",
        {
            "schema": guard.HEARTBEAT_SCHEMA,
            "run_id": run_id,
            "nonce": nonce,
            "heartbeat_at": time.time() - 503,
        },
    )
    birth = {
        "schema": guard.BIRTH_SCHEMA,
        "kind": "download_worker",
        "run_id": run_id,
        "nonce": nonce,
        "manifest_sha256": expected["command_sha256"],
        "argv_sha256": argv_hash,
        "pid": 999999,
        "start_ticks": 12345,
        "exe": "/usr/bin/python3",
        "cmdline_sha256": "c" * 64,
        "boot_id": "boot",
    }
    proof_dir = attempt / "process-proofs"
    proof_dir.mkdir()
    _json(proof_dir / "birth-999999-12345.json", birth)
    proof = {
        "schema": guard.PROOF_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "command_sha256": expected["command_sha256"],
        "pid": 999999,
        "start_ticks": 12345,
        "exe": "/usr/bin/python3",
        "cmdline_sha256": "c" * 64,
        "argv_sha256": argv_hash,
        "boot_id": "boot",
        "birth_sha256": _sha(guard.canonical_bytes(birth)),
    }
    expected_job = _sha(payload.read_bytes())
    expected_launch = _sha(launch_bytes)
    before = _tree(attempt)
    no_proof = guard.observe_download(
        root=owned,
        attempt_dir=attempt,
        attempt_id="0001",
        run_id=run_id,
        nonce=nonce,
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
    )
    after_no_proof = _tree(attempt)

    def run_collector():
        argv = observation.build_snapshot_argv(
            workspace=str(owned),
            comfy=str(owned / "ComfyUI"),
            commit="x",
            root=str(owned),
            attempt=str(attempt),
            run_id=run_id,
            nonce=nonce,
            guard_sha=observation.guard_sha256(),
            targets=[],
            validator_source="",
            mode="attempt",
            marker="d" * 32,
            guard_source_b64=observation.guard_source_b64(),
            expected_job_sha256=expected_job,
            expected_launch_sha256=expected_launch,
            expected_deadline_epoch=launch["deadline_epoch"],
            collector_budget=6.0,
        )
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        payload = json.loads(completed.stdout)
        return completed.returncode, payload

    collector_before_tree = _tree(attempt)
    collector_before_rc, collector_before = run_collector()
    collector_after_tree = _tree(attempt)
    observation._validate_snapshot(
        collector_before,
        marker="d" * 32,
        run_id=run_id,
        nonce=nonce,
        mode="attempt",
        attempt_path=str(attempt),
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
    )
    _json(proof_dir / "current-999999.json", proof)
    before_proof = _tree(attempt)
    with_proof = guard.observe_download(
        root=owned,
        attempt_dir=attempt,
        attempt_id="0001",
        run_id=run_id,
        nonce=nonce,
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
    )
    after_proof = _tree(attempt)
    collector_after_proof_tree = _tree(attempt)
    collector_after_proof_rc, collector_after_proof = run_collector()
    collector_after_proof_after_tree = _tree(attempt)
    observation._validate_snapshot(
        collector_after_proof,
        marker="d" * 32,
        run_id=run_id,
        nonce=nonce,
        mode="attempt",
        attempt_path=str(attempt),
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
    )
    real_signature = guard._control_signature
    removed = {"done": False}

    def signature_then_remove_ack(root, attempt_dir, deadline=None):
        result = real_signature(root, attempt_dir, deadline)
        if not removed["done"]:
            (Path(attempt_dir) / "worker-ack.json").unlink()
            removed["done"] = True
        return result

    guard._control_signature = signature_then_remove_ack
    try:
        child_disappeared = guard.observe_download(
            root=owned,
            attempt_dir=attempt,
            attempt_id="0001",
            run_id=run_id,
            nonce=nonce,
            expected_job_sha256=expected_job,
            expected_launch_sha256=expected_launch,
        )
    finally:
        guard._control_signature = real_signature
    _json(attempt / "worker-ack.json", ack)
    removed = {"done": False}

    def signature_then_remove_proof(root, attempt_dir, deadline=None):
        result = real_signature(root, attempt_dir, deadline)
        if not removed["done"]:
            (Path(attempt_dir) / "process-proofs" / "current-999999.json").unlink()
            removed["done"] = True
        return result

    guard._control_signature = signature_then_remove_proof
    try:
        proof_disappeared = guard.observe_download(
            root=owned,
            attempt_dir=attempt,
            attempt_id="0001",
            run_id=run_id,
            nonce=nonce,
            expected_job_sha256=expected_job,
            expected_launch_sha256=expected_launch,
        )
    finally:
        guard._control_signature = real_signature
    _json(proof_dir / "current-999999.json", proof)
    (attempt / "worker-owner.json").unlink()
    (attempt / "worker-ack.json").unlink()
    _json(
        attempt / "status.json",
        {
            "schema": guard.STATUS_SCHEMA,
            "run_id": run_id,
            "nonce": nonce,
            "status": "COMPLETE",
        },
    )
    complete_absent = guard.observe_download(
        root=owned,
        attempt_dir=attempt,
        attempt_id="0001",
        run_id=run_id,
        nonce=nonce,
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
    )
    collector_complete_tree = _tree(attempt)
    collector_complete_rc, collector_complete = run_collector()
    observation._validate_snapshot(
        collector_complete,
        marker="d" * 32,
        run_id=run_id,
        nonce=nonce,
        mode="attempt",
        attempt_path=str(attempt),
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
    )
    collector_complete_after_tree = _tree(attempt)

    def combo(owner_present, ack_present, status_payload):
        for name in ("worker-owner.json", "worker-ack.json"):
            target = attempt / name
            if target.exists():
                target.unlink()
        if owner_present:
            _json(attempt / "worker-owner.json", owner)
        if ack_present:
            _json(attempt / "worker-ack.json", ack)
        status_target = attempt / "status.json"
        if status_payload is None:
            if status_target.exists():
                status_target.unlink()
        else:
            _json(status_target, status_payload)
        return guard.observe_download(
            root=owned,
            attempt_dir=attempt,
            attempt_id="0001",
            run_id=run_id,
            nonce=nonce,
            expected_job_sha256=expected_job,
            expected_launch_sha256=expected_launch,
        )

    running_status = {
        "schema": guard.STATUS_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "status": "RUNNING",
    }
    complete_status = {
        "schema": guard.STATUS_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "status": "COMPLETE",
    }
    combos = {
        "owner_only": combo(True, False, running_status),
        "ack_only": combo(False, True, running_status),
        "both_absent_running": combo(False, False, running_status),
        "both_absent_complete": combo(False, False, complete_status),
        "both_absent_missing": combo(False, False, None),
        "both_present": combo(True, True, running_status),
    }
    _json(attempt / "worker-owner.json", owner)
    _json(attempt / "worker-ack.json", ack)
    _json(attempt / "status.json", running_status)
    missing = guard.observe_download(
        root=owned,
        attempt_dir=owned / "attempts" / "9999",
        attempt_id="9999",
        run_id=run_id,
        nonce=nonce,
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
    )
    symlink = owned / "attempts" / "0002"
    symlink.symlink_to(attempt, target_is_directory=True)
    symlink_result = guard.observe_download(
        root=owned,
        attempt_dir=symlink,
        attempt_id="0002",
        run_id=run_id,
        nonce=nonce,
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
    )
    real_lstat = guard.os.lstat

    def denied(path):
        if str(path).endswith("status.json"):
            raise PermissionError("denied")
        return real_lstat(path)

    guard.os.lstat = denied
    try:
        permission = guard.observe_download(
            root=owned,
            attempt_dir=attempt,
            attempt_id="0001",
            run_id=run_id,
            nonce=nonce,
            expected_job_sha256=expected_job,
            expected_launch_sha256=expected_launch,
        )
    finally:
        guard.os.lstat = real_lstat
    _json(attempt / "worker-owner.json", owner)
    _json(attempt / "worker-ack.json", ack)
    _json(attempt / "status.json", running_status)
    status_path = attempt / "status.json"
    status_path.unlink()
    os.mkfifo(status_path)
    fifo_started = time.monotonic()
    fifo_status = guard.observe_download(
        root=owned,
        attempt_dir=attempt,
        attempt_id="0001",
        run_id=run_id,
        nonce=nonce,
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
        deadline=time.monotonic() + 0.5,
    )
    fifo_elapsed = time.monotonic() - fifo_started
    status_path.unlink()
    _json(status_path, running_status)
    proof_path = proof_dir / "current-999999.json"
    proof_path.unlink()
    os.mkfifo(proof_path)
    proof_fifo_started = time.monotonic()
    fifo_proof = guard.observe_download(
        root=owned,
        attempt_dir=attempt,
        attempt_id="0001",
        run_id=run_id,
        nonce=nonce,
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
        deadline=time.monotonic() + 0.5,
    )
    proof_fifo_elapsed = time.monotonic() - proof_fifo_started
    proof_path.unlink()
    _json(proof_path, proof)
    stop = threading.Event()

    def toggler():
        index = 0
        while not stop.is_set():
            status = "RUNNING" if index % 2 == 0 else "COMPLETE"
            _json(
                attempt / "status.json",
                {
                    "schema": guard.STATUS_SCHEMA,
                    "run_id": run_id,
                    "nonce": nonce,
                    "status": status,
                },
            )
            index += 1
            time.sleep(0.005)

    thread = threading.Thread(target=toggler)
    thread.start()
    try:
        time.sleep(0.02)
        consistency = guard.observe_download(
            root=owned,
            attempt_dir=attempt,
            attempt_id="0001",
            run_id=run_id,
            nonce=nonce,
            expected_job_sha256=expected_job,
            expected_launch_sha256=expected_launch,
            deadline=time.monotonic() + 0.15,
        )
    finally:
        stop.set()
        thread.join()
    live_child = subprocess.Popen([sys.executable, str(worker), str(payload)])
    try:
        live_facts = guard._proc_facts(live_child.pid)
        live_owner = {**expected, "pid": live_child.pid, "argv_sha256": argv_hash}
        live_ack = dict(live_owner)
        _json(attempt / "worker-owner.json", live_owner)
        _json(attempt / "worker-ack.json", live_ack)
        _json(attempt / "status.json", running_status)
        live_birth = {
            "schema": guard.BIRTH_SCHEMA,
            "kind": "download_worker",
            "run_id": run_id,
            "nonce": nonce,
            "manifest_sha256": expected["command_sha256"],
            "argv_sha256": live_facts["argv_sha256"],
            "pid": live_child.pid,
            "start_ticks": live_facts["start_ticks"],
            "exe": live_facts["exe"],
            "cmdline_sha256": live_facts["cmdline_sha256"],
            "boot_id": live_facts["boot_id"],
        }
        _json(
            proof_dir / f"birth-{live_child.pid}-{live_facts['start_ticks']}.json",
            live_birth,
        )
        live_proof = {
            "schema": guard.PROOF_SCHEMA,
            "run_id": run_id,
            "nonce": nonce,
            "command_sha256": expected["command_sha256"],
            "pid": live_child.pid,
            "start_ticks": live_facts["start_ticks"],
            "exe": live_facts["exe"],
            "cmdline_sha256": live_facts["cmdline_sha256"],
            "argv_sha256": live_facts["argv_sha256"],
            "boot_id": live_facts["boot_id"],
            "birth_sha256": _sha(guard.canonical_bytes(live_birth)),
        }
        _json(proof_dir / f"current-{live_child.pid}.json", live_proof)
        live_decision = guard.observe_download(
            root=owned,
            attempt_dir=attempt,
            attempt_id="0001",
            run_id=run_id,
            nonce=nonce,
            expected_job_sha256=expected_job,
            expected_launch_sha256=expected_launch,
            expected_deadline_epoch=launch["deadline_epoch"],
        )
        collector_live_rc, collector_live = run_collector()
        observation._validate_snapshot(
            collector_live,
            marker="d" * 32,
            run_id=run_id,
            nonce=nonce,
            mode="attempt",
            attempt_path=str(attempt),
            expected_job_sha256=expected_job,
            expected_launch_sha256=expected_launch,
        )
    finally:
        live_child.terminate()
        try:
            live_child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            live_child.kill()
            live_child.wait(timeout=5)
    for name in ("worker-owner.json", "worker-ack.json"):
        target = attempt / name
        if target.exists():
            target.unlink()
    status_target = attempt / "status.json"
    if status_target.exists():
        status_target.unlink()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,pathlib,sys;pathlib.Path(sys.argv[1]).write_text(json.dumps({'schema':sys.argv[2],'run_id':sys.argv[3],'nonce':sys.argv[4],'status':'COMPLETE'}))",
            str(status_target),
            guard.STATUS_SCHEMA,
            run_id,
            nonce,
        ],
        check=True,
    )
    complete_worker = guard.observe_download(
        root=owned,
        attempt_dir=attempt,
        attempt_id="0001",
        run_id=run_id,
        nonce=nonce,
        expected_job_sha256=expected_job,
        expected_launch_sha256=expected_launch,
        expected_deadline_epoch=launch["deadline_epoch"],
    )
    summary = {
        "status": "OK",
        "no_proof_state": no_proof.get("state"),
        "no_proof_tree_unchanged": before == after_no_proof,
        "no_proof_created_lock": any(
            "lock" in name.lower() for name in after_no_proof
        ),
        "with_proof_state": with_proof.get("state"),
        "with_proof_process": with_proof.get("process"),
        "with_proof_tree_unchanged": before_proof == after_proof,
        "missing_state": missing.get("state"),
        "symlink_state": symlink_result.get("state"),
        "permission_state": permission.get("state"),
        "consistency_state": consistency.get("state"),
        "consistency_reason": consistency.get("reason"),
        "consistency_status": (
            (consistency.get("status") or {}).get("status")
            if consistency.get("state") == "OK"
            else None
        ),
        "collector_before_rc": collector_before_rc,
        "collector_before_marker": collector_before.get("marker"),
        "collector_before_attempt_state": (collector_before.get("attempt") or {}).get("state"),
        "collector_before_tree_unchanged": collector_before_tree == collector_after_tree,
        "collector_after_proof_rc": collector_after_proof_rc,
        "collector_after_proof_attempt_state": (collector_after_proof.get("attempt") or {}).get("state"),
        "collector_after_proof_process": ((collector_after_proof.get("attempt") or {}).get("decision") or {}).get("process"),
        "collector_after_proof_tree_unchanged": collector_after_proof_tree == collector_after_proof_after_tree,
        "complete_absent_state": complete_absent.get("state"),
        "complete_absent_process": complete_absent.get("process"),
        "collector_complete_absent_state": (collector_complete.get("attempt") or {}).get("state"),
        "collector_complete_absent_process": ((collector_complete.get("attempt") or {}).get("decision") or {}).get("process"),
        "collector_complete_tree_unchanged": collector_complete_tree == collector_complete_after_tree,
        "child_disappeared_state": child_disappeared.get("state"),
        "proof_disappeared_state": proof_disappeared.get("state"),
        "owner_only_state": combos["owner_only"].get("state"),
        "ack_only_state": combos["ack_only"].get("state"),
        "both_absent_running_state": combos["both_absent_running"].get("state"),
        "both_absent_complete_state": combos["both_absent_complete"].get("state"),
        "both_absent_complete_process": combos["both_absent_complete"].get("process"),
        "both_absent_missing_state": combos["both_absent_missing"].get("state"),
        "both_present_state": combos["both_present"].get("state"),
        "both_present_process": combos["both_present"].get("process"),
        "fifo_status_state": fifo_status.get("state"),
        "fifo_status_elapsed": round(fifo_elapsed, 3),
        "fifo_proof_state": fifo_proof.get("state"),
        "fifo_proof_elapsed": round(proof_fifo_elapsed, 3),
        "live_state": live_decision.get("state"),
        "live_process": live_decision.get("process"),
        "live_pid": live_child.pid,
        "collector_live_state": (collector_live.get("attempt") or {}).get("state"),
        "collector_live_process": ((collector_live.get("attempt") or {}).get("decision") or {}).get("process"),
        "live_child_gone": live_child.poll() is not None,
        "complete_worker_state": complete_worker.get("state"),
        "complete_worker_process": complete_worker.get("process"),
    }
    ok = (
        summary["no_proof_state"] == "INCONCLUSIVE"
        and summary["no_proof_tree_unchanged"] is True
        and summary["no_proof_created_lock"] is False
        and summary["with_proof_state"] == "OK"
        and summary["with_proof_process"] == "DEAD"
        and summary["with_proof_tree_unchanged"] is True
        and summary["missing_state"] == "MISSING"
        and summary["symlink_state"] == "UNREADABLE"
        and summary["permission_state"] == "UNREADABLE"
        and summary["consistency_state"] in {"INCONCLUSIVE", "OK"}
        and (
            summary["consistency_state"] != "OK"
            or summary["consistency_status"] in {"RUNNING", "COMPLETE"}
        )
        and summary["collector_before_rc"] == 0
        and summary["collector_before_marker"] == "d" * 32
        and summary["collector_before_attempt_state"] == "INCONCLUSIVE"
        and summary["collector_before_tree_unchanged"] is True
        and summary["collector_after_proof_rc"] == 0
        and summary["collector_after_proof_attempt_state"] == "OK"
        and summary["collector_after_proof_process"] == "DEAD"
        and summary["collector_after_proof_tree_unchanged"] is True
        and summary["complete_absent_state"] == "OK"
        and summary["complete_absent_process"] == "ABSENT"
        and summary["collector_complete_absent_state"] == "OK"
        and summary["collector_complete_absent_process"] == "ABSENT"
        and summary["collector_complete_tree_unchanged"] is True
        and summary["child_disappeared_state"] in {"UNREADABLE", "INCONCLUSIVE"}
        and summary["proof_disappeared_state"] in {"UNREADABLE", "INCONCLUSIVE"}
        and summary["owner_only_state"] == "UNREADABLE"
        and summary["ack_only_state"] == "UNREADABLE"
        and summary["both_absent_running_state"] == "UNREADABLE"
        and summary["both_absent_complete_state"] == "OK"
        and summary["both_absent_complete_process"] == "ABSENT"
        and summary["both_absent_missing_state"] == "MISSING"
        and summary["both_present_state"] == "OK"
        and summary["both_present_process"] == "DEAD"
        and summary["fifo_status_state"] in {"UNREADABLE", "INCONCLUSIVE"}
        and summary["fifo_status_elapsed"] < 2.0
        and summary["fifo_proof_state"] in {"UNREADABLE", "INCONCLUSIVE"}
        and summary["fifo_proof_elapsed"] < 2.0
        and summary["live_state"] == "OK"
        and summary["live_process"] == "LIVE"
        and summary["collector_live_state"] == "OK"
        and summary["collector_live_process"] == "LIVE"
        and summary["live_child_gone"] is True
        and summary["complete_worker_state"] == "OK"
        and summary["complete_worker_process"] == "ABSENT"
    )
    summary["ok"] = ok
    print(json.dumps(summary, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
