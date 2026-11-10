from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.i1 import comfy, comfy_supervisor, process_guard
from scripts.i1.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROOF = {
    "schema": "anime-video-studio/i1-process-proof/v1",
    "run_id": "r",
    "nonce": "n",
    "command_sha256": "a" * 64,
    "pid": 777,
    "start_ticks": 999,
    "exe": "/usr/bin/python3",
    "cmdline_sha256": "b" * 64,
    "argv_sha256": "c" * 64,
    "boot_id": "boot",
    "started_at": 1.0,
}


class FakeRemote:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.puts = []

    def _local(self, remote: str) -> Path:
        return self.workspace / remote[len("/workspace/") :]

    def put_file(self, identity, remote_path, local_path, **kwargs):
        self.puts.append(remote_path)
        target = self._local(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local_path).read_bytes())

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout):
        return json.loads(self._local(remote_path).read_text(encoding="utf-8"))

    def run_remote(self, identity, argv, timeout=None, **kwargs):
        args = list(argv)
        if args[0] == "mkdir":
            return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)
        if "i1_comfy_supervisor.py" in " ".join(args):
            for name, value in (
                (
                    "comfy-owner.json",
                    {
                        "schema": "anime-video-studio/p1-v3-process-owner/v1",
                        "kind": "comfy",
                        "run_id": "r",
                        "nonce": "n",
                        "command_sha256": "a" * 64,
                        "pid": 777,
                    },
                ),
                (
                    "comfy-ack.json",
                    {
                        "schema": "anime-video-studio/p1-v3-process-owner/v1",
                        "kind": "comfy",
                        "run_id": "r",
                        "nonce": "n",
                        "command_sha256": "a" * 64,
                        "pid": 777,
                    },
                ),
            ):
                path = self.workspace / name
                path.write_text(json.dumps(value), encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "anime-video-studio/p1-v3-launch-result/v1",
                        "kind": "comfy",
                        "run_id": "r",
                        "nonce": "n",
                        "status": "STARTED",
                        "command_sha256": "a" * 64,
                        "pid": 777,
                        "process_proof": PROOF,
                    }
                ),
                stderr="",
                timed_out=False,
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)


def test_cold_root_comfy_stage_includes_launcher_and_dependencies(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".avs-i1").mkdir(parents=True)
    profile = load_profile(ROOT / "config" / "i1" / "wan22-14b-ab.profile.json")
    runner = FakeRemote(workspace)
    manifest = comfy.stage_launch(
        runner,
        object(),
        root="/workspace",
        workspace="/workspace",
        profile=profile,
        run_id="r",
        nonce="n",
    )
    assert manifest["kind"] == "comfy"
    for name in (
        "v3_remote_launcher.py",
        "i1_comfy_supervisor.py",
        "i1_process_guard.py",
        "i1_util.py",
        "scripts/p1/v3_remote_fetch_helper.py",
        ".avs-i1/comfy-entry/main.py",
        ".avs-i1/comfy-entry/i1_child_entry.py",
        "comfy-launch.json",
    ):
        assert (workspace / name).is_file(), name
    puts_before_start = list(runner.puts)
    started = comfy.start(
        runner, object(), root="/workspace", run_id="r", nonce="n"
    )
    assert runner.puts == puts_before_start, "local client must not upload proof files"
    assert started["launch"]["status"] == "STARTED"
    assert started["owner"]["pid"] == 777
    assert started["proof"]["schema"] == process_guard.PROOF_SCHEMA


def _manifest() -> dict:
    return {
        "schema": "anime-video-studio/p1-v3-launch/v1",
        "kind": "comfy",
        "run_id": "r",
        "nonce": "n",
        "root": "/workspace",
        "log": "/workspace/comfy.log",
        "owner": "/workspace/comfy-owner.json",
        "ack": "/workspace/comfy-ack.json",
        "python": "/workspace/p1-venv/bin/python",
        "comfy_dir": "/workspace/ComfyUI",
        "main": "/workspace/ComfyUI/main.py",
        "port": 8188,
    }


def _expected(manifest: dict) -> tuple[dict, str]:
    return (
        process_guard.expected_owner(manifest),
        process_guard.expected_argv_sha256(manifest),
    )


def _records(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    manifest = _manifest()
    expected, argv_hash = _expected(manifest)
    owner = {
        **expected,
        "pid": 999999,
        "started_at": 1.0,
        "argv_sha256": argv_hash,
    }
    ack = dict(owner)
    owner_path = tmp_path / "comfy-owner.json"
    ack_path = tmp_path / "comfy-ack.json"
    owner_path.write_text(json.dumps(owner), encoding="utf-8")
    ack_path.write_text(json.dumps(ack), encoding="utf-8")
    return owner_path, ack_path, owner, ack


def _facts(owner: dict, *, start_ticks: int = 42, argv_sha256: str | None = None) -> dict:
    return {
        "pid": owner["pid"],
        "start_ticks": start_ticks,
        "state": "S",
        "exe": "/usr/bin/python3",
        "cmdline_sha256": "d" * 64,
        "argv_sha256": argv_sha256 or owner["argv_sha256"],
        "boot_id": "boot",
    }


def _write_birth(proof_dir: Path, owner: dict, facts: dict) -> Path:
    proof_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": process_guard.BIRTH_SCHEMA,
        "kind": owner["kind"],
        "run_id": owner["run_id"],
        "nonce": owner["nonce"],
        "manifest_sha256": owner["command_sha256"],
        "created_at": 1.0,
        **facts,
    }
    path = proof_dir / f"birth-{facts['pid']}-{facts['start_ticks']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_dead_same_incarnation_is_archived_after_full_validation(
    tmp_path: Path, monkeypatch
) -> None:
    owner_path, ack_path, owner, _ = _records(tmp_path)
    proof_dir = tmp_path / "comfy-process-proofs"
    _write_birth(proof_dir, owner, _facts(owner))
    monkeypatch.setattr(
        process_guard,
        "_proc_facts",
        lambda pid: (_ for _ in ()).throw(OSError("gone")),
    )
    monkeypatch.setattr(process_guard, "_confirmed_absent", lambda pid: True)
    decision = process_guard.assess_owner(
        root=tmp_path,
        manifest=_manifest(),
        owner_path=owner_path,
        ack_path=ack_path,
        proof_dir=proof_dir,
    )
    assert decision["action"] == "ARCHIVE_DEAD"
    assert decision["birth"]["schema"] == process_guard.BIRTH_SCHEMA


def test_foreign_or_altered_owner_is_retained_byte_for_byte(
    tmp_path: Path, monkeypatch
) -> None:
    cases = {
        "wrong_nonce": lambda owner, ack: owner.update(nonce="foreign"),
        "wrong_kind": lambda owner, ack: owner.update(kind="download_worker"),
        "missing_pid": lambda owner, ack: owner.pop("pid"),
        "wrong_command": lambda owner, ack: owner.update(command_sha256="f" * 64),
        "ack_altered": lambda owner, ack: ack.update(pid=123),
    }
    for label, mutate in cases.items():
        case = tmp_path / label
        case.mkdir()
        owner_path, ack_path, owner, ack = _records(case)
        mutate(owner, ack)
        owner_path.write_text(json.dumps(owner), encoding="utf-8")
        ack_path.write_text(json.dumps(ack), encoding="utf-8")
        before_owner = owner_path.read_bytes()
        before_ack = ack_path.read_bytes()
        decision = process_guard.assess_owner(
            root=case,
            manifest=_manifest(),
            owner_path=owner_path,
            ack_path=ack_path,
            proof_dir=case / "comfy-process-proofs",
        )
        assert decision["action"] == "REFUSE", label
        assert owner_path.read_bytes() == before_owner, label
        assert ack_path.read_bytes() == before_ack, label
        assert not (case / "comfy-archive").exists(), label


def test_live_owner_proof_attach_and_missing_pointer_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    owner_path, ack_path, owner, _ = _records(tmp_path)
    proof_dir = tmp_path / "comfy-process-proofs"
    facts = _facts(owner)
    _write_birth(proof_dir, owner, facts)
    monkeypatch.setattr(process_guard, "_proc_facts", lambda pid: dict(facts))
    decision = process_guard.assess_owner(
        root=tmp_path,
        manifest=_manifest(),
        owner_path=owner_path,
        ack_path=ack_path,
        proof_dir=proof_dir,
    )
    assert decision["action"] == "ATTACH"
    assert decision["process"] == "LIVE"
    assert decision["recovered"] is True
    pointer = proof_dir / "current-999999.json"
    assert pointer.is_file()
    # Second assessment must attach from the durable proof, not a blind PID.
    again = process_guard.assess_owner(
        root=tmp_path,
        manifest=_manifest(),
        owner_path=owner_path,
        ack_path=ack_path,
        proof_dir=proof_dir,
    )
    assert again["action"] == "ATTACH"
    assert again["recovered"] is False


def test_live_pid_with_changed_cmdline_is_refused(tmp_path: Path, monkeypatch) -> None:
    owner_path, ack_path, owner, _ = _records(tmp_path)
    proof_dir = tmp_path / "comfy-process-proofs"
    _write_birth(proof_dir, owner, _facts(owner))
    facts = _facts(owner, argv_sha256="e" * 64)
    monkeypatch.setattr(process_guard, "_proc_facts", lambda pid: dict(facts))
    before = owner_path.read_bytes()
    decision = process_guard.assess_owner(
        root=tmp_path,
        manifest=_manifest(),
        owner_path=owner_path,
        ack_path=ack_path,
        proof_dir=proof_dir,
    )
    assert decision["action"] == "REFUSE"
    assert owner_path.read_bytes() == before
    assert not (tmp_path / "comfy-archive").exists()


def test_started_incarnation_overwrites_a_stale_same_pid_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, owner, _ = _records(tmp_path)
    manifest = _manifest()
    expected, argv_hash = _expected(manifest)
    proof_dir = tmp_path / "comfy-process-proofs"
    facts = _facts(owner, start_ticks=2)
    _write_birth(proof_dir, owner, facts)
    old = {
        "schema": process_guard.PROOF_SCHEMA,
        "run_id": "r",
        "nonce": "n",
        "command_sha256": owner["command_sha256"],
        "pid": 999999,
        "start_ticks": 1,
        "exe": "/old",
        "cmdline_sha256": "1" * 64,
        "argv_sha256": owner["argv_sha256"],
        "boot_id": "boot",
        "birth_sha256": "0" * 64,
    }
    (proof_dir / "current-999999.json").write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(process_guard, "_proc_facts", lambda pid: dict(facts))
    state = process_guard._process_state(
        owner,
        proof_dir,
        expected=expected,
        expected_argv_hash=argv_hash,
        fresh=True,
    )
    assert state["process"] == "LIVE"
    pointer = json.loads((proof_dir / "current-999999.json").read_text(encoding="utf-8"))
    assert pointer["start_ticks"] == 2
    # Without fresh, a same-PID but different incarnation is refused.
    (proof_dir / "current-999999.json").write_text(json.dumps(old), encoding="utf-8")
    stale = process_guard._process_state(
        owner,
        proof_dir,
        expected=expected,
        expected_argv_hash=argv_hash,
    )
    assert stale["process"] == "MISMATCH"


def test_supervisor_live_attach_never_waits_for_port(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    launch = tmp_path / "comfy-launch.json"
    launch.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "read_manifest",
        lambda path, root: {"port": 8188, "kind": "comfy", "run_id": "r", "nonce": "n"},
    )
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "assess_owner",
        lambda **kwargs: {"action": "ATTACH", "process": "LIVE"},
    )
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "launch_with_proof",
        lambda root, path: {"schema": "result", "status": "ATTACHED", "pid": 7},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("live attach must not wait for a free port")

    monkeypatch.setattr(comfy_supervisor, "_wait_port_free", forbidden)
    assert comfy_supervisor.main([str(tmp_path), str(launch)]) == 0
    assert '"status": "ATTACHED"' in capsys.readouterr().out


def test_supervisor_dead_owner_archives_only_after_decision_and_waits_for_spawn(
    tmp_path: Path, monkeypatch
) -> None:
    launch = tmp_path / "comfy-launch.json"
    launch.write_text("{}", encoding="utf-8")
    owner = tmp_path / "comfy-owner.json"
    ack = tmp_path / "comfy-ack.json"
    owner.write_text("owner-bytes", encoding="utf-8")
    ack.write_text("ack-bytes", encoding="utf-8")
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "read_manifest",
        lambda path, root: {"port": 8188, "kind": "comfy", "run_id": "r", "nonce": "n"},
    )
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "assess_owner",
        lambda **kwargs: {"action": "ARCHIVE_DEAD", "process": "DEAD"},
    )
    calls = []
    monkeypatch.setattr(comfy_supervisor, "_wait_port_free", lambda port: calls.append(port) or True)
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "launch_with_proof",
        lambda root, path: {"schema": "result", "status": "STARTED", "pid": 8},
    )
    assert comfy_supervisor.main([str(tmp_path), str(launch)]) == 0
    assert calls == [8188]
    assert not owner.exists() and not ack.exists()
    archived = list((tmp_path / "comfy-archive").glob("*.dead-*"))
    assert len(archived) == 2


def test_supervisor_refusal_retains_owner_and_ack(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "comfy-launch.json"
    launch.write_text("{}", encoding="utf-8")
    owner = tmp_path / "comfy-owner.json"
    ack = tmp_path / "comfy-ack.json"
    owner.write_text("owner-bytes", encoding="utf-8")
    ack.write_text("ack-bytes", encoding="utf-8")
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "read_manifest",
        lambda path, root: {"port": 8188, "kind": "comfy", "run_id": "r", "nonce": "n"},
    )
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "assess_owner",
        lambda **kwargs: {"action": "REFUSE", "reason": "foreign owner"},
    )
    monkeypatch.setattr(
        comfy_supervisor.guard,
        "launch_with_proof",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    assert comfy_supervisor.main([str(tmp_path), str(launch)]) == 3
    assert owner.read_text(encoding="utf-8") == "owner-bytes"
    assert ack.read_text(encoding="utf-8") == "ack-bytes"
    assert not (tmp_path / "comfy-archive").exists()

def _download_attempt(
    root: Path,
    attempt_id: str = "0001",
    *,
    run_id: str = "r",
    nonce: str = "n",
    worker_bytes: bytes = b"shim",
    payload_bytes: bytes = b'{"job":1}',
) -> tuple[Path, dict, dict, str, dict]:
    import hashlib

    attempt = root / "attempts" / attempt_id
    attempt.mkdir(parents=True)
    worker = attempt / "v3_download_worker.py"
    payload = attempt / "job.json"
    worker.write_bytes(worker_bytes)
    payload.write_bytes(payload_bytes)
    manifest = {
        "schema": process_guard.LAUNCH_SCHEMA,
        "kind": "download_worker",
        "run_id": run_id,
        "nonce": nonce,
        "root": str(attempt),
        "worker": str(worker),
        "worker_sha256": hashlib.sha256(worker_bytes).hexdigest(),
        "payload": str(payload),
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "deadline_epoch": 9_999_999_999,
        "status": str(attempt / "status.json"),
        "progress": str(attempt / "progress.json"),
        "heartbeat": str(attempt / "heartbeat.json"),
        "log": str(attempt / "download.log"),
        "owner": str(attempt / "worker-owner.json"),
        "ack": str(attempt / "worker-ack.json"),
    }
    (attempt / "launch.json").write_text(json.dumps(manifest), encoding="utf-8")
    expected = process_guard.expected_owner(manifest)
    argv_hash = process_guard.expected_argv_sha256(manifest)
    owner = {**expected, "pid": 4242, "argv_sha256": argv_hash}
    ack = dict(owner)
    (attempt / "worker-owner.json").write_text(json.dumps(owner), encoding="utf-8")
    (attempt / "worker-ack.json").write_text(json.dumps(ack), encoding="utf-8")
    (attempt / "status.json").write_text(
        json.dumps({"schema": process_guard.STATUS_SCHEMA, "run_id": run_id, "nonce": nonce, "status": "RUNNING"}),
        encoding="utf-8",
    )
    facts = {
        "pid": 4242,
        "start_ticks": 7,
        "state": "S",
        "exe": "/usr/bin/python3",
        "cmdline_sha256": "a" * 64,
        "argv_sha256": argv_hash,
        "boot_id": "boot",
    }
    proof_dir = attempt / "process-proofs"
    proof_dir.mkdir()
    birth = {
        "schema": process_guard.BIRTH_SCHEMA,
        "kind": "download_worker",
        "run_id": run_id,
        "nonce": nonce,
        "manifest_sha256": expected["command_sha256"],
        "created_at": 1.0,
        **facts,
    }
    (proof_dir / "birth-4242-7.json").write_text(json.dumps(birth), encoding="utf-8")
    return attempt, manifest, expected, argv_hash, facts


def test_process_live_requires_child_birth_baseline_and_no_heuristic(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, owner, _ = _records(tmp_path)
    manifest = _manifest()
    expected, argv_hash = _expected(manifest)
    facts = _facts(owner)
    monkeypatch.setattr(process_guard, "_proc_facts", lambda pid: dict(facts))
    proof_dir = tmp_path / "comfy-process-proofs"
    state = process_guard._process_state(
        owner, proof_dir, expected=expected, expected_argv_hash=argv_hash
    )
    assert state["process"] == "MISSING_BIRTH"
    assert not (proof_dir / "current-999999.json").exists()
    # A baseline for another exact incarnation is present; same argv/new
    # ticks/exe/boot with no matching child birth record is still refused.
    _write_birth(proof_dir, owner, facts)
    for override in (
        {"start_ticks": 43},
        {"exe": "/other/python"},
        {"boot_id": "other-boot"},
    ):
        candidate = {**facts, **override}
        monkeypatch.setattr(process_guard, "_proc_facts", lambda pid, c=candidate: dict(c))
        state = process_guard._process_state(
            owner, proof_dir, expected=expected, expected_argv_hash=argv_hash
        )
        assert state["process"] in {"MISSING_BIRTH", "MISMATCH"}
        assert not (proof_dir / "current-999999.json").exists()


def test_inspect_download_validates_current_manifest_and_birth(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "owned"
    attempt, manifest, expected, argv_hash, facts = _download_attempt(root)
    monkeypatch.setattr(process_guard, "_proc_facts", lambda pid: dict(facts))
    result = process_guard.inspect_download(
        root=root, attempt_dir=attempt, run_id="r", nonce="n"
    )
    assert result["state"] == "OK"
    assert result["process"] == "LIVE"
    assert result["manifest_sha256"] == expected["command_sha256"]

    # A different attempt with the same run/nonce and copied exact process
    # proof must not be accepted against its own manifest.
    other_attempt, _, _, _, _ = _download_attempt(
        root, "0002", worker_bytes=b"other-shim", payload_bytes=b'{"job":2}'
    )
    for name in ("worker-owner.json", "worker-ack.json", "status.json"):
        (other_attempt / name).write_bytes((attempt / name).read_bytes())
    other_proof = other_attempt / "process-proofs"
    for path in other_proof.glob("*.json"):
        path.unlink()
    for path in (attempt / "process-proofs").glob("*.json"):
        (other_proof / path.name).write_bytes(path.read_bytes())
    refused = process_guard.inspect_download(
        root=root, attempt_dir=other_attempt, run_id="r", nonce="n"
    )
    assert refused["state"] == "UNREADABLE"
    assert refused.get("process") != "LIVE"

    # Missing current manifest is UNREADABLE, not MISSING.
    (attempt / "launch.json").unlink()
    missing_manifest = process_guard.inspect_download(
        root=root, attempt_dir=attempt, run_id="r", nonce="n"
    )
    assert missing_manifest["state"] == "UNREADABLE"
    assert missing_manifest.get("process") != "LIVE"


def test_inspect_download_refuses_changed_job_and_old_command(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "owned"
    attempt, manifest, expected, argv_hash, facts = _download_attempt(root)
    monkeypatch.setattr(process_guard, "_proc_facts", lambda pid: dict(facts))
    # Job bytes changed after the manifest was written.
    (attempt / "job.json").write_bytes(b'{"tampered":true}')
    tampered = process_guard.inspect_download(
        root=root, attempt_dir=attempt, run_id="r", nonce="n"
    )
    assert tampered["state"] == "UNREADABLE"
    # Restore the payload and mutate owner command identity.
    (attempt / "job.json").write_bytes(b'{"job":1}')
    owner = json.loads((attempt / "worker-owner.json").read_text(encoding="utf-8"))
    owner["command_sha256"] = "f" * 64
    (attempt / "worker-owner.json").write_text(json.dumps(owner), encoding="utf-8")
    changed = process_guard.inspect_download(
        root=root, attempt_dir=attempt, run_id="r", nonce="n"
    )
    assert changed["state"] == "UNREADABLE"
    # An old-command birth (different argv hash) is not a live proof.
    owner["command_sha256"] = expected["command_sha256"]
    (attempt / "worker-owner.json").write_text(json.dumps(owner), encoding="utf-8")
    birth_path = next((attempt / "process-proofs").glob("birth-*.json"))
    birth = json.loads(birth_path.read_text(encoding="utf-8"))
    birth["argv_sha256"] = "0" * 64
    birth_path.write_text(json.dumps(birth), encoding="utf-8")
    old_command = process_guard.inspect_download(
        root=root, attempt_dir=attempt, run_id="r", nonce="n"
    )
    assert old_command["state"] == "UNREADABLE"
    assert old_command.get("process") != "LIVE"

