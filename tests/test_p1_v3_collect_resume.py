from __future__ import annotations

import base64
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.p1 import remote as remote_module
from scripts.p1 import v3_lifecycle, v3_operator
from scripts.p1.remote import RemoteIdentity, RemoteResult, RemoteStreamPartialError

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = b"exact-mp4-fixture-bytes-" * 20
BLOB = b"fake-ed25519-key-blob"
FINGERPRINT = "SHA256:" + base64.b64encode(hashlib.sha256(BLOB).digest()).decode().rstrip("=")


def _identity(tmp_path: Path) -> RemoteIdentity:
    identity = tmp_path / "id_ed25519"
    known = tmp_path / "known_hosts"
    identity.write_text("not-a-real-key", encoding="utf-8")
    known.write_text(
        f"[example.test]:22 ssh-ed25519 {base64.b64encode(BLOB).decode()}\n",
        encoding="utf-8",
    )
    return RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=identity,
        known_hosts=known,
    )


class _CommandTransport:
    """Fake SSH command transport, not a replacement for collect/fetch."""

    ssh_keygen_path = shutil.which("ssh-keygen")

    def __init__(self, source: Path, remote_path: str, *, interrupt_after: int = 40):
        self.source = source
        self.remote_path = remote_path
        self.interrupt_after = interrupt_after
        self.interrupted = False
        self.staged: list[str] = []
        self.stream_calls: list[int] = []
        self.error_mode = "partial"

    def _header(self, offset: int) -> bytes:
        data = self.source.read_bytes()
        return (
            json.dumps(
                {
                    "schema": "anime-video-studio/p1-v3-stream/v1",
                    "path": self.remote_path,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "offset": offset,
                    "length": len(data) - offset,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )

    def put_file(self, identity, remote_path, local_path, **kwargs):
        self.staged.append(remote_path)

    def run_remote(self, identity, args, timeout=None):
        return RemoteResult(tuple(args), 0, "", "")

    def stream_remote_to_file(self, identity, remote_args, target, *, timeout, max_payload_bytes, **kwargs):
        offset = int(remote_args[4])
        self.stream_calls.append(offset)
        data = self.source.read_bytes()
        header = self._header(offset)
        if not self.interrupted:
            partial = data[offset : offset + self.interrupt_after]
            target.write_bytes(partial)
            self.interrupted = True
            if self.error_mode == "generic":
                raise remote_module.RemoteCommandError("simulated generic disconnect")
            raise RemoteStreamPartialError(
                "simulated validated partial",
                header=json.loads(header),
                payload_size=len(partial),
                target=target,
                returncode=255,
                timed_out=False,
            )
        target.write_bytes(data[offset:])
        return header.split(b"\n", 1)[0], len(data) - offset


def _setup(tmp_path: Path) -> tuple[Path, Path, RemoteIdentity, _CommandTransport]:
    run = tmp_path / "run"
    v3_operator.prepare(run_dir=run)
    identity = _identity(tmp_path)
    now = int(time.time())
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=run.name,
        session=v3_lifecycle.SessionBinding(
            order_id=99001,
            server_id=112213,
            created_at=now - 5,
            currency="USD",
            cap_usd="1.00",
            duration_seconds=3600,
            planned_cancel_at=now + 300,
            hard_end_at=now + 600,
            quote_hash="a" * 64,
        ),
        bundle_hashes={"config": "a" * 64},
        phase="READY",
    )
    lifecycle = v3_lifecycle.update_phase(
        run / "lifecycle.json",
        lifecycle,
        phase="READY",
        pipeline={
            "ssh_identity": {
                "host": identity.host,
                "user": identity.user,
                "port": identity.port,
                "known_hosts": str(identity.known_hosts),
                "fingerprint": FINGERPRINT,
            }
        },
    )
    (run / "remote-export.json").write_text(
        json.dumps(
            {
                "schema": "anime-video-studio/p1-v3-remote-export/v1",
                "run_id": run.name,
                "nonce": lifecycle.nonce,
                "session": lifecycle.session.to_dict(),
                "ssh_identity": lifecycle.pipeline["ssh_identity"],
                "job_id": "job-1",
                "prompt_id": "prompt-1",
                "remote_root": "/workspace",
                "remote_artifact_path": "/workspace/result.mp4",
                "remote_helper_path": "/workspace/scripts/p1/v3_remote_fetch_helper.py",
                "relative_path": "result.mp4",
                "filename": "result.mp4",
                "size": len(PAYLOAD),
                "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "remote-result.mp4"
    source.write_bytes(PAYLOAD)
    transport = _CommandTransport(source, "/workspace/result.mp4")
    return run, source, identity, transport


def _collect_args(run: Path, source: Path) -> list[str]:
    return [
        "collect",
        "--run-dir",
        str(run),
        "--remote-host",
        "example.test",
        "--remote-identity",
        str(run.parent / "id_ed25519"),
        "--remote-known-hosts",
        str(run.parent / "known_hosts"),
        "--expected-fingerprint",
        FINGERPRINT,
        "--remote-artifact-path",
        "/workspace/result.mp4",
        "--remote-helper-path",
        "/workspace/scripts/p1/v3_remote_fetch_helper.py",
        "--remote-expected-size",
        str(len(PAYLOAD)),
        "--remote-expected-sha256",
        hashlib.sha256(PAYLOAD).hexdigest(),
        "--json",
    ]


def test_cli_collect_interrupted_then_resumes_same_partial(tmp_path: Path, monkeypatch, capsys) -> None:
    run, source, identity, transport = _setup(tmp_path)
    monkeypatch.setattr(v3_operator, "RemoteRunner", lambda: transport)
    args = _collect_args(run, source)

    first = v3_operator.main(args)
    assert first == 2
    first_out = json.loads(capsys.readouterr().out)
    assert first_out["status"] == "REFUSED"
    partial = run / "collect-transfer" / "artifact.mp4.part"
    assert partial.is_file()
    assert partial.stat().st_size == 40
    state = json.loads((run / "collect-transfer.json").read_text())
    assert state["status"] == "FAILED"
    assert state["offset"] == 40
    assert transport.stream_calls == [0]

    second = v3_operator.main(args)
    assert second == 0, capsys.readouterr().out
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "COLLECTED_DECODE_NOT_VERIFIED"
    assert partial.read_bytes() == PAYLOAD
    assert transport.stream_calls == [0, 40]
    state = json.loads((run / "collect-transfer.json").read_text())
    assert state["status"] == "COMPLETE"
    assert state["offset"] == len(PAYLOAD)


def test_corrupt_prefix_preserves_partial_and_refuses(tmp_path: Path, monkeypatch, capsys) -> None:
    run, source, identity, transport = _setup(tmp_path)
    monkeypatch.setattr(v3_operator, "RemoteRunner", lambda: transport)
    args = _collect_args(run, source)
    assert v3_operator.main(args) == 2
    capsys.readouterr()
    partial = run / "collect-transfer" / "artifact.mp4.part"
    original = partial.read_bytes()
    corrupted = bytes([original[0] ^ 0xFF]) + original[1:]
    partial.write_bytes(corrupted)
    before_calls = list(transport.stream_calls)
    assert v3_operator.main(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "REFUSED"
    assert partial.read_bytes() == corrupted
    assert transport.stream_calls == before_calls


def test_foreign_transfer_identity_refuses_without_deleting(tmp_path: Path, monkeypatch, capsys) -> None:
    run, source, identity, transport = _setup(tmp_path)
    monkeypatch.setattr(v3_operator, "RemoteRunner", lambda: transport)
    args = _collect_args(run, source)
    assert v3_operator.main(args) == 2
    capsys.readouterr()
    partial = run / "collect-transfer" / "artifact.mp4.part"
    before = partial.read_bytes()
    state_path = run / "collect-transfer.json"
    state = json.loads(state_path.read_text())
    state["expected_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert v3_operator.main(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "REFUSED"
    assert partial.read_bytes() == before
    assert json.loads(state_path.read_text())["expected_sha256"] == "0" * 64

def test_generic_disconnect_does_not_publish_unvalidated_partial(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    run, source, identity, transport = _setup(tmp_path)
    transport.error_mode = "generic"
    monkeypatch.setattr(v3_operator, "RemoteRunner", lambda: transport)
    assert v3_operator.main(_collect_args(run, source)) == 2
    capsys.readouterr()
    assert not (run / "collect-transfer" / "artifact.mp4.part").exists()
