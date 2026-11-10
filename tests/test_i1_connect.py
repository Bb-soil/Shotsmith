from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.i1 import ops
from scripts.i1.errors import HostTrustError


@dataclass
class Proc:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def _key(seed: bytes) -> tuple[str, str]:
    b64 = base64.b64encode(seed).decode()
    fp = "SHA256:" + base64.b64encode(hashlib.sha256(seed).digest()).decode().rstrip("=")
    return b64, fp


class Runner:
    def run_remote(self, identity, argv, timeout=None, **kwargs):
        return Proc(0)


def test_connect_explicit_trust_and_changed_key_block(tmp_path: Path) -> None:
    b64, fp = _key(b"i1-key")
    scanned = {"value": b64}

    def trust_runner(argv, **kwargs):
        if argv[0].endswith("ssh-keyscan") or "ssh-keyscan" in argv[0]:
            return Proc(0, f"example.test ssh-ed25519 {scanned['value']}\n")
        # ssh-keygen -F
        if len(argv) >= 5 and argv[1] == "-F":
            return Proc(0, f"example.test ssh-ed25519 {b64}\n")
        return Proc(0)

    args = ops.ConnectArgs(
        name="srv",
        host="example.test",
        user="root",
        port=22,
        workspace="/workspace",
        agent=True,
        accept_fingerprint=fp,
    )
    result = ops.connect(
        args=args,
        state_dir=tmp_path,
        runner=Runner(),
        trust_runner=trust_runner,
    )
    assert result["status"] == "CONNECTED"
    assert result["fingerprint"] == fp
    record = (tmp_path / "profiles" / "srv.json").read_text(encoding="utf-8")
    assert "PRIVATE" not in record
    assert "passphrase" not in record.lower()
    # Unknown changed host key must be refused and must not remove the old entry.
    scanned["value"] = base64.b64encode(b"changed-key").decode()
    with pytest.raises(HostTrustError):
        ops.connect(
            args=ops.ConnectArgs(
                name="other",
                host="example.test",
                user="root",
                port=22,
                workspace="/workspace",
                agent=True,
                accept_fingerprint="SHA256:" + base64.b64encode(hashlib.sha256(b"changed-key").digest()).decode().rstrip("="),
            ),
            state_dir=tmp_path,
            runner=Runner(),
            trust_runner=trust_runner,
        )


def test_connect_requires_accept_in_noninteractive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    b64, _ = _key(b"key2")

    def trust_runner(argv, **kwargs):
        return Proc(0, f"example.test ssh-ed25519 {b64}\n")

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(HostTrustError):
        ops.connect(
            args=ops.ConnectArgs(
                name="srv",
                host="example.test",
                user="root",
                port=22,
                workspace="/workspace",
                agent=True,
            ),
            state_dir=tmp_path,
            runner=Runner(),
            trust_runner=trust_runner,
        )
