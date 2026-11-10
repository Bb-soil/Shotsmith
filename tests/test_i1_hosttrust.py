from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.i1 import hosttrust


@dataclass
class Proc:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def _key(seed: bytes = b"i1-host-key") -> tuple[str, str]:
    blob = seed
    b64 = base64.b64encode(blob).decode()
    fp = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return b64, fp


def test_unknown_host_requires_exact_explicit_fingerprint(tmp_path: Path) -> None:
    b64, fp = _key()
    known = tmp_path / "known_hosts"

    def keyscan(argv, **kwargs):
        return Proc(0, f"example.test ssh-ed25519 {b64}\n")

    with pytest.raises(hosttrust.HostTrustError):
        hosttrust.trust_scanned_key(
            known_hosts=known,
            host="example.test",
            port=22,
            expected_fingerprint="SHA256:wrong",
            runner=keyscan,
        )
    assert not known.exists() or known.read_text() == ""
    kept = hosttrust.trust_scanned_key(
        known_hosts=known,
        host="example.test",
        port=22,
        expected_fingerprint=fp,
        runner=keyscan,
    )
    assert kept.fingerprint == fp
    assert b64 in known.read_text()
    assert "PRIVATE KEY" not in known.read_text()


def test_verify_trusted_and_changed_key_never_removes_entry(tmp_path: Path) -> None:
    b64, fp = _key()
    known = tmp_path / "known_hosts"
    known.write_text(f"example.test ssh-ed25519 {b64}\n", encoding="utf-8")

    def keygen(argv, **kwargs):
        return Proc(0, f"example.test ssh-ed25519 {b64}\n")

    hosttrust.verify_trusted(
        known_hosts=known,
        host="example.test",
        port=22,
        expected_fingerprint=fp,
        runner=keygen,
        live_keys=hosttrust.parse_keyscan(f"example.test ssh-ed25519 {b64}", host="example.test", port=22),
    )
    other_b64, _ = _key(b"changed")
    before = known.read_text()
    with pytest.raises(hosttrust.HostTrustError):
        hosttrust.verify_trusted(
            known_hosts=known,
            host="example.test",
            port=22,
            expected_fingerprint="SHA256:other",
            runner=keygen,
        )
    assert known.read_text() == before


def test_live_scan_mismatch_is_refused() -> None:
    b64, fp = _key()
    other_b64, other_fp = _key(b"other")
    live = hosttrust.parse_keyscan(f"example.test ssh-ed25519 {other_b64}", host="example.test", port=22)
    with pytest.raises(hosttrust.HostTrustError):
        hosttrust.verify_trusted(
            known_hosts=Path("does-not-matter"),
            host="example.test",
            port=22,
            expected_fingerprint=fp,
            live_keys=live,
        )


def test_standard_and_nonstandard_known_hosts_lookup(tmp_path: Path) -> None:
    b64, fp = _key(b"port-key")
    for port, name in ((22, "example.test"), (2222, "[example.test]:2222")):
        known = tmp_path / f"known-{port}"
        known.write_text(f"{name} ssh-ed25519 {b64}\n", encoding="utf-8")

        def keygen(argv, **kwargs):
            return Proc(0, f"{name} ssh-ed25519 {b64}\n")

        hosttrust.verify_trusted(
            known_hosts=known,
            host="example.test",
            port=port,
            expected_fingerprint=fp,
            runner=keygen,
        )


def test_host_key_name_and_automatic_removal_forbidden() -> None:
    assert hosttrust.host_key_name("h", 22) == "h"
    assert hosttrust.host_key_name("h", 2222) == "[h]:2222"
    with pytest.raises(hosttrust.HostTrustError):
        hosttrust.remove_untrusted_key(known_hosts="x", host="h", port=22)
