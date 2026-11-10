from __future__ import annotations

import os
from pathlib import Path

from scripts.p1.remote import RemoteIdentity, RemoteRunner
from scripts.p1.v3_bootstrap import build_tunnel_argv


def test_agent_identity_omits_identity_file(tmp_path: Path) -> None:
    known = tmp_path / "known_hosts"
    known.write_text("", encoding="utf-8")
    identity = RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=None,
        known_hosts=known,
    )
    runner = RemoteRunner(
        ssh_path="/usr/bin/ssh",
        scp_path="/usr/bin/scp",
        ssh_keygen_path="/usr/bin/ssh-keygen",
    )
    argv = runner.ssh_argv(identity, ("true",))
    assert "-i" not in argv
    tunnel = build_tunnel_argv(identity, local_port=8189, remote_port=8188)
    assert "-i" not in tunnel
    assert "BatchMode=yes" in tunnel
    assert "StrictHostKeyChecking=yes" in tunnel


def test_i1_standard_known_hosts_global_isolation_and_algorithms(tmp_path: Path) -> None:
    known = tmp_path / "known_hosts"
    known.write_text("example.test ssh-ed25519 AAAA\n", encoding="utf-8")
    identity = RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=None,
        known_hosts=known,
        standard_known_hosts=True,
        global_known_hosts=Path(os.devnull),
        host_key_algorithms="ssh-ed25519",
    )
    runner = RemoteRunner(
        ssh_path="/usr/bin/ssh",
        scp_path="/usr/bin/scp",
        ssh_keygen_path="/usr/bin/ssh-keygen",
    )
    argv = runner.ssh_argv(identity, ("true",))
    assert identity.host_key_name == "example.test"
    assert any(str(item).startswith("GlobalKnownHostsFile=") for item in argv)
    assert "HostKeyAlgorithms=ssh-ed25519" in argv
    assert "UserKnownHostsFile=" + str(known) in argv


def test_legacy_p1_bracket_known_hosts_is_preserved(tmp_path: Path) -> None:
    known = tmp_path / "known_hosts"
    known.write_text("[example.test]:22 ssh-ed25519 AAAA\n", encoding="utf-8")
    identity = RemoteIdentity(
        host="example.test",
        user="root",
        port=22,
        identity_file=None,
        known_hosts=known,
    )
    assert identity.host_key_name == "[example.test]:22"
    argv = RemoteRunner(
        ssh_path="/usr/bin/ssh",
        scp_path="/usr/bin/scp",
        ssh_keygen_path="/usr/bin/ssh-keygen",
    ).ssh_argv(identity, ("true",))
    assert "GlobalKnownHostsFile" not in " ".join(argv)


def test_windows_make_identity_tunnel_config_passes_validator(tmp_path: Path) -> None:
    from scripts.i1 import remote as i1_remote
    from scripts.p1 import v3_tunnel_child
    from scripts.p1.v3_bootstrap import OwnedTunnel

    known = tmp_path / "known_hosts"
    known.write_text("example.test ssh-ed25519 AAAA\n", encoding="utf-8")
    global_known = tmp_path / "global-empty"
    global_known.write_text("", encoding="utf-8")
    record = {
        "identity": {
            "host": "example.test",
            "user": "root",
            "port": 22,
            "fingerprint": "SHA256:x",
            "workspace": "/workspace",
            "profile_id": "p",
            "profile_sha256": "b" * 64,
        },
        "auth": {"kind": "agent"},
        "known_hosts": str(known),
        "global_known_hosts": str(global_known),
        "host_key_type": "ssh-ed25519",
    }
    identity = i1_remote.make_identity(record)
    tunnel = OwnedTunnel(
        identity=identity,
        local_port=8189,
        remote_port=8188,
        log_path=tmp_path / "tunnel.log",
        run_id="r",
        nonce="n",
        owner_path=tmp_path / "tunnel-owner.json",
        expected_fingerprint="SHA256:x",
    )
    config = tunnel._config_payload()
    assert v3_tunnel_child.validate_config(config)["global_known_hosts"] == str(global_known)
