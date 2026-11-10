from __future__ import annotations

from pathlib import Path

import pytest

from scripts.i1 import state


def _new(**overrides):
    values = dict(
        identity_key_value="a" * 64,
        host="h",
        user="u",
        port=22,
        fingerprint="SHA256:x",
        workspace="/workspace",
        profile_id="wan22-14b-ab",
        profile_sha256="b" * 64,
    )
    values.update(overrides)
    return state.new_state(**values)


def test_identity_key_changes_for_server_workspace_or_profile() -> None:
    base = dict(
        host="h",
        user="u",
        port=22,
        fingerprint="SHA256:x",
        workspace="/workspace",
        profile_sha256="b" * 64,
    )
    first = state.identity_key(**base)
    assert first != state.identity_key(**{**base, "host": "h2"})
    assert first != state.identity_key(**{**base, "fingerprint": "SHA256:y"})
    assert first != state.identity_key(**{**base, "workspace": "/other"})
    assert first != state.identity_key(**{**base, "profile_sha256": "c" * 64})


def test_state_transition_and_no_ready_from_install() -> None:
    value = _new()
    value = state.transition(value, "INSPECTED")
    value = state.transition(value, "PLANNED")
    value = state.transition(value, "INSTALLING")
    value = state.transition(value, "INSTALLED")
    assert value["phase"] == "INSTALLED"
    with pytest.raises(state.StateError):
        state.transition(value, "READY")
    verified = state.transition(value, "VERIFYING")
    ready = state.transition(verified, "READY")
    assert ready["phase"] == "READY"


def test_interrupted_resume_and_failed_terminal() -> None:
    value = state.transition(state.transition(state.transition(_new(), "INSPECTED"), "PLANNED"), "INSTALLING")
    interrupted = state.transition(value, "INTERRUPTED")
    assert state.transition(interrupted, "INSTALLING")["phase"] == "INSTALLING"
    failed = state.transition(value, "FAILED")
    with pytest.raises(state.StateError):
        state.transition(failed, "INSTALLING")


def test_record_roundtrip_has_no_secret_fields(tmp_path: Path) -> None:
    record = {
        "schema": state.RECORD_SCHEMA,
        "name": "srv",
        "identity": {
            "host": "example.test",
            "user": "root",
            "port": 22,
            "fingerprint": "SHA256:x",
            "workspace": "/workspace",
            "profile_id": "wan22-14b-ab",
            "profile_sha256": "b" * 64,
        },
        "auth": {"kind": "key", "path": str(tmp_path / "id")},
        "known_hosts": str(tmp_path / "known_hosts"),
    }
    state.save_record(tmp_path, "srv", record)
    loaded = state.load_record(tmp_path, "srv")
    assert loaded["auth"]["path"] == record["auth"]["path"]
    assert "private" not in repr(loaded).lower()
    assert state.resolve_name(tmp_path, None) == "srv"


def test_state_is_keyed_by_identity(tmp_path: Path) -> None:
    record1 = {
        "schema": state.RECORD_SCHEMA,
        "name": "a",
        "identity": {
            "host": "h",
            "user": "u",
            "port": 22,
            "fingerprint": "SHA256:x",
            "workspace": "/workspace",
            "profile_id": "wan22-14b-ab",
            "profile_sha256": "b" * 64,
        },
        "auth": {"kind": "agent"},
        "known_hosts": str(tmp_path / "kh"),
    }
    record2 = {
        **record1,
        "name": "b",
        "identity": {
            **record1["identity"],
            "host": "h2",
        },
    }
    s1 = state.state_for_record(tmp_path, record1)
    s2 = state.state_for_record(tmp_path, record2)
    assert s1["identity_key"] != s2["identity_key"]
    assert s1["run_id"] != s2["run_id"]
    state.save_state_for_record(tmp_path, record1, s1)
    with pytest.raises(state.StateError):
        state.save_state_for_record(tmp_path, record2, s1)
