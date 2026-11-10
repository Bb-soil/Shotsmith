"""Local durable state and strict phase transitions for I1."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .util import (
    atomic_json,
    read_json,
    sha256_mapping,
    validate_name,
    validate_posix_absolute,
    validate_workspace,
)

STATE_SCHEMA = "anime-video-studio/i1-state/v1"
RECORD_SCHEMA = "anime-video-studio/i1-profile-record/v1"

PHASES = (
    "NEW",
    "INSPECTED",
    "PLANNED",
    "INSTALLING",
    "INTERRUPTED",
    "INSTALLED",
    "VERIFYING",
    "READY",
    "FAILED",
    "OFFLINE_VERIFIED",
)

_TRANSITIONS: dict[str, frozenset[str]] = {
    "NEW": frozenset({"INSPECTED", "FAILED"}),
    "INSPECTED": frozenset({"PLANNED", "INSTALLING", "FAILED"}),
    "PLANNED": frozenset({"INSTALLING", "FAILED"}),
    "INSTALLING": frozenset({"INTERRUPTED", "INSTALLED", "FAILED"}),
    "INTERRUPTED": frozenset({"INSTALLING", "FAILED"}),
    "INSTALLED": frozenset({"VERIFYING", "FAILED"}),
    "VERIFYING": frozenset({"READY", "FAILED", "INSTALLED"}),
    "READY": frozenset({"FAILED", "VERIFYING"}),
    "FAILED": frozenset(),
    "OFFLINE_VERIFIED": frozenset({"FAILED"}),
}


class StateError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def target_id(
    *, host: str, user: str, port: int, fingerprint: str
) -> str:
    payload = {
        "host": host,
        "user": user,
        "port": int(port),
        "fingerprint": fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def identity_key(
    *,
    host: str,
    user: str,
    port: int,
    fingerprint: str,
    workspace: str,
    profile_sha256: str,
) -> str:
    payload = {
        "host": host,
        "user": user,
        "port": int(port),
        "fingerprint": fingerprint,
        "workspace": workspace,
        "profile_sha256": profile_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def state_path(state_dir: Path | str, key: str) -> Path:
    if not isinstance(key, str) or len(key) != 64:
        raise StateError("identity key is invalid")
    return Path(state_dir) / "runs" / key / "state.json"


def new_state(
    *,
    identity_key_value: str,
    host: str,
    user: str,
    port: int,
    fingerprint: str,
    workspace: str,
    profile_id: str,
    profile_sha256: str,
    target_id_value: str | None = None,
) -> dict[str, Any]:
    if not isinstance(profile_sha256, str) or len(profile_sha256) != 64:
        raise StateError("profile hash is invalid")
    workspace = validate_workspace(workspace)
    target = target_id_value or target_id(
        host=host, user=user, port=port, fingerprint=fingerprint
    )
    return {
        "schema": STATE_SCHEMA,
        "identity_key": identity_key_value,
        "run_id": uuid.uuid4().hex,
        "nonce": uuid.uuid4().hex,
        "identity": {
            "host": host,
            "user": user,
            "port": int(port),
            "fingerprint": fingerprint,
            "workspace": workspace,
            "profile_id": profile_id,
            "profile_sha256": profile_sha256,
            "target_id": target,
        },
        "phase": "NEW",
        "stage": None,
        "remote_facts": {},
        "plan": {},
        "progress": {},
        "artifacts": {},
        "worker": {},
        "comfy": {},
        "tunnel": {},
        "ui": {},
        "readiness": {},
        "observation": {},
        "last_error": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


def validate_state(data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(data, Mapping) or data.get("schema") != STATE_SCHEMA:
        raise StateError("state schema is invalid")
    for key in ("identity_key", "run_id", "nonce", "phase"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise StateError(f"state {key} is invalid")
    if data["phase"] not in PHASES:
        raise StateError("state phase is unknown")
    identity = data.get("identity")
    if not isinstance(identity, Mapping):
        raise StateError("state identity is missing")
    if identity.get("workspace") != validate_workspace(identity.get("workspace")):
        raise StateError("state workspace is invalid")
    if not isinstance(identity.get("profile_sha256"), str) or len(
        identity["profile_sha256"]
    ) != 64:
        raise StateError("state profile hash is invalid")
    expected_target = target_id(
        host=str(identity.get("host", "")),
        user=str(identity.get("user", "")),
        port=int(identity.get("port", 0)),
        fingerprint=str(identity.get("fingerprint", "")),
    )
    if identity.get("target_id") != expected_target:
        raise StateError("state target identity is invalid")
    for key in ("remote_facts", "plan", "progress", "artifacts", "worker", "comfy", "tunnel", "ui", "readiness", "observation"):
        value = data.get(key)
        if key == "observation" and value is None:
            # Legacy records predate observation metadata; absence is unknown,
            # never a fabricated confirmation timestamp.
            value = {}
        if not isinstance(value, Mapping):
            raise StateError(f"state {key} must be an object")
    return dict(data)


def write_state(path: Path | str, state: Mapping[str, Any]) -> None:
    payload = validate_state(state)
    payload["updated_at"] = now_iso()
    atomic_json(path, payload)


def load_state(path: Path | str) -> dict[str, Any]:
    try:
        return validate_state(read_json(path))
    except (OSError, ValueError) as exc:
        raise StateError(f"state is unavailable: {path}") from exc


def transition(
    state: Mapping[str, Any],
    phase: str,
    *,
    stage: str | None = None,
    error: str | None = None,
    **updates: Any,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise StateError(f"unknown phase {phase}")
    current = str(state.get("phase"))
    if phase != current and phase not in _TRANSITIONS.get(current, frozenset()):
        raise StateError(f"illegal phase transition {current} -> {phase}")
    result = dict(state)
    result["phase"] = phase
    result["stage"] = stage
    result["last_error"] = error
    for key, value in updates.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    result["updated_at"] = now_iso()
    return validate_state(result)


def update_stage(state: Mapping[str, Any], stage: str | None) -> dict[str, Any]:
    result = dict(state)
    result["stage"] = stage
    result["updated_at"] = now_iso()
    return validate_state(result)


def run_dir(state_dir: Path | str, key: str) -> Path:
    return Path(state_dir) / "runs" / key


def record_path(state_dir: Path | str, name: str) -> Path:
    return Path(state_dir) / "profiles" / (validate_name(name, "profile name") + ".json")


def known_hosts_path(state_dir: Path | str, name: str | None = None) -> Path:
    base = Path(state_dir)
    if name is None:
        return base / "known_hosts"
    return base / "known_hosts" / validate_name(name, "profile name")


def global_known_hosts_path(state_dir: Path | str, name: str) -> Path:
    return Path(state_dir) / "known_hosts" / (validate_name(name, "profile name") + ".global-empty")


def save_record(state_dir: Path | str, name: str, record: Mapping[str, Any]) -> Path:
    payload = dict(record)
    payload["schema"] = RECORD_SCHEMA
    payload["name"] = validate_name(name, "profile name")
    payload["updated_at"] = now_iso()
    path = record_path(state_dir, name)
    atomic_json(path, payload)
    return path


def load_record(state_dir: Path | str, name: str) -> dict[str, Any]:
    path = record_path(state_dir, name)
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        raise StateError(f"connection profile is missing: {name}") from exc
    if data.get("schema") != RECORD_SCHEMA or data.get("name") != name:
        raise StateError("connection profile identity is invalid")
    identity = data.get("identity")
    if not isinstance(identity, Mapping):
        raise StateError("connection profile identity is missing")
    for key in ("host", "user", "fingerprint", "workspace", "profile_sha256"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise StateError(f"connection profile {key} is invalid")
    validate_workspace(identity["workspace"])
    if not isinstance(identity.get("port"), int):
        raise StateError("connection profile port is invalid")
    auth = data.get("auth")
    if not isinstance(auth, Mapping) or auth.get("kind") not in {"key", "agent"}:
        raise StateError("connection profile auth is invalid")
    if auth.get("kind") == "key" and not isinstance(auth.get("path"), str):
        raise StateError("connection profile key path is invalid")
    return data


def list_records(state_dir: Path | str) -> list[dict[str, Any]]:
    folder = Path(state_dir) / "profiles"
    if not folder.is_dir():
        return []
    records = []
    for path in sorted(folder.glob("*.json")):
        try:
            records.append(load_record(state_dir, path.stem))
        except StateError:
            continue
    return records


def resolve_name(state_dir: Path | str, name: str | None) -> str:
    if name:
        return validate_name(name, "profile name")
    records = list_records(state_dir)
    if len(records) == 1:
        return str(records[0]["name"])
    if not records:
        raise StateError("no connection profile; run worker connect first")
    raise StateError("multiple connection profiles; pass --name")


def default_record(state_dir: Path | str, name: str | None) -> tuple[str, dict[str, Any]]:
    resolved = resolve_name(state_dir, name)
    return resolved, load_record(state_dir, resolved)


def state_for_record(state_dir: Path | str, record: Mapping[str, Any]) -> dict[str, Any]:
    identity = record["identity"]
    key = identity_key(
        host=identity["host"],
        user=identity["user"],
        port=int(identity["port"]),
        fingerprint=identity["fingerprint"],
        workspace=identity["workspace"],
        profile_sha256=identity["profile_sha256"],
    )
    path = state_path(state_dir, key)
    if not path.is_file():
        return new_state(
            identity_key_value=key,
            host=identity["host"],
            user=identity["user"],
            port=int(identity["port"]),
            fingerprint=identity["fingerprint"],
            workspace=identity["workspace"],
            profile_id=identity.get("profile_id", "wan22-14b-ab"),
            profile_sha256=identity["profile_sha256"],
        )
    return load_state(path)


def save_state_for_record(
    state_dir: Path | str, record: Mapping[str, Any], state: Mapping[str, Any]
) -> Path:
    identity = record["identity"]
    key = identity_key(
        host=identity["host"],
        user=identity["user"],
        port=int(identity["port"]),
        fingerprint=identity["fingerprint"],
        workspace=identity["workspace"],
        profile_sha256=identity["profile_sha256"],
    )
    if state.get("identity_key") != key:
        raise StateError("state belongs to a different target/profile")
    path = state_path(state_dir, key)
    write_state(path, state)
    return path


__all__ = [
    "PHASES",
    "RECORD_SCHEMA",
    "STATE_SCHEMA",
    "StateError",
    "default_record",
    "identity_key",
    "known_hosts_path",
    "list_records",
    "load_record",
    "load_state",
    "new_state",
    "now_iso",
    "record_path",
    "resolve_name",
    "run_dir",
    "save_record",
    "save_state_for_record",
    "state_for_record",
    "state_path",
    "transition",
    "update_stage",
    "validate_state",
    "write_state",
]
