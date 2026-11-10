"""Focused independent Clore supervisor child.

It validates one complete immutable session binding, writes a fresh ack and
heartbeat only after valid initialization, then runs the existing CloreSession
watch loop. It is intentionally separate from SSH/GUI/backend.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p1 import clore_session, v3_lifecycle


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("supervisor config must be an object")
    return value


def _atomic(path: Path, payload: dict) -> None:
    v3_lifecycle._atomic_write(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def validate_config(path: Path) -> dict:
    data = _read(path)
    required = {
        "schema",
        "run_id",
        "nonce",
        "state_path",
        "quote_path",
        "ack_path",
        "heartbeat_path",
        "session",
    }
    if set(data) != required:
        raise SystemExit("supervisor config has unknown/missing keys")
    if data["schema"] != "anime-video-studio/p1-v3-supervisor-config/v1":
        raise SystemExit("supervisor config schema is invalid")
    for key in ("run_id", "nonce", "state_path", "quote_path", "ack_path", "heartbeat_path"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise SystemExit(f"supervisor config {key} is invalid")
    binding = v3_lifecycle.SessionBinding.from_dict(data["session"])
    state = clore_session.load_state(Path(data["state_path"]), safety_mode=True)
    binding.assert_matches_state(state)
    return data


def _load_bridge(expected_server_id: int):
    from scripts.p1.clore_bridge import CloreBridge

    return CloreBridge(
        allow_live=True, expected_server_id=expected_server_id
    )


def _binding_sha256(binding) -> str:
    return hashlib.sha256(
        json.dumps(
            binding.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _ack_payload(data: dict, binding, status: str, config_sha256: str, binding_sha256: str) -> dict:
    return {
        "schema": "anime-video-studio/p1-v3-supervisor-ack/v1",
        "status": status,
        "run_id": data["run_id"],
        "nonce": data["nonce"],
        "pid": os.getpid(),
        "order_id": binding.order_id,
        "created_at": binding.created_at,
        "config_sha256": config_sha256,
        "binding_sha256": binding_sha256,
        "ack_at": time.time(),
    }


def _heartbeat_payload(data: dict, config_sha256: str, binding_sha256: str) -> dict:
    return {
        "schema": "anime-video-studio/p1-v3-supervisor-heartbeat/v1",
        "run_id": data["run_id"],
        "nonce": data["nonce"],
        "at": time.time(),
        "pid": os.getpid(),
        "config_sha256": config_sha256,
        "binding_sha256": binding_sha256,
    }


async def _watch(data: dict, *, config_sha256: str | None = None) -> int:
    state_path = Path(data["state_path"])
    ack_path = Path(data["ack_path"])
    heartbeat_path = Path(data["heartbeat_path"])
    binding = v3_lifecycle.SessionBinding.from_dict(data["session"])
    binding_sha256 = _binding_sha256(binding)
    if config_sha256 is None:
        config_path = data.get("config_path")
        if isinstance(config_path, str) and Path(config_path).is_file():
            config_sha256 = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        else:
            config_sha256 = ""
    state = clore_session.load_state(state_path, safety_mode=True)
    binding.assert_matches_state(state)
    bridge = _load_bridge(state.server_id)
    if hasattr(bridge, "expected_currency"):
        bridge.expected_currency = state.currency
    session = clore_session.CloreSession(state_path, bridge=bridge)
    first = await session.evaluate()
    if first.status != "ARMED":
        _atomic(
            ack_path,
            _ack_payload(data, binding, first.status, config_sha256, binding_sha256),
        )
        return 2
    _atomic(
        heartbeat_path,
        _heartbeat_payload(data, config_sha256, binding_sha256),
    )
    _atomic(
        ack_path,
        _ack_payload(data, binding, "ARMED", config_sha256, binding_sha256),
    )
    try:
        while True:
            _atomic(
                heartbeat_path,
                _heartbeat_payload(data, config_sha256, binding_sha256),
            )
            outcome = await session.evaluate()
            if outcome.status in {"ENDED_REPORTED", "LIVE_BLOCKED"}:
                # A terminal state must not leave an ARMED ack behind: any
                # later readback would otherwise accept activity.
                _atomic(
                    ack_path,
                    _ack_payload(
                        data,
                        binding,
                        outcome.status,
                        config_sha256,
                        binding_sha256,
                    ),
                )
                try:
                    heartbeat_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return 0 if outcome.status == "ENDED_REPORTED" else 2
            await asyncio.sleep(min(30.0, max(1.0, session.readback_poll_seconds)))
    finally:
        await bridge.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v3_supervisor_child.py")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.config)
        data = validate_config(config_path)
        if args.validate_config:
            return 0
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        return asyncio.run(_watch(data, config_sha256=config_sha256))
    except Exception as exc:
        print(f"supervisor child failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
