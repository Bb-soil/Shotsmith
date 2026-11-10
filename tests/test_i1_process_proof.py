from __future__ import annotations

import pytest

from scripts.i1 import remote
from scripts.i1.errors import RemoteFailure


PROOF = {
    "schema": "anime-video-studio/i1-process-proof/v1",
    "run_id": "r",
    "nonce": "n",
    "command_sha256": "a" * 64,
    "pid": 123,
    "start_ticks": 777,
    "exe": "/usr/bin/python3",
    "cmdline_sha256": "b" * 64,
    "boot_id": "boot",
}


def test_process_proof_accepts_same_starttime_and_rejects_pid_reuse(monkeypatch) -> None:
    current = {key: PROOF[key] for key in ("pid", "start_ticks", "exe", "cmdline_sha256", "boot_id")}
    monkeypatch.setattr(remote, "capture_process", lambda *a, **k: dict(current))
    remote.verify_process_proof(
        PROOF,
        runner=object(),
        identity=object(),
        run_id="r",
        nonce="n",
        command_sha256="a" * 64,
    )
    current["start_ticks"] = 778
    with pytest.raises(RemoteFailure):
        remote.verify_process_proof(
            PROOF,
            runner=object(),
            identity=object(),
            run_id="r",
            nonce="n",
            command_sha256="a" * 64,
        )
