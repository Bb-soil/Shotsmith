from __future__ import annotations

import pytest

from anime_studio.errors import InstanceAlreadyRunning
from anime_studio.lock import InstanceLock
from tests.helpers import ServerProcess, running_server


def test_lock_is_exclusive_and_reusable(tmp_path) -> None:
    path = tmp_path / "instance.lock"
    first = InstanceLock(path).acquire()
    try:
        with pytest.raises(InstanceAlreadyRunning):
            InstanceLock(path).acquire()
    finally:
        first.release()

    second = InstanceLock(path).acquire()
    second.release()


def test_second_server_fails_fast_on_same_db(tmp_path) -> None:
    with running_server(tmp_path) as first:
        second = ServerProcess(
            db_path=tmp_path / "p0.sqlite3",
            artifacts_dir=tmp_path / "artifacts",
            log_path=tmp_path / "second.log",
        )
        try:
            with pytest.raises(RuntimeError):
                second.start()
        finally:
            second.stop()
        assert "InstanceAlreadyRunning" in second.log_tail() or "second instance" in second.log_tail()
        assert first.client is not None
        assert first.client.get("/health").status_code == 200
