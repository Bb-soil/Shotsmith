from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.p1 import remote

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "p1" / "v3_remote_fetch_helper.py"


def _run_helper(*args: str, timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        capture_output=True,
        timeout=timeout,
    )


def _symlink(link: Path, target: str, *, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this host")


def test_helper_read_rejects_leaf_symlink_before_content_read(tmp_path: Path) -> None:
    sentinel = tmp_path / "outside.json"
    sentinel.write_text('{"secret": "sentinel-bytes"}', encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "status.json"
    _symlink(link, str(sentinel))
    result = _run_helper("read", str(link), "1024", str(root))
    assert result.returncode != 0
    assert b"sentinel-bytes" not in result.stdout
    assert sentinel.read_bytes() == b'{"secret": "sentinel-bytes"}'


def test_helper_read_rejects_parent_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "status.json"
    sentinel.write_text("outside-bytes", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "linked"
    _symlink(link, str(outside), directory=True)
    result = _run_helper("read", str(link / "status.json"), "1024", str(root))
    assert result.returncode != 0
    assert b"outside-bytes" not in result.stdout
    assert sentinel.read_text(encoding="utf-8") == "outside-bytes"


def test_helper_rejects_directory_and_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    assert _run_helper("read", str(root), "1024", str(root)).returncode != 0
    assert _run_helper("read", str(outside), "1024", str(root)).returncode != 0


def test_helper_read_and_stream_exact_bytes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "status.json"
    target.write_text('{"ok": true}', encoding="utf-8")
    read = _run_helper("read", str(target), "1024", str(root))
    assert read.returncode == 0, read.stderr
    envelope = json.loads(read.stdout)
    assert envelope["schema"] == "anime-video-studio/p1-v3-read/v1"
    assert envelope["content"] == '{"ok": true}'
    assert envelope["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()

    stream = _run_helper("stream", str(target), "6", str(root))
    assert stream.returncode == 0, stream.stderr
    header, payload = stream.stdout.split(b"\n", 1)
    parsed = json.loads(header)
    assert parsed["size"] == target.stat().st_size
    assert parsed["offset"] == 6
    assert payload == target.read_bytes()[6:]


def _receiver(dest: Path, root: Path, *, data: bytes, expected: str | None = None, mode: str = "noclobber"):
    return subprocess.run(
        [sys.executable, "-c", remote._REMOTE_RECEIVER, str(dest), str(root), expected or "", mode],
        input=data,
        capture_output=True,
        timeout=10,
    )


def test_receiver_noclobber_preserves_foreign_bytes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    dest = root / "helper.py"
    sentinel = b"foreign-sentinel"
    dest.write_bytes(sentinel)
    result = _receiver(dest, root, data=b"new", expected=hashlib.sha256(b"new").hexdigest())
    assert result.returncode != 0
    assert dest.read_bytes() == sentinel


def test_receiver_same_hash_is_idempotent_but_foreign_hash_refuses(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    dest = root / "helper.py"
    data = b"exact-helper-bytes"
    dest.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert _receiver(dest, root, data=data, expected=expected, mode="same").returncode == 0
    assert dest.read_bytes() == data
    result = _receiver(
        dest,
        root,
        data=b"other",
        expected=hashlib.sha256(b"other").hexdigest(),
        mode="same",
    )
    assert result.returncode != 0
    assert dest.read_bytes() == data


def test_receiver_rejects_symlink_leaf_and_parent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "helper.py"
    sentinel.write_bytes(b"outside-sentinel")
    link = root / "helper.py"
    _symlink(link, str(sentinel))
    result = _receiver(link, root, data=b"new", expected=hashlib.sha256(b"new").hexdigest())
    assert result.returncode != 0
    assert sentinel.read_bytes() == b"outside-sentinel"

    link.unlink()
    parent = root / "linked"
    _symlink(parent, str(outside), directory=True)
    result = _receiver(parent / "other.py", root, data=b"new", expected=hashlib.sha256(b"new").hexdigest())
    assert result.returncode != 0
    assert not (outside / "other.py").exists()
