from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.i1 import stock_nodes

ROOT = Path(__file__).resolve().parents[1]
STOCK_SOURCE = (
    ROOT
    / "docs"
    / "i1-evidence"
    / "codex-review-20261029"
    / "pinned-websocket_image_save.py"
)
STOCK_SHA256 = "3727e903705dba69924c390775a0d6aedfeb717400800e08793fcb77a8c331da"
GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is required for this fixture")


def _write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8", newline="\n")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert GIT is not None
    return subprocess.run(
        [GIT, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "ComfyUI"
    custom = repo / "custom_nodes"
    custom.mkdir(parents=True)
    (repo / "comfy").mkdir()
    _write(repo / "comfy" / "__init__.py", "")
    _write(repo / "nodes.py", "")
    _write(repo / "requirements.txt", "")
    shutil.copy2(STOCK_SOURCE, custom / "websocket_image_save.py")
    _write(custom / "example_node.py.example", "stock\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "stock fixture")
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, commit


def _validate(repo: Path, commit: str):
    return stock_nodes.run_local(repo, commit)


def _codes(data: dict) -> set[str]:
    return {str(item.get("code")) for item in data.get("violations", [])}


def _paths(data: dict) -> set[str]:
    return {str(item.get("path")) for item in data.get("violations", [])}


def _tree_snapshot(repo: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(repo.rglob("*")):
        rel = str(path.relative_to(repo)).replace("\\", "/")
        if path.is_symlink():
            snapshot[rel] = "link:" + os.readlink(path)
        elif path.is_file():
            snapshot[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            snapshot[rel] = "dir"
    return snapshot


def test_clean_pinned_tree_with_exact_public_stock_passes(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    stock = repo / "custom_nodes" / "websocket_image_save.py"
    assert hashlib.sha256(stock.read_bytes()).hexdigest() == STOCK_SHA256
    proc, data = _validate(repo, commit)
    assert proc.returncode == 0, proc.stderr
    assert data["ok"] is True
    assert data["object_format"] == "sha1"
    assert data["tracked"] == [
        "custom_nodes/example_node.py.example",
        "custom_nodes/websocket_image_save.py",
    ]
    assert data["violations"] == []
    assert data["cache_unverified"] == []
    assert data["cache_dirs"] == []


def test_untracked_node_example_metadata_and_empty_directory_block(
    tmp_path: Path,
) -> None:
    repo, commit = _repo(tmp_path)
    custom = repo / "custom_nodes"
    _write(custom / "evil.py", "raise SystemExit(1)\n")
    _write(custom / "evil.py.example", "example\n")
    _write(custom / ".gitkeep", "")
    _write(custom / "metadata.json", "{}\n")
    (custom / "empty_dir").mkdir()
    proc, data = _validate(repo, commit)
    assert proc.returncode != 0
    assert data["ok"] is False
    assert "untracked" in _codes(data)
    paths = _paths(data)
    for expected in (
        "custom_nodes/evil.py",
        "custom_nodes/evil.py.example",
        "custom_nodes/.gitkeep",
        "custom_nodes/metadata.json",
        "custom_nodes/empty_dir",
    ):
        assert expected in paths


def test_altered_tracked_stock_blocks_even_with_assume_unchanged(
    tmp_path: Path,
) -> None:
    repo, commit = _repo(tmp_path)
    stock = repo / "custom_nodes" / "websocket_image_save.py"
    original = stock.read_bytes()
    stock.write_bytes(original + b"# tampered\n")
    _git(
        repo,
        "update-index",
        "--assume-unchanged",
        "custom_nodes/websocket_image_save.py",
    )
    assert _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip() == ""
    proc, data = _validate(repo, commit)
    assert proc.returncode != 0
    assert data["ok"] is False
    assert "content" in _codes(data)
    assert "custom_nodes/websocket_image_save.py" in _paths(data)
    assert stock.read_bytes() == original + b"# tampered\n"
    _git(
        repo,
        "update-index",
        "--no-assume-unchanged",
        "custom_nodes/websocket_image_save.py",
    )
    stock.write_bytes(original)


def test_wrong_revision_blocks(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    proc, data = _validate(repo, "0" * 40)
    assert proc.returncode != 0
    assert data["ok"] is False
    assert "revision" in _codes(data)


def test_cache_for_tracked_source_is_allowed_but_unverified(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    cache = repo / "custom_nodes" / "__pycache__"
    cache.mkdir()
    pyc = cache / f"websocket_image_save.{sys.implementation.cache_tag}.pyc"
    pyc.write_bytes(b"fixture-bytecode")
    proc, data = _validate(repo, commit)
    assert proc.returncode == 0, proc.stderr
    assert data["ok"] is True
    assert "custom_nodes/__pycache__" in data["cache_dirs"]
    assert "custom_nodes/__pycache__/" + pyc.name in data["cache_unverified"]
    assert "custom_nodes/__pycache__/" + pyc.name not in data["tracked"]
    assert not any(item.get("code") == "untracked" for item in data["violations"])


def test_empty_cache_directory_is_allowed_and_reported(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    (repo / "custom_nodes" / "__pycache__").mkdir()
    proc, data = _validate(repo, commit)
    assert proc.returncode == 0, proc.stderr
    assert data["ok"] is True
    assert data["cache_dirs"] == ["custom_nodes/__pycache__"]
    assert data["cache_unverified"] == []


def test_unknown_cache_basename_and_orphan_pyc_block(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    cache = repo / "custom_nodes" / "__pycache__"
    cache.mkdir()
    _write(cache / "evil.cpython-312.pyc", b"evil")
    proc, data = _validate(repo, commit)
    assert proc.returncode != 0
    assert "untracked" in _codes(data)
    assert "custom_nodes/__pycache__/evil.cpython-312.pyc" in _paths(data)


def test_symlinked_stock_file_blocks_and_preserves_outside(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    stock = repo / "custom_nodes" / "websocket_image_save.py"
    outside = tmp_path / "outside-stock.py"
    outside.write_text("UNCHANGED\n", encoding="utf-8")
    stock.unlink()
    try:
        stock.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    proc, data = _validate(repo, commit)
    assert proc.returncode != 0
    assert "symlink" in _codes(data)
    assert outside.read_text(encoding="utf-8") == "UNCHANGED\n"


def test_symlinked_cache_file_blocks_and_preserves_outside(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    cache = repo / "custom_nodes" / "__pycache__"
    cache.mkdir()
    outside = tmp_path / "outside-cache"
    outside.write_bytes(b"UNCHANGED")
    pyc = cache / f"websocket_image_save.{sys.implementation.cache_tag}.pyc"
    try:
        pyc.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    proc, data = _validate(repo, commit)
    assert proc.returncode != 0
    assert "symlink" in _codes(data)
    assert outside.read_bytes() == b"UNCHANGED"


def test_validator_does_not_write_checkout(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    before = _tree_snapshot(repo)
    proc, data = _validate(repo, commit)
    assert proc.returncode == 0, proc.stderr
    assert data["ok"] is True
    assert _tree_snapshot(repo) == before
