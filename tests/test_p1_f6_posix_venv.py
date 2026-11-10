"""F6 regressions: keep the owned venv interpreter path on POSIX.

The real POSIX proof lives in the stdlib-only
``tests/posix/p1_f6_posix_venv_regression.py`` so it can run under WSL/local
POSIX even when pytest is unavailable. This file is only a pytest wrapper plus
Windows-specific regular-executable checks; it never mocks ``os.name``.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from scripts.p1 import v3_remote_launcher

ROOT = pathlib.Path(__file__).resolve().parents[1]
POSIX_SCRIPT = ROOT / "tests" / "posix" / "p1_f6_posix_venv_regression.py"


def _windows_to_wsl(path: pathlib.Path) -> str:
    drive, rest = os.path.splitdrive(str(path))
    if not drive:
        return str(path).replace("\\", "/")
    return "/mnt/" + drive.rstrip(":").lower() + "/" + rest.replace("\\", "/").lstrip("/")


def _posix_python(repo: pathlib.Path) -> tuple[list[str], list[str]] | None:
    """Return (argv prefix, [script, repo] args) for a real POSIX Python."""
    if os.name != "nt":
        return [sys.executable], [str(POSIX_SCRIPT), "--repo", str(repo)]
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return None
    try:
        probe = subprocess.run(
            [wsl, "-e", "python3", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    return (
        [wsl, "-e", "python3"],
        [_windows_to_wsl(POSIX_SCRIPT), "--repo", _windows_to_wsl(repo)],
    )


def test_posix_temp_venv_is_preserved_through_validated_argv() -> None:
    runner = _posix_python(ROOT)
    if runner is None:
        pytest.skip("no local POSIX Python (WSL or native) available")
    prefix, args = runner
    result = subprocess.run(
        [*prefix, *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    checks = payload["checks"]
    assert checks["selected_equals_lexical_venv_python"] is True
    assert checks["selected_is_symlink"] is True
    assert checks["selected_in_venv"] is True
    assert checks["selected_prefix_equals_venv"] is True
    assert checks["venv_only_module_imported"] is True
    assert checks["old_resolved_style_lost_venv"] is True
    assert checks["external_symlink_target_rejected"] is True
    assert checks["traversal_rejected"] is True
    assert checks["parent_symlink_rejected"] is True
    assert checks["external_venv_direct_prefix_is_foreign"] is True
    assert checks["parent_venv_link_rejected"] is True
    assert checks["parent_venv_link_sentinel_unchanged"] is True
    assert checks["bin_venv_link_rejected"] is True
    assert checks["bin_venv_link_sentinel_unchanged"] is True
    assert checks["inside_parent_symlink_rejected"] is True
    assert checks["inside_parent_sentinel_unchanged"] is True


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _kill_pid(pid: int) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except OSError:
        pass


def _comfy_manifest(root: pathlib.Path, comfy: pathlib.Path, python: str) -> dict:
    return {
        "schema": "anime-video-studio/p1-v3-launch/v1",
        "kind": "comfy",
        "run_id": "f6-windows-run",
        "nonce": "f6-windows-nonce",
        "root": str(root),
        "python": python,
        "comfy_dir": str(comfy),
        "main": str(comfy / "main.py"),
        "port": 8188,
        "log": str(root / "comfy.log"),
        "owner": str(root / "comfy-owner.json"),
        "ack": str(root / "comfy-ack.json"),
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows regular-executable path")
def test_windows_regular_python_executable_is_accepted(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "owned"
    comfy = root / "ComfyUI"
    bin_dir = root / "p1-venv" / "bin"
    comfy.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    (comfy / "main.py").write_text("print('placeholder')\n", encoding="utf-8")
    python = bin_dir / "python.exe"
    shutil.copy2(sys.executable, python)
    assert python.is_file() and not os.path.islink(python)

    manifest = _comfy_manifest(root, comfy, str(python))
    manifest_path = root / "comfy-launch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checked, _ = v3_remote_launcher._validate_manifest(
        manifest, manifest_path=str(manifest_path)
    )
    assert checked["argv"][0] == str(python)
    assert checked["argv"][1] == str(comfy / "main.py")


@pytest.mark.skipif(os.name != "nt", reason="Windows regular-executable path")
def test_windows_regular_python_executable_starts_and_registers_owner(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "owned"
    comfy = root / "ComfyUI"
    bin_dir = root / "p1-venv" / "bin"
    comfy.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    (comfy / "main.py").write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
    python = bin_dir / "python.exe"
    shutil.copy2(sys.executable, python)
    assert not os.path.islink(python)

    manifest = _comfy_manifest(root, comfy, str(python))
    manifest["port"] = _free_port()
    manifest_path = root / "comfy-launch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "p1" / "v3_remote_launcher.py"),
        "--launch",
        str(manifest_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "STARTED"
    pid = payload["pid"]
    try:
        owner = json.loads((root / "comfy-owner.json").read_text(encoding="utf-8"))
        ack = json.loads((root / "comfy-ack.json").read_text(encoding="utf-8"))
        assert owner["pid"] == pid == ack["pid"]
        assert (
            owner["command_sha256"]
            == ack["command_sha256"]
            == payload["command_sha256"]
        )
        second = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert second.returncode == 0, second.stdout + second.stderr
        attached = json.loads(second.stdout)
        assert attached["status"] == "ATTACHED"
        assert attached["pid"] == pid
        assert json.loads((root / "comfy-owner.json").read_text())["pid"] == pid
    finally:
        _kill_pid(pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction path")
def test_windows_parent_junction_is_rejected(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "owned"
    comfy = root / "ComfyUI"
    comfy.mkdir(parents=True)
    (comfy / "main.py").write_text("print('placeholder')\n", encoding="utf-8")
    outside = tmp_path / "outside-venv"
    (outside / "bin").mkdir(parents=True)
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"JUNCTION_SENTINEL")
    try:
        os.symlink(sys.executable, outside / "bin" / "python.exe")
    except (OSError, NotImplementedError):
        pytest.skip("Windows symlink creation not permitted")

    junction = root / "p1-venv"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("Windows junction creation not permitted")
    try:
        manifest = _comfy_manifest(root, comfy, str(junction / "bin" / "python.exe"))
        manifest_path = root / "comfy-launch.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(v3_remote_launcher.LauncherError) as excinfo:
            v3_remote_launcher._validate_manifest(
                manifest, manifest_path=str(manifest_path)
            )
        reason = str(excinfo.value)
        assert "parent" in reason or "symlink or junction" in reason
        assert sentinel.read_bytes() == b"JUNCTION_SENTINEL"
    finally:
        try:
            os.rmdir(junction)
        except OSError:
            pass


def test_traversal_in_python_path_is_rejected(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "owned"
    comfy = root / "ComfyUI"
    bin_dir = root / "p1-venv" / "bin"
    comfy.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    (comfy / "main.py").write_text("print('placeholder')\n", encoding="utf-8")
    traversal = os.path.join(str(root), "p1-venv", "..", "p1-venv", "bin", "python.exe")
    manifest = _comfy_manifest(root, comfy, traversal)
    manifest_path = root / "comfy-launch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(v3_remote_launcher.LauncherError) as excinfo:
        v3_remote_launcher._validate_manifest(
            manifest, manifest_path=str(manifest_path)
        )
    assert "traversal" in str(excinfo.value)


def test_python_path_outside_owned_root_is_rejected(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "owned"
    comfy = root / "ComfyUI"
    comfy.mkdir(parents=True)
    (comfy / "main.py").write_text("print('placeholder')\n", encoding="utf-8")
    outside = tmp_path / "outside-python.exe"
    outside.write_bytes(b"not-a-real-interpreter")
    manifest = _comfy_manifest(root, comfy, str(outside))
    manifest_path = root / "comfy-launch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(v3_remote_launcher.LauncherError) as excinfo:
        v3_remote_launcher._validate_manifest(
            manifest, manifest_path=str(manifest_path)
        )
    assert "escapes the owned root" in str(excinfo.value)
