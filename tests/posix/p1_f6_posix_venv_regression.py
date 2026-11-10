#!/usr/bin/env python3
"""Independent offline POSIX regression for P1 F6 (venv preservation).

This file is deliberately stdlib-only so it can run on a real POSIX Python
(e.g. WSL ``python3``) even when pytest is not installed. It creates a real
temporary venv with ``--without-pip`` (nothing is downloaded or installed),
writes a venv-only module into that venv's site-packages, builds the actual
ComfyUI launch manifest, validates it through the real
``_validate_manifest`` and then executes the selected ``argv[0]``.

The assertions prove that the selected executable is the lexical
``p1-venv/bin/python`` path and that executing it still sees ``sys.prefix`` of
the venv and imports the venv-only module. They also prove that unsafe external
symlink targets and traversal are still refused.

No network, pip, SSH, GPU, cloud or repository mutation is performed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback

MARKER = "F6_VENV_ONLY_OK"
INSPECT = (
    "import json, sys\n"
    "try:\n"
    "    import p1_f6_venv_only as fixture\n"
    "    marker = fixture.MARKER\n"
    "except Exception as exc:  # pragma: no cover - diagnostic only\n"
    "    marker = 'IMPORT_FAILED:' + type(exc).__name__\n"
    "print(json.dumps({'prefix': sys.prefix, 'base_prefix': sys.base_prefix, "
    "'in_venv': sys.prefix != sys.base_prefix, 'executable': sys.executable, "
    "'module_marker': marker}))\n"
)


def load_launcher(repo: pathlib.Path):
    source = repo / "scripts" / "p1" / "v3_remote_launcher.py"
    spec = importlib.util.spec_from_file_location("f6_v3_remote_launcher", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def inspect(python: str) -> dict:
    output = subprocess.check_output([python, "-c", INSPECT], text=True)
    return json.loads(output)


def expect_launcher_error(launcher, callable_, label: str) -> str:
    try:
        callable_()
    except launcher.LauncherError as exc:
        return str(exc)
    raise AssertionError(f"{label}: expected LauncherError, but validation succeeded")


def build_manifest(root: pathlib.Path, python: str, comfy: pathlib.Path) -> dict:
    return {
        "schema": "anime-video-studio/p1-v3-launch/v1",
        "kind": "comfy",
        "run_id": "f6-posix-run",
        "nonce": "f6-posix-nonce",
        "root": str(root),
        "python": python,
        "comfy_dir": str(comfy),
        "main": str(comfy / "main.py"),
        "port": 8188,
        "log": str(root / "comfy.log"),
        "owner": str(root / "comfy-owner.json"),
        "ack": str(root / "comfy-ack.json"),
    }


def create_venv(path: pathlib.Path, marker: str) -> tuple[str, pathlib.Path]:
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    python = str(path / "bin" / "python")
    assert os.path.islink(python), "venv/bin/python is not a symlink"
    # Offline venv-only fixture: no pip install, no network, no copy of
    # third-party dependencies.
    purelib = pathlib.Path(
        subprocess.check_output(
            [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            text=True,
        ).strip()
    )
    assert purelib.is_dir(), f"venv site-packages missing: {purelib}"
    (purelib / "p1_f6_venv_only.py").write_text(
        f"MARKER = {marker!r}\n", encoding="utf-8"
    )
    return python, purelib


def write_manifest(root: pathlib.Path, python: str) -> tuple[dict, pathlib.Path]:
    comfy = root / "ComfyUI"
    comfy.mkdir(parents=True, exist_ok=True)
    (comfy / "main.py").write_text("print('placeholder')\n", encoding="utf-8")
    manifest = build_manifest(root, python, comfy)
    manifest_path = root / "comfy-launch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path


def run(repo: pathlib.Path) -> dict:
    launcher = load_launcher(repo)
    summary: dict = {"repo": str(repo), "python": sys.version, "checks": {}}

    with tempfile.TemporaryDirectory(prefix="p1-f6-posix-") as temp:
        work = pathlib.Path(temp)
        root = work / "owned"
        comfy = root / "ComfyUI"
        venv = root / "p1-venv"
        venv_python, _ = create_venv(venv, MARKER)
        resolved = os.path.realpath(venv_python)
        assert resolved != venv_python, "venv/bin/python did not resolve elsewhere"

        manifest, manifest_path = write_manifest(root, venv_python)
        checked, command_hash = launcher._validate_manifest(
            manifest, manifest_path=str(manifest_path)
        )
        selected = checked["argv"][0]
        direct = inspect(venv_python)
        through_launcher = inspect(selected)
        old_style = inspect(resolved)
        old_module = subprocess.run(
            [resolved, "-c", "import p1_f6_venv_only"],
            capture_output=True,
            text=True,
        )

        assertions = {
            "selected_equals_lexical_venv_python": selected == venv_python,
            "selected_is_symlink": os.path.islink(selected),
            "selected_is_absolute_inside_root": os.path.commonpath(
                [selected, str(root)]
            )
            == str(root),
            "direct_in_venv": bool(direct["in_venv"]),
            "selected_in_venv": bool(through_launcher["in_venv"]),
            "selected_prefix_equals_direct": through_launcher["prefix"]
            == direct["prefix"],
            "selected_base_prefix_equals_direct": through_launcher["base_prefix"]
            == direct["base_prefix"],
            "selected_prefix_equals_venv": pathlib.Path(through_launcher["prefix"])
            == venv,
            "venv_only_module_imported": through_launcher["module_marker"] == MARKER,
            "old_resolved_style_lost_venv": bool(direct["in_venv"])
            and not bool(old_style["in_venv"]),
            "old_resolved_style_cannot_import_venv_module": old_module.returncode != 0,
            "command_hash_is_sha256": isinstance(command_hash, str)
            and len(command_hash) == 64,
        }
        summary["checks"].update(assertions)
        failed = [name for name, ok in assertions.items() if not ok]
        assert not failed, f"failed assertions: {failed}"
        summary["selected"] = selected
        summary["resolved_old_style"] = resolved
        summary["direct"] = direct
        summary["through_launcher"] = through_launcher
        summary["old_style"] = old_style

        # F6 parent-link bypass: a foreign venv has its own venv-only module
        # and sentinel. The final python link points at an allowed base
        # interpreter, so only the real parent-chain check can reject it.
        foreign_venv = work / "foreign-venv"
        foreign_python, _ = create_venv(foreign_venv, "F6_FOREIGN_VENV")
        foreign_direct = inspect(foreign_python)
        foreign_sentinel = foreign_venv / "sentinel.txt"
        foreign_sentinel.write_bytes(b"FOREIGN_SENTINEL")
        summary["foreign_venv"] = foreign_direct
        summary["checks"]["external_venv_direct_prefix_is_foreign"] = (
            foreign_direct["prefix"] == str(foreign_venv)
            and foreign_direct["module_marker"] == "F6_FOREIGN_VENV"
        )
        assert summary["checks"]["external_venv_direct_prefix_is_foreign"]
        assert os.path.realpath(foreign_python) != foreign_python

        # Variant 1: owned/p1-venv is a symlink to the external venv.
        root_parent_link = work / "owned-parent-link"
        root_parent_link.mkdir()
        os.symlink(foreign_venv, root_parent_link / "p1-venv")
        parent_link_python = str(root_parent_link / "p1-venv" / "bin" / "python")
        assert os.path.islink(parent_link_python)
        assert os.path.realpath(parent_link_python) == os.path.realpath(foreign_python)
        parent_manifest, parent_manifest_path = write_manifest(
            root_parent_link, parent_link_python
        )
        parent_link_reason = expect_launcher_error(
            launcher,
            lambda: launcher._validate_manifest(
                parent_manifest, manifest_path=str(parent_manifest_path)
            ),
            "external venv parent link",
        )
        summary["checks"]["parent_venv_link_rejected"] = (
            "parent" in parent_link_reason
        )
        summary["checks"]["parent_venv_link_sentinel_unchanged"] = (
            foreign_sentinel.read_bytes() == b"FOREIGN_SENTINEL"
        )
        assert summary["checks"]["parent_venv_link_rejected"], parent_link_reason
        assert summary["checks"]["parent_venv_link_sentinel_unchanged"]

        # Variant 2: owned/p1-venv/bin is a symlink to the external venv bin.
        root_bin_link = work / "owned-bin-link"
        (root_bin_link / "p1-venv").mkdir(parents=True)
        os.symlink(foreign_venv / "bin", root_bin_link / "p1-venv" / "bin")
        bin_link_python = str(root_bin_link / "p1-venv" / "bin" / "python")
        assert os.path.islink(bin_link_python)
        assert os.path.realpath(bin_link_python) == os.path.realpath(foreign_python)
        bin_manifest, bin_manifest_path = write_manifest(
            root_bin_link, bin_link_python
        )
        bin_link_reason = expect_launcher_error(
            launcher,
            lambda: launcher._validate_manifest(
                bin_manifest, manifest_path=str(bin_manifest_path)
            ),
            "external venv bin link",
        )
        summary["checks"]["bin_venv_link_rejected"] = "parent" in bin_link_reason
        summary["checks"]["bin_venv_link_sentinel_unchanged"] = (
            foreign_sentinel.read_bytes() == b"FOREIGN_SENTINEL"
        )
        assert summary["checks"]["bin_venv_link_rejected"], bin_link_reason
        assert summary["checks"]["bin_venv_link_sentinel_unchanged"]

        # Variant 3: a symlinked parent that resolves inside the owned root
        # must still be rejected as a non-canonical parent chain.
        root_inside_link = work / "owned-inside-link"
        actual_venv = root_inside_link / "actual-venv"
        (actual_venv / "bin").mkdir(parents=True)
        os.symlink(sys.executable, actual_venv / "bin" / "python")
        inside_sentinel = actual_venv / "sentinel.txt"
        inside_sentinel.write_bytes(b"INSIDE_SENTINEL")
        os.symlink(actual_venv, root_inside_link / "p1-venv")
        inside_python = str(root_inside_link / "p1-venv" / "bin" / "python")
        assert os.path.islink(inside_python)
        inside_manifest, inside_manifest_path = write_manifest(
            root_inside_link, inside_python
        )
        inside_reason = expect_launcher_error(
            launcher,
            lambda: launcher._validate_manifest(
                inside_manifest, manifest_path=str(inside_manifest_path)
            ),
            "inside-root parent symlink",
        )
        summary["checks"]["inside_parent_symlink_rejected"] = (
            "symlink or junction" in inside_reason
        )
        summary["checks"]["inside_parent_sentinel_unchanged"] = (
            inside_sentinel.read_bytes() == b"INSIDE_SENTINEL"
        )
        assert summary["checks"]["inside_parent_symlink_rejected"], inside_reason
        assert summary["checks"]["inside_parent_sentinel_unchanged"]

        # Unsafe external final symlink target must still be rejected.
        outside = work / "outside"
        outside.mkdir()
        foreign = outside / "python3.12"
        foreign.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        foreign.chmod(0o755)
        unsafe_link = venv / "bin" / "python-unsafe"
        os.symlink(foreign, unsafe_link)
        unsafe_manifest = build_manifest(root, str(unsafe_link), comfy)
        unsafe_reason = expect_launcher_error(
            launcher,
            lambda: launcher._validate_manifest(
                unsafe_manifest, manifest_path=str(manifest_path)
            ),
            "external python symlink target",
        )
        summary["checks"]["external_symlink_target_rejected"] = (
            "outside the expected interpreter roots" in unsafe_reason
        )
        summary["checks"]["external_target_untouched"] = (
            foreign.read_text(encoding="utf-8") == "#!/bin/sh\nexit 0\n"
        )
        assert summary["checks"]["external_symlink_target_rejected"], unsafe_reason
        assert summary["checks"]["external_target_untouched"]

        # Traversal in the interpreter path must still be rejected.
        traversal = f"{venv}/../p1-venv/bin/python"
        traversal_manifest = build_manifest(root, traversal, comfy)
        traversal_reason = expect_launcher_error(
            launcher,
            lambda: launcher._validate_manifest(
                traversal_manifest, manifest_path=str(manifest_path)
            ),
            "python traversal",
        )
        summary["checks"]["traversal_rejected"] = "traversal" in traversal_reason
        assert summary["checks"]["traversal_rejected"], traversal_reason

        # A non-final symlink component must still be rejected (main.py path).
        linked = root / "linked-comfy"
        os.symlink(outside, linked)
        linked_manifest = build_manifest(root, venv_python, comfy)
        linked_manifest["main"] = str(linked / "main.py")
        linked_reason = expect_launcher_error(
            launcher,
            lambda: launcher._validate_manifest(
                linked_manifest, manifest_path=str(manifest_path)
            ),
            "main symlink component",
        )
        summary["checks"]["parent_symlink_rejected"] = (
            "symlink or junction" in linked_reason
        )
        assert summary["checks"]["parent_symlink_rejected"], linked_reason

        # A non-final symlink path is still rejected outside the one allowed
        # Python final-symlink case.
        summary["checks"]["non_python_symlink_not_allowed"] = (
            "must not be a symlink"
            in expect_launcher_error(
                launcher,
                lambda: launcher._validate_path(
                    str(root), str(unsafe_link), "log", allow_missing=True
                ),
                "log symlink",
            )
        )
        assert summary["checks"]["non_python_symlink_not_allowed"]

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    try:
        summary = run(pathlib.Path(args.repo).resolve())
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
