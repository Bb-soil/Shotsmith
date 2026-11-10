#!/usr/bin/env python3
"""Production Comfy entry shim + frozen launcher + tiny fake checkout.

Proves the staged shim replaces sys.path[0] with the verified real checkout so
sibling modules and nested packages resolve exactly like ``python main.py``.
No PYTHONPATH injection is used; the child writes its origin evidence and birth
record before the fake main runs.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from scripts.p1.v3_bootstrap import comfy_launch_manifest
    from scripts.p1.v3_remote_launcher import launch

    root = Path(tempfile.mkdtemp(prefix="i1-comfy-shim-"))
    try:
        workspace = root / "workspace"
        comfy = workspace / "ComfyUI"
        comfy.mkdir(parents=True)
        _write(
            comfy / "sibling_probe.py",
            "ORIGIN = __file__\n",
        )
        _write(
            comfy / "nested_probe" / "__init__.py",
            "",
        )
        _write(
            comfy / "nested_probe" / "helper.py",
            "ORIGIN = __file__\n",
        )
        _write(
            comfy / "main.py",
            "import json,sys\n"
            "from pathlib import Path\n"
            "import sibling_probe\n"
            "from nested_probe import helper\n"
            "birth=Path(__file__).resolve().parent.parent/'comfy-process-proofs'\n"
            "origin={\n"
            " 'sys_path_0': sys.path[0],\n"
            " 'sibling_file': sibling_probe.ORIGIN,\n"
            " 'nested_file': helper.ORIGIN,\n"
            " 'main_file': __file__,\n"
            " 'sys_executable': sys.executable,\n"
            " 'sys_prefix': sys.prefix,\n"
            " 'birth_present': any(birth.glob(f'birth-{__import__(\"os\").getpid()}-*.json')),\n"
            "}\n"
            "Path(__file__).resolve().parent.parent.joinpath('comfy-origin.json').write_text(json.dumps(origin),encoding='utf-8')\n",
        )
        entry = workspace / ".avs-i1" / "comfy-entry"
        entry.mkdir(parents=True)
        shutil.copy2(repo / "scripts" / "i1" / "comfy_entry_shim.py", entry / "main.py")
        shutil.copy2(repo / "scripts" / "i1" / "child_entry.py", entry / "i1_child_entry.py")
        port = _free_port()
        manifest = comfy_launch_manifest(
            remote_root=str(workspace),
            workspace=str(workspace),
            run_id="shim-run",
            nonce="shim-nonce",
            remote_port=port,
        )
        venv_python = workspace / "p1-venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(sys.executable)
        manifest["python"] = str(venv_python)
        manifest["main"] = str(entry / "main.py")
        launch_path = workspace / "comfy-launch.json"
        launch_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        old_pythonpath = os.environ.pop("PYTHONPATH", None)
        try:
            result = launch(str(launch_path))
        finally:
            if old_pythonpath is not None:
                os.environ["PYTHONPATH"] = old_pythonpath
        origin_path = workspace / "comfy-origin.json"
        deadline = time.time() + 20.0
        while time.time() < deadline and not origin_path.is_file():
            time.sleep(0.05)
        if not origin_path.is_file():
            print(json.dumps({"status": "ORIGIN_NOT_WRITTEN", "launch": result}, ensure_ascii=False))
            return 1
        origin = json.loads(origin_path.read_text(encoding="utf-8"))
        expected_dir = str(comfy.resolve())
        checks = {
            "sys_path_0": origin.get("sys_path_0") == expected_dir,
            "sibling_origin": Path(origin.get("sibling_file", "")).resolve() == (comfy / "sibling_probe.py").resolve(),
            "nested_origin": Path(origin.get("nested_file", "")).resolve() == (comfy / "nested_probe" / "helper.py").resolve(),
            "main_origin": Path(origin.get("main_file", "")).resolve() == (comfy / "main.py").resolve(),
            "birth_before_work": bool(origin.get("birth_present")),
        }
        payload = {"status": "OFFLINE_VERIFIED" if all(checks.values()) else "ORIGIN_MISMATCH", "checks": checks, "origin": origin, "launch": result}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0 if all(checks.values()) else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
