#!/usr/bin/env python3
"""Local POSIX harness for I1: tiny fixtures, real venv, no external network.

Run under WSL:
  python3 tests/posix/i1_posix_harness.py --repo /mnt/c/.../anime-video-studio

It exercises the actual setup script and the actual I1 probe with a fake
ComfyUI checkout, fake torch/nvidia-smi and tiny model bytes.  It never
downloads models, installs external packages or contacts a cloud service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

STOCK_SOURCE = (
    "docs/i1-evidence/codex-review-20261029/pinned-websocket_image_save.py"
)
STOCK_SHA256 = "3727e903705dba69924c390775a0d6aedfeb717400800e08793fcb77a8c331da"




class LocalRunner:
    """Local implementation of the production put/run interface for the probe."""

    def __init__(self, *, root: Path, env: dict[str, str]) -> None:
        self.root = root
        self.env = env
        self.commands: list[tuple[str, ...]] = []

    def put_file(self, identity, remote_path, local_path, **kwargs):
        target = Path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local_path).read_bytes())

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout):
        return json.loads(Path(remote_path).read_text(encoding="utf-8"))

    def run_remote(self, identity, argv, timeout=None, **kwargs):
        args = list(argv)
        if args and args[0] == "python3":
            args[0] = sys.executable
        self.commands.append(tuple(args))
        result = subprocess.run(
            args,
            cwd=str(self.root),
            capture_output=True,
            text=True,
            timeout=timeout or 120.0,
            env=self.env,
        )
        return SimpleNamespace(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _make_git(path: Path) -> None:
    _write(
        path,
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "if [ \"$1\" = \"clone\" ]; then\n"
        "  dest=\"${@: -1}\"\n"
        "  mkdir -p \"$dest/.git\" \"$dest/custom_nodes\"\n"
        "  : > \"$dest/requirements.txt\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"-C\" ]; then\n"
        "  dir=\"$2\"; shift 2\n"
        "  if [ \"$1\" = \"rev-parse\" ] && [ \"$2\" = \"--show-toplevel\" ]; then cd \"$dir\" && pwd; exit 0; fi\n"
        "  if [ \"$1\" = \"rev-parse\" ] && [ \"$2\" = \"HEAD\" ]; then echo \"${I1_FAKE_COMMIT:?}\"; exit 0; fi\n"
        "  if [ \"$1\" = \"status\" ]; then exit 0; fi\n"
        "  if [ \"$1\" = \"checkout\" ]; then exit 0; fi\n"
        "fi\n"
        "if [ \"$1\" = \"--version\" ]; then echo 'git version 0'; exit 0; fi\n"
        "exit 0\n",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_git(real_git: str, repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [real_git, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _make_real_tree_git(path: Path, *, real_git: str, pinned: str) -> None:
    """Test-only git: pinned revision alias, real tree/status/index operations."""

    script = f"""#!/usr/bin/env bash
set -euo pipefail
real={shlex.quote(real_git)}
if [ "${{1:-}}" = "--version" ]; then echo 'git version fixture'; exit 0; fi
if [ "${{1:-}}" = "-C" ]; then
  dir="$2"; shift 2
  if [ "${{1:-}}" = "rev-parse" ] && {{ [ "${{2:-}}" = "HEAD" ] || [ "${{2:-}}" = "--verify" ]; }}; then
    echo "${{I1_FAKE_COMMIT:-{pinned}}}"; exit 0
  fi
  if [ "${{1:-}}" = "rev-parse" ] && [ "${{2:-}}" = "--show-toplevel" ]; then
    exec "$real" -C "$dir" rev-parse --show-toplevel
  fi
  if [ "${{1:-}}" = "status" ]; then
    exec "$real" -C "$dir" status "${{@:2}}"
  fi
  if [ "${{1:-}}" = "ls-tree" ]; then
    shift
    out=()
    replaced=0
    for a in "$@"; do
      if [ "$replaced" = 0 ] && [[ "$a" != -* ]] && [ "$a" != "custom_nodes" ] && [ "$a" != "--" ]; then
        out+=("HEAD"); replaced=1
      else
        out+=("$a")
      fi
    done
    exec "$real" -C "$dir" ls-tree "${{out[@]}}"
  fi
  exec "$real" -C "$dir" "$@"
fi
exec "$real" "$@"
"""
    _write(path, script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_nvidia_smi(path: Path) -> None:
    _write(path, "#!/usr/bin/env bash\necho 'Fake GPU, 24564, 999.0'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _snapshot(paths) -> dict[str, str]:
    return {str(path): _sha(path) for path in paths if path.is_file()}


def _make_fake_venv(venv: Path) -> None:
    """Create a real pyvenv.cfg + symlinked interpreter without ensurepip.

    The Ubuntu image in this environment has no python3-venv/ensurepip package.
    This fixture still gives Python a genuine venv prefix and lexical bin/python;
    pip is a tiny stdlib-only fixture so no external install is attempted.
    """

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = venv / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    (venv / "bin").mkdir()
    _write(
        venv / "pyvenv.cfg",
        f"home = {Path(sys.executable).parent}\n"
        "include-system-site-packages = true\n"
        f"version = {sys.version.split()[0]}\n",
    )
    (venv / "bin" / "python").symlink_to(sys.executable)
    (venv / "bin" / "python3").symlink_to(sys.executable)
    _write(
        site / "pip" / "__init__.py",
        "__version__ = 'fixture'\n",
    )
    _write(
        site / "pip" / "__main__.py",
        "import os\n"
        "import sys\n"
        "a=sys.argv[1:]\n"
        "if a[:1]==['check']:\n"
        "    print('No broken requirements found.')\n"
        "    raise SystemExit(0)\n"
        "if a[:1]==['freeze']:\n"
        "    print('torch==2.9.1')\n"
        "    raise SystemExit(0)\n"
        "if a[:1]==['install']:\n"
        "    p=os.environ.get('I1_PIP_SENTINEL')\n"
        "    if p:\n"
        "        open(p,'w').write('installed')\n"
        "    print('fixture pip install (no-op)')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(0)\n",
    )
    _fake_torch_at(site)


def _fake_torch_at(site: Path) -> None:
    torch = site / "torch"
    _write(
        torch / "__init__.py",
        "__version__ = '2.9.1'\n"
        "class _Version:\n"
        "    cuda = '12.8'\n"
        "version = _Version()\n"
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "cuda = _Cuda()\n",
    )
    _write(site / "torch-2.9.1.dist-info" / "METADATA", "Metadata-Version: 2.1\nName: torch\nVersion: 2.9.1\n")


def _object_info(profile: dict, repo: Path) -> dict:
    fixture = json.loads((repo / "tests" / "fixtures" / "p1" / "savevideo-object_info-7a283320.json").read_text(encoding="utf-8"))
    save = fixture["SaveVideo"]
    data = {"SaveVideo": save}
    for name in profile["required_classes"]:
        if name == "SaveVideo":
            continue
        required = {}
        for input_name in profile["required_inputs"].get(name, []):
            required[input_name] = ["*", {}]
        data[name] = {"input": {"required": required}, "output": ["*", "*", "*"]}
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    root = Path(tempfile.mkdtemp(prefix="i1-posix-"))
    try:
        real_git = shutil.which("git")
        if not real_git:
            print(json.dumps({"status": "GIT_MISSING"}, ensure_ascii=False))
            return 1
        bin_dir = root / "bin"
        bin_dir.mkdir()
        _make_nvidia_smi(bin_dir / "nvidia-smi")
        env = dict(os.environ)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env["PIP_NO_INDEX"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        profile_data = json.loads((repo / "config" / "i1" / "wan22-14b-ab.profile.json").read_text(encoding="utf-8"))
        pinned = profile_data["comfyui"]["commit"]
        profile_data["profile_id"] = "posix-tiny"
        payloads = {}
        for index, item in enumerate(profile_data["models"]):
            data = f"posix-tiny-{index}".encode()
            payloads[item["target"]] = data
            item["size_bytes"] = len(data)
            item["sha256"] = hashlib.sha256(data).hexdigest()
            item["url"] = f"https://example.invalid/{index}.bin"
        profile_data["disk"]["install_overhead_bytes"] = 0
        profile_data["disk"]["reserve_bytes"] = 0

        workspace = root / "workspace"
        owned = workspace / ".avs-i1"
        comfy = workspace / "ComfyUI"
        owned.mkdir(parents=True)
        comfy.mkdir(parents=True)
        (comfy / "comfy").mkdir()
        _write(comfy / "comfy" / "__init__.py", "")
        _write(comfy / "nodes.py", "")
        _write(comfy / "requirements.txt", "")
        (comfy / "custom_nodes").mkdir()
        (comfy / "custom_nodes" / "example_node.py.example").write_text("stock\n")
        shutil.copy2(repo / STOCK_SOURCE, comfy / "custom_nodes" / "websocket_image_save.py")
        _run_git(real_git, comfy, "init", "-q")
        _run_git(real_git, comfy, "config", "user.email", "fixture@example.invalid")
        _run_git(real_git, comfy, "config", "user.name", "fixture")
        _run_git(real_git, comfy, "config", "core.autocrlf", "false")
        _run_git(real_git, comfy, "add", "-A")
        _run_git(real_git, comfy, "commit", "-qm", "pinned stock fixture")
        fixture_commit = _run_git(real_git, comfy, "rev-parse", "HEAD").stdout.strip()
        _make_real_tree_git(bin_dir / "git", real_git=real_git, pinned=pinned)
        env["I1_FAKE_COMMIT"] = pinned
        setup_env = dict(env)
        sentinel = root / "pip-sentinel"
        setup_env["I1_PIP_SENTINEL"] = str(sentinel)

        for target, data in payloads.items():
            path = comfy / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        attempt_dir = owned / "attempts" / "0001"
        attempt_dir.mkdir(parents=True)
        profile_path = attempt_dir / "profile.json"
        _write(profile_path, json.dumps(profile_data, indent=2, sort_keys=True) + "\n")
        from scripts.i1.profile import validate_profile  # type: ignore

        profile = validate_profile(profile_data)
        api_source = repo / "workflows" / "P1" / "wan22-14b-shot02-alley-v1.api.json"
        api_target = owned / "workflow" / "posix-tiny.api.json"
        api_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(api_source, api_target)
        object_info_path = workspace / "p1-evidence" / "object_info.json"
        system_stats_path = workspace / "p1-evidence" / "system_stats.json"
        _write(object_info_path, json.dumps(_object_info(profile_data, repo), indent=2))
        _write(system_stats_path, json.dumps({"devices": [{"name": "Fake GPU", "vram_total": 1}], "system": {"comfyui_version": "fake"}}))

        venv = workspace / "p1-venv"
        _make_fake_venv(venv)
        torch_init = next((venv / "lib").glob("*/site-packages/torch/__init__.py"))
        for name in ("probe.py", "util.py", "profile.py"):
            source = repo / "scripts" / "i1" / name
            target = owned / ("i1_probe.py" if name == "probe.py" else "i1_" + name)
            shutil.copy2(source, target)
        shutil.copy2(repo / "scripts" / "p1" / "preflight.py", owned / "preflight.py")
        shutil.copy2(repo / "scripts" / "p1" / "setup_plan.py", owned / "setup_plan.py")

        from scripts.i1.setup import render_setup  # type: ignore

        _write(owned / "setup.sh", render_setup(profile, workspace=str(workspace)))

        def run_setup(env_override=None):
            return subprocess.run(
                ["bash", "setup.sh"],
                cwd=str(owned),
                env=env_override if env_override is not None else setup_env,
                capture_output=True,
                text=True,
                timeout=120,
            )

        setup = run_setup()
        if setup.returncode != 0:
            print(json.dumps({"status": "SETUP_FAILED", "stdout": setup.stdout, "stderr": setup.stderr}, ensure_ascii=False))
            return 1
        if not sentinel.is_file():
            print(json.dumps({"status": "PIP_NOT_REACHED_ON_CLEAN_SETUP"}, ensure_ascii=False))
            return 1

        protected = [comfy / item["target"] for item in profile_data["models"]]
        protected += [venv / "pyvenv.cfg", torch_init]
        before = _snapshot(protected)
        cache_dir = comfy / "custom_nodes" / "__pycache__"
        cache_dir.mkdir()
        cache_file = cache_dir / f"websocket_image_save.{sys.implementation.cache_tag}.pyc"
        py_compile.compile(
            str(comfy / "custom_nodes" / "websocket_image_save.py"),
            cfile=str(cache_file),
            doraise=True,
        )

        setup2 = run_setup()
        if setup2.returncode != 0:
            print(json.dumps({"status": "SETUP_RESTART_FAILED", "stdout": setup2.stdout, "stderr": setup2.stderr}, ensure_ascii=False))
            return 1
        if "custom_nodes: runtime cache" not in setup2.stdout:
            print(json.dumps({"status": "CACHE_NOTE_MISSING", "stdout": setup2.stdout, "stderr": setup2.stderr}, ensure_ascii=False))
            return 1
        if _snapshot(protected) != before:
            print(json.dumps({"status": "RESTART_CHANGED_PROTECTED_FILES"}, ensure_ascii=False))
            return 1

        from scripts.i1.inspect import inspect_remote  # type: ignore

        inspection = inspect_remote(
            LocalRunner(root=owned, env={**setup_env, "I1_FAKE_COMMIT": pinned}),
            object(),
            workspace=str(workspace),
            profile=profile,
        )
        if not inspection.get("compatible"):
            print(json.dumps({"status": "INSPECT_STOCK_FAILED", "inspection": inspection}, ensure_ascii=False))
            return 1
        if not any(item.get("code") == "custom_nodes_cache" for item in inspection.get("warnings", [])):
            print(json.dumps({"status": "INSPECT_CACHE_WARNING_MISSING", "inspection": inspection}, ensure_ascii=False))
            return 1
        if not inspection.get("facts", {}).get("custom_nodes", {}).get("ok"):
            print(json.dumps({"status": "INSPECT_VALIDATOR_NOT_OK", "inspection": inspection}, ensure_ascii=False))
            return 1

        from scripts.i1 import stock_nodes  # type: ignore

        nodes_runner = LocalRunner(root=owned, env={**setup_env, "I1_FAKE_COMMIT": pinned})

        def validate_nodes_now():
            result = nodes_runner.run_remote(
                object(), stock_nodes.remote_argv(str(comfy), pinned)
            )
            return stock_nodes.parse_output(result.stdout), result

        def has_symlink_violation(data):
            return any(item.get("code") == "symlink" for item in data.get("violations", []))

        dotgit = comfy / ".git"
        dotgit_real = comfy / ".git-real"
        dotgit.rename(dotgit_real)
        try:
            dotgit.symlink_to(dotgit_real)
            data, result = validate_nodes_now()
            if result.returncode == 0 or not has_symlink_violation(data):
                print(json.dumps({"status": "GIT_SYMLINK_NOT_REJECTED", "data": data}, ensure_ascii=False))
                return 1
        finally:
            if dotgit.is_symlink():
                dotgit.unlink()
            if dotgit_real.exists() and not dotgit.exists():
                dotgit_real.rename(dotgit)

        custom_nodes_dir = comfy / "custom_nodes"
        custom_nodes_real = comfy / "custom_nodes-real"
        custom_nodes_dir.rename(custom_nodes_real)
        try:
            custom_nodes_dir.symlink_to(custom_nodes_real)
            data, result = validate_nodes_now()
            if result.returncode == 0 or not has_symlink_violation(data):
                print(json.dumps({"status": "CUSTOM_NODES_DIR_SYMLINK_NOT_REJECTED", "data": data}, ensure_ascii=False))
                return 1
        finally:
            if custom_nodes_dir.is_symlink():
                custom_nodes_dir.unlink()
            if custom_nodes_real.exists() and not custom_nodes_dir.exists():
                custom_nodes_real.rename(custom_nodes_dir)

        negative_cases = []

        def negative(name, *, env_override=None, expect=None):
            sentinel.unlink(missing_ok=True)
            snapshot = _snapshot(protected)
            proc = run_setup(env_override)
            combined = proc.stdout + proc.stderr
            ok = (
                proc.returncode != 0
                and "SETUP_COMPLETE" not in proc.stdout
                and not sentinel.exists()
                and _snapshot(protected) == snapshot
                and (expect is None or expect in combined)
            )
            if ok:
                negative_cases.append(name)
            if not ok:
                print(
                    json.dumps(
                        {
                            "status": "NEGATIVE_CASE_FAILED",
                            "name": name,
                            "returncode": proc.returncode,
                            "stdout": proc.stdout[-1000:],
                            "stderr": proc.stderr[-1000:],
                        },
                        ensure_ascii=False,
                    )
                )
            return ok

        stock = comfy / "custom_nodes" / "websocket_image_save.py"
        stock_original = stock.read_bytes()

        evil = comfy / "custom_nodes" / "evil.py"
        _write(evil, "raise SystemExit(1)\n")
        if not negative("unknown_node", expect="custom_nodes"):
            return 1
        evil.unlink()

        evil_example = comfy / "custom_nodes" / "evil.py.example"
        _write(evil_example, "example\n")
        if not negative("untracked_example", expect="custom_nodes"):
            return 1
        evil_example.unlink()

        stock.write_bytes(stock_original + b"# tampered\n")
        if not negative("changed_stock_dirty", expect="modified tracked files"):
            return 1
        stock.write_bytes(stock_original)

        _run_git(real_git, comfy, "update-index", "--assume-unchanged", "custom_nodes/websocket_image_save.py")
        stock.write_bytes(stock_original + b"# hidden\n")
        if _run_git(real_git, comfy, "status", "--porcelain", "--untracked-files=no").stdout.strip():
            print(json.dumps({"status": "ASSUME_UNCHANGED_NOT_HIDDEN"}, ensure_ascii=False))
            return 1
        if not negative("changed_stock_assume_unchanged", expect="custom_nodes"):
            return 1
        _run_git(real_git, comfy, "update-index", "--no-assume-unchanged", "custom_nodes/websocket_image_save.py")
        stock.write_bytes(stock_original)

        outside = root / "outside-sentinel"
        outside.write_text("UNCHANGED")
        link = comfy / "custom_nodes" / "evil.py"
        link.symlink_to(outside)
        if not negative("symlink_node", expect="symlink"):
            return 1
        if outside.read_text() != "UNCHANGED":
            print(json.dumps({"status": "SYMLINK_OUTSIDE_CHANGED"}, ensure_ascii=False))
            return 1
        link.unlink()

        cache_file.unlink(missing_ok=True)
        cache_file.symlink_to(outside)
        if not negative("symlink_cache", expect="symlink"):
            return 1
        if outside.read_text() != "UNCHANGED":
            print(json.dumps({"status": "CACHE_SYMLINK_OUTSIDE_CHANGED"}, ensure_ascii=False))
            return 1
        shutil.rmtree(cache_dir, ignore_errors=True)

        wrong_env = dict(setup_env)
        wrong_env["I1_FAKE_COMMIT"] = "0" * 40
        if not negative("wrong_commit", env_override=wrong_env, expect="commit"):
            return 1

        nodes = comfy / "nodes.py"
        nodes_original = nodes.read_bytes()
        nodes.write_text("dirty\n", encoding="utf-8", newline="\n")
        if not negative("dirty_core", expect="modified tracked files"):
            return 1
        nodes.write_bytes(nodes_original)

        log_link = owned / "logs" / "setup.log"
        if log_link.exists() or log_link.is_symlink():
            log_link.unlink()
        log_link.symlink_to(outside)
        sentinel.unlink(missing_ok=True)
        symlink_setup = run_setup()
        if symlink_setup.returncode == 0 or outside.read_text() != "UNCHANGED":
            print(json.dumps({"status": "SYMLINK_NOT_REJECTED"}, ensure_ascii=False))
            return 1
        log_link.unlink()

        from scripts.i1 import ops, readiness  # type: ignore

        runner = LocalRunner(root=owned, env={**setup_env, "I1_FAKE_COMMIT": pinned})
        ctx = SimpleNamespace(
            profile=profile,
            state={"run_id": "r", "nonce": "n"},
            runner=runner,
            identity=object(),
        )
        evidence = ops._run_probe(
            ctx,
            root=str(owned),
            workspace=str(workspace),
            object_info_path=str(object_info_path),
            system_stats_path=str(system_stats_path),
            deadline_epoch=time.time() + 120,
            attempt="0001",
            verification="v0001",
        )
        report = readiness.evaluate(
            probe=evidence,
            profile=profile,
            tunnel_health={"system": {"comfyui_version": "fake"}},
            ui={"ui_status": "DERIVED_FROM_API_NOT_LIVE_VERIFIED", "ui_graph_rendered": False},
        )
        result = {
            "status": "OFFLINE_VERIFIED" if report.get("ready") else "NOT_READY",
            "probe_schema": evidence.get("schema"),
            "models": len(evidence.get("models", [])),
            "venv_executable": evidence.get("python", {}).get("executable"),
            "venv_prefix": evidence.get("python", {}).get("prefix"),
            "setup_restart": True,
            "setup_symlink_rejected": True,
            "stock_nodes_regressions": True,
            "negative_cases": negative_cases,
            "directory_symlink_regressions": True,
            "inspect_stock_ok": True,
            "cache_repeat_ok": True,
            "stock_file_sha256": STOCK_SHA256,
            "fixture_commit_alias": pinned,
            "fixture_real_head": fixture_commit,
            "ready_local_fixture": report.get("ready"),
            "warnings": report.get("warnings", []),
            "blockers": report.get("blockers", []),
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if report.get("ready") else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
