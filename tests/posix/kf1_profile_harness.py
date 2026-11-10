#!/usr/bin/env python3
"""KF1 isolated POSIX tiny worker/probe execution.

This harness runs under WSL.  It builds a real lexical-venv fixture, fake CUDA,
fake Comfy checkout/git and tiny model files, stages the production I1
profile/util/probe/preflight modules into an isolated directory, and executes
the actual staged probe.  The probe process has no repo root import fallback;
only the fixture's fake torch path is on PYTHONPATH.

Test-only fixture adapter: the staged copy of i1_profile.py has only the
KNOWN_PROFILES total_bytes values patched to the tiny fixture sums.  Production
validators, model checks, probe schema and readiness checks are not bypassed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _stage_module(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def _fake_object_info(profile: dict, repo: Path) -> dict:
    """Finite actual-like object_info, not a ``*`` cache.

    ``LoadImage.image`` is a clean-worker empty enum with ``image_upload``.
    Classes/inputs/outputs/enums/ranges/model names are realistic; a superset
    including Wan classes covers the bundle probe without reusing hidden ``*``.
    """

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tests.kf1_object_info import build_object_info  # type: ignore

    data = build_object_info(load_image_files=[], include_wan=True)
    # The actual probe must exercise a genuine clean-worker empty upload enum,
    # not a wildcard that silently accepts any filename.
    assert data["LoadImage"]["input"]["required"]["image"][0] == []
    assert data["LoadImage"]["input"]["required"]["image"][1]["image_upload"] is True
    return data


def _fake_git(path: Path, real_git: str, pinned: str) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
real={real_git!r}
if [ "${{1:-}}" = "--version" ]; then echo 'git version fixture'; exit 0; fi
if [ "${{1:-}}" = "-C" ]; then
  dir="$2"; shift 2
  if [ "${{1:-}}" = "rev-parse" ] && {{ [ "${{2:-}}" = "HEAD" ] || [ "${{2:-}}" = "--verify" ]; }}; then echo "${{I1_FAKE_COMMIT:-{pinned}}}"; exit 0; fi
  exec "$real" -C "$dir" "$@"
fi
exec "$real" "$@"
"""
    _write(path, script)
    path.chmod(path.stat().st_mode | 0o111)


def _make_tiny_profile(profile: dict, workspace: Path, comfy: Path, payloads: dict[str, bytes]) -> dict:
    data = json.loads(json.dumps(profile))
    for item in data["models"]:
        payload = f"tiny-{item['role']}-{item['id']}".encode()
        payloads[item["target"]] = payload
        item["size_bytes"] = len(payload)
        item["sha256"] = _sha(payload)
        item["url"] = "https://example.invalid/tiny.bin"
        item["binary_verified_at_derivation"] = False
    data["disk"]["install_overhead_bytes"] = 0
    data["disk"]["reserve_bytes"] = 0
    return data


def _run_probe(
    *,
    profile_data: dict,
    profile_path: Path,
    owned: Path,
    workspace: Path,
    comfy: Path,
    venv_python: Path,
    workflows: dict[str, dict],
    object_info: Path,
    system_stats: Path,
    env: dict[str, str],
    deadline: float,
) -> tuple[int, dict | None, str]:
    workflow_specs = []
    for workflow_id, meta in workflows.items():
        workflow_specs.append(
            {
                "workflow_id": workflow_id,
                "path": str(owned / "workflow" / f"{workflow_id}.api.json"),
                "sha256": meta["api_sha256"],
                "required_classes": meta["required_classes"],
                "required_inputs": meta["required_inputs"],
                "node_classes": meta["node_classes"],
                "graph_inputs": meta["graph_inputs"],
                "output_specs": [{"workflow_id": workflow_id, **meta["output"]}],
            }
        )
    primary = workflow_specs[0]
    plan = {
        "schema": "anime-video-studio/i1-runtime-plan/v1",
        "run_id": "r",
        "nonce": "n",
        "workspace": str(workspace),
        "owned_root": str(owned),
        "comfy_dir": str(comfy),
        "venv_python": str(venv_python),
        "profile_path": str(profile_path),
        "profile_sha256": "0" * 64,
        "workflow_api_path": primary["path"],
        "workflow_sha256": primary["sha256"],
        "object_info_path": str(object_info),
        "system_stats_path": str(system_stats),
        "preflight_path": str(owned / "preflight.py"),
        "expected_comfy_commit": profile_data["comfyui"]["commit"],
        "expected_model_count": len(profile_data["models"]),
        "required_classes": primary["required_classes"],
        "required_inputs": primary["required_inputs"],
        "node_classes": primary["node_classes"],
        "graph_inputs": primary["graph_inputs"],
        "workflows": workflow_specs,
        "output_specs": [spec for item in workflow_specs for spec in item["output_specs"]],
        "models": [
            {
                "id": item["id"],
                "role": item["role"],
                "target": item["target"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in profile_data["models"]
        ],
        "deadline_epoch": deadline,
    }
    # The staged production profile copy must have its canonical hash in the plan.
    sys.path.insert(0, str(owned))
    try:
        import i1_profile as staged_profile  # type: ignore

        plan["profile_sha256"] = staged_profile.load_profile(profile_path).sha256
    finally:
        sys.path.pop(0)
    plan_path = owned / "runtime-plan.json"
    _write(plan_path, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    proc = subprocess.run(
        [str(venv_python), str(owned / "probe.py"), "--plan", str(plan_path)],
        cwd=str(owned),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    payload = None
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = None
    return proc.returncode, payload, proc.stderr[-2000:]


def _run_worker(
    *, profile_path: Path, profile_data: dict, owned: Path, workspace: Path,
    comfy: Path, venv_python: Path, env: dict[str, str], label: str,
) -> dict:
    attempt = owned / "attempts" / "0001"
    attempt.mkdir(parents=True, exist_ok=True)
    staged_profile = attempt / "profile.json"
    staged_profile.write_bytes(profile_path.read_bytes())
    target_home = workspace / "target-home"
    target_id = hashlib.sha256(label.encode()).hexdigest()
    target_lock = target_home / ".cache" / "anime-video-studio" / "i1" / "targets" / f"{target_id}.lock"
    target_lock.parent.mkdir(parents=True, exist_ok=True)
    status = attempt / "status.json"
    progress = attempt / "progress.json"
    heartbeat = attempt / "heartbeat.json"
    job = attempt / "job.json"
    payload = {
        "schema": "anime-video-studio/i1-download-job/v1",
        "run_id": "r",
        "nonce": "n",
        "workspace": str(workspace),
        "owned_root": str(attempt),
        "attempt_dir": str(attempt),
        "comfy_root": str(comfy),
        "profile_path": str(staged_profile),
        "profile_sha256": "",
        "target_home": str(target_home),
        "target_id": target_id,
        "target_lock_path": str(target_lock),
        "deadline_epoch": time.time() + 120,
        "log": str(attempt / "download.log"),
        "status_path": str(status),
        "progress_path": str(progress),
        "heartbeat_path": str(heartbeat),
        "required_bytes": 0,
    }
    # Compute profile hash with the staged patched profile.
    sys.path.insert(0, str(owned))
    try:
        import i1_profile as staged_profile_module  # type: ignore

        payload["profile_sha256"] = staged_profile_module.load_profile(staged_profile).sha256
    finally:
        sys.path.pop(0)
    _write(job, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    proc = subprocess.run(
        [str(venv_python), str(owned / "worker.py"), str(job)],
        cwd=str(owned), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    report = None
    if status.is_file():
        report = json.loads(status.read_text(encoding="utf-8"))
    return {
        "exit": proc.returncode,
        "status": report.get("status") if isinstance(report, dict) else None,
        "files": len(report.get("files", [])) if isinstance(report, dict) else 0,
        "skipped": sum(1 for item in report.get("files", []) if item.get("skipped")) if isinstance(report, dict) else 0,
        "stderr": proc.stderr[-1000:],
    }
def main() -> int:
    global REPO
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    REPO = Path(args.repo).resolve()
    sys.path.insert(0, str(REPO))
    from tests.posix.i1_posix_harness import _make_fake_venv, _make_nvidia_smi  # type: ignore

    root = Path(tempfile.mkdtemp(prefix="kf1-posix-"))
    try:
        bin_dir = root / "bin"
        bin_dir.mkdir()
        _make_nvidia_smi(bin_dir / "nvidia-smi")
        real_git = shutil.which("git")
        if not real_git:
            print(json.dumps({"status": "GIT_MISSING"}))
            return 1
        env = dict(os.environ)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env["PIP_NO_INDEX"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        workspace = root / "workspace"
        owned = workspace / ".avs-i1"
        comfy = workspace / "ComfyUI"
        owned.mkdir(parents=True)
        comfy.mkdir(parents=True)
        (comfy / "comfy").mkdir()
        _write(comfy / "comfy" / "__init__.py", "")
        _write(comfy / "nodes.py", "")
        (comfy / "custom_nodes").mkdir()
        (comfy / "custom_nodes" / "example_node.py.example").write_text("stock\n")
        shutil.copy2(REPO / "docs/i1-evidence/codex-review-20261029/pinned-websocket_image_save.py", comfy / "custom_nodes" / "websocket_image_save.py")
        subprocess.run([real_git, "-C", str(comfy), "init", "-q"], check=True)
        subprocess.run([real_git, "-C", str(comfy), "config", "user.email", "fixture@example.invalid"], check=True)
        subprocess.run([real_git, "-C", str(comfy), "config", "user.name", "fixture"], check=True)
        subprocess.run([real_git, "-C", str(comfy), "config", "core.autocrlf", "false"], check=True)
        subprocess.run([real_git, "-C", str(comfy), "add", "-A"], check=True)
        subprocess.run([real_git, "-C", str(comfy), "commit", "-qm", "pinned stock fixture"], check=True)
        _fake_git(bin_dir / "git", real_git, "7a28332047b6af638dc6c7f72ef3481a718769fd")

        venv = workspace / "p1-venv"
        _make_fake_venv(venv)
        venv_python = venv / "bin" / "python"
        env["I1_FAKE_COMMIT"] = "7a28332047b6af638dc6c7f72ef3481a718769fd"
        env["PYTHONPATH"] = str(next((venv / "lib").glob("*/site-packages")))

        # Stage production modules under isolated names; no repo path is used by
        # the probe process.
        stage = owned
        for source, name in (
            (REPO / "scripts/i1/util.py", "i1_util.py"),
            (REPO / "scripts/i1/probe.py", "probe.py"),
            (REPO / "scripts/i1/worker.py", "worker.py"),
            (REPO / "scripts/p1/preflight.py", "preflight.py"),
            (REPO / "scripts/p1/setup_plan.py", "setup_plan.py"),
        ):
            _stage_module(source, stage / name)
        profile_source = (REPO / "scripts/i1/profile.py").read_text(encoding="utf-8")
        # Test-only DATA patch: registry totals become the tiny fixture sums.
        k1_payloads: dict[str, bytes] = {}
        bundle_payloads: dict[str, bytes] = {}
        k1_raw = json.loads((REPO / "config/i1/keyframes-qwen-edit-2511.profile.json").read_text(encoding="utf-8"))
        bundle_raw = json.loads((REPO / "config/i1/bundle-wan22-qwen-edit-2511.profile.json").read_text(encoding="utf-8"))
        k1_tiny = _make_tiny_profile(k1_raw, workspace, comfy, k1_payloads)
        bundle_tiny = _make_tiny_profile(bundle_raw, workspace, comfy, bundle_payloads)
        tiny_totals = {
            "keyframes-qwen-edit-2511": sum(item["size_bytes"] for item in k1_tiny["models"]),
            "bundle-wan22-qwen-edit-2511": sum(item["size_bytes"] for item in bundle_tiny["models"]),
        }
        profile_source = profile_source.replace("30_172_239_743", str(tiny_totals["keyframes-qwen-edit-2511"]))
        profile_source = profile_source.replace("65_751_447_622", str(tiny_totals["bundle-wan22-qwen-edit-2511"]))
        _write(stage / "i1_profile.py", profile_source)
        sys.path.insert(0, str(stage))
        try:
            import i1_profile as staged_profile  # type: ignore

            k1 = staged_profile.validate_profile(k1_tiny)
            bundle = staged_profile.validate_profile(bundle_tiny)
        finally:
            sys.path.pop(0)

        def prepare_profile(profile, raw, payloads, label):
            profile_path = stage / f"{label}-profile.json"
            # Source inventory path is irrelevant for this probe; keep exact
            # workflow pins and tiny model metadata.
            _write(profile_path, json.dumps(raw, indent=2, sort_keys=True) + "\n")
            for target, data in payloads.items():
                path = comfy / target
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            workflows = {}
            for workflow_id, meta in profile.workflows.items():
                source = REPO / meta["api_path"]
                dest = stage / "workflow" / f"{workflow_id}.api.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(source.read_bytes())
                workflows[workflow_id] = meta
            object_info = workspace / "p1-evidence" / f"object_info-{label}.json"
            system_stats = workspace / "p1-evidence" / f"system_stats-{label}.json"
            object_info.parent.mkdir(parents=True, exist_ok=True)
            _write(object_info, json.dumps(_fake_object_info(raw, REPO), indent=2))
            _write(system_stats, json.dumps({"devices": [{"name": "Fake GPU", "vram_total": 1}], "system": {"comfyui_version": "fake"}}))
            return profile_path, workflows, object_info, system_stats

        results = {}
        for label, profile, raw, payloads in (
            ("k1", k1, k1_tiny, k1_payloads),
            ("bundle", bundle, bundle_tiny, bundle_payloads),
        ):
            profile_path, workflows, object_info, system_stats = prepare_profile(profile, raw, payloads, label)
            code, payload, stderr = _run_probe(
                profile_data=raw,
                profile_path=profile_path,
                owned=owned,
                workspace=workspace,
                comfy=comfy,
                venv_python=venv_python,
                workflows=workflows,
                object_info=object_info,
                system_stats=system_stats,
                env=env,
                deadline=time.time() + 120,
            )
            if code != 0 or not payload:
                print(json.dumps({"status": "PROBE_FAILED", "label": label, "exit": code, "stderr": stderr}))
                return 1
            results[label] = {
                "models": len(payload.get("models", [])),
                "workflows": len(payload.get("workflows", [])),
                "outputs": sorted(
                    {
                        item.get("class")
                        for workflow in payload.get("workflows", [])
                        for item in workflow.get("output_checks", [])
                    }
                ),
                "schema": payload.get("schema"),
                "worker": _run_worker(
                    profile_path=profile_path,
                    profile_data=raw,
                    owned=owned,
                    workspace=workspace,
                    comfy=comfy,
                    venv_python=venv_python,
                    env=env,
                    label=label,
                ),
            }

        # Missing Qwen file must fail the actual staged probe.
        missing = next((comfy / item["target"] for item in k1_tiny["models"] if item["role"] == "diffusion"), None)
        assert missing is not None
        profile_path, workflows, object_info, system_stats = prepare_profile(k1, k1_tiny, k1_payloads, "k1-missing")
        missing.unlink()
        code, _payload, _stderr = _run_probe(
            profile_data=k1_tiny, profile_path=profile_path, owned=owned, workspace=workspace,
            comfy=comfy, venv_python=venv_python, workflows=workflows,
            object_info=object_info, system_stats=system_stats, env=env,
            deadline=time.time() + 120,
        )
        if code == 0:
            print(json.dumps({"status": "MISSING_MODEL_NOT_REJECTED"}))
            return 1
        # Restore the deliberately missing file before the byte-tamper probe.
        missing.write_bytes(k1_payloads[missing.relative_to(comfy).as_posix()])

        # A stored READY must not authorize a later run when model bytes change.
        tamper_target = next(iter(k1_payloads))
        tamper_path = comfy / tamper_target
        tamper_path.write_bytes(k1_payloads[tamper_target] + b"tamper")
        profile_path, workflows, object_info, system_stats = prepare_profile(
            k1, k1_tiny, k1_payloads, "k1-tamper"
        )
        # prepare_profile rewrites tiny payloads, so tamper again after it.
        tamper_path.write_bytes(k1_payloads[tamper_target] + b"tamper")
        tamper_code, _tamper_payload, _tamper_stderr = _run_probe(
            profile_data=k1_tiny,
            profile_path=profile_path,
            owned=owned,
            workspace=workspace,
            comfy=comfy,
            venv_python=venv_python,
            workflows=workflows,
            object_info=object_info,
            system_stats=system_stats,
            env=env,
            deadline=time.time() + 120,
        )
        if tamper_code == 0:
            print(json.dumps({"status": "MODEL_TAMPER_NOT_REJECTED"}))
            return 1
        tamper_path.write_bytes(k1_payloads[tamper_target])

        print(
            json.dumps(
                {
                    "status": "OFFLINE_VERIFIED",
                    "k1": results["k1"],
                    "bundle": results["bundle"],
                    "missing_model_exit": code,
                    "model_tamper_exit": tamper_code,
                    "model_tamper_rejected": tamper_code != 0,
                    "probe": "actual staged production probe.py (no HTTP/POST code path)",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())




