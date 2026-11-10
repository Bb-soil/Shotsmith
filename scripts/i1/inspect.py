"""Read-only remote diagnostics for I1; never mkdir/stage/install."""

from __future__ import annotations

import json
from typing import Any, Mapping

from scripts.p1.remote import RemoteRunner

from . import observation
from . import remote as i1_remote
from . import stock_nodes
from .profile import Profile
from .util import format_bytes, validate_workspace


def _text(result: Any) -> str:
    if getattr(result, "timed_out", False):
        return ""
    value = getattr(result, "stdout", "")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _ok(result: Any) -> bool:
    return not getattr(result, "timed_out", False) and int(getattr(result, "returncode", 1)) == 0


def _run(runner: RemoteRunner, identity: Any, argv: tuple[str, ...], timeout: float = 15.0) -> Any:
    return runner.run_remote(identity, argv, timeout=timeout)


def _check(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    message: str,
) -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def _probe_kind(result: Any) -> str:
    return str(getattr(result, "probe_kind", "ok"))


def _unknown(result: Any) -> bool:
    return _probe_kind(result) in {"timeout", "missing", "not_run"}


def _inconclusive_inspection(
    workspace: str, result: observation.ObservationResult
) -> dict[str, Any]:
    message = (
        "remote observation is inconclusive; no environment fact is treated as "
        "missing and no install mutation is allowed"
    )
    return {
        "compatible": False,
        "conclusive": False,
        "facts": {
            "workspace": workspace,
            "comfy_dir": i1_remote.comfy_dir(workspace),
        },
        "issues": [
            {
                "severity": "warning",
                "code": "observation",
                "message": message,
            }
        ],
        "blockers": [],
        "warnings": [{"severity": "warning", "code": "observation", "message": message}],
        "observation": result.as_dict(),
    }


def _missing_python_inspection(
    workspace: str,
    result: observation.ObservationResult,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    presence = snapshot.get("tool_presence")
    presence = presence if isinstance(presence, Mapping) else {}
    facts: dict[str, Any] = {
        "workspace": workspace,
        "comfy_dir": i1_remote.comfy_dir(workspace),
        "os": None,
        "arch": None,
        "python3": None,
        "venv_module": None,
        "torch": None,
        "gpu": [],
        "comfy_exists": None,
        "comfy_git": None,
        "comfy_commit": None,
        "custom_nodes": None,
        "venv_exists": None,
        "pip_check": None,
        "models": [],
    }
    issues: list[dict[str, str]] = [
        {
            "severity": "blocker",
            "code": "python3",
            "message": "python3 is unavailable (confirmed by framed remote observation)",
        }
    ]
    for name in ("bash", "git"):
        value = presence.get(name)
        if value is False:
            facts[name] = None
            issues.append(
                {
                    "severity": "blocker",
                    "code": name,
                    "message": f"{name} is unavailable (confirmed by framed remote observation)",
                }
            )
        elif value is True:
            facts[name] = "present (version not probed)"
        else:
            facts[name] = None
            issues.append(
                {
                    "severity": "warning",
                    "code": f"{name}_unknown",
                    "message": f"{name} presence was not observed",
                }
            )
    for name in ("sh", "nvidia-smi"):
        if presence.get(name) is None:
            issues.append(
                {
                    "severity": "warning",
                    "code": f"{name}_unknown",
                    "message": f"{name} presence was not observed",
                }
            )
    issues.append(
        {
            "severity": "warning",
            "code": "observation_incomplete",
            "message": "python3 is missing; other environment facts remain unconfirmed",
        }
    )
    blockers = [item for item in issues if item["severity"] == "blocker"]
    warnings = [item for item in issues if item["severity"] == "warning"]
    return {
        "compatible": False,
        "conclusive": False,
        "facts": facts,
        "issues": issues,
        "blockers": blockers,
        "warnings": warnings,
        "observation": result.as_dict(),
    }


def inspect_remote(
    runner: RemoteRunner,
    identity: Any,
    *,
    workspace: str,
    profile: Profile,
) -> dict[str, Any]:
    workspace = validate_workspace(workspace)
    comfy = i1_remote.comfy_dir(workspace)
    venv = i1_remote.venv_python(workspace)
    issues: list[dict[str, str]] = []
    facts: dict[str, Any] = {"workspace": workspace, "comfy_dir": comfy}
    conclusive = True
    snapshot_result = None
    if isinstance(runner, RemoteRunner):
        snapshot_result = observation.capture_snapshot(
            runner,
            identity,
            workspace=workspace,
            profile=profile,
            root=i1_remote.owned_root(workspace),
        )
        if snapshot_result.kind != "OK":
            return _inconclusive_inspection(workspace, snapshot_result)
        snapshot = snapshot_result.snapshot or {}
        if snapshot.get("python3") is False:
            return _missing_python_inspection(workspace, snapshot_result, snapshot)
        try:
            runner = observation.SnapshotRunner(snapshot)
        except ValueError as exc:
            snapshot_result.kind = "PROTOCOL_ERROR"
            snapshot_result.diagnostic = f"snapshot command protocol error: {type(exc).__name__}"
            return _inconclusive_inspection(workspace, snapshot_result)

    uname = _run(runner, identity, ("uname", "-s"))
    facts["os"] = _text(uname).strip()
    if _unknown(uname) or not _ok(uname) or not facts["os"]:
        facts["os"] = None
        conclusive = False
        _check(issues, "warning", "os_unknown", "remote OS probe was inconclusive")
    elif facts["os"] != "Linux":
        _check(issues, "blocker", "os", "remote OS is not Linux")
    arch_result = _run(runner, identity, ("uname", "-m"))
    arch = _text(arch_result).strip()
    facts["arch"] = arch
    if _unknown(arch_result) or not _ok(arch_result) or not arch:
        facts["arch"] = None
        conclusive = False
        _check(issues, "warning", "arch_unknown", "remote architecture probe was inconclusive")
    elif arch not in {"x86_64", "amd64"}:
        _check(issues, "blocker", "arch", "remote architecture is not linux/amd64")
    uid = _text(_run(runner, identity, ("id", "-u"))).strip()
    facts["uid"] = uid
    home_result = _run(runner, identity, ("python3", "-c", observation.HOME_CODE))
    home = _text(home_result).strip()
    facts["home"] = home if home.startswith("/") else None
    if _unknown(home_result):
        conclusive = False
        _check(issues, "warning", "home_unknown", "remote home probe was inconclusive")
    elif not facts["home"]:
        _check(issues, "blocker", "home", "remote home directory is unavailable")

    for name, argv in (
        ("bash", ("bash", "--version")),
        ("git", ("git", "--version")),
        ("python3", ("python3", "--version")),
    ):
        result = _run(runner, identity, argv)
        facts[name] = _text(result).splitlines()[0] if _text(result).strip() else None
        if _unknown(result):
            conclusive = False
            _check(issues, "warning", f"{name}_unknown", f"{name} probe was inconclusive")
        elif not _ok(result):
            _check(issues, "blocker", name, f"{name} is unavailable")
    venv_module = _run(runner, identity, ("python3", "-c", "import venv"))
    facts["venv_module"] = _ok(venv_module) if not _unknown(venv_module) else None
    if _unknown(venv_module):
        conclusive = False
        _check(issues, "warning", "venv_unknown", "python3 venv probe was inconclusive")
    elif not _ok(venv_module):
        _check(issues, "blocker", "venv", "python3 venv module is unavailable")

    containment_code = observation.CONTAINMENT_CODE
    containment = _run(runner, identity, ("python3", "-c", containment_code, workspace))
    facts["containment"] = {}
    if _unknown(containment):
        conclusive = False
        _check(issues, "warning", "containment_unknown", "containment probe was inconclusive")
    elif _ok(containment):
        try:
            facts["containment"] = json.loads(_text(containment))
        except json.JSONDecodeError:
            facts["containment"] = {}
            conclusive = False
            _check(issues, "warning", "containment_unknown", "containment output was not JSON")
    else:
        conclusive = False
        _check(issues, "warning", "containment_unknown", "containment probe failed")
    if facts["containment"].get("symlinks"):
        _check(issues, "blocker", "symlink", "unsafe symlink before setup mutation")
    if facts["containment"].get("outside"):
        _check(issues, "blocker", "containment", "existing path resolves outside workspace")
    if facts["containment"].get("foreign_venv"):
        _check(issues, "blocker", "foreign_venv", "p1-venv exists without our pyvenv.cfg")

    writable = _run(runner, identity, ("test", "-w", workspace))
    facts["workspace_writable"] = _ok(writable) if not _unknown(writable) else None
    if _unknown(writable):
        conclusive = False
        _check(issues, "warning", "workspace_unknown", "workspace writability probe was inconclusive")
    elif not _ok(writable):
        _check(issues, "blocker", "workspace", "workspace is not writable")

    df = _run(runner, identity, ("df", "-Pk", workspace))
    if _unknown(df):
        facts["disk_free_bytes"] = None
        conclusive = False
        _check(issues, "warning", "disk_unknown", "workspace free-space probe was inconclusive")
    else:
        try:
            line = [item for item in _text(df).splitlines() if item.strip()][-1]
            facts["disk_free_bytes"] = int(line.split()[3]) * 1024
        except (IndexError, ValueError):
            facts["disk_free_bytes"] = None
            _check(issues, "blocker", "disk", "cannot read workspace free space")
    mem_result = _run(runner, identity, ("python3", "-c", observation.MEM_CODE))
    try:
        facts["memory"] = json.loads(_text(mem_result))
    except Exception:
        facts["memory"] = {}
    if _unknown(mem_result):
        conclusive = False
        _check(issues, "warning", "memory_unknown", "memory probe was inconclusive")
    ram_total = int(facts["memory"].get("MemTotal", 0) or 0)
    if ram_total and ram_total < int(profile.data["runtime"]["ram_target_bytes"]):
        _check(issues, "warning", "ram", "RAM is below the 64 GB experiment target")
    if not ram_total and not _unknown(mem_result):
        _check(issues, "warning", "ram", "cannot read total RAM")

    smi = _run(
        runner,
        identity,
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
    )
    facts["gpu"] = []
    if _ok(smi):
        for line in _text(smi).splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) >= 3:
                try:
                    memory_mib = int(float(parts[1]))
                except ValueError:
                    memory_mib = 0
                facts["gpu"].append(
                    {"name": parts[0], "memory_bytes": memory_mib * 1024 * 1024, "driver": parts[2]}
                )
    if _unknown(smi):
        conclusive = False
        _check(issues, "warning", "gpu_unknown", "nvidia-smi probe was inconclusive")
    elif not facts["gpu"]:
        _check(issues, "blocker", "gpu", "nvidia-smi did not report a GPU")
    elif max(item["memory_bytes"] for item in facts["gpu"]) < int(profile.data["runtime"]["vram_target_bytes"]):
        _check(issues, "warning", "vram", "VRAM is below the 24 GB experiment target")

    python_cmd = venv if _ok(_run(runner, identity, ("test", "-x", venv))) else "python3"
    facts["inspect_python"] = python_cmd
    torch_code = observation.TORCH_CODE
    torch_result = _run(runner, identity, (python_cmd, "-c", torch_code), timeout=30.0)
    facts["torch"] = None
    if _ok(torch_result):
        try:
            facts["torch"] = json.loads(_text(torch_result))
        except json.JSONDecodeError:
            facts["torch"] = None
    if _unknown(torch_result):
        conclusive = False
        _check(issues, "warning", "cuda_unknown", "torch/CUDA probe was inconclusive")
    elif not facts["torch"] or facts["torch"].get("available") is not True:
        _check(issues, "blocker", "cuda", "torch/CUDA is not available to the installer")
    elif not str(facts["torch"].get("version", "")).startswith(
        str(profile.data["base_torch"].get("expected_prefix", "2.9"))
    ):
        _check(issues, "warning", "torch", "base torch version differs from the expected prefix")

    comfy_result = _run(runner, identity, ("test", "-d", comfy))
    comfy_exists = _ok(comfy_result) if not _unknown(comfy_result) else None
    facts["comfy_exists"] = comfy_exists
    if _unknown(comfy_result):
        conclusive = False
        _check(issues, "warning", "comfy_unknown", "ComfyUI presence probe was inconclusive")
    if comfy_exists is False:
        facts["comfy_git"] = False
        facts["comfy_commit"] = None
    elif comfy_exists:
        git_result = _run(runner, identity, ("test", "-d", comfy + "/.git"))
        git_dir = _ok(git_result) if not _unknown(git_result) else None
        facts["comfy_git"] = git_dir
        if _unknown(git_result):
            conclusive = False
            _check(issues, "warning", "comfy_git_unknown", "ComfyUI git probe was inconclusive")
        elif not git_dir:
            _check(issues, "blocker", "comfy_git", "ComfyUI exists but is not a git checkout")
            facts["custom_nodes"] = None
            facts["custom_nodes_present"] = None
        else:
            commit_result = _run(runner, identity, ("git", "-C", comfy, "rev-parse", "HEAD"))
            commit = _text(commit_result).strip()
            facts["comfy_commit"] = commit if not _unknown(commit_result) else None
            if _unknown(commit_result):
                conclusive = False
                _check(issues, "warning", "comfy_commit_unknown", "ComfyUI commit probe was inconclusive")
            elif commit != profile.comfy["commit"]:
                _check(issues, "blocker", "comfy_commit", "ComfyUI commit differs from the pinned profile")
            dirty_result = _run(
                runner,
                identity,
                ("git", "-C", comfy, "status", "--porcelain", "--untracked-files=no"),
            )
            dirty = _text(dirty_result).strip()
            facts["comfy_dirty"] = bool(dirty) if not _unknown(dirty_result) else None
            if _unknown(dirty_result):
                conclusive = False
                _check(issues, "warning", "comfy_dirty_unknown", "ComfyUI dirty probe was inconclusive")
            elif dirty:
                _check(issues, "blocker", "comfy_dirty", "ComfyUI has modified tracked files")
            if commit == profile.comfy["commit"] and not dirty and not _unknown(dirty_result):
                nodes_result = _run(
                    runner,
                    identity,
                    stock_nodes.remote_argv(comfy, profile.comfy["commit"]),
                    timeout=60.0,
                )
                nodes = stock_nodes.parse_output(_text(nodes_result))
                facts["custom_nodes"] = nodes
                facts["custom_nodes_present"] = bool(nodes.get("tracked"))
                if _unknown(nodes_result):
                    conclusive = False
                    _check(
                        issues,
                        "warning",
                        "custom_nodes_unknown",
                        "custom_nodes: проверка не завершена (inconclusive)",
                    )
                elif not _ok(nodes_result) or not nodes.get("ok"):
                    _check(
                        issues,
                        "blocker",
                        "custom_nodes",
                        "custom_nodes: проверка не пройдена: "
                        + stock_nodes.describe_violations(nodes),
                    )
                elif nodes.get("cache_unverified") or nodes.get("cache_dirs"):
                    _check(
                        issues,
                        "warning",
                        "custom_nodes_cache",
                        "custom_nodes: runtime __pycache__ присутствует; содержимое не проверено",
                    )
            else:
                facts["custom_nodes"] = None
                facts["custom_nodes_present"] = None
        requirements_result = _run(
            runner, identity, ("test", "-f", comfy + "/requirements.txt")
        )
        facts["requirements_present"] = (
            _ok(requirements_result) if not _unknown(requirements_result) else None
        )
        if _unknown(requirements_result):
            conclusive = False
            _check(issues, "warning", "requirements_unknown", "requirements probe was inconclusive")
        elif not facts["requirements_present"]:
            _check(issues, "blocker", "requirements", "ComfyUI requirements.txt is missing")
    else:
        facts["comfy_git"] = None
        facts["comfy_commit"] = None

    venv_result = _run(runner, identity, ("test", "-x", venv))
    venv_exists = _ok(venv_result) if not _unknown(venv_result) else None
    facts["venv_exists"] = venv_exists
    if _unknown(venv_result):
        conclusive = False
        _check(issues, "warning", "venv_exists_unknown", "venv presence probe was inconclusive")
    if venv_exists:
        pip_check = _run(runner, identity, (venv, "-m", "pip", "check"), timeout=60.0)
        facts["pip_check"] = _ok(pip_check) if not _unknown(pip_check) else None
        if _unknown(pip_check):
            conclusive = False
            _check(issues, "warning", "pip_check_unknown", "pip check probe was inconclusive")
        elif not _ok(pip_check):
            _check(issues, "blocker", "pip_check", "existing venv has a pip dependency conflict")
    else:
        facts["pip_check"] = None

    models = []
    for item in profile.models:
        target = f"{comfy}/{item['target']}"
        result = _run(runner, identity, ("stat", "-c", "%s", target))
        actual = None
        if _ok(result):
            try:
                actual = int(_text(result).strip())
            except ValueError:
                actual = None
        if _unknown(result):
            conclusive = False
            _check(
                issues,
                "warning",
                "model_unknown",
                f"model size probe was inconclusive: {item['id']}",
            )
        models.append(
            {
                "id": item["id"],
                "target": item["target"],
                "expected_size": item["size_bytes"],
                "actual_size": actual,
                "present": actual is not None,
                "size_matches": actual == item["size_bytes"],
            }
        )
    facts["models"] = models
    existing_valid = sum(
        int(item["expected_size"]) for item in models if item.get("size_matches")
    )
    facts["disk_required_bytes"] = max(0, profile.disk_required_bytes - existing_valid)
    facts["existing_model_bytes"] = existing_valid
    if (
        facts["disk_free_bytes"] is not None
        and facts["disk_free_bytes"] < facts["disk_required_bytes"]
    ):
        _check(
            issues,
            "blocker",
            "disk",
            f"free {format_bytes(facts['disk_free_bytes'])} < additional required "
            f"{format_bytes(facts['disk_required_bytes'])}",
        )
    blockers = [item for item in issues if item["severity"] == "blocker"]
    warnings = [item for item in issues if item["severity"] == "warning"]
    return {
        "compatible": not blockers and conclusive,
        "conclusive": conclusive,
        "facts": facts,
        "issues": issues,
        "blockers": blockers,
        "warnings": warnings,
        "observation": (
            snapshot_result.as_dict() if snapshot_result is not None else None
        ),
    }


__all__ = ["inspect_remote"]
