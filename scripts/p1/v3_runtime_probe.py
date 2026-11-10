"""Fixed remote runtime-evidence probe for P1 v3.

This file is staged into the owned remote root and executed by the owned venv
python. It reads a machine-generated plan, streams model bytes (never copies
weights), validates object_info and emits a compact JSON fact set on stdout.
It does not import secrets, call a provider, or use a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

SCHEMA = "anime-video-studio/p1-runtime-evidence/v1"
PLAN_SCHEMA = "anime-video-studio/p1-runtime-evidence-plan/v1"
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
MAX_OBJECT_INFO_BYTES = 16 * 1024 * 1024


class ProbeError(Exception):
    pass


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path, *, deadline_epoch: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if deadline_epoch is not None and time.time() >= deadline_epoch:
                raise ProbeError("deadline expired while hashing model bytes")
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _run_bounded(argv: list[str], *, deadline_epoch: float | None, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    if deadline_epoch is not None:
        remaining = deadline_epoch - time.time()
        if remaining <= 0:
            raise ProbeError("deadline expired before a bounded subprocess")
        timeout = min(timeout, max(0.05, remaining))
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read runtime plan: {type(exc).__name__}") from exc
    if not isinstance(data, dict) or data.get("schema") != PLAN_SCHEMA:
        raise ProbeError("runtime plan schema is invalid")
    data["_plan_sha256"] = _sha_bytes(raw)
    return data


def _require_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProbeError(f"{name} is missing")
    path = Path(value)
    if not (
        path.is_absolute()
        or PurePosixPath(value.replace("\\", "/")).is_absolute()
    ):
        raise ProbeError(f"{name} must be absolute")
    return path


def _inside(path: Path, root: Path, name: str) -> Path:
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError as exc:
        raise ProbeError(f"{name} cannot be resolved") from exc
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ProbeError(f"{name} escapes the owned workspace")
    return resolved


def _model_path(workspace: Path, target: Any) -> Path:
    if not isinstance(target, str) or not target:
        raise ProbeError("model target is missing")
    relative = PurePosixPath(target.replace("\\", "/"))
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ProbeError("model target is not a normalized relative path")
    if not relative.as_posix().startswith("models/"):
        raise ProbeError("model target must be under models/")
    path = workspace / Path(*relative.parts)
    resolved = _inside(path, workspace / "models", "model target")
    if not resolved.is_file():
        raise ProbeError(f"model file is missing: {target}")
    if resolved.is_symlink():
        raise ProbeError(f"model file is a symlink: {target}")
    return resolved


def _torch_facts() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise ProbeError(f"torch import failed: {type(exc).__name__}") from exc
    version = str(getattr(torch, "__version__", ""))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not version or not cuda_version:
        raise ProbeError("torch/CUDA version is unavailable")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        raise ProbeError("torch.cuda is unavailable")
    available = bool(cuda.is_available())
    facts: dict[str, Any] = {
        "version": version,
        "cuda_version": str(cuda_version),
        "cuda_available": available,
        "device_count": 0,
        "device_name": None,
        "device_capability": None,
        "total_vram_bytes": None,
    }
    if available:
        try:
            facts["device_count"] = int(cuda.device_count())
        except Exception:
            facts["device_count"] = 0
        try:
            facts["device_name"] = str(cuda.get_device_name(0))
        except Exception:
            facts["device_name"] = None
        try:
            capability = cuda.get_device_capability(0)
            facts["device_capability"] = list(capability) if isinstance(capability, tuple) else capability
        except Exception:
            facts["device_capability"] = None
        try:
            facts["total_vram_bytes"] = int(cuda.get_device_properties(0).total_memory)
        except Exception:
            facts["total_vram_bytes"] = None
    return facts


def _module_origin(name: str, module: Any) -> Path | None:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        return None
    return Path(origin).resolve()


def _comfy_origin(module: Any, checkout: Path) -> tuple[Path, str]:
    """Require file-backed package/module or one exact checkout namespace."""

    expected_package = (checkout / "comfy").resolve()
    expected_file_package = (checkout / "comfy.py").resolve()
    expected_init = (checkout / "comfy" / "__init__.py").resolve()
    origin = _module_origin("comfy", module)
    spec = getattr(module, "__spec__", None)
    spec_raw = getattr(spec, "submodule_search_locations", None)
    path_raw = getattr(module, "__path__", None)
    search_raw = spec_raw if spec_raw is not None else path_raw
    search_locations: list[Path] = []
    if search_raw is not None:
        try:
            search_locations = [Path(str(item)).resolve() for item in search_raw]
        except (TypeError, OSError) as exc:
            raise ProbeError("comfy search locations are invalid") from exc
    if spec_raw is not None and path_raw is not None:
        try:
            path_locations = [Path(str(item)).resolve() for item in path_raw]
        except (TypeError, OSError) as exc:
            raise ProbeError("comfy __path__ locations are invalid") from exc
        if path_locations != search_locations:
            raise ProbeError("comfy __path__ differs from its verified spec origins")
    if origin is None:
        if len(search_locations) != 1 or search_locations[0] != expected_package:
            raise ProbeError("foreign comfy namespace search locations")
        if not expected_package.is_dir():
            raise ProbeError("comfy namespace directory is missing")
        return expected_package, "namespace"
    if origin not in {expected_init, expected_file_package}:
        raise ProbeError("foreign comfy module origin is ambiguous")
    if search_locations and search_locations != [expected_package]:
        raise ProbeError("foreign comfy package search locations")
    return origin, "file"


def _nodes_origin(module: Any, checkout: Path) -> Path:
    expected = (checkout / "nodes.py").resolve()
    origin = _module_origin("nodes", module)
    if origin is None or origin != expected:
        raise ProbeError("foreign nodes module origin is ambiguous")
    return origin


def _comfy_facts(comfy_dir: Path, expected_commit: Any) -> dict[str, Any]:
    if not isinstance(expected_commit, str) or not HEX40.match(expected_commit):
        raise ProbeError("expected ComfyUI commit is invalid")
    if not comfy_dir.is_dir():
        raise ProbeError("ComfyUI directory is missing")
    checkout = comfy_dir.resolve()
    for name in ("comfy", "nodes"):
        if name in sys.modules:
            raise ProbeError(f"foreign pre-imported module is not allowed: {name}")

    top = _run_bounded(
        ["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
        deadline_epoch=None,
        timeout=30.0,
    )
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != checkout:
        raise ProbeError("ComfyUI directory is not a verified checkout root")
    result = _run_bounded(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        deadline_epoch=None,
        timeout=30.0,
    )
    if result.returncode != 0:
        raise ProbeError("cannot read actual ComfyUI commit")
    actual = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not HEX40.match(actual):
        raise ProbeError("actual ComfyUI commit is invalid")
    if actual != expected_commit:
        raise ProbeError("actual ComfyUI commit differs from the source pin")

    old_cwd = Path.cwd()
    inserted = str(checkout)
    sys.path.insert(0, inserted)
    try:
        os.chdir(checkout)
        try:
            import comfy
            import nodes
        except Exception as exc:
            raise ProbeError(f"ComfyUI import failed: {type(exc).__name__}") from exc
    finally:
        try:
            os.chdir(old_cwd)
        finally:
            try:
                sys.path.remove(inserted)
            except ValueError:
                pass

    comfy_origin, comfy_kind = _comfy_origin(comfy, checkout)
    nodes_origin = _nodes_origin(nodes, checkout)
    return {
        "commit": actual,
        "import_ok": True,
        "import_error": None,
        "module": str(comfy_origin),
        "module_kind": comfy_kind,
        "nodes_module": str(nodes_origin),
        "checkout_root": str(checkout),
    }


def _pip_facts(deadline_epoch: float | None) -> dict[str, Any]:
    check = _run_bounded(
        [sys.executable, "-m", "pip", "check"],
        deadline_epoch=deadline_epoch,
        timeout=60.0,
    )
    if check.returncode != 0:
        raise ProbeError("pip check failed")
    freeze = _run_bounded(
        [sys.executable, "-m", "pip", "freeze"],
        deadline_epoch=deadline_epoch,
        timeout=60.0,
    )
    if freeze.returncode != 0:
        raise ProbeError("pip freeze failed")
    freeze_text = (freeze.stdout or "").replace("\r\n", "\n")
    lines = [line for line in freeze_text.splitlines() if line.strip()]
    return {
        "check_exit": check.returncode,
        "check_stdout_sha256": _sha_bytes((check.stdout or "").encode("utf-8")),
        "freeze_sha256": _sha_bytes(freeze_text.encode("utf-8")),
        "freeze_line_count": len(lines),
    }


def _object_info_facts(
    path: Path,
    *,
    preflight_path: Path,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ProbeError("object_info file is missing")
    if path.is_symlink():
        raise ProbeError("object_info file is a symlink")
    size = path.stat().st_size
    if size <= 0 or size > MAX_OBJECT_INFO_BYTES:
        raise ProbeError("object_info file size is invalid")
    data = path.read_bytes()
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"object_info is not valid JSON: {type(exc).__name__}") from exc
    if not isinstance(parsed, dict):
        raise ProbeError("object_info must be an object")
    sys.path.insert(0, str(preflight_path.parent))
    try:
        import preflight  # type: ignore
    except Exception as exc:
        raise ProbeError(f"preflight validator import failed: {type(exc).__name__}") from exc
    summary = preflight.summarize_object_info(parsed)
    descriptors = None
    if class_names is not None:
        if not isinstance(class_names, list) or not all(
            isinstance(item, str) for item in class_names
        ):
            raise ProbeError("plan required_classes is invalid")
        descriptors = {}
        for name in class_names:
            descriptor = parsed.get(name)
            if not isinstance(descriptor, dict):
                raise ProbeError(f"object_info class descriptor is missing: {name}")
            descriptors[name] = descriptor
    return {
        "sha256": _sha_bytes(data),
        "size_bytes": size,
        **summary,
        **({"descriptors": descriptors} if descriptors is not None else {}),
    }


def _system_stats_facts(path: Path | None, *, required: bool) -> dict[str, Any] | None:
    if path is None:
        if required:
            raise ProbeError("system_stats path is missing")
        return None
    if not path.is_file():
        if required:
            raise ProbeError("system_stats file is missing")
        return None
    if path.is_symlink():
        raise ProbeError("system_stats file is a symlink")
    data = path.read_bytes()
    if not data or len(data) > MAX_OBJECT_INFO_BYTES:
        raise ProbeError("system_stats file size is invalid")
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"system_stats is not valid JSON: {type(exc).__name__}") from exc
    if not isinstance(parsed, dict):
        raise ProbeError("system_stats must be an object")
    raw_devices = parsed.get("devices")
    if not isinstance(raw_devices, list):
        raw_devices = []
    devices: list[dict[str, Any]] = []
    for item in raw_devices:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        device_type = item.get("type") or item.get("device_type")
        memory = (
            item.get("vram_total")
            or item.get("memory_total")
            or item.get("vram_total_bytes")
            or item.get("total_memory")
        )
        devices.append(
            {
                "name": name,
                "type": device_type,
                "index": item.get("index"),
                "vram_total": memory,
            }
        )
    if required and not devices:
        raise ProbeError("system_stats has no CUDA devices")
    system = parsed.get("system") if isinstance(parsed.get("system"), Mapping) else {}
    return {
        "sha256": _sha_bytes(data),
        "devices": devices,
        "comfyui_version": system.get("comfyui_version"),
    }


def _graph_facts(
    plan: Mapping[str, Any],
    object_info: Mapping[str, Any],
) -> dict[str, Any]:
    required_classes = plan.get("required_classes")
    graph_inputs = plan.get("graph_inputs")
    required_inputs = plan.get("required_inputs")
    node_classes = plan.get("node_classes")
    if not isinstance(required_classes, list) or not all(isinstance(item, str) for item in required_classes):
        raise ProbeError("plan required_classes is invalid")
    if not isinstance(node_classes, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in node_classes.items()
    ):
        raise ProbeError("plan node_classes is invalid")
    present = set(object_info.get("classes") or [])
    missing = [name for name in required_classes if name not in present]
    input_names = object_info.get("input_names")
    if not isinstance(input_names, Mapping):
        raise ProbeError("object_info input names are missing")
    input_checks: list[dict[str, Any]] = []
    if graph_inputs is not None:
        if not isinstance(graph_inputs, list):
            raise ProbeError("plan graph_inputs is invalid")
        for item in graph_inputs:
            if not isinstance(item, Mapping):
                raise ProbeError("plan graph_inputs entry is invalid")
            node_id = item.get("node_id")
            class_name = item.get("class")
            input_name = item.get("input")
            if not all(isinstance(value, str) and value for value in (node_id, class_name, input_name)):
                raise ProbeError("plan graph_inputs identity is invalid")
            input_checks.append(
                {
                    "node_id": node_id,
                    "class": class_name,
                    "input": input_name,
                    "present": input_name in set(input_names.get(class_name) or []),
                }
            )
    else:
        if not isinstance(required_inputs, Mapping):
            raise ProbeError("plan required_inputs is invalid")
        for class_name, names in required_inputs.items():
            if not isinstance(names, list):
                raise ProbeError("plan required_inputs entry is invalid")
            actual = set(input_names.get(class_name) or [])
            for name in names:
                input_checks.append(
                    {
                        "class": class_name,
                        "input": name,
                        "present": name in actual,
                    }
                )
    return {
        "node_classes": dict(node_classes),
        "required_classes": required_classes,
        "missing_graph_classes": missing,
        "input_checks": input_checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v3_runtime_probe.py")
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    try:
        plan_path = Path(args.plan)
        plan = _load_plan(plan_path)
        workspace = _require_path(plan.get("workspace"), "workspace")
        owned_root = _require_path(plan.get("owned_root"), "owned_root")
        venv_python = _require_path(plan.get("venv_python"), "venv_python")
        comfy_dir = _require_path(plan.get("comfy_dir"), "comfy_dir")
        object_info_path = _require_path(plan.get("object_info_path"), "object_info_path")
        preflight_path = _require_path(plan.get("preflight_path"), "preflight_path")
        system_stats_raw = plan.get("system_stats_path")
        system_stats_path = (
            _require_path(system_stats_raw, "system_stats_path")
            if system_stats_raw is not None
            else None
        )
        require_system_stats = plan.get("require_system_stats") is True
        deadline_epoch = plan.get("deadline_epoch")
        if isinstance(deadline_epoch, bool) or not isinstance(deadline_epoch, (int, float)):
            raise ProbeError("deadline_epoch is missing")
        deadline = float(deadline_epoch)
        if time.time() >= deadline:
            raise ProbeError("deadline expired before probe started")
        if workspace.resolve() != comfy_dir.resolve() and workspace.resolve() not in comfy_dir.resolve().parents:
            raise ProbeError("comfy_dir escapes workspace")
        _inside(object_info_path, workspace, "object_info_path")
        if system_stats_path is not None:
            _inside(system_stats_path, workspace, "system_stats_path")
        _inside(preflight_path, owned_root, "preflight_path")
        if venv_python.resolve() != Path(sys.executable).resolve():
            raise ProbeError("probe is not executing under the planned owned venv")

        models: list[dict[str, Any]] = []
        expected_models = plan.get("models")
        if not isinstance(expected_models, list) or not expected_models:
            raise ProbeError("plan models are missing")
        for item in expected_models:
            if not isinstance(item, Mapping):
                raise ProbeError("plan model entry is invalid")
            model_id = item.get("id")
            target = item.get("target")
            expected_size = item.get("size_bytes")
            expected_sha = item.get("sha256")
            if not isinstance(model_id, str) or not model_id:
                raise ProbeError("model id is missing")
            if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
                raise ProbeError(f"model {model_id} expected size is invalid")
            if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
                raise ProbeError(f"model {model_id} expected SHA256 is invalid")
            path = _model_path(workspace, target)
            if time.time() >= deadline:
                raise ProbeError("deadline expired before model hashing")
            actual_size = path.stat().st_size
            actual_sha = _sha_file(path, deadline_epoch=deadline)
            if actual_size != expected_size or actual_sha.lower() != expected_sha.lower():
                raise ProbeError(f"model {model_id} bytes differ from the source pin")
            models.append(
                {
                    "id": model_id,
                    "target": target,
                    "size_bytes": expected_size,
                    "sha256": expected_sha.lower(),
                    "actual_size_bytes": actual_size,
                    "actual_sha256": actual_sha,
                }
            )

        torch_facts = _torch_facts()
        comfy_facts = _comfy_facts(comfy_dir, plan.get("expected_comfy_commit"))
        pip_facts = _pip_facts(deadline)
        object_info = _object_info_facts(
            object_info_path,
            preflight_path=preflight_path,
            class_names=plan.get("required_classes"),
        )
        system_stats = _system_stats_facts(
            system_stats_path, required=require_system_stats
        )
        graph = _graph_facts(plan, object_info)
        if object_info.get("validation_issues"):
            raise ProbeError("object_info validation failed")
        if object_info.get("missing_classes"):
            raise ProbeError("object_info is missing required classes")
        dynamic = object_info.get("dynamic_combo")
        if not isinstance(dynamic, Mapping) or dynamic.get("issues") or dynamic.get("selected") != {
            "format": "mp4",
            "codec": "h264",
            "encoding": "auto",
        }:
            raise ProbeError("object_info nested SaveVideo DynamicCombo facts are invalid")
        if graph.get("missing_graph_classes"):
            raise ProbeError("workflow graph classes are missing from object_info")
        if any(item.get("present") is not True for item in graph.get("input_checks", [])):
            raise ProbeError("workflow graph inputs are missing from object_info")
        if time.time() >= deadline:
            raise ProbeError("deadline expired before evidence was emitted")

        host_key = plan.get("host_key")
        if not isinstance(host_key, Mapping):
            raise ProbeError("plan host_key is missing")
        payload = {
            "schema": SCHEMA,
            "run_id": plan.get("run_id"),
            "nonce": plan.get("nonce"),
            "session_binding_sha256": plan.get("session_binding_sha256"),
            "source_bundle_sha256": plan.get("source_bundle_sha256"),
            "source_approval_sha256": plan.get("source_approval_sha256"),
            "host_key": dict(host_key),
            "python": {
                "version": platform.python_version(),
                "executable": str(Path(sys.executable)),
                "prefix": str(sys.prefix),
            },
            "torch": torch_facts,
            "comfyui": comfy_facts,
            "pip": pip_facts,
            "models": models,
            "object_info": object_info,
            "graph": graph,
            **({"system_stats": system_stats} if system_stats is not None else {}),
            "collector": {
                "sha256": _sha_file(Path(__file__)),
                "plan_sha256": plan.get("_plan_sha256"),
            },
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "deadline_epoch": deadline,
            "provenance": {
                "kind": "owned_ssh_collector",
                "hardware_attestation": False,
                "model_bytes_hashed_in_place": True,
            },
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    except (ProbeError, OSError, subprocess.SubprocessError) as exc:
        print(f"runtime probe failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
