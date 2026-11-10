"""Actual remote runtime probe for I1.

Runs under the owned lexical venv.  It reads only a machine plan, hashes model
bytes in place, validates the exact API workflow against object_info, and emits
compact facts.  It never imports Clore/session policy and never POSTs /prompt.
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

try:
    import i1_profile as profile_module
    import i1_util as util
    import preflight
except ImportError:  # pragma: no cover
    from scripts.i1 import profile as profile_module  # type: ignore
    from scripts.i1 import util  # type: ignore
    from scripts.p1 import preflight  # type: ignore

SCHEMA = "anime-video-studio/i1-runtime-evidence/v1"
PLAN_SCHEMA = "anime-video-studio/i1-runtime-plan/v1"
MAX_JSON_BYTES = 16 * 1024 * 1024


class ProbeError(Exception):
    pass


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path, *, deadline: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if time.time() >= deadline:
                raise ProbeError("deadline expired while hashing")
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: list[str], *, deadline: float, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    remaining = deadline - time.time()
    if remaining <= 0:
        raise ProbeError("deadline expired before subprocess")
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=min(timeout, max(0.05, remaining)),
        stdin=subprocess.DEVNULL,
    )


def _inside(path: Path, root: Path, name: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ProbeError(f"{name} escapes its root")
    return resolved


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("cannot read runtime plan") from exc
    if not isinstance(data, dict) or data.get("schema") != PLAN_SCHEMA:
        raise ProbeError("runtime plan schema is invalid")
    data["_plan_sha256"] = _sha_bytes(raw)
    return data


def _decode(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout or ""


def _torch_facts(deadline: float) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise ProbeError(f"torch import failed: {type(exc).__name__}") from exc
    version = str(getattr(torch, "__version__", ""))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not version or not cuda_version:
        raise ProbeError("torch/CUDA version is unavailable")
    if not torch.cuda.is_available():
        raise ProbeError("torch.cuda.is_available() is false")
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        deadline=deadline,
    )
    if smi.returncode != 0:
        raise ProbeError("nvidia-smi failed")
    devices = []
    for line in _decode(smi).splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 3:
            try:
                memory = int(float(parts[1])) * 1024 * 1024
            except ValueError:
                memory = 0
            devices.append({"name": parts[0], "memory_bytes": memory, "driver": parts[2]})
    if not devices:
        raise ProbeError("nvidia-smi reported no GPU")
    return {
        "version": version,
        "cuda": str(cuda_version),
        "available": True,
        "devices": devices,
    }


def _comfy_module_origin(module: Any, checkout: Path) -> Path:
    """Accept the pinned package, including ComfyUI's native namespace package."""
    expected = checkout / "comfy"
    resolved = expected.resolve()
    if resolved != expected or not expected.is_dir():
        raise ProbeError("foreign ComfyUI module origin")
    filename = getattr(module, "__file__", None)
    if filename is not None:
        origin = Path(filename).resolve()
        if origin != expected / "__init__.py" or not origin.is_file():
            raise ProbeError("foreign ComfyUI module origin")
        return origin
    spec = getattr(module, "__spec__", None)
    paths = list(getattr(module, "__path__", ()))
    spec_paths = list(getattr(spec, "submodule_search_locations", ()) or ())
    if (
        getattr(spec, "origin", None) is not None
        or (expected / "__init__.py").exists()
        or len(paths) != 1
        or len(spec_paths) != 1
        or Path(paths[0]).resolve() != expected
        or Path(spec_paths[0]).resolve() != expected
    ):
        raise ProbeError("foreign ComfyUI namespace origin")
    return expected


def _comfy_facts(comfy_dir: Path, expected_commit: Any, deadline: float) -> dict[str, Any]:
    if not isinstance(expected_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit):
        raise ProbeError("expected ComfyUI commit is invalid")
    if not comfy_dir.is_dir():
        raise ProbeError("ComfyUI directory is missing")
    checkout = comfy_dir.resolve()
    top = _run(["git", "-C", str(checkout), "rev-parse", "--show-toplevel"], deadline=deadline)
    if top.returncode != 0 or Path(_decode(top).strip()).resolve() != checkout:
        raise ProbeError("ComfyUI is not a verified checkout root")
    head = _run(["git", "-C", str(checkout), "rev-parse", "HEAD"], deadline=deadline)
    actual = _decode(head).strip()
    if head.returncode != 0 or actual != expected_commit:
        raise ProbeError("actual ComfyUI commit differs from the pin")
    dirty = _run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        deadline=deadline,
    )
    if dirty.returncode != 0 or _decode(dirty).strip():
        raise ProbeError("ComfyUI has modified tracked files")
    old_cwd = Path.cwd()
    sys.path.insert(0, str(checkout))
    try:
        os.chdir(checkout)
        try:
            import comfy
            import nodes
        except Exception as exc:
            raise ProbeError(f"ComfyUI import failed: {type(exc).__name__}") from exc
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(checkout))
        except ValueError:
            pass
    comfy_origin = _comfy_module_origin(comfy, checkout)
    nodes_origin = Path(getattr(nodes, "__file__", "")).resolve()
    if checkout not in comfy_origin.parents or nodes_origin != (checkout / "nodes.py").resolve():
        raise ProbeError("foreign ComfyUI/nodes module origin")
    return {
        "commit": actual,
        "import_ok": True,
        "checkout_root": str(checkout),
        "comfy_module": str(comfy_origin),
        "nodes_module": str(nodes_origin),
    }


def _pip_facts(deadline: float) -> dict[str, Any]:
    check = _run([sys.executable, "-m", "pip", "check"], deadline=deadline)
    if check.returncode != 0:
        raise ProbeError("pip check failed")
    freeze = _run([sys.executable, "-m", "pip", "freeze"], deadline=deadline)
    if freeze.returncode != 0:
        raise ProbeError("pip freeze failed")
    text = _decode(freeze).replace("\r\n", "\n")
    return {
        "check_exit": check.returncode,
        "check_stdout_sha256": _sha_bytes((check.stdout or "").encode("utf-8")),
        "freeze_sha256": _sha_bytes(text.encode("utf-8")),
        "freeze_line_count": len([line for line in text.splitlines() if line.strip()]),
    }


def _input_type(spec: Mapping[str, Any], name: str) -> Any:
    raw = spec.get("input")
    if not isinstance(raw, Mapping):
        return None
    for group in ("required", "optional"):
        values = raw.get(group)
        if isinstance(values, Mapping) and name in values:
            value = values[name]
            if isinstance(value, list) and value:
                return value[0]
            return None
    return None


def _input_enum(spec: Mapping[str, Any], name: str) -> list[Any] | None:
    raw = spec.get("input")
    if not isinstance(raw, Mapping):
        return None
    for group in ("required", "optional"):
        values = raw.get(group)
        if isinstance(values, Mapping) and name in values:
            value = values[name]
            if isinstance(value, list) and value and isinstance(value[0], list):
                return list(value[0])
            return None
    return None


def _input_definition(
    spec: Mapping[str, Any], name: str
) -> tuple[Any, list[Any] | None, dict[str, Any]]:
    """Return ``(type, enum, options)`` for an actual object_info input."""

    raw = spec.get("input")
    if not isinstance(raw, Mapping):
        return None, None, {}
    for group in ("required", "optional"):
        values = raw.get(group)
        if isinstance(values, Mapping) and name in values:
            value = values[name]
            if not isinstance(value, list) or not value:
                return None, None, {}
            head = value[0]
            options = value[1] if len(value) > 1 and isinstance(value[1], Mapping) else {}
            if isinstance(head, list):
                return "COMBO", list(head), dict(options)
            return head, None, dict(options)
    return None, None, {}


def _check_literal_range(
    value: Any, options: Mapping[str, Any], *, class_name: Any, input_name: Any
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    minimum = options.get("min")
    maximum = options.get("max")
    if isinstance(minimum, (int, float)) and value < minimum:
        raise ProbeError(
            f"object_info range mismatch for {class_name}.{input_name}: {value} < {minimum}"
        )
    if isinstance(maximum, (int, float)) and value > maximum:
        raise ProbeError(
            f"object_info range mismatch for {class_name}.{input_name}: {value} > {maximum}"
        )


def _output_type(spec: Mapping[str, Any], slot: int) -> Any:
    output = spec.get("output")
    if not isinstance(output, list) or slot >= len(output):
        return None
    value = output[slot]
    if isinstance(value, list) and value:
        return value[0]
    return value


def _output_checks(
    data: Mapping[str, Any], output_specs: Any
) -> list[dict[str, Any]]:
    if not isinstance(output_specs, list) or not output_specs:
        raise ProbeError("plan output_specs is missing")
    result: list[dict[str, Any]] = []
    for spec in output_specs:
        if not isinstance(spec, Mapping):
            raise ProbeError("output spec is malformed")
        class_name = spec.get("class")
        if class_name not in {"SaveImage", "SaveVideo"}:
            raise ProbeError("output spec class is invalid")
        class_spec = data.get(class_name)
        if not isinstance(class_spec, Mapping):
            raise ProbeError(f"object_info is missing output class {class_name}")
        inputs = preflight._input_names(class_spec)
        if class_name == "SaveImage":
            present = "images" in inputs and "filename_prefix" in inputs
            if not present:
                raise ProbeError("SaveImage schema is missing images/filename_prefix")
            result.append(
                {
                    "workflow_id": spec.get("workflow_id"),
                    "class": class_name,
                    "input": "images",
                    "present": True,
                    "dynamic": False,
                }
            )
            continue
        savevideo = preflight._savevideo_dynamic_facts(data)
        if savevideo.get("issues"):
            raise ProbeError("SaveVideo dynamic schema is invalid")
        if savevideo.get("selected") != {"format": "mp4", "codec": "h264", "encoding": "auto"}:
            raise ProbeError("SaveVideo nested mp4/h264/auto path is missing")
        result.append(
            {
                "workflow_id": spec.get("workflow_id"),
                "class": class_name,
                "input": "format",
                "present": True,
                "dynamic": True,
                "selected": savevideo.get("selected"),
            }
        )
    return result


def _object_info_facts(
    path: Path,
    *,
    required_classes: list[str],
    required_inputs: Mapping[str, list[str]],
    graph_inputs: list[Mapping[str, Any]],
    output_specs: Any,
    deadline: float,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError("object_info file is missing/unsafe")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_JSON_BYTES:
        raise ProbeError("object_info size is invalid")
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("object_info is not JSON") from exc
    if not isinstance(data, dict):
        raise ProbeError("object_info must be an object")
    missing = [name for name in required_classes if not isinstance(data.get(name), Mapping)]
    if missing:
        raise ProbeError(f"object_info is missing classes: {', '.join(missing)}")
    input_checks = []
    for class_name, names in required_inputs.items():
        spec = data.get(class_name)
        if not isinstance(spec, Mapping):
            raise ProbeError(f"object_info class is missing: {class_name}")
        existing = preflight._input_names(spec)
        for name in names:
            input_checks.append(
                {
                    "class": class_name,
                    "input": name,
                    "type": _input_type(spec, name),
                    "present": name in existing,
                }
            )
    link_checks = []
    for item in graph_inputs:
        if not isinstance(item, Mapping) or item.get("kind") != "link":
            continue
        target_class = item.get("class")
        source_class = item.get("source_class")
        target_spec = data.get(target_class)
        source_spec = data.get(source_class)
        if not isinstance(target_spec, Mapping) or not isinstance(source_spec, Mapping):
            raise ProbeError("graph link class is missing from object_info")
        target_type = _input_type(target_spec, str(item.get("input")))
        source_type = _output_type(source_spec, int(item.get("source_slot", -1)))
        link_checks.append(
            {
                "node_id": item.get("node_id"),
                "input": item.get("input"),
                "source_class": source_class,
                "source_slot": item.get("source_slot"),
                "target_type": target_type,
                "source_type": source_type,
                "compatible": bool(target_type)
                and bool(source_type)
                and (
                    target_type == source_type
                    or str(target_type) == "*"
                    or str(source_type) == "*"
                ),
            }
        )
    if any(check["compatible"] is not True for check in link_checks):
        raise ProbeError("workflow linked input/slot types are incompatible with object_info")
    job_binding_required: list[dict[str, Any]] = []
    for item in graph_inputs:
        if not isinstance(item, Mapping) or item.get("kind") != "literal" or "value" not in item:
            continue
        target_spec = data.get(str(item.get("class")))
        if not isinstance(target_spec, Mapping):
            continue
        value = item.get("value")
        input_name = str(item.get("input"))
        class_name = item.get("class")
        _kind, allowed, options = _input_definition(target_spec, input_name)
        if allowed is not None and value not in allowed:
            # A clean worker's LoadImage enum is only the currently present
            # files and may legitimately not contain the bound upload yet.  The
            # schema capability + a trusted upload receipt authorize the job;
            # the template filename is never treated as an installed file.
            if (
                str(class_name) == "LoadImage"
                and input_name == "image"
                and options.get("image_upload") is True
            ):
                job_binding_required.append(
                    {
                        "node_id": item.get("node_id"),
                        "class": class_name,
                        "input": input_name,
                        "value": value,
                        "status": "JOB_BINDING_REQUIRED",
                        "reason": "dynamic_image_upload",
                    }
                )
                continue
            raise ProbeError(
                "object_info enum mismatch for "
                f"{item.get('class')}.{item.get('input')}"
            )
        _check_literal_range(
            value, options, class_name=class_name, input_name=input_name
        )
    output_checks = _output_checks(data, output_specs)
    result = {
        "sha256": _sha_bytes(raw),
        "size_bytes": len(raw),
        "classes": required_classes,
        "input_checks": input_checks,
        "link_checks": link_checks,
        "output_checks": output_checks,
        "job_binding_required": job_binding_required,
    }
    if any(
        isinstance(spec, Mapping) and spec.get("class") == "SaveVideo"
        for spec in output_specs
    ):
        result["savevideo_dynamic"] = preflight._savevideo_dynamic_facts(data)
    return result


def _system_stats_facts(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError("system_stats file is missing/unsafe")
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ProbeError("system_stats size is invalid")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("system_stats is not JSON") from exc
    if not isinstance(data, dict):
        raise ProbeError("system_stats must be an object")
    devices = []
    raw_devices = data.get("devices")
    if isinstance(raw_devices, list):
        for item in raw_devices:
            if not isinstance(item, Mapping):
                continue
            memory = (
                item.get("vram_total")
                or item.get("memory_total")
                or item.get("vram_total_bytes")
                or item.get("total_memory")
            )
            devices.append(
                {
                    "name": item.get("name"),
                    "type": item.get("type") or item.get("device_type"),
                    "index": item.get("index"),
                    "vram_total": memory,
                }
            )
    if not devices:
        raise ProbeError("system_stats reports no CUDA device")
    system = data.get("system") if isinstance(data.get("system"), Mapping) else {}
    return {
        "sha256": _sha_bytes(raw),
        "devices": devices,
        "comfyui_version": system.get("comfyui_version"),
    }


def _workflow_facts(
    path: Path,
    *,
    expected_sha: str,
    required_classes: list[str],
    required_inputs: Mapping[str, list[str]],
    node_classes: Mapping[str, str],
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError("API workflow is missing/unsafe")
    raw = path.read_bytes()
    actual = _sha_bytes(raw)
    if actual != expected_sha:
        raise ProbeError("API workflow hash differs from the profile")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("API workflow is not JSON") from exc
    if not isinstance(data, dict) or not data:
        raise ProbeError("API workflow must be a non-empty object")
    present_classes = []
    for node_id, expected_class in node_classes.items():
        node = data.get(node_id)
        if not isinstance(node, Mapping) or node.get("class_type") != expected_class:
            raise ProbeError(f"API workflow node {node_id} class mismatch")
    present_classes = sorted({str(node.get("class_type")) for node in data.values() if isinstance(node, Mapping)})
    if sorted(present_classes) != sorted(required_classes):
        raise ProbeError("API workflow classes differ from the profile")
    for node_id, node in data.items():
        if not isinstance(node, Mapping) or not isinstance(node.get("inputs"), Mapping):
            raise ProbeError(f"API workflow node {node_id} is malformed")
        for input_name, value in node["inputs"].items():
            if isinstance(value, list) and (
                len(value) != 2 or value[0] not in data or not isinstance(value[1], int)
            ):
                raise ProbeError(f"API workflow edge {node_id}.{input_name} is malformed")
    return {
        "sha256": actual,
        "classes": present_classes,
        "required_inputs": dict(required_inputs),
        "node_count": len(data),
    }


def _model_facts(
    plan: Mapping[str, Any], *, comfy_dir: Path, deadline: float
) -> list[dict[str, Any]]:
    models = plan.get("models")
    expected_count = plan.get("expected_model_count", 4)
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ProbeError("plan expected_model_count is invalid")
    if not isinstance(models, list) or len(models) != expected_count:
        raise ProbeError(f"plan must pin exactly {expected_count} models")
    result = []
    for item in models:
        if not isinstance(item, Mapping):
            raise ProbeError("plan model entry is invalid")
        target = item.get("target")
        if not isinstance(target, str):
            raise ProbeError("plan model target is missing")
        relative = PurePosixPath(target.replace("\\", "/"))
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ProbeError("plan model target is unsafe")
        path = _inside(comfy_dir / Path(*relative.parts), comfy_dir / "models", "model")
        if not path.is_file() or path.is_symlink():
            raise ProbeError(f"model file is missing/unsafe: {target}")
        expected_size = item.get("size_bytes")
        expected_sha = item.get("sha256")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
            raise ProbeError("plan model size is invalid")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
            raise ProbeError("plan model hash is invalid")
        actual_size = path.stat().st_size
        actual_sha = _sha_file(path, deadline=deadline)
        if actual_size != expected_size or actual_sha.lower() != expected_sha.lower():
            raise ProbeError(f"model bytes differ from the pin: {target}")
        result.append(
            {
                "id": item.get("id"),
                "target": target,
                "size_bytes": expected_size,
                "sha256": expected_sha.lower(),
                "actual_size_bytes": actual_size,
                "actual_sha256": actual_sha,
            }
        )
    return result


def run(plan_path: Path) -> dict[str, Any]:
    plan = _read_plan(plan_path)
    deadline = plan.get("deadline_epoch")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise ProbeError("plan deadline is missing")
    deadline = float(deadline)
    if time.time() >= deadline:
        raise ProbeError("deadline expired before probe")
    workspace = Path(str(plan.get("workspace", "")))
    owned_root = Path(str(plan.get("owned_root", "")))
    comfy_dir = Path(str(plan.get("comfy_dir", "")))
    venv_python = Path(str(plan.get("venv_python", "")))
    profile_path = Path(str(plan.get("profile_path", "")))
    object_info_path = Path(str(plan.get("object_info_path", "")))
    system_stats_path = Path(str(plan.get("system_stats_path", "")))
    workflow_specs = plan.get("workflows")
    if not isinstance(workflow_specs, list) or not workflow_specs:
        workflow_specs = [
            {
                "workflow_id": "wan",
                "path": plan.get("workflow_api_path"),
                "sha256": plan.get("workflow_sha256"),
                "required_classes": plan.get("required_classes"),
                "required_inputs": plan.get("required_inputs"),
                "node_classes": plan.get("node_classes"),
                "graph_inputs": plan.get("graph_inputs"),
                "output_specs": plan.get("output_specs")
                or [
                    {
                        "workflow_id": "wan",
                        "class": "SaveVideo",
                        "node_id": None,
                        "keys": ["images"],
                        "media_type": "video/mp4",
                        "dynamic": True,
                    }
                ],
            }
        ]
    paths = [
        ("workspace", workspace),
        ("owned_root", owned_root),
        ("comfy_dir", comfy_dir),
        ("venv_python", venv_python),
        ("profile_path", profile_path),
        ("object_info_path", object_info_path),
        ("system_stats_path", system_stats_path),
    ]
    for spec in workflow_specs:
        if not isinstance(spec, Mapping):
            raise ProbeError("plan workflow spec is malformed")
        spec_path = Path(str(spec.get("path", "")))
        if not spec_path.is_absolute():
            raise ProbeError("plan workflow path must be absolute")
        paths.append((f"workflow[{spec.get('workflow_id')}]", spec_path))
    for label, path in paths:
        if not path.is_absolute():
            raise ProbeError(f"plan {label} must be absolute")
    _inside(profile_path, owned_root, "profile_path")
    _inside(object_info_path, workspace, "object_info_path")
    _inside(system_stats_path, workspace, "system_stats_path")
    _inside(comfy_dir, workspace, "comfy_dir")
    for spec in workflow_specs:
        _inside(Path(str(spec["path"])), owned_root, "workflow_api_path")
    if Path(sys.executable).resolve() != venv_python.resolve():
        raise ProbeError("probe is not executing under the planned venv python")
    venv_root = venv_python.parent.parent
    if Path(sys.prefix).resolve() != venv_root.resolve():
        raise ProbeError("probe sys.prefix is not the owned venv")
    profile = profile_module.load_profile(profile_path)
    if profile.sha256 != plan.get("profile_sha256"):
        raise ProbeError("profile hash differs from plan")
    expected_commit = plan.get("expected_comfy_commit")
    models = _model_facts(plan, comfy_dir=comfy_dir, deadline=deadline)
    torch_facts = _torch_facts(deadline)
    comfy_facts = _comfy_facts(comfy_dir, expected_commit, deadline)
    pip_facts = _pip_facts(deadline)
    system_stats = _system_stats_facts(system_stats_path)
    workflow_evidence = []
    primary_workflow = None
    primary_object_info = None
    for spec in workflow_specs:
        required_classes = spec.get("required_classes")
        required_inputs = spec.get("required_inputs")
        graph_inputs = spec.get("graph_inputs")
        node_classes = spec.get("node_classes")
        if not isinstance(required_classes, list) or not isinstance(required_inputs, Mapping):
            raise ProbeError("plan required graph facts are missing")
        if not isinstance(graph_inputs, list):
            raise ProbeError("plan graph_inputs is missing")
        if not isinstance(node_classes, Mapping):
            raise ProbeError("plan node_classes is missing")
        output_specs = spec.get("output_specs")
        object_info = _object_info_facts(
            object_info_path,
            required_classes=required_classes,
            required_inputs=required_inputs,
            graph_inputs=graph_inputs,
            output_specs=output_specs,
            deadline=deadline,
        )
        if any(check.get("present") is not True for check in object_info["input_checks"]):
            raise ProbeError("object_info is missing workflow input types")
        workflow = _workflow_facts(
            Path(str(spec["path"])),
            expected_sha=str(spec.get("sha256") or ""),
            required_classes=required_classes,
            required_inputs=required_inputs,
            node_classes=node_classes,
        )
        item = {
            "workflow_id": str(spec.get("workflow_id") or "wan"),
            "sha256": workflow["sha256"],
            "classes": workflow["classes"],
            "required_inputs": workflow["required_inputs"],
            "node_count": workflow["node_count"],
            "input_checks": object_info["input_checks"],
            "link_checks": object_info["link_checks"],
            "output_checks": object_info["output_checks"],
            "job_binding_required": object_info.get("job_binding_required", []),
        }
        workflow_evidence.append(item)
        if primary_workflow is None:
            primary_workflow = workflow
            primary_object_info = object_info
    if primary_workflow is None or primary_object_info is None:
        raise ProbeError("no workflow evidence was produced")
    if time.time() >= deadline:
        raise ProbeError("deadline expired before evidence emission")
    return {
        "schema": SCHEMA,
        "run_id": plan.get("run_id"),
        "nonce": plan.get("nonce"),
        "profile_sha256": profile.sha256,
        "workflow_sha256": primary_workflow["sha256"],
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable)),
            "prefix": str(Path(sys.prefix)),
            "venv_python": str(venv_python),
        },
        "torch": torch_facts,
        "comfyui": comfy_facts,
        "pip": pip_facts,
        "models": models,
        "object_info": primary_object_info,
        "workflow": primary_workflow,
        "workflows": workflow_evidence,
        "system_stats": system_stats,
        "collector": {"sha256": _sha_file(Path(__file__), deadline=deadline), "plan_sha256": plan.get("_plan_sha256")},
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {"kind": "owned_ssh_probe", "model_bytes_hashed_in_place": True, "no_prompt_post": True},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="i1_probe.py")
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    try:
        payload = run(Path(args.plan))
    except (ProbeError, OSError, subprocess.SubprocessError) as exc:
        print(f"i1 probe failed: {util.redact(str(exc), limit=600)}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

