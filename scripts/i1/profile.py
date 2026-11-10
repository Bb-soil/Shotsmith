"""Explicit adapter from the pinned 14B source inventory to an I1 profile.

The original inventory is a source-of-truth document with its own schema.  I1
does not reinterpret it as a 5B manifest; it derives a separate, checked profile
whose hash is pinned in the local connection record and in every remote job.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

try:  # stageable as a top-level module inside the owned remote root
    from .util import (
        HEX40_RE,
        HEX64_RE,
        canonical_bytes,
        sha256_bytes,
        sha256_file,
        sha256_mapping,
        validate_name,
        validate_posix_absolute,
        validate_relative_posix,
    )
except ImportError:  # pragma: no cover - exercised by staged self-checks
    from i1_util import (  # type: ignore
        HEX40_RE,
        HEX64_RE,
        canonical_bytes,
        sha256_bytes,
        sha256_file,
        sha256_mapping,
        validate_name,
        validate_posix_absolute,
        validate_relative_posix,
    )

INVENTORY_SCHEMA = "anime-studio/ab-source-inventory/v1"
PROFILE_SCHEMA = "anime-video-studio/i1-profile/v1"
PROFILE_SCHEMA_V2 = "anime-video-studio/i1-profile/v2"
COMFY_COMMIT = "7a28332047b6af638dc6c7f72ef3481a718769fd"
COMFY_SOURCE = "https://github.com/Comfy-Org/ComfyUI"
BASE_IMAGE = "runpod/pytorch:1.4.1-cu1281-torch291-ubuntu2404"
GIB = 1024**3

# Only these explicit, source-derived profiles may be validated.  There is no
# global "any count / any graph" switch; an unknown profile id is rejected.
KNOWN_PROFILES: dict[str, dict[str, Any]] = {
    "wan22-14b-ab": {
        "schema": PROFILE_SCHEMA,
        "models": 4,
        "role_counts": {"diffusion": 2, "text_encoder": 1, "vae": 1},
        "workflows": ("wan",),
        "outputs": {"SaveVideo"},
        "total_bytes": 35_579_207_879,
    },
    "keyframes-qwen-edit-2511": {
        "schema": PROFILE_SCHEMA_V2,
        "models": 3,
        "role_counts": {"diffusion": 1, "text_encoder": 1, "vae": 1},
        "workflows": ("qwen-1", "qwen-2", "qwen-3"),
        "outputs": {"SaveImage"},
        "total_bytes": 30_172_239_743,
    },
    "bundle-wan22-qwen-edit-2511": {
        "schema": PROFILE_SCHEMA_V2,
        "models": 7,
        "role_counts": {"diffusion": 3, "text_encoder": 2, "vae": 2},
        "workflows": ("wan", "qwen-1", "qwen-2", "qwen-3"),
        "outputs": {"SaveVideo", "SaveImage"},
        "total_bytes": 65_751_447_622,
    },
}


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    data: dict[str, Any]
    sha256: str

    @property
    def profile_id(self) -> str:
        return str(self.data["profile_id"])

    @property
    def models(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data["models"])

    @property
    def workflow(self) -> dict[str, Any]:
        if self.data.get("schema") == PROFILE_SCHEMA_V2:
            workflows = self.data["workflows"]
            return workflows[str(self.data["primary_workflow"])]
        return self.data["workflow"]

    @property
    def workflows(self) -> dict[str, dict[str, Any]]:
        if self.data.get("schema") == PROFILE_SCHEMA_V2:
            return {
                str(key): dict(value)
                for key, value in self.data["workflows"].items()
            }
        return {"wan": dict(self.data["workflow"])}

    def workflow_items(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        return tuple(sorted(self.workflows.items()))

    @property
    def primary_workflow_id(self) -> str:
        if self.data.get("schema") == PROFILE_SCHEMA_V2:
            return str(self.data["primary_workflow"])
        return "wan"

    @property
    def output_specs(self) -> tuple[dict[str, Any], ...]:
        if self.data.get("schema") == PROFILE_SCHEMA_V2:
            result = []
            for workflow_id, workflow in self.workflow_items():
                output = dict(workflow["output"])
                output["workflow_id"] = workflow_id
                result.append(output)
            return tuple(result)
        return (
            {
                "workflow_id": "wan",
                "class": "SaveVideo",
                "node_id": None,
                "keys": ["images"],
                "media_type": "video/mp4",
                "dynamic": True,
            },
        )

    @property
    def model_policy(self) -> dict[str, Any]:
        if self.data.get("schema") == PROFILE_SCHEMA_V2:
            return dict(self.data["models_policy"])
        return {
            "mode": "legacy-v1",
            "expected_count": 4,
            "allowed_roles": ["diffusion", "text_encoder", "vae"],
            "role_counts": {"diffusion": 2, "text_encoder": 1, "vae": 1},
        }

    @property
    def comfy(self) -> dict[str, Any]:
        return self.data["comfyui"]

    @property
    def required_classes(self) -> tuple[str, ...]:
        if self.data.get("schema") == PROFILE_SCHEMA_V2:
            return tuple(self.workflow["required_classes"])
        return tuple(self.data["required_classes"])

    @property
    def required_inputs(self) -> dict[str, list[str]]:
        if self.data.get("schema") == PROFILE_SCHEMA_V2:
            return {
                key: list(value)
                for key, value in self.workflow["required_inputs"].items()
            }
        return {key: list(value) for key, value in self.data["required_inputs"].items()}

    @property
    def disk_required_bytes(self) -> int:
        models = sum(int(item["size_bytes"]) for item in self.models)
        return models + int(self.data["disk"]["install_overhead_bytes"]) + int(
            self.data["disk"]["reserve_bytes"]
        )

    def model_bytes(self) -> int:
        return sum(int(item["size_bytes"]) for item in self.models)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ProfileError(f"{path} must be a JSON object")
    return value


def _validate_https(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ProfileError(f"{name} must be an https URL")
    if any(char in value for char in ("\x00", "\n", "\r", " ", "\t")):
        raise ProfileError(f"{name} contains whitespace/control characters")
    if "@" in value.split("/", 3)[2]:
        raise ProfileError(f"{name} must not contain userinfo")
    return value


def target_from_inventory_path(path: Any) -> str:
    validate_relative_posix(path, "inventory model path")
    pure = PurePosixPath(str(path))
    if len(pure.parts) < 2 or pure.parts[0] != "split_files":
        raise ProfileError("inventory model path must start with split_files/")
    relative = PurePosixPath(*pure.parts[1:])
    target = PurePosixPath("models") / relative
    if len(relative.parts) < 2:
        raise ProfileError("inventory model path must include a category and file")
    return str(target)


def _inventory_models(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ProfileError("inventory schema is not the pinned 14B source inventory")
    models = inventory.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise ProfileError("inventory must contain exactly four 14B profile files")
    result: list[dict[str, Any]] = []
    targets: set[str] = set()
    ids: set[str] = set()
    for item in models:
        if not isinstance(item, Mapping):
            raise ProfileError("every inventory model entry must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            raise ProfileError("inventory model name is invalid")
        model_id = name.removesuffix(".safetensors")
        model_id = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")
        if not model_id:
            raise ProfileError("inventory model id is empty")
        if model_id in ids:
            raise ProfileError("inventory model ids must be unique")
        ids.add(model_id)
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ProfileError(f"{model_id}: size_bytes is invalid")
        checksum = item.get("expected_sha256")
        if not isinstance(checksum, str) or not HEX64_RE.fullmatch(checksum):
            raise ProfileError(f"{model_id}: expected_sha256 is invalid")
        target = target_from_inventory_path(item.get("path"))
        if target in targets:
            raise ProfileError("inventory model targets must be unique")
        targets.add(target)
        category = PurePosixPath(str(item["path"])).parts[1]
        role = {
            "diffusion_models": "diffusion",
            "vae": "vae",
            "text_encoders": "text_encoder",
        }.get(category, category)
        result.append(
            {
                "id": model_id,
                "name": name,
                "role": role,
                "repo_id": str(item.get("repo_id") or ""),
                "revision": str(item.get("revision") or ""),
                "source_path": str(item["path"]),
                "target": target,
                "size_bytes": size,
                "sha256": checksum.lower(),
                "url": _validate_https(item.get("url"), f"{model_id}.url"),
                "license": item.get("license_declared"),
                "license_source": item.get("license_source"),
                "binary_verified_at_derivation": False,
            }
        )
    if sum(int(item["size_bytes"]) for item in result) != int(
        inventory.get("total_bytes", -1)
    ):
        raise ProfileError("inventory total_bytes does not equal the model sum")
    return result


def _validate_api_workflow(
    data: Mapping[str, Any],
) -> tuple[
    list[str], dict[str, list[str]], dict[str, str], list[dict[str, Any]]
]:
    if not isinstance(data, Mapping) or not data:
        raise ProfileError("API workflow must be a non-empty object")
    nodes: dict[str, Mapping[str, Any]] = {}
    node_classes: dict[str, str] = {}
    for node_id, node in data.items():
        if str(node_id).startswith("_"):
            continue
        if not isinstance(node_id, str) or not node_id:
            raise ProfileError("API workflow node ids must be strings")
        if not isinstance(node, Mapping):
            raise ProfileError(f"workflow node {node_id} must be an object")
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(class_type, str) or not class_type:
            raise ProfileError(f"workflow node {node_id} lacks class_type")
        if not isinstance(inputs, Mapping):
            raise ProfileError(f"workflow node {node_id} lacks inputs")
        nodes[node_id] = node
        node_classes[node_id] = class_type
    if not nodes:
        raise ProfileError("API workflow has no nodes")
    if "TEMPLATE_NOT_EXECUTABLE" in json.dumps(data, ensure_ascii=False):
        raise ProfileError("template workflow is not installable")
    required_classes = sorted({node["class_type"] for node in nodes.values()})
    required_inputs: dict[str, set[str]] = {}
    graph_inputs: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        class_type = str(node["class_type"])
        for input_name, value in node["inputs"].items():
            name = str(input_name)
            if isinstance(value, list):
                if (
                    len(value) != 2
                    or not isinstance(value[0], str)
                    or value[0] not in nodes
                    or isinstance(value[1], bool)
                    or not isinstance(value[1], int)
                    or value[1] < 0
                ):
                    raise ProfileError(f"workflow edge {node_id}.{input_name} is malformed")
                required_inputs.setdefault(class_type, set()).add(name)
                graph_inputs.append(
                    {
                        "node_id": node_id,
                        "class": class_type,
                        "input": name,
                        "kind": "link",
                        "source_node": value[0],
                        "source_class": str(nodes[value[0]]["class_type"]),
                        "source_slot": int(value[1]),
                    }
                )
                continue
            if "." in name:
                # SaveVideo DynamicCombo child keys are checked separately by the probe.
                graph_inputs.append(
                    {
                        "node_id": node_id,
                        "class": class_type,
                        "input": name,
                        "kind": "dynamic",
                    }
                )
                continue
            required_inputs.setdefault(class_type, set()).add(name)
            graph_inputs.append(
                {
                    "node_id": node_id,
                    "class": class_type,
                    "input": name,
                    "kind": "literal",
                }
            )
    return (
        required_classes,
        {key: sorted(value) for key, value in sorted(required_inputs.items())},
        dict(sorted(node_classes.items())),
        sorted(graph_inputs, key=lambda item: (item["node_id"], item["input"])),
    )


def _validate_ui_workflow(data: Mapping[str, Any]) -> int:
    nodes = data.get("nodes")
    links = data.get("links")
    if not isinstance(nodes, list) or not nodes:
        raise ProfileError("UI workflow has no nodes")
    if not isinstance(links, list):
        raise ProfileError("UI workflow has no links list")
    ids: set[int] = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ProfileError("UI workflow node must be an object")
        node_id = node.get("id")
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id <= 0:
            raise ProfileError("UI workflow node id is invalid")
        if node_id in ids:
            raise ProfileError("UI workflow node ids must be unique")
        ids.add(node_id)
        if not isinstance(node.get("type"), str) or not node["type"]:
            raise ProfileError("UI workflow node type is invalid")
    for link in links:
        if not isinstance(link, list) or len(link) < 5:
            raise ProfileError("UI workflow link is malformed")
        if not isinstance(link[0], int) or link[1] not in ids or link[3] not in ids:
            raise ProfileError("UI workflow link endpoints are invalid")
    return len(nodes)


def derive_profile(
    *,
    inventory_path: Path | str,
    api_workflow_path: Path | str,
    ui_workflow_path: Path | str | None,
    profile_id: str = "wan22-14b-ab",
    display_name: str = "Wan 2.2 14B A→B (FP8, native FLF2V)",
    repo_root: Path | str | None = None,
) -> Profile:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    inv_path = Path(inventory_path)
    api_path = Path(api_workflow_path)
    inventory = _read_json(inv_path)
    api = _read_json(api_path)
    models = _inventory_models(inventory)
    required_classes, required_inputs, node_classes, graph_inputs = _validate_api_workflow(api)
    comfy_commit = inventory.get("comfy_commit")
    if comfy_commit != COMFY_COMMIT:
        raise ProfileError("inventory ComfyUI commit is not the pinned 14B commit")
    workflow: dict[str, Any] = {
        "api_path": _relative_to(root, api_path),
        "api_sha256": sha256_file(api_path),
        "ui_path": None,
        "ui_sha256": None,
        "ui_status": "UI_GRAPH_NOT_VERIFIED",
    }
    if ui_workflow_path is not None:
        ui_path = Path(ui_workflow_path)
        ui = _read_json(ui_path)
        _validate_ui_workflow(ui)
        workflow["ui_path"] = _relative_to(root, ui_path)
        workflow["ui_sha256"] = sha256_file(ui_path)
        workflow["ui_status"] = "DERIVED_FROM_API_NOT_LIVE_VERIFIED"
    data: dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "profile_id": validate_name(profile_id, "profile_id"),
        "display_name": display_name,
        "source_inventory": {
            "path": _relative_to(root, inv_path),
            "sha256": sha256_file(inv_path),
            "schema": INVENTORY_SCHEMA,
        },
        "comfyui": {
            "source_url": COMFY_SOURCE,
            "commit": comfy_commit,
            "expected_imports": ["comfy", "nodes"],
        },
        "base_torch": {
            "image": BASE_IMAGE,
            "expected_prefix": "2.9",
            "expected_cuda_prefix": "12.8",
            "never_reinstall": True,
        },
        "models": models,
        "workflow": workflow,
        "required_classes": required_classes,
        "required_inputs": required_inputs,
        "node_classes": node_classes,
        "graph_inputs": graph_inputs,
        "runtime": {
            "vram_target_bytes": 24 * GIB,
            "ram_target_bytes": 64 * GIB,
            "comfy_remote_port": 8188,
            "local_tunnel_port": 8189,
        },
        "disk": {
            "install_overhead_bytes": 6 * GIB,
            "reserve_bytes": 5 * GIB,
        },
    }
    return Profile(data=data, sha256=sha256_mapping(data))


def _relative_to(root: Path, path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def validate_profile(data: Mapping[str, Any]) -> Profile:
    if not isinstance(data, Mapping):
        raise ProfileError("profile must be an object")
    schema = data.get("schema")
    if schema == PROFILE_SCHEMA:
        return _validate_v1(data)
    if schema == PROFILE_SCHEMA_V2:
        return _validate_v2(data)
    raise ProfileError("profile schema is invalid")


def _validate_v1(data: Mapping[str, Any]) -> Profile:
    if data.get("schema") != PROFILE_SCHEMA:
        raise ProfileError("profile schema is invalid")
    profile_id = validate_name(data.get("profile_id"), "profile_id")
    if profile_id != data.get("profile_id"):
        raise ProfileError("profile_id is invalid")
    registry = KNOWN_PROFILES.get(profile_id)
    if registry is not None and registry.get("schema") != PROFILE_SCHEMA:
        raise ProfileError("profile_id is not a known v1 profile")
    comfy = data.get("comfyui")
    if not isinstance(comfy, Mapping) or comfy.get("commit") != COMFY_COMMIT:
        raise ProfileError("profile ComfyUI commit is not pinned")
    models = data.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise ProfileError("profile must pin exactly four models")
    seen: set[str] = set()
    total = 0
    for item in models:
        if not isinstance(item, Mapping):
            raise ProfileError("profile model must be an object")
        target = validate_relative_posix(item.get("target"), "model.target")
        if not target.startswith("models/"):
            raise ProfileError("model target must be under models/")
        if target in seen:
            raise ProfileError("profile model targets must be unique")
        seen.add(target)
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ProfileError("model size_bytes is invalid")
        checksum = item.get("sha256")
        if not isinstance(checksum, str) or not HEX64_RE.fullmatch(checksum):
            raise ProfileError("model sha256 is invalid")
        _validate_https(item.get("url"), f"{item.get('id')}.url")
        total += size
    workflow = data.get("workflow")
    if not isinstance(workflow, Mapping):
        raise ProfileError("profile workflow is missing")
    if not isinstance(workflow.get("api_path"), str) or not HEX64_RE.fullmatch(
        str(workflow.get("api_sha256"))
    ):
        raise ProfileError("workflow api pin is invalid")
    ui_path = workflow.get("ui_path")
    ui_sha = workflow.get("ui_sha256")
    if ui_path is not None:
        validate_relative_posix(ui_path, "workflow.ui_path")
        if not isinstance(ui_sha, str) or not HEX64_RE.fullmatch(ui_sha):
            raise ProfileError("workflow ui hash is invalid")
    required = data.get("required_classes")
    if not isinstance(required, list) or not required or not all(
        isinstance(item, str) and item for item in required
    ):
        raise ProfileError("required_classes is invalid")
    inputs = data.get("required_inputs")
    if not isinstance(inputs, Mapping):
        raise ProfileError("required_inputs is invalid")
    for class_name, names in inputs.items():
        if not isinstance(class_name, str) or not isinstance(names, list):
            raise ProfileError("required_inputs entry is invalid")
        if any(not isinstance(item, str) or not item for item in names):
            raise ProfileError("required_inputs name is invalid")
    graph_inputs = data.get("graph_inputs")
    if not isinstance(graph_inputs, list):
        raise ProfileError("graph_inputs is invalid")
    for item in graph_inputs:
        if not isinstance(item, Mapping):
            raise ProfileError("graph_inputs entry is invalid")
        for key in ("node_id", "class", "input", "kind"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ProfileError(f"graph_inputs.{key} is invalid")
        if item["kind"] == "link":
            for key in ("source_node", "source_class"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise ProfileError(f"graph_inputs.{key} is invalid")
            if isinstance(item.get("source_slot"), bool) or not isinstance(
                item.get("source_slot"), int
            ):
                raise ProfileError("graph_inputs.source_slot is invalid")
    runtime = data.get("runtime")
    disk = data.get("disk")
    if not isinstance(runtime, Mapping) or not isinstance(disk, Mapping):
        raise ProfileError("runtime/disk profile section is missing")
    if not isinstance(runtime.get("comfy_remote_port"), int):
        raise ProfileError("comfy_remote_port is invalid")
    if not isinstance(disk.get("install_overhead_bytes"), int) or not isinstance(
        disk.get("reserve_bytes"), int
    ):
        raise ProfileError("disk reserves are invalid")
    return Profile(data=dict(data), sha256=sha256_mapping(data))


def _validate_workflow_meta(workflow_id: str, meta: Any) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        raise ProfileError(f"workflow {workflow_id} must be an object")
    api_path = validate_relative_posix(meta.get("api_path"), f"workflow.{workflow_id}.api_path")
    if not HEX64_RE.fullmatch(str(meta.get("api_sha256", ""))):
        raise ProfileError(f"workflow {workflow_id} api hash is invalid")
    required = meta.get("required_classes")
    if not isinstance(required, list) or not required or not all(
        isinstance(item, str) and item for item in required
    ):
        raise ProfileError(f"workflow {workflow_id} required_classes is invalid")
    inputs = meta.get("required_inputs")
    if not isinstance(inputs, Mapping):
        raise ProfileError(f"workflow {workflow_id} required_inputs is invalid")
    for class_name, names in inputs.items():
        if not isinstance(class_name, str) or not isinstance(names, list):
            raise ProfileError(f"workflow {workflow_id} required_inputs entry is invalid")
        if any(not isinstance(item, str) or not item for item in names):
            raise ProfileError(f"workflow {workflow_id} required_inputs name is invalid")
    graph_inputs = meta.get("graph_inputs")
    if not isinstance(graph_inputs, list):
        raise ProfileError(f"workflow {workflow_id} graph_inputs is invalid")
    for item in graph_inputs:
        if not isinstance(item, Mapping):
            raise ProfileError(f"workflow {workflow_id} graph_inputs entry is invalid")
        for key in ("node_id", "class", "input", "kind"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ProfileError(f"workflow {workflow_id}.graph_inputs.{key} is invalid")
        if item["kind"] == "link":
            for key in ("source_node", "source_class"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise ProfileError(f"workflow {workflow_id}.graph_inputs.{key} is invalid")
            if isinstance(item.get("source_slot"), bool) or not isinstance(
                item.get("source_slot"), int
            ):
                raise ProfileError(f"workflow {workflow_id}.graph_inputs.source_slot is invalid")
    output = meta.get("output")
    if not isinstance(output, Mapping):
        raise ProfileError(f"workflow {workflow_id} output is invalid")
    if output.get("class") not in {"SaveImage", "SaveVideo"}:
        raise ProfileError(f"workflow {workflow_id} output class is invalid")
    keys = output.get("keys")
    if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ProfileError(f"workflow {workflow_id} output keys are invalid")
    if output.get("media_type") not in {"image/png", "video/mp4"}:
        raise ProfileError(f"workflow {workflow_id} output media_type is invalid")
    node_classes = meta.get("node_classes")
    if not isinstance(node_classes, Mapping) or not node_classes:
        raise ProfileError(f"workflow {workflow_id} node_classes is invalid")
    return {
        **dict(meta),
        "api_path": api_path,
        "required_classes": list(required),
        "required_inputs": {key: list(value) for key, value in inputs.items()},
        "graph_inputs": [dict(item) for item in graph_inputs],
        "output": dict(output),
        "node_classes": dict(node_classes),
    }


def _validate_v2(data: Mapping[str, Any]) -> Profile:
    profile_id = validate_name(data.get("profile_id"), "profile_id")
    if profile_id != data.get("profile_id"):
        raise ProfileError("profile_id is invalid")
    registry = KNOWN_PROFILES.get(profile_id)
    if not isinstance(registry, Mapping) or registry.get("schema") != PROFILE_SCHEMA_V2:
        raise ProfileError("profile_id is not a known v2 profile")
    comfy = data.get("comfyui")
    if not isinstance(comfy, Mapping) or comfy.get("commit") != COMFY_COMMIT:
        raise ProfileError("profile ComfyUI commit is not pinned")
    policy = data.get("models_policy")
    if not isinstance(policy, Mapping):
        raise ProfileError("models_policy is missing")
    if policy.get("mode") != "allowlist":
        raise ProfileError("models_policy.mode must be allowlist")
    expected_count = registry["models"]
    if policy.get("expected_count") != expected_count:
        raise ProfileError("models_policy.expected_count does not match the known profile")
    if policy.get("role_counts") != registry["role_counts"]:
        raise ProfileError("models_policy.role_counts does not match the known profile")
    allowed_roles = policy.get("allowed_roles")
    if not isinstance(allowed_roles, list) or sorted(allowed_roles) != ["diffusion", "text_encoder", "vae"]:
        raise ProfileError("models_policy.allowed_roles is invalid")
    models = data.get("models")
    if not isinstance(models, list) or len(models) != expected_count:
        raise ProfileError(f"profile must pin exactly {expected_count} models")
    seen: set[str] = set()
    ids: set[str] = set()
    roles: dict[str, int] = {}
    total = 0
    for item in models:
        if not isinstance(item, Mapping):
            raise ProfileError("profile model must be an object")
        model_id = item.get("id")
        if not isinstance(model_id, str) or not validate_name(model_id, "model.id") == model_id:
            raise ProfileError("model.id is invalid")
        if model_id in ids:
            raise ProfileError("model ids must be unique")
        ids.add(model_id)
        role = item.get("role")
        if role not in {"diffusion", "text_encoder", "vae"}:
            raise ProfileError("model role is not allowed")
        roles[str(role)] = roles.get(str(role), 0) + 1
        target = validate_relative_posix(item.get("target"), "model.target")
        if not target.startswith("models/"):
            raise ProfileError("model target must be under models/")
        if target in seen:
            raise ProfileError("profile model targets must be unique")
        seen.add(target)
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ProfileError("model size_bytes is invalid")
        checksum = item.get("sha256")
        if not isinstance(checksum, str) or not HEX64_RE.fullmatch(checksum):
            raise ProfileError("model sha256 is invalid")
        _validate_https(item.get("url"), f"{model_id}.url")
        total += size
    if roles != registry["role_counts"]:
        raise ProfileError("profile model role counts do not match the known profile")
    if total != registry["total_bytes"]:
        raise ProfileError("profile model total_bytes do not match the known profile")
    workflows = data.get("workflows")
    if not isinstance(workflows, Mapping):
        raise ProfileError("workflows is missing")
    if tuple(sorted(workflows)) != tuple(sorted(registry["workflows"])):
        raise ProfileError("workflow ids do not match the known profile")
    validated = {
        str(workflow_id): _validate_workflow_meta(str(workflow_id), meta)
        for workflow_id, meta in workflows.items()
    }
    primary = data.get("primary_workflow")
    if primary not in validated:
        raise ProfileError("primary_workflow is invalid")
    outputs = {str(meta["output"]["class"]) for meta in validated.values()}
    if not registry["outputs"].issubset(outputs):
        raise ProfileError("profile outputs do not match the known profile")
    runtime = data.get("runtime")
    disk = data.get("disk")
    if not isinstance(runtime, Mapping) or not isinstance(disk, Mapping):
        raise ProfileError("runtime/disk profile section is missing")
    if not isinstance(runtime.get("comfy_remote_port"), int):
        raise ProfileError("comfy_remote_port is invalid")
    if not isinstance(disk.get("install_overhead_bytes"), int) or not isinstance(
        disk.get("reserve_bytes"), int
    ):
        raise ProfileError("disk reserves are invalid")
    result = dict(data)
    result["workflows"] = validated
    return Profile(data=result, sha256=sha256_mapping(result))


def load_profile(path: Path | str) -> Profile:
    return validate_profile(_read_json(Path(path)))


def build_profile_from_files(
    *,
    repo_root: Path | str,
    inventory_path: Path | str,
    api_workflow_path: Path | str,
    ui_workflow_path: Path | str | None,
    profile_id: str = "wan22-14b-ab",
) -> Profile:
    return derive_profile(
        inventory_path=inventory_path,
        api_workflow_path=api_workflow_path,
        ui_workflow_path=ui_workflow_path,
        profile_id=profile_id,
        repo_root=repo_root,
    )


def verify_profile_sources(profile: Profile, repo_root: Path | str) -> None:
    root = Path(repo_root)
    checks: list[tuple[str, str]] = []
    inventory = profile.data.get("source_inventory")
    if isinstance(inventory, Mapping) and isinstance(inventory.get("path"), str):
        checks.append((inventory["path"], inventory["sha256"]))
    for workflow_id, workflow in profile.workflow_items():
        checks.append((workflow["api_path"], workflow["api_sha256"]))
    for workflow_id, workflow in profile.workflow_items():
        ui_path = workflow.get("ui_path")
        if ui_path:
            checks.append((ui_path, workflow["ui_sha256"]))
    for relative, expected in checks:
        path = root / relative
        if not path.is_file():
            raise ProfileError(f"profile source is missing: {relative}")
        if sha256_file(path) != expected:
            raise ProfileError(f"profile source hash mismatch: {relative}")


def download_specs(profile: Profile) -> list[dict[str, Any]]:
    return [
        {
            "model_id": item["id"],
            "url": item["url"],
            "target": item["target"],
            "size_bytes": int(item["size_bytes"]),
            "sha256": item["sha256"],
        }
        for item in profile.models
    ]


def write_profile(path: Path | str, profile: Profile) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(profile.data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "BASE_IMAGE",
    "COMFY_COMMIT",
    "COMFY_SOURCE",
    "INVENTORY_SCHEMA",
    "KNOWN_PROFILES",
    "PROFILE_SCHEMA",
    "PROFILE_SCHEMA_V2",
    "Profile",
    "ProfileError",
    "build_profile_from_files",
    "derive_profile",
    "download_specs",
    "load_profile",
    "target_from_inventory_path",
    "validate_profile",
    "verify_profile_sources",
    "write_profile",
]
