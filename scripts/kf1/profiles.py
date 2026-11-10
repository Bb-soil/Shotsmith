"""Derived I1 profile/v2 artifacts for the opt-in KF1 model sets.

Original source inventories, Wan v1 profile, K1 drafts and the example plan are
never modified.  These builders only read them and write derived profiles.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .errors import PlanError
from .source_contract import COMFY_COMMIT

ROOT = Path(__file__).resolve().parents[2]
PROFILE_V2 = "anime-video-studio/i1-profile/v2"
WAN_PROFILE = ROOT / "config" / "i1" / "wan22-14b-ab.profile.json"
K1_MANIFEST = ROOT / "manifests" / "K1.qwen-edit-2511.sources.draft.json"
K1_PROFILE = ROOT / "config" / "i1" / "keyframes-qwen-edit-2511.profile.json"
BUNDLE_PROFILE = ROOT / "config" / "i1" / "bundle-wan22-qwen-edit-2511.profile.json"
K1_WORKFLOWS = {
    "qwen-1": (1, ROOT / "workflows" / "K1" / "qwen-edit-2511-1ref.api.json"),
    "qwen-2": (2, ROOT / "workflows" / "K1" / "qwen-edit-2511-2ref.api.json"),
    "qwen-3": (3, ROOT / "workflows" / "K1" / "qwen-edit-2511-3ref.api.json"),
}
K1_OUTPUT = {
    "class": "SaveImage",
    "node_id": "70",
    "keys": ["images"],
    "media_type": "image/png",
    "dynamic": False,
}
WAN_OUTPUT = {
    "class": "SaveVideo",
    "node_id": "83",
    "keys": ["images"],
    "media_type": "video/mp4",
    "dynamic": True,
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"Не удалось прочитать {path}: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise PlanError(f"{path} должен быть JSON-объектом")
    return data


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_id(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", name.removesuffix(".safetensors")).strip("-")


def _workflow_meta(path: Path, *, output: Mapping[str, Any]) -> dict[str, Any]:
    data = _read_json(path)
    nodes = {key: value for key, value in data.items() if not str(key).startswith("_")}
    if not nodes:
        raise PlanError(f"workflow {path} пуст")
    required_classes = sorted({str(node["class_type"]) for node in nodes.values() if isinstance(node, Mapping)})
    required_inputs: dict[str, set[str]] = {}
    node_classes: dict[str, str] = {}
    graph_inputs: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str):
            raise PlanError(f"workflow {path}: node {node_id} некорректен")
        class_type = str(node["class_type"])
        node_classes[str(node_id)] = class_type
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            raise PlanError(f"workflow {path}: node {node_id} без inputs")
        for input_name, value in inputs.items():
            name = str(input_name)
            if isinstance(value, list):
                if len(value) != 2 or not isinstance(value[0], str):
                    raise PlanError(f"workflow {path}: edge {node_id}.{name} некорректен")
                required_inputs.setdefault(class_type, set()).add(name)
                graph_inputs.append(
                    {
                        "node_id": str(node_id),
                        "class": class_type,
                        "input": name,
                        "kind": "link",
                        "source_node": value[0],
                        "source_class": str(nodes[value[0]]["class_type"]),
                        "source_slot": int(value[1]),
                    }
                )
            elif "." in name:
                graph_inputs.append(
                    {"node_id": str(node_id), "class": class_type, "input": name, "kind": "dynamic"}
                )
            else:
                required_inputs.setdefault(class_type, set()).add(name)
                graph_inputs.append(
                    {"node_id": str(node_id), "class": class_type, "input": name, "kind": "literal", "value": value}
                )
    return {
        "api_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "api_sha256": _sha_file(path),
        "required_classes": required_classes,
        "required_inputs": {key: sorted(value) for key, value in sorted(required_inputs.items())},
        "node_classes": dict(sorted(node_classes.items())),
        "graph_inputs": sorted(graph_inputs, key=lambda item: (item["node_id"], item["input"])),
        "output": dict(output),
    }


def _k1_models() -> list[dict[str, Any]]:
    manifest = _read_json(K1_MANIFEST)
    if manifest.get("schema") != "anime-studio/keyframe-source-inventory/v1":
        raise PlanError("K1 inventory schema неожиданна")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise PlanError("K1 inventory должен содержать 3 модели")
    result = []
    for item in models:
        if not isinstance(item, Mapping):
            raise PlanError("K1 inventory model некорректен")
        name = str(item.get("source_path", "")).rsplit("/", 1)[-1]
        if not name.endswith(".safetensors"):
            raise PlanError("K1 model name некорректен")
        result.append(
            {
                "id": _model_id(name),
                "name": name,
                "role": str(item.get("role")),
                "repo_id": str(item.get("repo_id") or ""),
                "revision": str(item.get("revision") or ""),
                "source_path": str(item.get("source_path") or ""),
                "target": str(item.get("target") or ""),
                "size_bytes": int(item.get("size_bytes") or 0),
                "sha256": str(item.get("expected_sha256") or "").lower(),
                "url": str(item.get("url") or ""),
                "license": item.get("license_declared"),
                "license_source": item.get("license_source"),
                "binary_verified_at_derivation": False,
            }
        )
    if sum(item["size_bytes"] for item in result) != int(manifest.get("total_bytes", -1)):
        raise PlanError("K1 inventory total_bytes не совпадает")
    return result


def _k1_workflows() -> dict[str, dict[str, Any]]:
    result = {}
    for workflow_id, (_count, path) in K1_WORKFLOWS.items():
        result[workflow_id] = _workflow_meta(path, output=K1_OUTPUT)
    return result


def _base_runtime(wan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "comfyui": dict(wan["comfyui"]),
        "base_torch": dict(wan["base_torch"]),
        "runtime": dict(wan["runtime"]),
        "disk": dict(wan["disk"]),
    }


def build_keyframe_profile() -> dict[str, Any]:
    wan = _read_json(WAN_PROFILE)
    models = _k1_models()
    data = {
        "schema": PROFILE_V2,
        "profile_id": "keyframes-qwen-edit-2511",
        "display_name": "Qwen Image Edit 2511 FP8 (native, 1/2/3 refs)",
        "model_set": "keyframes-qwen-edit-2511",
        "models": models,
        "models_policy": {
            "mode": "allowlist",
            "expected_count": 3,
            "allowed_roles": ["diffusion", "text_encoder", "vae"],
            "role_counts": {"diffusion": 1, "text_encoder": 1, "vae": 1},
        },
        "workflows": _k1_workflows(),
        "primary_workflow": "qwen-3",
        "source_inventory": {
            "path": str(K1_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha_file(K1_MANIFEST),
            "schema": "anime-studio/keyframe-source-inventory/v1",
        },
        **_base_runtime(wan),
    }
    data["comfyui"]["commit"] = COMFY_COMMIT
    return data


def build_bundle_profile() -> dict[str, Any]:
    wan = _read_json(WAN_PROFILE)
    if wan.get("schema") != "anime-video-studio/i1-profile/v1":
        raise PlanError("Wan profile должен оставаться v1")
    k1_models = _k1_models()
    models = [dict(item) for item in wan["models"]] + k1_models
    targets = [item["target"] for item in models]
    if len(targets) != len(set(targets)):
        raise PlanError("Bundle model targets должны быть уникальны")
    workflows: dict[str, dict[str, Any]] = {
        "wan": {
            "api_path": wan["workflow"]["api_path"],
            "api_sha256": wan["workflow"]["api_sha256"],
            "required_classes": list(wan["required_classes"]),
            "required_inputs": dict(wan["required_inputs"]),
            "node_classes": dict(wan["node_classes"]),
            "graph_inputs": list(wan["graph_inputs"]),
            "output": dict(WAN_OUTPUT),
        }
    }
    workflows.update(_k1_workflows())
    data = {
        "schema": PROFILE_V2,
        "profile_id": "bundle-wan22-qwen-edit-2511",
        "display_name": "Wan 2.2 14B + Qwen Image Edit 2511 (shared pinned runtime)",
        "model_set": "wan22-14b-ab+keyframes-qwen-edit-2511",
        "models": models,
        "models_policy": {
            "mode": "allowlist",
            "expected_count": 7,
            "allowed_roles": ["diffusion", "text_encoder", "vae"],
            "role_counts": {"diffusion": 3, "text_encoder": 2, "vae": 2},
        },
        "workflows": workflows,
        "primary_workflow": "wan",
        "source_inventory": {
            "paths": [
                str(wan.get("source_inventory", {}).get("path") or "manifests/P1.wan22-14b-ab.sources.json"),
                str(K1_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            ]
        },
        **_base_runtime(wan),
    }
    data["comfyui"]["commit"] = COMFY_COMMIT
    return data


def write_derived_profiles(*, force: bool = False) -> tuple[Path, Path]:
    for path in (K1_PROFILE, BUNDLE_PROFILE):
        if path.exists() and not force:
            raise PlanError(f"Derived profile уже существует: {path}; используйте force")
    K1_PROFILE.write_text(
        json.dumps(build_keyframe_profile(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    BUNDLE_PROFILE.write_text(
        json.dumps(build_bundle_profile(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return K1_PROFILE, BUNDLE_PROFILE


__all__ = [
    "BUNDLE_PROFILE",
    "K1_PROFILE",
    "PROFILE_V2",
    "build_bundle_profile",
    "build_keyframe_profile",
    "write_derived_profiles",
]

