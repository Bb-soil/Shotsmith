"""Pinned source contract for KF1.

The tests use the actual snapshots under docs/k1-evidence/.../source, not only a
hand-written table.  This module extracts class/input/output names from those
snapshots and validates a bound workflow against them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "docs" / "k1-evidence" / "kf1-implementation-20261029" / "source"
PINS_PATH = SOURCE_DIR / "pins.json"
COMFY_COMMIT = "7a28332047b6af638dc6c7f72ef3481a718769fd"

CLASS_SOURCE = {
    "LoadImage": "nodes.py",
    "ImageScale": "nodes.py",
    "UNETLoader": "nodes.py",
    "CLIPLoader": "nodes.py",
    "VAELoader": "nodes.py",
    "VAEEncode": "nodes.py",
    "KSampler": "nodes.py",
    "VAEDecode": "nodes.py",
    "SaveImage": "nodes.py",
    "ModelSamplingAuraFlow": "nodes_model_advanced.py",
    "CFGNorm": "nodes_cfg.py",
    "TextEncodeQwenImageEditPlus": "nodes_qwen.py",
}

# Input/output type contract.  Names are checked against the source snapshots;
# this map supplies the runtime type used for link compatibility.
TYPE_CONTRACT: dict[str, dict[str, Any]] = {
    "LoadImage": {
        "required": {"image": "COMBO"},
        "optional": {},
        "outputs": ["IMAGE", "MASK"],
    },
    "ImageScale": {
        "required": {
            "image": "IMAGE",
            "upscale_method": "COMBO",
            "width": "INT",
            "height": "INT",
            "crop": "COMBO",
        },
        "optional": {},
        "outputs": ["IMAGE"],
    },
    "UNETLoader": {
        "required": {"unet_name": "COMBO", "weight_dtype": "COMBO"},
        "optional": {},
        "outputs": ["MODEL"],
    },
    "CLIPLoader": {
        "required": {"clip_name": "COMBO", "type": "COMBO"},
        "optional": {"device": "COMBO"},
        "outputs": ["CLIP"],
    },
    "VAELoader": {
        "required": {"vae_name": "COMBO"},
        "optional": {},
        "outputs": ["VAE"],
    },
    "ModelSamplingAuraFlow": {
        "required": {"model": "MODEL", "shift": "FLOAT"},
        "optional": {"sampling": "COMBO"},
        "outputs": ["MODEL"],
    },
    "CFGNorm": {
        "required": {"model": "MODEL", "strength": "FLOAT"},
        "optional": {"pre_cfg": "BOOLEAN"},
        "outputs": ["MODEL"],
    },
    "TextEncodeQwenImageEditPlus": {
        "required": {"clip": "CLIP", "prompt": "STRING"},
        "optional": {"vae": "VAE", "image1": "IMAGE", "image2": "IMAGE", "image3": "IMAGE"},
        "outputs": ["CONDITIONING"],
    },
    "VAEEncode": {
        "required": {"pixels": "IMAGE", "vae": "VAE"},
        "optional": {},
        "outputs": ["LATENT"],
    },
    "KSampler": {
        "required": {
            "model": "MODEL",
            "seed": "INT",
            "steps": "INT",
            "cfg": "FLOAT",
            "sampler_name": "COMBO",
            "scheduler": "COMBO",
            "positive": "CONDITIONING",
            "negative": "CONDITIONING",
            "latent_image": "LATENT",
            "denoise": "FLOAT",
        },
        "optional": {},
        "outputs": ["LATENT"],
    },
    "VAEDecode": {
        "required": {"samples": "LATENT", "vae": "VAE"},
        "optional": {},
        "outputs": ["IMAGE"],
    },
    "SaveImage": {
        "required": {"images": "IMAGE", "filename_prefix": "STRING"},
        "optional": {},
        "outputs": ["IMAGE"],
    },
    "SaveVideo": {
        "required": {
            "video": "VIDEO",
            "filename_prefix": "STRING",
            "format": "COMBO",
            "format.codec": "STRING",
            "format.codec.encoding": "STRING",
        },
        "optional": {"codec": "COMFY_DYNAMICCOMBO_V3"},
        "outputs": ["VIDEO"],
    },
}


def _class_block(text: str, class_name: str) -> str | None:
    match = re.search(rf"(?m)^class\s+{re.escape(class_name)}\b", text)
    if not match:
        return None
    start = match.start()
    next_class = re.search(r"(?m)^class\s+\w+", text[match.end():])
    end = match.end() + next_class.start() if next_class else len(text)
    return text[start:end]


def _balanced_block(text: str, marker: str) -> str | None:
    index = text.find(marker)
    if index < 0:
        return None
    brace = text.find("{", index)
    if brace < 0:
        return None
    depth = 0
    for pos in range(brace, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : pos]
    return None


def _keys_in_block(block: str | None) -> list[str]:
    if not block:
        return []
    result: list[str] = []
    depth = 0
    in_string = False
    escape = False
    index = 0
    while index < len(block):
        char = block[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            # A key is a quoted name at depth 0 followed by optional spaces and ':'.
            end = index + 1
            escaped = False
            while end < len(block):
                if escaped:
                    escaped = False
                elif block[end] == "\\":
                    escaped = True
                elif block[end] == '"':
                    break
                end += 1
            if end < len(block):
                name = block[index + 1 : end]
                after = end + 1
                while after < len(block) and block[after] in " \t\r\n":
                    after += 1
                if depth == 0 and after < len(block) and block[after] == ":" and re.fullmatch(
                    r"[A-Za-z0-9_.]+", name
                ):
                    result.append(name)
                index = end + 1
                continue
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth = max(0, depth - 1)
        index += 1
    return result


def _legacy_contract(text: str, class_name: str) -> dict[str, Any] | None:
    block = _class_block(text, class_name)
    if block is None:
        return None
    required = _keys_in_block(_balanced_block(block, '"required"'))
    optional = _keys_in_block(_balanced_block(block, '"optional"'))
    returns_match = re.search(r"RETURN_TYPES\s*=\s*\(([^)]*)\)", block, re.S)
    outputs = re.findall(r'"([A-Z0-9_]+)"', returns_match.group(1)) if returns_match else []
    if not outputs:
        base_match = re.search(rf"(?m)^class\s+{re.escape(class_name)}\((\w+)\)", block)
        if base_match and base_match.group(1) != class_name:
            base = _legacy_contract(text, base_match.group(1))
            if base:
                outputs = list(base.get("outputs") or [])
    types = {}
    for name in required + optional:
        match = re.search(
            rf'"{re.escape(name)}"\s*:\s*\(\s*"([A-Z0-9_]+)"',
            block,
        )
        if match:
            types[name] = match.group(1)
    return {
        "required": required,
        "optional": optional,
        "outputs": outputs,
        "types": types,
    }


def _call_args(block: str, method: str) -> list[str]:
    result: list[str] = []
    pattern = re.compile(rf"io\.\w+\.{re.escape(method)}\(")
    for match in pattern.finditer(block):
        start = match.end()
        depth = 1
        in_string = False
        escape = False
        index = start
        while index < len(block) and depth > 0:
            char = block[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        if depth == 0:
            result.append(block[start : index - 1])
    return result


def _v3_contract(text: str, class_name: str) -> dict[str, Any] | None:
    block = _class_block(text, class_name)
    if block is None:
        return None
    required: list[str] = []
    optional: list[str] = []
    for args in _call_args(block, "Input"):
        name_match = re.search(r'"([^"]+)"', args)
        if not name_match:
            continue
        name = name_match.group(1)
        if "optional=True" in args:
            optional.append(name)
        else:
            required.append(name)
    outputs = [
        match.group(1).upper()
        for match in re.finditer(r"io\.(\w+)\.Output\(", block)
    ]
    type_map = {
        name: {
            "Image": "IMAGE",
            "Clip": "CLIP",
            "Vae": "VAE",
            "String": "STRING",
            "Int": "INT",
            "Float": "FLOAT",
            "Boolean": "BOOLEAN",
            "Combo": "COMBO",
            "Model": "MODEL",
            "Conditioning": "CONDITIONING",
        }.get(kind, kind.upper())
        for kind, name in re.findall(r'io\.(\w+)\.Input\("([^"]+)"', block)
    }
    return {
        "required": required,
        "optional": optional,
        "outputs": outputs,
        "types": type_map,
    }


def _enum_values(text: str, class_name: str, input_name: str) -> list[str]:
    block = _class_block(text, class_name)
    if block is None:
        return []
    match = re.search(
        rf'"{re.escape(input_name)}"\s*:\s*\(\[(.*?)\]',
        block,
        re.S,
    )
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))
def load_source_contract() -> dict[str, dict[str, Any]]:
    contract: dict[str, dict[str, Any]] = {}
    for class_name, filename in CLASS_SOURCE.items():
        path = SOURCE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"pinned source snapshot missing: {path}")
        text = path.read_text(encoding="utf-8")
        if class_name in {"TextEncodeQwenImageEditPlus", "CFGNorm"}:
            extracted = _v3_contract(text, class_name)
        else:
            extracted = _legacy_contract(text, class_name)
        if extracted is None:
            raise ValueError(f"class {class_name} not found in pinned {filename}")
        expected = TYPE_CONTRACT[class_name]
        if sorted(extracted["required"]) != sorted(expected["required"]):
            raise ValueError(
                f"{class_name}: source required inputs differ: "
                f"{extracted['required']} != {list(expected['required'])}"
            )
        if sorted(extracted["optional"]) != sorted(expected["optional"]):
            raise ValueError(
                f"{class_name}: source optional inputs differ: "
                f"{extracted['optional']} != {list(expected['optional'])}"
            )
        if [item for item in extracted["outputs"]] != list(expected["outputs"]):
            raise ValueError(f"{class_name}: source outputs differ")
        enums = {}
        for name in ("type", "upscale_method", "crop", "weight_dtype"):
            values = _enum_values(text, class_name, name)
            if values:
                enums[name] = values
        contract[class_name] = {
            "required": list(extracted["required"]),
            "optional": list(extracted["optional"]),
            "enums": enums,
            "types": dict(extracted.get("types") or {}),
            "outputs": list(expected["outputs"]),
            "file": filename,
        }
    return contract


def source_hashes() -> dict[str, str]:
    result = {}
    for path in sorted(SOURCE_DIR.glob("*")):
        if path.is_file():
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def load_pins() -> dict[str, Any]:
    if not PINS_PATH.is_file():
        raise FileNotFoundError(f"pins file missing: {PINS_PATH}")
    return json.loads(PINS_PATH.read_text(encoding="utf-8"))


def validate_workflow(workflow: Mapping[str, Any], *, allow_placeholder: bool = False) -> list[str]:
    issues: list[str] = []
    contract = load_source_contract()
    nodes = {key: value for key, value in workflow.items() if not str(key).startswith("_")}
    for node_id, node in nodes.items():
        if not isinstance(node, Mapping):
            issues.append(f"node {node_id} is not an object")
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if class_type not in contract:
            issues.append(f"node {node_id} has unknown class {class_type!r}")
            continue
        if not isinstance(inputs, Mapping):
            issues.append(f"node {node_id} has no inputs")
            continue
        declared = set(contract[class_type]["required"]) | set(contract[class_type]["optional"])
        for input_name, value in inputs.items():
            name = str(input_name)
            if name not in declared:
                issues.append(f"node {node_id} ({class_type}) undeclared input {name!r}")
            if isinstance(value, list):
                if len(value) != 2 or not isinstance(value[0], str):
                    issues.append(f"node {node_id}.{name} malformed edge")
                    continue
                source = nodes.get(value[0])
                if not isinstance(source, Mapping):
                    issues.append(f"node {node_id}.{name} references missing node")
                    continue
                source_class = source.get("class_type")
                source_contract = contract.get(str(source_class))
                if source_contract is None or not isinstance(value[1], int):
                    issues.append(f"node {node_id}.{name} source class/slot invalid")
                    continue
                if value[1] < 0 or value[1] >= len(source_contract["outputs"]):
                    issues.append(f"node {node_id}.{name} source slot out of range")
                    continue
                expected_type = contract[class_type]["types"].get(name) or TYPE_CONTRACT[class_type]["required"].get(name) or TYPE_CONTRACT[class_type]["optional"].get(name)
                actual_type = source_contract["outputs"][value[1]]
                if expected_type != actual_type:
                    issues.append(
                        f"node {node_id}.{name} link {actual_type} != {expected_type}"
                    )
        for input_name, value in inputs.items():
            if isinstance(value, list):
                continue
            expected_type = contract[class_type]["types"].get(str(input_name)) or TYPE_CONTRACT[class_type]["required"].get(str(input_name)) or TYPE_CONTRACT[class_type]["optional"].get(str(input_name))
            if expected_type == "INT" and (isinstance(value, bool) or not isinstance(value, int)):
                issues.append(f"node {node_id}.{input_name} must be INT")
            elif expected_type == "FLOAT" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                issues.append(f"node {node_id}.{input_name} must be FLOAT")
            elif expected_type == "STRING" and not isinstance(value, str):
                issues.append(f"node {node_id}.{input_name} must be STRING")
            elif expected_type == "BOOLEAN" and not isinstance(value, bool):
                issues.append(f"node {node_id}.{input_name} must be BOOLEAN")
            enum_values = contract[class_type].get("enums", {}).get(str(input_name))
            if enum_values and value not in enum_values:
                issues.append(
                    f"node {node_id}.{input_name} value {value!r} not in pinned enum"
                )
        for required in contract[class_type]["required"]:
            if required not in inputs:
                issues.append(f"node {node_id} ({class_type}) missing required {required!r}")
        if not allow_placeholder and (
            "{{" in json.dumps(inputs, ensure_ascii=False)
            or "KF1_BIND_REQUIRED" in json.dumps(inputs, ensure_ascii=False)
        ):
            issues.append(f"node {node_id} still contains a bind placeholder")
    return issues


def source_contract_report() -> dict[str, Any]:
    contract = load_source_contract()
    dynamic = {
        "LoadImage.image": "folder_paths file enum",
        "UNETLoader.unet_name": "folder_paths file enum",
        "CLIPLoader.clip_name": "folder_paths file enum",
        "VAELoader.vae_name": "folder_paths file enum",
        "KSampler.sampler_name": "comfy.samplers runtime enum",
        "KSampler.scheduler": "comfy.samplers runtime enum",
    }
    return {
        "schema": "anime-studio/kf1-source-contract/v1",
        "comfy_commit": COMFY_COMMIT,
        "checked_classes": sorted(contract),
        "dynamic_inputs": dynamic,
        "source_dynamic_not_verified": sorted(dynamic),
        "actual_object_info": "NOT_RUN",
        "source_extraction": "static_regex_from_pinned_snapshots",
        "full_source_ast_proof": "NOT_IMPLEMENTED",
        "static_schema_scope": "class/input/output names, types and pinned static enums",
        "numeric_ranges": "RUNTIME_OBJECT_INFO_ONLY",
        "dynamic_enums": "RUNTIME_OBJECT_INFO_ONLY",
        "note": (
            "Names/types/outputs/static enums come from pinned snapshots via a "
            "bounded regex extractor, not a full AST proof. Numeric min/max and "
            "dynamic file/sampler enums are verified only against actual "
            "object_info at runtime (installation capability and bound-job gate)."
        ),
    }

__all__ = [
    "CLASS_SOURCE",
    "COMFY_COMMIT",
    "PINS_PATH",
    "SOURCE_DIR",
    "TYPE_CONTRACT",
    "load_pins",
    "load_source_contract",
    "source_hashes",
    "source_contract_report",
    "validate_workflow",
]







