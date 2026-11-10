"""Schema-aware binding for a ComfyUI API workflow.

P1 intentionally does not guess real model nodes or weights. A workflow file is
an API JSON document whose node ids/types/edges are pinned by the P1 config and
manifest. This module validates that mapping and injects only the declared
scalar inputs.

Two workflows are kept distinct: ``p1-i2v.api.template.json`` is a synthetic
``TEMPLATE_NOT_EXECUTABLE`` binding fixture, while
``wan22-5b-shot01.api.draft.json`` is a manual, source-checked API adaptation of
the pinned official template. Neither is a live-verified or GPU-proven workflow.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import WorkflowBindingError


TEMPLATE_MARKER = "TEMPLATE_NOT_EXECUTABLE"


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise WorkflowBindingError(f"Cannot read workflow {path}: {exc}") from exc
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowBindingError(f"Workflow is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowBindingError("Workflow JSON must be an object of node id -> node")
    return value, hashlib.sha256(data).hexdigest()


def load_workflow(path: Path | str) -> tuple[dict[str, Any], str]:
    """Return workflow object and the sha256 of the exact file bytes."""

    return _read_json(Path(path))


def safe_remote_filename(value: str) -> str:
    """Validate a filename used in a remote artifact reference.

    Rejects separators, drive letters, traversal, control/bidi characters and
    non-NFC names. The returned value is already NFC-normalized.
    """

    if not isinstance(value, str) or not value:
        raise WorkflowBindingError("Remote filename must be a non-empty string")
    if len(value) > 255:
        raise WorkflowBindingError("Remote filename is too long")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise WorkflowBindingError("Remote filename must already be NFC-normalized")
    if value in {".", ".."}:
        raise WorkflowBindingError("Remote filename cannot be '.' or '..'")
    if any(ch in value for ch in ("/", "\\", ":", "\x00")):
        raise WorkflowBindingError("Remote filename contains a path separator")
    for char in value:
        if unicodedata.category(char) in {"Cc", "Cf"}:
            raise WorkflowBindingError("Remote filename contains control characters")
    stem = value.split(".", 1)[0].upper()
    if stem in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise WorkflowBindingError("Remote filename is a reserved Windows name")
    return value


def safe_relative_subfolder(value: str | None) -> str:
    """Validate a ComfyUI subfolder, keeping it relative and separator-safe."""

    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise WorkflowBindingError("Remote subfolder must be a string")
    if len(value) > 512:
        raise WorkflowBindingError("Remote subfolder is too long")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise WorkflowBindingError("Remote subfolder must already be NFC-normalized")
    if value.startswith(("/", "\\")) or "\\" in value or ":" in value:
        raise WorkflowBindingError("Remote subfolder must be relative")
    parts = value.split("/")
    for part in parts:
        if part in {"", ".", ".."}:
            raise WorkflowBindingError("Remote subfolder contains traversal")
        if any(unicodedata.category(char) in {"Cc", "Cf"} for char in part):
            raise WorkflowBindingError("Remote subfolder contains control characters")
    return value


@dataclass(frozen=True)
class FieldMapping:
    node_id: str
    input_name: str


@dataclass(frozen=True)
class OutputMapping:
    node_id: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowBinding:
    node_types: Mapping[str, str]
    prompt: FieldMapping
    seed: FieldMapping
    image: FieldMapping
    width: FieldMapping
    height: FieldMapping
    frames: FieldMapping
    fps: FieldMapping
    output: OutputMapping
    optional_node_types: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowBinding":
        if not isinstance(data, Mapping):
            raise WorkflowBindingError("binding must be an object")

        def field_mapping(key: str) -> FieldMapping:
            raw = data.get(key)
            if not isinstance(raw, Mapping):
                raise WorkflowBindingError(f"binding.{key} must be an object")
            node = raw.get("node")
            inp = raw.get("input")
            if not isinstance(node, str) or not node:
                raise WorkflowBindingError(f"binding.{key}.node must be a string")
            if not isinstance(inp, str) or not inp:
                raise WorkflowBindingError(f"binding.{key}.input must be a string")
            return FieldMapping(node, inp)

        raw_types = data.get("node_types")
        if not isinstance(raw_types, Mapping) or not raw_types:
            raise WorkflowBindingError("binding.node_types must be a non-empty object")
        node_types = {}
        for node, class_type in raw_types.items():
            if not isinstance(node, str) or not isinstance(class_type, str):
                raise WorkflowBindingError("binding.node_types entries must be strings")
            node_types[node] = class_type

        raw_optional = data.get("optional_node_types") or {}
        if not isinstance(raw_optional, Mapping):
            raise WorkflowBindingError("binding.optional_node_types must be an object")
        optional = {}
        for node, class_type in raw_optional.items():
            if not isinstance(node, str) or not isinstance(class_type, str):
                raise WorkflowBindingError(
                    "binding.optional_node_types entries must be strings"
                )
            optional[node] = class_type

        raw_output = data.get("output")
        if not isinstance(raw_output, Mapping):
            raise WorkflowBindingError("binding.output must be an object")
        output_node = raw_output.get("node")
        output_keys = raw_output.get("keys")
        if not isinstance(output_node, str) or not output_node:
            raise WorkflowBindingError("binding.output.node must be a string")
        if (
            not isinstance(output_keys, list)
            or not output_keys
            or any(not isinstance(key, str) or not key for key in output_keys)
        ):
            raise WorkflowBindingError("binding.output.keys must be a non-empty list")

        mappings = {
            "prompt": field_mapping("prompt"),
            "seed": field_mapping("seed"),
            "image": field_mapping("image"),
            "width": field_mapping("width"),
            "height": field_mapping("height"),
            "frames": field_mapping("frames"),
            "fps": field_mapping("fps"),
        }
        seen: dict[tuple[str, str], str] = {}
        for name, mapping in mappings.items():
            slot = (mapping.node_id, mapping.input_name)
            if slot in seen:
                raise WorkflowBindingError(
                    f"binding.{name} duplicates binding.{seen[slot]} at "
                    f"{mapping.node_id}.{mapping.input_name}"
                )
            seen[slot] = name

        return cls(
            node_types=node_types,
            optional_node_types=optional,
            prompt=mappings["prompt"],
            seed=mappings["seed"],
            image=mappings["image"],
            width=mappings["width"],
            height=mappings["height"],
            frames=mappings["frames"],
            fps=mappings["fps"],
            output=OutputMapping(output_node, tuple(output_keys)),
        )

    def _all_mappings(self) -> tuple[FieldMapping, ...]:
        return (
            self.prompt,
            self.seed,
            self.image,
            self.width,
            self.height,
            self.frames,
            self.fps,
        )

    def validate(self, workflow: Mapping[str, Any]) -> list[str]:
        issues: list[str] = []
        if not isinstance(workflow, Mapping):
            return ["workflow must be an object"]

        nodes = {key: value for key, value in workflow.items() if not key.startswith("_")}
        if not nodes:
            issues.append("workflow has no nodes")
        for node_id, node in nodes.items():
            if not isinstance(node_id, str):
                issues.append("workflow node ids must be strings")
                continue
            if not isinstance(node, Mapping):
                issues.append(f"node {node_id} must be an object")
                continue
            class_type = node.get("class_type")
            inputs = node.get("inputs")
            if not isinstance(class_type, str) or not class_type:
                issues.append(f"node {node_id} is missing class_type")
            if not isinstance(inputs, Mapping):
                issues.append(f"node {node_id} is missing inputs")
                continue
            for input_name, value in inputs.items():
                if not isinstance(value, list):
                    continue
                if (
                    len(value) != 2
                    or not isinstance(value[0], str)
                    or isinstance(value[1], bool)
                    or not isinstance(value[1], int)
                    or value[1] < 0
                ):
                    issues.append(
                        f"node {node_id}.{input_name} has a malformed edge"
                    )
                    continue
                if value[0] not in nodes:
                    issues.append(
                        f"node {node_id}.{input_name} references missing node "
                        f"{value[0]!r}"
                    )

        for node_id, expected in self.node_types.items():
            node = nodes.get(node_id)
            if not isinstance(node, Mapping):
                issues.append(f"required node {node_id} is missing")
                continue
            actual = node.get("class_type")
            if actual != expected:
                issues.append(
                    f"node {node_id} class_type is {actual!r}, expected {expected!r}"
                )
        for node_id, expected in self.optional_node_types.items():
            node = nodes.get(node_id)
            if node is None:
                continue
            if not isinstance(node, Mapping) or node.get("class_type") != expected:
                issues.append(
                    f"optional node {node_id} class_type must be {expected!r}"
                )

        scalar_expected = {
            self.prompt: str,
            self.image: str,
            self.seed: int,
            self.width: int,
            self.height: int,
            self.frames: int,
            self.fps: int,
        }
        for mapping, expected in scalar_expected.items():
            node = nodes.get(mapping.node_id)
            if not isinstance(node, Mapping):
                issues.append(f"binding node {mapping.node_id} is missing")
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, Mapping):
                issues.append(f"binding node {mapping.node_id} has no inputs")
                continue
            if mapping.input_name not in inputs:
                issues.append(
                    f"node {mapping.node_id} is missing input {mapping.input_name!r}"
                )
            elif isinstance(inputs[mapping.input_name], list):
                issues.append(
                    f"node {mapping.node_id}.{mapping.input_name} is linked, "
                    "but the binding requires a scalar input"
                )
            else:
                value = inputs[mapping.input_name]
                if expected is int:
                    if isinstance(value, bool) or not isinstance(value, int):
                        issues.append(
                            f"node {mapping.node_id}.{mapping.input_name} must be "
                            "an integer"
                        )
                elif not isinstance(value, expected):
                    issues.append(
                        f"node {mapping.node_id}.{mapping.input_name} must be "
                        f"{expected.__name__}"
                    )

        output_node = nodes.get(self.output.node_id)
        if not isinstance(output_node, Mapping):
            issues.append(f"output node {self.output.node_id} is missing")
        return issues

    def bind(
        self,
        workflow: Mapping[str, Any],
        *,
        prompt: str,
        seed: int,
        image_name: str,
        width: int,
        height: int,
        frames: int,
        fps: int,
    ) -> dict[str, Any]:
        issues = self.validate(workflow)
        if issues:
            raise WorkflowBindingError("; ".join(issues))
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkflowBindingError("prompt must be a non-empty string")
        for name, value, lower, upper in (
            ("seed", seed, 0, 2_147_483_647),
            ("width", width, 16, 8192),
            ("height", height, 16, 8192),
            ("frames", frames, 1, 1000),
            ("fps", fps, 1, 120),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise WorkflowBindingError(f"{name} must be an integer")
            if value < lower or value > upper:
                raise WorkflowBindingError(
                    f"{name} must be between {lower} and {upper}"
                )
        safe_image = safe_remote_filename(image_name)

        bound = copy.deepcopy(dict(workflow))
        bound.pop("_meta", None)
        bound.pop("_template", None)
        values = {
            self.prompt: prompt,
            self.seed: seed,
            self.image: safe_image,
            self.width: width,
            self.height: height,
            self.frames: frames,
            self.fps: fps,
        }
        for mapping, value in values.items():
            node = bound.get(mapping.node_id)
            if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
                raise WorkflowBindingError(
                    f"cannot set {mapping.node_id}.{mapping.input_name}"
                )
            node["inputs"][mapping.input_name] = value
        return {
            key: value
            for key, value in bound.items()
            if not key.startswith("_")
        }

    @staticmethod
    def is_template(workflow: Mapping[str, Any]) -> bool:
        try:
            return TEMPLATE_MARKER in json.dumps(workflow, ensure_ascii=False)
        except (TypeError, ValueError):
            return False


def workflow_sha256(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise WorkflowBindingError(f"Cannot hash workflow {path}: {exc}") from exc


__all__ = [
    "FieldMapping",
    "OutputMapping",
    "WorkflowBinding",
    "TEMPLATE_MARKER",
    "load_workflow",
    "safe_relative_subfolder",
    "safe_remote_filename",
    "workflow_sha256",
]
