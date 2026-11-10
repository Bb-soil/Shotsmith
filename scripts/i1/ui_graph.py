"""Build a best-effort UI workflow from the exact pinned API graph.

ComfyUI's API JSON is not a canvas workflow.  This converter produces a derived
UI JSON from the *same* API nodes and values (never from an older reference
workflow with different prompts).  It is deliberately marked NOT_LIVE_VERIFIED
until a real frontend loads it; the exact API file remains the fallback.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .util import sha256_mapping

# Output ports are read from the pinned 14B UI schema; they are types only.
OUTPUTS: dict[str, list[tuple[str, str]]] = {
    "CLIPLoader": [("CLIP", "CLIP")],
    "UNETLoader": [("MODEL", "MODEL")],
    "VAELoader": [("VAE", "VAE")],
    "LoadImage": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    "CLIPTextEncode": [("CONDITIONING", "CONDITIONING")],
    "ModelSamplingSD3": [("MODEL", "MODEL")],
    "WanFirstLastFrameToVideo": [
        ("positive", "CONDITIONING"),
        ("negative", "CONDITIONING"),
        ("latent", "LATENT"),
    ],
    "KSamplerAdvanced": [("LATENT", "LATENT")],
    "VAEDecode": [("IMAGE", "IMAGE")],
    "CreateVideo": [("VIDEO", "VIDEO")],
    "SaveVideo": [],
}

INPUTS: dict[str, list[tuple[str, str]]] = {
    "CLIPLoader": [],
    "UNETLoader": [],
    "VAELoader": [],
    "LoadImage": [],
    "CLIPTextEncode": [("clip", "CLIP")],
    "ModelSamplingSD3": [("model", "MODEL")],
    "WanFirstLastFrameToVideo": [
        ("positive", "CONDITIONING"),
        ("negative", "CONDITIONING"),
        ("vae", "VAE"),
        ("start_image", "IMAGE"),
        ("end_image", "IMAGE"),
    ],
    "KSamplerAdvanced": [
        ("model", "MODEL"),
        ("positive", "CONDITIONING"),
        ("negative", "CONDITIONING"),
        ("latent_image", "LATENT"),
    ],
    "VAEDecode": [("samples", "LATENT"), ("vae", "VAE")],
    "CreateVideo": [("images", "IMAGE"), ("audio", "AUDIO")],
    "SaveVideo": [("video", "VIDEO")],
}

WIDGET_ORDER: dict[str, list[str]] = {
    "CLIPLoader": ["clip_name", "type", "device"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "VAELoader": ["vae_name"],
    "LoadImage": ["image", "upload"],
    "CLIPTextEncode": ["text"],
    "ModelSamplingSD3": ["shift"],
    "WanFirstLastFrameToVideo": ["width", "height", "length", "batch_size"],
    "KSamplerAdvanced": [
        "add_noise",
        "noise_seed",
        "control_after_generate",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "start_at_step",
        "end_at_step",
        "return_with_leftover_noise",
    ],
    "CreateVideo": ["fps"],
    "SaveVideo": ["filename_prefix", "format", "format.codec", "format.codec.encoding"],
}

DEFAULT_WIDGETS: dict[str, Any] = {
    "upload": "image",
    "control_after_generate": "fixed",
    "batch_size": 1,
    "weight_dtype": "default",
    "device": "default",
}


class UiGraphError(ValueError):
    pass


def _int_node_ids(api: Mapping[str, Any]) -> dict[str, int]:
    keys = [key for key in api if not str(key).startswith("_")]
    numeric = [key for key in keys if str(key).isdigit() and int(key) > 0]
    if len(numeric) == len(keys):
        return {key: int(key) for key in keys}
    return {key: index + 1 for index, key in enumerate(keys)}


def derive_ui_graph(api: Mapping[str, Any], *, graph_id: str | None = None) -> dict[str, Any]:
    if not isinstance(api, Mapping) or not api:
        raise UiGraphError("API workflow must be a non-empty object")
    node_ids = _int_node_ids(api)
    links: list[list[Any]] = []
    link_id = 1
    nodes: list[dict[str, Any]] = []
    api_nodes = {
        key: value
        for key, value in api.items()
        if not str(key).startswith("_") and isinstance(value, Mapping)
    }
    if len(api_nodes) != len(node_ids):
        raise UiGraphError("API workflow contains non-object nodes")

    for order, (api_id, spec) in enumerate(api_nodes.items()):
        class_type = spec.get("class_type")
        api_inputs = spec.get("inputs")
        if not isinstance(class_type, str) or not isinstance(api_inputs, Mapping):
            raise UiGraphError(f"API node {api_id} is malformed")
        output_specs = OUTPUTS.get(class_type)
        input_specs = INPUTS.get(class_type)
        if output_specs is None or input_specs is None:
            raise UiGraphError(f"unsupported UI class: {class_type}")
        target_inputs: dict[str, list[Any]] = {}
        for input_name, value in api_inputs.items():
            if isinstance(value, list):
                if (
                    len(value) != 2
                    or not isinstance(value[0], str)
                    or value[0] not in node_ids
                ):
                    raise UiGraphError(f"API edge {api_id}.{input_name} is malformed")
                target_inputs[str(input_name)] = [value[0], value[1]]
        ui_inputs: list[dict[str, Any]] = []
        for name, type_name in input_specs:
            if name not in target_inputs:
                continue
            source_id, source_slot = target_inputs[name]
            current_link = link_id
            link_id += 1
            links.append(
                [
                    current_link,
                    node_ids[source_id],
                    int(source_slot),
                    node_ids[api_id],
                    len(ui_inputs),
                    type_name,
                ]
            )
            ui_inputs.append(
                {
                    "name": name,
                    "type": type_name,
                    "link": current_link,
                }
            )
        widgets: list[Any] = []
        widget_keys = WIDGET_ORDER.get(class_type, [])
        for key in widget_keys:
            if key in api_inputs and not isinstance(api_inputs[key], list):
                widgets.append(copy.deepcopy(api_inputs[key]))
            elif key in DEFAULT_WIDGETS:
                widgets.append(DEFAULT_WIDGETS[key])
            elif key in ("filename_prefix", "text", "image"):
                widgets.append("")
            elif key in ("shift", "fps", "width", "height", "length", "steps", "cfg", "noise_seed", "start_at_step", "end_at_step"):
                widgets.append(0)
            else:
                widgets.append("")
        ui_outputs = [
            {"name": name, "type": type_name, "slot_index": index, "links": []}
            for index, (name, type_name) in enumerate(output_specs)
        ]
        nodes.append(
            {
                "id": node_ids[api_id],
                "type": class_type,
                "pos": [100 + (order % 4) * 360, 100 + (order // 4) * 220],
                "size": [320, 140],
                "flags": {},
                "order": order,
                "mode": 0,
                "inputs": ui_inputs,
                "outputs": ui_outputs,
                "properties": {"Node name for S&R": class_type},
                "widgets_values": widgets,
            }
        )

    by_id = {node["id"]: node for node in nodes}
    for link in links:
        _link_id, source_id, source_slot, _target_id, _target_slot, _type = link
        source = by_id[source_id]
        outputs = source["outputs"]
        if source_slot >= len(outputs):
            raise UiGraphError("UI edge references a missing output slot")
        outputs[source_slot]["links"].append(_link_id)

    data = {
        "id": graph_id or "i1-derived-" + sha256_mapping(api)[:12],
        "revision": 0,
        "last_node_id": max(by_id),
        "last_link_id": link_id - 1,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.65, "offset": [100, 100]},
            "frontendVersion": "1.32.1",
            "ue_links": [],
        },
        "version": "0.4",
    }
    return data


def validate_ui_graph(data: Mapping[str, Any]) -> int:
    nodes = data.get("nodes")
    links = data.get("links")
    if not isinstance(nodes, list) or not nodes or not isinstance(links, list):
        raise UiGraphError("UI graph nodes/links are missing")
    ids: set[int] = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            raise UiGraphError("UI node is not an object")
        node_id = node.get("id")
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id <= 0:
            raise UiGraphError("UI node id is invalid")
        if node_id in ids:
            raise UiGraphError("UI node ids are duplicated")
        ids.add(node_id)
        if not isinstance(node.get("type"), str) or not node["type"]:
            raise UiGraphError("UI node type is invalid")
        if not isinstance(node.get("inputs"), list) or not isinstance(
            node.get("outputs"), list
        ):
            raise UiGraphError("UI node inputs/outputs are missing")
    for link in links:
        if not isinstance(link, list) or len(link) < 5:
            raise UiGraphError("UI link is malformed")
        if link[1] not in ids or link[3] not in ids:
            raise UiGraphError("UI link endpoints are missing")
    return len(nodes)


def derived_prompts_match_api(api: Mapping[str, Any], ui: Mapping[str, Any]) -> bool:
    api_text = sorted(
        str(node.get("inputs", {}).get("text"))
        for node in api.values()
        if isinstance(node, Mapping)
        and node.get("class_type") == "CLIPTextEncode"
        and isinstance(node.get("inputs"), Mapping)
    )
    ui_nodes = ui.get("nodes")
    if not isinstance(ui_nodes, list):
        return False
    ui_text = []
    for node in ui_nodes:
        if (
            isinstance(node, Mapping)
            and node.get("type") == "CLIPTextEncode"
            and isinstance(node.get("widgets_values"), list)
            and node["widgets_values"]
        ):
            ui_text.append(str(node["widgets_values"][0]))
    return sorted(ui_text) == api_text


def write_ui_graph(path: Path | str, data: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "UiGraphError",
    "derive_ui_graph",
    "derived_prompts_match_api",
    "validate_ui_graph",
    "write_ui_graph",
]
