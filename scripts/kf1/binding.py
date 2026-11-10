"""Exact, confined construction of a KF1 prompt/graph binding."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from anime_studio.workflow import load_workflow, safe_remote_filename

from .errors import BindingError
from .plan import (
    DEFAULT_CFG,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    INVARIANT_VOCAB,
    SHOT_MODES,
    WORKFLOW_FILES,
)
from .source_contract import validate_workflow

ROOT = Path(__file__).resolve().parents[2]
POSITIVE_PROMPT_NODE = "30"
NEGATIVE_PROMPT_NODE = "31"
SAMPLER_NODE = "50"
SHIFT_NODE = "23"
CFG_NORM_NODE = "24"
OUTPUT_NODE = "70"
LOAD_NODES = {1: "1", 2: "2", 3: "3"}

# Generic negative terms only.  No human anatomy or anime-specific wording.
NEGATIVE_PROMPT = (
    "collage, contact sheet, multiple panels, text, watermark, "
    "inconsistent perspective, malformed structure"
)

_ROLE_ALIASES = {
    "character": "identity",
    "character_reference": "identity",
    "background_reference": "environment",
}
_ROLE_PHRASES = {
    "camera": "camera position",
    "lens": "lens perspective",
    "background": "background",
    "environment": "environment",
    "position": "subject position",
    "apparent_scale": "apparent subject scale",
    "pose": "pose",
    "gaze": "gaze/head direction",
    "identity": "subject identity",
    "style": "rendering style",
    "lighting": "lighting",
    "expression": "expression",
    "wardrobe": "wardrobe/design",
    "subject_count": "subject count",
    "framing": "framing",
    "composition": "composition",
}


def _normalize_shot_mode(shot_mode: str | None, mode: str | None) -> str:
    if shot_mode in SHOT_MODES:
        return shot_mode
    if shot_mode is not None:
        raise BindingError("shot_mode должен быть continue_shot или new_shot")
    if mode == "b":
        return "continue_shot"
    if mode == "a_new":
        return "new_shot"
    raise BindingError("Нужен shot_mode=continue_shot|new_shot")


def _validate_prompt_lists(invariants: Sequence[str], allowed_changes: Sequence[str]) -> None:
    inv = list(invariants)
    changes = list(allowed_changes)
    for name, values in (("invariants", inv), ("allowed_changes", changes)):
        for value in values:
            if value not in INVARIANT_VOCAB:
                raise BindingError(f"{name}: неизвестный токен {value!r}")
        if len(values) != len(set(values)):
            raise BindingError(f"{name}: повторяющийся токен")
    overlap = sorted(set(inv) & set(changes))
    if overlap:
        raise BindingError(f"invariants/allowed_changes пересекаются: {overlap}")


def _render_tokens(values: Sequence[str]) -> str:
    return ", ".join(_ROLE_PHRASES.get(value, value) for value in values)


def _negative_prompt(extra: str | None) -> str:
    if extra is None or not str(extra).strip():
        return NEGATIVE_PROMPT
    return f"{NEGATIVE_PROMPT}, {str(extra).strip()}"


def _role_sentence(
    index: int,
    role: str,
    shot_mode: str,
    *,
    identity_index: int | None = None,
    identity_total: int | None = None,
) -> str:
    role = _ROLE_ALIASES.get(role, role)
    if role == "anchor":
        if shot_mode == "continue_shot":
            return (
                f"Picture {index} is the current scene anchor for composition and "
                "continuity."
            )
        return (
            f"Picture {index} is a previous-shot composition reference; use it only "
            "as explicitly instructed."
        )
    if role == "identity":
        sentence = (
            f"Picture {index} defines subject identity and appearance only. Do not "
            "copy its layout, panels, captions, background, camera, or subject count."
        )
        if identity_total and identity_total > 1 and identity_index is not None:
            sentence += (
                f" This is identity reference {identity_index} of {identity_total}; "
                "use both identity references only for consistent identity, "
                "appearance and requested views."
            )
        return sentence
    if role == "style":
        return (
            f"Picture {index} defines rendering style only. Do not copy its "
            "composition, camera, subject count, or specific subject."
        )
    if role == "environment":
        return (
            f"Picture {index} defines the environment/world only; it does not define "
            "camera or subject placement."
        )
    raise BindingError(f"Неизвестная роль reference: {role}")


def stable_binding_hash(
    *,
    ref_count: int,
    roles: Sequence[str],
    edit_instruction: str,
    description: str,
    style: str,
    mode: str | None = None,
    shot_mode: str | None = None,
    invariants: Sequence[str] = (),
    allowed_changes: Sequence[str] = (),
    subject_label: str = "subject",
    negative_extra: str | None = None,
    role_map: Mapping[str, str] | None = None,
    seed: int,
    candidate_geometry: tuple[int, int] | list[int] | None = None,
    plan_sha256: str | None = None,
    parent_version: int | None = None,
    settings: Mapping[str, Any] | None = None,
    root: Path | str = ROOT,
) -> tuple[str, str, str, str]:
    workflow, workflow_sha, _path = load_variant(ref_count, root=root)
    width, height = _validate_geometry(candidate_geometry or (704, 1280))
    normalized_mode = _normalize_shot_mode(shot_mode, mode)
    _validate_prompt_lists(invariants, allowed_changes)
    positive = build_prompt(
        shot_mode=normalized_mode,
        edit_instruction=edit_instruction,
        description=description,
        style=style,
        roles=tuple(roles),
        role_map=role_map,
        invariants=tuple(invariants),
        allowed_changes=tuple(allowed_changes),
        subject_label=subject_label,
    )
    negative = _negative_prompt(negative_extra)
    normalized_settings = {
        "steps": int((settings or {}).get("steps", DEFAULT_STEPS)),
        "cfg": float((settings or {}).get("cfg", DEFAULT_CFG)),
        "sampler_name": str((settings or {}).get("sampler_name", DEFAULT_SAMPLER)),
        "scheduler": str((settings or {}).get("scheduler", DEFAULT_SCHEDULER)),
        "denoise": float((settings or {}).get("denoise", 1.0)),
        "shift": float((settings or {}).get("shift", DEFAULT_SHIFT)),
        "cfg_norm_strength": float((settings or {}).get("cfg_norm_strength", 1.0)),
    }
    digest = _sha(
        {
            "workflow_sha256": workflow_sha,
            "roles": list(roles),
            "positive": positive,
            "negative": negative,
            "settings": normalized_settings,
            "seed": int(seed),
            "candidate_geometry": [width, height],
            "plan_sha256": plan_sha256,
            "parent_version": parent_version,
            "shot_mode": normalized_mode,
            "invariants": list(invariants),
            "allowed_changes": list(allowed_changes),
            "subject_label": subject_label,
        }
    )
    return digest, positive, workflow_sha, negative


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def ordered_roles(
    *,
    anchor: str,
    character: str | None,
    environment: str | None,
) -> list[tuple[str, str]]:
    """Role order is fixed: anchor, original character, optional environment."""

    if not anchor:
        raise BindingError("anchor обязателен")
    roles: list[tuple[str, str]] = [("anchor", anchor)]
    if character:
        roles.append(("character", character))
    if environment:
        roles.append(("environment", environment))
    if len(roles) > 3:
        raise BindingError("Qwen EditPlus принимает максимум 3 reference")
    if not 1 <= len(roles) <= 3:
        raise BindingError("Нужно 1..3 reference")
    return roles


def _validate_semantic_roles(role_list: Sequence[str]) -> dict[str, int]:
    counts = {"anchor": 0, "identity": 0, "environment": 0, "style": 0}
    for role in role_list:
        if role not in counts:
            raise BindingError(f"Неизвестная роль reference: {role}")
        counts[role] += 1
    if not 1 <= len(role_list) <= 3:
        raise BindingError("Нужно 1..3 reference")
    if counts["anchor"] > 1:
        raise BindingError("anchor допускается только один")
    if counts["identity"] > 2:
        raise BindingError("identity references: максимум 2")
    if counts["environment"] > 1:
        raise BindingError("environment reference: максимум 1")
    if counts["style"] > 1:
        raise BindingError("style reference: максимум 1")
    return counts


def _role_sentences(roles: Sequence[str], shot_mode: str) -> str:
    counts = _validate_semantic_roles(roles)
    identity_total = counts["identity"]
    identity_index = 0
    sentences = []
    for index, role in enumerate(roles, start=1):
        if role == "identity":
            identity_index += 1
            sentences.append(
                _role_sentence(
                    index,
                    role,
                    shot_mode,
                    identity_index=identity_index,
                    identity_total=identity_total,
                )
            )
        else:
            sentences.append(_role_sentence(index, role, shot_mode))
    return " ".join(sentences)


def build_prompt(
    *,
    shot_mode: str | None = None,
    mode: str | None = None,
    edit_instruction: str,
    description: str,
    style: str,
    roles: Sequence[str],
    role_map: Mapping[str, str] | None = None,
    invariants: Sequence[str] = (),
    allowed_changes: Sequence[str] = (),
    subject_label: str = "subject",
) -> str:
    normalized_mode = _normalize_shot_mode(shot_mode, mode)
    _validate_prompt_lists(invariants, allowed_changes)
    if not edit_instruction.strip():
        raise BindingError("edit_instruction обязателен для генерации")
    if "{{" in edit_instruction or "KF1_BIND_REQUIRED" in edit_instruction:
        raise BindingError("edit_instruction содержит placeholder")
    role_list = [
        (role_map or {}).get(role, _ROLE_ALIASES.get(role, role)) for role in roles
    ]
    if len(roles) != len(set(roles)):
        raise BindingError("Повторяющийся reference id")
    _validate_semantic_roles(role_list)
    if "anchor" in role_list and role_list[0] != "anchor":
        raise BindingError("anchor должен быть Picture 1")
    lead = "Create one clean production keyframe."
    if normalized_mode == "continue_shot":
        lead += " Continue the current shot."
    else:
        lead += " Establish a new shot exactly as instructed."
    context = (
        f" Scene context (preserve its meaning): {description.strip()}."
        if description.strip()
        else ""
    )
    style_text = f" Style/constraints: {style.strip()}." if style.strip() else ""
    subject_text = (
        f" Subject: {subject_label.strip()}."
        if subject_label.strip() and subject_label.strip() != "subject"
        else ""
    )
    keep_text = (
        f" Keep: {_render_tokens(invariants)}." if invariants else ""
    )
    change_text = (
        f" Change only: {_render_tokens(allowed_changes)}."
        if allowed_changes
        else " Change only what the exact instruction states."
    )
    refs = _role_sentences(role_list, normalized_mode)
    prompt = (
        f"{lead}{context}{style_text}{subject_text}{keep_text}{change_text} "
        f"Exact instruction: {edit_instruction.strip()} "
        "Keep coherent perspective and one coherent image. "
        "Output one single image without panels, captions or duplicates."
    )
    if refs:
        prompt += " " + refs
    if "{{" in prompt or "KF1_BIND_REQUIRED" in prompt:
        raise BindingError("Собранный prompt всё ещё содержит placeholder")
    return prompt


def workflow_path(ref_count: int, *, root: Path | str = ROOT) -> Path:
    try:
        relative = WORKFLOW_FILES[int(ref_count)]
    except KeyError as exc:
        raise BindingError("Нужно 1..3 reference") from exc
    path = Path(root) / relative
    if not path.is_file():
        raise BindingError(f"Нет derived workflow: {path}")
    return path


def _validate_geometry(value: tuple[int, int] | list[int]) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise BindingError("candidate_geometry должен быть [width, height]")
    width, height = int(value[0]), int(value[1])
    if not 64 <= width <= 4096 or not 64 <= height <= 4096:
        raise BindingError("candidate_geometry вне 64..4096")
    if width % 8 or height % 8:
        raise BindingError("candidate_geometry должен быть кратен 8")
    return width, height


def load_variant(ref_count: int, *, root: Path | str = ROOT) -> tuple[dict[str, Any], str, Path]:
    path = workflow_path(ref_count, root=root)
    workflow, digest = load_workflow(path)
    return workflow, digest, path


def bind(
    *,
    ref_count: int,
    image_names: Mapping[str, str],
    edit_instruction: str,
    description: str,
    style: str,
    mode: str | None = None,
    shot_mode: str | None = None,
    invariants: Sequence[str] = (),
    allowed_changes: Sequence[str] = (),
    subject_label: str = "subject",
    negative_extra: str | None = None,
    role_map: Mapping[str, str] | None = None,
    roles: Sequence[str] | None = None,
    seed: int,
    version: int,
    plan_sha256: str,
    scene_id: str,
    role: str,
    candidate_geometry: tuple[int, int] | list[int] | None = None,
    parent_version: int | None = None,
    settings: Mapping[str, Any] | None = None,
    root: Path | str = ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow, workflow_sha, path = load_variant(ref_count, root=root)
    normalized_mode = _normalize_shot_mode(shot_mode, mode)
    _validate_prompt_lists(invariants, allowed_changes)
    if roles is None:
        expected_roles = ["anchor"]
        if "character" in image_names:
            expected_roles.append("character")
        if "environment" in image_names:
            expected_roles.append("environment")
    else:
        expected_roles = list(roles)
    if len(expected_roles) != ref_count:
        raise BindingError("ref_count не соответствует ролям")
    if not 1 <= len(expected_roles) <= 3:
        raise BindingError("Нужно 1..3 reference")
    if len(expected_roles) != len(set(expected_roles)):
        raise BindingError("Повторяющаяся роль reference")
    if "anchor" in expected_roles and expected_roles[0] != "anchor":
        raise BindingError("anchor должен быть Picture 1")
    for role_name in expected_roles:
        if role_name not in image_names:
            raise BindingError(f"Нет upload name для роли {role_name}")
    safe_names = {name: safe_remote_filename(value) for name, value in image_names.items()}
    positive = build_prompt(
        shot_mode=normalized_mode,
        edit_instruction=edit_instruction,
        description=description,
        style=style,
        roles=expected_roles,
        role_map=role_map,
        invariants=invariants,
        allowed_changes=allowed_changes,
        subject_label=subject_label,
    )
    negative = _negative_prompt(negative_extra)
    bound = copy.deepcopy(workflow)
    bound.pop("_meta", None)
    bound.pop("_template", None)
    width, height = _validate_geometry(candidate_geometry or (704, 1280))
    scale = bound.get("10")
    if not isinstance(scale, dict) or scale.get("class_type") != "ImageScale":
        raise BindingError("В workflow нет ImageScale 10")
    scale["inputs"]["width"] = width
    scale["inputs"]["height"] = height
    scale["inputs"]["crop"] = "center"
    for slot in (1, 2, 3):
        node = bound.get(LOAD_NODES[slot])
        if slot <= ref_count:
            if not isinstance(node, dict) or node.get("class_type") != "LoadImage":
                raise BindingError(f"В workflow нет LoadImage node {slot}")
            role_name = expected_roles[slot - 1]
            node["inputs"]["image"] = safe_names[role_name]
        elif node is not None:
            # The exact 1/2/3 variant should not carry extra LoadImage nodes.
            raise BindingError(f"Workflow {ref_count}ref содержит лишний LoadImage {slot}")
    for node_id in (POSITIVE_PROMPT_NODE, NEGATIVE_PROMPT_NODE):
        node = bound.get(node_id)
        if not isinstance(node, dict) or node.get("class_type") != "TextEncodeQwenImageEditPlus":
            raise BindingError(f"Нет TextEncodeQwenImageEditPlus {node_id}")
        node["inputs"]["prompt"] = positive if node_id == POSITIVE_PROMPT_NODE else negative
    sampler = bound.get(SAMPLER_NODE)
    if not isinstance(sampler, dict) or sampler.get("class_type") != "KSampler":
        raise BindingError("Нет KSampler")
    sampler["inputs"].update(
        {
            "seed": int(seed),
            "steps": int((settings or {}).get("steps", DEFAULT_STEPS)),
            "cfg": float((settings or {}).get("cfg", DEFAULT_CFG)),
            "sampler_name": str((settings or {}).get("sampler_name", DEFAULT_SAMPLER)),
            "scheduler": str((settings or {}).get("scheduler", DEFAULT_SCHEDULER)),
            "denoise": float((settings or {}).get("denoise", 1.0)),
        }
    )
    shift = bound.get(SHIFT_NODE)
    if not isinstance(shift, dict) or shift.get("class_type") != "ModelSamplingAuraFlow":
        raise BindingError("Нет ModelSamplingAuraFlow")
    shift["inputs"]["shift"] = float((settings or {}).get("shift", DEFAULT_SHIFT))
    cfg_norm = bound.get(CFG_NORM_NODE)
    if not isinstance(cfg_norm, dict) or cfg_norm.get("class_type") != "CFGNorm":
        raise BindingError("Нет CFGNorm")
    cfg_norm["inputs"]["strength"] = float((settings or {}).get("cfg_norm_strength", 1.0))
    output = bound.get(OUTPUT_NODE)
    if not isinstance(output, dict) or output.get("class_type") != "SaveImage":
        raise BindingError("Нет SaveImage")
    safe_scene = "".join(char if char.isalnum() or char in "._-" else "_" for char in scene_id)
    safe_role = "".join(char if char.isalnum() or char in "._-" else "_" for char in role)
    prefix = f"anime-studio/kf1/{plan_sha256[:12]}/{safe_scene}/{safe_role}/v{int(version):04d}"
    output["inputs"]["filename_prefix"] = prefix
    issues = validate_workflow(bound)
    if issues:
        raise BindingError("; ".join(issues))
    payload = json.dumps(bound, ensure_ascii=False)
    if "{{" in payload or "KF1_BIND_REQUIRED" in payload:
        raise BindingError("Bound workflow содержит placeholder")
    binding = {
        "schema": "anime-studio/kf1-binding/v1",
        "workflow_path": str(path),
        "workflow_sha256": workflow_sha,
        "ref_count": ref_count,
        "roles": expected_roles,
        "role_semantics": [
            (role_map or {}).get(name, _ROLE_ALIASES.get(name, name))
            for name in expected_roles
        ],
        "uploads": [{"role": name, "remote_name": safe_names[name]} for name in expected_roles],
        "positive_prompt": positive,
        "negative_prompt": negative,
        "edit_instruction": edit_instruction,
        "description": description,
        "style": style,
        "shot_mode": normalized_mode,
        "mode": normalized_mode,
        "invariants": list(invariants),
        "allowed_changes": list(allowed_changes),
        "subject_label": subject_label,
        "negative_extra": negative_extra,
        "seed": int(seed),
        "settings": {
            "steps": int(sampler["inputs"]["steps"]),
            "cfg": float(sampler["inputs"]["cfg"]),
            "sampler_name": str(sampler["inputs"]["sampler_name"]),
            "scheduler": str(sampler["inputs"]["scheduler"]),
            "denoise": float(sampler["inputs"]["denoise"]),
            "shift": float(shift["inputs"]["shift"]),
            "cfg_norm_strength": float(cfg_norm["inputs"]["strength"]),
        },
        "output_node": OUTPUT_NODE,
        "output_keys": ["images"],
        "media_type": "image/png",
        "filename_prefix": prefix,
        "expected_output": {
            "node_id": OUTPUT_NODE,
            "keys": ["images"],
            "subfolder": prefix.rsplit("/", 1)[0],
            "filename_prefix": prefix.rsplit("/", 1)[-1],
            "version": int(version),
        },
        "stable_binding_sha256": _sha(
            {
                "workflow_sha256": workflow_sha,
                "roles": expected_roles,
                "positive": positive,
                "negative": negative,
                "settings": {
                    "steps": int(sampler["inputs"]["steps"]),
                    "cfg": float(sampler["inputs"]["cfg"]),
                    "sampler_name": str(sampler["inputs"]["sampler_name"]),
                    "scheduler": str(sampler["inputs"]["scheduler"]),
                    "denoise": float(sampler["inputs"]["denoise"]),
                    "shift": float(shift["inputs"]["shift"]),
                    "cfg_norm_strength": float(cfg_norm["inputs"]["strength"]),
                },
                "seed": seed,
                "candidate_geometry": [width, height],
                "plan_sha256": plan_sha256,
                "parent_version": parent_version,
                "shot_mode": normalized_mode,
                "invariants": list(invariants),
                "allowed_changes": list(allowed_changes),
                "subject_label": subject_label,
            }
        ),
        "binding_sha256": _sha(
            {
                "workflow": bound,
                "workflow_sha256": workflow_sha,
                "uploads": expected_roles,
                "positive": positive,
                "negative": negative,
                "settings": sampler["inputs"],
                "seed": seed,
            }
        ),
    }
    return bound, binding


def verify_saved_binding(
    binding: Mapping[str, Any],
    *,
    plan_sha256: str,
    parent_version: int,
    candidate_geometry: tuple[int, int] | list[int],
    expected_workflow_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify an immutable saved binding from its actual payload.

    This is used for v1 recovery/export/approval.  It never rebuilds the old
    prompt from current source text; it hashes the saved positive/negative,
    roles, settings, workflow and bound graph exactly as stored.
    """

    if not isinstance(binding, Mapping):
        raise BindingError("saved binding не объект")
    bound = binding.get("bound_workflow")
    if not isinstance(bound, Mapping) or not bound:
        raise BindingError("saved binding без bound_workflow")
    roles = binding.get("roles")
    if not isinstance(roles, list) or not 1 <= len(roles) <= 3:
        raise BindingError("saved binding roles некорректны")
    if len(roles) != len(set(str(item) for item in roles)):
        raise BindingError("saved binding roles повторяются")
    ref_count = binding.get("ref_count")
    if ref_count != len(roles):
        raise BindingError("saved binding ref_count не совпадает с roles")
    workflow_sha = str(binding.get("workflow_sha256") or "")
    if expected_workflow_sha256 is not None and workflow_sha != expected_workflow_sha256:
        raise BindingError("saved binding workflow hash не совпадает с pin")
    positive = binding.get("positive_prompt")
    negative = binding.get("negative_prompt")
    if not isinstance(positive, str) or not isinstance(negative, str):
        raise BindingError("saved binding prompt/negative некорректны")
    if "{{" in positive or "KF1_BIND_REQUIRED" in positive:
        raise BindingError("saved binding positive содержит placeholder")
    if "{{" in negative or "KF1_BIND_REQUIRED" in negative:
        raise BindingError("saved binding negative содержит placeholder")
    settings = binding.get("settings")
    if not isinstance(settings, Mapping):
        raise BindingError("saved binding settings некорректны")
    normalized_settings = {
        "steps": int(settings.get("steps", DEFAULT_STEPS)),
        "cfg": float(settings.get("cfg", DEFAULT_CFG)),
        "sampler_name": str(settings.get("sampler_name", DEFAULT_SAMPLER)),
        "scheduler": str(settings.get("scheduler", DEFAULT_SCHEDULER)),
        "denoise": float(settings.get("denoise", 1.0)),
        "shift": float(settings.get("shift", DEFAULT_SHIFT)),
        "cfg_norm_strength": float(settings.get("cfg_norm_strength", 1.0)),
    }
    width, height = _validate_geometry(candidate_geometry)
    stable_payload = {
        "workflow_sha256": workflow_sha,
        "roles": roles,
        "positive": positive,
        "negative": negative,
        "settings": normalized_settings,
        "seed": int(binding.get("seed", 0)),
        "candidate_geometry": [width, height],
        "plan_sha256": plan_sha256,
        "parent_version": parent_version,
    }
    if "shot_mode" in binding:
        stable_payload.update(
            {
                "shot_mode": binding.get("shot_mode"),
                "invariants": list(binding.get("invariants") or []),
                "allowed_changes": list(binding.get("allowed_changes") or []),
                "subject_label": binding.get("subject_label", "subject"),
            }
        )
    stable_expected = _sha(stable_payload)
    if stable_expected != binding.get("stable_binding_sha256"):
        raise BindingError(
            "saved stable_binding_sha256 не совпадает с actual saved payload"
        )
    sampler = bound.get(SAMPLER_NODE)
    if not isinstance(sampler, Mapping):
        raise BindingError("saved bound workflow без KSampler")
    sampler_inputs = sampler.get("inputs")
    if not isinstance(sampler_inputs, Mapping):
        raise BindingError("saved bound KSampler inputs некорректны")
    full_expected = _sha(
        {
            "workflow": bound,
            "workflow_sha256": workflow_sha,
            "uploads": roles,
            "positive": positive,
            "negative": negative,
            "settings": sampler_inputs,
            "seed": int(binding.get("seed", 0)),
        }
    )
    if full_expected != binding.get("binding_sha256"):
        raise BindingError(
            "saved binding_sha256 не совпадает с actual saved payload/graph"
        )
    uploads = binding.get("uploads")
    if not isinstance(uploads, list) or len(uploads) != len(roles):
        raise BindingError("saved binding uploads некорректны")
    for index, role in enumerate(roles, start=1):
        upload = uploads[index - 1]
        if not isinstance(upload, Mapping) or upload.get("role") != role:
            raise BindingError("saved binding upload role order не совпадает")
        name = upload.get("name") or upload.get("remote_name")
        node = bound.get(LOAD_NODES[index])
        if (
            not isinstance(node, Mapping)
            or node.get("class_type") != "LoadImage"
            or node.get("inputs", {}).get("image") != name
        ):
            raise BindingError("saved binding LoadImage не совпадает с upload order")
    for node_id in (POSITIVE_PROMPT_NODE, NEGATIVE_PROMPT_NODE):
        node = bound.get(node_id)
        if not isinstance(node, Mapping) or node.get("class_type") != "TextEncodeQwenImageEditPlus":
            raise BindingError("saved binding без TextEncodeQwenImageEditPlus")
        expected_prompt = positive if node_id == POSITIVE_PROMPT_NODE else negative
        if node.get("inputs", {}).get("prompt") != expected_prompt:
            raise BindingError("saved binding prompt node не совпадает")
    issues = validate_workflow(bound)
    if issues:
        raise BindingError("; ".join(issues))
    return {
        "roles": roles,
        "workflow_sha256": workflow_sha,
        "stable_binding_sha256": stable_expected,
        "binding_sha256": full_expected,
        "ref_count": len(roles),
    }


__all__ = [
    "NEGATIVE_PROMPT",
    "build_prompt",
    "bind",
    "verify_saved_binding",
    "load_variant",
    "ordered_roles",
    "workflow_path",
]

