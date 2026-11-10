"""Plain-text KF2 import: UTF-8/BOM, multiline Russian and numbered proposals.

The parser never translates text and never invents camera/action fields.  A
numbered list produces a proposal with descriptions only; the user confirms and
edits it before a plan exists.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .errors import TextImportError
from .paths import sha256_bytes

TERMINATOR = "---"
LITERAL_TERMINATOR = "\\---"
NUMBERED = re.compile(r"^\s*(\d+)[.)]\s*(.*)$")
KEY_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]*):(.*)$")
MULTILINE_KEYS = {
    "title",
    "description",
    "action",
    "camera",
    "style",
    "notes",
    "a_instruction",
    "b_instruction",
    "subject_label",
}
SCENE_KEYS = {
    "description",
    "action",
    "camera",
    "a_instruction",
    "b_instruction",
    "subjects_visible",
    "subjects_absent",
    "shared_invariants",
    "shared_allowed_changes",
}


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]


def _new_scene(index: int) -> dict[str, Any]:
    return {
        "id": f"shot-{index:02d}",
        "description": "",
        "action": "",
        "camera": "",
        "subjects_visible": [],
        "subjects_absent": [],
        "shared_state": "",
        "shared_invariants": [],
        "shared_allowed_changes": [],
        "initial": {
            "kind": "first_frame" if index == 1 else "new_shot",
            "asset_id": None,
            "instruction": "",
            "instruction_mode": "manual",
            "state_a": "",
            "references": [],
            "invariants": [],
            "allowed_changes": [],
            "use_previous_anchor": False,
            "require_anchor_approval": True,
        },
        "final": {
            "kind": "continue",
            "asset_id": None,
            "instruction": "",
            "instruction_mode": "manual",
            "state_b": "",
            "references": [],
            "invariants": [],
            "allowed_changes": [],
            "require_anchor_approval": False if index == 1 else True,
        },
    }


def _apply_scene_value(scene: dict[str, Any], key: str, value: str) -> None:
    if key in {"description", "action", "camera"}:
        scene[key] = value
    elif key == "a_instruction":
        scene["initial"]["instruction"] = value
    elif key == "b_instruction":
        scene["final"]["instruction"] = value
    elif key == "subjects_visible":
        scene["subjects_visible"] = _split_list(value)
    elif key == "subjects_absent":
        scene["subjects_absent"] = _split_list(value)
    elif key == "shared_invariants":
        scene["shared_invariants"] = _split_list(value)
    elif key == "shared_allowed_changes":
        scene["shared_allowed_changes"] = _split_list(value)


def _parse_keyvalue(text: str) -> dict[str, Any]:
    lines = text.split("\n")
    fields: dict[str, Any] = {}
    scenes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    warnings: list[str] = []
    unparsed: list[str] = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue
        match = KEY_LINE.match(raw)
        if not match:
            unparsed.append(raw)
            warnings.append(f"строка без key: сохранена как unparsed: {stripped!r}")
            index += 1
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        if key == "scene":
            current = _new_scene(len(scenes) + 1)
            if value:
                current["id"] = value
            scenes.append(current)
            index += 1
            continue
        if not value and key in MULTILINE_KEYS:
            next_is_key = False
            if index + 1 < len(lines):
                next_match = KEY_LINE.match(lines[index + 1])
                if next_match:
                    next_key = next_match.group(1).strip().lower().replace(" ", "_")
                    next_is_key = next_key in {
                        "scene", "title", "description", "action", "camera", "style",
                        "notes", "a_instruction", "b_instruction", "subject_label",
                        "subject_kind", "subjects_visible", "subjects_absent",
                        "shared_invariants", "shared_allowed_changes", "orientation",
                        "generation_limit", "seed", "subject", "subjects",
                    }
            if next_is_key:
                index += 1
                value = ""
                terminated = True
                collected = []
            else:
                collected = []
                index += 1
                terminated = False
                while index < len(lines):
                    candidate = lines[index]
                    if candidate.strip() == TERMINATOR:
                        terminated = True
                        index += 1
                        break
                    if candidate.rstrip() == LITERAL_TERMINATOR:
                        collected.append(TERMINATOR)
                    else:
                        collected.append(candidate)
                    index += 1
            value = "\n".join(collected).strip("\n")
            if not terminated:
                warnings.append(f"multiline {key} ??? terminator {TERMINATOR}")
        else:
            index += 1
        if key in SCENE_KEYS:
            if current is None:
                warnings.append(f"{key} вне scene проигнорирован как proposal")
                continue
            _apply_scene_value(current, key, value)
            continue
        if key in {"title", "orientation", "style", "notes", "subject_label", "subject_kind"}:
            fields[key] = value
            continue
        if key in {"generation_limit", "seed"}:
            try:
                fields[key] = int(value)
            except ValueError:
                warnings.append(f"{key} не число; сохранён как текст")
                fields[key] = value
            continue
        if key in {"subject", "subjects"}:
            fields[key] = _split_list(value)
            continue
        warnings.append(f"неизвестный key {key!r}; сохранён в extras")
        fields.setdefault("extras", {})[key] = value
    if not scenes:
        warnings.append("key-value текст не содержит scene:")
    return {
        "mode": "keyvalue",
        "fields": fields,
        "scenes": scenes,
        "warnings": warnings,
        "unparsed": unparsed,
    }


def _parse_numbered(text: str) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    warnings: list[str] = []
    current: dict[str, Any] | None = None
    for raw in text.split("\n"):
        match = NUMBERED.match(raw)
        if match:
            current = _new_scene(len(scenes) + 1)
            scenes.append(current)
            if match.group(2).strip():
                current["description"] = match.group(2).strip()
            continue
        if current is None:
            if raw.strip():
                warnings.append("текст до первого номера сохранён как unparsed")
            continue
        if raw.strip():
            current["description"] = (
                current["description"] + "\n" + raw.rstrip()
            ).strip("\n")
    if not scenes:
        warnings.append("numbered parser не нашёл строк вида '1.' / '2.'")
    return {
        "mode": "numbered",
        "fields": {},
        "scenes": scenes,
        "warnings": warnings,
        "unparsed": [],
    }


def parse_text_document(data: bytes, *, numbered: bool = False) -> dict[str, Any]:
    if not data:
        raise TextImportError("текстовый файл пуст")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TextImportError("текст должен быть UTF-8 (BOM допускается)") from exc
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    parsed = _parse_numbered(normalised) if numbered else _parse_keyvalue(normalised)
    parsed.update(
        {
            "schema": "anime-studio/kf2-text-proposal/v1",
            "source_sha256": sha256_bytes(data),
            "source_bytes": len(data),
            "original_text_sha256": sha256_bytes(text.encode("utf-8")),
        }
    )
    return parsed


def apply_proposal(project: dict[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only explicit proposal fields; never synthesize narrative."""

    fields = proposal.get("fields")
    if isinstance(fields, Mapping):
        for key in ("title", "orientation", "style", "notes", "subject_label"):
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                project[key] = value.strip()
        for key in ("generation_limit", "seed"):
            value = fields.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                project[key] = value
        orientation = fields.get("orientation")
        if orientation == "portrait":
            project["orientation"] = "portrait"
            project["candidate_geometry"] = [704, 1280]
            project["video_geometry"] = [448, 816]
        elif orientation == "landscape":
            project["orientation"] = "landscape"
            project["candidate_geometry"] = [1280, 704]
            project["video_geometry"] = [816, 448]
    scenes = project.setdefault("scenes", [])
    by_id = {str(scene.get("id")): scene for scene in scenes if isinstance(scene, dict)}
    next_index = len(scenes) + 1
    for proposed in proposal.get("scenes") or []:
        if not isinstance(proposed, Mapping):
            continue
        scene_id = str(proposed.get("id") or f"shot-{next_index:02d}")
        scene = by_id.get(scene_id)
        if scene is None:
            scene = _new_scene(next_index)
            scene["id"] = scene_id
            scenes.append(scene)
            by_id[scene_id] = scene
            next_index += 1
        for key in ("description", "action", "camera"):
            value = proposed.get(key)
            if isinstance(value, str) and value:
                scene[key] = value
        defaults = _new_scene(1)
        scene.setdefault("initial", defaults["initial"])
        scene.setdefault("final", defaults["final"])
        for source_key, target_key in (("initial", "initial"), ("final", "final")):
            source = proposed.get(source_key)
            if isinstance(source, Mapping):
                value = source.get("instruction")
                if isinstance(value, str) and value:
                    scene[target_key]["instruction"] = value
                    # An explicit non-empty imported instruction is a raw manual
                    # override; a previous states-mode flag must not hide it.
                    scene[target_key]["instruction_mode"] = "manual"
        for key in ("subjects_visible", "subjects_absent", "shared_invariants", "shared_allowed_changes"):
            value = proposed.get(key)
            if isinstance(value, list):
                scene[key] = [str(item) for item in value if str(item).strip()]
    for index, scene in enumerate(scenes):
        final = scene.get("final")
        if isinstance(final, dict) and final.get("require_anchor_approval") is None:
            final["require_anchor_approval"] = False if index == 0 else True
    project["import"] = {
        "kind": "text",
        "mode": proposal.get("mode"),
        "source_sha256": proposal.get("source_sha256"),
        "proposal_schema": proposal.get("schema"),
        "needs_confirmation": True,
    }
    return project


__all__ = [
    "LITERAL_TERMINATOR",
    "TERMINATOR",
    "apply_proposal",
    "parse_text_document",
]
