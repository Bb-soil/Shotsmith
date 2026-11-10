"""Russian, offline KF2 project-authoring CLI.

The authoring commands never create a Comfy client and never perform network,
SSH, GPU, upload or POST work.  `prepare` publishes a stable plan and reuses the
existing KF1 dry-run; the real ledger/approval path remains KF1-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from scripts.kf1.plan import resolve_plan

from .assets import (
    IMAGE_ROLES,
    SUBJECT_KINDS,
    asset_usage,
    change_asset_role,
    import_image,
    import_script,
    read_provenance_file,
)
from .compile import (
    ORIENTATIONS,
    _compose_state_instruction,
    validate_and_compile,
)
from .errors import KF2Error, ProjectConflict, ProjectError, ProjectNotFound
from .paths import (
    DEFAULT_PROJECTS_DIR,
    MAX_PLAN_BYTES,
    ROOT,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    checked_exists,
    checked_unlink,
    ensure_managed_dir,
    is_reparse,
    read_bytes_limited,
    sha256_bytes,
    validate_local_destination,
)
from .store import ProjectStore, _validate_old_plan_bytes, authoring_digest, now_utc
from .text_import import apply_proposal, parse_text_document

DEFAULT_KF1_STATE = ROOT / "data" / "kf1"
CLEAR = "-"
LITERAL_CLEAR = "\\-"
ROLE_LABELS = {
    "character": "персонаж",
    "creature": "существо",
    "object": "объект",
    "environment": "окружение",
    "style": "стиль",
    "provided_keyframe": "готовый ключевой кадр",
}
KIND_LABELS = {
    "first_frame": "A1: готовый первый кадр",
    "new_shot": "A: новый кадр",
    "provided_keyframe": "готовый ключевой кадр",
    "continue": "B: продолжение с той же камерой",
}
SEMANTIC_BY_ASSET_ROLE = {
    "character": "identity",
    "creature": "identity",
    "object": "identity",
    "environment": "environment",
    "style": "style",
}
ROLE_BY_SEMANTIC = {
    "identity": "identity",
    "environment": "environment",
    "style": "style",
}


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _store(args: argparse.Namespace) -> ProjectStore:
    return ProjectStore(args.projects_dir)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--projects-dir", default=None)
    parser.add_argument("--json", action="store_true")


def _short(value: str, limit: int = 120) -> str:
    value = value.replace("\n", "\\n")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _ask(
    prompt: str,
    *,
    default: str = "",
    input_fn: Callable[[str], str] | None = None,
    allow_clear: bool = False,
) -> str:
    if input_fn is None:
        input_fn = input
    suffix = f" [{_short(default)}]" if default else ""
    value = input_fn(f"{prompt}{suffix}: ").strip()
    if allow_clear and value == CLEAR:
        return ""
    if value == LITERAL_CLEAR:
        return CLEAR
    return value or default


def _ask_yes(
    prompt: str,
    *,
    default: bool,
    input_fn: Callable[[str], str] | None = None,
) -> bool:
    value = _ask(
        prompt + " (y/n)",
        default="y" if default else "n",
        input_fn=input_fn,
    ).lower()
    return value in {"y", "yes", "д", "да"}


def _ask_list(
    prompt: str,
    current: list[str],
    *,
    input_fn: Callable[[str], str] | None = None,
    allow_clear: bool = True,
) -> list[str]:
    value = _ask(
        prompt,
        default=", ".join(str(item) for item in current),
        input_fn=input_fn,
        allow_clear=allow_clear,
    )
    if value in {"", CLEAR}:
        return []
    if value == LITERAL_CLEAR:
        return [CLEAR]
    return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]


def _ask_multiline(
    prompt: str,
    current: str,
    *,
    input_fn: Callable[[str], str] | None = None,
    on_change: Callable[[str], None] | None = None,
) -> str:
    """Real console multiline: --- finishes; \\--- is literal; # is kept."""

    if input_fn is None:
        input_fn = input
    print(f"{prompt}; текущее: {_short(current) if current else '<пусто>'}")
    print("Ввод: --- завершить, \\--- literal, # сохраняется, - очистить, Enter оставить.")
    first = input_fn(f"{prompt} (строка 1): ")
    if first == CLEAR:
        if on_change is not None:
            on_change("")
        return ""
    if first == "":
        return current
    lines = [first]
    value = "\n".join(lines)
    if on_change is not None:
        on_change(value)
    while True:
        line = input_fn("... ")
        if line.strip() == "---":
            break
        if line.rstrip() == "\\---":
            lines.append("---")
        else:
            lines.append(line)
        value = "\n".join(lines)
        if on_change is not None:
            on_change(value)
    return value


def _auto_id() -> str:
    return "proj-" + uuid.uuid4().hex[:20]


def _save(store: ProjectStore, project: dict[str, Any]) -> dict[str, Any]:
    saved = store.save(
        str(project["id"]), project, expected_revision=int(project["revision"])
    )
    project.clear()
    project.update(saved)
    return project


def _subject(project: Mapping[str, Any], subject_id: str) -> dict[str, Any] | None:
    for subject in project.get("subjects") or []:
        if subject.get("id") == subject_id:
            return subject
    return None


def _ensure_subject(
    project: dict[str, Any], subject_id: str, *, kind: str, label: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", subject_id):
        raise ProjectError("subject ID: lower-case safe id")
    if kind not in SUBJECT_KINDS:
        raise ProjectError("subject kind должен быть character/creature/object/environment")
    existing = _subject(project, subject_id)
    if existing is not None:
        if existing.get("kind") != kind:
            raise ProjectError(f"subject {subject_id} уже имеет kind={existing.get('kind')}")
        return existing
    subject = {"id": subject_id, "label": label, "kind": kind, "views": []}
    project.setdefault("subjects", []).append(subject)
    return subject


def _asset_by_id(project: Mapping[str, Any], asset_id: str) -> dict[str, Any] | None:
    for item in project.get("library") or []:
        if item.get("id") == asset_id:
            return item
    return None


def _add_image_to_project(
    project: dict[str, Any],
    project_dir: Path,
    *,
    role: str,
    path: str,
    subject_id: str | None = None,
    subject_label: str | None = None,
    view: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if role in {"character", "creature", "object"}:
        if role != "object" and not subject_id:
            raise ProjectError(f"роль {role} требует --subject")
        if subject_id:
            _ensure_subject(
                project,
                subject_id,
                kind=role,
                label=subject_label or subject_id,
            )
    metadata = import_image(
        project_dir,
        path,
        role=role,
        subject_id=subject_id,
        view=view,
        provenance=provenance,
        library=project.get("library") or [],
        scripts=project.get("scripts") or [],
    )
    metadata["imported_utc"] = now_utc()
    existing = _asset_by_id(project, str(metadata["id"]))
    if existing is not None:
        for key in ("sha256", "stored_path", "role", "subject_id", "view"):
            if existing.get(key) != metadata.get(key):
                raise ProjectError(f"asset association collision: {metadata['id']}")
        if provenance is not None:
            prov = dict(metadata.get("provenance") or {})
            if str(prov.get("content_sha256", "")).lower() != str(existing.get("sha256", "")).lower():
                raise ProjectError("provenance.content_sha256 не совпадает с asset SHA")
            existing["provenance"] = prov
        return existing
    project.setdefault("library", []).append(metadata)
    if subject_id:
        subject = _subject(project, subject_id)
        assert subject is not None
        if view and not any(item.get("asset_id") == metadata["id"] for item in subject.setdefault("views", [])):
            subject["views"].append({"asset_id": metadata["id"], "label": view})
    return metadata


def _refs_text(project: Mapping[str, Any], refs: list[Mapping[str, Any]]) -> str:
    return ", ".join(f"{item.get('asset_id')}:{item.get('role')}" for item in refs)


def _parse_refs(project: Mapping[str, Any], text: str) -> list[dict[str, str]]:
    if text.strip() == CLEAR:
        return []
    assets = {item.get("id"): item for item in project.get("library") or []}
    result: list[dict[str, str]] = []
    for raw in text.replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            asset_id, role = item.split(":", 1)
        else:
            asset_id, role = item, ""
        asset_id = asset_id.strip()
        role = role.strip().lower()
        asset = assets.get(asset_id)
        if asset is None:
            raise ProjectError(f"unknown asset в references: {asset_id}")
        if not role:
            role = {
                "character": "identity",
                "creature": "identity",
                "object": "identity",
                "environment": "environment",
                "style": "style",
            }.get(str(asset.get("role")), "")
        if role not in {"identity", "style", "environment"}:
            raise ProjectError(
                f"reference {asset_id}: role должна быть identity/style/environment"
            )
        result.append({"asset_id": asset_id, "role": role})
    return result


def _asset_display(project: Mapping[str, Any], asset_id: str) -> str:
    asset = _asset_by_id(project, asset_id)
    if asset is None:
        return asset_id
    subject = ""
    if asset.get("subject_id"):
        item = _subject(project, str(asset["subject_id"]))
        subject = str((item or {}).get("label") or asset["subject_id"])
    view = f" view={asset.get('view')}" if asset.get("view") else ""
    subject_text = f" subject={subject}" if subject else ""
    return (
        f"{asset_id} \"{asset.get('original_name')}\" "
        f"role={ROLE_LABELS.get(str(asset.get('role')), asset.get('role'))}"
        f"{subject_text}{view}"
    )


def _reference_semantic(asset: Mapping[str, Any]) -> str | None:
    return SEMANTIC_BY_ASSET_ROLE.get(str(asset.get("role")))


def _reference_choice_issues(
    project: Mapping[str, Any],
    refs: list[Mapping[str, Any]],
    *,
    anchor_active: bool,
) -> list[str]:
    issues: list[str] = []
    total = len(refs) + (1 if anchor_active else 0)
    if total < 1 or total > 3:
        issues.append(f"итого references включая anchor: {total}; допустимо 1..3")
    counts = {"identity": 0, "environment": 0, "style": 0}
    seen_assets: set[str] = set()
    identity_shas: dict[str, str] = {}
    assets = {item.get("id"): item for item in project.get("library") or []}
    for ref in refs:
        asset_id = str(ref.get("asset_id"))
        asset = assets.get(asset_id)
        if asset is None:
            issues.append(f"unknown asset {asset_id}")
            continue
        semantic = ref.get("role")
        if semantic not in counts:
            issues.append(f"{asset_id}: role должна быть identity/environment/style")
            continue
        expected = _reference_semantic(asset)
        if expected != semantic:
            issues.append(
                f"{asset_id}: role {semantic} не соответствует роли asset {asset.get('role')}"
            )
            continue
        if asset_id in seen_assets:
            issues.append(f"{asset_id} выбран дважды")
        seen_assets.add(asset_id)
        counts[semantic] += 1
        if semantic == "identity":
            sha = str(asset.get("sha256") or "")
            if sha in identity_shas and identity_shas[sha] != asset_id:
                issues.append(
                    f"две identity refs имеют одинаковый content SHA256: "
                    f"{identity_shas[sha]} и {asset_id}"
                )
            identity_shas[sha] = asset_id
    if counts["identity"] > 2:
        issues.append("identity references: максимум 2")
    if counts["environment"] > 1:
        issues.append("environment reference: максимум 1")
    if counts["style"] > 1:
        issues.append("style reference: максимум 1")
    return issues


def _choose_refs(
    project: Mapping[str, Any],
    current: list[Mapping[str, Any]],
    *,
    frame_id: str,
    anchor_active: bool,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> list[dict[str, str]]:
    assets = list(project.get("library") or [])
    if not assets:
        print_fn("Библиотека пуста; сначала добавьте изображения.")
        return [dict(item) for item in current if isinstance(item, Mapping)]
    print_fn(f"Референсы {frame_id}: Picture 1 — anchor, дальше выбранные по порядку.")
    for index, asset in enumerate(assets, 1):
        print_fn(f"  {index}. {_asset_display(project, str(asset.get('id')))}")
    default = ", ".join(str(item.get("asset_id")) for item in current if isinstance(item, Mapping))
    raw = _ask(
        "Номера через запятую (0/- = очистить; Enter = оставить; ID:role = advanced)",
        default=default,
        input_fn=input_fn,
        allow_clear=True,
    )
    if raw == "":
        return []
    if raw == default and default:
        return [dict(item) for item in current if isinstance(item, Mapping)]
    candidate: list[dict[str, str]] = []
    for token in [part.strip() for part in re.split(r"[,;]", raw) if part.strip()]:
        if token == "0":
            candidate = []
            continue
        if token.isdigit():
            number = int(token)
            if not 1 <= number <= len(assets):
                raise ProjectError(f"нет asset с номером {number}")
            asset = assets[number - 1]
        elif ":" in token or token.startswith("asset-"):
            parsed = _parse_refs(project, token)
            candidate.extend(parsed)
            continue
        else:
            raise ProjectError(f"неизвестный выбор reference: {token!r}")
        semantic = _reference_semantic(asset)
        if semantic is None:
            raise ProjectError(
                f"{asset.get('id')}: provided_keyframe нельзя использовать как reference; "
                "назначьте его как A1 или исправьте роль"
            )
        candidate.append({"asset_id": str(asset.get("id")), "role": semantic})
    issues = _reference_choice_issues(project, candidate, anchor_active=anchor_active)
    if issues:
        print_fn("Выбор references невалиден:")
        for issue in issues:
            print_fn(f"  [BLOCK] {issue}")
        if not _ask_yes(
            "Сохранить этот невалидный выбор как DRAFT-блокер?",
            default=False,
            input_fn=input_fn,
        ):
            print_fn("Старый выбор references сохранён.")
            return [dict(item) for item in current if isinstance(item, Mapping)]
    return candidate


def _picture_lines(
    project: Mapping[str, Any],
    scene: Mapping[str, Any],
    role: str,
) -> list[str]:
    frame = scene.get("initial" if role == "A" else "final")
    if not isinstance(frame, Mapping):
        return []
    anchor = role == "B" or bool(frame.get("use_previous_anchor"))
    lines: list[str] = []
    index = 1
    if anchor:
        lines.append(
            f"Picture {index}: anchor текущего кадра"
            + (" (A1/previous B)" if role == "B" else " (previous B)")
        )
        index += 1
    for ref in frame.get("references") or []:
        if not isinstance(ref, Mapping):
            continue
        asset_id = str(ref.get("asset_id"))
        lines.append(f"Picture {index}: {_asset_display(project, asset_id)}")
        index += 1
    return lines


def _copy_description_choice(
    scene: Mapping[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
) -> bool:
    if not str(scene.get("description") or "").strip():
        return False
    return _ask_yes(
        "Скопировать точное описание сцены в начало/конец состояния?",
        default=False,
        input_fn=input_fn,
    )


def _blank_scene(index: int, scene_id: str) -> dict[str, Any]:
    first = index == 0
    return {
        "id": scene_id,
        "description": "",
        "action": "",
        "camera": "",
        "subjects_visible": [],
        "subjects_absent": [],
        "shared_state": "",
        "shared_invariants": [],
        "shared_allowed_changes": [],
        "initial": {
            "kind": "first_frame" if first else "new_shot",
            "asset_id": None,
            "instruction": "",
            "instruction_mode": "manual",
            "state_a": "",
            "references": [],
            "invariants": [],
            "allowed_changes": [],
            "use_previous_anchor": False,
            "require_anchor_approval": False,
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
            "require_anchor_approval": False if first else True,
        },
    }


def _scene_by_id(project: Mapping[str, Any], scene_id: str) -> dict[str, Any]:
    for scene in project.get("scenes") or []:
        if scene.get("id") == scene_id:
            return scene
    raise ProjectError(f"scene не найдена: {scene_id}")


def _latest_provided(project: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = [item for item in project.get("library") or [] if item.get("role") == "provided_keyframe"]
    return candidates[-1] if candidates else None


def _choose_provided_asset(
    project: dict[str, Any],
    *,
    prompt: str,
    current: str | None,
    input_fn: Callable[[str], str] | None = None,
) -> str | None:
    candidates = [item for item in project.get("library") or [] if item.get("role") == "provided_keyframe"]
    if not candidates:
        print("Нет provided_keyframe assets; сначала добавьте изображение role=provided_keyframe.")
        return current
    print("Готовые ключевые кадры (role=provided_keyframe):")
    for index, item in enumerate(candidates, 1):
        print(
            f"  {index}. {item.get('id')}  {item.get('original_name')} "
            f"{item.get('width')}x{item.get('height')}"
        )
    default = current if current in {item.get("id") for item in candidates} else str(candidates[-1].get("id"))
    value = _ask(
        prompt + " (номер или ID; '-' = нет)",
        default=default,
        input_fn=input_fn,
        allow_clear=True,
    )
    if value == "":
        return None
    if value.isdigit() and 1 <= int(value) <= len(candidates):
        return str(candidates[int(value) - 1]["id"])
    asset = _asset_by_id(project, value)
    if asset is None:
        raise ProjectError(f"asset не найден: {value}")
    if asset.get("role") != "provided_keyframe":
        raise ProjectError(
            f"asset {value} имеет role={asset.get('role')}; A1/provided требует provided_keyframe"
        )
    return value


def _contextual_approval(project: Mapping[str, Any], scene: Mapping[str, Any], role: str) -> bool:
    if role == "A":
        if not scene.get("initial", {}).get("use_previous_anchor"):
            return False
        index = next((i for i, item in enumerate(project.get("scenes") or []) if item.get("id") == scene.get("id")), 0)
        if index <= 0:
            return False
        previous = (project.get("scenes") or [])[index - 1]
        return bool(previous.get("final", {}).get("kind") == "continue")
    index = next((i for i, item in enumerate(project.get("scenes") or []) if item.get("id") == scene.get("id")), 0)
    return bool(index > 0 and scene.get("initial", {}).get("kind") == "new_shot")


def _ask_approval(
    project: Mapping[str, Any],
    scene: dict[str, Any],
    role: str,
    *,
    current: bool,
    input_fn: Callable[[str], str] | None = None,
) -> bool:
    possible = _contextual_approval(project, scene, role)
    default = bool(current) if current and possible else possible
    value = _ask_yes("Требовать owner approval anchor?", default=default, input_fn=input_fn)
    if value and not possible:
        print(
            "Approval недоступен для first_frame/provided anchor; оставлено false. "
            "Используйте generated anchor (например, A2/B2)."
        )
        return False
    return value


def _save_field(store: ProjectStore, project: dict[str, Any], print_fn: Callable[[str], None]) -> None:
    try:
        _save(store, project)
    except ProjectError as exc:
        print_fn(f"Не сохранено: {exc}")
        raise


def _edit_initial(
    store: ProjectStore,
    project: dict[str, Any],
    scene: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    initial = scene.setdefault("initial", _blank_scene(0, str(scene["id"]))["initial"])
    first = project.get("scenes") and (project["scenes"][0].get("id") == scene.get("id"))
    if first:
        initial["kind"] = "first_frame"
        print("A1: kind=first_frame (фиксировано для первой сцены).")
    else:
        kind = _ask(
            "A kind (new_shot или provided_keyframe; continue не поддержан)",
            default=str(initial.get("kind") or "new_shot"),
            input_fn=input_fn,
        )
        if kind == "continue":
            print("A continue не поддержан; используйте B той же сцены или новую сцену.")
        elif kind in {"new_shot", "provided_keyframe"}:
            initial["kind"] = kind
        else:
            print("Неизвестный A kind; оставлено прежнее.")
    _save_field(store, project, print_fn)
    if initial.get("kind") in {"first_frame", "provided_keyframe"}:
        current = initial.get("asset_id")
        try:
            selected = _choose_provided_asset(
                project,
                prompt="A asset",
                current=str(current) if current else None,
                input_fn=input_fn,
            )
        except ProjectError as exc:
            print_fn(f"Ошибка: {exc}")
            selected = current
        initial["asset_id"] = selected
        _save_field(store, project, print_fn)
    if initial.get("kind") == "new_shot":
        initial["instruction"] = _ask_multiline(
            "A exact instruction",
            str(initial.get("instruction") or ""),
            input_fn=input_fn,
            on_change=lambda value: (initial.__setitem__("instruction", value), _save_field(store, project, print_fn)),
        )
        initial["instruction_mode"] = "manual"
        initial["invariants"] = _ask_list(
            "A invariants (через запятую; - очистить)",
            list(initial.get("invariants") or []),
            input_fn=input_fn,
        )
        _save_field(store, project, print_fn)
        initial["allowed_changes"] = _ask_list(
            "A allowed changes (через запятую; - очистить)",
            list(initial.get("allowed_changes") or []),
            input_fn=input_fn,
        )
        _save_field(store, project, print_fn)
    if not first:
        initial["use_previous_anchor"] = _ask_yes(
            "A использовать previous-B anchor?",
            default=bool(initial.get("use_previous_anchor")),
            input_fn=input_fn,
        )
        _save_field(store, project, print_fn)
    if initial.get("kind") == "new_shot":
        initial["require_anchor_approval"] = _ask_approval(
            project,
            scene,
            "A",
            current=bool(initial.get("require_anchor_approval")),
            input_fn=input_fn,
        )
        _save_field(store, project, print_fn)
    try:
        initial["references"] = _parse_refs(
            project,
            _ask(
                "A references (asset:role через запятую; - очистить)",
                default=_refs_text(project, list(initial.get("references") or [])),
                input_fn=input_fn,
                allow_clear=True,
            ),
        )
        _save_field(store, project, print_fn)
    except ProjectError as exc:
        print_fn(f"References не изменены: {exc}")


def _edit_final(
    store: ProjectStore,
    project: dict[str, Any],
    scene: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    final = scene.setdefault("final", _blank_scene(0, str(scene["id"]))["final"])
    kind = _ask(
        "B kind (continue или provided_keyframe; new_shot не поддержан)",
        default=str(final.get("kind") or "continue"),
        input_fn=input_fn,
    )
    if kind == "new_shot":
        print("B new camera не поддержан; создайте новую сцену. Оставлено continue.")
    elif kind in {"continue", "provided_keyframe"}:
        final["kind"] = kind
    _save_field(store, project, print_fn)
    if final.get("kind") == "provided_keyframe":
        try:
            selected = _choose_provided_asset(
                project,
                prompt="B provided asset",
                current=str(final.get("asset_id") or "") or None,
                input_fn=input_fn,
            )
        except ProjectError as exc:
            print_fn(f"Ошибка: {exc}")
            selected = final.get("asset_id")
        final["asset_id"] = selected
        _save_field(store, project, print_fn)
    else:
        final["instruction"] = _ask_multiline(
            "B exact instruction",
            str(final.get("instruction") or ""),
            input_fn=input_fn,
            on_change=lambda value: (final.__setitem__("instruction", value), _save_field(store, project, print_fn)),
        )
        final["instruction_mode"] = "manual"
        final["invariants"] = _ask_list(
            "B invariants (камера добавляется v2 автоматически; - очистить)",
            list(final.get("invariants") or []),
            input_fn=input_fn,
        )
        _save_field(store, project, print_fn)
        final["allowed_changes"] = _ask_list(
            "B allowed changes (camera/lens запрещены; - очистить)",
            list(final.get("allowed_changes") or []),
            input_fn=input_fn,
        )
        _save_field(store, project, print_fn)
        final["require_anchor_approval"] = _ask_approval(
            project,
            scene,
            "B",
            current=bool(final.get("require_anchor_approval")),
            input_fn=input_fn,
        )
        _save_field(store, project, print_fn)
    try:
        final["references"] = _parse_refs(
            project,
            _ask(
                "B references без anchor (asset:role; - очистить)",
                default=_refs_text(project, list(final.get("references") or [])),
                input_fn=input_fn,
                allow_clear=True,
            ),
        )
        _save_field(store, project, print_fn)
    except ProjectError as exc:
        print_fn(f"References не изменены: {exc}")


def _choose_subject_list(
    project: Mapping[str, Any],
    current: list[str],
    *,
    prompt: str,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> list[str]:
    subjects = list(project.get("subjects") or [])
    if not subjects:
        return []
    print_fn("Субъекты:")
    for index, subject in enumerate(subjects, 1):
        print_fn(f"  {index}. {subject.get('id')} kind={subject.get('kind')} label={subject.get('label')}")
    raw = _ask(
        prompt + " (номера через запятую; 0/- = нет; Enter = оставить)",
        default=", ".join(str(item) for item in current),
        input_fn=input_fn,
        allow_clear=True,
    )
    if raw in {"", "0"}:
        return []
    result: list[str] = []
    for token in [part.strip() for part in re.split(r"[,;]", raw) if part.strip()]:
        if token.isdigit():
            number = int(token)
            if not 1 <= number <= len(subjects):
                raise ProjectError(f"нет subject с номером {number}")
            subject_id = str(subjects[number - 1].get("id"))
        else:
            subject_id = token
        if not any(str(item.get("id")) == subject_id for item in subjects):
            raise ProjectError(f"unknown subject: {subject_id}")
        if subject_id not in result:
            result.append(subject_id)
    return result


def _choose_a1_asset(
    project: Mapping[str, Any],
    *,
    current: str | None,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> str | None:
    candidates = [
        item for item in project.get("library") or []
        if item.get("role") == "provided_keyframe"
    ]
    if not candidates:
        print_fn(
            "Нет provided_keyframe. Добавьте изображение (меню 1) или исправьте "
            "роль существующего изображения (меню 10 → Роль)."
        )
        return current
    print_fn("Готовые кадры для A1 (role=provided_keyframe):")
    for index, item in enumerate(candidates, 1):
        print_fn(
            f"  {index}. {item.get('id')} \"{item.get('original_name')}\" "
            f"{item.get('width')}x{item.get('height')} "
            f"role={ROLE_LABELS.get(str(item.get('role')), item.get('role'))}"
        )
    default = str(current) if current in {item.get("id") for item in candidates} else "1"
    raw = _ask(
        "A1 asset (номер или ID; Enter = оставить)",
        default=default,
        input_fn=input_fn,
        allow_clear=True,
    )
    if raw == "":
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(candidates):
        return str(candidates[int(raw) - 1]["id"])
    asset = _asset_by_id(project, raw)
    if asset is None or asset.get("role") != "provided_keyframe":
        raise ProjectError("A1 требует существующий provided_keyframe asset")
    return str(asset["id"])


def _assign_a1_source(
    store: ProjectStore,
    project: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    scenes = list(project.get("scenes") or [])
    if not scenes:
        print_fn("Сначала добавьте сцену (меню 2).")
        return
    scene = scenes[0]
    initial = scene.setdefault("initial", _blank_scene(0, str(scene["id"]))["initial"])
    try:
        selected = _choose_a1_asset(
            project,
            current=str(initial.get("asset_id") or "") or None,
            input_fn=input_fn,
            print_fn=print_fn,
        )
    except ProjectError as exc:
        print_fn(f"Ошибка: {exc}")
        return
    if selected is None:
        print_fn("A1 не изменён.")
        return
    initial["kind"] = "first_frame"
    initial["asset_id"] = selected
    _save_field(store, project, print_fn)
    print_fn(f"A1 назначен: {selected} (только scene {scene.get('id')}).")


def _edit_frame_states(
    store: ProjectStore,
    project: dict[str, Any],
    scene: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    print_fn(f"Состояния сцены {scene.get('id')}: {scene.get('description') or '<без описания>'}")
    scene["action"] = _ask_multiline(
        "Действие/изменение",
        str(scene.get("action") or ""),
        input_fn=input_fn,
        on_change=lambda value: (scene.__setitem__("action", value), _save_field(store, project, print_fn)),
    )
    _save_field(store, project, print_fn)
    scene["camera"] = _ask_multiline(
        "Ракурс/камера",
        str(scene.get("camera") or ""),
        input_fn=input_fn,
        on_change=lambda value: (scene.__setitem__("camera", value), _save_field(store, project, print_fn)),
    )
    _save_field(store, project, print_fn)
    scene["shared_state"] = _ask_multiline(
        "Что общее и неизменно",
        str(scene.get("shared_state") or ""),
        input_fn=input_fn,
        on_change=lambda value: (scene.__setitem__("shared_state", value), _save_field(store, project, print_fn)),
    )
    _save_field(store, project, print_fn)
    copy_description = _copy_description_choice(scene, input_fn=input_fn)
    initial = scene.setdefault("initial", _blank_scene(0, str(scene["id"]))["initial"])
    final = scene.setdefault("final", _blank_scene(0, str(scene["id"]))["final"])
    if copy_description:
        text = str(scene.get("description") or "")
        if initial.get("kind") not in {"provided_keyframe"}:
            initial["state_a"] = text
        if final.get("kind") == "continue":
            final["state_b"] = text
    else:
        if initial.get("kind") == "first_frame":
            initial["state_a"] = _ask_multiline(
                "Что видно в начале (A1, необязательная аннотация)",
                str(initial.get("state_a") or ""),
                input_fn=input_fn,
                on_change=lambda value: (initial.__setitem__("state_a", value), _save_field(store, project, print_fn)),
            )
            _save_field(store, project, print_fn)
        elif initial.get("kind") == "new_shot":
            initial["state_a"] = _ask_multiline(
                "Что видно в начале (A)",
                str(initial.get("state_a") or ""),
                input_fn=input_fn,
                on_change=lambda value: (initial.__setitem__("state_a", value), _save_field(store, project, print_fn)),
            )
            _save_field(store, project, print_fn)
        if final.get("kind") == "continue":
            final["state_b"] = _ask_multiline(
                "Что должно быть в конце (B)",
                str(final.get("state_b") or ""),
                input_fn=input_fn,
                on_change=lambda value: (final.__setitem__("state_b", value), _save_field(store, project, print_fn)),
            )
            _save_field(store, project, print_fn)
    labels = {
        str(item.get("id")): str(item.get("label") or item.get("id"))
        for item in project.get("subjects") or []
    }
    for role, frame in (("A", initial), ("B", final)):
        if frame.get("kind") in {"provided_keyframe"} or (
            role == "A" and frame.get("kind") == "first_frame"
        ):
            continue
        target = frame.get("state_a" if role == "A" else "state_b")
        if not isinstance(target, str) or not target.strip():
            frame["instruction_mode"] = "states"
            print_fn(f"{role}: состояние пусто; можно оставить DRAFT и увидеть blocker в preview.")
            continue
        frame["instruction_mode"] = "states"
        composed = _compose_state_instruction(scene, frame, role, labels)
        print_fn(f"Итоговая инструкция {role}:")
        print_fn(composed)
        if _ask_yes(
            f"Изменить итоговую инструкцию {role} вручную?",
            default=False,
            input_fn=input_fn,
        ):
            edited = _ask_multiline(
                f"Итоговая инструкция {role} (ручной режим)",
                composed,
                input_fn=input_fn,
            )
            frame["instruction"] = edited
            frame["instruction_mode"] = "exact"
        _save_field(store, project, print_fn)


def _edit_frame_references(
    store: ProjectStore,
    project: dict[str, Any],
    scene: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    role = _ask("Состояние A или B", default="A", input_fn=input_fn).upper()
    if role not in {"A", "B"}:
        print_fn("Нужно A или B.")
        return
    frame = scene.get("initial" if role == "A" else "final")
    if not isinstance(frame, dict):
        print_fn("Состояние не найдено.")
        return
    if frame.get("kind") in {"first_frame", "provided_keyframe"}:
        print_fn(
            f"{role}: references не требуются для provided/first_frame; "
            "выбор не меняется."
        )
        return
    anchor_active = role == "B" or bool(frame.get("use_previous_anchor"))
    try:
        refs = _choose_refs(
            project,
            list(frame.get("references") or []),
            frame_id=f"{scene.get('id')}/{role}",
            anchor_active=anchor_active,
            input_fn=input_fn,
            print_fn=print_fn,
        )
    except ProjectError as exc:
        print_fn(f"Ошибка: {exc}")
        return
    frame["references"] = refs
    _save_field(store, project, print_fn)
    print_fn(
        f"{role}: references сохранены ({len(refs)}). "
        f"Карта: {', '.join(_picture_lines(project, scene, role)) or '<нет>'}"
    )


def _completion_wizard(
    store: ProjectStore,
    project: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    scenes = list(project.get("scenes") or [])
    if not scenes:
        print_fn("Сцен нет; сначала добавьте сцену (меню 2).")
        return
    first = scenes[0]
    initial_first = first.setdefault("initial", _blank_scene(0, str(first["id"]))["initial"])
    if not initial_first.get("asset_id"):
        print_fn("A1: нужен provided_keyframe.")
        _assign_a1_source(store, project, input_fn=input_fn, print_fn=print_fn)
    for number, scene in enumerate(scenes, 1):
        print_fn(f"\nсцена {number} из {len(scenes)}: {scene.get('description') or '<без описания>'}")
        try:
            if not str(scene.get("action") or "").strip():
                scene["action"] = _ask_multiline(
                    "ACTION (действие/изменение; Enter = оставить пустым)",
                    "",
                    input_fn=input_fn,
                    on_change=lambda value: (scene.__setitem__("action", value), _save_field(store, project, print_fn)),
                )
                _save_field(store, project, print_fn)
            if not str(scene.get("camera") or "").strip():
                scene["camera"] = _ask_multiline(
                    "CAMERA (ракурс; Enter = оставить пустым)",
                    "",
                    input_fn=input_fn,
                    on_change=lambda value: (scene.__setitem__("camera", value), _save_field(store, project, print_fn)),
                )
                _save_field(store, project, print_fn)
            if not scene.get("subjects_visible"):
                scene["subjects_visible"] = _choose_subject_list(
                    project,
                    list(scene.get("subjects_visible") or []),
                    prompt="Видимые subjects",
                    input_fn=input_fn,
                    print_fn=print_fn,
                )
                _save_field(store, project, print_fn)
            if not scene.get("subjects_absent"):
                scene["subjects_absent"] = _choose_subject_list(
                    project,
                    list(scene.get("subjects_absent") or []),
                    prompt="Отсутствующие subjects",
                    input_fn=input_fn,
                    print_fn=print_fn,
                )
                _save_field(store, project, print_fn)
            if not str(scene.get("shared_state") or "").strip():
                scene["shared_state"] = _ask_multiline(
                    "SHARED (что общее и неизменно; Enter = оставить пустым)",
                    "",
                    input_fn=input_fn,
                    on_change=lambda value: (scene.__setitem__("shared_state", value), _save_field(store, project, print_fn)),
                )
                _save_field(store, project, print_fn)
            initial = scene.setdefault("initial", _blank_scene(0, str(scene["id"]))["initial"])
            final = scene.setdefault("final", _blank_scene(0, str(scene["id"]))["final"])
            labels = {
                str(item.get("id")): str(item.get("label") or item.get("id"))
                for item in project.get("subjects") or []
            }
            for role, frame in (("A", initial), ("B", final)):
                if frame.get("kind") in {"provided_keyframe"} or (
                    role == "A" and frame.get("kind") == "first_frame"
                ):
                    continue
                key = "state_a" if role == "A" else "state_b"
                if not str(frame.get(key) or "").strip():
                    frame[key] = _ask_multiline(
                        "Что видно в начале (A)" if role == "A" else "Что должно быть в конце (B)",
                        "",
                        input_fn=input_fn,
                        on_change=lambda value, frame=frame, key=key: (
                            frame.__setitem__(key, value),
                            _save_field(store, project, print_fn),
                        ),
                    )
                    _save_field(store, project, print_fn)
                if str(frame.get(key) or "").strip():
                    frame["instruction_mode"] = "states"
                    composed = _compose_state_instruction(scene, frame, role, labels)
                    print_fn(f"Итоговая инструкция {role}:")
                    print_fn(composed)
                    if _ask_yes(
                        f"Изменить итоговую инструкцию {role} вручную?",
                        default=False,
                        input_fn=input_fn,
                    ):
                        edited = _ask_multiline(
                            f"Итоговая инструкция {role} (ручной режим)",
                            composed,
                            input_fn=input_fn,
                        )
                        frame["instruction"] = edited
                        frame["instruction_mode"] = "exact"
                        _save_field(store, project, print_fn)
                    _save_field(store, project, print_fn)
            for role, frame in (("A", initial), ("B", final)):
                if frame.get("kind") in {"first_frame", "provided_keyframe"}:
                    continue
                anchor_active = role == "B" or bool(frame.get("use_previous_anchor"))
                current_refs = list(frame.get("references") or [])
                if not current_refs or _reference_choice_issues(
                    project, current_refs, anchor_active=anchor_active
                ):
                    frame["references"] = _choose_refs(
                        project,
                        current_refs,
                        frame_id=f"{scene.get('id')}/{role}",
                        anchor_active=anchor_active,
                        input_fn=input_fn,
                        print_fn=print_fn,
                    )
                    _save_field(store, project, print_fn)
                print_fn(
                    f"Карта {role}: "
                    + (", ".join(_picture_lines(project, scene, role)) or "<нет>")
                )
        except (EOFError, KeyboardInterrupt):
            _save_field(store, project, print_fn)
            print_fn("DRAFT сохранён; продолжайте project resume.")
            return
        except ProjectError as exc:
            print_fn(f"Ошибка: {exc}; сцена сохранена как DRAFT.")
            _save_field(store, project, print_fn)
    print_fn(
        "Готово для ручной проверки. Prepare не запускался автоматически; "
        "следующий шаг показан в preview."
    )


def _scene_menu(
    store: ProjectStore,
    project: dict[str, Any],
    scene: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    while True:
        print_fn(
            f"\nСцена {scene.get('id')}: {_short(str(scene.get('description') or '<без описания>'))}"
        )
        print_fn("1 описание   2 действие   3 камера   4 видимые subjects   5 отсутствующие subjects")
        print_fn("6 shared invariants   7 shared allowed   8 A state   9 B state   10 показать   0 назад")
        print_fn("11 references A/B   12 состояния A/B (понятные поля)")
        try:
            choice = input_fn("Поле сцены: ").strip()
        except (EOFError, KeyboardInterrupt):
            print_fn("Сцена сохранена; продолжайте через project resume.")
            return
        try:
            if choice in {"0", "q", ""}:
                return
            if choice == "1":
                scene["description"] = _ask_multiline(
                    "Описание сцены", str(scene.get("description") or ""), input_fn=input_fn,
                    on_change=lambda value: (scene.__setitem__("description", value), _save_field(store, project, print_fn)),
                )
            elif choice == "2":
                scene["action"] = _ask_multiline(
                    "Действие", str(scene.get("action") or ""), input_fn=input_fn,
                    on_change=lambda value: (scene.__setitem__("action", value), _save_field(store, project, print_fn)),
                )
            elif choice == "3":
                scene["camera"] = _ask_multiline(
                    "Камера", str(scene.get("camera") or ""), input_fn=input_fn,
                    on_change=lambda value: (scene.__setitem__("camera", value), _save_field(store, project, print_fn)),
                )
            elif choice == "4":
                scene["subjects_visible"] = _ask_list(
                    "Видимые subject IDs (через запятую; - очистить)",
                    list(scene.get("subjects_visible") or []), input_fn=input_fn,
                )
                _save_field(store, project, print_fn)
            elif choice == "5":
                scene["subjects_absent"] = _ask_list(
                    "Отсутствующие subject IDs (через запятую; - очистить)",
                    list(scene.get("subjects_absent") or []), input_fn=input_fn,
                )
                _save_field(store, project, print_fn)
            elif choice == "6":
                scene["shared_invariants"] = _ask_list(
                    "Shared invariants (через запятую; - очистить)",
                    list(scene.get("shared_invariants") or []), input_fn=input_fn,
                )
                _save_field(store, project, print_fn)
            elif choice == "7":
                scene["shared_allowed_changes"] = _ask_list(
                    "Shared allowed changes (через запятую; - очистить)",
                    list(scene.get("shared_allowed_changes") or []), input_fn=input_fn,
                )
                _save_field(store, project, print_fn)
            elif choice == "8":
                _edit_initial(store, project, scene, input_fn=input_fn, print_fn=print_fn)
            elif choice == "9":
                _edit_final(store, project, scene, input_fn=input_fn, print_fn=print_fn)
            elif choice == "10":
                _print_scene(project, scene)
            elif choice == "11":
                _edit_frame_references(store, project, scene, input_fn=input_fn, print_fn=print_fn)
            elif choice == "12":
                _edit_frame_states(store, project, scene, input_fn=input_fn, print_fn=print_fn)
            else:
                print_fn("Неизвестное поле.")
        except (EOFError, KeyboardInterrupt):
            print_fn("Сцена сохранена; продолжайте через project resume.")
            return
        except ProjectError as exc:
            print_fn(f"Ошибка: {exc}")
            continue


def _add_scene(
    store: ProjectStore,
    project: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any] | None:
    if input_fn is None:
        input_fn = input
    while True:
        try:
            scene_id = _ask(
                "ID сцены",
                default=f"shot-{len(project.get('scenes') or []) + 1:02d}",
                input_fn=input_fn,
            )
        except (EOFError, KeyboardInterrupt):
            print_fn("Новая сцена не создана.")
            return None
        if any(item.get("id") == scene_id for item in project.get("scenes") or []):
            print_fn(f"Сцена {scene_id} уже существует; выберите другой ID.")
            continue
        scene = _blank_scene(len(project.get("scenes") or []), scene_id)
        project.setdefault("scenes", []).append(scene)
        try:
            _save_field(store, project, print_fn)
        except ProjectError:
            project["scenes"].pop()
            return None
        print_fn(f"Сцена {scene_id} создана (DRAFT, можно заполнять по полям).")
        _scene_menu(store, project, scene, input_fn=input_fn, print_fn=print_fn)
        return scene


def _edit_scene(
    store: ProjectStore,
    project: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    scene_id = _ask("ID сцены для редактирования", input_fn=input_fn)
    try:
        scene = _scene_by_id(project, scene_id)
    except ProjectError as exc:
        print_fn(f"Ошибка: {exc}")
        return
    _scene_menu(store, project, scene, input_fn=input_fn, print_fn=print_fn)


def _edit_project_settings(
    store: ProjectStore,
    project: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    while True:
        print_fn(
            f"\nНастройки: title={project.get('title')} orientation={project.get('orientation')} "
            f"limit={project.get('generation_limit')} seed={project.get('seed')}"
        )
        print_fn("1 title   2 orientation (atomic geometry)   3 style   4 generation_limit   5 seed   6 subject_label   0 назад")
        try:
            choice = input_fn("Настройка: ").strip()
        except (EOFError, KeyboardInterrupt):
            print_fn("Настройки сохранены.")
            return
        if choice in {"0", "q", ""}:
            return
        try:
            if choice == "1":
                value = _ask("Название", default=str(project.get("title") or ""), input_fn=input_fn)
                if not value:
                    raise ProjectError("title не должен быть пустым")
                project["title"] = value
            elif choice == "2":
                value = _ask(
                    "Ориентация (portrait/landscape)",
                    default=str(project.get("orientation") or "portrait"),
                    input_fn=input_fn,
                )
                if value not in ORIENTATIONS:
                    raise ProjectError("ориентация должна быть portrait или landscape")
                project["orientation"] = value
                project["candidate_geometry"] = list(ORIENTATIONS[value][0])
                project["video_geometry"] = list(ORIENTATIONS[value][1])
            elif choice == "3":
                project["style"] = _ask_multiline(
                    "Style/общие ограничения", str(project.get("style") or ""), input_fn=input_fn,
                    on_change=lambda value: (project.__setitem__("style", value), _save_field(store, project, print_fn)),
                )
            elif choice == "4":
                raw = _ask("generation_limit (1..99)", default=str(project.get("generation_limit", 3)), input_fn=input_fn)
                value = int(raw)
                if not 1 <= value <= 99:
                    raise ProjectError("generation_limit должен быть 1..99")
                project["generation_limit"] = value
            elif choice == "5":
                raw = _ask("seed", default=str(project.get("seed", 424242)), input_fn=input_fn)
                value = int(raw)
                if not 0 <= value <= 2_147_483_647:
                    raise ProjectError("seed вне диапазона")
                project["seed"] = value
            elif choice == "6":
                project["subject_label"] = _ask(
                    "subject.label", default=str(project.get("subject_label") or ""),
                    input_fn=input_fn, allow_clear=True,
                )
            else:
                print_fn("Неизвестная настройка.")
                continue
            _save_field(store, project, print_fn)
        except (ValueError, ProjectError) as exc:
            print_fn(f"Ошибка: {exc}")
        except (EOFError, KeyboardInterrupt):
            print_fn("Настройки сохранены.")
            return


def _reassign_asset(
    project: dict[str, Any],
    asset: dict[str, Any],
    *,
    subject_id: str | None,
    view: str | None,
) -> None:
    role = str(asset.get("role"))
    if role in {"character", "creature"} and not subject_id:
        raise ProjectError(f"роль {role} требует existing subject ID")
    if role in {"environment", "style", "provided_keyframe"} and subject_id:
        raise ProjectError(f"роль {role} не принимает subject")
    target_subject = None
    if subject_id:
        target_subject = _subject(project, subject_id)
        if target_subject is None:
            raise ProjectError(f"subject не существует: {subject_id}")
        if target_subject.get("kind") != role:
            raise ProjectError(
                f"subject {subject_id} kind={target_subject.get('kind')} не совпадает с role={role}"
            )
    # Remove the asset association from every subject, then add exactly one.
    for subject in project.get("subjects") or []:
        views = subject.get("views")
        if isinstance(views, list):
            subject["views"] = [
                item for item in views
                if not isinstance(item, Mapping) or item.get("asset_id") != asset.get("id")
            ]
    asset["subject_id"] = subject_id
    asset["view"] = view or None
    if target_subject is not None and view:
        target_subject.setdefault("views", []).append(
            {"asset_id": asset["id"], "label": view}
        )


def _edit_asset(
    store: ProjectStore,
    project: dict[str, Any],
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    items = list(project.get("library") or [])
    if not items:
        print_fn("Library пуст.")
        return
    for index, item in enumerate(items, 1):
        print_fn(
            f"  {index}. {item.get('id')} role={ROLE_LABELS.get(str(item.get('role')), item.get('role'))}"
            f" ({item.get('role')}) subject={item.get('subject_id')} view={item.get('view')}"
            f" provenance={'yes' if item.get('provenance') else 'no'}"
        )
    raw = _ask("Asset (номер или ID)", input_fn=input_fn)
    asset = items[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(items) else _asset_by_id(project, raw)
    if asset is None:
        print_fn("Asset не найден.")
        return
    current_role = str(asset.get("role"))
    try:
        role = _ask(
            "Роль (character/creature/object/environment/style/provided_keyframe; Enter=оставить)",
            default=current_role,
            input_fn=input_fn,
        )
        if role not in IMAGE_ROLES:
            raise ProjectError("неизвестная роль изображения")
        role_changed = role != current_role
        if role_changed:
            usage = asset_usage(project, str(asset.get("id")))
            print_fn(
                "Влияние смены роли: "
                f"A1={usage['a1_source']}, frame sources={usage['frame_sources']}, "
                f"references={usage['references']}, subject views={usage['subject_views']}."
            )
            print_fn(
                "Image bytes/provenance не меняются; asset ID сохраняется. "
                "A1/provided и старые references не сбрасываются автоматически."
            )
            if not _ask_yes(
                "Применить смену роли?",
                default=False,
                input_fn=input_fn,
            ):
                print_fn("Роль не изменена.")
                return
        subject_id = None
        if role in {"character", "creature", "object"}:
            current_subject = str(asset.get("subject_id") or "")
            subject = _subject(project, current_subject) if current_subject else None
            if subject is not None and subject.get("kind") != role:
                current_subject = ""
            subject_id = _ask(
                "Existing Subject ID (для object можно -)",
                default=current_subject,
                input_fn=input_fn,
                allow_clear=True,
            ) or None
            if role in {"character", "creature"} and not subject_id:
                raise ProjectError(f"роль {role} требует существующий Subject ID")
        view = _ask(
            "View label (пусто/- = убрать)",
            default=str(asset.get("view") or ""),
            input_fn=input_fn,
            allow_clear=True,
        ) or None
        prov_path = _ask(
            "Provenance JSON (пусто=оставить, - = очистить)",
            input_fn=input_fn,
            allow_clear=False,
        )
        new_provenance = None
        clear_provenance = prov_path == "-"
        if prov_path and not clear_provenance:
            new_provenance = read_provenance_file(prov_path)
            if str(new_provenance.get("content_sha256", "")).lower() != str(asset.get("sha256", "")).lower():
                raise ProjectError("provenance.content_sha256 не совпадает с asset SHA")
        change_asset_role(
            project,
            str(asset.get("id")),
            role=role,
            subject_id=subject_id,
            view=view,
        )
        if clear_provenance:
            asset["provenance"] = {}
        elif new_provenance is not None:
            asset["provenance"] = new_provenance
        _save_field(store, project, print_fn)
        print_fn("Association/role/provenance обновлены; asset ID и image bytes не меняются.")
    except ProjectError as exc:
        print_fn(f"Ошибка: {exc}")

def _menu(
    store: ProjectStore,
    project: dict[str, Any],
    project_dir: Path,
    *,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    if input_fn is None:
        input_fn = input
    while True:
        print_fn(
            f"\n{project.get('title')} [{project.get('id')}] rev {project.get('revision')} "
            f"status {project.get('status')}"
        )
        print_fn("1 добавить изображение   2 добавить сцену   3 редактировать сцену")
        print_fn("4 preview                5 импорт текста    6 сохранить/выйти")
        print_fn("7 library/subjects       8 добавить subject   9 настройки проекта   10 association asset")
        print_fn("11 назначить A1          12 довести проект до готовности")
        try:
            choice = input_fn("Выбор: ").strip()
        except (EOFError, KeyboardInterrupt):
            try:
                _save(store, project)
            except ProjectError as exc:
                print_fn(f"Не сохранено: {exc}")
                return project
            print_fn("Draft сохранён; продолжайте через project resume.")
            return project
        try:
            if choice in {"6", "save", "exit", "q", ""}:
                _save(store, project)
                print_fn("Сохранено.")
                return project
            if choice == "1":
                role = _ask(
                    "Роль (character/creature/object/environment/style/provided_keyframe)",
                    input_fn=input_fn,
                )
                if role not in IMAGE_ROLES:
                    raise ProjectError("неизвестная role")
                path = _ask("Путь к изображению (кавычки/пробелы/кириллица ok)", input_fn=input_fn)
                subject_id = None
                subject_label = None
                if role in {"character", "creature", "object"}:
                    subject_id = _ask("Subject ID (для object можно -)", input_fn=input_fn, allow_clear=True) or None
                    if subject_id:
                        subject_label = _ask("Subject label", default=subject_id, input_fn=input_fn)
                view = _ask("View label (front/side/...; - = нет)", input_fn=input_fn, allow_clear=True) or None
                prov_path = _ask("Provenance JSON (пусто/- = нет)", input_fn=input_fn, allow_clear=True)
                provenance = read_provenance_file(prov_path) if prov_path else None
                metadata = _add_image_to_project(
                    project, project_dir, role=role, path=path, subject_id=subject_id,
                    subject_label=subject_label, view=view, provenance=provenance,
                )
                _save(store, project)
                print_fn(f"Asset: {metadata.get('id')} ({ROLE_LABELS.get(role, role)})")
                continue
            if choice == "2":
                _add_scene(store, project, input_fn=input_fn, print_fn=print_fn)
                continue
            if choice == "3":
                _edit_scene(store, project, input_fn=input_fn, print_fn=print_fn)
                continue
            if choice == "4":
                payload = _summary(project, project_dir)
                _print_summary(payload)
                continue
            if choice == "5":
                path = _ask("Путь к .txt (UTF-8/BOM; multiline до ---)", input_fn=input_fn)
                numbered = _ask_yes("Numbered parser (1./2.)?", default=False, input_fn=input_fn)
                confirm = _ask_yes("Применить proposal в draft?", default=False, input_fn=input_fn)
                project, proposal = _import_text_to_project(
                    project, project_dir, path, numbered=numbered, confirm=confirm
                )
                _save(store, project)
                print_fn(f"Proposal: {proposal.get('mode')}, scenes={len(proposal.get('scenes') or [])}")
                for warning in proposal.get("warnings") or []:
                    print_fn(f"  [!] {warning}")
                continue
            if choice == "7":
                _print_library(project)
                continue
            if choice == "8":
                subject_id = _ask("Subject ID (lower-case safe id)", input_fn=input_fn)
                kind = _ask("Kind (character/creature/object/environment)", default="character", input_fn=input_fn)
                label = _ask("Label", default=subject_id, input_fn=input_fn)
                _ensure_subject(project, subject_id, kind=kind, label=label)
                _save(store, project)
                continue
            if choice == "9":
                _edit_project_settings(store, project, input_fn=input_fn, print_fn=print_fn)
                continue
            if choice == "10":
                _edit_asset(store, project, input_fn=input_fn, print_fn=print_fn)
                continue
            if choice == "11":
                _assign_a1_source(store, project, input_fn=input_fn, print_fn=print_fn)
                continue
            if choice == "12":
                _completion_wizard(store, project, input_fn=input_fn, print_fn=print_fn)
                continue
            print_fn("Неизвестный выбор.")
        except (EOFError, KeyboardInterrupt):
            try:
                _save(store, project)
            except ProjectError as exc:
                print_fn(f"Не сохранено: {exc}")
                return project
            print_fn("Draft сохранён; продолжайте через project resume.")
            return project
        except ProjectError as exc:
            print_fn(f"Ошибка: {exc}")
            continue
    return project


def _import_text_to_project(
    project: dict[str, Any],
    project_dir: Path,
    path: str,
    *,
    numbered: bool,
    confirm: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    script, text = import_script(
        project_dir, path, library=project.get("library") or [], scripts=project.get("scripts") or []
    )
    script["imported_utc"] = now_utc()
    if not any(item.get("id") == script["id"] for item in project.get("scripts") or []):
        project.setdefault("scripts", []).append(script)
    stored = project_dir / script["stored_path"]
    data = read_bytes_limited(stored, max_bytes=4 * 1024 * 1024, label="script")
    proposal = parse_text_document(data, numbered=numbered)
    proposals = ensure_managed_dir(project_dir, project_dir / "proposals", label="proposals directory")
    proposal_path = proposals / f"{script['sha256']}.json"
    atomic_write_json(proposal_path, proposal)
    project["pending_proposal"] = {
        "path": proposal_path.relative_to(project_dir).as_posix(),
        "sha256": script["sha256"],
        "mode": proposal.get("mode"),
        "needs_confirmation": True,
    }
    if confirm:
        project = apply_proposal(project, proposal)
        project.pop("pending_proposal", None)
        project["import"] = {
            "kind": "text",
            "mode": proposal.get("mode"),
            "source_sha256": proposal.get("source_sha256"),
            "proposal_schema": proposal.get("schema"),
            "needs_confirmation": False,
        }
    return project, proposal


def _fix_text(code: str, blocker: Mapping[str, Any]) -> str:
    if code == "FIRST_FRAME":
        return "выберите loaded provided_keyframe для A1"
    if code == "FIRST_FRAME_ROLE":
        return "A1 asset не имеет role provided_keyframe; исправьте A1 или роль изображения"
    if code in {"MISSING_INSTRUCTION", "MISSING_STATE"}:
        return "заполните Начало (A) / Конец (B) состояния или exact instruction"
    if code == "MISSING_ACTION":
        return "заполните ACTION/изменение для этой сцены"
    if code == "MISSING_CAMERA":
        return "заполните CAMERA/ракурс для этой сцены"
    if code == "EMPTY_REFERENCES":
        return "выберите 1–3 references; с anchor допустимо 2 identity + anchor"
    if code == "REF_LIMIT":
        return "слишком много references; уберите лишние — ничего не будет отброшено автоматически"
    if code == "IDENTITY_REF_LIMIT":
        return "identity references: максимум 2 в одном кадре"
    if code == "DUPLICATE_ROLE":
        return "environment/style: максимум 1 в одном кадре"
    if code == "DUPLICATE_ASSET_REF":
        return "один asset выбран дважды; оставьте один reference"
    if code == "DUPLICATE_REF_CONTENT":
        return "две identity refs указывают одинаковый файл; выберите разные изображения"
    if code == "REFERENCE_CATALOG_LIMIT":
        return "каталог references превышает 64 used refs; уменьшите число выбранных"
    if code == "GENERATION_LIMIT":
        return "увеличьте generation_limit вручную до числа generated frames; автоматического повышения лимита/расходов нет"
    if code == "INVALID_INSTRUCTION_MODE":
        return "исправьте instruction_mode: manual, states или exact"
    if code == "B_CAMERA_CHANGE_UNSUPPORTED":
        return "B не меняет camera/lens; для новой камеры создайте новую сцену"
    if code == "ANCHOR_APPROVAL_UNSUPPORTED":
        return "approval недоступен для provided/first-frame anchor; оставьте false"
    if code in {"SUBJECT_VIEW", "SUBJECT_VIEW_LABEL"}:
        return "исправьте subject/view association"
    if code in {"PROVIDED_PROVENANCE_REQUIRED", "PROVIDED_PROVENANCE_HASH", "PROVIDED_PROVENANCE_FRAME", "PROVIDED_PROVENANCE_SHA", "PROVIDED_PROVENANCE_VERSION"}:
        return "исправьте provenance provided-кадра; произвольный файл не выдаётся за KF1 provenance"
    if code == "UNKNOWN_SUBJECT":
        return "выберите subject из существующей библиотеки subjects"
    if code == "SUBJECT_STATE_CONFLICT":
        return "subject не может быть одновременно visible и absent"
    if code in {"TOKEN_OVERLAP", "UNKNOWN_TOKEN", "DUPLICATE_TOKEN"}:
        return "исправьте invariants/allowed_changes в расширенном редакторе"
    return f"исправьте {code}: {blocker.get('message') or ''}".strip()


def _fix_menu_path(code: str, scene: str | None, frame: str | None) -> str:
    scene_text = scene or "<scene>"
    if code == "FIRST_FRAME":
        return "меню 11 → выбрать provided-кадр"
    if code == "FIRST_FRAME_ROLE":
        return f"меню 11 → выбрать A1; или меню 10 → Роль изображения (scene {scene_text})"
    if code in {"MISSING_INSTRUCTION", "MISSING_STATE", "INVALID_INSTRUCTION_MODE"}:
        return f"меню 3 → сцена {scene_text} → 12"
    if code in {"MISSING_ACTION", "MISSING_CAMERA"}:
        return f"меню 3 → сцена {scene_text} → 12"
    if code in {
        "EMPTY_REFERENCES",
        "REF_LIMIT",
        "IDENTITY_REF_LIMIT",
        "DUPLICATE_ROLE",
        "DUPLICATE_ASSET_REF",
        "DUPLICATE_REF_CONTENT",
        "REFERENCE_CATALOG_LIMIT",
    }:
        return f"меню 3 → сцена {scene_text} → 11"
    if code == "GENERATION_LIMIT":
        return "меню 9 → 4 (лимит/расходы — explicit owner choice)"
    if code == "B_CAMERA_CHANGE_UNSUPPORTED":
        return f"меню 3 → сцена {scene_text} → 12/9; новая камера — новая сцена"
    if code in {"SUBJECT_VIEW", "SUBJECT_VIEW_LABEL"}:
        return "меню 10 → association/Роль"
    if code in {"PROVIDED_PROVENANCE_REQUIRED", "PROVIDED_PROVENANCE_HASH", "PROVIDED_PROVENANCE_FRAME", "PROVIDED_PROVENANCE_SHA", "PROVIDED_PROVENANCE_VERSION"}:
        return (
            f"меню 10 → asset → «Provenance JSON» (attach/correct metadata); "
            f"либо меню 3 → сцена {scene_text} → 8/9 чтобы сменить готовый кадр"
        )
    return f"меню 3 → сцена {scene_text}"


def _next_fixes(
    project: Mapping[str, Any],
    blockers: list[Mapping[str, Any]],
    *,
    gaps: list[Mapping[str, Any]] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    scene_order = {
        str(scene.get("id")): index
        for index, scene in enumerate(project.get("scenes") or [])
        if isinstance(scene, Mapping)
    }
    role_order = {"A": 0, "B": 1, "shared": 2}
    items = [(0, item) for item in blockers] + [(1, item) for item in gaps or []]

    def key(pair: tuple[int, Mapping[str, Any]]) -> tuple[int, int, int, int, str]:
        is_gap, item = pair
        code = str(item.get("code"))
        scene = str(item.get("scene") or "")
        frame = str(item.get("frame") or "")
        role = frame.rsplit("/", 1)[-1] if "/" in frame else "shared"
        return (
            is_gap,
            0 if code in {"FIRST_FRAME", "FIRST_FRAME_ROLE"} else 1,
            scene_order.get(scene, 999),
            role_order.get(role, 3),
            code,
        )

    result: list[dict[str, Any]] = []
    for _is_gap, blocker in sorted(items, key=key):
        code = str(blocker.get("code"))
        scene = str(blocker.get("scene") or "") or None
        frame = str(blocker.get("frame") or "") or None
        target = frame or scene or "project"
        menu_path = _fix_menu_path(code, scene, frame)
        reason = _fix_text(code, blocker)
        item = {
            "code": code,
            "scene": scene,
            "frame": frame,
            "target": target,
            "menu_path": menu_path,
            "text": f"{target} [{code}]: {reason} → {menu_path}",
            "detail": str(blocker.get("message") or ""),
        }
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _completion_gaps(project: Mapping[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for scene in project.get("scenes") or []:
        if not isinstance(scene, Mapping):
            continue
        scene_id = str(scene.get("id") or "")
        if not str(scene.get("action") or "").strip():
            gaps.append(
                {
                    "code": "MISSING_ACTION",
                    "scene": scene_id,
                    "message": "ACTION не заполнен; можно оставить DRAFT или заполнить в guided flow",
                }
            )
        if not str(scene.get("camera") or "").strip():
            gaps.append(
                {
                    "code": "MISSING_CAMERA",
                    "scene": scene_id,
                    "message": "CAMERA/ракурс не заполнен; можно оставить DRAFT или заполнить в guided flow",
                }
            )
    return gaps


def _summary(project: Mapping[str, Any], project_dir: Path) -> dict[str, Any]:
    compiled = validate_and_compile(project, project_dir)
    prepared = project.get("prepared")
    stale = bool(
        isinstance(prepared, Mapping)
        and prepared.get("project_sha256") != authoring_digest(project)
    )
    if compiled["blockers"]:
        status = "INCOMPLETE"
    elif prepared and not stale:
        status = "PREPARED"
    elif stale:
        status = "STALE_PREPARE"
    else:
        status = "READY_TO_PREPARE"
    blocker_list = list(compiled["blockers"] or [])
    gap_list = _completion_gaps(project)
    fixes = _next_fixes(project, blocker_list, gaps=gap_list)
    can_prepare = status in {"READY_TO_PREPARE", "STALE_PREPARE"}
    if status == "INCOMPLETE":
        next_command = f'.\\avs.ps1 project resume "{project.get("id")}"'
    elif can_prepare:
        next_command = f'.\\avs.ps1 project prepare "{project.get("id")}"'
    else:
        next_command = (
            f'.\\avs.ps1 keyframes generate --plan "{project_dir / "plan.json"}" --json'
        )
    return {
        "schema": "anime-studio/kf2-project-preview/v1",
        "id": project.get("id"),
        "title": project.get("title"),
        "orientation": project.get("orientation"),
        "candidate_geometry": project.get("candidate_geometry"),
        "video_geometry": project.get("video_geometry"),
        "revision": project.get("revision"),
        "content_sha256": authoring_digest(project),
        "status": status,
        "can_prepare": can_prepare,
        "blockers": compiled["blockers"],
        "warnings": compiled["warnings"],
        "frames": compiled.get("frames", []),
        "generated_count": compiled.get("generated_count", 0),
        "reference_catalog_count": compiled.get("reference_catalog_count", 0),
        "reference_catalog_limit": compiled.get("reference_catalog_limit", 64),
        "approval_required_frames": compiled.get("approval_required_frames", []),
        "prepared": prepared,
        "plan_sha256": prepared.get("plan_sha256") if isinstance(prepared, Mapping) else None,
        "plan_path": str(project_dir / "plan.json"),
        "library": project.get("library") or [],
        "subjects": project.get("subjects") or [],
        "scenes": project.get("scenes") or [],
        "library_count": len(project.get("library") or []),
        "subject_count": len(project.get("subjects") or []),
        "scene_count": len(project.get("scenes") or []),
        "next_fix": fixes[0] if fixes else None,
        "next_fixes": fixes,
        "remaining_fix_count": max(0, len(blocker_list) + len(gap_list) - len(fixes)),
        "next_command": next_command,
    }


def _print_library(project: Mapping[str, Any]) -> None:
    print("Библиотека:")
    for item in project.get("library") or []:
        role = ROLE_LABELS.get(str(item.get("role")), str(item.get("role")))
        print(
            f"  {item.get('id')} роль={role} ({item.get('role')}) "
            f"subject={item.get('subject_id')} view={item.get('view')} "
            f"{item.get('original_name')} {item.get('width')}x{item.get('height')} "
            f"provenance={'yes' if item.get('provenance') else 'no'}"
        )
    print("Субъекты:")
    for subject in project.get("subjects") or []:
        print(
            f"  {subject.get('id')} kind={subject.get('kind')} label={subject.get('label')} "
            f"views={[v.get('asset_id') for v in subject.get('views') or []]}"
        )


def _print_scene(project: Mapping[str, Any], scene: Mapping[str, Any]) -> None:
    labels = {str(item.get("id")): str(item.get("label") or item.get("id")) for item in project.get("subjects") or []}
    print(f"Сцена {scene.get('id')}:")
    print(f"  описание: {scene.get('description') or '<нет>'}")
    print(f"  действие: {scene.get('action') or '<нет>'}")
    print(f"  камера: {scene.get('camera') or '<нет>'}")
    print(f"  видимые: {', '.join(f'{labels.get(str(x), x)} ({x})' for x in scene.get('subjects_visible') or []) or '<нет>'}")
    print(f"  отсутствующие: {', '.join(f'{labels.get(str(x), x)} ({x})' for x in scene.get('subjects_absent') or []) or '<нет>'}")
    print(f"  shared invariants: {scene.get('shared_invariants') or []}")
    print(f"  shared allowed: {scene.get('shared_allowed_changes') or []}")
    print(f"  shared state: {scene.get('shared_state') or '<нет>'}")
    for role in ("initial", "final"):
        frame = scene.get(role) or {}
        state_key = "state_a" if role == "initial" else "state_b"
        print(
            f"  {role}: kind={frame.get('kind')} ({KIND_LABELS.get(str(frame.get('kind')), frame.get('kind'))}) "
            f"anchor={frame.get('use_previous_anchor') if role == 'initial' else 'current A'} "
            f"approval={frame.get('require_anchor_approval')} refs={_refs_text(project, list(frame.get('references') or []))}"
        )
        print(f"    mode={frame.get('instruction_mode') or 'manual'} {state_key}: {frame.get(state_key) or '<нет>'}")
        print(f"    instruction: {frame.get('instruction') or '<нет>'}")


def _print_summary(payload: Mapping[str, Any]) -> None:
    print(f"Проект: {payload.get('title')} ({payload.get('id')})")
    print(
        f"Формат: {payload.get('orientation')} "
        f"candidate={payload.get('candidate_geometry')} video={payload.get('video_geometry')}"
    )
    print(f"Статус: {payload.get('status')} revision={payload.get('revision')}")
    print("Библиотека:")
    for item in payload.get("library") or []:
        role = ROLE_LABELS.get(str(item.get("role")), str(item.get("role")))
        print(
            f"  {item.get('id')} роль={role} ({item.get('role')}) subject={item.get('subject_id')} "
            f"view={item.get('view')} {item.get('original_name')} {item.get('width')}x{item.get('height')} "
            f"provenance={'yes' if item.get('provenance') else 'no'}"
        )
    print("Субъекты:")
    for subject in payload.get("subjects") or []:
        print(f"  {subject.get('id')} kind={subject.get('kind')} label={subject.get('label')}")
    frames = {str(frame.get("frame_id")): frame for frame in payload.get("frames") or []}
    print("Сцены:")
    for scene in payload.get("scenes") or []:
        _print_scene(payload, scene)
        for role in ("A", "B"):
            frame = frames.get(f"{scene.get('id')}/{role}")
            if frame and frame.get("generate"):
                print(f"    итоговая инструкция {role}: {frame.get('instruction')}")
                print(
                    f"      reference_roles={frame.get('reference_roles')} "
                    f"количество={len(frame.get('reference_roles') or [])} "
                    f"approval={frame.get('approval_required')}"
                )
                for picture in _picture_lines(payload, scene, role):
                    print(f"      {picture}")
    print(
        f"Reference catalog: {payload.get('reference_catalog_count', 0)}/"
        f"{payload.get('reference_catalog_limit', 64)} used refs."
    )
    for warning in payload.get("warnings") or []:
        print(f"  [!] {warning}")
    for blocker in payload.get("blockers") or []:
        print(f"  [BLOCK] {blocker.get('code')}: {blocker.get('message')}")
    if payload.get("prepared"):
        print(f"Подготовленный plan SHA256: {payload.get('plan_sha256')}")
    if payload.get("status") == "INCOMPLETE":
        fixes = list(payload.get("next_fixes") or [])
        if fixes:
            first = fixes[0]
            print(f"Рекомендуемое следующее действие: {first.get('text')}")
            for item in fixes[1:]:
                print(f"  - {item.get('text')}")
            if payload.get("remaining_fix_count"):
                print(f"  - ещё {payload.get('remaining_fix_count')} шагов смотрите в project preview --json")
        print("Prepare не предлагается, пока есть blockers. Исправьте draft через project resume.")
    else:
        if payload.get("next_fix"):
            print(f"Рекомендуемое улучшение: {payload['next_fix'].get('text')}")
        print(f"Следующее: {payload.get('next_command')}")


def _print_prepare(payload: Mapping[str, Any], *, file: Any = None) -> None:
    if file is None:
        file = sys.stdout
    if payload.get("status") == "BLOCKED":
        print("Подготовка заблокирована:", file=file)
        project_preview = payload.get("project") or {}
        fixes = list(project_preview.get("next_fixes") or [])
        if fixes:
            print(f"Рекомендуемое следующее действие: {fixes[0].get('text')}", file=file)
            for item in fixes[1:]:
                print(f"  - {item.get('text')}", file=file)
        for blocker in payload.get("blockers") or []:
            print(f"  [BLOCK] {blocker.get('code')}: {blocker.get('message')}", file=file)
        print("Исправьте draft: project resume/preview; plan не публиковался.", file=file)
        return
    print(f"Подготовка: {payload.get('status')}")
    print(f"План: {payload.get('plan_path')}")
    print(f"Plan SHA256: {payload.get('plan_sha256')}")
    dryrun = payload.get("dryrun") or {}
    print(f"Dry-run: {dryrun.get('status')}")
    print(f"Сгенерированных кадров: {len(dryrun.get('post_plan') or [])}; post_plan={dryrun.get('post_plan')}")
    plan = payload.get("plan_path")
    print(f"Проверка: .\\avs.ps1 keyframes status --plan \"{plan}\"")
    print(f"Dry-run повтор: .\\avs.ps1 keyframes generate --plan \"{plan}\" --json")
    print("Execute/upload/POST не выполнялись; отдельный запуск — только после owner approval.")


def _cmd_create(args: argparse.Namespace) -> tuple[int, Any]:
    store = _store(args)
    project_id = args.id
    title = args.title
    orientation = args.orientation
    interactive = not args.no_interactive
    if interactive:
        try:
            if not project_id:
                project_id = _ask("Project ID (lower-case; пусто = auto)", default=_auto_id())
            if not title:
                title = _ask("Название проекта", default="Новый проект")
            if not orientation:
                orientation = _ask("Ориентация (portrait/landscape)", default="portrait")
        except (EOFError, KeyboardInterrupt):
            project_id = project_id or _auto_id()
            title = title or "Новый проект"
            orientation = orientation or "portrait"
    else:
        project_id = project_id or _auto_id()
        title = title or "Новый проект"
        orientation = orientation or "portrait"
    if orientation not in ORIENTATIONS:
        raise ProjectError("orientation должен быть portrait или landscape")
    project, project_dir = store.create(project_id, title=title, orientation=orientation)
    if args.from_text:
        project, proposal = _import_text_to_project(
            project, project_dir, args.from_text,
            numbered=bool(args.from_text_numbered), confirm=True,
        )
        _save(store, project)
        if args.json:
            payload = _summary(project, project_dir)
            payload["proposal"] = proposal
            return 0, payload
    if interactive and not args.json:
        project = _menu(store, project, project_dir)
    payload = _summary(project, project_dir)
    return 0, payload


def _cmd_resume(args: argparse.Namespace) -> tuple[int, Any]:
    store = _store(args)
    project = store.load(args.id)
    project_dir = store.directory(args.id)
    if args.json:
        return 0, _summary(project, project_dir)
    _menu(store, project, project_dir)
    return 0, None


def _cmd_preview(args: argparse.Namespace) -> tuple[int, Any]:
    store = _store(args)
    project = store.load(args.id)
    return 0, _summary(project, store.directory(args.id))


def _cmd_list(args: argparse.Namespace) -> tuple[int, Any]:
    return 0, {"schema": "anime-studio/kf2-project-list/v1", "projects": _store(args).list()}


def _cmd_add_image(args: argparse.Namespace) -> tuple[int, Any]:
    store = _store(args)
    project = store.load(args.id)
    project_dir = store.directory(args.id)
    provenance = read_provenance_file(args.provenance) if args.provenance else None
    metadata = _add_image_to_project(
        project, project_dir, role=args.role, path=args.path,
        subject_id=args.subject, subject_label=args.subject_label,
        view=args.view, provenance=provenance,
    )
    _save(store, project)
    return 0, {"schema": "anime-studio/kf2-asset-import/v1", "asset": metadata, "project": _summary(project, project_dir)}


def _cmd_import_text(args: argparse.Namespace) -> tuple[int, Any]:
    store = _store(args)
    project = store.load(args.id)
    project_dir = store.directory(args.id)
    project, proposal = _import_text_to_project(
        project, project_dir, args.file, numbered=bool(args.numbered), confirm=bool(args.confirm)
    )
    _save(store, project)
    return 0, {
        "schema": "anime-studio/kf2-text-import/v1",
        "proposal": proposal,
        "project": _summary(project, project_dir),
    }


def _archive_plan(project_dir: Path, old: bytes) -> None:
    history = ensure_managed_dir(project_dir, project_dir / "history" / "plans", label="plan history")
    target = history / f"plan-{sha256_bytes(old)}.json"
    if checked_exists(target, label="plan history"):
        if read_bytes_limited(target, max_bytes=MAX_PLAN_BYTES, label="plan history") != old:
            raise ProjectError(f"plan history collision/tamper: {target}")
    else:
        atomic_write_bytes(target, old, exclusive=True)


def _dryrun(plan_path: Path, state_dir: Path) -> dict[str, Any]:
    from scripts.kf1 import cli as kf1_cli

    args = SimpleNamespace(
        state_dir=str(state_dir),
        plan=str(plan_path),
        execute=False,
        until_frame=None,
        json=True,
        raise_limit=False,
        base_url=None,
        connection=None,
        i1_state_dir=str(ROOT / "data" / "i1"),
        profile=str(ROOT / "config" / "i1" / "keyframes-qwen-edit-2511.profile.json"),
        token_env=None,
        deadline=900.0,
    )
    code, payload = kf1_cli._cmd_generate(args)
    if code != 0 or not isinstance(payload, dict):
        raise ProjectError("KF1 dry-run не вернул structured payload")
    return payload


def _cmd_prepare(args: argparse.Namespace) -> tuple[int, Any]:
    store = _store(args)
    lock = store.lock(args.id)
    try:
        store.recover_prepare_journal_locked(args.id)
        project = store._load_locked(args.id)
        project_dir = store.directory(args.id)
        revision = int(project["revision"])
        state_dir = (
            Path(args.kf1_state_dir)
            if args.kf1_state_dir
            else project_dir / ".kf1-dry-run"
        )
        if not state_dir.is_absolute():
            state_dir = ROOT / state_dir
        state_dir = validate_local_destination(state_dir, label="kf1-state-dir")
        compiled = validate_and_compile(project, project_dir)
        if compiled["blockers"]:
            return 2, {
                "schema": "anime-studio/kf2-project-prepare/v1",
                "status": "BLOCKED",
                "blockers": compiled["blockers"],
                "warnings": compiled["warnings"],
                "project": _summary(project, project_dir),
            }
        plan = compiled["plan"]
        assert isinstance(plan, dict)
        plan_data = canonical_json_bytes(plan)
        if len(plan_data) > MAX_PLAN_BYTES:
            raise ProjectError(f"plan превышает {MAX_PLAN_BYTES} bytes")
        plan_path = project_dir / "plan.json"
        candidate = project_dir / f".plan-candidate-{uuid.uuid4().hex}.json"
        try:
            atomic_write_bytes(candidate, plan_data)
            try:
                resolved = resolve_plan(candidate)
            except Exception as exc:
                raise ProjectError(f"compiled plan не проходит resolve_plan: {exc}") from exc
        finally:
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        old_project_bytes = read_bytes_limited(
            project_dir / "project.json", max_bytes=4 * 1024 * 1024, label="project.json"
        )
        old_plan_bytes = None
        if checked_exists(plan_path, label="plan.json"):
            old_plan_bytes = read_bytes_limited(plan_path, max_bytes=MAX_PLAN_BYTES, label="plan.json")
        prepared_old = project.get("prepared")
        old_plan_canonical = None
        if old_plan_bytes is not None:
            if not isinstance(prepared_old, Mapping):
                raise ProjectError("old plan exists без prepared metadata; пересборка небезопасна")
            old_plan_canonical = prepared_old.get("plan_sha256")
            if (
                not isinstance(old_plan_canonical, str)
                or len(old_plan_canonical) != 64
                or any(char not in "0123456789abcdef" for char in old_plan_canonical.lower())
            ):
                raise ProjectError("prepared.plan_sha256 должен быть 64 hex")
            if prepared_old.get("plan_path") != "plan.json":
                raise ProjectError("prepared.plan_path должен быть stable plan.json")
            _validate_old_plan_bytes(
                old_plan_bytes,
                project_dir=project_dir,
                project=project,
                expected_raw_sha=sha256_bytes(old_plan_bytes),
                expected_canonical_sha=old_plan_canonical,
            )
        elif prepared_old is not None:
            raise ProjectError("Есть метаданные подготовки, но отсутствует прежний plan.json")
        store._write_prepare_journal_locked(
            args.id,
            old_project_bytes=old_project_bytes,
            old_plan_bytes=old_plan_bytes,
            old_plan_canonical_sha256=old_plan_canonical,
        )
        try:
            if old_plan_bytes != plan_data:
                if old_plan_bytes is not None:
                    _archive_plan(project_dir, old_plan_bytes)
                atomic_write_bytes(plan_path, plan_data)
            dryrun = _dryrun(plan_path, state_dir)
            prepared = {
                "plan_path": "plan.json",
                "plan_sha256": resolved.plan_sha256,
                "project_sha256": authoring_digest(project),
                "prepared_utc": now_utc(),
                "dryrun_state_dir": str(state_dir),
                "dryrun_status": dryrun.get("status"),
                "post_plan": dryrun.get("post_plan"),
                "approval_required_frames": compiled.get("approval_required_frames", []),
            }
            project["prepared"] = prepared
            project["status"] = "PREPARED"
            saved = store._save_locked(args.id, project, expected_revision=revision)
            project.clear()
            project.update(saved)
            store._clear_prepare_journal_locked(args.id)
        except Exception:
            store.recover_prepare_journal_locked(args.id)
            raise
        return 0, {
            "schema": "anime-studio/kf2-project-prepare/v1",
            "status": "PREPARED",
            "plan_path": str(plan_path),
            "plan_sha256": resolved.plan_sha256,
            "dryrun": dryrun,
            "prepared": prepared,
            "project": _summary(project, project_dir),
        }
    finally:
        lock.release()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avs.ps1 project",
        description="KF2: локальный authoring-проект. Без network/GPU/execute.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="создать draft; по умолчанию wizard")
    _common(create)
    create.add_argument("--id", default=None)
    create.add_argument("--title", default=None)
    create.add_argument("--orientation", choices=sorted(ORIENTATIONS), default=None)
    create.add_argument("--from-text", default=None)
    create.add_argument("--from-text-numbered", action="store_true")
    create.add_argument("--no-interactive", action="store_true")
    for name, help_text in (("resume", "продолжить draft"), ("edit", "редактировать draft")):
        p = sub.add_parser(name, help=help_text)
        _common(p)
        p.add_argument("id")
    preview = sub.add_parser("preview", help="полный preview/blockers")
    _common(preview)
    preview.add_argument("id")
    list_p = sub.add_parser("list", help="список проектов")
    _common(list_p)
    add = sub.add_parser("add-image", help="immutable import изображения")
    _common(add)
    add.add_argument("id")
    add.add_argument("--role", required=True, choices=sorted(IMAGE_ROLES))
    add.add_argument("--path", required=True)
    add.add_argument("--subject", default=None)
    add.add_argument("--subject-label", default=None)
    add.add_argument("--view", default=None)
    add.add_argument("--provenance", default=None)
    text = sub.add_parser("import-text", help="UTF-8/BOM .txt -> proposal")
    _common(text)
    text.add_argument("id")
    text.add_argument("--file", required=True)
    text.add_argument("--numbered", action="store_true")
    text.add_argument("--confirm", action="store_true")
    prep = sub.add_parser("prepare", help="transactional compile + KF1 dry-run")
    _common(prep)
    prep.add_argument("id")
    prep.add_argument("--kf1-state-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        code, payload = {
            "create": _cmd_create,
            "resume": _cmd_resume,
            "edit": _cmd_resume,
            "preview": _cmd_preview,
            "list": _cmd_list,
            "add-image": _cmd_add_image,
            "import-text": _cmd_import_text,
            "prepare": _cmd_prepare,
        }[args.command](args)
        if args.json and payload is not None:
            _print_json(payload)
        elif payload is not None:
            if args.command == "preview":
                _print_summary(payload)
            elif args.command == "prepare":
                _print_prepare(payload, file=sys.stderr if code == 2 else sys.stdout)
            elif args.command in {"create", "resume"}:
                _print_summary(payload)
            elif args.command == "add-image":
                print(f"Asset: {payload.get('asset', {}).get('id')} импортирован.")
            elif args.command == "import-text":
                proposal = payload.get("proposal") or {}
                print(f"Proposal: {proposal.get('mode')}, scenes={len(proposal.get('scenes') or [])}")
            elif args.command == "list":
                for item in payload.get("projects", []):
                    print(f"{item.get('id')}\t{item.get('status')}\t{item.get('title')}")
        return code
    except KF2Error as exc:
        if getattr(args, "json", False):
            _print_json({"status": "ERROR", "error": type(exc).__name__, "message": str(exc)})
        else:
            print(f"Ошибка: {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 2)
    except KeyboardInterrupt:
        print("Прервано. Изменения сохранены по последнему полю; используйте project resume.", file=sys.stderr)
        return 5
    except Exception as exc:
        if getattr(args, "json", False):
            _print_json({"status": "ERROR", "error": type(exc).__name__, "message": str(exc)})
        else:
            print(f"Непредвиденная ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
