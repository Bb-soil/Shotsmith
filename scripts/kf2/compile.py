"""Compile an authored KF2 project into the existing KF1 v2 plan contract.

No KF1 validator, binding, approval, ledger or workflow code is modified here.
The compiler only produces data and refuses silently dropping references.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.kf1.plan import (
    DEFAULT_CFG,
    DEFAULT_GENERATION_LIMIT,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    DEFAULT_SEED,
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    INVARIANT_VOCAB,
    MAX_USED_REFERENCES,
    PLAN_SCHEMA_V2,
    SAFE_ID,
)

from .assets import SUBJECT_KINDS, validate_library_asset
from .errors import CompileError
from .paths import ROOT

PORTRAIT_CANDIDATE = [704, 1280]
PORTRAIT_VIDEO = [448, 816]
LANDSCAPE_CANDIDATE = [1280, 704]
LANDSCAPE_VIDEO = [816, 448]
ORIENTATIONS = {
    "portrait": (PORTRAIT_CANDIDATE, PORTRAIT_VIDEO),
    "landscape": (LANDSCAPE_CANDIDATE, LANDSCAPE_VIDEO),
}
SEMANTIC_ROLES = {"identity", "style", "environment"}
ASSET_TO_SEMANTIC = {
    "character": "identity",
    "creature": "identity",
    "object": "identity",
    "environment": "environment",
    "style": "style",
}
LEGACY_REF_ID = {
    "identity": "character_reference",
    "environment": "background_reference",
    "style": "style_reference",
}
MAX_IDENTITY_REFS_PER_FRAME = 2
MAX_ENVIRONMENT_REFS_PER_FRAME = 1
MAX_STYLE_REFS_PER_FRAME = 1
PROJECT_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _block(
    blockers: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    scene: str | None = None,
    frame: str | None = None,
) -> None:
    item: dict[str, Any] = {"code": code, "message": message}
    if scene is not None:
        item["scene"] = scene
    if frame is not None:
        item["frame"] = frame
    blockers.append(item)


def _warn(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _tokens(value: Any, label: str, blockers: list[dict[str, Any]], *, scene: str, frame: str) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        _block(blockers, "INVALID_TOKENS", f"{label} должен быть списком", scene=scene, frame=frame)
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in INVARIANT_VOCAB:
            _block(blockers, "UNKNOWN_TOKEN", f"{label}: неизвестный токен {item!r}", scene=scene, frame=frame)
            continue
        if item in result:
            _block(blockers, "DUPLICATE_TOKEN", f"{label}: повтор токена {item!r}", scene=scene, frame=frame)
            continue
        result.append(item)
    return result


def _merge_tokens(*lists: list[str]) -> list[str]:
    result: list[str] = []
    for values in lists:
        for value in values:
            if value not in result:
                result.append(value)
    return result


def _subject_label(project: Mapping[str, Any], subjects: list[Mapping[str, Any]]) -> str:
    explicit = project.get("subject_label")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if len(subjects) == 1 and isinstance(subjects[0].get("label"), str):
        return str(subjects[0]["label"])
    return "subject"


def _frame_defaults(kind: str, scene_index: int, *, final: bool) -> dict[str, Any]:
    return {
        "kind": kind,
        "asset_id": None,
        "instruction": "",
        "references": [],
        "invariants": [],
        "allowed_changes": [],
        "use_previous_anchor": False,
        "require_anchor_approval": True if final else False,
    }


def _validate_provenance(
    asset: Mapping[str, Any], blockers: list[dict[str, Any]], *, frame_id: str
) -> dict[str, Any] | None:
    provenance = asset.get("provenance")
    if not isinstance(provenance, Mapping) or not any(
        key in provenance for key in ("frame_id", "plan_sha256", "revision_id", "version")
    ):
        _block(
            blockers,
            "PROVIDED_PROVENANCE_REQUIRED",
            "provided_frame требует existing public provenance; произвольная картинка не выдаётся за KF1 provenance",
            frame=frame_id,
        )
        return None
    result = dict(provenance)
    content = result.get("content_sha256")
    if not isinstance(content, str) or content.lower() != str(asset.get("sha256", "")).lower():
        _block(blockers, "PROVIDED_PROVENANCE_HASH", "provenance.content_sha256 не совпадает с asset SHA", frame=frame_id)
    frame_id_value = result.get("frame_id")
    if not isinstance(frame_id_value, str) or not frame_id_value.strip():
        _block(blockers, "PROVIDED_PROVENANCE_FRAME", "provenance.frame_id обязателен", frame=frame_id)
    for key in ("plan_sha256", "revision_id"):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64:
            _block(blockers, "PROVIDED_PROVENANCE_SHA", f"provenance.{key} должен быть SHA256", frame=frame_id)
    version = result.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        _block(blockers, "PROVIDED_PROVENANCE_VERSION", "provenance.version должен быть >=1", frame=frame_id)
    return result


def _asset_path(project_dir: Path, asset: Mapping[str, Any]) -> str:
    path = Path(str(asset["actual_path"]))
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _new_ref_id(asset_id: str, role: str, catalog: Mapping[str, Any]) -> str:
    base = f"ref-{asset_id}"
    if len(base) > 64:
        base = base[:64]
    candidate = base
    counter = 1
    while candidate in catalog:
        suffix = hashlib.sha256(f"{asset_id}|{role}|{counter}".encode("utf-8")).hexdigest()[:8]
        candidate = (base[:55] + "-" + suffix)[:64]
        counter += 1
    return candidate


def _catalog_ref_id(
    item: Mapping[str, Any],
    *,
    catalog: dict[str, dict[str, Any]],
    asset_ref_ids: dict[tuple[str, str], str],
    used_aliases: set[str],
) -> str:
    asset_id = str(item["asset_id"])
    role = str(item["role"])
    key = (asset_id, role)
    existing = asset_ref_ids.get(key)
    if existing is not None:
        return existing
    alias = LEGACY_REF_ID.get(role)
    if alias is not None and alias not in used_aliases:
        ref_id = alias
        used_aliases.add(alias)
    else:
        ref_id = _new_ref_id(asset_id, role, catalog)
    catalog[ref_id] = {"asset_id": asset_id, "role": role, "asset": item["asset"]}
    asset_ref_ids[key] = ref_id
    return ref_id


def _compose_state_instruction(
    scene: Mapping[str, Any],
    frame: Mapping[str, Any],
    role: str,
    subject_labels: Mapping[str, str],
) -> str:
    target = frame.get("state_a" if role == "A" else "state_b")
    parts = [f"Target {role} state: {str(target).strip()}"]
    shared = scene.get("shared_state")
    if isinstance(shared, str) and shared.strip():
        parts.append("Common and unchanged: " + shared.strip())
    action = scene.get("action")
    if isinstance(action, str) and action.strip():
        parts.append("Action/change: " + action.strip())
    camera = scene.get("camera")
    if isinstance(camera, str) and camera.strip():
        parts.append("Camera: " + camera.strip())
    visible = [subject_labels.get(str(item), str(item)) for item in scene.get("subjects_visible") or []]
    absent = [subject_labels.get(str(item), str(item)) for item in scene.get("subjects_absent") or []]
    if visible:
        parts.append("Visible subjects: " + ", ".join(visible))
    if absent:
        parts.append("Absent subjects: " + ", ".join(absent))
    return "\n".join(part for part in parts if part)


def _instruction_mode(scene: Mapping[str, Any], frame: Mapping[str, Any]) -> str:
    mode = frame.get("instruction_mode")
    if mode is None:
        mode = scene.get("instruction_mode")
    if mode in {None, "manual", "states", "exact"}:
        return str(mode or "manual")
    return "invalid"


def _reference_selection(
    project: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]],
    scene: Mapping[str, Any],
    frame: Mapping[str, Any],
    *,
    frame_id: str,
    blockers: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    selected = frame.get("references") or []
    if not isinstance(selected, list):
        _block(blockers, "INVALID_REFS", "references должен быть списком", scene=str(scene.get("id")), frame=frame_id)
        return []
    visible = {str(item) for item in scene.get("subjects_visible") or []}
    absent = {str(item) for item in scene.get("subjects_absent") or []}
    result: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    identity_shas: dict[str, str] = {}
    counts = {"identity": 0, "style": 0, "environment": 0}
    for index, item in enumerate(selected):
        if not isinstance(item, Mapping):
            _block(blockers, "INVALID_REFS", f"references[{index}] должен быть объектом", frame=frame_id)
            continue
        asset_id = item.get("asset_id")
        role = item.get("role")
        if not isinstance(asset_id, str) or asset_id not in assets:
            _block(blockers, "UNKNOWN_ASSET", f"unknown asset_id {asset_id!r}", frame=frame_id)
            continue
        if role not in SEMANTIC_ROLES:
            _block(blockers, "INVALID_REF_ROLE", f"reference role должен быть identity/style/environment: {role!r}", frame=frame_id)
            continue
        asset = assets[asset_id]
        expected = ASSET_TO_SEMANTIC.get(str(asset.get("role")))
        if expected is None:
            _block(blockers, "PROVIDED_AS_REF", "provided_keyframe нельзя использовать как reference", frame=frame_id)
            continue
        if expected != role:
            _block(
                blockers,
                "REF_ROLE_MISMATCH",
                f"asset {asset_id} role {asset.get('role')} не соответствует {role}",
                frame=frame_id,
            )
            continue
        if asset_id in seen_assets:
            _block(
                blockers,
                "DUPLICATE_ASSET_REF",
                f"asset {asset_id} выбран дважды в одном frame; оставьте один reference",
                frame=frame_id,
            )
            continue
        if role == "identity":
            if counts["identity"] >= MAX_IDENTITY_REFS_PER_FRAME:
                _block(
                    blockers,
                    "IDENTITY_REF_LIMIT",
                    f"identity references: максимум {MAX_IDENTITY_REFS_PER_FRAME} в frame",
                    frame=frame_id,
                )
                continue
            sha = str(asset.get("sha256") or "")
            previous = identity_shas.get(sha)
            if previous is not None and previous != asset_id:
                _block(
                    blockers,
                    "DUPLICATE_REF_CONTENT",
                    f"две identity refs имеют одинаковый content SHA256 ({asset_id} и {previous}); выберите разные изображения",
                    frame=frame_id,
                )
                continue
            identity_shas[sha] = asset_id
        elif role == "environment":
            if counts["environment"] >= MAX_ENVIRONMENT_REFS_PER_FRAME:
                _block(
                    blockers,
                    "DUPLICATE_ROLE",
                    "environment reference: максимум 1 в frame",
                    frame=frame_id,
                )
                continue
        elif role == "style":
            if counts["style"] >= MAX_STYLE_REFS_PER_FRAME:
                _block(
                    blockers,
                    "DUPLICATE_ROLE",
                    "style reference: максимум 1 в frame",
                    frame=frame_id,
                )
                continue
        subject_id = asset.get("subject_id")
        if role == "identity" and subject_id:
            if str(subject_id) in absent:
                _block(blockers, "ABSENT_SUBJECT_REF", f"subject {subject_id} отсутствует в scene", frame=frame_id)
                continue
            if visible and str(subject_id) not in visible:
                _block(blockers, "REF_NOT_VISIBLE", f"subject {subject_id} не в subjects_visible", frame=frame_id)
                continue
        if role == "identity" and not subject_id:
            _warn(warnings, f"{frame_id}: identity asset без subject_id; identity не привязана к visible/absent")
        counts[role] += 1
        seen_assets.add(asset_id)
        result.append({"asset_id": asset_id, "role": role, "asset": asset})
    for subject_id in sorted(visible):
        if not any(
            item["role"] == "identity" and item["asset"].get("subject_id") == subject_id
            for item in result
        ):
            _warn(warnings, f"{frame_id}: visible subject {subject_id} без identity reference; identity не гарантируется")
    return result


def _compose_instruction(
    scene: Mapping[str, Any], instruction: str, subject_labels: Mapping[str, str]
) -> str:
    parts = [instruction.strip()]
    action = scene.get("action")
    camera = scene.get("camera")
    if isinstance(action, str) and action.strip():
        parts.append("Action: " + action.strip())
    if isinstance(camera, str) and camera.strip():
        parts.append("Camera: " + camera.strip())
    visible = [subject_labels.get(str(item), str(item)) for item in scene.get("subjects_visible") or []]
    absent = [subject_labels.get(str(item), str(item)) for item in scene.get("subjects_absent") or []]
    if visible:
        parts.append("Visible subjects: " + ", ".join(visible))
    if absent:
        parts.append("Absent subjects: " + ", ".join(absent))
    return "\n".join(part for part in parts if part)


def validate_and_compile(project: Mapping[str, Any], project_dir: Path) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    warnings: list[str] = []
    frames: list[dict[str, Any]] = []
    plan: dict[str, Any] | None = None

    if project.get("schema") != "anime-studio/kf2-project/v1":
        _block(blockers, "PROJECT_SCHEMA", "project schema должен быть anime-studio/kf2-project/v1")
    title = project.get("title")
    if not isinstance(title, str) or not title.strip():
        _block(blockers, "MISSING_TITLE", "title обязателен")
    orientation = project.get("orientation")
    if orientation not in ORIENTATIONS:
        _block(blockers, "ORIENTATION", "orientation должен быть portrait или landscape")
        candidate, video = PORTRAIT_CANDIDATE, PORTRAIT_VIDEO
    else:
        candidate, video = ORIENTATIONS[orientation]
        if project.get("candidate_geometry") != candidate:
            _block(blockers, "GEOMETRY_MISMATCH", f"candidate_geometry должен быть {candidate}")
        if project.get("video_geometry") != video:
            _block(blockers, "GEOMETRY_MISMATCH", f"video_geometry должен быть {video}")

    library = project.get("library") or []
    if not isinstance(library, list):
        _block(blockers, "LIBRARY", "library должен быть списком")
        library = []
    assets: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(library):
        if not isinstance(raw, Mapping):
            _block(blockers, "ASSET", f"library[{index}] должен быть объектом")
            continue
        asset_id = raw.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            _block(blockers, "ASSET_ID", f"library[{index}].id обязателен")
            continue
        if asset_id in assets:
            _block(blockers, "ASSET_DUPLICATE", f"duplicate asset id {asset_id}")
            continue
        try:
            checked = validate_library_asset(project_dir, raw)
        except Exception as exc:
            _block(blockers, "ASSET_INVALID", f"{asset_id}: {exc}")
            continue
        if checked.get("role") not in {
            "character",
            "creature",
            "object",
            "environment",
            "style",
            "provided_keyframe",
        }:
            _block(blockers, "ASSET_ROLE", f"{asset_id}: неизвестная role")
            continue
        assets[asset_id] = checked

    subjects_raw = project.get("subjects") or []
    if not isinstance(subjects_raw, list):
        _block(blockers, "SUBJECTS", "subjects должен быть списком")
        subjects_raw = []
    subjects: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(subjects_raw):
        if not isinstance(raw, Mapping):
            _block(blockers, "SUBJECT", f"subjects[{index}] должен быть объектом")
            continue
        subject_id = raw.get("id")
        if not isinstance(subject_id, str) or not PROJECT_SAFE_ID.fullmatch(subject_id):
            _block(blockers, "SUBJECT_ID", f"invalid subject id {subject_id!r}")
            continue
        if subject_id in subjects:
            _block(blockers, "SUBJECT_DUPLICATE", f"duplicate subject id {subject_id}")
            continue
        kind = raw.get("kind")
        if kind not in SUBJECT_KINDS:
            _block(blockers, "SUBJECT_KIND", f"{subject_id}: kind должен быть character/creature/object/environment")
            continue
        label = raw.get("label")
        if not isinstance(label, str) or not label.strip():
            _block(blockers, "SUBJECT_LABEL", f"{subject_id}: label обязателен")
            continue
        views = raw.get("views") or []
        if not isinstance(views, list):
            _block(blockers, "SUBJECT_VIEWS", f"{subject_id}: views должен быть списком")
            views = []
        for view in views:
            if not isinstance(view, Mapping) or view.get("asset_id") not in assets:
                _block(blockers, "SUBJECT_VIEW", f"{subject_id}: unknown view asset")
                continue
            asset = assets[str(view["asset_id"])]
            if asset.get("subject_id") != subject_id:
                _block(blockers, "SUBJECT_VIEW", f"{subject_id}: asset subject_id не совпадает")
            if kind != "environment" and asset.get("role") != kind:
                _block(blockers, "SUBJECT_VIEW_ROLE", f"{subject_id}: asset role не совпадает с kind")
            if kind == "environment" and asset.get("role") != "environment":
                _block(blockers, "SUBJECT_VIEW_ROLE", f"{subject_id}: environment asset ожидается")
        for view in views:
            asset = assets.get(str(view.get("asset_id")))
            if asset is not None:
                label = view.get("label")
                if not isinstance(label, str) or not label.strip():
                    _block(blockers, "SUBJECT_VIEW_LABEL", f"{subject_id}: view label пуст", scene=subject_id)
                elif asset.get("view") and asset.get("view") != label:
                    _block(
                        blockers,
                        "SUBJECT_VIEW_LABEL",
                        f"{subject_id}: view label {label!r} не совпадает с asset.view {asset.get('view')!r}",
                        scene=subject_id,
                    )
        subjects[subject_id] = {**dict(raw), "views": views}
    for asset_id, asset in assets.items():
        subject_id = asset.get("subject_id")
        if not subject_id:
            continue
        subject = subjects.get(str(subject_id))
        if subject is None:
            _block(blockers, "SUBJECT_VIEW", f"{asset_id}: subject {subject_id} не существует")
            continue
        asset_view = asset.get("view")
        if asset_view:
            found = any(
                isinstance(view, Mapping)
                and view.get("asset_id") == asset_id
                and view.get("label") == asset_view
                for view in subject.get("views") or []
            )
            if not found:
                _block(
                    blockers,
                    "SUBJECT_VIEW_LABEL",
                    f"{asset_id}: subject.views не содержит association {asset_view!r}",
                )
    subject_labels = {
        subject_id: str(value.get("label", subject_id))
        for subject_id, value in subjects.items()
    }

    scenes = project.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        _block(blockers, "SCENES", "нужна хотя бы одна scene")
        scenes = []
    scene_ids: set[str] = set()
    compiled_scenes: list[dict[str, Any]] = []
    reference_catalog: dict[str, dict[str, Any]] = {}
    asset_ref_ids: dict[tuple[str, str], str] = {}
    used_aliases: set[str] = set()
    previous_b_generated = False

    for scene_index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            _block(blockers, "SCENE", f"scenes[{scene_index}] должен быть объектом")
            continue
        scene_id = scene.get("id")
        if not isinstance(scene_id, str) or not SAFE_ID.fullmatch(scene_id) or scene_id in scene_ids:
            _block(blockers, "SCENE_ID", f"invalid/duplicate scene id {scene_id!r}")
            continue
        scene_ids.add(scene_id)
        description = scene.get("description")
        if not isinstance(description, str) or not description.strip():
            _block(blockers, "SCENE_DESCRIPTION", "description обязателен", scene=scene_id)
            description = ""
        visible = scene.get("subjects_visible") or []
        absent = scene.get("subjects_absent") or []
        if not isinstance(visible, list) or not isinstance(absent, list):
            _block(blockers, "SUBJECTS_STATE", "subjects_visible/subjects_absent должны быть списками", scene=scene_id)
            visible, absent = [], []
        visible_ids = {str(item) for item in visible}
        absent_ids = {str(item) for item in absent}
        for subject_id in visible_ids | absent_ids:
            if subject_id not in subjects:
                _block(blockers, "UNKNOWN_SUBJECT", f"unknown subject {subject_id}", scene=scene_id)
        if visible_ids & absent_ids:
            _block(blockers, "SUBJECT_STATE_CONFLICT", "subject одновременно visible и absent", scene=scene_id)
        shared_inv = _tokens(scene.get("shared_invariants"), "shared_invariants", blockers, scene=scene_id, frame="shared")
        shared_allowed = _tokens(scene.get("shared_allowed_changes"), "shared_allowed_changes", blockers, scene=scene_id, frame="shared")
        if set(shared_inv) & set(shared_allowed):
            _block(blockers, "TOKEN_OVERLAP", "shared invariants/allowed_changes пересекаются", scene=scene_id)

        compiled_scene = {"id": scene_id, "description": description.strip(), "a": {}, "b": {}}
        for role, final in (("A", False), ("B", True)):
            key = "initial" if role == "A" else "final"
            raw_frame = scene.get(key)
            if not isinstance(raw_frame, Mapping):
                _block(blockers, "FRAME", f"{key} должен быть объектом", scene=scene_id, frame=f"{scene_id}/{role}")
                continue
            frame_id = f"{scene_id}/{role}"
            kind = raw_frame.get("kind")
            if role == "B":
                initial_raw = scene.get("initial") if isinstance(scene, Mapping) else None
                anchor_generated = bool(
                    scene_index > 0
                    and isinstance(initial_raw, Mapping)
                    and initial_raw.get("kind") == "new_shot"
                )
            elif raw_frame.get("use_previous_anchor"):
                anchor_generated = previous_b_generated
            else:
                anchor_generated = False
            if role == "A" and scene_index == 0:
                if kind != "first_frame":
                    _block(blockers, "A1_SOURCE", "A1 должен быть first_frame", scene=scene_id, frame=frame_id)
            elif role == "A":
                if kind not in {"new_shot", "provided_keyframe"}:
                    _block(
                        blockers,
                        "A_NEW_SHOT_ONLY",
                        "A следующей сцены поддерживает только new_shot; для той же камеры используйте B или новую сцену",
                        scene=scene_id,
                        frame=frame_id,
                    )
            else:
                if kind not in {"continue", "provided_keyframe"}:
                    _block(
                        blockers,
                        "B_CONTINUE_ONLY",
                        "B поддерживает только continue_shot; новую камеру оформляйте новой сценой",
                        scene=scene_id,
                        frame=frame_id,
                    )

            if role == "A" and scene_index == 0 and kind == "first_frame":
                asset_id = raw_frame.get("asset_id")
                if not isinstance(asset_id, str) or asset_id not in assets:
                    _block(blockers, "FIRST_FRAME", "A1 требует provided_keyframe asset", scene=scene_id, frame=frame_id)
                    continue
                asset = assets[asset_id]
                if asset.get("role") != "provided_keyframe":
                    _block(blockers, "FIRST_FRAME_ROLE", "A1 asset должен иметь role provided_keyframe", scene=scene_id, frame=frame_id)
                    continue
                compiled_scene["a"] = {"source": "first_frame"}
                frames.append(
                    {
                        "frame_id": frame_id,
                        "scene_id": scene_id,
                        "role": role,
                        "generate": False,
                        "source": "first_frame",
                        "approval_required": False,
                        "asset_id": asset_id,
                    }
                )
                continue

            if kind == "provided_keyframe":
                asset_id = raw_frame.get("asset_id")
                if not isinstance(asset_id, str) or asset_id not in assets:
                    _block(blockers, "PROVIDED_ASSET", "provided_keyframe требует asset_id", scene=scene_id, frame=frame_id)
                    continue
                asset = assets[asset_id]
                if asset.get("role") != "provided_keyframe":
                    _block(blockers, "PROVIDED_ROLE", "provided frame asset должен иметь role provided_keyframe", scene=scene_id, frame=frame_id)
                    continue
                if role == "A" and scene_index == 0:
                    provenance = dict(asset.get("provenance") or {})
                    provenance["content_sha256"] = asset.get("sha256")
                else:
                    provenance = _validate_provenance(asset, blockers, frame_id=frame_id)
                if provenance is None:
                    continue
                compiled_scene["a" if role == "A" else "b"] = {
                    "source": "first_frame" if role == "A" and scene_index == 0 else "provided_frame",
                    **(
                        {"provided_frame": {"path": _asset_path(project_dir, asset), "sha256": asset.get("sha256"), "provenance": provenance}}
                        if not (role == "A" and scene_index == 0)
                        else {}
                    ),
                }
                if role == "A" and scene_index == 0:
                    compiled_scene["a"] = {"source": "first_frame"}
                frames.append(
                    {
                        "frame_id": frame_id,
                        "scene_id": scene_id,
                        "role": role,
                        "generate": False,
                        "source": "first_frame" if role == "A" and scene_index == 0 else "provided_frame",
                        "approval_required": False,
                        "asset_id": asset_id,
                    }
                )
                continue

            mode = _instruction_mode(scene, raw_frame)
            if mode == "invalid":
                _block(
                    blockers,
                    "INVALID_INSTRUCTION_MODE",
                    "instruction_mode должен быть manual, states или exact",
                    scene=scene_id,
                    frame=frame_id,
                )
                mode = "manual"
            instruction = raw_frame.get("instruction")
            if mode == "states":
                target_key = "state_a" if role == "A" else "state_b"
                target = raw_frame.get(target_key)
                if not isinstance(target, str) or not target.strip():
                    _block(
                        blockers,
                        "MISSING_STATE",
                        f"режим states требует непустое состояние {target_key}",
                        scene=scene_id,
                        frame=frame_id,
                    )
                    instruction = ""
                else:
                    instruction = _compose_state_instruction(
                        scene, raw_frame, role, subject_labels
                    )
            elif not isinstance(instruction, str) or not instruction.strip():
                _block(blockers, "MISSING_INSTRUCTION", "нужна exact user edit_instruction", scene=scene_id, frame=frame_id)
                instruction = ""
            selected = _reference_selection(
                project, assets, scene, raw_frame, frame_id=frame_id, blockers=blockers, warnings=warnings
            )
            selected_ref_ids: list[str] = []
            for item in selected:
                ref_id = _catalog_ref_id(
                    item,
                    catalog=reference_catalog,
                    asset_ref_ids=asset_ref_ids,
                    used_aliases=used_aliases,
                )
                selected_ref_ids.append(ref_id)
                if len(reference_catalog) > MAX_USED_REFERENCES:
                    _block(
                        blockers,
                        "REFERENCE_CATALOG_LIMIT",
                        f"reference catalog превышает {MAX_USED_REFERENCES} used refs; "
                        "уберите лишние references или уменьшите число сцен",
                        scene=scene_id,
                        frame=frame_id,
                    )
                    break

            anchor = role == "B" or bool(raw_frame.get("use_previous_anchor"))
            reference_roles = (["anchor"] if anchor else []) + selected_ref_ids
            if not reference_roles:
                _block(blockers, "EMPTY_REFERENCES", "generated frame требует 1..3 references", scene=scene_id, frame=frame_id)
            if len(reference_roles) > 3:
                _block(
                    blockers,
                    "REF_LIMIT",
                    f"выбрано {len(reference_roles)} reference: {reference_roles}; v2 допускает 1..3, ничего не отброшено",
                    scene=scene_id,
                    frame=frame_id,
                )
            if role == "B":
                invariants = _merge_tokens(shared_inv, _tokens(raw_frame.get("invariants"), "invariants", blockers, scene=scene_id, frame=frame_id), ["camera"])
                allowed = _merge_tokens(shared_allowed, _tokens(raw_frame.get("allowed_changes"), "allowed_changes", blockers, scene=scene_id, frame=frame_id))
                if "camera" in allowed or "lens" in allowed:
                    _block(
                        blockers,
                        "B_CAMERA_CHANGE_UNSUPPORTED",
                        "B не может менять camera/lens; создайте новую сцену вместо скрытой смены shot_mode",
                        scene=scene_id,
                        frame=frame_id,
                    )
                shot_mode = "continue_shot"
            else:
                invariants = _merge_tokens(shared_inv, _tokens(raw_frame.get("invariants"), "invariants", blockers, scene=scene_id, frame=frame_id))
                allowed = _merge_tokens(shared_allowed, _tokens(raw_frame.get("allowed_changes"), "allowed_changes", blockers, scene=scene_id, frame=frame_id))
                shot_mode = "new_shot"
            overlap = set(invariants) & set(allowed)
            if overlap:
                _block(blockers, "TOKEN_OVERLAP", f"invariants/allowed_changes пересекаются: {sorted(overlap)}", scene=scene_id, frame=frame_id)
            requested_approval = raw_frame.get("require_anchor_approval", None)
            if requested_approval is None:
                require_approval = bool(anchor and anchor_generated)
            elif not isinstance(requested_approval, bool):
                _block(blockers, "APPROVAL_FLAG", "require_anchor_approval должен быть bool", scene=scene_id, frame=frame_id)
                require_approval = bool(anchor and anchor_generated)
            else:
                require_approval = requested_approval
            if not anchor:
                require_approval = False
            if require_approval and not anchor_generated:
                _block(
                    blockers,
                    "ANCHOR_APPROVAL_UNSUPPORTED",
                    "owner approval доступен только для generated anchor; first_frame/provided нельзя одобрить через existing approval",
                    scene=scene_id,
                    frame=frame_id,
                )
            edit_instruction = (
                instruction
                if mode in {"states", "exact"}
                else _compose_instruction(scene, instruction, subject_labels)
            )
            compiled_scene["a" if role == "A" else "b"] = {
                "source": "generate_from_scene_a" if role == "B" else (
                    "first_frame" if scene_index == 0 else "generate_from_previous_b"
                ),
                "shot_mode": shot_mode,
                "reference_roles": reference_roles,
                "invariants": invariants,
                "allowed_changes": allowed,
                "require_anchor_approval": require_approval,
                "edit_instruction": edit_instruction,
            }
            frames.append(
                {
                    "frame_id": frame_id,
                    "scene_id": scene_id,
                    "role": role,
                    "generate": True,
                    "source": "generate_from_scene_a" if role == "B" else (
                        "first_frame" if scene_index == 0 else "generate_from_previous_b"
                    ),
                    "shot_mode": shot_mode,
                    "reference_roles": reference_roles,
                    "instruction": edit_instruction,
                    "approval_required": require_approval,
                }
            )
        compiled_scenes.append(compiled_scene)
        final_raw = scene.get("final") if isinstance(scene, Mapping) else None
        previous_b_generated = bool(
            isinstance(final_raw, Mapping) and final_raw.get("kind") == "continue"
        )

    generated = [frame for frame in frames if frame["generate"]]
    if blockers:
        return {
            "plan": None,
            "blockers": blockers,
            "warnings": warnings,
            "frames": frames,
            "generated_count": len(generated),
            "reference_catalog_count": len(reference_catalog),
            "reference_catalog_limit": MAX_USED_REFERENCES,
        }
    first_frame_asset = None
    if scenes and isinstance(scenes[0], Mapping):
        initial = scenes[0].get("initial")
        if isinstance(initial, Mapping):
            first_frame_asset = assets.get(str(initial.get("asset_id")))
    if first_frame_asset is None:
        _block(blockers, "FIRST_FRAME", "A1 provided_keyframe asset не найден")
        return {
            "plan": None,
            "blockers": blockers,
            "warnings": warnings,
            "frames": frames,
            "generated_count": len(generated),
            "reference_catalog_count": len(reference_catalog),
            "reference_catalog_limit": MAX_USED_REFERENCES,
        }
    generation_limit = project.get("generation_limit", max(DEFAULT_GENERATION_LIMIT, len(generated)))
    if not isinstance(generation_limit, int) or isinstance(generation_limit, bool) or not 1 <= generation_limit <= 99:
        _block(blockers, "GENERATION_LIMIT", "generation_limit должен быть 1..99")
        generation_limit = max(DEFAULT_GENERATION_LIMIT, len(generated))
    elif generation_limit < len(generated):
        _block(blockers, "GENERATION_LIMIT", f"generation_limit {generation_limit} меньше generated frames {len(generated)}")
    seed = project.get("seed", DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2_147_483_647:
        _block(blockers, "SEED", "seed должен быть 0..2147483647")
        seed = DEFAULT_SEED
    if blockers:
        return {
            "plan": None,
            "blockers": blockers,
            "warnings": warnings,
            "frames": frames,
            "generated_count": len(generated),
            "reference_catalog_count": len(reference_catalog),
            "reference_catalog_limit": MAX_USED_REFERENCES,
        }

    style = project.get("style") if isinstance(project.get("style"), str) else ""
    subject = None
    subject_values = list(subjects.values())
    if subject_values:
        subject = {
            "label": _subject_label(project, subject_values),
            "kind": str(subject_values[0].get("kind", "subject")),
        }
    policy = {
        "candidate_geometry": candidate,
        "video_geometry": video,
        "normalization": "explicit proportional contain, never stretch",
        "overwrite_inputs": False,
        "automatic_video_submit": False,
        "first_frame_policy": "contain",
        "anchor_policy": "contain",
        "character_policy": "contain",
        "environment_policy": "contain",
        "video_policy": "contain",
    }
    legacy_refs: dict[str, str] = {}
    plan_references: list[dict[str, Any]] = []
    for ref_id, entry in reference_catalog.items():
        asset = entry["asset"]
        path = _asset_path(project_dir, asset)
        if ref_id == "character_reference":
            legacy_refs["character_reference"] = path
        elif ref_id == "background_reference":
            legacy_refs["background_reference"] = path
        else:
            plan_references.append(
                {
                    "id": ref_id,
                    "role": entry["role"],
                    "path": path,
                    "sha256": asset.get("sha256"),
                }
            )
    plan = {
        "schema": PLAN_SCHEMA_V2,
        "title": str(title).strip(),
        "language": "ru",
        **({"subject": subject} if subject else {}),
        "style": style,
        "first_frame": _asset_path(project_dir, first_frame_asset),
        **legacy_refs,
        **({"references": plan_references} if plan_references else {}),
        "scenes": compiled_scenes,
        "output_policy": policy,
        "generation_limit": generation_limit,
        "seed": seed,
        "seed_policy": "per_frame",
        "notes": str(project.get("notes") or ""),
    }
    return {
        "plan": plan,
        "blockers": blockers,
        "warnings": warnings,
        "frames": frames,
        "generated_count": len(generated),
        "reference_catalog_count": len(reference_catalog),
        "reference_catalog_limit": MAX_USED_REFERENCES,
        "approval_required_frames": [
            frame["frame_id"] for frame in generated if frame.get("approval_required")
        ],
    }


__all__ = [
    "LANDSCAPE_CANDIDATE",
    "LANDSCAPE_VIDEO",
    "ORIENTATIONS",
    "PORTRAIT_CANDIDATE",
    "PORTRAIT_VIDEO",
    "validate_and_compile",
]
