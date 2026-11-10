"""Plan loading/validation for KF1.

The checked-in example JSON is read-only.  A user may also start from a plain
text template or interactive questions; JSON is never required.  All user-facing
text (including Russian scene descriptions) is preserved verbatim.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .errors import PlanError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PLAN = ROOT / "config" / "keyframes" / "two-scenes.example.json"
K1_MANIFEST = ROOT / "manifests" / "K1.qwen-edit-2511.sources.draft.json"
PROFILE_ID = "keyframes-qwen-edit-2511"
PLAN_SCHEMA = "anime-studio/keyframe-plan/v1"
PLAN_SCHEMA_V2 = "anime-studio/keyframe-plan/v2"
PLAN_SCHEMAS = {PLAN_SCHEMA, PLAN_SCHEMA_V2}
EXAMPLE_SCHEMA = "anime-studio/keyframe-plan-example/v1"
PLAN_SCHEMA_VERSIONS = {
    PLAN_SCHEMA: 1,
    EXAMPLE_SCHEMA: 1,
    PLAN_SCHEMA_V2: 2,
}
SHOT_MODES = {"continue_shot", "new_shot"}
REFERENCE_ROLE_VOCAB = {"identity", "style", "environment"}
# Bounded global catalog of actually used references (excluding first_frame).
# This is intentionally much smaller than KF2's 1000-asset library.
MAX_USED_REFERENCES = 64
MAX_IDENTITY_REFS_PER_FRAME = 2
MAX_ENVIRONMENT_REFS_PER_FRAME = 1
MAX_STYLE_REFS_PER_FRAME = 1
# These are deliberately domain-neutral.  No person/anime-specific terms are
# hardcoded in the prompt builder; a plan may put its own wording in
# description/edit_instruction/style.
INVARIANT_VOCAB = {
    "camera",
    "lens",
    "background",
    "environment",
    "position",
    "apparent_scale",
    "pose",
    "gaze",
    "identity",
    "style",
    "lighting",
    "expression",
    "wardrobe",
    "subject_count",
    "framing",
    "composition",
}
DEFAULT_SEED = 424242
DEFAULT_STEPS = 40
DEFAULT_CFG = 4.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_SHIFT = 3.1
DEFAULT_GENERATION_LIMIT = 3
DEFAULT_CANDIDATE = (704, 1280)
DEFAULT_VIDEO = (448, 816)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|KF1_BIND_REQUIRED")

WORKFLOW_FILES: dict[int, str] = {
    1: "workflows/K1/qwen-edit-2511-1ref.api.json",
    2: "workflows/K1/qwen-edit-2511-2ref.api.json",
    3: "workflows/K1/qwen-edit-2511-3ref.api.json",
}


@dataclass(frozen=True)
class RefInfo:
    role: str
    value: str
    path: Path
    sha256: str
    size_bytes: int
    width: int
    height: int
    format: str


@dataclass(frozen=True)
class FrameSpec:
    scene_id: str
    role: str
    frame_id: str
    generate: bool
    source: str
    edit_instruction: str | None
    anchor_frame: str | None
    anchor_role: str | None
    description: str
    shot_mode: str | None = None
    invariants: tuple[str, ...] = ()
    allowed_changes: tuple[str, ...] = ()
    reference_roles: tuple[str, ...] = ()
    require_anchor_approval: bool = False
    negative_prompt: str | None = None
    provided_path: Path | None = None
    provided_sha256: str | None = None
    provided_provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedPlan:
    path: Path
    data: dict[str, Any]
    plan_sha256: str
    refs: dict[str, RefInfo]
    output_policy: dict[str, Any]
    frames: tuple[FrameSpec, ...]
    pins: dict[str, Any]
    settings: dict[str, Any]
    schema_version: int = 1
    provided_frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    reference_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    subject_label: str = "subject"

    @property
    def title(self) -> str:
        return str(self.data.get("title") or self.path.stem)

    @property
    def generation_limit(self) -> int:
        return int(self.data.get("generation_limit", DEFAULT_GENERATION_LIMIT))

    @property
    def generated_frames(self) -> tuple[FrameSpec, ...]:
        return tuple(frame for frame in self.frames if frame.generate)

    def scene_count(self) -> int:
        return len(self.data.get("scenes") or [])


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _as_int(value: Any, name: str, *, minimum: int = 1, maximum: int = 10_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PlanError(f"{name} должен быть целым числом {minimum}..{maximum}")
    return value


def _as_geometry(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", value)
        if not match:
            raise PlanError(f"{name} должен быть WxH")
        value = [int(match.group(1)), int(match.group(2))]
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise PlanError(f"{name} должен быть списком из двух целых")
    width, height = int(value[0]), int(value[1])
    if not 16 <= width <= 8192 or not 16 <= height <= 8192:
        raise PlanError(f"{name} вне допустимого диапазона")
    return width, height


def _safe_text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PlanError(f"{name} должен быть строкой")
    text = value.strip()
    if not text and not allow_empty:
        raise PlanError(f"{name} не должен быть пустым")
    if "\x00" in text:
        raise PlanError(f"{name} содержит NUL")
    return text


def _resolve_ref(
    root: Path,
    value: Any,
    *,
    role: str,
    required: bool,
    expected_sha256: str | None = None,
) -> RefInfo | None:
    if value in (None, ""):
        if required:
            raise PlanError(f"Не задан обязательный {role}")
        return None
    text = _safe_text(value, role)
    candidate = Path(text)
    path = candidate if candidate.is_absolute() else root / candidate
    if path.is_symlink():
        raise PlanError(f"{role} не должен быть symlink: {text}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PlanError(f"{role} не найден: {text}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise PlanError(f"{role} должен быть обычным файлом без symlink: {text}")
    raw = resolved.read_bytes()
    if not raw:
        raise PlanError(f"{role} пуст: {text}")
    # Imported lazily so plan validation remains usable before Pillow import in
    # offline source-contract checks.
    from .normalize import inspect_image

    info = inspect_image(resolved)
    actual_sha = _sha_bytes(raw)
    if expected_sha256 is not None:
        expected = str(expected_sha256).lower()
        if len(expected) != 64 or any(
            char not in "0123456789abcdef" for char in expected
        ):
            raise PlanError(f"{role}: sha256 должен быть 64 hex")
        if actual_sha != expected:
            raise PlanError(
                f"{role}: SHA256 не совпадает с ожидаемым "
                f"{expected} != {actual_sha}"
            )
    return RefInfo(
        role=role,
        value=text,
        path=resolved,
        sha256=actual_sha,
        size_bytes=len(raw),
        width=int(info["width"]),
        height=int(info["height"]),
        format=str(info["format"]),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"Не удалось прочитать план {path}: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise PlanError("План должен быть JSON-объектом")
    return data


def parse_text_plan(text: str) -> dict[str, Any]:
    """Parse the simple line-oriented text template.

    Values are everything after the first colon; quotes are optional.  This is
    deliberately not YAML/JSON: a Russian scene description is never parsed as
    syntax.
    """

    data: dict[str, Any] = {"schema": PLAN_SCHEMA, "scenes": []}
    current: dict[str, Any] | None = None
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise PlanError(f"Строка текстового плана без ':' — {line!r}")
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key == "scene":
            current = {"id": value, "a": {}, "b": {}}
            data["scenes"].append(current)
            continue
        if current is not None and key in {
            "description",
            "a_source",
            "a_edit",
            "b_source",
            "b_edit",
        }:
            if key == "description":
                current["description"] = value
            elif key == "a_source":
                current["a"]["source"] = value
            elif key == "a_edit":
                current["a"]["edit_instruction"] = value
            elif key == "b_source":
                current["b"]["source"] = value
            elif key == "b_edit":
                current["b"]["edit_instruction"] = value
            continue
        current = None
        if key in {"title", "language", "first_frame", "character_reference", "background_reference", "style", "normalization"}:
            data[key] = value
        elif key in {"generation_limit", "seed"}:
            try:
                data[key] = int(value)
            except ValueError as exc:
                raise PlanError(f"{key} должен быть числом") from exc
        elif key in {"candidate_geometry", "video_geometry"}:
            data[key] = value
        elif key in {"first_frame_policy", "anchor_policy", "character_policy", "environment_policy", "video_policy"}:
            data.setdefault("output_policy", {})[key] = value
        else:
            raise PlanError(f"Неизвестное поле текстового плана: {key}")
    if not data["scenes"]:
        raise PlanError("В текстовом плане нет ни одной сцены")
    return data


def interactive_plan(
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Ask simple questions; never require the user to write JSON."""

    def ask(prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        value = input_fn(f"{prompt}{suffix}: ").strip()
        return value or default

    print_fn("Создание плана опорных кадров. Пустой ответ использует значение в скобках.")
    data: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "title": ask("Название", "Новые опорные кадры"),
        "language": ask("Язык описаний", "ru"),
        "first_frame": ask("Первый кадр (A1, обязательный)"),
        "character_reference": ask("Постоянный character reference (пусто = нет)", ""),
        "background_reference": ask("Environment reference (пусто = нет)", ""),
        "style": ask("Стиль/общие ограничения", "Сохранить стиль исходного кадра."),
        "seed": int(ask("Seed", str(DEFAULT_SEED))),
        "generation_limit": int(ask("Лимит генераций", str(DEFAULT_GENERATION_LIMIT))),
        "output_policy": {
            "candidate_geometry": [704, 1280],
            "video_geometry": [448, 816],
            "normalization": "explicit proportional contain/crop, never stretch",
            "overwrite_inputs": False,
            "automatic_video_submit": False,
            "first_frame_policy": "preserve",
            "anchor_policy": ask("Политика anchor (contain/crop)", "contain"),
            "character_policy": "contain",
            "environment_policy": "contain",
            "video_policy": "contain",
        },
        "scenes": [],
    }
    if not data["first_frame"]:
        raise PlanError("Первый кадр обязателен")
    index = 1
    while True:
        scene_default = f"shot-{index:02d}"
        scene_id = input_fn(
            f"ID сцены {index} (пусто = закончить) [{scene_default}]: "
        ).strip()
        if not scene_id:
            break
        scene: dict[str, Any] = {
            "id": scene_id,
            "description": ask(f"Сцена {scene_id}: русское описание"),
            "a": {},
            "b": {},
        }
        if index == 1:
            scene["a"]["source"] = "first_frame"
        else:
            scene["a"]["source"] = "generate_from_previous_b"
            scene["a"]["edit_instruction"] = ask(
                f"Сцена {scene_id}: новое положение камеры/действие для A"
            )
        scene["b"]["source"] = "generate_from_scene_a"
        scene["b"]["edit_instruction"] = ask(
            f"Сцена {scene_id}: что меняется в B (камера/фон сохраняются)"
        )
        data["scenes"].append(scene)
        index += 1
    if not data["scenes"]:
        raise PlanError("Нужна хотя бы одна сцена")
    return data


def _normalize_output_policy(raw: Any) -> dict[str, Any]:
    data = dict(raw) if isinstance(raw, Mapping) else {}
    candidate = _as_geometry(data.get("candidate_geometry", DEFAULT_CANDIDATE), "candidate_geometry")
    video = _as_geometry(data.get("video_geometry", DEFAULT_VIDEO), "video_geometry")
    policy = {
        "candidate_geometry": list(candidate),
        "video_geometry": list(video),
        "normalization": _safe_text(
            data.get("normalization", "explicit proportional contain/crop, never stretch"),
            "output_policy.normalization",
        ),
        "overwrite_inputs": data.get("overwrite_inputs", False),
        "automatic_video_submit": data.get("automatic_video_submit", False),
        "first_frame_policy": _safe_text(
            data.get("first_frame_policy", "preserve"), "first_frame_policy"
        ),
        "anchor_policy": _safe_text(data.get("anchor_policy", "contain"), "anchor_policy"),
        "character_policy": _safe_text(
            data.get("character_policy", "contain"), "character_policy"
        ),
        "environment_policy": _safe_text(
            data.get("environment_policy", "contain"), "environment_policy"
        ),
        "video_policy": _safe_text(data.get("video_policy", "contain"), "video_policy"),
    }
    for key in ("first_frame_policy", "anchor_policy", "character_policy", "environment_policy", "video_policy"):
        if policy[key] not in {"preserve", "contain", "crop"}:
            raise PlanError(f"{key} должен быть preserve/contain/crop")
    if policy["overwrite_inputs"] is not False:
        raise PlanError("overwrite_inputs должен быть false: исходные refs неизменны")
    if policy["automatic_video_submit"] is not False:
        raise PlanError("automatic_video_submit должен быть false")
    return policy


def _source_pins(root: Path) -> dict[str, Any]:
    manifest_path = root / K1_MANIFEST.relative_to(ROOT)
    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"Не найден source inventory K1: {manifest_path}") from exc
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise PlanError("K1 source inventory должен содержать 3 модели")
    model_pins = []
    for item in models:
        if not isinstance(item, Mapping):
            raise PlanError("Некорректная модель в K1 inventory")
        model_pins.append(
            {
                "role": item.get("role"),
                "target": item.get("target"),
                "size_bytes": item.get("size_bytes"),
                "sha256": str(item.get("expected_sha256") or "").lower(),
            }
        )
    workflow_pins = {}
    for count, relative in WORKFLOW_FILES.items():
        path = root / relative
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise PlanError(f"Не найден derived workflow: {path}") from exc
        workflow_pins[str(count)] = {
            "path": relative,
            "sha256": _sha_bytes(raw),
        }
    return {
        "profile_id": PROFILE_ID,
        "manifest": {
            "path": str(K1_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha_bytes(manifest_path.read_bytes()),
        },
        "models": model_pins,
        "workflows": workflow_pins,
    }


def _validate_token_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise PlanError(f"{name} должен быть списком")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PlanError(f"{name}: элементы должны быть непустыми строками")
        token = item.strip()
        if token not in INVARIANT_VOCAB:
            raise PlanError(f"{name}: неизвестный токен {token!r}")
        if token in result:
            raise PlanError(f"{name}: повторяющийся токен {token!r}")
        result.append(token)
    return result


def _validate_shot_fields(
    raw_role: Mapping[str, Any], *, frame_label: str, is_b: bool
) -> dict[str, Any]:
    shot_mode = raw_role.get("shot_mode")
    if shot_mode not in SHOT_MODES:
        raise PlanError(
            f"{frame_label}.shot_mode должен быть continue_shot или new_shot"
        )
    invariants = _validate_token_list(
        raw_role.get("invariants"), f"{frame_label}.invariants"
    )
    allowed = _validate_token_list(
        raw_role.get("allowed_changes"), f"{frame_label}.allowed_changes"
    )
    overlap = sorted(set(invariants) & set(allowed))
    if overlap:
        raise PlanError(
            f"{frame_label}: invariants и allowed_changes пересекаются: {overlap}"
        )
    if shot_mode == "continue_shot":
        if "camera" not in invariants:
            raise PlanError(
                f"{frame_label}: continue_shot требует camera в invariants"
            )
        if "camera" in allowed or "lens" in allowed:
            raise PlanError(
                f"{frame_label}: continue_shot не может разрешать camera/lens change"
            )
    if is_b and shot_mode != "continue_shot":
        raise PlanError(f"{frame_label}.b: ожидается shot_mode=continue_shot")
    if not is_b and shot_mode != "new_shot":
        raise PlanError(f"{frame_label}.a: ожидается shot_mode=new_shot")
    roles = raw_role.get("reference_roles")
    if not isinstance(roles, list) or not 1 <= len(roles) <= 3:
        raise PlanError(f"{frame_label}.reference_roles должен содержать 1..3 роли")
    role_list: list[str] = []
    for item in roles:
        if not isinstance(item, str) or not item.strip():
            raise PlanError(f"{frame_label}.reference_roles: пустая роль")
        role = item.strip()
        if role in role_list:
            raise PlanError(f"{frame_label}.reference_roles: повтор {role!r}")
        role_list.append(role)
    if is_b and "anchor" not in role_list:
        raise PlanError(f"{frame_label}.b: reference_roles обязан включать anchor")
    if "anchor" in role_list and role_list[0] != "anchor":
        raise PlanError(
            f"{frame_label}: anchor должен быть Picture 1 (порядок reference_roles)"
        )
    approval = raw_role.get("require_anchor_approval", False)
    if not isinstance(approval, bool):
        raise PlanError(f"{frame_label}.require_anchor_approval должен быть bool")
    negative = raw_role.get("negative")
    if negative is not None:
        negative = _safe_text(negative, f"{frame_label}.negative")
        if PLACEHOLDER_RE.search(negative):
            raise PlanError(f"{frame_label}.negative содержит placeholder")
    return {
        "shot_mode": shot_mode,
        "invariants": invariants,
        "allowed_changes": allowed,
        "reference_roles": role_list,
        "require_anchor_approval": approval,
        **({"negative": negative} if negative is not None else {}),
    }


def _validate_provided_frame(
    raw_role: Mapping[str, Any], *, frame_label: str
) -> dict[str, Any]:
    value = raw_role.get("provided_frame")
    if not isinstance(value, Mapping):
        raise PlanError(f"{frame_label}.provided_frame должен быть объектом")
    path = _safe_text(value.get("path"), f"{frame_label}.provided_frame.path")
    sha = value.get("sha256")
    if sha is not None:
        sha = _safe_text(sha, f"{frame_label}.provided_frame.sha256")
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha.lower()):
            raise PlanError(f"{frame_label}.provided_frame.sha256 должен быть 64 hex")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PlanError(f"{frame_label}.provided_frame.provenance должен быть объектом")
    for key in ("content_sha256",):
        item = provenance.get(key)
        if not isinstance(item, str) or len(item) != 64:
            raise PlanError(
                f"{frame_label}.provided_frame.provenance.{key} должен быть SHA256"
            )
    if sha and provenance.get("content_sha256") != sha.lower():
        raise PlanError(
            f"{frame_label}.provided_frame.sha256 и provenance.content_sha256 различаются"
        )
    return {
        "path": path,
        **({"sha256": sha.lower()} if sha else {}),
        "provenance": dict(provenance),
    }


def _validate_scenes(
    data: Mapping[str, Any], *, schema_version: int
) -> list[dict[str, Any]]:
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise PlanError("План должен содержать хотя бы одну сцену")
    result = []
    ids: set[str] = set()
    for index, raw in enumerate(scenes):
        if not isinstance(raw, Mapping):
            raise PlanError(f"scenes[{index}] должен быть объектом")
        scene_id = _safe_text(raw.get("id"), f"scenes[{index}].id")
        if not SAFE_ID.fullmatch(scene_id):
            raise PlanError(f"Недопустимый scene id: {scene_id!r}")
        if scene_id in ids:
            raise PlanError(f"Повторяющийся scene id: {scene_id}")
        ids.add(scene_id)
        scene = {
            "id": scene_id,
            "description": _safe_text(raw.get("description"), f"{scene_id}.description"),
            "a": {},
            "b": {},
        }
        for role in ("a", "b"):
            raw_role = raw.get(role)
            if not isinstance(raw_role, Mapping):
                raise PlanError(f"{scene_id}.{role} должен быть объектом")
            source = _safe_text(raw_role.get("source"), f"{scene_id}.{role}.source")
            allowed_sources = {
                "first_frame",
                "generate_from_previous_b",
                "generate_from_scene_a",
            }
            if schema_version == 2:
                allowed_sources.add("provided_frame")
            if source not in allowed_sources:
                raise PlanError(f"{scene_id}.{role}: неизвестный source {source!r}")
            frame_label = f"{scene_id}.{role}"
            frame = {"source": source}
            if source == "provided_frame":
                if schema_version != 2:
                    raise PlanError(f"{frame_label}: provided_frame доступен только в v2")
                if index == 0 and role == "a":
                    raise PlanError(f"{frame_label}: A1 задаётся через first_frame")
                frame["provided_frame"] = _validate_provided_frame(
                    raw_role, frame_label=frame_label
                )
                if "edit_instruction" in raw_role and raw_role["edit_instruction"] is not None:
                    frame["edit_instruction"] = _safe_text(
                        raw_role["edit_instruction"], f"{frame_label}.edit_instruction"
                    )
            else:
                if source in {"generate_from_previous_b", "generate_from_scene_a"}:
                    if schema_version == 2:
                        frame.update(
                            _validate_shot_fields(
                                raw_role,
                                frame_label=frame_label,
                                is_b=(role == "b"),
                            )
                        )
                edit = raw_role.get("edit_instruction")
                if edit is not None:
                    edit = _safe_text(edit, f"{frame_label}.edit_instruction")
                    if PLACEHOLDER_RE.search(edit):
                        raise PlanError(
                            f"{frame_label}: placeholder запрещён; нужен точный edit instruction"
                        )
                    frame["edit_instruction"] = edit
            scene[role] = frame
        if index == 0:
            if scene["a"]["source"] != "first_frame":
                raise PlanError("A первой сцены должен быть source=first_frame")
            if scene["b"]["source"] not in {"generate_from_scene_a", "provided_frame"}:
                raise PlanError(f"{scene_id}.b: ожидается generate_from_scene_a")
            if scene["b"]["source"] == "generate_from_scene_a" and not scene["b"].get(
                "edit_instruction"
            ):
                raise PlanError(f"{scene_id}.b: нужен edit_instruction")
        else:
            if scene["a"]["source"] not in {
                "generate_from_previous_b",
                "generate_from_scene_a",
                "provided_frame",
            }:
                raise PlanError(f"{scene_id}.a: ожидается generate_from_previous_b")
            if scene["a"]["source"] != "provided_frame" and not scene["a"].get(
                "edit_instruction"
            ):
                raise PlanError(f"{scene_id}.a: нужен новый camera/edit instruction")
            if scene["b"]["source"] not in {"generate_from_scene_a", "provided_frame"}:
                raise PlanError(f"{scene_id}.b: ожидается generate_from_scene_a")
            if scene["b"]["source"] == "generate_from_scene_a" and not scene["b"].get(
                "edit_instruction"
            ):
                raise PlanError(f"{scene_id}.b: нужен edit_instruction")
        result.append(scene)
    return result


def _resolve_v2_refs(
    root_path: Path, data: Mapping[str, Any]
) -> tuple[dict[str, RefInfo], dict[str, dict[str, Any]], str]:
    first = _resolve_ref(root_path, data.get("first_frame"), role="first_frame", required=True)
    if first is None:
        raise PlanError("first_frame обязателен")
    refs: dict[str, RefInfo] = {"first_frame": first}
    provenance: dict[str, dict[str, Any]] = {}

    def add_ref(ref_id: str, value: Any, role: str, *, required: bool, expected: str | None) -> None:
        info = _resolve_ref(
            root_path,
            value,
            role=ref_id,
            required=required,
            expected_sha256=expected,
        )
        if info is None:
            return
        # RefInfo.role carries the semantic role, not the plan id.
        info = type(info)(
            role=role,
            value=info.value,
            path=info.path,
            sha256=info.sha256,
            size_bytes=info.size_bytes,
            width=info.width,
            height=info.height,
            format=info.format,
        )
        if ref_id in refs:
            raise PlanError(f"Повторяющийся reference id: {ref_id}")
        if len(refs) - 1 >= MAX_USED_REFERENCES:
            raise PlanError(
                f"reference catalog превышает {MAX_USED_REFERENCES} used refs"
            )
        refs[ref_id] = info

    add_ref(
        "character_reference",
        data.get("character_reference"),
        "identity",
        required=False,
        expected=None,
    )
    add_ref(
        "background_reference",
        data.get("background_reference"),
        "environment",
        required=False,
        expected=None,
    )
    raw_refs = data.get("references") or []
    if not isinstance(raw_refs, list):
        raise PlanError("references должен быть списком")
    for index, raw in enumerate(raw_refs):
        if not isinstance(raw, Mapping):
            raise PlanError(f"references[{index}] должен быть объектом")
        ref_id = _safe_text(raw.get("id"), f"references[{index}].id")
        if ref_id == "anchor":
            raise PlanError("reference id 'anchor' зарезервирован для current A")
        if not SAFE_ID.fullmatch(ref_id):
            raise PlanError(f"Недопустимый reference id: {ref_id!r}")
        role = _safe_text(raw.get("role"), f"references[{index}].role")
        if role not in REFERENCE_ROLE_VOCAB:
            raise PlanError(
                f"references[{index}].role должен быть identity/style/environment"
            )
        expected = raw.get("sha256")
        if expected is not None:
            expected = _safe_text(expected, f"references[{index}].sha256")
        add_ref(
            ref_id,
            raw.get("path"),
            role,
            required=True,
            expected=expected,
        )
        if isinstance(raw.get("provenance"), Mapping):
            provenance[ref_id] = dict(raw["provenance"])
    subject = data.get("subject")
    label = "subject"
    if subject is not None:
        if not isinstance(subject, Mapping):
            raise PlanError("subject должен быть объектом")
        label = _safe_text(subject.get("label", "subject"), "subject.label")
    return refs, provenance, label


def _resolve_provided_frames(
    root_path: Path,
    scenes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        scene_id = scene["id"]
        for role in ("a", "b"):
            raw = scene[role]
            if raw.get("source") != "provided_frame":
                continue
            frame_id = f"{scene_id}/{role.upper()}"
            value = raw["provided_frame"]
            info = _resolve_ref(
                root_path,
                value["path"],
                role=f"provided:{frame_id}",
                required=True,
                expected_sha256=value.get("sha256"),
            )
            assert info is not None
            provenance = dict(value["provenance"])
            if provenance.get("content_sha256") != info.sha256:
                raise PlanError(
                    f"{frame_id}: provided_frame provenance.content_sha256 "
                    "не совпадает с actual file"
                )
            if not isinstance(provenance.get("frame_id"), str) or not provenance[
                "frame_id"
            ].strip():
                raise PlanError(
                    f"{frame_id}: provided_frame provenance.frame_id должен быть строкой"
                )
            for key in ("plan_sha256", "revision_id", "content_sha256"):
                if not isinstance(provenance.get(key), str) or len(provenance[key]) != 64:
                    raise PlanError(
                        f"{frame_id}: provided_frame.provenance.{key} должен быть SHA256"
                    )
            if not isinstance(provenance.get("version"), int) or provenance["version"] < 1:
                raise PlanError(
                    f"{frame_id}: provided_frame.provenance.version должен быть >=1"
                )
            result[frame_id] = {
                "path": str(info.path),
                "sha256": info.sha256,
                "size_bytes": info.size_bytes,
                "width": info.width,
                "height": info.height,
                "format": info.format,
                "value": info.value,
                "provenance": provenance,
            }
    return result


def _validate_reference_roles(
    frames: list[FrameSpec], refs: Mapping[str, RefInfo]
) -> None:
    for frame in frames:
        if not frame.generate:
            continue
        roles = list(frame.reference_roles)
        if not roles:
            raise PlanError(f"{frame.frame_id}: reference_roles пуст")
        if not 1 <= len(roles) <= 3:
            raise PlanError(f"{frame.frame_id}: нужно 1..3 reference")
        if "anchor" in roles and roles[0] != "anchor":
            raise PlanError(f"{frame.frame_id}: anchor должен быть Picture 1")
        if frame.role == "B" and "anchor" not in roles:
            raise PlanError(f"{frame.frame_id}: B требует anchor")
        counts = {"anchor": 0, "identity": 0, "environment": 0, "style": 0}
        identity_shas: dict[str, list[str]] = {}
        for role in roles:
            if role == "anchor":
                counts["anchor"] += 1
                continue
            if role not in refs:
                raise PlanError(f"{frame.frame_id}: неизвестный reference id {role!r}")
            semantic = refs[role].role
            if semantic not in REFERENCE_ROLE_VOCAB:
                raise PlanError(
                    f"{frame.frame_id}: reference {role!r} имеет неизвестную роль {semantic!r}"
                )
            counts[semantic] += 1
            if semantic == "identity":
                identity_shas.setdefault(refs[role].sha256, []).append(role)
        if counts["anchor"] > 1:
            raise PlanError(f"{frame.frame_id}: anchor допускается только один")
        if counts["identity"] > MAX_IDENTITY_REFS_PER_FRAME:
            raise PlanError(
                f"{frame.frame_id}: identity refs {counts['identity']} превышает "
                f"{MAX_IDENTITY_REFS_PER_FRAME}"
            )
        if counts["environment"] > MAX_ENVIRONMENT_REFS_PER_FRAME:
            raise PlanError(
                f"{frame.frame_id}: environment refs {counts['environment']} превышает "
                f"{MAX_ENVIRONMENT_REFS_PER_FRAME}"
            )
        if counts["style"] > MAX_STYLE_REFS_PER_FRAME:
            raise PlanError(
                f"{frame.frame_id}: style refs {counts['style']} превышает "
                f"{MAX_STYLE_REFS_PER_FRAME}"
            )
        duplicate_identity = [
            values for values in identity_shas.values() if len(values) > 1
        ]
        if duplicate_identity:
            raise PlanError(
                f"{frame.frame_id}: две identity refs указывают одинаковый content SHA256 "
                f"({', '.join(duplicate_identity[0])}); выберите разные изображения"
            )


def _frame_from_scene(
    *,
    scene: Mapping[str, Any],
    role: str,
    frame_id: str,
    generate: bool,
    source: str,
    anchor_frame: str | None,
    anchor_role: str | None,
    provided: Mapping[str, Any] | None,
) -> FrameSpec:
    raw = scene[role.lower()]
    provenance = dict(provided["provenance"]) if provided else None
    return FrameSpec(
        scene_id=str(scene["id"]),
        role=role,
        frame_id=frame_id,
        generate=generate,
        source=source,
        edit_instruction=raw.get("edit_instruction"),
        anchor_frame=anchor_frame,
        anchor_role=anchor_role,
        description=str(scene["description"]),
        shot_mode=raw.get("shot_mode"),
        invariants=tuple(raw.get("invariants") or ()),
        allowed_changes=tuple(raw.get("allowed_changes") or ()),
        reference_roles=tuple(raw.get("reference_roles") or ()),
        require_anchor_approval=bool(raw.get("require_anchor_approval", False)),
        negative_prompt=raw.get("negative"),
        provided_path=Path(str(provided["path"])) if provided else None,
        provided_sha256=str(provided["sha256"]) if provided else None,
        provided_provenance=provenance,
    )


def resolve_plan(path: Path | str, *, root: Path | str | None = None) -> ResolvedPlan:
    root_path = Path(root) if root is not None else ROOT
    plan_path = Path(path)
    if not plan_path.is_absolute():
        plan_path = root_path / plan_path
    data = _load_json(plan_path)
    schema = data.get("schema")
    if schema not in PLAN_SCHEMA_VERSIONS:
        raise PlanError("Неизвестная схема плана")
    schema_version = PLAN_SCHEMA_VERSIONS[str(schema)]
    if data.get("language") not in {None, "ru"}:
        raise PlanError("Пока поддерживается language=ru (описания сохраняются)")
    _safe_text(data.get("title"), "title", allow_empty=True)

    if schema_version == 1:
        first = _resolve_ref(root_path, data.get("first_frame"), role="first_frame", required=True)
        character = _resolve_ref(
            root_path, data.get("character_reference"), role="character_reference", required=False
        )
        background = _resolve_ref(
            root_path, data.get("background_reference"), role="background_reference", required=False
        )
        if first is None:
            raise PlanError("first_frame обязателен")
        refs = {"first_frame": first}
        if character is not None:
            refs["character_reference"] = character
        if background is not None:
            refs["background_reference"] = background
        subject_label = "subject"
        provided_frames: dict[str, dict[str, Any]] = {}
        reference_provenance: dict[str, dict[str, Any]] = {}
    else:
        refs, reference_provenance, subject_label = _resolve_v2_refs(root_path, data)
        first = refs["first_frame"]

    scenes = _validate_scenes(data, schema_version=schema_version)
    raw_policy = dict(data.get("output_policy") or {})
    for key in (
        "candidate_geometry",
        "video_geometry",
        "normalization",
        "overwrite_inputs",
        "automatic_video_submit",
        "first_frame_policy",
        "anchor_policy",
        "character_policy",
        "environment_policy",
        "video_policy",
    ):
        if key in data and key not in raw_policy:
            raw_policy[key] = data[key]
    output_policy = _normalize_output_policy(raw_policy)
    seed = _as_int(data.get("seed", DEFAULT_SEED), "seed", minimum=0, maximum=2_147_483_647)
    if data.get("seed_policy", "fixed") not in {"fixed", "per_frame"}:
        raise PlanError("seed_policy: fixed или per_frame")
    limit = _as_int(
        data.get("generation_limit", DEFAULT_GENERATION_LIMIT),
        "generation_limit",
        minimum=1,
        maximum=99,
    )
    pins = _source_pins(root_path)
    settings = {
        "seed": seed,
        "steps": DEFAULT_STEPS,
        "cfg": DEFAULT_CFG,
        "sampler_name": DEFAULT_SAMPLER,
        "scheduler": DEFAULT_SCHEDULER,
        "shift": DEFAULT_SHIFT,
        "denoise": 1.0,
        "cfg_norm_strength": 1.0,
    }
    provided_frames = (
        _resolve_provided_frames(root_path, scenes) if schema_version == 2 else {}
    )
    frames: list[FrameSpec] = []
    previous_b: str | None = None
    for index, scene in enumerate(scenes):
        scene_id = scene["id"]
        a_id = f"{scene_id}/A"
        if index == 0:
            frames.append(
                _frame_from_scene(
                    scene=scene,
                    role="A",
                    frame_id=a_id,
                    generate=False,
                    source="first_frame",
                    anchor_frame=None,
                    anchor_role=None,
                    provided=None,
                )
            )
            anchor = a_id
        else:
            source_a = scene["a"]["source"]
            provided_a = provided_frames.get(a_id)
            frames.append(
                _frame_from_scene(
                    scene=scene,
                    role="A",
                    frame_id=a_id,
                    generate=source_a != "provided_frame",
                    source=source_a,
                    anchor_frame=previous_b,
                    anchor_role="previous_b",
                    provided=provided_a,
                )
            )
            anchor = a_id
        b_id = f"{scene_id}/B"
        source_b = scene["b"]["source"]
        provided_b = provided_frames.get(b_id)
        frames.append(
            _frame_from_scene(
                scene=scene,
                role="B",
                frame_id=b_id,
                generate=source_b != "provided_frame",
                source=source_b,
                anchor_frame=anchor,
                anchor_role="current_a",
                provided=provided_b,
            )
        )
        previous_b = b_id
    if schema_version == 2:
        _validate_reference_roles(frames, refs)

    if schema_version == 1:
        canonical = {
            "schema": PLAN_SCHEMA,
            "title": data.get("title"),
            "language": "ru",
            "refs": {
                key: {
                    "value": info.value,
                    "sha256": info.sha256,
                    "size_bytes": info.size_bytes,
                    "width": info.width,
                    "height": info.height,
                    "format": info.format,
                }
                for key, info in refs.items()
            },
            "scenes": scenes,
            "output_policy": output_policy,
            "settings": settings,
            "generation_limit": limit,
            "source_pins": pins,
        }
    else:
        canonical = {
            "schema": PLAN_SCHEMA_V2,
            "title": data.get("title"),
            "language": "ru",
            "subject_label": subject_label,
            "style": _safe_text(data.get("style"), "style", allow_empty=True),
            "refs": {
                key: {
                    "role": info.role,
                    "value": info.value,
                    "sha256": info.sha256,
                    "size_bytes": info.size_bytes,
                    "width": info.width,
                    "height": info.height,
                    "format": info.format,
                }
                for key, info in refs.items()
            },
            "provided_frames": provided_frames,
            "reference_provenance": reference_provenance,
            "scenes": scenes,
            "output_policy": output_policy,
            "settings": settings,
            "generation_limit": limit,
            "source_pins": pins,
        }
    plan_sha = _sha_bytes(_canonical(canonical))
    resolved_data = dict(data)
    resolved_data["schema"] = schema
    resolved_data["scenes"] = scenes
    resolved_data["generation_limit"] = limit
    resolved_data["seed"] = seed
    resolved_data["output_policy"] = output_policy
    if schema_version == 2:
        resolved_data["subject_label"] = subject_label
    return ResolvedPlan(
        path=plan_path,
        data=resolved_data,
        plan_sha256=plan_sha,
        refs=refs,
        output_policy=output_policy,
        frames=tuple(frames),
        pins=pins,
        settings=settings,
        schema_version=schema_version,
        provided_frames=provided_frames,
        reference_provenance=reference_provenance,
        subject_label=subject_label,
    )


def _human_ref(info: RefInfo) -> str:
    return (
        f"{info.role}: {info.path} "
        f"({info.width}x{info.height}, {info.format}, {info.size_bytes} bytes) "
        f"sha256={info.sha256}"
    )


def preview_lines(plan: ResolvedPlan) -> list[str]:
    from .normalize import inspect_image

    lines = [
        f"План: {plan.title}",
        f"Plan SHA256: {plan.plan_sha256}",
        f"Сцен: {plan.scene_count()}; генерируемых кадров: {len(plan.generated_frames)}",
    ]
    for info in plan.refs.values():
        lines.append("  " + _human_ref(info))
        if info.role == "character_reference":
            lines.append(
                "  [!] character reference — это лист/несколько ракурсов; "
                "качество и перенос layout не гарантированы, нужна визуальная приёмка."
            )
    lines.append(
        "A1: provided first_frame сохраняется байт-в-байт; исходный файл не перезаписывается."
    )
    lines.append(
        "Нормализация: "
        f"anchor={plan.output_policy['anchor_policy']}, "
        f"character={plan.output_policy['character_policy']}, "
        f"environment={plan.output_policy['environment_policy']}, "
        f"candidate={plan.output_policy['candidate_geometry']}, "
        f"video={plan.output_policy['video_geometry']}"
    )
    for frame in plan.frames:
        if frame.generate:
            edit = frame.edit_instruction or ""
            if len(edit) > 80:
                edit = edit[:80] + "..."
            details = f"  {frame.frame_id}: generate anchor={frame.anchor_frame}"
            if plan.schema_version == 2:
                details += (
                    f"; shot_mode={frame.shot_mode}; refs={list(frame.reference_roles)}"
                    f"; invariants={list(frame.invariants)}"
                    f"; allowed={list(frame.allowed_changes)}"
                )
                if frame.require_anchor_approval:
                    details += "; approval=required"
            details += f"; edit={edit}"
            lines.append(details)
        elif frame.source == "provided_frame":
            lines.append(
                f"  {frame.frame_id}: preserve provided_frame "
                f"sha256={frame.provided_sha256} (no POST)"
            )
        else:
            lines.append(f"  {frame.frame_id}: preserve provided first_frame (no POST)")
    lines.append(
        f"Лимит генераций: {plan.generation_limit}; ожидаемые POST: "
        f"{len(plan.generated_frames)} (без retry)."
    )
    lines.append("Автоматическая отправка видео: запрещена. LLM/imagegen: не используются.")
    return lines


def plan_to_json(plan: ResolvedPlan) -> dict[str, Any]:
    return {
        "schema": "anime-studio/kf1-plan-resolved/v1",
        "path": str(plan.path),
        "plan_sha256": plan.plan_sha256,
        "title": plan.title,
        "scenes": plan.data["scenes"],
        "refs": {
            key: {
                "path": str(info.path),
                "sha256": info.sha256,
                "size_bytes": info.size_bytes,
                "width": info.width,
                "height": info.height,
                "format": info.format,
            }
            for key, info in plan.refs.items()
        },
        "output_policy": plan.output_policy,
        "settings": plan.settings,
        "generation_limit": plan.generation_limit,
        "schema_version": plan.schema_version,
        "subject_label": plan.subject_label,
        "reference_provenance": plan.reference_provenance,
        "provided_frames": plan.provided_frames,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "scene_id": frame.scene_id,
                "role": frame.role,
                "generate": frame.generate,
                "source": frame.source,
                "anchor_frame": frame.anchor_frame,
                "shot_mode": frame.shot_mode,
                "reference_roles": list(frame.reference_roles),
                "invariants": list(frame.invariants),
                "allowed_changes": list(frame.allowed_changes),
                "require_anchor_approval": frame.require_anchor_approval,
                "negative_prompt": frame.negative_prompt,
                "provided_path": str(frame.provided_path) if frame.provided_path else None,
                "provided_sha256": frame.provided_sha256,
            }
            for frame in plan.frames
        ],
        "source_pins": plan.pins,
    }


def load_example() -> dict[str, Any]:
    return _load_json(EXAMPLE_PLAN)


__all__ = [
    "DEFAULT_CFG",
    "DEFAULT_GENERATION_LIMIT",
    "DEFAULT_SAMPLER",
    "DEFAULT_SCHEDULER",
    "DEFAULT_SEED",
    "DEFAULT_SHIFT",
    "DEFAULT_STEPS",
    "EXAMPLE_PLAN",
    "EXAMPLE_SCHEMA",
    "FrameSpec",
    "INVARIANT_VOCAB",
    "K1_MANIFEST",
    "MAX_ENVIRONMENT_REFS_PER_FRAME",
    "MAX_IDENTITY_REFS_PER_FRAME",
    "MAX_STYLE_REFS_PER_FRAME",
    "MAX_USED_REFERENCES",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_V2",
    "PLAN_SCHEMAS",
    "PROFILE_ID",
    "REFERENCE_ROLE_VOCAB",
    "SHOT_MODES",
    "RefInfo",
    "ResolvedPlan",
    "WORKFLOW_FILES",
    "interactive_plan",
    "load_example",
    "parse_text_plan",
    "plan_to_json",
    "preview_lines",
    "resolve_plan",
]
