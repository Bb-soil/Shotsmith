"""Immutable, content-addressed KF2 project assets."""

from __future__ import annotations

import io
import json
import os
import struct
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.kf1.normalize import validate_png_bytes

from .errors import AssetError
from .paths import (
    MAX_ASSETS,
    MAX_IMAGE_BYTES,
    MAX_PROJECT_BYTES,
    MAX_SCRIPT_BYTES,
    assert_contained,
    atomic_write_bytes,
    checked_exists,
    ensure_managed_dir,
    is_reparse,
    read_bytes_limited,
    resolve_local_source,
    sha256_bytes,
)

try:  # Pillow is optional in the narrow venv; PNG fallback must keep working.
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - exercised in the narrow venv
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = OSError  # type: ignore[assignment,misc]

if Image is not None:
    Image.MAX_IMAGE_PIXELS = 50_000_000

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_ROLES = {"character", "creature", "object", "environment", "style", "provided_keyframe"}
SUBJECT_KINDS = {"character", "creature", "object", "environment"}
FORMAT_EXTENSION = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
ALLOWED_FORMATS = set(FORMAT_EXTENSION)
MAX_PIXELS = 50_000_000


def _looks_like_png(data: bytes) -> bool:
    return data.startswith(b"\x89PNG\r\n\x1a\n")


def _png_dimensions(data: bytes, *, label: str) -> tuple[int, int, int, int]:
    """Parse bounded PNG IHDR before any decoder/allocation."""

    if len(data) < 29 or not _looks_like_png(data):
        raise AssetError(f"{label}: не PNG header")
    length = struct.unpack(">I", data[8:12])[0]
    if data[12:16] != b"IHDR" or length != 13:
        raise AssetError(f"{label}: некорректный PNG IHDR")
    width, height = struct.unpack(">II", data[16:24])
    bit_depth, color_type = data[24], data[25]
    if width <= 0 or height <= 0:
        raise AssetError(f"{label}: некорректные PNG dimensions")
    if width * height > MAX_PIXELS:
        raise AssetError(f"{label}: decoded size превышает {MAX_PIXELS} pixels")
    if bit_depth not in {1, 2, 4, 8, 16}:
        raise AssetError(f"{label}: некорректная PNG bit depth")
    if color_type not in {0, 2, 3, 4, 6}:
        raise AssetError(f"{label}: некорректный PNG color type")
    return width, height, bit_depth, color_type


def _decode_png(data: bytes, *, label: str) -> dict[str, Any]:
    width, height, _, _ = _png_dimensions(data, label=label)
    if Image is not None:
        try:
            with Image.open(io.BytesIO(data)) as image:
                if bool(getattr(image, "is_animated", False)) or int(
                    getattr(image, "n_frames", 1)
                ) != 1:
                    raise AssetError(f"{label}: animated/multiframe PNG не поддерживается")
                if image.size != (width, height):
                    raise AssetError(f"{label}: PNG header/decoder size mismatch")
        except AssetError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise AssetError(f"{label}: PNG не декодируется") from exc
    try:
        info = validate_png_bytes(data)
    except Exception as exc:
        raise AssetError(f"{label}: PNG не декодируется") from exc
    if int(info["width"]) * int(info["height"]) > MAX_PIXELS:
        raise AssetError(f"{label}: decoded size превышает {MAX_PIXELS} pixels")
    return {
        "format": "PNG",
        "width": int(info["width"]),
        "height": int(info["height"]),
        "orientation": int(info.get("orientation", 1) or 1),
        "mode": str(info.get("mode", "")),
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }


def decode_image_bytes(data: bytes, *, label: str = "image") -> dict[str, Any]:
    """Decode actual bytes; extension is never treated as proof of format."""

    if not data:
        raise AssetError(f"{label}: пустой файл")
    if _looks_like_png(data):
        return _decode_png(data, label=label)
    if Image is None:
        raise AssetError(
            f"{label}: JPEG/WebP требуют Pillow; в этом окружении доступен только PNG"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getattr(Image, "DecompressionBombWarning", Warning))
            with Image.open(io.BytesIO(data)) as image:
                fmt = str(image.format or "").upper()
                if fmt not in ALLOWED_FORMATS:
                    raise AssetError(f"{label}: неподдерживаемый формат {fmt or '?'}")
                if bool(getattr(image, "is_animated", False)) or int(
                    getattr(image, "n_frames", 1)
                ) != 1:
                    raise AssetError(
                        f"{label}: animated/multiframe изображения не поддерживаются"
                    )
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
                    raise AssetError(
                        f"{label}: decoded size {width}x{height} превышает {MAX_PIXELS} pixels"
                    )
                # Dimensions are checked before decompression/allocation.
                image.load()
                orientation = int(image.getexif().get(274, 1) or 1)
                mode = str(image.mode)
    except AssetError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise AssetError(f"{label}: изображение не декодируется") from exc
    except Exception as exc:
        if type(exc).__name__ in {"DecompressionBombError", "DecompressionBombWarning"}:
            raise AssetError(f"{label}: decompression bomb rejected") from exc
        raise AssetError(f"{label}: изображение не декодируется") from exc
    return {
        "format": fmt,
        "width": width,
        "height": height,
        "orientation": orientation,
        "mode": mode,
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }


def _asset_path(project_dir: Path, relative: str) -> Path:
    path = project_dir / relative
    assert_contained(project_dir, path)
    return path


def _existing_asset(
    project_dir: Path, relative: str, *, expected_sha: str, expected_size: int
) -> bool:
    path = _asset_path(project_dir, relative)
    if not checked_exists(path, label="project asset"):
        return False
    data = read_bytes_limited(path, max_bytes=MAX_IMAGE_BYTES, label="project asset")
    if len(data) != expected_size or sha256_bytes(data) != expected_sha:
        raise AssetError(f"project asset повреждён или подменён: {path}")
    return True


def _used_bytes(library: Iterable[Mapping[str, Any]], scripts: Iterable[Mapping[str, Any]]) -> int:
    return sum(int(item.get("size_bytes") or 0) for item in library) + sum(
        int(item.get("size_bytes") or 0) for item in scripts
    )


def _association_id(
    sha: str, role: str, subject_id: str | None, view: str | None
) -> str:
    association = f"{sha}|{role}|{subject_id or ''}|{view or ''}"
    return "asset-" + sha[:12] + "-" + sha256_bytes(association.encode("utf-8"))[:8]


def _asset_id_for(
    sha: str,
    role: str,
    subject_id: str | None,
    view: str | None,
    library: Iterable[Mapping[str, Any]],
) -> str:
    short = _association_id(sha, role, subject_id, view)
    for item in library:
        if item.get("id") == short and (
            item.get("sha256") != sha
            or item.get("role") != role
            or item.get("subject_id") != subject_id
            or item.get("view") != view
        ):
            association = f"{sha}|{role}|{subject_id or ''}|{view or ''}"
            return "asset-" + sha + "-" + sha256_bytes(association.encode("utf-8"))[:8]
    return short


def import_image(
    project_dir: Path,
    source_value: str,
    *,
    role: str,
    subject_id: str | None = None,
    view: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    library: Iterable[Mapping[str, Any]] = (),
    scripts: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Copy one local image into the project without modifying the source."""

    if role not in IMAGE_ROLES:
        raise AssetError(f"неизвестная роль изображения: {role!r}")
    if role in {"character", "creature"} and not subject_id:
        raise AssetError(f"роль {role} требует явный --subject")
    if role in {"environment", "style", "provided_keyframe"} and subject_id:
        raise AssetError(f"роль {role} не принимает --subject")
    source = resolve_local_source(
        source_value,
        label="image",
        allowed_suffixes=ALLOWED_IMAGE_EXTENSIONS,
    )
    data = read_bytes_limited(source, max_bytes=MAX_IMAGE_BYTES, label="image")
    info = decode_image_bytes(data, label=str(source))
    fmt = info["format"]
    if fmt not in ALLOWED_FORMATS:
        raise AssetError(f"image format не поддержан: {fmt}")
    sha = str(info["sha256"])
    ext = FORMAT_EXTENSION[fmt]
    library_list = list(library)
    scripts_list = list(scripts)
    if len(library_list) >= MAX_ASSETS:
        raise AssetError(f"project library ограничен {MAX_ASSETS} assets")
    total = _used_bytes(library_list, scripts_list)
    if total + len(data) > MAX_PROJECT_BYTES:
        raise AssetError(f"project storage limit {MAX_PROJECT_BYTES} bytes exceeded")
    relative = f"assets/{sha}{ext}"
    target = _asset_path(project_dir, relative)
    ensure_managed_dir(project_dir, target.parent, label="assets directory")
    reused = _existing_asset(
        project_dir, relative, expected_sha=sha, expected_size=len(data)
    )
    if not reused:
        atomic_write_bytes(target, data, exclusive=True)
        if is_reparse(target):
            raise AssetError(f"project asset unexpectedly became reparse: {target}")
        if sha256_bytes(target.read_bytes()) != sha:
            raise AssetError(f"project asset hash mismatch after copy: {target}")
    prov = dict(provenance or {})
    content_sha = prov.get("content_sha256")
    if content_sha is not None and str(content_sha).lower() != sha:
        raise AssetError("provenance.content_sha256 не совпадает с actual image bytes")
    prov["content_sha256"] = sha
    return {
        "id": _asset_id_for(sha, role, subject_id, view, library_list),
        "original_name": source.name,
        "source_path": str(source),
        "stored_path": relative,
        "sha256": sha,
        "size_bytes": len(data),
        "width": int(info["width"]),
        "height": int(info["height"]),
        "format": fmt,
        "orientation": int(info.get("orientation", 1) or 1),
        "role": role,
        "subject_id": subject_id,
        "view": view,
        "provenance": prov,
        "reused": reused,
    }


def import_script(
    project_dir: Path,
    source_value: str,
    *,
    library: Iterable[Mapping[str, Any]] = (),
    scripts: Iterable[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], str]:
    """Preserve a .txt script byte-for-byte and return decoded UTF-8 text."""

    source = resolve_local_source(source_value, label="script", allowed_suffixes={".txt"})
    data = read_bytes_limited(source, max_bytes=MAX_SCRIPT_BYTES, label="script")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AssetError("script должен быть UTF-8 (BOM допускается)") from exc
    sha = sha256_bytes(data)
    library_list = list(library)
    scripts_list = list(scripts)
    if len(library_list) + len(scripts_list) >= MAX_ASSETS:
        raise AssetError(f"project storage limited to {MAX_ASSETS} assets/scripts")
    if _used_bytes(library_list, scripts_list) + len(data) > MAX_PROJECT_BYTES:
        raise AssetError(f"project storage limit {MAX_PROJECT_BYTES} bytes exceeded")
    relative = f"assets/scripts/{sha}.txt"
    target = _asset_path(project_dir, relative)
    ensure_managed_dir(project_dir, target.parent, label="scripts directory")
    if checked_exists(target, label="script asset"):
        existing = read_bytes_limited(target, max_bytes=MAX_SCRIPT_BYTES, label="script asset")
        if sha256_bytes(existing) != sha or existing != data:
            raise AssetError(f"script asset collision/tamper: {target}")
        reused = True
    else:
        atomic_write_bytes(target, data, exclusive=True)
        reused = False
    return (
        {
            "id": "script-" + sha[:16],
            "original_name": source.name,
            "source_path": str(source),
            "stored_path": relative,
            "sha256": sha,
            "size_bytes": len(data),
            "imported_utc": "",
            "reused": reused,
        },
        text,
    )


def read_provenance_file(path_value: str) -> dict[str, Any]:
    path = resolve_local_source(path_value, label="provenance", allowed_suffixes={".json"})
    data = read_bytes_limited(path, max_bytes=1024 * 1024, label="provenance")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetError("provenance JSON не читается") from exc
    if not isinstance(value, dict):
        raise AssetError("provenance JSON должен быть объектом")
    for key in ("content_sha256", "plan_sha256", "revision_id"):
        item = value.get(key)
        if item is not None and (
            not isinstance(item, str)
            or len(item) != 64
            or any(char not in "0123456789abcdef" for char in item.lower())
        ):
            raise AssetError(f"provenance.{key} должен быть 64 hex")
    return value


def validate_library_asset(project_dir: Path, asset: Mapping[str, Any]) -> dict[str, Any]:
    relative = asset.get("stored_path")
    if not isinstance(relative, str) or not relative.startswith("assets/"):
        raise AssetError("asset.stored_path должен начинаться с assets/")
    path = _asset_path(project_dir, relative)
    data = read_bytes_limited(path, max_bytes=MAX_IMAGE_BYTES, label="project asset")
    actual_sha = sha256_bytes(data)
    if actual_sha != asset.get("sha256"):
        raise AssetError(f"asset SHA256 mismatch: {relative}")
    if len(data) != int(asset.get("size_bytes") or -1):
        raise AssetError(f"asset size mismatch: {relative}")
    info = decode_image_bytes(data, label=relative)
    if info["format"] != asset.get("format"):
        raise AssetError(f"asset format mismatch: {relative}")
    if info["width"] != int(asset.get("width") or -1) or info["height"] != int(
        asset.get("height") or -1
    ):
        raise AssetError(f"asset decoded geometry mismatch: {relative}")
    return {**dict(asset), "actual_path": str(path), **info}


ASSET_REFERENCE_SEMANTIC = {
    "character": "identity",
    "creature": "identity",
    "object": "identity",
    "environment": "environment",
    "style": "style",
}


def asset_usage(project: Mapping[str, Any], asset_id: str) -> dict[str, int]:
    """Count authored uses without changing any frame or asset."""

    result = {
        "frame_sources": 0,
        "references": 0,
        "subject_views": 0,
        "a1_source": 0,
    }
    scenes = project.get("scenes") or []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            continue
        for key in ("initial", "final"):
            frame = scene.get(key)
            if not isinstance(frame, Mapping):
                continue
            if frame.get("asset_id") == asset_id:
                result["frame_sources"] += 1
                if index == 0 and key == "initial":
                    result["a1_source"] += 1
            for ref in frame.get("references") or []:
                if isinstance(ref, Mapping) and ref.get("asset_id") == asset_id:
                    result["references"] += 1
    for subject in project.get("subjects") or []:
        if not isinstance(subject, Mapping):
            continue
        for view in subject.get("views") or []:
            if isinstance(view, Mapping) and view.get("asset_id") == asset_id:
                result["subject_views"] += 1
    return result


def change_asset_role(
    project: dict[str, Any],
    asset_id: str,
    *,
    role: str,
    subject_id: str | None = None,
    view: str | None = None,
) -> dict[str, Any]:
    """Change association metadata while preserving the stable asset id/bytes.

    Frame assignments and references keep their asset id. Reference semantic
    overrides are updated only when the new role still has a reference semantic.
    A provided_keyframe keeps existing frame/reference assignments for the caller
    to resolve explicitly; nothing is silently unset or dropped.
    """

    if role not in IMAGE_ROLES:
        raise AssetError(f"неизвестная роль изображения: {role!r}")
    asset = next(
        (item for item in project.get("library") or [] if item.get("id") == asset_id),
        None,
    )
    if not isinstance(asset, dict):
        raise AssetError(f"asset не найден: {asset_id}")
    clean_subject = subject_id or None
    clean_view = view or None
    if role in {"character", "creature"} and not clean_subject:
        raise AssetError(f"роль {role} требует существующий subject")
    if role in {"environment", "style", "provided_keyframe"} and clean_subject:
        raise AssetError(f"роль {role} не принимает subject")
    target_subject = None
    if clean_subject:
        target_subject = next(
            (
                item
                for item in project.get("subjects") or []
                if isinstance(item, Mapping) and item.get("id") == clean_subject
            ),
            None,
        )
        if target_subject is None:
            raise AssetError(f"subject не существует: {clean_subject}")
        if target_subject.get("kind") != role:
            raise AssetError(
                f"subject {clean_subject} kind={target_subject.get('kind')} "
                f"не совпадает с role={role}"
            )
    if clean_view is not None and not isinstance(clean_view, str):
        raise AssetError("view должен быть строкой или None")

    asset["role"] = role
    asset["subject_id"] = clean_subject
    asset["view"] = clean_view
    for subject in project.get("subjects") or []:
        if not isinstance(subject, dict):
            continue
        views = subject.get("views")
        if isinstance(views, list):
            subject["views"] = [
                item
                for item in views
                if not isinstance(item, Mapping) or item.get("asset_id") != asset_id
            ]
    if target_subject is not None and clean_view:
        target_subject.setdefault("views", []).append(
            {"asset_id": asset_id, "label": clean_view}
        )

    semantic = ASSET_REFERENCE_SEMANTIC.get(role)
    if semantic is not None:
        for scene in project.get("scenes") or []:
            if not isinstance(scene, Mapping):
                continue
            for key in ("initial", "final"):
                frame = scene.get(key)
                if not isinstance(frame, Mapping):
                    continue
                for ref in frame.get("references") or []:
                    if isinstance(ref, dict) and ref.get("asset_id") == asset_id:
                        ref["role"] = semantic
    return {
        "asset_id": asset_id,
        "role": role,
        "subject_id": clean_subject,
        "view": clean_view,
        "usage": asset_usage(project, asset_id),
    }


__all__ = [
    "ALLOWED_FORMATS",
    "ALLOWED_IMAGE_EXTENSIONS",
    "ASSET_REFERENCE_SEMANTIC",
    "IMAGE_ROLES",
    "MAX_PIXELS",
    "SUBJECT_KINDS",
    "asset_usage",
    "change_asset_role",
    "decode_image_bytes",
    "import_image",
    "import_script",
    "read_provenance_file",
    "validate_library_asset",
]
