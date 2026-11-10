"""Orientation-safe, never-stretching image normalization for KF1.

Inputs are never modified.  Every derived file is written atomically to a new
path.  PNG output is validated by decoding it again after the atomic replace.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:  # optional dependency for JPEG/WebP/orientation; PNG fallback is stdlib
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:  # pragma: no cover - exercised in the narrow venv
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    UnidentifiedImageError = OSError  # type: ignore[assignment,misc]

from . import png as png_mod

if Image is not None:
    Image.MAX_IMAGE_PIXELS = 50_000_000
from .errors import NormalizationError


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def inspect_image(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise NormalizationError(f"Изображение не является обычным файлом: {target}")
    if Image is None:
        try:
            decoded = png_mod.read_png(target.read_bytes())
        except (OSError, png_mod.PNGError) as exc:
            raise NormalizationError(f"Не удалось декодировать PNG: {target}") from exc
        return {
            "width": decoded.width,
            "height": decoded.height,
            "format": "PNG",
            "mode": "RGBA",
            "orientation": 1,
            "size_bytes": target.stat().st_size,
            "sha256": _sha_bytes(target.read_bytes()),
        }
    try:
        with Image.open(target) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > 50_000_000:
                raise NormalizationError(f"Размер изображения вне лимита: {target}")
            image.load()
            fmt = str(image.format or "").upper()
            mode = str(image.mode)
            orientation = int(image.getexif().get(274, 1) or 1)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise NormalizationError(f"Не удалось декодировать изображение: {target}") from exc
    if width <= 0 or height <= 0:
        raise NormalizationError(f"Некорректные размеры изображения: {target}")
    return {
        "width": width,
        "height": height,
        "format": fmt,
        "mode": mode,
        "orientation": orientation,
        "size_bytes": target.stat().st_size,
        "sha256": _sha_bytes(target.read_bytes()),
    }


def validate_png_bytes(data: bytes, *, max_bytes: int = 100 * 1024 * 1024) -> dict[str, Any]:
    if not data or len(data) > max_bytes:
        raise NormalizationError("PNG пуст или превышает допустимый размер")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise NormalizationError("Ответ не является PNG")
    if Image is None:
        try:
            decoded = png_mod.read_png(data)
        except png_mod.PNGError as exc:
            raise NormalizationError("PNG не декодируется") from exc
        return {
            "width": decoded.width,
            "height": decoded.height,
            "format": "PNG",
            "mode": "RGBA",
            "orientation": 1,
            "size_bytes": len(data),
            "sha256": _sha_bytes(data),
        }
    temp = None
    try:
        handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False)
        temp = Path(handle.name)
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        with Image.open(temp) as image:
            image.load()
            width, height = image.size
            fmt = str(image.format or "").upper()
            mode = str(image.mode)
            orientation = int(image.getexif().get(274, 1) or 1)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise NormalizationError("PNG не декодируется") from exc
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    if fmt != "PNG":
        raise NormalizationError("Декодированный файл не PNG")
    return {
        "width": width,
        "height": height,
        "format": fmt,
        "mode": mode,
        "orientation": orientation,
        "size_bytes": len(data),
        "sha256": _sha_bytes(data),
    }


def _target_box(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise NormalizationError("target geometry должен быть [width, height]")
    return int(value[0]), int(value[1])


def normalize_image(
    src: Path | str,
    dst: Path | str,
    *,
    policy: str,
    target: Any,
    pad: bool = False,
    label: str = "image",
) -> dict[str, Any]:
    """Create a derived image without touching ``src``.

    ``policy``:
    - ``preserve``: exact byte copy unless EXIF orientation requires transposing.
    - ``contain``: proportional fit inside target, optional white padding.
    - ``crop``: proportional cover + centered crop (never stretch).
    """

    source = Path(src)
    destination = Path(dst)
    if source.resolve() == destination.resolve():
        raise NormalizationError(f"{label}: нельзя писать поверх исходника")
    if policy not in {"preserve", "contain", "crop"}:
        raise NormalizationError(f"{label}: неизвестная политика {policy!r}")
    inspect = inspect_image(source)
    orientation = int(inspect["orientation"])
    if policy == "preserve" and orientation in {0, 1}:
        data = source.read_bytes()
        _atomic_write_bytes(destination, data)
        out_info = inspect_image(destination)
        return _record(source, destination, policy, inspect, out_info, pad=False)
    if policy == "preserve" and orientation not in {0, 1}:
        policy = "contain"
        target = (inspect["width"], inspect["height"])
        pad = False

    width, height = _target_box(target)
    if Image is None:
        try:
            decoded = png_mod.read_png(source.read_bytes())
            if policy == "contain":
                data = png_mod.contain(decoded, width, height, pad=pad)
            elif policy == "crop":
                data = png_mod.crop(decoded, width, height)
            else:
                raise NormalizationError("unreachable policy")
            _atomic_write_bytes(destination, data)
            validate_png_bytes(destination.read_bytes())
        except png_mod.PNGError as exc:
            raise NormalizationError(f"{label}: PNG fallback failed: {exc}") from exc
        out_info = inspect_image(destination)
        return _record(source, destination, policy, inspect, out_info, pad=pad)
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            if policy == "contain":
                fitted = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
                if pad:
                    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
                    left = (width - fitted.width) // 2
                    top = (height - fitted.height) // 2
                    canvas.alpha_composite(fitted, (left, top))
                    fitted = canvas
            elif policy == "crop":
                fitted = ImageOps.fit(
                    image,
                    (width, height),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
            else:  # pragma: no cover - guarded above
                raise NormalizationError("unreachable policy")
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent,
                prefix=destination.name + ".", suffix=".png.tmp", delete=False
            )
            temp = Path(handle.name)
            try:
                with handle:
                    fitted.save(handle, format="PNG")
                    handle.flush()
                    os.fsync(handle.fileno())
                # Re-decode the exact bytes before publishing.
                validate_png_bytes(temp.read_bytes())
                os.replace(temp, destination)
            finally:
                temp.unlink(missing_ok=True)
    except NormalizationError:
        raise
    except (OSError, ValueError) as exc:
        raise NormalizationError(f"{label}: нормализация не удалась: {exc}") from exc
    out_info = inspect_image(destination)
    return _record(source, destination, policy, inspect, out_info, pad=pad)


def _record(
    source: Path,
    destination: Path,
    policy: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    pad: bool,
) -> dict[str, Any]:
    return {
        "schema": "anime-studio/kf1-normalization/v1",
        "policy": policy,
        "pad": bool(pad),
        "source": {
            "path": str(source),
            "sha256": before["sha256"],
            "size_bytes": before["size_bytes"],
            "width": before["width"],
            "height": before["height"],
            "format": before["format"],
            "orientation": before["orientation"],
        },
        "output": {
            "path": str(destination),
            "sha256": after["sha256"],
            "size_bytes": after["size_bytes"],
            "width": after["width"],
            "height": after["height"],
            "format": after["format"],
            "orientation": after["orientation"],
        },
        "stretch": False,
    }


def make_thumbnail(src: Path | str, dst: Path | str, *, max_size: int = 320) -> dict[str, Any]:
    source = Path(src)
    destination = Path(dst)
    if source.resolve() == destination.resolve():
        raise NormalizationError("thumbnail: нельзя писать поверх исходника")
    if Image is None:
        try:
            decoded = png_mod.read_png(source.read_bytes())
            data = png_mod.thumbnail(decoded, max_size)
            _atomic_write_bytes(destination, data)
            validate_png_bytes(destination.read_bytes())
        except png_mod.PNGError as exc:
            raise NormalizationError(f"thumbnail fallback failed: {exc}") from exc
        return {"path": str(destination), **inspect_image(destination)}
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=destination.name + ".", suffix=".tmp",
                delete=False,
            )
            temp = Path(handle.name)
            try:
                with handle:
                    image.save(handle, format="PNG")
                    handle.flush()
                    os.fsync(handle.fileno())
                validate_png_bytes(temp.read_bytes())
                os.replace(temp, destination)
            finally:
                temp.unlink(missing_ok=True)
    except NormalizationError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise NormalizationError(f"thumbnail не создан: {exc}") from exc
    info = inspect_image(destination)
    return {"path": str(destination), **info}


__all__ = [
    "inspect_image",
    "make_thumbnail",
    "normalize_image",
    "validate_png_bytes",
]

