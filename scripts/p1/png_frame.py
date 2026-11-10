"""Bounded, dependency-free PNG8 RGB frame preparation for P1 v3.

This is deliberately not a general image library. It accepts only the exact
subset needed by the checked-in v3 start frame:

- PNG signature, IHDR first, one or more IDAT chunks, IEND last;
- bit depth 8, color type 2 (RGB), no interlace;
- standard PNG row filters 0..4;
- dimensions bounded by ``MAX_DIMENSION``;
- a deterministic integer area-average resize.

Unsupported PNG variants fail closed with ``PngFrameError`` instead of being
silently approximated. No network, environment or third-party package is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any, Mapping

MAX_DIMENSION = 8192
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_DECLARED_PIXELS = MAX_DIMENSION * MAX_DIMENSION
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
POLICY_SCHEMA = "anime-video-studio/p1-v3-frame-policy/v1"


class PngFrameError(Exception):
    """The input is not the supported PNG8 RGB subset or a policy is invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PngFrameError(f"cannot stat PNG: {exc}") from exc
    if size <= 0 or size > MAX_FILE_BYTES:
        raise PngFrameError("PNG size is outside the bounded range")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PngFrameError(f"cannot read PNG: {exc}") from exc


def _parse_chunks(data: bytes) -> tuple[dict[str, Any], bytes]:
    if not data.startswith(PNG_SIGNATURE):
        raise PngFrameError("not a PNG file")
    offset = len(PNG_SIGNATURE)
    ihdr: dict[str, Any] | None = None
    idat_parts: list[bytes] = []
    seen_iend = False
    chunk_count = 0
    while offset < len(data):
        if len(data) - offset < 12:
            raise PngFrameError("truncated PNG chunk header")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        if length > MAX_FILE_BYTES:
            raise PngFrameError("PNG chunk is too large")
        end = offset + 12 + length
        if end > len(data):
            raise PngFrameError("truncated PNG chunk")
        payload = data[offset + 8 : offset + 8 + length]
        declared_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if declared_crc != actual_crc:
            raise PngFrameError(f"bad CRC for PNG chunk {chunk_type!r}")
        chunk_count += 1
        if chunk_count > 10_000:
            raise PngFrameError("too many PNG chunks")
        if chunk_type == b"IHDR":
            if ihdr is not None or offset != len(PNG_SIGNATURE):
                raise PngFrameError("IHDR must be the first and only IHDR chunk")
            if length != 13:
                raise PngFrameError("IHDR length is invalid")
            (
                width,
                height,
                bit_depth,
                color_type,
                compression,
                filter_method,
                interlace,
            ) = struct.unpack(">IIBBBBB", payload)
            if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
                raise PngFrameError("PNG dimensions are outside the bounded range")
            if width * height > MAX_DECLARED_PIXELS:
                raise PngFrameError("PNG pixel count is outside the bounded range")
            if bit_depth != 8 or color_type != 2:
                raise PngFrameError("only 8-bit RGB (color type 2) PNG is supported")
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise PngFrameError("only non-interlaced, standard-filter PNG is supported")
            ihdr = {
                "width": width,
                "height": height,
                "bit_depth": bit_depth,
                "color_type": color_type,
                "interlace": interlace,
            }
        elif chunk_type == b"IDAT":
            if ihdr is None:
                raise PngFrameError("IDAT before IHDR")
            idat_parts.append(payload)
        elif chunk_type == b"IEND":
            if length != 0:
                raise PngFrameError("IEND must be empty")
            if offset + 12 != len(data):
                raise PngFrameError("trailing bytes after IEND")
            seen_iend = True
        else:
            # Only ancillary chunks that cannot affect pixels are skipped.
            # Unknown critical chunks are rejected.
            if not (chunk_type[0] & 0x20):
                raise PngFrameError(f"unsupported critical PNG chunk {chunk_type!r}")
        offset = end
        if seen_iend:
            break
    if ihdr is None or not idat_parts or not seen_iend:
        raise PngFrameError("PNG is missing IHDR/IDAT/IEND")
    return ihdr, b"".join(idat_parts)


def _decompress_bounded(compressed: bytes, expected: int) -> bytes:
    if expected <= 0 or expected > MAX_DECLARED_PIXELS * 4 + MAX_DIMENSION * 8:
        raise PngFrameError("declared PNG data size is outside the bounded range")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected + 1)
    except zlib.error as exc:
        raise PngFrameError(f"invalid PNG zlib stream: {exc}") from exc
    if len(raw) != expected:
        raise PngFrameError(
            f"decompressed PNG size {len(raw)} does not match expected {expected}"
        )
    if decompressor.unconsumed_tail:
        raise PngFrameError("PNG stream expands beyond the declared image size")
    if not decompressor.eof:
        raise PngFrameError("PNG zlib stream did not reach EOF")
    if decompressor.unused_data:
        raise PngFrameError("PNG zlib stream has trailing unused data")
    return raw


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw: bytes, width: int, height: int) -> bytearray:
    row_bytes = width * 3
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise PngFrameError("PNG raw byte length does not match dimensions")
    out = bytearray(width * height * 3)
    previous = bytearray(row_bytes)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        scan = raw[offset : offset + row_bytes]
        offset += row_bytes
        current = bytearray(row_bytes)
        if filter_type == 0:
            current[:] = scan
        elif filter_type == 1:
            for i in range(row_bytes):
                left = current[i - 3] if i >= 3 else 0
                current[i] = (scan[i] + left) & 0xFF
        elif filter_type == 2:
            for i in range(row_bytes):
                current[i] = (scan[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            for i in range(row_bytes):
                left = current[i - 3] if i >= 3 else 0
                up = previous[i]
                current[i] = (scan[i] + ((left + up) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(row_bytes):
                left = current[i - 3] if i >= 3 else 0
                up = previous[i]
                up_left = previous[i - 3] if i >= 3 else 0
                current[i] = (scan[i] + _paeth(left, up, up_left)) & 0xFF
        else:
            raise PngFrameError(f"unsupported PNG filter type {filter_type}")
        start = row * row_bytes
        out[start : start + row_bytes] = current
        previous = current
    return out


def decode_png_rgb(path: Path | str) -> tuple[int, int, bytes]:
    """Decode a bounded PNG8 RGB file into ``(width, height, rgb_bytes)``."""

    data = _read_bounded(Path(path))
    ihdr, compressed = _parse_chunks(data)
    width = int(ihdr["width"])
    height = int(ihdr["height"])
    expected = height * (width * 3 + 1)
    raw = _decompress_bounded(compressed, expected)
    pixels = _unfilter(raw, width, height)
    return width, height, bytes(pixels)


def _check_box(box: tuple[int, int, int, int], width: int, height: int) -> None:
    if len(box) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise PngFrameError("crop box must contain four integers")
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise PngFrameError("crop box is outside the PNG bounds")


def crop_rgb(
    width: int,
    height: int,
    pixels: bytes,
    box: tuple[int, int, int, int],
) -> tuple[int, int, bytes]:
    _check_box(box, width, height)
    if len(pixels) != width * height * 3:
        raise PngFrameError("pixel buffer length does not match dimensions")
    left, top, right, bottom = box
    cropped_width = right - left
    cropped_height = bottom - top
    row_bytes = cropped_width * 3
    result = bytearray(row_bytes * cropped_height)
    for row in range(cropped_height):
        source_start = ((top + row) * width + left) * 3
        target_start = row * row_bytes
        result[target_start : target_start + row_bytes] = pixels[
            source_start : source_start + row_bytes
        ]
    return cropped_width, cropped_height, bytes(result)


def _axis_weights(source_length: int, target_length: int) -> list[tuple[list[tuple[int, int]], int]]:
    """Return integer overlap weights for each target pixel.

    Source pixel ``i`` spans ``[i*target_length, (i+1)*target_length)``.
    Target pixel ``x`` spans ``[x*source_length, (x+1)*source_length)``.
    Overlap lengths are integers and sum to ``source_length`` per target pixel.
    """

    if not (1 <= source_length <= MAX_DIMENSION and 1 <= target_length <= MAX_DIMENSION):
        raise PngFrameError("resize dimensions are outside the bounded range")
    weights: list[tuple[list[tuple[int, int]], int]] = []
    for x in range(target_length):
        start = x * source_length
        end = (x + 1) * source_length
        first = start // target_length
        last = (end + target_length - 1) // target_length
        parts: list[tuple[int, int]] = []
        total = 0
        for i in range(first, min(last, source_length)):
            overlap = min(end, (i + 1) * target_length) - max(start, i * target_length)
            if overlap > 0:
                parts.append((i, overlap))
                total += overlap
        if total != source_length:
            raise PngFrameError("internal resize weight invariant failed")
        weights.append((parts, total))
    return weights


def resize_area_rgb(
    width: int,
    height: int,
    pixels: bytes,
    target_width: int,
    target_height: int,
) -> bytes:
    """Deterministic integer area-average resize for RGB8.

    The resize is separable. Each output pixel is the weighted average of the
    source pixels it covers, with integer weights divided by the exact source
    span. This is intended for downscaling the checked-in 913x1660 crop to
    704x1280 and is not a general-purpose resampler.
    """

    if len(pixels) != width * height * 3:
        raise PngFrameError("pixel buffer length does not match dimensions")
    if not (1 <= target_width <= width and 1 <= target_height <= height):
        raise PngFrameError("this bounded resizer only supports downscaling")
    x_weights = _axis_weights(width, target_width)
    y_weights = _axis_weights(height, target_height)

    # Horizontal pass: for every source row, produce target_width RGB pixels.
    horizontal = bytearray(height * target_width * 3)
    for row in range(height):
        source_row = row * width * 3
        target_row = row * target_width * 3
        for x, (parts, total) in enumerate(x_weights):
            r = g = b = 0
            for index, weight in parts:
                source = source_row + index * 3
                r += pixels[source] * weight
                g += pixels[source + 1] * weight
                b += pixels[source + 2] * weight
            out = target_row + x * 3
            horizontal[out] = (r + total // 2) // total
            horizontal[out + 1] = (g + total // 2) // total
            horizontal[out + 2] = (b + total // 2) // total

    # Vertical pass.
    result = bytearray(target_width * target_height * 3)
    for y, (parts, total) in enumerate(y_weights):
        target_row = y * target_width * 3
        for x in range(target_width):
            r = g = b = 0
            for index, weight in parts:
                source = (index * target_width + x) * 3
                r += horizontal[source] * weight
                g += horizontal[source + 1] * weight
                b += horizontal[source + 2] * weight
            out = target_row + x * 3
            result[out] = (r + total // 2) // total
            result[out + 1] = (g + total // 2) // total
            result[out + 2] = (b + total // 2) // total
    return bytes(result)


def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", crc)
    )


def encode_png_rgb(width: int, height: int, pixels: bytes) -> bytes:
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
        raise PngFrameError("output dimensions are outside the bounded range")
    if len(pixels) != width * height * 3:
        raise PngFrameError("pixel buffer length does not match output dimensions")
    raw = bytearray(height * (width * 3 + 1))
    source_offset = 0
    target_offset = 0
    for _row in range(height):
        raw[target_offset] = 0  # filter type None
        target_offset += 1
        raw[target_offset : target_offset + width * 3] = pixels[
            source_offset : source_offset + width * 3
        ]
        source_offset += width * 3
        target_offset += width * 3
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressed = zlib.compress(bytes(raw), 9)
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp_name = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            temp_name.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"))


def build_policy(
    *,
    source_path: Path,
    source_sha256: str,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    crop_box: tuple[int, int, int, int],
    derived_sha256: str,
    derived_size_bytes: int,
) -> dict[str, Any]:
    left, top, right, bottom = crop_box
    return {
        "schema": POLICY_SCHEMA,
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "width": source_width,
            "height": source_height,
            "format": "PNG8 RGB non-interlaced",
        },
        "target": {
            "width": target_width,
            "height": target_height,
            "format": "PNG8 RGB non-interlaced",
            "sha256": derived_sha256,
            "size_bytes": derived_size_bytes,
        },
        "crop": {
            "box_left_top_right_bottom": [left, top, right, bottom],
            "width": right - left,
            "height": bottom - top,
            "aspect": "11:20",
            "top_crop_pixels": top,
            "bottom_crop_pixels": source_height - bottom,
            "left_crop_pixels": left,
            "right_crop_pixels": source_width - right,
        },
        "resize": {
            "filter": "integer_area_average_rgb8",
            "scale_width": f"{target_width}/{right - left}",
            "scale_height": f"{target_height}/{bottom - top}",
            "anisotropic": (right - left) * target_height
            != (bottom - top) * target_width,
        },
        "note": (
            "Bounded local crop/resize only. The 6 px top crop removes background, "
            "not the head; source hash is verified before use."
        ),
    }


def derive_frame(
    *,
    source_path: Path | str,
    output_path: Path | str,
    policy_path: Path | str,
    expected_source_sha256: str,
    target_width: int = 704,
    target_height: int = 1280,
    crop_box: tuple[int, int, int, int] = (14, 6, 927, 1666),
) -> dict[str, Any]:
    """Create or verify the derived v3 frame; fail closed on any mismatch."""

    source = Path(source_path)
    output = Path(output_path)
    policy = Path(policy_path)
    if not source.is_file():
        raise PngFrameError(f"source frame is missing: {source}")
    actual_source_sha = _sha256_file(source)
    if actual_source_sha != expected_source_sha256.lower():
        raise PngFrameError("source frame SHA256 does not match the pinned value")
    source_width, source_height, source_pixels = decode_png_rgb(source)
    if (source_width, source_height) != (941, 1672):
        raise PngFrameError(
            f"source frame dimensions are {source_width}x{source_height}, expected 941x1672"
        )
    _check_box(crop_box, source_width, source_height)
    crop_width, crop_height, cropped = crop_rgb(
        source_width, source_height, source_pixels, crop_box
    )
    resized = resize_area_rgb(
        crop_width, crop_height, cropped, target_width, target_height
    )
    encoded = encode_png_rgb(target_width, target_height, resized)
    derived_sha = _sha256_bytes(encoded)

    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise PngFrameError("existing derived frame is not a regular file")
        if not policy.is_file():
            raise PngFrameError("derived frame exists without a policy file")
        try:
            existing_policy = json.loads(policy.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PngFrameError(f"cannot read existing policy: {exc}") from exc
        existing_sha = _sha256_file(output)
        expected = existing_policy.get("target", {}).get("sha256")
        if existing_sha != expected:
            raise PngFrameError("derived frame hash does not match its policy")
        if expected != derived_sha or existing_sha != derived_sha:
            raise PngFrameError(
                "derived frame/policy does not match the locally recomputed pinned output"
            )
        if existing_policy.get("source", {}).get("sha256") != expected_source_sha256:
            raise PngFrameError("existing policy source hash does not match the pin")
        existing_box = existing_policy.get("crop", {}).get(
            "box_left_top_right_bottom"
        )
        if existing_box != list(crop_box):
            raise PngFrameError("existing policy crop box does not match")
        if (
            existing_policy.get("target", {}).get("width") != target_width
            or existing_policy.get("target", {}).get("height") != target_height
        ):
            raise PngFrameError("existing policy target dimensions do not match")
        return existing_policy

    payload = build_policy(
        source_path=source,
        source_sha256=actual_source_sha,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        crop_box=crop_box,
        derived_sha256=derived_sha,
        derived_size_bytes=len(encoded),
    )
    _atomic_write_bytes(output, encoded)
    _atomic_write_json(policy, payload)
    return payload


__all__ = [
    "MAX_DIMENSION",
    "POLICY_SCHEMA",
    "PngFrameError",
    "build_policy",
    "crop_rgb",
    "decode_png_rgb",
    "derive_frame",
    "encode_png_rgb",
    "resize_area_rgb",
]
