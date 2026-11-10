"""Tiny stdlib PNG reader/writer used when Pillow is unavailable.

KF1 only needs 8-bit non-interlaced RGB/RGBA/gray PNGs.  Pillow is still used
when installed for better format/orientation handling; this module keeps the CLI
usable in the project's narrow venv without adding a new runtime dependency.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Any

SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PIXELS = 50_000_000


class PNGError(ValueError):
    pass


@dataclass(frozen=True)
class PNGImage:
    width: int
    height: int
    color_type: int
    bit_depth: int
    interlace: int
    rgba: bytes


def _read_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if not data.startswith(SIGNATURE):
        raise PNGError("not PNG")
    chunks: list[tuple[bytes, bytes]] = []
    offset = len(SIGNATURE)
    seen_ihdr = False
    seen_iend = False
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(data):
            raise PNGError("truncated PNG chunk")
        payload = data[start:end]
        declared = struct.unpack(">I", data[end : end + 4])[0]
        if zlib.crc32(chunk_type + payload) & 0xFFFFFFFF != declared:
            raise PNGError("PNG CRC mismatch")
        if chunk_type == b"IHDR":
            if seen_ihdr or chunks:
                raise PNGError("IHDR must be exactly first")
            seen_ihdr = True
        elif not seen_ihdr:
            raise PNGError("IHDR must be first")
        if chunk_type == b"IEND":
            if seen_iend or length != 0:
                raise PNGError("invalid IEND")
            seen_iend = True
            chunks.append((chunk_type, payload))
            offset = end + 4
            break
        if seen_iend:
            raise PNGError("data after IEND")
        chunks.append((chunk_type, payload))
        offset = end + 4
    if not chunks or chunks[0][0] != b"IHDR":
        raise PNGError("missing IHDR")
    if not seen_iend:
        raise PNGError("missing IEND")
    if offset != len(data):
        raise PNGError("trailing data after IEND")
    if not any(item[0] == b"IDAT" for item in chunks):
        raise PNGError("missing IDAT")
    return chunks


def _unfilter(raw: bytes, width: int, height: int, bpp: int) -> bytes:
    stride = width * bpp
    result = bytearray(height * stride)
    previous = bytearray(stride)
    pos = 0
    for row in range(height):
        if pos >= len(raw):
            raise PNGError("truncated scanline")
        filter_type = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        if len(line) != stride:
            raise PNGError("truncated scanline data")
        pos += stride
        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                up = previous[i]
                up_left = previous[i - bpp] if i >= bpp else 0
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                line[i] = (line[i] + predictor) & 0xFF
        else:
            raise PNGError("unsupported PNG filter")
        result[row * stride : (row + 1) * stride] = line
        previous = line
    return bytes(result)


def read_png(data: bytes, *, max_pixels: int = MAX_PIXELS) -> PNGImage:
    chunks = _read_chunks(data)
    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        raise PNGError("invalid IHDR")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if width <= 0 or height <= 0:
        raise PNGError("invalid dimensions")
    if width * height > max_pixels:
        raise PNGError("PNG pixel count exceeds limit")
    if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        raise PNGError("only 8-bit non-interlaced PNG is supported")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise PNGError("only gray/RGB/RGBA PNG is supported")
    compressed = b"".join(payload for chunk_type, payload in chunks if chunk_type == b"IDAT")
    if not compressed:
        raise PNGError("missing IDAT")
    expected = height * (1 + width * channels)
    if expected > max_pixels * 4 + height:
        raise PNGError("PNG decoded size exceeds limit")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected + 1)
    except zlib.error as exc:
        raise PNGError("invalid PNG zlib data") from exc
    if (
        len(raw) != expected
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise PNGError("PNG stream length/terminal mismatch")
    pixels = _unfilter(raw, width, height, channels)
    rgba = bytearray(width * height * 4)
    for index in range(width * height):
        source = pixels[index * channels : (index + 1) * channels]
        if color_type == 0:
            gray = source[0]
            rgba[index * 4 : index * 4 + 4] = bytes((gray, gray, gray, 255))
        elif color_type == 2:
            rgba[index * 4 : index * 4 + 4] = bytes((source[0], source[1], source[2], 255))
        elif color_type == 4:
            gray, alpha = source
            rgba[index * 4 : index * 4 + 4] = bytes((gray, gray, gray, alpha))
        else:
            rgba[index * 4 : index * 4 + 4] = bytes(source)
    return PNGImage(width, height, color_type, bit_depth, interlace, bytes(rgba))


def write_png(width: int, height: int, rgba: bytes) -> bytes:
    if len(rgba) != width * height * 4:
        raise PNGError("RGBA buffer size mismatch")
    raw = bytearray()
    stride = width * 4
    for row in range(height):
        raw.append(0)
        raw.extend(rgba[row * stride : (row + 1) * stride])
    compressed = zlib.compress(bytes(raw), 6)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def _bilinear(image: PNGImage, width: int, height: int) -> bytes:
    source = image.rgba
    sw, sh = image.width, image.height
    out = bytearray(width * height * 4)
    for y in range(height):
        sy = (y + 0.5) * sh / height - 0.5
        y0 = max(0, min(sh - 1, int(sy)))
        y1 = max(0, min(sh - 1, y0 + 1))
        fy = max(0.0, min(1.0, sy - y0))
        for x in range(width):
            sx = (x + 0.5) * sw / width - 0.5
            x0 = max(0, min(sw - 1, int(sx)))
            x1 = max(0, min(sw - 1, x0 + 1))
            fx = max(0.0, min(1.0, sx - x0))
            base = (y * width + x) * 4
            for channel in range(4):
                p00 = source[(y0 * sw + x0) * 4 + channel]
                p10 = source[(y0 * sw + x1) * 4 + channel]
                p01 = source[(y1 * sw + x0) * 4 + channel]
                p11 = source[(y1 * sw + x1) * 4 + channel]
                top = p00 * (1 - fx) + p10 * fx
                bottom = p01 * (1 - fx) + p11 * fx
                out[base + channel] = int(round(top * (1 - fy) + bottom * fy))
    return bytes(out)


def contain(image: PNGImage, width: int, height: int, *, pad: bool) -> bytes:
    scale = min(width / image.width, height / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = _bilinear(image, new_w, new_h)
    if not pad:
        return write_png(new_w, new_h, resized)
    canvas = bytearray(width * height * 4)
    for index in range(width * height):
        canvas[index * 4 : index * 4 + 3] = b"\xff\xff\xff"
        canvas[index * 4 + 3] = 255
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    for y in range(new_h):
        dest = ((top + y) * width + left) * 4
        source = y * new_w * 4
        canvas[dest : dest + new_w * 4] = resized[source : source + new_w * 4]
    return write_png(width, height, bytes(canvas))


def crop(image: PNGImage, width: int, height: int) -> bytes:
    scale = max(width / image.width, height / image.height)
    new_w = max(width, int(round(image.width * scale)))
    new_h = max(height, int(round(image.height * scale)))
    resized = _bilinear(image, new_w, new_h)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    out = bytearray(width * height * 4)
    for y in range(height):
        source = ((top + y) * new_w + left) * 4
        dest = y * width * 4
        out[dest : dest + width * 4] = resized[source : source + width * 4]
    return write_png(width, height, bytes(out))


def thumbnail(image: PNGImage, max_size: int) -> bytes:
    scale = min(max_size / image.width, max_size / image.height, 1.0)
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    return write_png(width, height, _bilinear(image, width, height))


__all__ = [
    "MAX_PIXELS",
    "PNGError",
    "PNGImage",
    "contain",
    "crop",
    "read_png",
    "thumbnail",
    "write_png",
]
