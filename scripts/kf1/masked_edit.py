"""Offline, deterministic spatial-edit bundles. Never submits a ComfyUI job."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from . import png
from .errors import KF1Error

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "shotsmith/masked-edit/v1"
PIN = "7a28332047b6af638dc6c7f72ef3481a718769fd"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise KF1Error(f"{label}: требуется целое >= {minimum}")
    return value


def _image(item: dict, base: Path) -> tuple[bytes, png.PNGImage, Path]:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise KF1Error("Исходник должен содержать path и sha256")
    path = Path(item["path"])
    if not path.is_absolute():
        path = base / path
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 60 * 1024 * 1024:
        raise KF1Error(f"Небезопасный/слишком большой исходник: {path}")
    data = path.read_bytes()
    if sha(data) != item.get("sha256"):
        raise KF1Error(f"SHA256 исходника изменился: {path}")
    image = png.read_png(data, max_pixels=12_000_000)
    # Compositing below intentionally works on opaque RGB sources only.
    if any(image.rgba[i] != 255 for i in range(3, len(image.rgba), 4)):
        raise KF1Error("Исходники должны быть непрозрачными PNG")
    return data, image, path.resolve()


def _rect(raw: Any, width: int, height: int, label: str) -> tuple[int, int, int, int]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise KF1Error(f"{label}: ожидается [x,y,width,height]")
    x, y, w, h = (_integer(v, label) for v in raw)
    if w == 0 or h == 0 or x + w > width or y + h > height:
        raise KF1Error(f"{label}: прямоугольник вне изображения")
    return x, y, w, h


def rectangle_mask(width: int, height: int, regions: list, feather: int) -> bytes:
    """White=editable; feather lies INSIDE the declared regions, never outside."""
    mask = bytearray(width * height)
    for region in regions:
        x, y, w, h = _rect(region, width, height, "edit_regions")
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                distance = min(
                    xx - x if x > 0 else 255,
                    x + w - 1 - xx if x + w < width else 255,
                    yy - y if y > 0 else 255,
                    y + h - 1 - yy if y + h < height else 255,
                )
                alpha = 255 if feather == 0 else min(255, round(255 * distance / feather))
                i = yy * width + xx
                mask[i] = max(mask[i], alpha)
    return bytes(mask)


def polygon_cutout(image: png.PNGImage, points: list, flip: bool) -> png.PNGImage:
    if not isinstance(points, list) or not 3 <= len(points) <= 128:
        raise KF1Error("pose_polygon: нужны 3..128 точек")
    coords = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise KF1Error("pose_polygon: неверная точка")
        x, y = (_integer(v, "pose_polygon") for v in point)
        if x >= image.width or y >= image.height:
            raise KF1Error("pose_polygon: точка вне изображения")
        coords.append((x, y))
    left, top = min(x for x, _ in coords), min(y for _, y in coords)
    right, bottom = max(x for x, _ in coords) + 1, max(y for _, y in coords) + 1
    width, height = right - left, bottom - top
    if width < 2 or height < 2:
        raise KF1Error("pose_polygon: вырожденный контур")
    out = bytearray(width * height * 4)
    edges = list(zip(coords, coords[1:] + coords[:1]))
    count = 0
    for y in range(top, bottom):
        scan = y + 0.5
        intersections = sorted(
            x1 + (scan - y1) * (x2 - x1) / (y2 - y1)
            for (x1, y1), (x2, y2) in edges
            if (y1 <= scan < y2) or (y2 <= scan < y1)
        )
        for a, b in zip(intersections[0::2], intersections[1::2]):
            for x in range(max(left, math.ceil(a - 0.5)), min(right, math.ceil(b - 0.5))):
                dest_x = right - 1 - x if flip else x - left
                dest = ((y - top) * width + dest_x) * 4
                src = (y * image.width + x) * 4
                out[dest:dest + 4] = image.rgba[src:src + 4]
                count += 1
    if count == 0:
        raise KF1Error("pose_polygon: пустой контур")
    return png.PNGImage(width, height, 6, 8, 0, bytes(out))


def composite_rgba(original: bytes, generated: bytes, mask: bytes) -> bytes:
    if len(original) != len(generated) or len(original) != len(mask) * 4:
        raise KF1Error("Геометрия composite не совпадает")
    out = bytearray(original)
    for i, alpha in enumerate(mask):
        if alpha:
            for c in range(3):
                j = i * 4 + c
                out[j] = (generated[j] * alpha + original[j] * (255 - alpha) + 127) // 255
    return bytes(out)


def build_graph(prompt: str, prefix: str, settings: dict | None = None, negative: str = "wrong pose, inconsistent perspective, text, watermark, harsh contrast") -> dict:
    settings = settings or {}
    graph = json.loads((ROOT / "workflows/K1/qwen-edit-2511-2ref.api.json").read_text(encoding="utf-8"))
    # The untouched anchor has NO path to Qwen conditioning or sampling.
    graph["1"]["inputs"]["image"] = prefix + "-original.png"
    graph["2"]["inputs"]["image"] = prefix + "-pose.png"
    del graph["10"]  # exact matching geometry; never resize/crop the canvas
    graph["4"] = {"class_type": "LoadImage", "inputs": {"image": prefix + "-guide.png"}}
    graph["5"] = {"class_type": "LoadImage", "inputs": {"image": prefix + "-mask.png"}}
    graph["6"] = {"class_type": "ImageToMask", "inputs": {"image": ["5", 0], "channel": "red"}}
    for node in ("30", "31"):
        graph[node]["inputs"]["image1"] = ["4", 0]
        graph[node]["inputs"]["image2"] = ["2", 0]
    graph["30"]["inputs"]["prompt"] = prompt
    graph["31"]["inputs"]["prompt"] = negative
    graph["40"]["inputs"]["pixels"] = ["4", 0]
    graph["41"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["40", 0], "mask": ["6", 0]}}
    graph["50"]["inputs"]["latent_image"] = ["41", 0]
    sampler = graph["50"]["inputs"]
    for key, value in {"seed": 424242, "steps": 40, "cfg": 3.0, "denoise": 1.0}.items():
        sampler[key] = settings.get(key, value)
    for key, lo, hi in (("steps", 1, 100), ("cfg", 0, 10), ("denoise", 0.01, 1), ("seed", 0, 2**64-1)):
        value = sampler[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
            raise KF1Error(f"Неверный параметр {key}")
        if key in ("steps", "seed") and type(value) is not int:
            raise KF1Error(f"{key} должен быть целым")
    graph["65"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": ["1", 0], "source": ["60", 0], "mask": ["6", 0],
        "x": 0, "y": 0, "resize_source": False}}
    graph["70"]["inputs"] = {"filename_prefix": "shotsmith/masked/" + prefix, "images": ["65", 0]}
    return graph


def prepare(spec_path: Path | str, output: Path | str) -> dict:
    spec_path, output = Path(spec_path).resolve(), Path(output)
    try:
        if spec_path.is_symlink() or spec_path.stat().st_size > 256 * 1024:
            raise KF1Error("Небезопасный/слишком большой spec")
        spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
        if not isinstance(spec, dict):
            raise KF1Error("Spec должен быть JSON-объектом")
        if spec.get("schema") != SCHEMA:
            raise KF1Error("Неверная схема masked-edit")
        raw, original, source_path = _image(spec["source"], spec_path.parent)
        _, pose, pose_path = _image(spec["pose"], spec_path.parent)
        w, h = original.width, original.height
        if w % 16 or h % 16:
            raise KF1Error("Размер кадра должен быть кратен 16")
        regions = spec["edit_regions"]
        if not isinstance(regions, list) or not 1 <= len(regions) <= 16:
            raise KF1Error("edit_regions: нужны 1..16 прямоугольников")
        feather = _integer(spec.get("feather", 12), "feather")
        if feather > 64:
            raise KF1Error("feather > 64")
        flip = spec.get("flip_pose", False)
        if type(flip) is not bool:
            raise KF1Error("flip_pose должен быть boolean")
        mask = rectangle_mask(w, h, regions, feather)
        if not any(mask) or all(mask):
            raise KF1Error("Маска должна оставлять непустую защищённую область")
        cut = polygon_cutout(pose, spec["pose_polygon"], flip)
        x, y, box_w, box_h = _rect(spec["pose_placement"], w, h, "pose_placement")
        fitted = png.read_png(png.contain(cut, box_w, box_h, pad=False))
        x += (box_w - fitted.width) // 2
        y += box_h - fitted.height  # ground contact is bottom-aligned
        gray = bytes([128, 128, 128, 255]) * (w * h)
        guide = bytearray(composite_rgba(original.rgba, gray, mask))
        reference = bytearray(bytes([128, 128, 128, 255]) * (fitted.width * fitted.height))
        for yy in range(fitted.height):
            for xx in range(fitted.width):
                src = (yy * fitted.width + xx) * 4
                alpha = fitted.rgba[src + 3]
                dst = ((y + yy) * w + x + xx) * 4
                if alpha and mask[dst // 4] != 255:
                    raise KF1Error("Поза должна целиком помещаться в непрозрачную часть edit mask")
                for c in range(3):
                    guide[dst + c] = (fitted.rgba[src + c] * alpha + guide[dst + c] * (255-alpha) + 127) // 255
                    reference[src + c] = (fitted.rgba[src + c] * alpha + 128 * (255-alpha) + 127) // 255
        prompt = spec["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
            raise KF1Error("Нужна непустая инструкция <=8000 символов")
        source_hashes = {
            "scripts/kf1/masked_edit.py": sha(Path(__file__).read_bytes()),
            "scripts/kf1/png.py": sha(Path(png.__file__).read_bytes()),
            "workflows/K1/qwen-edit-2511-2ref.api.json": sha((ROOT / "workflows/K1/qwen-edit-2511-2ref.api.json").read_bytes()),
        }
        identity = sha(json.dumps({"spec": spec, "sources": source_hashes}, sort_keys=True, ensure_ascii=False).encode())[:16]
        prefix = "sm-" + identity
        mask_rgba = bytes(channel for a in mask for channel in (a, a, a, 255))
        files = {
            prefix + "-original.png": raw,
            prefix + "-guide.png": png.write_png(w, h, bytes(guide)),
            prefix + "-mask.png": png.write_png(w, h, mask_rgba),
            prefix + "-pose.png": png.write_png(fitted.width, fitted.height, bytes(reference)),
        }
        negative = spec.get("negative_prompt", "wrong pose, inconsistent perspective, text, watermark, harsh contrast")
        if not isinstance(negative, str) or len(negative) > 8000:
            raise KF1Error("Неверный negative_prompt")
        graph = build_graph(prompt, prefix, spec.get("settings"), negative)
        files["workflow.api.json"] = (json.dumps(graph, ensure_ascii=False, indent=2) + "\n").encode()
        files["spec.json"] = (json.dumps(spec, ensure_ascii=False, indent=2) + "\n").encode()
        manifest = {
            "schema": SCHEMA, "status": "PREPARED_OFFLINE", "gpu_verified": False,
            "automatic_submit": False, "comfy_commit": PIN, "prefix": prefix,
            "source": {"path": str(source_path), "sha256": sha(raw)},
            "pose": {"path": str(pose_path), "sha256": spec["pose"]["sha256"]},
            "geometry": [w, h], "feather_inside": feather,
            "protected_pixels": sum(v == 0 for v in mask),
            "edit_pixels": sum(v != 0 for v in mask),
            "normalizer": "stdlib-png-v1 (same implementation with/without Pillow)",
            "classes": sorted({node["class_type"] for node in graph.values()}),
            "files": {name: sha(data) for name, data in files.items()},
            "builder_sources_sha256": source_hashes,
            "guide_is_generated_video_frame": False,
            "limitations": ["Pose polygon and placement are explicit owner-editable layout data, not automatic segmentation.",
                            "Only pixels with mask=0 are protected; repaired background inside mask is synthesized.",
                            "No OpenPose/ControlNet/new weights. Pose and seam quality remain GPU_NOT_RUN.",
                            "Experimental graph is separate from KF1 generate and existing I1 profile inventory."]}
        if output.exists() or output.is_symlink():
            raise KF1Error("Выход уже существует; выберите новую папку")
        # All validation/rendering precedes output creation. Never overwrite source/state.
        output.mkdir(parents=True)
        for name, data in files.items():
            (output / name).write_bytes(data)
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "PREPARED_OFFLINE", "output": str(output.resolve()),
                "guide": str((output / (prefix + "-guide.png")).resolve()),
                "mask": str((output / (prefix + "-mask.png")).resolve()),
                "protected_pixels": manifest["protected_pixels"], "gpu_verified": False}
    except (OSError, ValueError, KeyError, TypeError, png.PNGError) as exc:
        raise KF1Error(f"masked-prepare: {exc}") from exc


def verify(package: Path | str, image_path: Path | str) -> dict:
    """Check exact decoded RGB outside mask, not PNG compression or metadata."""
    package = Path(package)
    try:
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise KF1Error("Неверный manifest")
        for name, expected in manifest["files"].items():
            if Path(name).name != name or name in (".", ".."):
                raise KF1Error("Небезопасный путь manifest")
            item = package / name
            if item.is_symlink() or sha(item.read_bytes()) != expected:
                raise KF1Error("Пакет изменён: " + name)
        prefix = manifest["prefix"]
        if not isinstance(prefix, str) or not re.fullmatch(r"sm-[0-9a-f]{16}", prefix):
            raise KF1Error("Неверный prefix manifest")
        original = png.read_png((package / (prefix + "-original.png")).read_bytes(), max_pixels=12_000_000)
        mask = png.read_png((package / (prefix + "-mask.png")).read_bytes(), max_pixels=12_000_000)
        result = png.read_png(Path(image_path).read_bytes(), max_pixels=12_000_000)
        if (result.width, result.height) != (original.width, original.height):
            raise KF1Error("Размер результата изменился")
        if (mask.width, mask.height) != (original.width, original.height):
            raise KF1Error("Размер маски изменился")
        changed = sum(mask.rgba[i] == 0 and result.rgba[i:i+3] != original.rgba[i:i+3]
                      for i in range(0, len(original.rgba), 4))
        return {"status": "PASS" if changed == 0 else "FAIL", "changed_protected_pixels": changed,
                "pose_quality": "NOT_VERIFIED", "checks": "decoded RGB outside mask only"}
    except (OSError, ValueError, KeyError, TypeError, png.PNGError) as exc:
        raise KF1Error(f"masked-verify: {exc}") from exc
