"""Local, self-contained storyboard export for KF1."""

from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .approval import approval_status
from .errors import StoreError
from .normalize import make_thumbnail, normalize_image, validate_png_bytes
from .plan import ResolvedPlan


def _atomic_bytes(path: Path, data: bytes) -> None:
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise StoreError("export path escapes output root")
    return path


def _dependency_ok(plan: ResolvedPlan, state: Mapping[str, Any], frame_id: str) -> bool:
    """Compare parent original/output identity with the recorded derivative.

    The parent output hash and the normalized anchor derivative are different
    identities when custom geometry is used.  Compare parent-to-parent, then
    prove the recorded derivative by recomputing it from the exact parent file.
    """

    frame = next((item for item in plan.frames if item.frame_id == frame_id), None)
    if frame is None or not frame.anchor_frame:
        return True
    parent_state = state.get("frames", {}).get(frame.anchor_frame, {})
    if not isinstance(parent_state, Mapping) or parent_state.get("invalidated"):
        return False
    parent_current = parent_state.get("current_version")
    parent_info = parent_state.get("versions", {}).get(str(parent_current), {})
    parent_output = parent_info.get("output") if isinstance(parent_info, Mapping) else None
    if not isinstance(parent_output, Mapping) or not parent_output.get("path"):
        return False
    parent_path = Path(str(parent_output["path"]))
    if not parent_path.is_file():
        return False
    try:
        parent_bytes = parent_path.read_bytes()
        parent_decoded = validate_png_bytes(parent_bytes)
    except Exception:
        return False
    if (
        parent_decoded["sha256"] != parent_output.get("sha256")
        or parent_decoded["width"] != parent_output.get("width")
        or parent_decoded["height"] != parent_output.get("height")
    ):
        return False
    own_state = state.get("frames", {}).get(frame_id, {})
    own_current = own_state.get("current_version") if isinstance(own_state, Mapping) else None
    own_info = own_state.get("versions", {}).get(str(own_current), {}) if own_current else {}
    source_hashes = own_info.get("source_hashes", {}) if isinstance(own_info, Mapping) else {}
    if not isinstance(source_hashes, Mapping):
        return False
    parent_identity = source_hashes.get("anchor_source_sha256")
    if parent_identity is None:
        # Historical state predates the separated identity; only accept the old
        # anchor value when it is the parent output hash itself.
        parent_identity = source_hashes.get("anchor")
    if parent_identity != parent_output.get("sha256"):
        return False
    derivative_sha = source_hashes.get("anchor_derivative_sha256")
    policy = str(source_hashes.get("anchor_policy") or "")
    geometry = source_hashes.get("anchor_geometry") or plan.output_policy["candidate_geometry"]
    if derivative_sha is None:
        # Old state has no derivative provenance; parent identity is all we have.
        return True
    if not (isinstance(geometry, (list, tuple)) and len(geometry) == 2):
        return False
    target = (int(geometry[0]), int(geometry[1]))
    if policy == "preserve_geometry":
        return derivative_sha == parent_output.get("sha256") and (
            int(parent_output.get("width") or 0),
            int(parent_output.get("height") or 0),
        ) == target
    if policy == "lineage_only":
        return (
            source_hashes.get("anchor_uploaded") is False
            and derivative_sha == parent_output.get("sha256")
            and parent_identity == parent_output.get("sha256")
        )
    if policy not in {"preserve", "contain", "crop"}:
        return False
    with tempfile.TemporaryDirectory() as temp_dir:
        check_path = Path(temp_dir) / "anchor-check.png"
        try:
            record = normalize_image(
                parent_path,
                check_path,
                policy=policy,
                target=list(target),
                pad=(policy == "contain"),
                label="anchor-dependency",
            )
        except Exception:
            return False
    return (
        record["output"]["sha256"] == derivative_sha
        and (int(record["output"]["width"]), int(record["output"]["height"])) == target
    )


def export_storyboard(
    plan: ResolvedPlan,
    state: Mapping[str, Any],
    output_dir: Path | str,
    *,
    store: Any = None,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    manifest_scenes = []
    for scene in plan.data["scenes"]:
        scene_id = str(scene["id"])
        roles = []
        for role in ("A", "B"):
            frame_id = f"{scene_id}/{role}"
            frame_state = state.get("frames", {}).get(frame_id, {})
            current = frame_state.get("current_version") if isinstance(frame_state, Mapping) else None
            versions = frame_state.get("versions", {}) if isinstance(frame_state, Mapping) else {}
            current_info = versions.get(str(current), {}) if current else {}
            if (
                (isinstance(frame_state, Mapping) and frame_state.get("invalidated"))
                or not _dependency_ok(plan, state, frame_id)
            ):
                roles.append(
                    {
                        "role": role,
                        "status": "INVALIDATED",
                        "current_version": current,
                        "historical_versions": sorted(
                            int(key) for key in versions if str(key).isdigit()
                        ),
                    }
                )
                continue
            output = current_info.get("output") if isinstance(current_info, Mapping) else None
            if not isinstance(output, Mapping) or not output.get("path"):
                roles.append({"role": role, "status": "MISSING", "approval_status": "NOT_APPLICABLE"})
                continue
            source = Path(str(output["path"]))
            if not source.is_file():
                roles.append({"role": role, "status": "MISSING_FILE", "approval_status": "NOT_APPLICABLE"})
                continue
            data = source.read_bytes()
            info = validate_png_bytes(data)
            if (
                info["sha256"] != output.get("sha256")
                or info["width"] != output.get("width")
                or info["height"] != output.get("height")
            ):
                raise StoreError(
                    f"{frame_id}: PNG не совпадает с сохранённым provenance; export отклонён"
                )
            rel = f"assets/{scene_id}/{role}-v{int(current):04d}.png"
            target = _contained(root, rel)
            _atomic_bytes(target, data)
            thumb_rel = f"assets/{scene_id}/{role}-v{int(current):04d}.thumb.png"
            thumb = _contained(root, thumb_rel)
            make_thumbnail(target, thumb, max_size=320)
            video_size = tuple(plan.output_policy["video_geometry"])
            video_policy = str(plan.output_policy.get("video_policy", "contain"))
            if video_policy == "preserve":
                video_policy = "contain"
            wan_rel = f"assets/{scene_id}/{role}-v{int(current):04d}-wan-{video_size[0]}x{video_size[1]}.png"
            wan_target = _contained(root, wan_rel)
            wan_record = normalize_image(
                target,
                wan_target,
                policy=video_policy,
                target=list(video_size),
                pad=True,
                label="wan",
            )
            attempts = [
                item
                for item in state.get("attempts", [])
                if isinstance(item, Mapping) and item.get("frame_id") == frame_id
            ]
            attempt_id = current_info.get("attempt_id") if isinstance(current_info, Mapping) else None
            latest = next(
                (item for item in attempts if item.get("attempt_id") == attempt_id),
                attempts[-1] if attempts else {},
            )
            source_status = str(current_info.get("status") or "COMPLETE")
            roles.append(
                {
                    "role": role,
                    "status": "COMPLETE" if source_status in {"COMPLETE", "PRESERVED"} else source_status,
                    "source_status": source_status,
                    "preserved": source_status == "PRESERVED",
                    "version": int(current),
                    "path": rel,
                    "thumb": thumb_rel,
                    "sha256": info["sha256"],
                    "size_bytes": info["size_bytes"],
                    "width": info["width"],
                    "height": info["height"],
                    "wan_path": wan_rel,
                    "wan_sha256": wan_record["output"]["sha256"],
                    "wan_width": wan_record["output"]["width"],
                    "wan_height": wan_record["output"]["height"],
                    "wan_policy": video_policy,
                    "original": current_info.get("original") if role == "A" else None,
                    "normalization": current_info.get("candidate") if role == "A" else None,
                    "seed": latest.get("seed") if isinstance(latest, Mapping) else None,
                    "parameters": latest.get("settings") if isinstance(latest, Mapping) else None,
                    "model_pins": plan.pins.get("models"),
                    "workflow_pins": plan.pins.get("workflows"),
                    "lineage": {
                        "attempt_id": current_info.get("attempt_id"),
                        "binding_sha256": current_info.get("binding_sha256"),
                        "source_hashes": current_info.get("source_hashes"),
                        "prompt_id": latest.get("prompt_id") if isinstance(latest, Mapping) else None,
                        "versions": sorted(int(key) for key in versions if str(key).isdigit()),
                    },
                    "approval_status": (
                        approval_status(plan, store, state, frame_id)
                        if store is not None
                        else {"status": "NOT_AVAILABLE"}
                    ).get("status"),
                    "approval": (
                        approval_status(plan, store, state, frame_id)
                        if store is not None
                        else {"status": "NOT_AVAILABLE"}
                    ),
                    "visual_status": "NOT_REVIEWED",
                    "quality": "NOT_RUN",
                }
            )
        manifest_scenes.append(
            {
                "scene_id": scene_id,
                "description": scene["description"],
                "roles": roles,
            }
        )
    manifest = {
        "schema": "anime-studio/kf1-storyboard/v1",
        "plan_sha256": plan.plan_sha256,
        "plan_title": plan.title,
        "plan_path": str(plan.path),
        "scenes": manifest_scenes,
        "visual_status": "NOT_REVIEWED",
        "quality": "NOT_RUN",
        "video_submitted": False,
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }
    _atomic_json(root / "storyboard.json", manifest)
    _write_html(root, manifest)
    return manifest


def _write_html(root: Path, manifest: Mapping[str, Any]) -> None:
    rows = []
    for scene in manifest.get("scenes", []):
        scene_id = html.escape(str(scene.get("scene_id")))
        description = html.escape(str(scene.get("description")))
        cards = []
        for role in scene.get("roles", []):
            role_name = html.escape(str(role.get("role")))
            status = html.escape(str(role.get("status") or "MISSING"))
            version = html.escape(str(role.get("version") or ""))
            if role.get("status") not in {"COMPLETE", "PRESERVED"} or not role.get("path"):
                historical = role.get("historical_versions") or []
                history_text = (
                    f"; versions {', '.join(str(item) for item in historical)}"
                    if historical
                    else ""
                )
                version_text = f" v{version}" if version else ""
                cards.append(
                    f'<div class="card unready"><h3>{role_name}{version_text}</h3>'
                    f'<p>{status}{history_text}; no usable image</p></div>'
                )
                continue
            img = html.escape(str(role.get("thumb") or role.get("path")), quote=True)
            full = html.escape(str(role.get("path")), quote=True)
            approval_label = html.escape(
                str(role.get("approval_status") or "NOT_AVAILABLE")
            )
            cards.append(
                f'<div class="card"><h3>{role_name} v{version}</h3>'
                f'<a href="{full}"><img src="{img}" alt="{role_name} {version}"></a>'
                f'<p>{status}; visual NOT_REVIEWED; approval {approval_label}; '
                f'quality NOT_RUN</p></div>'
            )
        rows.append(
            f'<section><h2>{scene_id}</h2><p>{description}</p>'
            f'<div class="row">{"".join(cards)}</div></section>'
        )
    document = (
        "<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\">"
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; "
        "img-src 'self'; style-src 'unsafe-inline'\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>KF1 storyboard</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;background:#fafafa;color:#222}"
        "section{margin-bottom:2rem}.row{display:flex;gap:1rem;flex-wrap:wrap}"
        ".card{background:#fff;border:1px solid #ddd;padding:.75rem;max-width:360px}"
        ".card img{max-width:320px;height:auto;display:block}.missing{color:#a00}"
        "p{white-space:pre-wrap}</style></head><body>"
        "<h1>KF1 storyboard</h1>"
        f"<p>Plan SHA256: <code>{html.escape(str(manifest.get('plan_sha256')))}</code></p>"
        "<p>Visual status: NOT_REVIEWED. Quality: NOT_RUN. Video: not submitted.</p>"
        + "".join(rows)
        + "</body></html>"
    )
    _atomic_bytes(root / "index.html", document.encode("utf-8"))


__all__ = ["export_storyboard"]
