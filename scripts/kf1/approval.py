"""Exact, local owner approval for one immutable KF1 frame version.

Approval is a small sidecar next to the existing version artifact.  It never
copies pixels and never creates a hidden database.  It binds the revision,
frame, version, content hash, saved binding/intent/result, source hashes and
parent lineage.  It is a local owner attestation, not a signature and not an
automatic art-quality score.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .binding import verify_saved_binding
from .errors import ApprovalRequired, StoreError
from .normalize import validate_png_bytes
from .plan import FrameSpec, ResolvedPlan
from .store import PlanStore

SCHEMA = "anime-studio/kf1-approval/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(plan: ResolvedPlan, frame_id: str) -> FrameSpec:
    for frame in plan.frames:
        if frame.frame_id == frame_id:
            return frame
    raise StoreError(f"кадр не найден в плане: {frame_id}")


def _parent_version(plan: ResolvedPlan, state: Mapping[str, Any], frame: FrameSpec) -> int:
    if not frame.anchor_frame:
        return 1
    anchor_state = state.get("frames", {}).get(frame.anchor_frame, {})
    if not isinstance(anchor_state, Mapping):
        return 1
    return int(anchor_state.get("current_version") or 1)


def _parent_links(
    plan: ResolvedPlan, state: Mapping[str, Any], frame: FrameSpec
) -> list[dict[str, Any]]:
    if not frame.anchor_frame:
        return []
    anchor = _frame(plan, frame.anchor_frame)
    anchor_state = state.get("frames", {}).get(anchor.frame_id, {})
    if not isinstance(anchor_state, Mapping):
        raise StoreError(f"{frame.frame_id}: parent state отсутствует")
    version = int(anchor_state.get("current_version") or 0)
    info = anchor_state.get("versions", {}).get(str(version), {})
    if not isinstance(info, Mapping):
        raise StoreError(f"{frame.frame_id}: parent version отсутствует")
    output = info.get("output")
    if not isinstance(output, Mapping):
        raise StoreError(f"{frame.frame_id}: parent output отсутствует")
    link = {
        "frame_id": anchor.frame_id,
        "version": version,
        "status": info.get("status"),
        "content_sha256": str(output.get("sha256") or ""),
        "binding_sha256": str(info.get("binding_sha256") or ""),
        "source": anchor.source,
    }
    if anchor.source == "provided_frame":
        provided = plan.provided_frames.get(anchor.frame_id)
        if isinstance(provided, Mapping):
            link["provenance"] = dict(provided.get("provenance") or {})
    return [link]


def _validate_artifact(
    store: PlanStore, output: Mapping[str, Any], *, expected_sha256: str | None
) -> tuple[Path, dict[str, Any]]:
    path = Path(str(output.get("path") or ""))
    resolved = store.assert_artifact_contained(path)
    try:
        info = validate_png_bytes(resolved.read_bytes())
    except Exception as exc:
        raise StoreError(f"artifact не декодируется: {resolved}") from exc
    if (
        info["sha256"] != output.get("sha256")
        or info["width"] != output.get("width")
        or info["height"] != output.get("height")
    ):
        raise StoreError("artifact bytes/hash/geometry не совпадают с state")
    if expected_sha256 is not None and info["sha256"] != expected_sha256:
        raise StoreError("artifact SHA256 не совпадает с --sha256")
    return resolved, info


def validate_saved_version(
    plan: ResolvedPlan,
    store: PlanStore,
    state: Mapping[str, Any],
    frame: FrameSpec,
    version: int,
    version_info: Mapping[str, Any],
) -> dict[str, Any]:
    version_dir = store.version_dir(frame.frame_id, version)
    result_path = version_dir / "result.json"
    binding_path = version_dir / "binding.json"
    intent_path = version_dir / "intent.json"
    for path in (result_path, binding_path, intent_path):
        if not path.is_file():
            raise StoreError(f"{frame.frame_id}: отсутствует {path.name}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StoreError(f"{frame.frame_id}: saved receipt повреждён") from exc
    if not all(isinstance(item, Mapping) for item in (result, binding, intent)):
        raise StoreError(f"{frame.frame_id}: saved receipt не объект")
    if result.get("attempt_id") != version_info.get("attempt_id"):
        raise StoreError("result attempt_id не совпадает с state")
    if result.get("version") != version or result.get("frame_id") != frame.frame_id:
        raise StoreError("result frame/version не совпадает")
    if result.get("plan_sha256") != state.get("plan_sha256"):
        raise StoreError("result plan_sha256 не совпадает с revision")
    if result.get("revision_id") != state.get("revision_id"):
        raise StoreError("result revision_id не совпадает")
    if result.get("binding_sha256") != version_info.get("binding_sha256"):
        raise StoreError("result binding_sha256 не совпадает с state")
    if result.get("source_hashes") != version_info.get("source_hashes"):
        raise StoreError("result source_hashes не совпадает с state")
    output = result.get("output")
    if not isinstance(output, Mapping):
        raise StoreError("result output отсутствует")
    if output.get("path") != (version_info.get("output") or {}).get("path"):
        raise StoreError("result output path не совпадает с state")
    if output.get("sha256") != (version_info.get("output") or {}).get("sha256"):
        raise StoreError("result output sha256 не совпадает с state")

    roles = binding.get("roles")
    if not isinstance(roles, list) or not roles:
        raise StoreError("binding roles отсутствуют")
    workflow_pin = plan.pins.get("workflows", {}).get(str(len(roles)), {})
    expected_workflow = workflow_pin.get("sha256") if isinstance(workflow_pin, Mapping) else None
    parent_version = _parent_version(plan, state, frame)
    verified = verify_saved_binding(
        binding,
        plan_sha256=str(state.get("plan_sha256")),
        parent_version=parent_version,
        candidate_geometry=tuple(plan.output_policy["candidate_geometry"]),
        expected_workflow_sha256=expected_workflow,
    )
    if verified["stable_binding_sha256"] != version_info.get("binding_sha256"):
        raise StoreError("saved binding stable hash не совпадает с state")
    if intent.get("binding_sha256") != verified["stable_binding_sha256"]:
        raise StoreError("intent binding_sha256 не совпадает с saved binding")
    if intent.get("prompt") != binding.get("positive_prompt"):
        raise StoreError("intent prompt не совпадает с saved binding")
    if intent.get("negative_prompt") != binding.get("negative_prompt"):
        raise StoreError("intent negative не совпадает с saved binding")
    if intent.get("settings") != binding.get("settings"):
        raise StoreError("intent settings не совпадает с saved binding")
    if intent.get("workflow_sha256") != verified["workflow_sha256"]:
        raise StoreError("intent workflow_sha256 не совпадает")
    if intent.get("source_hashes") != version_info.get("source_hashes"):
        raise StoreError("intent source_hashes не совпадает с state")
    graph_sha = _sha(binding["bound_workflow"])
    if intent.get("bound_workflow_sha256") != graph_sha:
        raise StoreError("intent bound_workflow_sha256 не совпадает с actual graph")
    source_hashes = version_info.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise StoreError("state source_hashes отсутствуют")
    pins_sha = _sha(plan.pins)
    if source_hashes.get("pins_sha256") != pins_sha:
        raise StoreError("source_hashes.pins_sha256 не совпадает с plan pins")
    if source_hashes.get("plan_sha256") != state.get("plan_sha256"):
        raise StoreError("source_hashes.plan_sha256 не совпадает")
    if source_hashes.get("revision_id") != state.get("revision_id"):
        raise StoreError("source_hashes.revision_id не совпадает")
    parents = _parent_links(plan, state, frame)
    if parents:
        parent = parents[0]
        if source_hashes.get("anchor_source_sha256") != parent["content_sha256"]:
            raise StoreError("source_hashes.anchor_source_sha256 не совпадает с parent")
        if int(source_hashes.get("anchor_version") or 0) != int(parent["version"]):
            raise StoreError("source_hashes.anchor_version не совпадает с parent")
    return {
        "binding": dict(binding),
        "intent": dict(intent),
        "result": dict(result),
        "verified": verified,
        "parents": parents,
    }


def _valid_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise StoreError(f"{name} должен быть 64-hex SHA256")
    return value


def _source_state(
    store: PlanStore, frame: FrameSpec, provenance: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    revision_id = _valid_sha(provenance.get("revision_id"), "source revision_id")
    plan_sha = _valid_sha(provenance.get("plan_sha256"), "source plan_sha256")
    candidates: list[Path] = []
    if frame.provided_path is not None:
        provided = Path(frame.provided_path).resolve()
        for parent in provided.parents:
            if parent.name == revision_id:
                candidates.append(parent / "state.json")
    plans_root = store.base_root / "plans"
    if plans_root.is_dir():
        for plan_dir in sorted(plans_root.iterdir()):
            if plan_dir.is_dir() and re.fullmatch(r"[0-9a-f]{64}", plan_dir.name):
                candidates.append(plan_dir / "revisions" / revision_id / "state.json")
    seen: set[Path] = set()
    for state_path in candidates:
        if state_path in seen or not state_path.is_file():
            continue
        seen.add(state_path)
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"source state повреждён: {state_path}") from exc
        if not isinstance(data, dict):
            raise StoreError("source state не объект")
        if data.get("plan_sha256") != plan_sha or data.get("revision_id") != revision_id:
            continue
        return data, state_path
    raise StoreError("source original state не найден для provided reference")


def _validate_source_provenance(
    store: PlanStore,
    frame: FrameSpec,
    provenance: Mapping[str, Any],
    *,
    actual_content_sha256: str,
) -> dict[str, Any]:
    source_plan = _valid_sha(provenance.get("plan_sha256"), "source plan_sha256")
    source_revision = _valid_sha(provenance.get("revision_id"), "source revision_id")
    source_content = _valid_sha(
        provenance.get("content_sha256"), "source content_sha256"
    )
    source_frame = provenance.get("frame_id")
    if not isinstance(source_frame, str) or not source_frame.strip():
        raise StoreError("source frame_id должен быть непустой строкой")
    source_version = provenance.get("version")
    if isinstance(source_version, bool) or not isinstance(source_version, int) or source_version < 1:
        raise StoreError("source version должен быть int >= 1")
    if source_content != actual_content_sha256:
        raise StoreError("source content_sha256 не совпадает с imported artifact")
    source_binding = provenance.get("binding_sha256")
    if source_binding is not None:
        source_binding = _valid_sha(source_binding, "source binding_sha256")
    source_attempt = provenance.get("attempt_id")
    if source_attempt is not None and (
        not isinstance(source_attempt, str) or not source_attempt
    ):
        raise StoreError("source attempt_id должен быть непустой строкой")
    source_prompt = provenance.get("prompt_id")
    if source_prompt is not None and (
        not isinstance(source_prompt, str) or not source_prompt
    ):
        raise StoreError("source prompt_id должен быть непустой строкой")

    state, state_path = _source_state(store, frame, provenance)
    source_frame_state = state.get("frames", {}).get(source_frame)
    if not isinstance(source_frame_state, Mapping):
        raise StoreError("source frame state отсутствует")
    source_info = source_frame_state.get("versions", {}).get(str(source_version))
    if not isinstance(source_info, Mapping):
        raise StoreError("source version state отсутствует")
    if source_info.get("status") not in {"COMPLETE", "PRESERVED"}:
        raise StoreError("source version не COMPLETE/PRESERVED")
    output = source_info.get("output")
    if not isinstance(output, Mapping):
        raise StoreError("source output отсутствует")
    if output.get("sha256") != source_content:
        raise StoreError("source state output SHA не совпадает с provenance")
    if source_info.get("status") == "COMPLETE":
        if source_binding is None or source_info.get("binding_sha256") != source_binding:
            raise StoreError("source binding_sha256 не совпадает с source state")
        if source_attempt is None or source_info.get("attempt_id") != source_attempt:
            raise StoreError("source attempt_id не совпадает с source state")
        source_attempts = [
            item
            for item in state.get("attempts", [])
            if isinstance(item, Mapping) and item.get("attempt_id") == source_attempt
        ]
        if not source_attempts:
            raise StoreError("source attempt отсутствует в source state")
        if source_prompt is None or source_attempts[-1].get("prompt_id") != source_prompt:
            raise StoreError("source prompt_id не совпадает с source attempt")
        source_result_path = (
            state_path.parent
            / "frames"
            / source_frame.split("/", 1)[0]
            / (source_frame.split("/", 1)[1] if "/" in source_frame else "A")
            / f"v{source_version:04d}"
            / "result.json"
        )
        if not source_result_path.is_file():
            raise StoreError("source result receipt отсутствует")
        try:
            result = json.loads(source_result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError("source result receipt повреждён") from exc
        if (
            result.get("attempt_id") != source_attempt
            or result.get("version") != source_version
            or result.get("binding_sha256") != source_binding
            or (result.get("output") or {}).get("sha256") != source_content
        ):
            raise StoreError("source result receipt не совпадает с provenance")
    else:
        if source_binding is not None or source_attempt is not None or source_prompt is not None:
            raise StoreError("PRESERVED source не должен нести generated binding/attempt/prompt")
    return {
        "plan_sha256": source_plan,
        "revision_id": source_revision,
        "frame_id": source_frame,
        "version": source_version,
        "content_sha256": source_content,
        "binding_sha256": source_binding,
        "attempt_id": source_attempt,
        "prompt_id": source_prompt,
        "status": source_info.get("status"),
        "source_state_verified": True,
    }


def _validate_provided(
    plan: ResolvedPlan,
    store: PlanStore,
    state: Mapping[str, Any],
    frame: FrameSpec,
    version: int,
    version_info: Mapping[str, Any],
    actual_info: Mapping[str, Any],
) -> dict[str, Any]:
    if frame.source != "provided_frame":
        raise StoreError(f"{frame.frame_id}: не provided frame")
    if version != 1 or version_info.get("status") != "PRESERVED":
        raise StoreError(f"{frame.frame_id}: provided version должен быть v1 PRESERVED")
    provenance = dict(frame.provided_provenance or {})
    if provenance.get("content_sha256") != actual_info["sha256"]:
        raise StoreError("provided provenance content hash не совпадает с artifact")
    if frame.provided_sha256 != actual_info["sha256"]:
        raise StoreError("provided source SHA не совпадает с artifact")
    if frame.provided_path is None:
        raise StoreError("provided source path отсутствует")
    source = Path(frame.provided_path)
    if not source.is_file() or _sha_file(source) != provenance.get("content_sha256"):
        raise StoreError("provided source artifact изменился; approval невозможен")
    source_verified = _validate_source_provenance(
        store,
        frame,
        provenance,
        actual_content_sha256=actual_info["sha256"],
    )
    return {
        "parents": _parent_links(plan, state, frame),
        "provenance": provenance,
        "source": source_verified,
    }


def _validate_record(
    record: Mapping[str, Any],
    *,
    frame: FrameSpec,
    version: int,
    state: Mapping[str, Any],
    actual_info: Mapping[str, Any],
    actual_path: Path,
    saved: Mapping[str, Any],
    version_info: Mapping[str, Any],
    store_root: Path,
) -> None:
    required_keys = {
        "schema", "approval_id", "frame_id", "version", "plan_sha256",
        "revision_id", "content_sha256", "size_bytes", "width", "height",
        "artifact_path", "stable_binding_sha256", "binding_sha256",
        "bound_workflow_sha256", "attempt_id", "prompt_id", "source_hashes",
        "parent_links", "source_provenance", "owner_ack", "approved_by",
        "approved_at", "quality", "note",
    }
    missing = sorted(key for key in required_keys if key not in record)
    if missing:
        raise StoreError("approval record без обязательных полей: " + ", ".join(missing))
    if record.get("schema") != SCHEMA:
        raise StoreError("approval schema неверна")
    approval_id = record.get("approval_id")
    if not isinstance(approval_id, str) or len(approval_id) != 32 or any(
        char not in "0123456789abcdef" for char in approval_id
    ):
        raise StoreError("approval_id некорректен")
    if record.get("frame_id") != frame.frame_id:
        raise StoreError("approval frame_id не совпадает")
    if record.get("version") != int(version):
        raise StoreError("approval version не совпадает")
    if record.get("plan_sha256") != state.get("plan_sha256"):
        raise StoreError("approval plan_sha256 не совпадает")
    if record.get("revision_id") != state.get("revision_id"):
        raise StoreError("approval revision_id не совпадает")
    if record.get("content_sha256") != actual_info["sha256"]:
        raise StoreError("approval content_sha256 не совпадает")
    if record.get("size_bytes") != actual_info["size_bytes"]:
        raise StoreError("approval size_bytes не совпадает")
    if record.get("width") != actual_info["width"]:
        raise StoreError("approval width не совпадает")
    if record.get("height") != actual_info["height"]:
        raise StoreError("approval height не совпадает")
    expected_rel = actual_path.relative_to(store_root).as_posix()
    if record.get("artifact_path") != expected_rel:
        raise StoreError("approval artifact_path не совпадает")
    if record.get("stable_binding_sha256") != version_info.get("binding_sha256"):
        raise StoreError("approval stable_binding_sha256 не совпадает")
    if record.get("source_hashes") != version_info.get("source_hashes"):
        raise StoreError("approval source_hashes не совпадает")
    if record.get("parent_links") != saved["parents"]:
        raise StoreError("approval parent_links не совпадает")
    if frame.source == "provided_frame":
        if record.get("source_provenance") != saved["source"]:
            raise StoreError("approval source_provenance не совпадает с source state")
        if record.get("binding_sha256") is not None:
            raise StoreError("provided approval binding_sha256 должен быть null")
        if record.get("bound_workflow_sha256") is not None:
            raise StoreError("provided approval graph hash должен быть null")
        if record.get("attempt_id") is not None:
            raise StoreError("provided approval attempt_id должен быть null")
        if record.get("prompt_id") is not None:
            raise StoreError("provided approval prompt_id должен быть null")
    else:
        if record.get("source_provenance") is not None:
            raise StoreError("generated approval source_provenance должен быть null")
        if record.get("binding_sha256") != saved["verified"]["binding_sha256"]:
            raise StoreError("approval full binding_sha256 не совпадает")
        if record.get("bound_workflow_sha256") != saved["intent"].get(
            "bound_workflow_sha256"
        ):
            raise StoreError("approval bound_workflow_sha256 не совпадает")
        if record.get("attempt_id") != version_info.get("attempt_id"):
            raise StoreError("approval attempt_id не совпадает")
        expected_prompt_id = (saved["result"].get("output") or {}).get("prompt_id")
        if record.get("prompt_id") != expected_prompt_id:
            raise StoreError("approval prompt_id не совпадает")
    if record.get("owner_ack") is not True:
        raise StoreError("approval owner_ack должен быть typed True")
    if record.get("approved_by") != "owner":
        raise StoreError("approval approved_by должен быть owner")
    if record.get("quality") != "NOT_RUN":
        raise StoreError("approval quality должен быть NOT_RUN")
    if not isinstance(record.get("note"), str):
        raise StoreError("approval note должен быть строкой")
    approved_at = record.get("approved_at")
    if not isinstance(approved_at, str) or not approved_at.strip():
        raise StoreError("approval approved_at отсутствует")
    try:
        parsed = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StoreError("approval approved_at не ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise StoreError("approval approved_at должен быть timezone-aware")


def validate_approval(
    plan: ResolvedPlan,
    store: PlanStore,
    state: Mapping[str, Any],
    frame_id: str,
    version: int,
    *,
    expected_sha256: str | None = None,
    require_current: bool = True,
) -> dict[str, Any]:
    frame = _frame(plan, frame_id)
    frame_state = state.get("frames", {}).get(frame_id)
    if not isinstance(frame_state, Mapping):
        raise ApprovalRequired(f"{frame_id}: frame state отсутствует")
    if require_current and int(frame_state.get("current_version") or 0) != int(version):
        raise StoreError(f"{frame_id}: версия {version} не является текущей")
    if frame_state.get("invalidated"):
        raise StoreError(f"{frame_id}: кадр invalidated; approval недействителен")
    version_info = frame_state.get("versions", {}).get(str(version))
    if not isinstance(version_info, Mapping):
        raise StoreError(f"{frame_id}: version {version} отсутствует")
    if version_info.get("status") not in {"COMPLETE", "PRESERVED"}:
        raise StoreError(f"{frame_id}: version {version} не COMPLETE/PRESERVED")
    output = version_info.get("output")
    if not isinstance(output, Mapping):
        raise StoreError(f"{frame_id}: output отсутствует")
    actual_path, actual_info = _validate_artifact(
        store, output, expected_sha256=expected_sha256
    )
    if frame.source == "provided_frame":
        saved = _validate_provided(
            plan, store, state, frame, version, version_info, actual_info
        )
    else:
        saved = validate_saved_version(plan, store, state, frame, version, version_info)
    record = store.read_approval(frame_id, version)
    if record is None:
        raise ApprovalRequired(
            f"{frame_id} v{version}: approval отсутствует; сначала keyframes approve "
            f"--sha256 {actual_info['sha256']}"
        )
    _validate_record(
        record,
        frame=frame,
        version=version,
        state=state,
        actual_info=actual_info,
        actual_path=actual_path,
        saved=saved,
        version_info=version_info,
        store_root=store.root,
    )
    return {
        "record": dict(record),
        "artifact_path": actual_path,
        "content_sha256": actual_info["sha256"],
        "binding_sha256": version_info.get("binding_sha256"),
        "saved": saved,
    }


def approve_frame(
    plan: ResolvedPlan,
    store: PlanStore,
    *,
    frame_id: str,
    version: int,
    sha256: str,
    owner_ack: bool,
    note: str | None = None,
) -> dict[str, Any]:
    if owner_ack is not True:
        raise StoreError("approve требует explicit typed --owner-ack")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256.lower())
    ):
        raise StoreError("approve требует exact 64-hex --sha256")
    with store.lock():
        state = store.load_any(plan=plan)
        if state.get("plan_sha256") != plan.plan_sha256:
            raise StoreError("approval state не совпадает с revision")
        frame = _frame(plan, frame_id)
        frame_state = state.get("frames", {}).get(frame_id)
        if not isinstance(frame_state, Mapping):
            raise StoreError(f"{frame_id}: frame state отсутствует")
        if int(frame_state.get("current_version") or 0) != int(version):
            raise StoreError(f"{frame_id}: --version не является текущей")
        if frame_state.get("invalidated"):
            raise StoreError(f"{frame_id}: кадр invalidated")
        version_info = frame_state.get("versions", {}).get(str(version))
        if not isinstance(version_info, Mapping):
            raise StoreError(f"{frame_id}: version отсутствует")
        output = version_info.get("output")
        if not isinstance(output, Mapping):
            raise StoreError(f"{frame_id}: output отсутствует")
        artifact_path, info = _validate_artifact(store, output, expected_sha256=sha256)
        if frame.source == "provided_frame":
            saved = _validate_provided(
                plan, store, state, frame, version, version_info, info
            )
        else:
            saved = validate_saved_version(plan, store, state, frame, version, version_info)
        existing = store.read_approval(frame_id, version)
        if existing is not None:
            try:
                validated = validate_approval(
                    plan,
                    store,
                    state,
                    frame_id,
                    version,
                    expected_sha256=sha256,
                )
            except Exception as exc:
                raise StoreError(
                    "существующий approval не проходит full validation; "
                    "overwrite запрещён"
                ) from exc
            if validated["record"] == existing:
                return {"status": "ALREADY_APPROVED", "approval": existing}
            raise StoreError(
                "approval существует, но отличается от exact current record; "
                "overwrite запрещён"
            )
        record = {
            "schema": SCHEMA,
            "approval_id": uuid.uuid4().hex,
            "frame_id": frame_id,
            "version": int(version),
            "plan_sha256": str(state.get("plan_sha256")),
            "revision_id": str(state.get("revision_id")),
            "content_sha256": info["sha256"],
            "size_bytes": info["size_bytes"],
            "width": info["width"],
            "height": info["height"],
            "artifact_path": artifact_path.relative_to(store.root).as_posix(),
            "stable_binding_sha256": version_info.get("binding_sha256"),
            "binding_sha256": (
                saved["verified"]["binding_sha256"]
                if "verified" in saved
                else None
            ),
            "bound_workflow_sha256": (
                saved["intent"].get("bound_workflow_sha256")
                if "intent" in saved
                else None
            ),
            "attempt_id": version_info.get("attempt_id"),
            "prompt_id": (
                (saved["result"].get("output") or {}).get("prompt_id")
                if "result" in saved
                else None
            ),
            "source_hashes": dict(version_info.get("source_hashes") or {}),
            "parent_links": saved["parents"],
            "source_provenance": (
                saved["source"] if frame.source == "provided_frame" else None
            ),
            "approved_at": _now(),
            "approved_by": "owner",
            "owner_ack": True,
            "note": note or "",
            "quality": "NOT_RUN",
        }
        _validate_record(
            record,
            frame=frame,
            version=version,
            state=state,
            actual_info=info,
            actual_path=artifact_path,
            saved=saved,
            version_info=version_info,
            store_root=store.root,
        )
        path = store.write_approval(frame_id, version, record)
        return {"status": "APPROVED", "approval": record, "path": str(path)}


def approval_status(
    plan: ResolvedPlan,
    store: PlanStore,
    state: Mapping[str, Any],
    frame_id: str,
) -> dict[str, Any]:
    frame_state = state.get("frames", {}).get(frame_id, {})
    if not isinstance(frame_state, Mapping):
        return {"status": "MISSING", "reason": "no frame state"}
    version = int(frame_state.get("current_version") or 0)
    if version <= 0:
        return {"status": "MISSING", "reason": "no current version"}
    version_info = frame_state.get("versions", {}).get(str(version), {})
    if plan.schema_version == 1:
        return {"status": "NOT_APPLICABLE", "version": version, "reason": "v1 legacy"}
    frame = _frame(plan, frame_id)
    if (
        isinstance(version_info, Mapping)
        and version_info.get("status") == "PRESERVED"
        and frame.source != "provided_frame"
    ):
        return {"status": "NOT_APPLICABLE", "version": version, "reason": "first frame"}
    try:
        validated = validate_approval(plan, store, state, frame_id, version)
        return {
            "status": "APPROVED",
            "approval_id": validated["record"].get("approval_id"),
            "version": version,
            "content_sha256": validated["content_sha256"],
            "binding_sha256": validated["binding_sha256"],
            "revision_id": state.get("revision_id"),
            "parent_links": validated["record"].get("parent_links"),
            "approved_at": validated["record"].get("approved_at"),
        }
    except ApprovalRequired as exc:
        return {"status": "MISSING", "version": version, "reason": str(exc)}
    except Exception as exc:
        return {"status": "STALE", "version": version, "reason": str(exc)}


def approval_command(
    plan: ResolvedPlan,
    frame_id: str,
    version: int,
    sha256: str,
) -> str:
    return (
        f'.\\avs.ps1 keyframes approve --plan "{plan.path}" --frame "{frame_id}" '
        f"--version {int(version)} --sha256 {sha256} --owner-ack"
    )


__all__ = [
    "SCHEMA",
    "approve_frame",
    "approval_command",
    "approval_status",
    "validate_saved_version",
    "validate_approval",
]
