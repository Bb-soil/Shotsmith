"""Offline-first KF1 runner with durable intent, resume and explicit retry."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from . import binding as binding_mod
from .approval import approval_command, approval_status, validate_approval, validate_saved_version
from .comfy import (
    AmbiguousDispatch,
    ComfyClient,
    ComfyError,
    PromptLost,
    TransientRemoteError,
    ValidationRejected,
    history_status,
    output_from_history,
)
from .errors import (
    ApprovalRequired,
    BindingError,
    GenerationLimitError,
    StoreError,
    TerminalRemoteFailure,
    V1MigrationRequired,
)
from .normalize import inspect_image, normalize_image, validate_png_bytes
from .plan import FrameSpec, ResolvedPlan
from .runtime_gate import validate_bound_workflow
from .store import PlanStore

ROOT = Path(__file__).resolve().parents[2]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

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
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def _candidate_copy(src: Path, dst: Path, target: tuple[int, int], policy: str) -> dict[str, Any]:
    info = inspect_image(src)
    if int(info["width"]) == target[0] and int(info["height"]) == target[1] and policy in {
        "preserve",
        "contain",
        "crop",
    }:
        data = src.read_bytes()
        _atomic_bytes(dst, data)
        return {
            "policy": "preserve_geometry",
            "source": {"path": str(src), "sha256": info["sha256"], "width": info["width"], "height": info["height"]},
            "output": {"path": str(dst), "sha256": _sha_bytes(data), "width": target[0], "height": target[1]},
            "stretch": False,
        }
    record = normalize_image(
        src,
        dst,
        policy=policy,
        target=list(target),
        pad=(policy == "contain"),
        label=dst.name,
    )
    return record


def _preserve_ref(src: Path, dst: Path, policy: str, target: tuple[int, int]) -> dict[str, Any]:
    info = inspect_image(src)
    if policy == "preserve" and int(info["orientation"]) in {0, 1}:
        data = src.read_bytes()
        _atomic_bytes(dst, data)
        return {
            "policy": "preserve",
            "source": {"path": str(src), "sha256": info["sha256"], "width": info["width"], "height": info["height"]},
            "output": {"path": str(dst), "sha256": _sha_bytes(data), "width": info["width"], "height": info["height"]},
            "stretch": False,
        }
    return normalize_image(
        src, dst, policy=policy if policy != "preserve" else "contain", target=list(target), pad=False,
        label=dst.name,
    )


def _ref_filename_base(ref_id: str) -> str:
    if ref_id == "character_reference":
        return "character"
    if ref_id == "background_reference":
        return "environment"
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in ref_id
    )


_WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _windows_reserved_stem(name: str) -> bool:
    return name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_STEMS


def _hashed_ref_name(ref_id: str, counter: int) -> str:
    material = ref_id if counter == 1 else f"{ref_id}|{counter}"
    return "ref-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def allocate_ref_names(ref_ids: list[str]) -> dict[str, str]:
    """Deterministically allocate distinct safe local reference filenames.

    Legacy names are reserved first so old single-identity state/refs metadata
    stays stable.  Non-colliding custom IDs keep their historical sanitized
    name.  Windows device stems and actual case-insensitive collisions get a
    deterministic ref-ID hash; no valid v2 ID is banned or silently deduped.
    """

    ordered = [
        ref_id
        for ref_id in ref_ids
        if ref_id != "first_frame"
    ]
    ordered.sort(
        key=lambda ref_id: (
            0 if ref_id in {"character_reference", "background_reference"} else 1,
            ref_ids.index(ref_id),
        )
    )
    used: set[str] = set()
    result: dict[str, str] = {}
    for ref_id in ordered:
        base = _ref_filename_base(ref_id)
        counter = 1
        candidate = (
            _hashed_ref_name(ref_id, counter)
            if _windows_reserved_stem(base)
            else base
        )
        while candidate.casefold() in used or _windows_reserved_stem(candidate):
            counter += 1
            candidate = (
                _hashed_ref_name(ref_id, counter)
                if _windows_reserved_stem(base)
                else base[:50] + "-" + hashlib.sha256(
                    (ref_id if counter == 1 else f"{ref_id}|{counter}").encode("utf-8")
                ).hexdigest()[:12]
            )
        used.add(candidate.casefold())
        result[ref_id] = candidate + ".png"
    return result


def _semantic_ref_role(role: str) -> str:
    if role in {"identity", "character", "character_reference"}:
        return "identity"
    if role in {"environment", "background_reference"}:
        return "environment"
    if role in {"style", "style_reference"}:
        return "style"
    return "identity"


class Runner:
    def __init__(
        self,
        plan: ResolvedPlan,
        store: PlanStore,
        client: ComfyClient | None,
        *,
        plan_root: Path | None = None,
        attestation: Mapping[str, Any] | None = None,
    ) -> None:
        self.plan = plan
        self.store = store
        self.client = client
        self.plan_root = plan_root or store.root
        self.attestation = dict(attestation) if isinstance(attestation, Mapping) else None

    # --------------------------------------------------------------- status
    def status(self, state: Mapping[str, Any]) -> dict[str, Any]:
        frames = []
        for frame in self.plan.frames:
            entry = state.get("frames", {}).get(frame.frame_id, {})
            current = entry.get("current_version") if isinstance(entry, Mapping) else None
            version_info = (
                entry.get("versions", {}).get(str(current), {})
                if isinstance(entry, Mapping) and current
                else {}
            )
            attempts = self.store.attempts_for_frame(state, frame.frame_id)
            latest = attempts[-1] if attempts else None
            output = version_info.get("output") if isinstance(version_info, Mapping) else None
            approval = approval_status(self.plan, self.store, state, frame.frame_id)
            approval_command_text = None
            if (
                isinstance(output, Mapping)
                and current
                and isinstance(output.get("sha256"), str)
            ):
                approval_command_text = approval_command(
                    self.plan, frame.frame_id, int(current), str(output["sha256"])
                )
            frames.append(
                {
                    "frame_id": frame.frame_id,
                    "role": frame.role,
                    "generate": frame.generate,
                    "source": frame.source,
                    "current_version": current,
                    "status": version_info.get("status")
                    or (latest.get("status") if latest else "MISSING"),
                    "invalidated": bool(entry.get("invalidated")) if isinstance(entry, Mapping) else False,
                    "output": output,
                    "content_sha256": output.get("sha256") if isinstance(output, Mapping) else None,
                    "binding_sha256": version_info.get("binding_sha256") if isinstance(version_info, Mapping) else None,
                    "shot_mode": frame.shot_mode,
                    "reference_roles": list(frame.reference_roles),
                    "invariants": list(frame.invariants),
                    "allowed_changes": list(frame.allowed_changes),
                    "require_anchor_approval": frame.require_anchor_approval,
                    "approval_status": approval.get("status"),
                    "approval": approval,
                    "approval_command": approval_command_text,
                    "latest_attempt": latest,
                }
            )
        revisions = []
        for revision in self.store.iter_states(plan=self.plan):
            revisions.append(
                {
                    "plan_sha256": revision.get("plan_sha256"),
                    "plan_path": revision.get("plan_path"),
                    "attempts": len(revision.get("attempts", [])),
                    "prompt_ids": self.store.all_prompt_ids(revision),
                    "current": revision.get("plan_sha256") == state.get("plan_sha256"),
                }
            )
        return {
            "plan_sha256": self.plan.plan_sha256,
            "plan_path": str(self.plan.path),
            "schema_version": self.plan.schema_version,
            "generation_limit": self.plan.generation_limit,
            "extra_limit": int(state.get("extra_limit", 0)),
            "attempts_used": self.store.attempt_count(state),
            "frames": frames,
            "prompt_ids": self.store.all_prompt_ids(state),
            "all_prompt_ids": self.store.all_prompt_ids_all(plan=self.plan),
            "revisions": revisions,
        }

    # --------------------------------------------------------------- helpers
    def _frame(self, frame_id: str) -> FrameSpec:
        for frame in self.plan.frames:
            if frame.frame_id == frame_id:
                return frame
        raise StoreError(f"кадр не найден в плане: {frame_id}")

    def _prepare_refs(self) -> dict[str, dict[str, Any]]:
        refs_dir = self.store.root / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        target = tuple(self.plan.output_policy["candidate_geometry"])
        result: dict[str, dict[str, Any]] = {}
        first = self.plan.refs["first_frame"]
        a1_dir = self.store.version_dir(self.plan.frames[0].frame_id, 1)
        a1_dir.mkdir(parents=True, exist_ok=True)
        a1_path = a1_dir / "frame.png"
        if not a1_path.is_file():
            _atomic_bytes(a1_path, first.path.read_bytes())
        if _sha_file(a1_path) != first.sha256:
            raise StoreError("A1 copy hash не совпадает с исходным first_frame")
        candidate = a1_dir / "candidate.png"
        result["first_frame"] = {
            "original": {
                "path": str(first.path),
                "sha256": first.sha256,
                "width": first.width,
                "height": first.height,
                "format": first.format,
            },
            "frame": {"path": str(a1_path), "sha256": _sha_file(a1_path), "width": first.width, "height": first.height},
            "candidate": _candidate_copy(
                a1_path,
                candidate,
                target,
                str(self.plan.output_policy["anchor_policy"]),
            ),
        }
        names = allocate_ref_names(list(self.plan.refs))
        for ref_id, info in self.plan.refs.items():
            if ref_id == "first_frame":
                continue
            semantic = _semantic_ref_role(info.role)
            if semantic == "identity":
                policy = str(self.plan.output_policy["character_policy"])
            elif semantic == "environment":
                policy = str(self.plan.output_policy["environment_policy"])
            else:
                policy = str(self.plan.output_policy["character_policy"])
            out = refs_dir / names[ref_id]
            result[ref_id] = _preserve_ref(
                info.path,
                out,
                policy,
                target,
            )
        _atomic_json(refs_dir / "refs.json", result)
        return result

    def _anchor_candidate(
        self,
        state: Mapping[str, Any],
        frame: FrameSpec,
        refs: Mapping[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        target = tuple(self.plan.output_policy["candidate_geometry"])
        if frame.anchor_frame is None:
            raise StoreError(f"{frame.frame_id}: нет anchor")
        anchor_frame = self._frame(frame.anchor_frame)
        frame_state = state.get("frames", {}).get(anchor_frame.frame_id, {})
        current = frame_state.get("current_version") if isinstance(frame_state, Mapping) else None
        if isinstance(frame_state, Mapping) and frame_state.get("invalidated"):
            raise StoreError(
                f"{frame.frame_id}: anchor {anchor_frame.frame_id} invalidated; "
                "сначала regenerate/accept valid anchor"
            )
        info = (
            frame_state.get("versions", {}).get(str(current), {})
            if isinstance(frame_state, Mapping) and current
            else {}
        )
        if isinstance(info, Mapping) and info.get("status") in {"COMPLETE", "PRESERVED"}:
            output = info.get("output")
            if not isinstance(output, Mapping) or not output.get("path"):
                raise StoreError(f"{frame.frame_id}: anchor {anchor_frame.frame_id} не готов")
            source = Path(str(output["path"]))
            expected_sha = str(output.get("sha256") or "")
        elif not anchor_frame.generate:
            anchor = refs["first_frame"]
            source = Path(anchor["frame"]["path"])
            expected_sha = anchor["frame"]["sha256"]
        else:
            raise StoreError(f"{frame.frame_id}: anchor {anchor_frame.frame_id} не COMPLETE")
        if not source.is_file():
            raise StoreError(f"{frame.frame_id}: anchor file отсутствует")
        actual = _sha_file(source)
        if actual != expected_sha:
            raise StoreError(f"{frame.frame_id}: anchor hash изменился")
        dst = (
            self.store.root
            / "candidates"
            / (frame.frame_id.replace("/", "_") + f"-anchor-{actual[:12]}.png")
        )
        record = _candidate_copy(
            source,
            dst,
            target,
            str(self.plan.output_policy["anchor_policy"]),
        )
        record = dict(record)
        record["parent_output_sha256"] = expected_sha
        return dst, record

    def _ref_paths(
        self, refs: Mapping[str, Any], roles: list[str]
    ) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for role in roles:
            if role == "anchor":
                continue
            if role not in refs:
                raise BindingError(f"Нет local reference для роли {role}")
            result[role] = Path(refs[role]["output"]["path"])
        return result

    def _lineage_parent(
        self, state: Mapping[str, Any], frame: FrameSpec
    ) -> dict[str, Any] | None:
        if not frame.anchor_frame:
            return None
        anchor = self._frame(frame.anchor_frame)
        anchor_state = state.get("frames", {}).get(anchor.frame_id, {})
        if not isinstance(anchor_state, Mapping):
            raise StoreError(f"{frame.frame_id}: parent state отсутствует")
        if anchor_state.get("invalidated"):
            raise StoreError(f"{frame.frame_id}: parent {anchor.frame_id} invalidated")
        version = int(anchor_state.get("current_version") or 0)
        info = anchor_state.get("versions", {}).get(str(version), {})
        if not isinstance(info, Mapping) or info.get("status") not in {
            "COMPLETE",
            "PRESERVED",
        }:
            raise StoreError(f"{frame.frame_id}: parent {anchor.frame_id} не готов")
        output = info.get("output")
        if not isinstance(output, Mapping) or not output.get("path"):
            raise StoreError(f"{frame.frame_id}: parent output отсутствует")
        return {
            "frame_id": anchor.frame_id,
            "version": version,
            "status": info.get("status"),
            "content_sha256": str(output.get("sha256") or ""),
            "binding_sha256": str(info.get("binding_sha256") or ""),
            "width": int(output.get("width") or 0),
            "height": int(output.get("height") or 0),
            "source": anchor.source,
            "provenance": (
                dict(self.plan.provided_frames.get(anchor.frame_id, {}).get("provenance") or {})
                if anchor.source == "provided_frame"
                else {}
            ),
        }

    def _source_hashes(
        self,
        refs: Mapping[str, Any],
        *,
        roles: list[str],
        lineage: Mapping[str, Any] | None,
        anchor_record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = dict(anchor_record or {})
        source = record.get("source") if isinstance(record.get("source"), Mapping) else {}
        output = record.get("output") if isinstance(record.get("output"), Mapping) else {}
        result: dict[str, Any] = {
            "plan_sha256": self.plan.plan_sha256,
            "revision_id": self.plan.plan_sha256,
            "pins_sha256": _sha_bytes(
                json.dumps(
                    self.plan.pins, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            ),
        }
        if anchor_record is not None:
            anchor_sha = str(output.get("sha256") or "")
            result.update(
                {
                    "anchor": anchor_sha,
                    "anchor_version": lineage.get("version") if lineage else 1,
                    "anchor_source_sha256": str(
                        record.get("parent_output_sha256")
                        or (source or {}).get("sha256")
                        or anchor_sha
                    ),
                    "anchor_derivative_sha256": anchor_sha,
                    "anchor_policy": str(record.get("policy") or "preserve_geometry"),
                    "anchor_geometry": [
                        int(output.get("width") or 0),
                        int(output.get("height") or 0),
                    ],
                    "anchor_uploaded": True,
                }
            )
        elif lineage is not None:
            result.update(
                {
                    "anchor": str(lineage.get("content_sha256") or ""),
                    "anchor_version": int(lineage.get("version") or 1),
                    "anchor_source_sha256": str(lineage.get("content_sha256") or ""),
                    "anchor_derivative_sha256": str(lineage.get("content_sha256") or ""),
                    "anchor_policy": "lineage_only",
                    "anchor_geometry": [
                        int(lineage.get("width") or 0),
                        int(lineage.get("height") or 0),
                    ],
                    "anchor_uploaded": False,
                }
            )
            if lineage.get("provenance"):
                result["anchor_provenance"] = dict(lineage["provenance"])
        for role in roles:
            if role == "anchor":
                continue
            if role not in refs:
                raise BindingError(f"Нет normalized reference hash для роли {role}")
            result[role] = str(refs[role]["output"]["sha256"])
        return result

    def _output_valid(self, output: Mapping[str, Any]) -> bool:
        path = Path(str(output.get("path") or ""))
        if not path.is_file():
            return False
        try:
            info = validate_png_bytes(path.read_bytes())
        except Exception:
            return False
        return (
            info["sha256"] == output.get("sha256")
            and info["width"] == output.get("width")
            and info["height"] == output.get("height")
        )

    def _latest_attempt(self, state: Mapping[str, Any], frame_id: str) -> dict[str, Any] | None:
        attempts = self.store.attempts_for_frame(state, frame_id)
        return attempts[-1] if attempts else None

    def _recover_local_receipt(
        self,
        state: dict[str, Any],
        frame: FrameSpec,
        latest: Mapping[str, Any] | None,
        expected_hash: str,
        source_hashes: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Recover only the exact current intended attempt's receipt.

        Older COMPLETE receipts must never mask a newer retry that is still
        POSTING/SUBMITTED/AMBIGUOUS/FAILED.  If the latest attempt has no own
        receipt, the normal exact prompt-id resume path handles it.
        """

        if latest is None:
            return None
        attempt_id = str(latest.get("attempt_id") or "")
        version = int(latest.get("version", 0) or 0)
        if not attempt_id or version <= 0:
            return None
        receipt_path = self.store.version_dir(frame.frame_id, version) / "result.json"
        if not receipt_path.is_file():
            return None
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(receipt, Mapping):
            return None
        if (
            receipt.get("attempt_id") != attempt_id
            or int(receipt.get("version", 0) or 0) != version
            or receipt.get("plan_sha256") != state.get("plan_sha256")
            or receipt.get("binding_sha256") != expected_hash
            or receipt.get("source_hashes") != dict(source_hashes)
        ):
            if latest.get("status") in {"POSTING", "SUBMITTED"}:
                raise StoreError(
                    f"{frame.frame_id}: result receipt не принадлежит текущей "
                    "attempt/version/revision; новый POST запрещён"
                )
            return None
        output = receipt.get("output")
        if not isinstance(output, Mapping) or not self._output_valid(output):
            if latest.get("status") in {"POSTING", "SUBMITTED"}:
                raise StoreError(
                    f"{frame.frame_id}: result receipt output невалиден; новый POST запрещён"
                )
            return None
        self.store.complete_attempt(
            state,
            attempt_id,
            output=dict(output),
            binding_sha256=expected_hash,
            source_hashes=dict(source_hashes),
            receipt=dict(receipt),
        )
        return {
            "frame_id": frame.frame_id,
            "status": "COMPLETE",
            "version": version,
            "output": dict(output),
            "recovered_receipt": True,
        }

    def _invalidate_downstream(self, state: dict[str, Any], frame_id: str, reason: str) -> list[str]:
        ids = [frame.frame_id for frame in self.plan.frames]
        if frame_id not in ids:
            return []
        index = ids.index(frame_id)
        downstream = [candidate for candidate in ids[index + 1 :] if self._frame(candidate).generate]
        if downstream:
            self.store.mark_invalidated(state, downstream, reason)
        return downstream

    def _frame_seed(self, frame):
        base = int(self.plan.settings["seed"])
        if self.plan.data.get("seed_policy", "fixed") == "fixed":
            return base
        index = next(i for i, item in enumerate(self.plan.frames) if item.frame_id == frame.frame_id)
        return (base + index + 1) % 2147483648

    def _stable_expected(
        self,
        frame: FrameSpec,
        roles: list[str],
        source_hashes: Mapping[str, Any],
        parent_version: int | None = None,
    ) -> tuple[str, str]:
        if not frame.generate:
            raise BindingError(f"{frame.frame_id}: кадр не генерируется")
        if not frame.edit_instruction:
            raise BindingError(f"{frame.frame_id}: нет edit_instruction")
        if self.plan.schema_version == 2:
            if frame.shot_mode not in {"continue_shot", "new_shot"}:
                raise BindingError(f"{frame.frame_id}: нет explicit shot_mode")
            mode = frame.shot_mode
        else:
            if frame.role == "B":
                mode = "b"
            else:
                mode = "a_new"
        role_map = {
            role: ("anchor" if role == "anchor" else self.plan.refs[role].role)
            for role in roles
        }
        digest, _positive, _workflow_sha, _negative = binding_mod.stable_binding_hash(
            ref_count=len(roles),
            roles=roles,
            edit_instruction=frame.edit_instruction,
            description=frame.description,
            style=str(self.plan.data.get("style") or ""),
            shot_mode=mode if self.plan.schema_version == 2 else None,
            mode=None if self.plan.schema_version == 2 else mode,
            invariants=frame.invariants,
            allowed_changes=frame.allowed_changes,
            subject_label=self.plan.subject_label,
            negative_extra=frame.negative_prompt,
            role_map=role_map,
            seed=self._frame_seed(frame),
            candidate_geometry=tuple(self.plan.output_policy["candidate_geometry"]),
            plan_sha256=self.plan.plan_sha256,
            parent_version=parent_version,
            settings=self.plan.settings,
        )
        return digest, mode

    def _ensure_a1_state(self, state: dict[str, Any], refs: Mapping[str, Any]) -> None:
        frame = self.plan.frames[0]
        frame_state = state.setdefault("frames", {}).setdefault(
            frame.frame_id, {"current_version": 0, "versions": {}}
        )
        frame_state["current_version"] = max(1, int(frame_state.get("current_version") or 0))
        frame_state.setdefault("versions", {})["1"] = {
            "status": "PRESERVED",
            "output": dict(refs["first_frame"]["frame"]),
            "candidate": dict(refs["first_frame"]["candidate"]),
            "original": dict(refs["first_frame"]["original"]),
            "source_hashes": {"original": refs["first_frame"]["original"]["sha256"]},
        }
        self.store.save(state)

    def _ensure_provided_state(self, state: dict[str, Any], frame: FrameSpec) -> None:
        if frame.provided_path is None or frame.provided_sha256 is None:
            raise StoreError(f"{frame.frame_id}: provided frame metadata отсутствует")
        source = Path(frame.provided_path)
        actual = _sha_file(source)
        if actual != frame.provided_sha256:
            raise StoreError(
                f"{frame.frame_id}: provided source hash изменился "
                f"{frame.provided_sha256} != {actual}"
            )
        version_dir = self.store.ensure_version_dir(frame.frame_id, 1)
        target = version_dir / "frame.png"
        if target.is_file():
            existing = _sha_file(target)
            if existing != actual:
                raise StoreError(
                    f"{frame.frame_id}: existing provided copy hash не совпадает"
                )
        else:
            _atomic_bytes(target, source.read_bytes())
        if _sha_file(target) != actual:
            raise StoreError(f"{frame.frame_id}: provided atomic finalize hash mismatch")
        frame_state = state.setdefault("frames", {}).setdefault(
            frame.frame_id, {"current_version": 0, "versions": {}}
        )
        frame_state["current_version"] = max(1, int(frame_state.get("current_version") or 0))
        info = frame.provided_provenance or {}
        frame_state.setdefault("versions", {})["1"] = {
            "status": "PRESERVED",
            "output": {
                "path": str(target),
                "sha256": actual,
                "size_bytes": target.stat().st_size,
                "width": int(info.get("width") or 0) or None,
                "height": int(info.get("height") or 0) or None,
                "format": "PNG",
            },
            "original": {
                "path": str(source),
                "sha256": actual,
                "size_bytes": source.stat().st_size,
            },
            "provenance": dict(info),
            "source_hashes": {
                "original": actual,
                "provided_provenance": dict(info),
            },
        }
        # Fill actual geometry from the copied PNG (the plan already verified it).
        from .normalize import inspect_image

        decoded = inspect_image(target)
        frame_state["versions"]["1"]["output"].update(
            {
                "width": int(decoded["width"]),
                "height": int(decoded["height"]),
                "size_bytes": int(decoded["size_bytes"]),
            }
        )
        self.store.save(state)

    def _validate_saved_intent(
        self,
        state: Mapping[str, Any],
        frame: FrameSpec,
        version: int,
        attempt: Mapping[str, Any],
    ) -> dict[str, Any]:
        version_dir = self.store.version_dir(frame.frame_id, version)
        binding_path = version_dir / "binding.json"
        intent_path = version_dir / "intent.json"
        if not binding_path.is_file() or not intent_path.is_file():
            raise StoreError(
                f"{frame.frame_id} v{version}: saved binding/intent отсутствует; "
                "новая генерация v1 запрещена"
            )
        try:
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"{frame.frame_id}: saved binding/intent повреждён") from exc
        if not isinstance(binding, Mapping) or not isinstance(intent, Mapping):
            raise StoreError(f"{frame.frame_id}: saved binding/intent не объект")
        if attempt.get("frame_id") != frame.frame_id:
            raise StoreError("attempt frame_id не совпадает")
        if int(attempt.get("version") or 0) != int(version):
            raise StoreError("attempt version не совпадает")
        parent_version = 1
        if frame.anchor_frame:
            parent_version = int(
                state.get("frames", {}).get(frame.anchor_frame, {}).get("current_version") or 1
            )
        workflow_pin = self.plan.pins.get("workflows", {}).get(str(len(binding.get("roles", []))), {})
        expected_workflow = (
            workflow_pin.get("sha256") if isinstance(workflow_pin, Mapping) else None
        )
        verified = binding_mod.verify_saved_binding(
            binding,
            plan_sha256=self.plan.plan_sha256,
            parent_version=parent_version,
            candidate_geometry=tuple(self.plan.output_policy["candidate_geometry"]),
            expected_workflow_sha256=expected_workflow,
        )
        if verified["stable_binding_sha256"] != attempt.get("binding_sha256"):
            raise StoreError("attempt binding_sha256 не совпадает с actual binding")
        if verified["stable_binding_sha256"] != intent.get("binding_sha256"):
            raise StoreError("saved intent binding hash не совпадает")
        if binding.get("positive_prompt") != intent.get("prompt"):
            raise StoreError("saved intent prompt не совпадает")
        if binding.get("positive_prompt") != attempt.get("prompt"):
            raise StoreError("attempt prompt не совпадает")
        if binding.get("negative_prompt") != intent.get("negative_prompt"):
            raise StoreError("saved intent negative не совпадает")
        if binding.get("negative_prompt") != attempt.get("negative_prompt"):
            raise StoreError("attempt negative не совпадает")
        if binding.get("settings") != intent.get("settings"):
            raise StoreError("saved intent settings не совпадает")
        if binding.get("settings") != attempt.get("settings"):
            raise StoreError("attempt settings не совпадает")
        binding_seed = binding.get("seed")
        intent_seed = intent.get("seed")
        attempt_seed = attempt.get("seed")
        for value, name in (
            (binding_seed, "binding.seed"),
            (intent_seed, "intent.seed"),
            (attempt_seed, "attempt.seed"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise StoreError(f"saved {name} должен быть int")
        if not (binding_seed == intent_seed == attempt_seed):
            raise StoreError("saved seed не совпадает")
        graph_sha = _sha_bytes(
            json.dumps(
                binding["bound_workflow"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        if intent.get("bound_workflow_sha256") != graph_sha:
            raise StoreError("saved intent graph hash не совпадает")
        if attempt.get("bound_workflow_sha256") != graph_sha:
            raise StoreError("attempt graph hash не совпадает")
        if intent.get("workflow_sha256") != verified["workflow_sha256"]:
            raise StoreError("saved intent workflow hash не совпадает")
        if attempt.get("workflow_sha256") != verified["workflow_sha256"]:
            raise StoreError("attempt workflow hash не совпадает")
        if intent.get("source_hashes") != attempt.get("source_hashes"):
            raise StoreError("attempt source_hashes не совпадают с intent")
        source_hashes = attempt.get("source_hashes")
        if not isinstance(source_hashes, Mapping):
            raise StoreError("attempt source_hashes отсутствуют")
        pins_sha = _sha_bytes(
            json.dumps(
                self.plan.pins,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        if source_hashes.get("pins_sha256") != pins_sha:
            raise StoreError("attempt source_hashes.pins_sha256 не совпадает с plan pins")
        if source_hashes.get("plan_sha256") != self.plan.plan_sha256:
            raise StoreError("attempt source_hashes.plan_sha256 не совпадает с plan")
        if source_hashes.get("revision_id") != self.plan.plan_sha256:
            raise StoreError("attempt source_hashes.revision_id не совпадает с plan")
        if intent.get("uploads") != attempt.get("uploads"):
            raise StoreError("attempt uploads не совпадают с intent")
        if intent.get("expected_output") != attempt.get("expected_output"):
            raise StoreError("attempt expected_output не совпадает с intent")
        return {"binding": dict(binding), "intent": dict(intent), "verified": verified}

    def _run_v1_recovery_or_block(
        self,
        state: dict[str, Any],
        frame: FrameSpec,
        *,
        deadline_seconds: float,
        emit: Callable[[str], None],
    ) -> dict[str, Any]:
        frame_state = state.get("frames", {}).get(frame.frame_id, {})
        if not isinstance(frame_state, Mapping):
            raise V1MigrationRequired(
                f"{frame.frame_id}: v1 fresh generation запрещена. "
                "Создайте v2-копию плана с explicit shot_mode/invariants/"
                "allowed_changes/reference_roles; старый план/state не перезаписываются."
            )
        current = int(frame_state.get("current_version") or 0)
        current_info = frame_state.get("versions", {}).get(str(current), {}) if current else {}
        latest = self._latest_attempt(state, frame.frame_id)

        # The latest durable attempt owns the frame state.  An older COMPLETE
        # version must never hide a newer retry that is SUBMITTED/POSTING/etc.
        if latest is None:
            if (
                current
                and isinstance(current_info, Mapping)
                and current_info.get("status") in {"COMPLETE", "PRESERVED"}
                and self._output_valid(current_info.get("output") or {})
            ):
                if current_info.get("status") == "COMPLETE":
                    validate_saved_version(
                        self.plan, self.store, state, frame, current, current_info
                    )
                return {"frame_id": frame.frame_id, "status": "SKIPPED", "version": current}
            raise V1MigrationRequired(
                f"{frame.frame_id}: v1 fresh generation запрещена. "
                "Создайте v2-копию плана; старый план/state не перезаписываются."
            )

        status = str(latest.get("status") or "")
        version = int(latest.get("version") or 0)
        if version <= 0:
            raise StoreError(f"{frame.frame_id}: attempt без version; fail closed")
        if status == "COMPLETE":
            if (
                current
                and isinstance(current_info, Mapping)
                and current_info.get("status") in {"COMPLETE", "PRESERVED"}
                and self._output_valid(current_info.get("output") or {})
            ):
                if current_info.get("status") == "COMPLETE":
                    validate_saved_version(
                        self.plan, self.store, state, frame, current, current_info
                    )
                return {"frame_id": frame.frame_id, "status": "SKIPPED", "version": current}
            raise V1MigrationRequired(
                f"{frame.frame_id}: latest v1 COMPLETE не является current; "
                "fresh v1 generation запрещена, создайте v2-копию."
            )
        if status in {"SUBMITTED", "POSTING", "AMBIGUOUS"}:
            self._validate_saved_intent(state, frame, version, latest)
            prompt_id = latest.get("prompt_id")
            if isinstance(prompt_id, str) and prompt_id:
                if status == "SUBMITTED":
                    return self._poll_and_finalize(
                        state,
                        frame,
                        latest,
                        prompt_id,
                        self.store.version_dir(frame.frame_id, version),
                        dict(latest.get("source_hashes") or {}),
                        str(latest.get("binding_sha256") or ""),
                        deadline_seconds,
                        emit,
                    )
                recovered = self._recover_unknown_ack(
                    state, frame, latest, deadline_seconds, emit
                )
                if recovered is not None:
                    return recovered
                raise AmbiguousDispatch(
                    f"{frame.frame_id}: v1 prompt_id не восстановлен; новый POST запрещён"
                )
            recovered = self._recover_unknown_ack(
                state, frame, latest, deadline_seconds, emit
            )
            if recovered is not None:
                return recovered
            raise AmbiguousDispatch(
                f"{frame.frame_id}: v1 attempt без prompt_id; AMBIGUOUS, новый POST запрещён"
            )
        if status == "INTENT":
            self._validate_saved_intent(state, frame, version, latest)
            self._recover_unknown_ack(state, frame, latest, deadline_seconds, emit)
            raise V1MigrationRequired(
                f"{frame.frame_id}: v1 INTENT был pre-send; fresh v1 POST запрещён. "
                "Создайте v2-копию плана; старый state сохранён."
            )
        if status == "FAILED_TERMINAL":
            raise TerminalRemoteFailure(
                f"{frame.frame_id}: terminal failure; v1 retry требует v2 migration"
            )
        raise V1MigrationRequired(
            f"{frame.frame_id}: неизвестный v1 attempt status {status!r}; "
            "fresh v1 POST запрещён, нужен v2 migration."
        )

    # --------------------------------------------------------------- generate
    def run(
        self,
        *,
        execute: bool,
        deadline_seconds: float = 900.0,
        progress: Callable[[str], None] | None = None,
        raise_limit: bool = False,
        stop_after: str | None = None,
    ) -> dict[str, Any]:
        if not execute:
            raise StoreError("run(execute=False) не должен вызываться; используйте preview")
        if self.client is None:
            raise StoreError("нет ComfyClient")
        if stop_after is not None and stop_after not in {
            frame.frame_id for frame in self.plan.frames
        }:
            raise StoreError(f"unknown --until-frame: {stop_after}")
        emit = progress or (lambda _line: None)
        with self.store.lock():
            state = self.store.load(plan=self.plan)
            if self.attestation is not None:
                state["runtime_attestation"] = dict(self.attestation)
            if raise_limit:
                state["extra_limit"] = int(state.get("extra_limit", 0)) + 1
                self.store.save(state)
            refs = self._prepare_refs()
            self._ensure_a1_state(state, refs)
            self.store.save(state)
            summaries = []
            scene_order = [str(scene.get("id")) for scene in self.plan.data["scenes"]]
            total = len(scene_order)
            for frame in self.plan.frames:
                scene_number = scene_order.index(frame.scene_id) + 1
                if not frame.generate:
                    if frame.source == "provided_frame":
                        self._ensure_provided_state(state, frame)
                        emit(
                            f"сцена {scene_number}/{total} — {frame.scene_id} — "
                            f"{frame.role} — provided_frame PRESERVED (0 POST)"
                        )
                    else:
                        emit(
                            f"сцена {scene_number}/{total} — {frame.scene_id} — A — "
                            "сохранён provided first_frame"
                        )
                    summaries.append(
                        {"frame_id": frame.frame_id, "status": "PRESERVED", "version": 1}
                    )
                    if stop_after == frame.frame_id:
                        self.store.save(state)
                        return {
                            "status": "STOPPED_AFTER",
                            "stop_after": stop_after,
                            "frames": summaries,
                            "state": self.status(state),
                        }
                    continue
                emit(
                    f"сцена {scene_number}/{total} — {frame.scene_id} — "
                    f"{frame.role} — проверка/очередь"
                )
                result = self._run_generated_frame(
                    state, frame, refs, deadline_seconds=deadline_seconds, force_new_version=False,
                    emit=emit,
                )
                summaries.append(result)
                if result.get("status") == "WAITING_APPROVAL":
                    self.store.save(state)
                    return {
                        "status": "WAITING_APPROVAL",
                        "stop_after": stop_after,
                        "frame_id": frame.frame_id,
                        "frames": summaries,
                        "state": self.status(state),
                    }
                if result.get("status") == "AMBIGUOUS":
                    raise AmbiguousDispatch(
                        f"{frame.frame_id}: POST принят неизвестно; автоматический retry запрещён"
                    )
                if stop_after == frame.frame_id:
                    self.store.save(state)
                    return {
                        "status": "STOPPED_AFTER",
                        "stop_after": stop_after,
                        "frames": summaries,
                        "state": self.status(state),
                    }
            self.store.save(state)
            return {"status": "COMPLETE", "frames": summaries, "state": self.status(state)}

    def _run_generated_frame(
        self,
        state: dict[str, Any],
        frame: FrameSpec,
        refs: Mapping[str, Any],
        *,
        deadline_seconds: float,
        force_new_version: bool,
        emit: Callable[[str], None],
    ) -> dict[str, Any]:
        if self.plan.schema_version == 1:
            return self._run_v1_recovery_or_block(
                state, frame, deadline_seconds=deadline_seconds, emit=emit
            )
        if self.client is None:
            raise StoreError("нет ComfyClient")
        roles = list(frame.reference_roles)
        if not roles or len(roles) > 3:
            raise BindingError(f"{frame.frame_id}: нужно 1..3 reference")
        if len(roles) != len(set(roles)):
            raise BindingError(f"{frame.frame_id}: повторяющаяся роль reference")
        if "anchor" in roles and roles[0] != "anchor":
            raise BindingError(f"{frame.frame_id}: anchor должен быть Picture 1")
        if frame.role == "B" and "anchor" not in roles:
            raise BindingError(f"{frame.frame_id}: B требует anchor")
        ref_paths = self._ref_paths(refs, roles)
        lineage = self._lineage_parent(state, frame)
        parent_version = int(lineage["version"]) if lineage else 1
        anchor_path: Path | None = None
        anchor_record: dict[str, Any] | None = None
        if "anchor" in roles:
            anchor_path, anchor_record = self._anchor_candidate(state, frame, refs)
        source_hashes = self._source_hashes(
            refs,
            roles=roles,
            lineage=lineage,
            anchor_record=anchor_record,
        )
        expected_hash, mode = self._stable_expected(
            frame, roles, source_hashes, parent_version
        )

        frame_state = state.setdefault("frames", {}).setdefault(
            frame.frame_id, {"current_version": 0, "versions": {}}
        )
        latest = self._latest_attempt(state, frame.frame_id)
        if not force_new_version:
            receipt_result = self._recover_local_receipt(
                state, frame, latest, expected_hash, source_hashes
            )
            if receipt_result is not None:
                return receipt_result
        if latest and latest.get("status") in {"INTENT", "POSTING"}:
            recovered = self._recover_unknown_ack(
                state, frame, latest, deadline_seconds, emit
            )
            if recovered is not None:
                return recovered
        if latest and latest.get("status") == "AMBIGUOUS" and not force_new_version:
            raise AmbiguousDispatch(
                f"{frame.frame_id}: предыдущий POST AMBIGUOUS; нужен explicit retry-frame --owner-ack"
            )
        if latest and latest.get("status") == "FAILED_TERMINAL" and not force_new_version:
            raise TerminalRemoteFailure(
                f"{frame.frame_id}: terminal remote failure; quality auto-retry запрещён"
            )
        if (
            latest
            and latest.get("status") == "SUBMITTED"
            and isinstance(latest.get("prompt_id"), str)
            and latest.get("prompt_id")
            and not force_new_version
        ):
            return self._poll_and_finalize(
                state,
                frame,
                latest,
                str(latest["prompt_id"]),
                self.store.version_dir(frame.frame_id, int(latest["version"])),
                dict(latest.get("source_hashes") or {}),
                str(latest.get("binding_sha256") or ""),
                deadline_seconds,
                emit,
            )
        current_version = int(frame_state.get("current_version") or 0)
        current_info = frame_state.get("versions", {}).get(str(current_version), {})
        if (
            not force_new_version
            and not frame_state.get("invalidated")
            and isinstance(current_info, Mapping)
            and current_info.get("status") == "COMPLETE"
            and current_info.get("binding_sha256") == expected_hash
            and current_info.get("source_hashes") == source_hashes
            and self._output_valid(current_info.get("output") or {})
        ):
            emit(f"сцена {frame.scene_id} — {frame.role} — готово, skip (0 POST)")
            return {"frame_id": frame.frame_id, "status": "SKIPPED", "version": current_version}
        if frame.require_anchor_approval:
            if not frame.anchor_frame:
                raise ApprovalRequired(
                    f"{frame.frame_id}: require_anchor_approval без parent"
                )
            anchor_frame = self._frame(frame.anchor_frame)
            anchor_state = state.get("frames", {}).get(anchor_frame.frame_id, {})
            anchor_version = int(anchor_state.get("current_version") or 0)
            try:
                validate_approval(
                    self.plan,
                    self.store,
                    state,
                    anchor_frame.frame_id,
                    anchor_version,
                )
            except ApprovalRequired as exc:
                return {
                    "frame_id": frame.frame_id,
                    "status": "WAITING_APPROVAL",
                    "anchor_frame": anchor_frame.frame_id,
                    "anchor_version": anchor_version,
                    "message": str(exc),
                }
        allow_attempt_ids: set[str] = set()
        if (
            force_new_version
            and isinstance(latest, Mapping)
            and latest.get("status") == "AMBIGUOUS"
            and latest.get("frame_id") == frame.frame_id
            and isinstance(latest.get("attempt_id"), str)
        ):
            allow_attempt_ids.add(str(latest["attempt_id"]))
        self.store.assert_global_clean_for_new_work(
            allow_attempt_ids=allow_attempt_ids
        )
        needs_new = (
            force_new_version
            or current_version == 0
            or not isinstance(current_info, Mapping)
            or current_info.get("status") != "COMPLETE"
            or bool(frame_state.get("invalidated"))
            or current_info.get("binding_sha256") != expected_hash
            or current_info.get("source_hashes") != source_hashes
            or not self._output_valid(current_info.get("output") or {})
        )
        if needs_new:
            version = self.store.next_version(frame.frame_id)
        else:
            version = current_version
        version_dir = self.store.ensure_version_dir(frame.frame_id, version)
        if anchor_path is not None:
            materialized_anchor = version_dir / "anchor.png"
            if anchor_path != materialized_anchor:
                _atomic_bytes(materialized_anchor, anchor_path.read_bytes())
            anchor_path = materialized_anchor
        if current_version and isinstance(current_info, Mapping) and current_info.get("status") == "COMPLETE" and (
            current_info.get("binding_sha256") != expected_hash
            or current_info.get("source_hashes") != source_hashes
        ):
            self._invalidate_downstream(state, frame.frame_id, "anchor/input/binding changed")
            self.store.clear_invalidated(state, frame.frame_id)
        if force_new_version:
            self._invalidate_downstream(state, frame.frame_id, "explicit retry-frame")
        self.store.clear_invalidated(state, frame.frame_id)

        emit(f"сцена {frame.scene_id} — {frame.role} — upload refs")
        uploads = []
        image_names = {}
        upload_receipts: dict[str, dict[str, Any]] = {}
        for role in roles:
            if role == "anchor":
                if anchor_path is None:
                    raise BindingError(f"{frame.frame_id}: anchor path отсутствует")
                path = anchor_path
            else:
                path = ref_paths[role]
            remote = (
                f"kf1_{self.plan.plan_sha256[:12]}_{frame.scene_id.replace('/', '_')}_"
                f"{frame.role}_{role}_v{version}.png"
            )
            uploaded = self.client.upload_png(path, remote_name=remote)
            if not isinstance(uploaded, Mapping) or not uploaded.get("name"):
                raise StoreError("upload не вернул exact receipt")
            source_sha = _sha_file(path)
            if str(uploaded.get("source_sha256") or "") != source_sha:
                raise StoreError("upload receipt source hash не совпадает с файлом")
            receipt = {"role": role, **dict(uploaded), "source_sha256": source_sha}
            uploads.append(receipt)
            upload_receipts[str(uploaded["name"])] = dict(uploaded)
            image_names[role] = str(uploaded["name"])
        bound, binding_meta = binding_mod.bind(
            ref_count=len(roles),
            image_names=image_names,
            edit_instruction=frame.edit_instruction or "",
            description=frame.description,
            style=str(self.plan.data.get("style") or ""),
            roles=roles,
            shot_mode=frame.shot_mode,
            invariants=frame.invariants,
            allowed_changes=frame.allowed_changes,
            subject_label=self.plan.subject_label,
            negative_extra=frame.negative_prompt,
            role_map={
                role: ("anchor" if role == "anchor" else self.plan.refs[role].role)
                for role in roles
            },
            seed=self._frame_seed(frame),
            version=version,
            plan_sha256=self.plan.plan_sha256,
            scene_id=frame.scene_id,
            role=frame.role,
            candidate_geometry=tuple(self.plan.output_policy["candidate_geometry"]),
            parent_version=parent_version,
            settings=self.plan.settings,
        )
        stable_hash = str(binding_meta["stable_binding_sha256"])
        if stable_hash != expected_hash:
            raise BindingError("binding hash нестабилен до/после upload")
        object_info = self.client.object_info()
        gate = validate_bound_workflow(
            bound,
            object_info,
            upload_receipts=upload_receipts,
            expected_output=binding_meta.get("expected_output"),
        )
        binding_meta["uploads"] = uploads
        binding_meta["runtime_gate"] = gate
        binding_meta["bound_workflow"] = bound
        bound_workflow_sha256 = _sha_bytes(
            json.dumps(
                bound, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        binding_meta["bound_workflow_sha256"] = bound_workflow_sha256
        _atomic_json(version_dir / "binding.json", binding_meta)
        nonce = uuid.uuid4().hex
        correlation_key = f"kf1-{uuid.uuid4().hex}"
        intent = {
            "frame_id": frame.frame_id,
            "scene_id": frame.scene_id,
            "role": frame.role,
            "version": version,
            "nonce": nonce,
            "correlation_key": correlation_key,
            "binding_sha256": stable_hash,
            "bound_workflow_sha256": bound_workflow_sha256,
            "expected_output": dict(binding_meta["expected_output"]),
            "settings": dict(binding_meta["settings"]),
            "prompt_hash": _sha_bytes(binding_meta["positive_prompt"].encode("utf-8")),
            "prompt": binding_meta["positive_prompt"],
            "negative_prompt": binding_meta["negative_prompt"],
            "uploads": uploads,
            "runtime_gate": dict(gate),
            "source_hashes": source_hashes,
            "workflow_sha256": binding_meta["workflow_sha256"],
            "seed": self._frame_seed(frame),
            "roles": roles,
            "reference_roles": roles,
            "shot_mode": frame.shot_mode,
            "invariants": list(frame.invariants),
            "allowed_changes": list(frame.allowed_changes),
            "plan_schema": "v2",
            "intent_at": time.time(),
        }
        _atomic_json(version_dir / "intent.json", intent)
        attempt = self.store.begin_attempt(
            state,
            frame_id=frame.frame_id,
            version=version,
            intent=intent,
            allow_global_attempt_ids=allow_attempt_ids,
        )
        self.store.update_attempt(state, attempt["attempt_id"], status="POSTING")
        emit(f"сцена {frame.scene_id} — {frame.role} — очередь/POST")
        try:
            submission = self.client.submit(
                bound, client_id="anime-video-studio-kf1", correlation_key=correlation_key, nonce=nonce
            )
        except AmbiguousDispatch as exc:
            self.store.update_attempt(
                state, attempt["attempt_id"], status="AMBIGUOUS", error=str(exc), ambiguous_at=time.time()
            )
            emit(f"сцена {frame.scene_id} — {frame.role} — AMBIGUOUS")
            try:
                recovered = self.client.find_prompt_by_correlation(correlation_key, nonce)
            except Exception:
                recovered = None
            if recovered:
                self.store.update_attempt(
                    state,
                    attempt["attempt_id"],
                    status="SUBMITTED",
                    prompt_id=recovered,
                    recovered_at=time.time(),
                )
                emit(f"сцена {frame.scene_id} — {frame.role} — восстановлен exact prompt_id")
                return self._poll_and_finalize(
                    state,
                    frame,
                    attempt,
                    recovered,
                    version_dir,
                    source_hashes,
                    stable_hash,
                    deadline_seconds,
                    emit,
                )
            return {"frame_id": frame.frame_id, "status": "AMBIGUOUS", "version": version}
        except ValidationRejected as exc:
            self.store.update_attempt(
                state, attempt["attempt_id"], status="FAILED_TERMINAL", error=str(exc)
            )
            raise
        except Exception as exc:
            self.store.update_attempt(
                state, attempt["attempt_id"], status="AMBIGUOUS", error=type(exc).__name__
            )
            raise AmbiguousDispatch(f"{frame.frame_id}: submit raised {type(exc).__name__}") from exc
        self.store.update_attempt(
            state,
            attempt["attempt_id"],
            status="SUBMITTED",
            prompt_id=submission.prompt_id,
            submitted_at=time.time(),
        )
        emit(f"сцена {frame.scene_id} — {frame.role} — sampling/history {submission.prompt_id[:12]}")
        return self._poll_and_finalize(
            state,
            frame,
            attempt,
            submission.prompt_id,
            version_dir,
            source_hashes,
            stable_hash,
            deadline_seconds,
            emit,
        )

    def _select_version(self, state: Mapping[str, Any], frame: FrameSpec, force_new: bool) -> int:
        return self._select_version_and_dir(state, frame, force_new)[0]

    def _select_version_and_dir(
        self, state: Mapping[str, Any], frame: FrameSpec, force_new: bool
    ) -> tuple[int, Path]:
        frame_state = state.get("frames", {}).get(frame.frame_id, {})
        current = int(frame_state.get("current_version") or 0) if isinstance(frame_state, Mapping) else 0
        if force_new or current == 0:
            version = self.store.next_version(frame.frame_id)
        else:
            version = current
        path = self.store.ensure_version_dir(frame.frame_id, version)
        return version, path

    def _recover_unknown_ack(
        self,
        state: dict[str, Any],
        frame: FrameSpec,
        attempt: Mapping[str, Any],
        deadline_seconds: float,
        emit: Callable[[str], None],
    ) -> dict[str, Any] | None:
        if self.client is None:
            return None
        prompt_id = attempt.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id:
            self.store.update_attempt(state, str(attempt["attempt_id"]), status="SUBMITTED", prompt_id=prompt_id)
            return self._poll_and_finalize(
                state, frame, attempt, prompt_id, self.store.version_dir(frame.frame_id, int(attempt["version"])),
                dict(attempt.get("source_hashes") or {}), str(attempt.get("binding_sha256") or ""),
                deadline_seconds, emit,
            )
        if attempt.get("status") == "INTENT":
            self.store.update_attempt(state, str(attempt["attempt_id"]), status="ABORTED_PRE_SEND")
            return None
        try:
            found = self.client.find_prompt_by_correlation(
                str(attempt.get("correlation_key") or ""), str(attempt.get("nonce") or "")
            )
        except AmbiguousDispatch:
            found = None
        except Exception:
            found = None
        if found:
            self.store.update_attempt(
                state, str(attempt["attempt_id"]), status="SUBMITTED", prompt_id=found, recovered_at=time.time()
            )
            return self._poll_and_finalize(
                state, frame, attempt, found, self.store.version_dir(frame.frame_id, int(attempt["version"])),
                dict(attempt.get("source_hashes") or {}), str(attempt.get("binding_sha256") or ""),
                deadline_seconds, emit,
            )
        self.store.update_attempt(
            state, str(attempt["attempt_id"]), status="AMBIGUOUS", error="unknown ack; no correlation match"
        )
        raise AmbiguousDispatch(
            f"{frame.frame_id}: POSTING без prompt_id и без correlation match; AMBIGUOUS, retry запрещён"
        )

    def _poll_and_finalize(
        self,
        state: dict[str, Any],
        frame: FrameSpec,
        attempt: Mapping[str, Any],
        prompt_id: str,
        version_dir: Path,
        source_hashes: Mapping[str, Any],
        binding_hash: str,
        deadline_seconds: float,
        emit: Callable[[str], None],
    ) -> dict[str, Any]:
        if self.client is None:
            raise StoreError("нет ComfyClient")
        deadline = time.monotonic() + float(deadline_seconds)
        last_error = None
        while time.monotonic() < deadline:
            try:
                entry = self.client.history(prompt_id)
            except TransientRemoteError as exc:
                last_error = exc
                time.sleep(1.0)
                continue
            if entry is None:
                time.sleep(1.0)
                continue
            status, error = history_status(entry)
            if status in {"error", "failed"} or error:
                self.store.update_attempt(
                    state, str(attempt["attempt_id"]), status="FAILED_TERMINAL",
                    error=error or "remote terminal failure", completed_at=time.time(),
                )
                raise TerminalRemoteFailure(error or "remote terminal failure")
            try:
                output = output_from_history(
                    entry,
                    node_id=str(attempt.get("expected_output", {}).get("node_id", "70")),
                    keys=list(attempt.get("expected_output", {}).get("keys", ["images"])),
                    expected_output=attempt.get("expected_output"),
                    expected_graph_sha256=str(attempt.get("bound_workflow_sha256") or ""),
                    expected_prompt_id=prompt_id,
                    expected_correlation_key=str(attempt.get("correlation_key") or "") or None,
                    expected_nonce=str(attempt.get("nonce") or "") or None,
                    expected_source="anime-video-studio-kf1",
                )
            except ComfyError:
                if status in {"success", None}:
                    time.sleep(1.0)
                    continue
                raise
            emit(f"сцена {frame.scene_id} — {frame.role} — сохранение PNG")
            data = self.client.view(output)
            info = validate_png_bytes(data)
            expected_geometry = tuple(self.plan.output_policy["candidate_geometry"])
            if (
                (info["width"], info["height"]) != expected_geometry
                or int(info.get("orientation", 1)) != 1
            ):
                self.store.update_attempt(
                    state,
                    str(attempt["attempt_id"]),
                    status="FAILED_TERMINAL",
                    error=(
                        f"geometry mismatch: got {info['width']}x{info['height']}, "
                        f"expected {expected_geometry[0]}x{expected_geometry[1]}"
                    ),
                )
                raise TerminalRemoteFailure(
                    f"{frame.frame_id}: generated geometry mismatch; не маскирую resize"
                )
            target = version_dir / "frame.png"
            _atomic_bytes(target, data)
            final_info = validate_png_bytes(target.read_bytes())
            output_record = {
                "path": str(target),
                "sha256": final_info["sha256"],
                "size_bytes": final_info["size_bytes"],
                "width": final_info["width"],
                "height": final_info["height"],
                "format": "PNG",
                "source": output,
                "prompt_id": prompt_id,
            }
            receipt = {
                "output": output_record,
                "attempt_id": str(attempt["attempt_id"]),
                "frame_id": frame.frame_id,
                "version": int(attempt["version"]),
                "binding_sha256": binding_hash,
                "source_hashes": dict(source_hashes),
                "plan_sha256": self.plan.plan_sha256,
                "revision_id": state.get("revision_id"),
                "written_at": time.time(),
            }
            _atomic_json(version_dir / "result.json", receipt)
            self.store.complete_attempt(
                state,
                str(attempt["attempt_id"]),
                output=output_record,
                binding_sha256=binding_hash,
                source_hashes=dict(source_hashes),
                receipt=receipt,
            )
            emit(f"сцена {frame.scene_id} — {frame.role} — сохранено v{int(attempt['version']):04d}")
            return {
                "frame_id": frame.frame_id,
                "status": "COMPLETE",
                "version": int(attempt["version"]),
                "prompt_id": prompt_id,
                "output": output_record,
            }
        self.store.update_attempt(
            state, str(attempt["attempt_id"]), status="SUBMITTED",
            error=f"poll timeout: {type(last_error).__name__ if last_error else 'no terminal history'}",
        )
        raise TransientRemoteError(
            f"{frame.frame_id}: poll deadline без terminal history; POST не повторять, resume по prompt_id"
        )

    # --------------------------------------------------------------- retry
    def retry_frame(
        self,
        frame_id: str,
        *,
        owner_ack: bool,
        raise_limit: bool,
        deadline_seconds: float = 900.0,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if not owner_ack:
            raise StoreError("retry-frame требует explicit --owner-ack")
        if self.plan.schema_version == 1:
            raise V1MigrationRequired(
                "v1 retry-frame запрещён: создайте v2-копию плана; "
                "старый план/state не перезаписываются"
            )
        emit = progress or (lambda _line: None)
        with self.store.lock():
            state = self.store.load(plan=self.plan)
            if self.attestation is not None:
                state["runtime_attestation"] = dict(self.attestation)
                self.store.save(state)
            frame = self._frame(frame_id)
            if not frame.generate:
                raise StoreError("provided/first frame preserved; retry не применим")
            if frame.require_anchor_approval:
                if not frame.anchor_frame:
                    raise ApprovalRequired(
                        f"{frame_id}: require_anchor_approval без parent"
                    )
                anchor_frame = self._frame(frame.anchor_frame)
                anchor_state = state.get("frames", {}).get(anchor_frame.frame_id, {})
                anchor_version = int(anchor_state.get("current_version") or 0)
                validate_approval(
                    self.plan,
                    self.store,
                    state,
                    anchor_frame.frame_id,
                    anchor_version,
                )
            latest = self._latest_attempt(state, frame_id)
            if latest and latest.get("status") == "AMBIGUOUS" and not owner_ack:
                raise StoreError("AMBIGUOUS: нужен owner ack")
            if raise_limit:
                state["extra_limit"] = int(state.get("extra_limit", 0)) + 1
                self.store.save(state)
            refs = self._prepare_refs()
            result = self._run_generated_frame(
                state, frame, refs, deadline_seconds=deadline_seconds,
                force_new_version=True, emit=emit,
            )
            downstream = self._invalidate_downstream(state, frame_id, "explicit retry-frame")
            self.store.save(state)
            if result.get("status") == "AMBIGUOUS":
                raise AmbiguousDispatch(
                    f"{frame_id}: explicit retry снова AMBIGUOUS; новый auto-attempt запрещён"
                )
            return {
                "status": result.get("status"),
                "frame": result,
                "invalidated_downstream": downstream,
                "state": self.status(state),
            }


def make_runner(
    plan: ResolvedPlan,
    *,
    state_root: Path | str = ROOT / "data" / "kf1",
    client: ComfyClient | None = None,
) -> Runner:
    store = PlanStore(state_root, plan.plan_sha256, plan.path)
    return Runner(plan, store, client)


__all__ = ["Runner", "make_runner"]

