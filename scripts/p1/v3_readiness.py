"""Focused two-phase readiness for P1 v3.

Phase 1 (source/bootstrap) validates immutable source pins, owner approval,
fresh session/quote facts, host key and supervisor ownership. It deliberately
allows the source manifest to remain DRAFT because runtime evidence cannot exist
before bootstrap.

Phase 2 (runtime/generation) validates an explicit per-run live candidate built
only from collected runtime facts. The source DRAFT manifest is never edited and
the run/session/nonce is never recreated. Candidate files live in
``<run>/live`` and are immutable.

This module is offline-only. It performs no SSH, provider, account, environment
or secret access. The focused remote probe is a separate fixed script.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from anime_studio.p1_config import (
    CANDIDATE_SCHEMA,
    RUNTIME_EVIDENCE_SCHEMA,
    SOURCE_APPROVAL_SCHEMA,
    SOURCE_BUNDLE_ROLES,
    check_readiness,
    load_config,
    sha256_file,
    validate_candidate_manifest,
    validate_source_manifest,
)
from scripts.p1 import clore_session, owner_risk, preflight, v3_lifecycle


ROOT = Path(__file__).resolve().parents[2]
BINDING_SCHEMA = "anime-video-studio/p1-live-candidate-binding/v1"
QUOTE_OBSERVATION_SCHEMA = "anime-video-studio/p1-quote-observation/v1"
RUNTIME_PLAN_SCHEMA = "anime-video-studio/p1-runtime-evidence-plan/v1"
CURRENT_APPROVAL_DOC = ROOT / "docs" / "P1-session-02-approval-20261026.md"
DEFAULT_APPROVAL_VALID_HOURS = 24
DEFAULT_QUOTE_MAX_AGE_SECONDS = 15 * 60
DEFAULT_RUNTIME_MAX_AGE_SECONDS = 15 * 60
HOST_KEY_FACT_MAX_AGE_SECONDS = 15 * 60
SUPERVISOR_MAX_HEARTBEAT_AGE_SECONDS = 120.0
RUNTIME_PLAN_MAX_BYTES = 256 * 1024
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")

# These seven source hashes include the DRAFT manifest. A candidate is never
# allowed to rewrite them; the same values are bound into owner approval,
# lifecycle, candidate binding and runtime evidence.
SOURCE_PATHS: dict[str, str] = {
    "config": "config/p1.wan22-5b.shot01-v3.draft.json",
    "workflow": "workflows/P1/wan22-5b-shot01-v3.api.draft.json",
    "manifest": "manifests/P1.wan22-5b-shot01-v3.draft.json",
    "job": "config/p1.shot01-v3.job.json",
    "source": "assets/frames/shot-01-start-v3.png",
    "derived": "assets/frames/derived/shot-01-start-v3-704x1280.png",
    "policy": "assets/frames/derived/shot-01-start-v3-704x1280.policy.json",
}
FROZEN_SOURCE_BUNDLE: dict[str, str] = {
    "config": "e53a873ec71372c1bd47d17a905f8e4e713e8ded077b552e7b063469a3cab66a",
    "workflow": "4f5645d1a6af2c929a6260a609c0d3cf513db99bcd2522649bd02d4b40c52784",
    "manifest": "f6504159494585286eb56c97cb8de112386f231a92b77fd9629413d6d6d39099",
    "job": "097da38d046a5192e4f746223dfeefd0ec0656ffd83b804fd3e3b03529116d87",
    "source": "5e3029aa7edb9207c1503635606beeeed9b012995b2a296616668c84fe68083a",
    "derived": "925eccbe02a49b74bd2e8cd190fe26bd9557628bb996149f093b894136c53add",
    "policy": "1ddfa219c1bb33ea49f873aa4517c10962fc7bdcb24435727efba8ab1f7bdc5a",
}
EXPECTED_V3_MODELS: dict[str, dict[str, Any]] = {
    "wan22-ti2v-5b-fp16": {
        "role": "diffusion",
        "size_bytes": 9_999_658_848,
        "sha256": "456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e",
        "filename": "wan2.2_ti2v_5B_fp16.safetensors",
        "target": "models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
    },
    "wan22-vae": {
        "role": "vae",
        "size_bytes": 1_409_400_960,
        "sha256": "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156",
        "filename": "wan2.2_vae.safetensors",
        "target": "models/vae/wan2.2_vae.safetensors",
    },
    "umt5-xxl-fp8-scaled": {
        "role": "text_encoder",
        "size_bytes": 6_735_906_897,
        "sha256": "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
        "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "target": "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    },
}
EXPECTED_PLAN = {
    "width": 704,
    "height": 1280,
    "frames": 121,
    "fps": 24,
    "seed": 424242,
    "steps": 20,
    "cfg": 5,
    "sampler_name": "uni_pc",
    "scheduler": "simple",
}


class ReadinessError(Exception):
    """A source/generation readiness fact is missing or inconsistent."""


@dataclass(frozen=True)
class SourceApproval:
    path: Path
    payload: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class QuoteObservation:
    path: Path
    payload: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class SourceGate:
    passed: bool
    issues: tuple[str, ...]
    facts: Mapping[str, Any]


@dataclass(frozen=True)
class LiveCandidate:
    directory: Path
    manifest_path: Path
    config_path: Path
    evidence_path: Path
    raw_path: Path
    binding_path: Path
    binding: Mapping[str, Any]


@dataclass(frozen=True)
class GenerationGate:
    passed: bool
    issues: tuple[str, ...]
    facts: Mapping[str, Any]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_mapping(value: Mapping[str, Any]) -> str:
    text = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha_bytes(text.encode("utf-8"))


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"cannot read JSON {path}: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise ReadinessError(f"JSON {path} must be an object")
    return data


def _create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one immutable JSON file; never replace an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_text(payload).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ReadinessError(f"cannot create {path}: {type(exc).__name__}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_directory_create(path: Path, files: Mapping[str, bytes]) -> None:
    """Create an entire artifact directory atomically; never overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.parent / f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    temp.mkdir()
    try:
        for name, data in files.items():
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            import shutil

            shutil.rmtree(temp, ignore_errors=True)
        except OSError:
            pass
        raise


def _parse_moment(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ReadinessError(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReadinessError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _now(now: float | datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        return now.astimezone(timezone.utc)
    return datetime.fromtimestamp(float(now), tz=timezone.utc)


def _require_hex(value: Any, name: str, issues: list[str], length: int = 64) -> None:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value):
        issues.append(f"{name} must be {length} hex characters")


def _safe_rel(value: Any, name: str, issues: list[str], *, prefix: str | None = None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{name} is missing")
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or "\x00" in normalized:
        issues.append(f"{name} must be a normalized relative POSIX path")
        return None
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        issues.append(f"{name} must be a normalized relative POSIX path")
        return None
    if prefix is not None and not normalized.startswith(prefix):
        issues.append(f"{name} must be under {prefix}")
    return normalized


def source_bundle_hashes(root: Path | str = ROOT) -> dict[str, str]:
    root_path = Path(root)
    return {
        role: sha256_file(root_path / relative)
        for role, relative in SOURCE_PATHS.items()
    }


def source_bundle_digest(hashes: Mapping[str, str]) -> str:
    return _sha_mapping(hashes)


def session_binding_digest(binding: Any) -> str:
    if hasattr(binding, "to_dict"):
        payload = binding.to_dict()
    elif isinstance(binding, Mapping):
        payload = dict(binding)
    else:
        raise ReadinessError("session binding is not serializable")
    return _sha_mapping(payload)


def quote_observation_path(session_state: Path | str) -> Path:
    path = Path(session_state)
    return path.with_name(path.name + ".quote-observation.json")


def load_source_approval(path: Path | str) -> SourceApproval:
    approval_path = Path(path)
    payload = _read_json(approval_path)
    try:
        digest = sha256_file(approval_path)
    except Exception as exc:
        raise ReadinessError("cannot hash source approval artifact") from exc
    return SourceApproval(path=approval_path, payload=payload, sha256=digest)


def load_quote_observation(path: Path | str) -> QuoteObservation:
    observation_path = Path(path)
    payload = _read_json(observation_path)
    try:
        digest = sha256_file(observation_path)
    except Exception as exc:
        raise ReadinessError("cannot hash quote observation artifact") from exc
    return QuoteObservation(path=observation_path, payload=payload, sha256=digest)


def _source_dates(manifest: Mapping[str, Any], issues: list[str]) -> None:
    today = datetime.now(timezone.utc).date()

    def check(value: Any, name: str) -> None:
        if not isinstance(value, str):
            return
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return
        if parsed > today:
            issues.append(f"{name} is in the future")

    check(manifest.get("checked_at"), "manifest.checked_at")
    models = manifest.get("models")
    if isinstance(models, list):
        for index, model in enumerate(models):
            if isinstance(model, Mapping):
                check(model.get("license_reviewed_at"), f"manifest.models[{index}].license_reviewed_at")


def _validate_source_pins(
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    *,
    root: Path = ROOT,
) -> list[str]:
    issues: list[str] = []
    manifest = bundle.get("manifest")
    config = bundle.get("config")
    workflow = bundle.get("workflow")
    job = bundle.get("job")
    policy = bundle.get("policy")
    if not isinstance(manifest, Mapping):
        return ["source manifest is missing"]
    if not isinstance(workflow, Mapping):
        return ["source workflow is missing"]
    if not isinstance(job, Mapping):
        return ["source job is missing"]
    if not isinstance(policy, Mapping):
        return ["source policy is missing"]
    if config is None or not hasattr(config, "workflow_path"):
        return ["source runtime config is missing"]
    try:
        workflow_path = Path(getattr(config, "workflow_path"))
        actual_workflow = str(bundle.get("workflow_sha256") or "")
        issues.extend(
            validate_source_manifest(
                manifest,
                workflow_path=workflow_path,
                actual_workflow_sha256=actual_workflow,
                config=config,
            )
        )
    except Exception as exc:
        issues.append(f"source manifest validation failed: {type(exc).__name__}")

    for role in SOURCE_BUNDLE_ROLES:
        expected = FROZEN_SOURCE_BUNDLE.get(role)
        actual = source_hashes.get(role)
        if expected is None or actual != expected:
            issues.append(f"source_bundle.{role} does not match the frozen source pin")
    if manifest.get("status") != "DRAFT":
        issues.append("source manifest must remain DRAFT")
    manifest_workflow = manifest.get("workflow")
    if not isinstance(manifest_workflow, Mapping) or manifest_workflow.get("sha256") != source_hashes.get("workflow"):
        issues.append("manifest.workflow.sha256 does not match the source bundle")
    if config.workflow_sha256 != source_hashes.get("workflow"):
        issues.append("config.workflow_sha256 does not match the source bundle")

    _source_dates(manifest, issues)

    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != len(EXPECTED_V3_MODELS):
        issues.append("source manifest must pin exactly the three v3 models")
        models = []
    found_ids: set[str] = set()
    model_by_id: dict[str, Mapping[str, Any]] = {}
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            issues.append(f"manifest.models[{index}] must be an object")
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or model_id not in EXPECTED_V3_MODELS:
            issues.append(f"manifest.models[{index}].id is not an approved v3 model")
            continue
        found_ids.add(model_id)
        model_by_id[model_id] = model
        expected = EXPECTED_V3_MODELS[model_id]
        prefix = f"manifest.models[{model_id}]"
        if model.get("role") != expected["role"]:
            issues.append(f"{prefix}.role does not match the v3 pin")
        if model.get("size_bytes") != expected["size_bytes"]:
            issues.append(f"{prefix}.size_bytes does not match the v3 pin")
        if model.get("sha256") != expected["sha256"]:
            issues.append(f"{prefix}.sha256 does not match the v3 pin")
        if model.get("target") != expected["target"]:
            issues.append(f"{prefix}.target does not match the v3 pin")
        if Path(str(model.get("filename"))).name != expected["filename"]:
            issues.append(f"{prefix}.filename does not match the v3 pin")
        revision = model.get("revision")
        if not isinstance(revision, str) or not _HEX40.match(revision):
            issues.append(f"{prefix}.revision must be a 40-hex commit")
        _safe_rel(model.get("target"), f"{prefix}.target", issues, prefix="models/")
        size = model.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            issues.append(f"{prefix}.size_bytes must be a positive integer")
        if model.get("hosted_use") != "allowed":
            issues.append(f"{prefix}.hosted_use must be allowed")
        if model.get("commercial_use_status") != "allowed":
            issues.append(f"{prefix}.commercial_use_status must be allowed")
    if found_ids != set(EXPECTED_V3_MODELS):
        issues.append("source manifest models are incomplete")

    live_plan = manifest.get("live_plan")
    if not isinstance(live_plan, Mapping):
        issues.append("manifest.live_plan is missing")
    else:
        if live_plan.get("source_image_sha256") != source_hashes.get("source"):
            issues.append("manifest.live_plan.source_image_sha256 does not match source")
        if live_plan.get("derived_input_image_sha256") != source_hashes.get("derived"):
            issues.append("manifest.live_plan.derived_input_image_sha256 does not match derived")
        if _normalized_path(live_plan.get("source_image")) != SOURCE_PATHS["source"]:
            issues.append("manifest.live_plan.source_image path is not the frozen source")
        if _normalized_path(live_plan.get("derived_input_image")) != SOURCE_PATHS["derived"]:
            issues.append("manifest.live_plan.derived_input_image path is not the frozen derived frame")
        policy_crop = policy.get("crop", {}) if isinstance(policy, Mapping) else {}
        live_crop = live_plan.get("crop_policy", {})
        if not isinstance(live_crop, Mapping):
            issues.append("manifest.live_plan.crop_policy is missing")
        else:
            if live_crop.get("path") != SOURCE_PATHS["policy"]:
                issues.append("manifest.live_plan.crop_policy.path is not the frozen policy")
            if live_crop.get("box_left_top_right_bottom") != policy_crop.get("box_left_top_right_bottom"):
                issues.append("manifest.live_plan crop box differs from policy")
        if policy.get("source", {}).get("sha256") != source_hashes.get("source"):
            issues.append("policy source SHA256 differs from source")
        if policy.get("target", {}).get("sha256") != source_hashes.get("derived"):
            issues.append("policy target SHA256 differs from derived")
        if live_plan.get("generations") != 1:
            issues.append("manifest.live_plan.generations must be 1")
        if live_plan.get("automatic_retries") != 0:
            issues.append("manifest.live_plan.automatic_retries must be 0")
        if live_plan.get("max_session_minutes") != 60:
            issues.append("manifest.live_plan.max_session_minutes must be 60")

        for key, expected in EXPECTED_PLAN.items():
            if live_plan.get(key) != expected:
                issues.append(f"manifest.live_plan.{key} does not match the frozen plan")
        if job.get("seed") != EXPECTED_PLAN["seed"]:
            issues.append("job.seed does not match the frozen plan")
        if job.get("prompt") != bundle.get("positive"):
            issues.append("job.prompt does not match the frozen prompt")
        if config.width != EXPECTED_PLAN["width"] or config.height != EXPECTED_PLAN["height"]:
            issues.append("config dimensions do not match the frozen plan")
        if config.frames != EXPECTED_PLAN["frames"] or config.fps != EXPECTED_PLAN["fps"]:
            issues.append("config frames/fps do not match the frozen plan")
        if workflow.get("3", {}).get("inputs", {}).get("seed") != EXPECTED_PLAN["seed"]:
            issues.append("workflow node 3 seed does not match the frozen plan")
        if workflow.get("3", {}).get("inputs", {}).get("steps") != EXPECTED_PLAN["steps"]:
            issues.append("workflow node 3 steps do not match the frozen plan")
        if workflow.get("3", {}).get("inputs", {}).get("cfg") != EXPECTED_PLAN["cfg"]:
            issues.append("workflow node 3 cfg does not match the frozen plan")
        if workflow.get("3", {}).get("inputs", {}).get("sampler_name") != EXPECTED_PLAN["sampler_name"]:
            issues.append("workflow sampler does not match the frozen plan")
        if workflow.get("3", {}).get("inputs", {}).get("scheduler") != EXPECTED_PLAN["scheduler"]:
            issues.append("workflow scheduler does not match the frozen plan")
        if workflow.get("55", {}).get("inputs", {}).get("width") != EXPECTED_PLAN["width"]:
            issues.append("workflow width does not match the frozen plan")
        if workflow.get("55", {}).get("inputs", {}).get("height") != EXPECTED_PLAN["height"]:
            issues.append("workflow height does not match the frozen plan")
        if workflow.get("55", {}).get("inputs", {}).get("length") != EXPECTED_PLAN["frames"]:
            issues.append("workflow frames do not match the frozen plan")
        if workflow.get("57", {}).get("inputs", {}).get("fps") != EXPECTED_PLAN["fps"]:
            issues.append("workflow fps does not match the frozen plan")
        if workflow.get("56", {}).get("inputs", {}).get("image") != PurePosixPath(SOURCE_PATHS["derived"]).name:
            issues.append("workflow input image is not the frozen derived frame")
        for model_id, node_id, input_name in (
            ("wan22-ti2v-5b-fp16", "37", "unet_name"),
            ("umt5-xxl-fp8-scaled", "38", "clip_name"),
            ("wan22-vae", "39", "vae_name"),
        ):
            expected = EXPECTED_V3_MODELS[model_id]
            if workflow.get(node_id, {}).get("inputs", {}).get(input_name) != expected["filename"]:
                issues.append(f"workflow node {node_id}.{input_name} does not match {model_id}")
        if manifest.get("workflow", {}).get("source_template_commit") and not _HEX40.match(
            str(manifest["workflow"]["source_template_commit"])
        ):
            issues.append("manifest.workflow.source_template_commit must be 40 hex")

    return issues


def _normalized_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def _validate_source_approval(
    approval: SourceApproval,
    source_hashes: Mapping[str, str],
    *,
    now: datetime,
) -> list[str]:
    issues: list[str] = []
    payload = approval.payload
    if payload.get("schema") != SOURCE_APPROVAL_SCHEMA:
        issues.append(f"source approval schema must be {SOURCE_APPROVAL_SCHEMA!r}")
    if not isinstance(payload.get("approved_by"), str) or not payload["approved_by"].strip():
        issues.append("source approval approved_by is missing")
    created = None
    expires = None
    try:
        created = _parse_moment(payload.get("created_at"), "source approval created_at")
    except ReadinessError as exc:
        issues.append(str(exc))
    try:
        expires = _parse_moment(payload.get("expires_at"), "source approval expires_at")
    except ReadinessError as exc:
        issues.append(str(exc))
    if created is not None and created > now:
        issues.append("source approval created_at is in the future")
    if expires is not None and expires <= now:
        issues.append("source approval has expired")
    if created is not None and expires is not None and expires <= created:
        issues.append("source approval expires_at must be after created_at")

    doc = payload.get("approval_doc")
    if not isinstance(doc, Mapping):
        issues.append("source approval approval_doc is missing")
    else:
        doc_path_raw = doc.get("path")
        if not isinstance(doc_path_raw, str):
            issues.append("source approval approval_doc.path is missing")
        else:
            try:
                doc_path = Path(doc_path_raw)
                if not doc_path.is_absolute():
                    doc_path = ROOT / doc_path
                doc_path = doc_path.resolve()
            except OSError as exc:
                issues.append(f"source approval approval_doc.path is invalid: {type(exc).__name__}")
            else:
                if doc_path != CURRENT_APPROVAL_DOC.resolve():
                    issues.append("source approval is not bound to the current session02 approval document")
                if not doc_path.is_file():
                    issues.append("source approval document is missing")
                else:
                    try:
                        actual_doc = sha256_file(doc_path)
                    except Exception as exc:
                        issues.append(f"cannot hash source approval document: {type(exc).__name__}")
                    else:
                        _require_hex(doc.get("sha256"), "source approval approval_doc.sha256", issues)
                        if isinstance(doc.get("sha256"), str) and actual_doc.lower() != doc["sha256"].lower():
                            issues.append("source approval approval_doc.sha256 does not match the document")

    bundle = payload.get("source_bundle")
    if not isinstance(bundle, Mapping) or set(bundle) != set(SOURCE_BUNDLE_ROLES):
        issues.append("source approval source_bundle is missing/unknown")
    else:
        for role in SOURCE_BUNDLE_ROLES:
            _require_hex(bundle.get(role), f"source approval source_bundle.{role}", issues)
            if isinstance(bundle.get(role), str) and bundle[role].lower() != str(source_hashes.get(role, "")).lower():
                issues.append(f"source approval source_bundle.{role} differs from the current source")
        if payload.get("source_bundle_sha256") != source_bundle_digest(bundle):
            issues.append("source approval source_bundle_sha256 does not match source_bundle")
    manifest = payload.get("source_manifest")
    if not isinstance(manifest, Mapping):
        issues.append("source approval source_manifest is missing")
    else:
        _require_hex(manifest.get("sha256"), "source approval source_manifest.sha256", issues)
        if manifest.get("status") != "DRAFT":
            issues.append("source approval source_manifest.status must be DRAFT")
        if isinstance(manifest.get("sha256"), str) and manifest["sha256"].lower() != str(
            source_hashes.get("manifest", "")
        ).lower():
            issues.append("source approval source_manifest.sha256 differs from the current manifest")
        if not isinstance(manifest.get("path"), str) or not manifest["path"].strip():
            issues.append("source approval source_manifest.path is missing")

    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        issues.append("source approval scope is missing")
    else:
        if scope.get("purpose") != "bootstrap_candidate_only":
            issues.append("source approval scope.purpose must be bootstrap_candidate_only")
        if scope.get("orders") != 1:
            issues.append("source approval scope.orders must be 1")
        if scope.get("max_generations") != 1:
            issues.append("source approval scope.max_generations must be 1")
        if scope.get("automatic_retries") != 0:
            issues.append("source approval scope.automatic_retries must be 0")
        if scope.get("max_duration_seconds") != 3600:
            issues.append("source approval scope.max_duration_seconds must be 3600")
        try:
            cap = Decimal(str(scope.get("max_total_usd")))
        except (InvalidOperation, ValueError):
            issues.append("source approval scope.max_total_usd is invalid")
        else:
            if cap <= 0 or cap > Decimal("1.00"):
                issues.append("source approval scope.max_total_usd must be <= 1.00")
    return issues


def _approval_cap(approval: SourceApproval) -> Decimal | None:
    scope = approval.payload.get("scope")
    if not isinstance(scope, Mapping):
        return None
    try:
        cap = Decimal(str(scope.get("max_total_usd")))
    except (InvalidOperation, ValueError):
        return None
    return cap


def _validate_session(
    state: Any,
    binding: Any,
    *,
    now: datetime,
    effective_cap: Decimal | None = None,
) -> list[str]:
    issues: list[str] = []
    admission = getattr(state, "owner_risk", None)
    try:
        if not isinstance(state, clore_session.CloreSessionState):
            raise ReadinessError("session state has the wrong type")
        binding.assert_matches_state(state)
        v3_lifecycle.assert_deadline(binding, now=now.timestamp())
        v3_lifecycle.assert_budget(binding, state, now=now)
    except Exception as exc:
        issues.append(f"session identity/budget/deadline failed: {exc}")
    if getattr(state, "status", None) != "ARMED":
        issues.append("session state is not ARMED")
    try:
        cap = Decimal(str(binding.cap_usd))
    except (InvalidOperation, ValueError):
        issues.append("session cap is invalid")
        cap = None
    else:
        if cap <= 0 or cap > Decimal("1.00"):
            issues.append("session cap must be <= 1.00 USD")
    if cap is not None and effective_cap is not None:
        if cap > effective_cap:
            issues.append("session cap exceeds the effective owner approval cap")
    if admission is not None and binding.duration_seconds != 3600:
        issues.append("owner-risk session duration must be exactly 3600 seconds")
    elif binding.duration_seconds > 3600:
        issues.append("session duration exceeds 60 minutes")
    if binding.hard_end_at > binding.created_at + 3600:
        issues.append("session hard end exceeds provider creation + 60 minutes")
    if binding.planned_cancel_at > binding.created_at + 3300:
        issues.append("planned cancel exceeds provider creation + 55 minutes")
    quote_cap = effective_cap if effective_cap is not None else cap
    try:
        blockers = state.quote.launch_blockers(
            cap_usd=quote_cap if quote_cap is not None else Decimal("0"),
            minutes=60,
            admission=admission,
            now=now,
        )
    except Exception as exc:
        issues.append(f"quote evaluator failed: {type(exc).__name__}")
    else:
        issues.extend(f"quote: {blocker}" for blocker in blockers)
    if admission is not None:
        if getattr(state, "owner_risk_sha256", None) != getattr(
            binding, "owner_risk_admission_sha256", None
        ):
            issues.append("session owner_risk digest differs from state")
        admission_issues = owner_risk.validate_admission(
            admission, now=now, allow_expired=False
        )
        issues.extend(f"owner_risk: {issue}" for issue in admission_issues)
    return issues


def _validate_quote_observation(
    observation: QuoteObservation,
    *,
    state: Any,
    binding: Any,
    now: datetime,
    max_age_seconds: float | None = DEFAULT_QUOTE_MAX_AGE_SECONDS,
) -> list[str]:
    issues: list[str] = []
    payload = observation.payload
    if payload.get("schema") != QUOTE_OBSERVATION_SCHEMA:
        issues.append(f"quote observation schema must be {QUOTE_OBSERVATION_SCHEMA!r}")
    quote_path = payload.get("quote_path")
    state_path = payload.get("session_state_path")
    if not isinstance(quote_path, str) or not quote_path.strip():
        issues.append("quote observation quote_path is missing")
    if not isinstance(state_path, str) or not state_path.strip():
        issues.append("quote observation session_state_path is missing")
    if isinstance(quote_path, str) and quote_path.strip():
        try:
            actual_quote = sha256_file(Path(quote_path))
        except Exception as exc:
            issues.append(f"quote evidence file cannot be hashed: {type(exc).__name__}")
        else:
            _require_hex(payload.get("quote_sha256"), "quote observation quote_sha256", issues)
            if isinstance(payload.get("quote_sha256"), str) and actual_quote.lower() != payload["quote_sha256"].lower():
                issues.append("quote observation quote_sha256 does not match the quote file")
    if isinstance(state_path, str) and state_path.strip():
        try:
            actual_state = sha256_file(Path(state_path))
        except Exception as exc:
            issues.append(f"session state cannot be hashed for quote evidence: {type(exc).__name__}")
        else:
            _require_hex(payload.get("session_state_sha256"), "quote observation session_state_sha256", issues)
            if isinstance(payload.get("session_state_sha256"), str) and actual_state.lower() != payload["session_state_sha256"].lower():
                issues.append("quote observation session_state_sha256 does not match the state file")
    if payload.get("order_id") != binding.order_id or payload.get("created_at") != binding.created_at:
        issues.append("quote observation order identity does not match the session")
    try:
        observed = _parse_moment(payload.get("observed_at"), "quote observation observed_at")
    except ReadinessError as exc:
        issues.append(str(exc))
    else:
        if observed > now:
            issues.append("quote observation observed_at is in the future")
        if max_age_seconds is not None:
            age = (now - observed).total_seconds()
            if age > max_age_seconds:
                issues.append("quote observation is stale")
        effective = min(
            observed + timedelta(seconds=max_age_seconds)
            if max_age_seconds is not None
            else datetime.fromtimestamp(binding.hard_end_at, tz=timezone.utc),
            datetime.fromtimestamp(binding.planned_cancel_at, tz=timezone.utc),
            datetime.fromtimestamp(binding.hard_end_at, tz=timezone.utc),
        )
        if now >= effective:
            issues.append("quote observation is outside the session deadline")
    admission = getattr(state, "owner_risk", None)
    if admission is not None:
        try:
            admitted = owner_risk.OwnerRiskAdmission.from_dict(admission)
        except owner_risk.OwnerRiskError as exc:
            issues.append(f"owner-risk admission is invalid: {exc}")
        else:
            admitted_quote = str(admitted.quote.get("sha256") or "").lower()
            observed_quote = str(payload.get("quote_sha256") or "").lower()
            if admitted_quote != observed_quote:
                issues.append(
                    "quote observation quote_sha256 differs from the owner-risk admission"
                )
    return issues


def _validate_host_key(
    fact: Mapping[str, Any] | None,
    *,
    expected_fingerprint: str | None,
    remote: Any,
    now: datetime,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(fact, Mapping):
        return ["host key fact is missing"]
    fingerprint = fact.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.startswith("SHA256:"):
        issues.append("host key fact fingerprint is invalid")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        issues.append("host key fact fingerprint differs from the trusted fingerprint")
    _require_hex(fact.get("known_hosts_sha256"), "host key fact known_hosts_sha256", issues)
    if remote is not None:
        for key, attr in (("host", "host"), ("user", "user"), ("port", "port")):
            if fact.get(key) != getattr(remote, attr, None):
                issues.append(f"host key fact {key} does not match the selected remote")
    try:
        checked = _parse_moment(fact.get("checked_at"), "host key fact checked_at")
    except ReadinessError as exc:
        issues.append(str(exc))
    else:
        if checked > now:
            issues.append("host key fact checked_at is in the future")
        if (now - checked).total_seconds() > HOST_KEY_FACT_MAX_AGE_SECONDS:
            issues.append("host key fact is stale")
    return issues


def _validate_supervisor(
    fact: Mapping[str, Any] | None,
    *,
    run_id: str | None,
    nonce: str | None,
    binding_hash: str,
    now: datetime,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(fact, Mapping):
        return ["supervisor ack fact is missing"]
    if run_id is not None and fact.get("run_id") != run_id:
        issues.append("supervisor ack run_id identity mismatch")
    if nonce is not None and fact.get("nonce") != nonce:
        issues.append("supervisor ack nonce identity mismatch")
    if fact.get("status") != "ARMED":
        issues.append("supervisor ack is not ARMED")
    if fact.get("binding_sha256") != binding_hash:
        issues.append("supervisor ack binding hash mismatch")
    _require_hex(fact.get("config_sha256"), "supervisor ack config_sha256", issues)
    pid = fact.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        issues.append("supervisor ack pid is invalid")
    heartbeat = fact.get("heartbeat_at")
    if isinstance(heartbeat, bool) or not isinstance(heartbeat, (int, float)):
        issues.append("supervisor heartbeat timestamp is invalid")
    else:
        age = now.timestamp() - float(heartbeat)
        if age < -30 or age > SUPERVISOR_MAX_HEARTBEAT_AGE_SECONDS:
            issues.append("supervisor heartbeat is stale")
    return issues


def make_host_key_fact(
    remote: Any,
    expected_fingerprint: str,
    *,
    known_hosts: Path | str | None = None,
    checked_at: datetime | None = None,
) -> dict[str, Any]:
    known_path = Path(known_hosts if known_hosts is not None else remote.known_hosts)
    fact = {
        "fingerprint": expected_fingerprint,
        "known_hosts_sha256": sha256_file(known_path),
        "host": getattr(remote, "host", None),
        "user": getattr(remote, "user", None),
        "port": getattr(remote, "port", None),
        "checked_at": _iso(checked_at or datetime.now(timezone.utc)),
    }
    return fact


def make_supervisor_fact(
    record: Mapping[str, Any],
    ack: Mapping[str, Any],
    heartbeat: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    heartbeat_at = heartbeat.get("at")
    if not isinstance(heartbeat_at, (int, float)):
        raise ReadinessError("supervisor heartbeat has no timestamp")
    return {
        "run_id": record.get("run_id"),
        "nonce": record.get("nonce"),
        "pid": ack.get("pid", record.get("pid")),
        "status": ack.get("status"),
        "binding_sha256": ack.get("binding_sha256", record.get("binding_sha256")),
        "config_sha256": ack.get("config_sha256", record.get("config_sha256")),
        "heartbeat_at": float(heartbeat_at),
        "recorded_at": _iso(now or datetime.now(timezone.utc)),
    }


def validate_source_bundle(
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> list[str]:
    return _validate_source_pins(bundle, source_hashes)


def evaluate_source_gate(
    *,
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    approval: SourceApproval,
    quote_observation: QuoteObservation,
    session_state: Any,
    binding: Any,
    host_key_fact: Mapping[str, Any] | None,
    remote: Any,
    expected_fingerprint: str | None,
    supervisor_fact: Mapping[str, Any] | None = None,
    require_supervisor: bool = False,
    run_id: str | None = None,
    nonce: str | None = None,
    now: datetime | None = None,
    quote_max_age_seconds: float | None = DEFAULT_QUOTE_MAX_AGE_SECONDS,
) -> SourceGate:
    current = _now(now)
    issues: list[str] = []
    issues.extend(_validate_source_pins(bundle, source_hashes))
    issues.extend(_validate_source_approval(approval, source_hashes, now=current))
    approval_cap = _approval_cap(approval)
    if approval_cap is None:
        issues.append("source approval effective cap is missing/invalid")
    try:
        binding_cap = Decimal(str(binding.cap_usd))
    except (InvalidOperation, ValueError):
        binding_cap = None
    if approval_cap is not None and binding_cap is not None and binding_cap > approval_cap:
        issues.append("session cap exceeds the source approval cap")
    effective_cap = (
        min(Decimal("1.00"), approval_cap, binding_cap)
        if approval_cap is not None and binding_cap is not None
        else None
    )
    issues.extend(
        _validate_session(
            session_state, binding, now=current, effective_cap=effective_cap
        )
    )
    issues.extend(
        _validate_quote_observation(
            quote_observation,
            state=session_state,
            binding=binding,
            now=current,
            max_age_seconds=quote_max_age_seconds,
        )
    )
    issues.extend(
        _validate_host_key(
            host_key_fact,
            expected_fingerprint=expected_fingerprint,
            remote=remote,
            now=current,
        )
    )
    if require_supervisor:
        issues.extend(
            _validate_supervisor(
                supervisor_fact,
                run_id=run_id,
                nonce=nonce,
                binding_hash=session_binding_digest(binding),
                now=current,
            )
        )
    facts = {
        "source_bundle_sha256": source_bundle_digest(source_hashes),
        "source_bundle": dict(source_hashes),
        "source_approval_sha256": approval.sha256,
        "source_approval_path": str(approval.path),
        "quote_observation_sha256": quote_observation.sha256,
        "quote_observation_path": str(quote_observation.path),
        "session_binding_sha256": session_binding_digest(binding),
        "host_key": dict(host_key_fact) if isinstance(host_key_fact, Mapping) else None,
        "approval_expires_at": approval.payload.get("expires_at"),
        "effective_cap_usd": str(effective_cap) if effective_cap is not None else None,
        "financial_status": (
            "OWNER_ACCEPTED_FINANCIAL_RISK"
            if getattr(session_state, "owner_risk", None) is not None
            else "STRICT_QUOTE"
        ),
        "owner_risk_admission_id": (
            getattr(session_state, "owner_risk", {}) or {}
        ).get("admission_id")
        if isinstance(getattr(session_state, "owner_risk", None), Mapping)
        else None,
        "owner_risk_admission_sha256": getattr(
            session_state, "owner_risk_sha256", None
        ),
        "provider_billing_verified": False,
        "phase": "SOURCE_BOOTSTRAP",
    }
    return SourceGate(passed=not issues, issues=tuple(issues), facts=facts)


def create_source_approval(
    *,
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    output_path: Path | str,
    approval_doc: Path | str = CURRENT_APPROVAL_DOC,
    approved_by: str = "owner",
    cap_usd: str = "1.00",
    duration_seconds: int = 3600,
    valid_hours: int = DEFAULT_APPROVAL_VALID_HOURS,
    now: datetime | None = None,
) -> SourceApproval:
    current = _now(now)
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ReadinessError("approved_by is required")
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int):
        raise ReadinessError("duration_seconds must be an integer")
    if not 1 <= duration_seconds <= 3600:
        raise ReadinessError("duration_seconds must be between 1 and 3600")
    if isinstance(valid_hours, bool) or not isinstance(valid_hours, int) or not 1 <= valid_hours <= 24:
        raise ReadinessError("valid_hours must be between 1 and 24")
    try:
        cap = Decimal(str(cap_usd))
    except (InvalidOperation, ValueError) as exc:
        raise ReadinessError("cap_usd is invalid") from exc
    if cap <= 0 or cap > Decimal("1.00"):
        raise ReadinessError("cap_usd must be <= 1.00; the owner 2.00 USD limit does not widen the code cap")
    doc_path = Path(approval_doc)
    if not doc_path.is_absolute():
        doc_path = ROOT / doc_path
    doc_path = doc_path.resolve()
    if doc_path != CURRENT_APPROVAL_DOC.resolve():
        raise ReadinessError("source approval must reference the current session02 approval document")
    if not doc_path.is_file():
        raise ReadinessError("source approval document is missing")
    issues = _validate_source_pins(bundle, source_hashes)
    if issues:
        raise ReadinessError("source approval refused: " + "; ".join(issues))
    expires = current + timedelta(hours=valid_hours)
    payload = {
        "schema": SOURCE_APPROVAL_SCHEMA,
        "approval_id": f"source-{current.strftime('%Y%m%dT%H%M%SZ')}-{source_bundle_digest(source_hashes)[:8]}",
        "approved_by": approved_by,
        "created_at": _iso(current),
        "expires_at": _iso(expires),
        "approval_doc": {
            "path": str(doc_path),
            "sha256": sha256_file(doc_path),
        },
        "source_manifest": {
            "path": str((ROOT / SOURCE_PATHS["manifest"]).resolve()),
            "sha256": source_hashes["manifest"],
            "status": "DRAFT",
        },
        "source_bundle": dict(source_hashes),
        "source_bundle_sha256": source_bundle_digest(source_hashes),
        "scope": {
            "purpose": "bootstrap_candidate_only",
            "orders": 1,
            "max_generations": 1,
            "automatic_retries": 0,
            "max_total_usd": str(cap),
            "max_duration_seconds": duration_seconds,
        },
        "provenance": {
            "kind": "explicit_owner_attestation",
            "signature": None,
            "note": "Local owner attestation; this is not a cryptographic signature or host attestation.",
        },
    }
    path = Path(output_path)
    if path.exists():
        existing = load_source_approval(path)
        existing_payload = existing.payload
        same_scope = existing_payload.get("scope") == payload["scope"]
        same_source = existing_payload.get("source_bundle_sha256") == payload["source_bundle_sha256"]
        same_doc = (
            existing_payload.get("approval_doc", {}).get("sha256")
            == payload["approval_doc"]["sha256"]
        )
        same_owner = existing_payload.get("approved_by") == payload["approved_by"]
        if same_scope and same_source and same_doc and same_owner:
            return existing
        raise ReadinessError("source approval already exists and is immutable; refusing overwrite")
    try:
        _create_json(path, payload)
    except FileExistsError:
        raise ReadinessError("source approval already exists and is immutable; refusing overwrite")
    return load_source_approval(path)


def create_quote_observation(
    *,
    quote_path: Path | str,
    session_state: Path | str,
    binding: Any,
    output_path: Path | str | None = None,
    observed_at: datetime | None = None,
) -> QuoteObservation:
    quote = Path(quote_path)
    state_path = Path(session_state)
    if not quote.is_file():
        raise ReadinessError("quote machine file is missing")
    if not state_path.is_file():
        raise ReadinessError("session state is missing")
    artifact_created_at = datetime.now(timezone.utc)
    observed = observed_at
    observation_origin = "machine_file_observed_by_session_bind"
    if observed is None:
        try:
            quote_document = json.loads(quote.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReadinessError(
                f"cannot read quote observation source: {type(exc).__name__}"
            ) from exc
        if isinstance(quote_document, Mapping):
            raw_observed = quote_document.get("observed_at")
            if isinstance(raw_observed, str) and raw_observed.strip():
                observed = _parse_moment(raw_observed, "quote.observed_at")
                provenance = quote_document.get("provenance")
                if isinstance(provenance, Mapping):
                    origin = provenance.get("observation_origin") or provenance.get(
                        "kind"
                    )
                    if isinstance(origin, str) and origin.strip():
                        observation_origin = origin
    observed = observed or artifact_created_at
    target = Path(output_path) if output_path is not None else quote_observation_path(state_path)
    payload = {
        "schema": QUOTE_OBSERVATION_SCHEMA,
        "quote_path": str(quote.resolve()),
        "quote_sha256": sha256_file(quote),
        "session_state_path": str(state_path.resolve()),
        "session_state_sha256": sha256_file(state_path),
        "order_id": binding.order_id,
        "created_at": binding.created_at,
        "observed_at": _iso(observed),
        "artifact_created_at": _iso(artifact_created_at),
        "observation_origin": observation_origin,
        "provenance": "machine_file_observed_by_session_bind",
    }
    if target.exists():
        existing = load_quote_observation(target)
        stable = {
            key: existing.payload.get(key)
            for key in (
                "quote_path",
                "quote_sha256",
                "session_state_path",
                "session_state_sha256",
                "order_id",
                "created_at",
            )
        }
        expected_stable = {
            key: payload.get(key)
            for key in (
                "quote_path",
                "quote_sha256",
                "session_state_path",
                "session_state_sha256",
                "order_id",
                "created_at",
            )
        }
        if stable == expected_stable:
            return existing
        raise ReadinessError(
            "quote observation is immutable; a changed quote/state requires a new explicit session"
        )
    try:
        _create_json(target, payload)
    except FileExistsError:
        raise ReadinessError("quote observation is immutable; refusing overwrite")
    return load_quote_observation(target)


def _approval_block(approval: SourceApproval) -> dict[str, Any]:
    payload = dict(approval.payload)
    payload["artifact_sha256"] = approval.sha256
    payload["path"] = str(approval.path)
    return payload


def _candidate_config(
    *,
    live_dir: Path,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    raw = _read_json(ROOT / SOURCE_PATHS["config"])
    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        raise ReadinessError("source config runtime is missing")
    runtime["manifest_path"] = str((live_dir / "candidate-manifest.json").resolve())
    runtime["workflow_path"] = str((ROOT / SOURCE_PATHS["workflow"]).resolve())
    runtime["input_image_path"] = str((ROOT / SOURCE_PATHS["derived"]).resolve())
    runtime["workflow_sha256"] = source_hashes["workflow"]
    raw["runtime"] = runtime
    return raw


def _candidate_manifest(
    *,
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    binding: Any,
    approval: SourceApproval,
    quote_observation: QuoteObservation,
    host_key_fact: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_sha256: str,
    raw_sha256: str,
    raw_size_bytes: int,
    collector_sha256: str,
    plan_sha256: str,
    run_id: str,
    nonce: str,
    created_at: datetime,
    expires_at: datetime,
) -> dict[str, Any]:
    source_manifest = bundle["manifest"]
    candidate_manifest = copy.deepcopy(dict(source_manifest))
    candidate_manifest["manifest_id"] = f"{source_manifest.get('manifest_id')}-LIVE-{run_id}"
    candidate_manifest["status"] = "VERIFIED"
    candidate_manifest["verification"] = {
        "RUNTIME_READINESS": "VERIFIED",
        "SOURCE_STATUS": "DRAFT",
        "GPU_QUALITY_VERIFIED": False,
        "GENERATION_STATE": "NOT_RUN",
    }
    candidate_manifest["candidate"] = {
        "schema": CANDIDATE_SCHEMA,
        "kind": "live_candidate",
        "source_status": "DRAFT",
        "source_promoted": False,
        "source_manifest": {
            "path": str((ROOT / SOURCE_PATHS["manifest"]).resolve()),
            "sha256": source_hashes["manifest"],
            "status": "DRAFT",
        },
        "source_bundle": dict(source_hashes),
        "source_bundle_sha256": source_bundle_digest(source_hashes),
        "source_approval": _approval_block(approval),
        "session": binding.to_dict(),
        "session_binding_sha256": session_binding_digest(binding),
        "run_id": run_id,
        "nonce": nonce,
        "host_key": dict(host_key_fact),
        "collector": {"sha256": collector_sha256, "plan_sha256": plan_sha256},
        "evidence": {
            "path": "runtime-evidence.json",
            "sha256": evidence_sha256,
            "raw_path": "runtime-evidence.raw.json",
            "raw_sha256": raw_sha256,
            "raw_size_bytes": raw_size_bytes,
            "observed_at": evidence.get("observed_at"),
            "expires_at": _iso(expires_at),
        },
        "runtime_evidence": dict(evidence),
        "created_at": _iso(created_at),
        "provenance": {
            "kind": "derived_from_actual_remote_probe",
            "source_manifest_status": "DRAFT",
            "source_manifest_promoted": False,
            "note": "Candidate VERIFIED means source/runtime readiness only; GPU quality and generation remain NOT_RUN.",
        },
    }
    candidate_manifest.setdefault("readiness_notes", []).append(
        "Derived live candidate: source DRAFT is not promoted; GPU quality is not verified; generation has not run."
    )
    return candidate_manifest


def _candidate_binding(
    *,
    source_hashes: Mapping[str, str],
    binding: Any,
    approval: SourceApproval,
    quote_observation: QuoteObservation,
    host_key_fact: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_sha256: str,
    raw_sha256: str,
    raw_size_bytes: int,
    collector_sha256: str,
    plan_sha256: str,
    run_id: str,
    nonce: str,
    created_at: datetime,
    expires_at: datetime,
    file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": BINDING_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "source_manifest": {
            "path": str((ROOT / SOURCE_PATHS["manifest"]).resolve()),
            "sha256": source_hashes["manifest"],
            "status": "DRAFT",
        },
        "source_bundle": dict(source_hashes),
        "source_bundle_sha256": source_bundle_digest(source_hashes),
        "source_approval": _approval_block(approval),
        "quote_observation": {
            "path": str(quote_observation.path),
            "sha256": quote_observation.sha256,
            "observed_at": quote_observation.payload.get("observed_at"),
        },
        "session": binding.to_dict(),
        "session_binding_sha256": session_binding_digest(binding),
        "host_key": dict(host_key_fact),
        "collector": {"sha256": collector_sha256, "plan_sha256": plan_sha256},
        "evidence": {
            "path": "runtime-evidence.json",
            "sha256": evidence_sha256,
            "raw_path": "runtime-evidence.raw.json",
            "raw_sha256": raw_sha256,
            "raw_size_bytes": raw_size_bytes,
            "observed_at": evidence.get("observed_at"),
            "expires_at": _iso(expires_at),
        },
        "files": dict(file_hashes),
        "created_at": _iso(created_at),
        "provenance": {
            "kind": "two_phase_readiness",
            "source_draft_promoted": False,
            "gpu_quality_verified": False,
            "generation_state": "NOT_RUN",
        },
    }


def build_live_candidate(
    *,
    run_dir: Path | str,
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    binding: Any,
    lifecycle: Any,
    approval: SourceApproval,
    quote_observation: QuoteObservation,
    host_key_fact: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_raw: bytes,
    collector_sha256: str,
    plan_sha256: str,
    now: datetime | None = None,
    max_age_seconds: float = DEFAULT_RUNTIME_MAX_AGE_SECONDS,
) -> LiveCandidate:
    if not isinstance(evidence_raw, bytes) or not evidence_raw:
        raise ReadinessError("runtime evidence raw bytes are missing")
    try:
        parsed_raw = json.loads(evidence_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReadinessError("runtime evidence raw bytes are not valid UTF-8 JSON") from exc
    if not isinstance(parsed_raw, dict) or parsed_raw.get("schema") != RUNTIME_EVIDENCE_SCHEMA:
        raise ReadinessError("runtime evidence raw schema is invalid")
    if dict(evidence) != parsed_raw:
        raise ReadinessError("runtime evidence facts differ from the raw probe JSON")
    current = _now(now)
    run_id = getattr(lifecycle, "run_id", None)
    nonce = getattr(lifecycle, "nonce", None)
    if not isinstance(run_id, str) or not isinstance(nonce, str):
        raise ReadinessError("lifecycle run identity is missing")
    if lifecycle.session != binding:
        raise ReadinessError("candidate session does not match lifecycle session")
    if source_hashes != dict(getattr(lifecycle, "bundle_hashes", {})):
        raise ReadinessError("candidate source bundle does not match lifecycle bundle hashes")
    if evidence.get("run_id") != run_id or evidence.get("nonce") != nonce:
        raise ReadinessError("runtime probe run/nonce identity mismatch")
    if evidence.get("session_binding_sha256") != session_binding_digest(binding):
        raise ReadinessError("runtime probe session binding mismatch")
    if evidence.get("source_bundle_sha256") != source_bundle_digest(source_hashes):
        raise ReadinessError("runtime probe source bundle mismatch")
    if evidence.get("source_approval_sha256") != approval.sha256:
        raise ReadinessError("runtime probe source approval mismatch")
    if evidence.get("host_key") != dict(host_key_fact):
        raise ReadinessError("runtime probe host key fact mismatch")
    collector = evidence.get("collector")
    if not isinstance(collector, Mapping) or collector.get("sha256") != collector_sha256:
        raise ReadinessError("runtime probe collector hash mismatch")
    if collector.get("plan_sha256") != plan_sha256:
        raise ReadinessError("runtime probe plan hash mismatch")
    try:
        observed = _parse_moment(evidence.get("observed_at"), "runtime evidence observed_at")
    except ReadinessError as exc:
        raise ReadinessError(str(exc)) from exc
    if observed > current:
        raise ReadinessError("runtime evidence observed_at is in the future")
    if (current - observed).total_seconds() > max_age_seconds:
        raise ReadinessError("runtime evidence is stale")
    expires = min(
        observed + timedelta(seconds=max_age_seconds),
        datetime.fromtimestamp(binding.planned_cancel_at, tz=timezone.utc),
        datetime.fromtimestamp(binding.hard_end_at, tz=timezone.utc),
    )
    if current >= expires:
        raise ReadinessError("runtime evidence expires outside the session deadline")

    raw_sha = _sha_bytes(evidence_raw)
    evidence_payload = dict(evidence)
    evidence_payload["raw_sha256"] = raw_sha
    evidence_payload["raw_size_bytes"] = len(evidence_raw)
    evidence_payload["source_approval_sha256"] = approval.sha256
    evidence_payload["host_key"] = dict(host_key_fact)
    evidence_bytes = _json_text(evidence_payload).encode("utf-8")
    evidence_sha = _sha_bytes(evidence_bytes)

    live_dir = Path(run_dir) / "live"
    manifest_path = live_dir / "candidate-manifest.json"
    config_path = live_dir / "candidate-config.json"
    evidence_path = live_dir / "runtime-evidence.json"
    raw_path = live_dir / "runtime-evidence.raw.json"
    binding_path = live_dir / "candidate-binding.json"

    created_at = observed
    candidate_manifest = _candidate_manifest(
        bundle=bundle,
        source_hashes=source_hashes,
        binding=binding,
        approval=approval,
        quote_observation=quote_observation,
        host_key_fact=host_key_fact,
        evidence=evidence_payload,
        evidence_sha256=evidence_sha,
        raw_sha256=raw_sha,
        raw_size_bytes=len(evidence_raw),
        collector_sha256=collector_sha256,
        plan_sha256=plan_sha256,
        run_id=run_id,
        nonce=nonce,
        created_at=created_at,
        expires_at=expires,
    )
    candidate_config = _candidate_config(live_dir=live_dir, source_hashes=source_hashes)
    manifest_bytes = _json_text(candidate_manifest).encode("utf-8")
    config_bytes = _json_text(candidate_config).encode("utf-8")
    file_hashes = {
        "candidate_manifest": _sha_bytes(manifest_bytes),
        "candidate_config": _sha_bytes(config_bytes),
        "runtime_evidence": _sha_bytes(evidence_bytes),
        "runtime_evidence_raw": _sha_bytes(evidence_raw),
    }
    candidate_binding = _candidate_binding(
        source_hashes=source_hashes,
        binding=binding,
        approval=approval,
        quote_observation=quote_observation,
        host_key_fact=host_key_fact,
        evidence=evidence_payload,
        evidence_sha256=evidence_sha,
        raw_sha256=raw_sha,
        raw_size_bytes=len(evidence_raw),
        collector_sha256=collector_sha256,
        plan_sha256=plan_sha256,
        run_id=run_id,
        nonce=nonce,
        created_at=created_at,
        expires_at=expires,
        file_hashes=file_hashes,
    )
    binding_bytes = _json_text(candidate_binding).encode("utf-8")

    if live_dir.exists():
        existing = load_live_candidate(
            run_dir=run_dir,
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=lifecycle,
            now=current,
        )
        expected = {
            "candidate-manifest.json": manifest_bytes,
            "candidate-config.json": config_bytes,
            "runtime-evidence.json": evidence_bytes,
            "runtime-evidence.raw.json": evidence_raw,
            "candidate-binding.json": binding_bytes,
        }
        for name, data in expected.items():
            path = live_dir / name
            if not path.is_file() or path.read_bytes() != data:
                raise ReadinessError(
                    "existing live candidate is immutable and differs from the new source/evidence/identity"
                )
        return existing

    try:
        _atomic_directory_create(
            live_dir,
            {
                "candidate-manifest.json": manifest_bytes,
                "candidate-config.json": config_bytes,
                "runtime-evidence.json": evidence_bytes,
                "runtime-evidence.raw.json": evidence_raw,
                "candidate-binding.json": binding_bytes,
            },
        )
    except FileExistsError:
        return load_live_candidate(
            run_dir=run_dir,
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=lifecycle,
            now=current,
        )

    validation_config = load_config(config_path)
    issues = validate_candidate_manifest(
        candidate_manifest,
        workflow_path=validation_config.workflow_path,
        actual_workflow_sha256=str(bundle.get("workflow_sha256")),
        config=validation_config,
        manifest_path=manifest_path,
    )
    if issues:
        # The directory was created by this invocation; do not leave an invalid
        # candidate that a later command could mistake for evidence.
        try:
            import shutil

            shutil.rmtree(live_dir)
        except OSError:
            pass
        raise ReadinessError("live candidate validation failed: " + "; ".join(issues))
    return load_live_candidate(
        run_dir=run_dir,
        source_hashes=source_hashes,
        binding=binding,
        lifecycle=lifecycle,
        now=current,
    )


def load_live_candidate(
    *,
    run_dir: Path | str,
    source_hashes: Mapping[str, str],
    binding: Any,
    lifecycle: Any,
    now: datetime | None = None,
) -> LiveCandidate:
    current = _now(now)
    live_dir = Path(run_dir) / "live"
    manifest_path = live_dir / "candidate-manifest.json"
    config_path = live_dir / "candidate-config.json"
    evidence_path = live_dir / "runtime-evidence.json"
    raw_path = live_dir / "runtime-evidence.raw.json"
    binding_path = live_dir / "candidate-binding.json"
    for path in (manifest_path, config_path, evidence_path, raw_path, binding_path):
        if not path.is_file():
            raise ReadinessError(f"live candidate file is missing: {path.name}")
    binding_payload = _read_json(binding_path)
    if binding_payload.get("schema") != BINDING_SCHEMA:
        raise ReadinessError("live candidate binding schema is invalid")
    if binding_payload.get("run_id") != lifecycle.run_id or binding_payload.get("nonce") != lifecycle.nonce:
        raise ReadinessError("live candidate run/nonce differs from lifecycle")
    if binding_payload.get("session") != binding.to_dict():
        raise ReadinessError("live candidate session differs from the requested session")
    if binding_payload.get("source_bundle") != dict(source_hashes):
        raise ReadinessError("live candidate source bundle differs from source")
    if binding_payload.get("source_bundle_sha256") != source_bundle_digest(source_hashes):
        raise ReadinessError("live candidate source bundle digest mismatch")
    if binding_payload.get("session_binding_sha256") != session_binding_digest(binding):
        raise ReadinessError("live candidate session binding digest mismatch")
    files = binding_payload.get("files")
    if not isinstance(files, Mapping):
        raise ReadinessError("live candidate file hashes are missing")
    for name, path in (
        ("candidate_manifest", manifest_path),
        ("candidate_config", config_path),
        ("runtime_evidence", evidence_path),
        ("runtime_evidence_raw", raw_path),
    ):
        expected = files.get(name)
        if not isinstance(expected, str) or not _HEX64.match(expected):
            raise ReadinessError(f"live candidate file hash {name} is invalid")
        if sha256_file(path).lower() != expected.lower():
            raise ReadinessError(f"live candidate file hash mismatch: {name}")
    config = load_config(config_path)
    if Path(config.manifest_path).resolve() != manifest_path.resolve():
        raise ReadinessError("candidate config does not point at the candidate manifest")
    readiness = check_readiness(config_path)
    if not readiness.live_ready:
        raise ReadinessError("candidate readiness gate failed: " + "; ".join(readiness.issues))
    candidate_manifest = _read_json(manifest_path)
    issues = validate_candidate_manifest(
        candidate_manifest,
        workflow_path=config.workflow_path,
        actual_workflow_sha256=readiness.workflow_sha256,
        config=config,
        manifest_path=manifest_path,
    )
    if issues:
        raise ReadinessError("candidate validation failed: " + "; ".join(issues))
    evidence = _read_json(evidence_path)
    try:
        observed = _parse_moment(evidence.get("observed_at"), "runtime evidence observed_at")
    except ReadinessError as exc:
        raise ReadinessError(str(exc)) from exc
    if (current - observed).total_seconds() > DEFAULT_RUNTIME_MAX_AGE_SECONDS:
        raise ReadinessError("runtime evidence is stale")
    return LiveCandidate(
        directory=live_dir,
        manifest_path=manifest_path,
        config_path=config_path,
        evidence_path=evidence_path,
        raw_path=raw_path,
        binding_path=binding_path,
        binding=binding_payload,
    )


def evaluate_generation_gate(
    *,
    run_dir: Path | str,
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    approval: SourceApproval,
    quote_observation: QuoteObservation,
    session_state: Any,
    binding: Any,
    lifecycle: Any,
    host_key_fact: Mapping[str, Any] | None,
    remote: Any,
    expected_fingerprint: str | None,
    supervisor_fact: Mapping[str, Any] | None,
    now: datetime | None = None,
) -> GenerationGate:
    current = _now(now)
    issues: list[str] = []
    candidate: LiveCandidate | None = None
    try:
        candidate = load_live_candidate(
            run_dir=run_dir,
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=lifecycle,
            now=current,
        )
    except ReadinessError as exc:
        issues.append(str(exc))
    source_gate = evaluate_source_gate(
        bundle=bundle,
        source_hashes=source_hashes,
        approval=approval,
        quote_observation=quote_observation,
        session_state=session_state,
        binding=binding,
        host_key_fact=host_key_fact,
        remote=remote,
        expected_fingerprint=expected_fingerprint,
        supervisor_fact=supervisor_fact,
        require_supervisor=True,
        run_id=lifecycle.run_id,
        nonce=lifecycle.nonce,
        now=current,
        quote_max_age_seconds=None,
    )
    issues.extend(source_gate.issues)
    if candidate is not None:
        candidate_payload = _read_json(candidate.manifest_path)
        candidate_block = candidate_payload.get("candidate", {})
        if candidate.binding.get("source_approval", {}).get("artifact_sha256") != approval.sha256:
            issues.append("candidate binding approval artifact hash differs from the current approval")
        if candidate_block.get("source_approval", {}).get("artifact_sha256") != approval.sha256:
            issues.append("candidate manifest approval artifact hash differs from the current approval")
        if candidate.binding.get("source_manifest", {}).get("sha256") != source_hashes.get("manifest"):
            issues.append("candidate source manifest hash differs from source")
        if candidate.binding.get("host_key") != dict(host_key_fact or {}):
            candidate_host = candidate.binding.get("host_key")
            current_host = dict(host_key_fact or {})
            stable_keys = ("fingerprint", "known_hosts_sha256", "host", "user", "port")
            if not isinstance(candidate_host, Mapping) or any(
                candidate_host.get(key) != current_host.get(key) for key in stable_keys
            ):
                issues.append("candidate host key fact differs from the current trusted host key")
        if candidate_payload.get("candidate", {}).get("run_id") != lifecycle.run_id:
            issues.append("candidate run_id differs from lifecycle")
        if candidate_payload.get("candidate", {}).get("nonce") != lifecycle.nonce:
            issues.append("candidate nonce differs from lifecycle")
    facts = {
        **dict(source_gate.facts),
        "phase": "GENERATION",
        "candidate_present": candidate is not None,
        "candidate_manifest_sha256": sha256_file(candidate.manifest_path) if candidate else None,
        "candidate_config_sha256": sha256_file(candidate.config_path) if candidate else None,
        "candidate_evidence_sha256": sha256_file(candidate.evidence_path) if candidate else None,
        "candidate_raw_sha256": sha256_file(candidate.raw_path) if candidate else None,
        "candidate_binding_sha256": sha256_file(candidate.binding_path) if candidate else None,
    }
    return GenerationGate(passed=not issues, issues=tuple(issues), facts=facts)


def build_runtime_evidence_plan(
    *,
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    approval: SourceApproval,
    quote_observation: QuoteObservation,
    host_key_fact: Mapping[str, Any],
    binding: Any,
    lifecycle: Any,
    workspace: str = "/workspace",
    venv_python: str | None = None,
    comfy_dir: str | None = None,
    object_info_path: str | None = None,
    system_stats_path: str | None = None,
    preflight_path: str | None = None,
    owned_root: str | None = None,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    workspace_path = PurePosixPath(workspace)
    if not (
        workspace_path.is_absolute()
        or Path(workspace).is_absolute()
    ) or any(part in ("", ".", "..") for part in workspace_path.parts[1:]):
        raise ReadinessError("workspace must be an absolute normalized path")
    workspace_text = str(workspace_path)
    root_text = str(PurePosixPath(owned_root)) if owned_root is not None else workspace_text
    root_path = PurePosixPath(root_text)
    if not (
        root_path.is_absolute()
        or Path(root_text).is_absolute()
    ) or any(part in ("", ".", "..") for part in root_path.parts[1:]):
        raise ReadinessError("owned_root must be an absolute normalized path")
    venv = venv_python or str(workspace_path / "p1-venv" / "bin" / "python")
    comfy = comfy_dir or str(workspace_path / "ComfyUI")
    object_info = object_info_path or str(workspace_path / "p1-evidence" / "object_info.json")
    system_stats = system_stats_path or str(workspace_path / "p1-evidence" / "system_stats.json")
    preflight_file = preflight_path or str(root_path / "scripts" / "p1" / "preflight.py")
    if deadline_epoch is None:
        deadline_epoch = min(binding.planned_cancel_at, binding.hard_end_at)
    manifest = bundle["manifest"]
    models = []
    for model in manifest.get("models", []):
        if not isinstance(model, Mapping):
            raise ReadinessError("source model pin is invalid")
        models.append(
            {
                "id": model.get("id"),
                "target": model.get("target"),
                "size_bytes": model.get("size_bytes"),
                "sha256": model.get("sha256"),
            }
        )
    config = bundle.get("config")
    binding_raw = getattr(config, "binding_raw", {})
    node_classes = binding_raw.get("node_types") if isinstance(binding_raw, Mapping) else None
    if not isinstance(node_classes, Mapping):
        raise ReadinessError("source workflow binding node types are missing")
    workflow = bundle.get("workflow")
    if not isinstance(workflow, Mapping):
        raise ReadinessError("source workflow is missing")
    graph_inputs: list[dict[str, str]] = []
    for node_id, class_name in node_classes.items():
        node = workflow.get(str(node_id))
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if not isinstance(inputs, Mapping):
            raise ReadinessError(f"workflow node {node_id} inputs are missing")
        for input_name in inputs:
            text = str(input_name)
            if str(class_name) == "SaveVideo" and (
                text == "format" or text.startswith("format.")
            ):
                continue
            graph_inputs.append(
                {
                    "node_id": str(node_id),
                    "class": str(class_name),
                    "input": text,
                }
            )
    graph_inputs.sort(key=lambda item: (item["node_id"], item["class"], item["input"]))
    expected_classes = sorted({str(value) for value in node_classes.values()})
    return {
        "schema": RUNTIME_PLAN_SCHEMA,
        "run_id": lifecycle.run_id,
        "nonce": lifecycle.nonce,
        "session_binding_sha256": session_binding_digest(binding),
        "source_bundle_sha256": source_bundle_digest(source_hashes),
        "source_approval_sha256": approval.sha256,
        "host_key": dict(host_key_fact),
        "workspace": workspace_text,
        "owned_root": root_text,
        "venv_python": venv,
        "comfy_dir": comfy,
        "object_info_path": object_info,
        "system_stats_path": system_stats,
        "require_system_stats": True,
        "preflight_path": preflight_file,
        "models": models,
        "required_classes": expected_classes,
        "required_inputs": {name: sorted(values) for name, values in preflight.REQUIRED_INPUTS.items()},
        "graph_inputs": graph_inputs,
        "node_classes": {str(key): str(value) for key, value in node_classes.items()},
        "dynamic_path": ["format", "codec", "encoding"],
        "expected_comfy_commit": manifest.get("comfyui", {}).get("commit"),
        "deadline_epoch": float(deadline_epoch),
        "max_bytes": RUNTIME_PLAN_MAX_BYTES,
    }


__all__ = [
    "BINDING_SCHEMA",
    "CURRENT_APPROVAL_DOC",
    "DEFAULT_APPROVAL_VALID_HOURS",
    "DEFAULT_QUOTE_MAX_AGE_SECONDS",
    "DEFAULT_RUNTIME_MAX_AGE_SECONDS",
    "EXPECTED_V3_MODELS",
    "FROZEN_SOURCE_BUNDLE",
    "GenerationGate",
    "LiveCandidate",
    "QuoteObservation",
    "ReadinessError",
    "SOURCE_PATHS",
    "SourceApproval",
    "SourceGate",
    "build_live_candidate",
    "build_runtime_evidence_plan",
    "create_quote_observation",
    "create_source_approval",
    "evaluate_generation_gate",
    "evaluate_source_gate",
    "load_live_candidate",
    "load_quote_observation",
    "load_source_approval",
    "make_host_key_fact",
    "make_supervisor_fact",
    "quote_observation_path",
    "session_binding_digest",
    "source_bundle_digest",
    "source_bundle_hashes",
    "validate_source_bundle",
]
