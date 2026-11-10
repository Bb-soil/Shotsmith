"""Single offline-first operator entrypoint for P1 v3.

Commands: ``plan``, ``readiness``, ``fresh-quote``, ``owner-risk-approve``,
``approve-source``, ``session-bind``, ``prepare``, ``status``, ``run-once``,
``collect``.

Two phases are explicit: source/bootstrap readiness allows DRAFT and performs
owner approval/session/SSH/supervisor checks; generation readiness requires an
immutable per-run live candidate built from actual remote runtime facts. The
source DRAFT manifest is never promoted or rewritten. ``run-once`` uses the
existing ``anime_studio serve`` readiness gate and never creates a second queue.
Real remote execution remains gated and is reported as NOT_RUN/NOT_VERIFIED.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anime_studio.lock import InstanceLock
from anime_studio.p1_config import (
    check_readiness,
    load_config,
    load_manifest,
    sha256_file,
)
from anime_studio.workflow import load_workflow
from scripts.p1 import clore_export, clore_session, clore_startup, marketplace_quote, owner_risk, preflight, setup_plan, v3_backend, v3_bootstrap, v3_lifecycle, v3_readiness
from scripts.p1.remote import (
    RemoteError,
    RemoteIdentity,
    RemoteRunner,
    RemoteUnavailable,
    _validate_remote_absolute_path,
    run_bounded_process,
    read_worker_state,
    write_worker_state,
)

TASK_ID = "P1-V3-AUTOMATION-20261022"
SCHEMA = "anime-video-studio/p1-v3-operator/v1"
RUN_ROOT = ROOT / "data" / "p1-v3"
V3_CONFIG = ROOT / "config" / "p1.wan22-5b.shot01-v3.draft.json"
V3_WORKFLOW = ROOT / "workflows" / "P1" / "wan22-5b-shot01-v3.api.draft.json"
V3_MANIFEST = ROOT / "manifests" / "P1.wan22-5b-shot01-v3.draft.json"
V3_JOB = ROOT / "config" / "p1.shot01-v3.job.json"
V3_SOURCE = ROOT / "assets" / "frames" / "shot-01-start-v3.png"
V3_DERIVED = ROOT / "assets" / "frames" / "derived" / "shot-01-start-v3-704x1280.png"
V3_POLICY = ROOT / "assets" / "frames" / "derived" / "shot-01-start-v3-704x1280.policy.json"
SOURCE_SHA256 = "5e3029aa7edb9207c1503635606beeeed9b012995b2a296616668c84fe68083a"
DERIVED_SHA256 = "925eccbe02a49b74bd2e8cd190fe26bd9557628bb996149f093b894136c53add"
EXPECTED_VIDEO_WIDTH = 704
EXPECTED_VIDEO_HEIGHT = 1280
EXPECTED_VIDEO_FRAMES = 121
EXPECTED_VIDEO_FPS = 24
EXPECTED_VIDEO_DURATION = EXPECTED_VIDEO_FRAMES / EXPECTED_VIDEO_FPS
DECODE_TIMEOUT_SECONDS = 120.0
NOT_EXPORTED_EVIDENCE = (
    "history.json",
    "setup.log",
    "download.log",
    "render.log",
    "status.json",
    "costs.json",
)
COLLECT_ITEMS = (
    ("artifact.mp4", "artifact.mp4", True),
    ("input.png", "inputs/input.png", True),
    ("workflow.json", "inputs/workflow.json", True),
    ("manifest.json", "inputs/manifest.json", True),
    ("job.json", "inputs/job.json", True),
    ("policy.json", "inputs/policy.json", True),
    ("request.json", "request.json", False),
    ("runtime.json", "runtime/runtime.json", False),
    ("runtime/runtime-evidence.json", "runtime/runtime-evidence.json", False),
    ("runtime/runtime-evidence.raw.json", "runtime/runtime-evidence.raw.json", False),
    ("runtime/candidate-manifest.json", "runtime/candidate-manifest.json", False),
    ("runtime/candidate-config.json", "runtime/candidate-config.json", False),
    ("runtime/candidate-binding.json", "runtime/candidate-binding.json", False),
)


class OperatorError(Exception):
    """A bounded operator failure; callers must not claim success."""


class ReadinessBlocked(OperatorError):
    """The existing real-mode readiness gate refused live work."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path | str) -> str:
    return sha256_file(Path(path))


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
        newline="\n",
    )
    temp_name = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            temp_name.unlink(missing_ok=True)
        except OSError:
            pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, _json_text(payload))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise OperatorError(f"JSON {path} must be an object")
    return data


def _prompts_from_doc(path: Path = ROOT / "docs" / "P1-shot-01-v3.md") -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    fence = chr(96) * 3
    parts = text.split(fence)
    blocks = [part for part in parts[1::2] if part.startswith("text")]
    if len(blocks) != 2:
        raise OperatorError("v3 prompt document must contain exactly two text blocks")
    positive = blocks[0].split("\n", 1)[1].strip()
    negative = blocks[1].split("\n", 1)[1].strip()
    return positive, negative


def load_bundle() -> dict[str, Any]:
    config = load_config(V3_CONFIG)
    workflow, workflow_sha = load_workflow(V3_WORKFLOW)
    manifest = load_manifest(V3_MANIFEST)
    job = _read_json(V3_JOB)
    if not V3_DERIVED.is_file():
        raise OperatorError("derived v3 frame is missing")
    if not V3_POLICY.is_file():
        raise OperatorError("derived v3 frame policy is missing")
    source_sha = _sha_file(V3_SOURCE)
    derived_sha = _sha_file(V3_DERIVED)
    if source_sha != SOURCE_SHA256:
        raise OperatorError("v3 source SHA256 does not match the pin")
    if derived_sha != DERIVED_SHA256:
        raise OperatorError("v3 derived SHA256 does not match the pin")
    if config.workflow_sha256 != workflow_sha:
        raise OperatorError("v3 config workflow SHA256 does not match workflow file")
    if manifest.get("workflow", {}).get("sha256") != workflow_sha:
        raise OperatorError("v3 manifest workflow SHA256 does not match workflow file")
    positive, negative = _prompts_from_doc()
    workflow_positive = workflow.get("6", {}).get("inputs", {}).get("text")
    workflow_negative = workflow.get("7", {}).get("inputs", {}).get("text")
    if workflow_positive != positive:
        raise OperatorError("workflow node 6 positive prompt is not exact")
    if workflow_negative != negative:
        raise OperatorError("workflow node 7 negative prompt is not exact")
    if job.get("prompt") != positive:
        raise OperatorError("v3 job prompt is not exact")
    if V3_DERIVED.name != workflow.get("56", {}).get("inputs", {}).get("image"):
        raise OperatorError("workflow image binding does not match the derived frame")
    policy = _read_json(V3_POLICY)
    if policy.get("source", {}).get("sha256") != source_sha:
        raise OperatorError("v3 policy source SHA256 does not match the pin")
    if policy.get("target", {}).get("sha256") != derived_sha:
        raise OperatorError("v3 policy derived SHA256 does not match the pin")
    if policy.get("crop", {}).get("box_left_top_right_bottom") != [14, 6, 927, 1666]:
        raise OperatorError("v3 policy crop box does not match the pin")
    if policy.get("crop", {}).get("width") != 913 or policy.get("crop", {}).get("height") != 1660:
        raise OperatorError("v3 policy crop dimensions do not match the pin")
    if policy.get("target", {}).get("width") != 704 or policy.get("target", {}).get("height") != 1280:
        raise OperatorError("v3 policy target dimensions do not match the pin")
    if policy.get("resize", {}).get("anisotropic") is not False:
        raise OperatorError("v3 policy must be non-anisotropic")
    readiness = check_readiness(V3_CONFIG)
    return {
        "config": config,
        "workflow": workflow,
        "workflow_sha256": workflow_sha,
        "manifest": manifest,
        "job": job,
        "positive": positive,
        "negative": negative,
        "source_sha256": source_sha,
        "derived_sha256": derived_sha,
        "policy": policy,
        "readiness": readiness,
    }


def _setup_plan(bundle: Mapping[str, Any], workspace: str) -> setup_plan.SetupPlan:
    return setup_plan.build_setup_plan(
        bundle["manifest"],
        workspace=workspace,
        manifest_path=str(V3_MANIFEST.relative_to(ROOT)),
    )


def _preflight_lines(workspace: str) -> tuple[str, ...]:
    return preflight.build_preflight_plan(workspace=workspace)


def _collect_plan(local_dir: Path, staging: Path) -> clore_export.ExportPlan:
    items = tuple(
        clore_export.ExportItem(source=source, target=target, required=required)
        for source, target, required in COLLECT_ITEMS
    )
    return clore_export.build_export_plan(local_dir, source_dir=staging, items=items)


def build_plan(*, workspace: str = "/workspace") -> dict[str, Any]:
    bundle = load_bundle()
    readiness = bundle["readiness"]
    setup = _setup_plan(bundle, workspace)
    startup = clore_startup.assess_startup()
    plan = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PLAN_OFFLINE",
        "v3": {
            "source": {
                "path": str(V3_SOURCE.relative_to(ROOT)),
                "sha256": bundle["source_sha256"],
                "width": 941,
                "height": 1672,
            },
            "derived": {
                "path": str(V3_DERIVED.relative_to(ROOT)),
                "sha256": bundle["derived_sha256"],
                "width": 704,
                "height": 1280,
                "crop_box": [14, 6, 927, 1666],
                "crop_size": [913, 1660],
                "resize": "integer_area_average_rgb8",
                "anisotropic": False,
            },
            "workflow": {
                "path": str(V3_WORKFLOW.relative_to(ROOT)),
                "sha256": bundle["workflow_sha256"],
                "prompt_positive_exact": True,
                "prompt_negative_exact": True,
            },
            "manifest": {
                "path": str(V3_MANIFEST.relative_to(ROOT)),
                "status": bundle["manifest"].get("status"),
            },
            "params": {
                "seed": 424242,
                "steps": 20,
                "cfg": 5,
                "frames": 121,
                "fps": 24,
                "width": 704,
                "height": 1280,
                "sampler": "uni_pc",
                "scheduler": "simple",
                "lora": "NONE",
                "interpolation": "NONE",
            },
        },
        "readiness": {
            "live_ready": readiness.live_ready,
            "issues": list(readiness.issues),
            "gate": "anime_studio/p1_config.check_readiness",
        },
        "financial": {
            "status": "OWNER_RISK_POLICY_AVAILABLE_NO_ADMISSION",
            "session_bound": False,
            "provider_billing_verified": False,
            "policy_forecast_60m_usd": str(owner_risk.TOTAL_60M),
            "known_unknowns": [
                "minimum_billed_duration",
                "additional_mandatory_fees",
                "tax",
                "fx",
                "payment_fees",
            ],
        },
        "setup": {
            "downloads": [
                {
                    "model_id": spec.model_id,
                    "target": spec.target,
                    "size_bytes": spec.size_bytes,
                    "sha256": spec.sha256,
                }
                for spec in setup.downloads
            ],
            "base_torch_constraints": True,
        },
        "preflight": {
            "commands": list(_preflight_lines(workspace)),
            "shell_execution": "NOT_RUN_OFFLINE",
        },
        "startup": {
            "live_ready": startup.live_ready,
            "blockers": list(startup.blockers),
        },
        "collect": {
            "items": [item.target for item in _collect_plan(RUN_ROOT / "_plan", RUN_ROOT / "_plan" / "staging").items],
            "critical_first": "artifact.mp4",
            "decoder": _decoder_status(),
        },
        "live": {
            "status": "LIVE_NOT_RUN",
            "reason": "offline plan only; no SSH/provider/backend call",
        },
        "not_implemented": [
            "real SSH/provider/GPU/weights/install/execution was not run on this host",
            "DRAFT readiness still blocks real live mutation",
            "real ComfyUI output decode requires ffprobe+ffmpeg on this host",
            "exactly-once ComfyUI delivery and provider hard billing cap are not promised",
        ],
        "next_action": (
            "review plan; run prepare for an offline runbook, "
            "then resolve readiness blockers before any live request"
        ),
    }
    if not readiness.live_ready:
        plan["next_action"] = (
            "fix readiness issues or keep DRAFT; run-once --live will fail "
            "before mutation until the existing gate is satisfied"
        )
    return plan


def _decoder_status() -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    pyav = False
    try:
        import av  # type: ignore

        pyav = True
        _ = av
    except Exception:
        pyav = False
    if ffprobe and ffmpeg:
        return {
            "status": "AVAILABLE",
            "ffprobe": ffprobe,
            "ffmpeg": ffmpeg,
            "pyav": pyav,
            "full_decode": "ffprobe_metadata_plus_ffmpeg_full_decode",
        }
    return {
        "status": "NOT_VERIFIED",
        "ffprobe": ffprobe or "NOT_FOUND",
        "ffmpeg": ffmpeg or "NOT_FOUND",
        "pyav": pyav,
        "note": (
            "full decode requires ffprobe+ffmpeg; PyAV is not implemented; "
            "structural MP4 check is not a decode"
        ),
    }


def _disk_status(workspace: str, required_bytes: int) -> dict[str, Any]:
    path = Path(workspace)
    if not path.exists():
        return {
            "status": "NOT_RUN",
            "workspace": workspace,
            "required_bytes": required_bytes,
            "reason": "workspace path does not exist on this host",
        }
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {
            "status": "NOT_VERIFIED",
            "workspace": workspace,
            "required_bytes": required_bytes,
            "reason": f"disk usage failed: {type(exc).__name__}",
        }
    if usage.free < required_bytes:
        raise OperatorError(
            f"insufficient disk at {workspace}: {usage.free} < {required_bytes}"
        )
    return {
        "status": "OK",
        "workspace": workspace,
        "required_bytes": required_bytes,
        "free_bytes": usage.free,
    }


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _assert_clean_path(path: Path) -> None:
    """Reject symlink/junction/dangling components from anchor to path."""

    current = path
    while True:
        if current.exists() and _is_link(current):
            raise OperatorError(f"symlink/junction component is not allowed: {current}")
        if current.is_symlink():
            raise OperatorError(f"dangling symlink component is not allowed: {current}")
        if current.parent == current:
            break
        current = current.parent


def _validate_bundle_identity(prior: Any, expected: Mapping[str, str]) -> None:
    keys = set(expected)
    if (
        not isinstance(prior, dict)
        or set(prior) != keys
        or any(
            not isinstance(prior.get(key), str) or len(prior.get(key, "")) != 64
            for key in keys
        )
    ):
        raise OperatorError(
            "existing run is missing/incomplete immutable bundle identity; "
            "explicit new run is required"
        )
    if prior != dict(expected):
        raise OperatorError("existing run bundle/config hashes do not match current pins")


def _prevalidate_run_root(
    target: Path, bundle_hashes: Mapping[str, str] | None = None
) -> None:
    """Validate existing run identity before creating/acquiring its lock."""

    if not target.exists():
        return
    if not target.is_dir():
        raise OperatorError("run path exists but is not a directory")
    path = _validate_owned_path(target, target / "run.json", name="run.json")
    if path.is_file():
        payload = _read_json(path)
        if payload.get("schema") != SCHEMA or payload.get("task_id") != TASK_ID:
            raise OperatorError("existing run.json schema/task identity does not match")
        if payload.get("run_id") != target.name:
            raise OperatorError("existing run.json run_id does not match the run directory")
        if bundle_hashes is not None:
            _validate_bundle_identity(payload.get("bundle_hashes"), bundle_hashes)
        return
    if any(target.iterdir()):
        raise OperatorError("nonempty run without a valid run.json")


def _assert_no_link_components(path: Path, *, stop: Path) -> None:
    current = path
    stop_resolved = stop.resolve()
    while True:
        if current.exists() and _is_link(current):
            raise OperatorError(f"symlink/junction component is not allowed: {current}")
        if current.resolve() == stop_resolved:
            break
        if current.parent == current:
            break
        current = current.parent


def _validate_owned_path(target: Path, path: Path, *, name: str) -> Path:
    target_resolved = target.resolve()
    candidate = path.resolve()
    if candidate != target_resolved and target_resolved not in candidate.parents:
        raise OperatorError(f"{name} escapes the run namespace")
    _assert_no_link_components(path, stop=target)
    return candidate


def _bundle_hashes(bundle: Mapping[str, Any]) -> dict[str, str]:
    return {
        "config": _sha_file(V3_CONFIG),
        "workflow": _sha_file(V3_WORKFLOW),
        "manifest": _sha_file(V3_MANIFEST),
        "job": _sha_file(V3_JOB),
        "source": _sha_file(V3_SOURCE),
        "derived": _sha_file(V3_DERIVED),
        "policy": _sha_file(V3_POLICY),
    }


SESSION_OWNER_ROOT = RUN_ROOT / "session-owners"

_LIVE_PHASE_ORDER = {
    "BOUND_LIVE": 0,
    "UNBOUND_OFFLINE": 0,
    "SUPERVISOR_STARTED": 1,
    "ROOT_READY": 2,
    "STAGED": 3,
    "SETUP_COMPLETE": 4,
    "DOWNLOAD_STARTED": 5,
    "DOWNLOAD_RUNNING": 6,
    "DOWNLOAD_FAILED": 6,
    "DOWNLOAD_COMPLETE": 7,
    "OFFLINE_PREFLIGHT_COMPLETE": 8,
    "COMFY_STARTED": 9,
    "PREFLIGHT_COMPLETE": 10,
    "TUNNEL_READY": 11,
    "READY": 12,
    "GENERATION_READY": 13,
}


def _pipeline_rank(phase: str) -> int:
    return _LIVE_PHASE_ORDER.get(phase, 0)


def _advance_live_pipeline(
    *,
    target: Path,
    lifecycle_path: Path,
    lifecycle_value: v3_lifecycle.LifecycleState,
    session_binding: v3_lifecycle.SessionBinding,
    remote: RemoteIdentity,
    expected_fingerprint: str | None = None,
    workspace: str,
    bundle: Mapping[str, Any],
    setup_text: str,
    preflight_text: str,
    runner: Any | None = None,
    execute_downloads: bool,
    resume_dead: bool = False,
    tunnel_factory: Callable[..., Any] | None = None,
) -> v3_lifecycle.LifecycleState:
    """Advance the owned live pipeline one durable phase at a time.

    Order is fixed: supervisor ack (already recorded) -> owned root -> pinned
    setup/base-torch -> download complete -> preflight -> ComfyUI -> tunnel ->
    READY.  Re-running the same CLI resumes at the recorded phase instead of
    repeating a completed remote action.
    """

    runner = runner or RemoteRunner()
    root = str(PurePosixPath(workspace))
    helper_path = str(PurePosixPath(root) / "scripts" / "p1" / "v3_remote_fetch_helper.py")
    remote_deadline = min(
        float(session_binding.planned_cancel_at),
        float(session_binding.hard_end_at),
    ) - 60.0
    if time.time() >= remote_deadline:
        raise OperatorError("session deadline is too close for remote bootstrap")
    remote_monotonic = time.monotonic() + max(0.05, remote_deadline - time.time())
    phase = lifecycle_value.phase
    rank = _pipeline_rank(phase)
    pipeline = dict(lifecycle_value.pipeline or {})

    def advance(new_phase: str, **extra: Any) -> v3_lifecycle.LifecycleState:
        nonlocal lifecycle_value, phase, rank, pipeline
        pipeline.update(extra)
        lifecycle_value = v3_lifecycle.update_phase(
            lifecycle_path,
            lifecycle_value,
            phase=new_phase,
            pipeline=pipeline,
        )
        phase = new_phase
        rank = _pipeline_rank(new_phase)
        return lifecycle_value

    if rank < 2:
        prepared = v3_bootstrap.prepare_remote_root(
            runner=runner,
            identity=remote,
            remote_root=root,
            timeout=v3_lifecycle.clip_remote_timeout(remote_deadline, 30.0),
            deadline_monotonic=remote_monotonic,
            run_id=target.name,
            nonce=lifecycle_value.nonce,
            session=session_binding.to_dict(),
        )
        advance("ROOT_READY", root=prepared["remote_root"], disk_kb=prepared["disk_kb"])
    if rank < 3:
        v3_bootstrap.stage_bootstrap(
            runner=runner,
            identity=remote,
            remote_root=root,
            setup_text=setup_text,
            preflight_text=preflight_text,
            launch_manifest=v3_bootstrap.comfy_launch_manifest(
                remote_root=root,
                workspace=workspace,
                run_id=target.name,
                nonce=lifecycle_value.nonce,
            ),
            timeout=v3_lifecycle.clip_remote_timeout(remote_deadline, 60.0),
            deadline_monotonic=remote_monotonic,
        )
        advance("STAGED")
    if rank < 4:
        v3_bootstrap.run_setup(
            runner=runner,
            identity=remote,
            remote_root=root,
            timeout=v3_lifecycle.clip_remote_timeout(remote_deadline, 3600.0),
            deadline_monotonic=remote_monotonic,
        )
        advance("SETUP_COMPLETE")
    if execute_downloads and rank < 7:
        setup_plan_obj = _setup_plan(bundle, workspace)
        required = sum(spec.size_bytes for spec in setup_plan_obj.downloads)
        deadline_epoch = min(
            float(session_binding.planned_cancel_at),
            float(session_binding.hard_end_at),
        ) - 60.0
        if time.time() >= deadline_epoch:
            raise OperatorError("planned cancellation deadline is too close for downloads")
        job_payload = {
            "schema": "anime-video-studio/p1-v3-download-job/v1",
            "run_id": target.name,
            "nonce": lifecycle_value.nonce,
            "deadline_epoch": deadline_epoch,
            "manifest_sha256": _bundle_hashes(bundle)["manifest"],
            "required_bytes": required,
        }
        download_job_path = _validate_owned_path(
            target, target / "download-job.json", name="download job"
        )
        _write_json(download_job_path, job_payload)
        worker = v3_lifecycle.RemoteDownloadWorker(
            runner=runner,
            identity=remote,
            remote_root=root,
            run_id=target.name,
            nonce=lifecycle_value.nonce,
            local_worker=ROOT / "scripts" / "p1" / "v3_download_worker.py",
            local_setup_plan=ROOT / "scripts" / "p1" / "setup_plan.py",
            local_manifest=V3_MANIFEST,
        )
        staged = worker.stage(
            payload=job_payload,
            manifest_sha256=job_payload["manifest_sha256"],
            required_bytes=required,
            timeout=v3_lifecycle.clip_remote_timeout(deadline_epoch, 60.0),
            deadline_monotonic=remote_monotonic,
            resume_dead=resume_dead,
        )
        advance(
            "DOWNLOAD_STARTED",
            launch_manifest=staged["launch_manifest"],
            worker=staged["worker"],
            payload=staged["payload"],
        )
        started = worker.start(
            launch_manifest=staged["launch_manifest"],
            timeout=v3_lifecycle.clip_remote_timeout(deadline_epoch, 60.0),
            deadline_monotonic=remote_monotonic,
            resume_dead=resume_dead,
        )
        advance("DOWNLOAD_RUNNING", started=started.get("status"))
        if started.get("status") == "ATTACHED_COMPLETE":
            state = started["state"]
        else:
            state = worker.wait(
                deadline_epoch=deadline_epoch,
                poll_seconds=1.0,
            )
        if state["status"].get("status") != "COMPLETE":
            advance(
                "DOWNLOAD_FAILED",
                progress=state.get("progress"),
                status=state.get("status"),
            )
            raise OperatorError("remote download worker did not complete")
        advance(
            "DOWNLOAD_COMPLETE",
            progress=state.get("progress"),
            heartbeat=state.get("heartbeat"),
            owner=state.get("owner"),
        )
    if execute_downloads and rank < 8:
        v3_bootstrap.run_preflight(
            runner=runner,
            identity=remote,
            remote_root=root,
            workspace=workspace,
            timeout=v3_lifecycle.clip_remote_timeout(remote_deadline, 600.0),
            deadline_monotonic=remote_monotonic,
            mode="offline",
        )
        advance("OFFLINE_PREFLIGHT_COMPLETE")
    if execute_downloads and rank < 9:
        comfy = v3_bootstrap.start_comfy(
            runner=runner,
            identity=remote,
            remote_root=root,
            run_id=target.name,
            nonce=lifecycle_value.nonce,
            timeout=v3_lifecycle.clip_remote_timeout(remote_deadline, 120.0),
            helper_path=helper_path,
            deadline_monotonic=remote_monotonic,
        )
        advance("COMFY_STARTED", comfy=comfy)
    if execute_downloads and rank < 10:
        v3_bootstrap.run_preflight(
            runner=runner,
            identity=remote,
            remote_root=root,
            workspace=workspace,
            timeout=v3_lifecycle.clip_remote_timeout(remote_deadline, 600.0),
            deadline_monotonic=remote_monotonic,
            mode="api",
        )
        advance("PREFLIGHT_COMPLETE")
    if execute_downloads and rank < 11:
        tunnel_owner = _validate_owned_path(
            target, target / "tunnel-owner.json", name="tunnel owner"
        )
        factory = tunnel_factory or v3_bootstrap.OwnedTunnel
        if tunnel_owner.is_file():
            tunnel = factory.attach(
                identity=remote,
                owner_path=tunnel_owner,
                run_id=target.name,
                nonce=lifecycle_value.nonce,
                expected_fingerprint=expected_fingerprint,
                deadline_monotonic=remote_monotonic,
            )
            attached = True
        else:
            tunnel = factory(
                identity=remote,
                local_port=v3_bootstrap.DEFAULT_LOCAL_TUNNEL_PORT,
                remote_port=v3_bootstrap.DEFAULT_COMFY_REMOTE_PORT,
                log_path=_validate_owned_path(
                    target, target / "tunnel.log", name="tunnel log"
                ),
                run_id=target.name,
                nonce=lifecycle_value.nonce,
                owner_path=tunnel_owner,
                expected_fingerprint=expected_fingerprint,
            ).start(deadline_monotonic=remote_monotonic)
            attached = False
        health = tunnel.health(
            timeout=v3_lifecycle.clip_remote_timeout(remote_deadline, 60.0),
            deadline_monotonic=remote_monotonic,
        )
        tunnel.verify_mapping(deadline_monotonic=remote_monotonic)
        advance(
            "TUNNEL_READY",
            tunnel={
                "owner": str(tunnel_owner),
                "local_port": v3_bootstrap.DEFAULT_LOCAL_TUNNEL_PORT,
                "remote_port": v3_bootstrap.DEFAULT_COMFY_REMOTE_PORT,
                "attached": attached,
                "health": bool(health),
            },
        )
    if execute_downloads and rank < 12:
        advance("READY")
    return lifecycle_value



def _financial_status(binding: v3_lifecycle.SessionBinding | None) -> str:
    if binding is None:
        return "NOT_BOUND"
    if binding.owner_risk_admission_sha256:
        return "OWNER_ACCEPTED_FINANCIAL_RISK"
    return "STRICT_QUOTE"


def _session_key(binding: v3_lifecycle.SessionBinding) -> str:
    # Stable claim key: the immutable order/server/created_at identity.  Cap,
    # quote hash and deadlines are compared from the full binding under lock,
    # so a second quote cannot claim the same session via a different hash.
    payload = json.dumps(
        {
            "order_id": binding.order_id,
            "server_id": binding.server_id,
            "created_at": binding.created_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha_bytes(payload.encode("utf-8"))


def _register_session_owner(
    binding: v3_lifecycle.SessionBinding, *, run_dir: Path, nonce: str
) -> Path:
    SESSION_OWNER_ROOT.mkdir(parents=True, exist_ok=True)
    path = SESSION_OWNER_ROOT / f"{_session_key(binding)}.json"
    expected = {
        "schema": "anime-video-studio/p1-v3-session-owner/v1",
        "run_dir": str(run_dir),
        "nonce": nonce,
        "session": binding.to_dict(),
    }
    try:
        lock = InstanceLock(path.with_suffix(".lock")).acquire()
    except Exception as exc:
        raise OperatorError("session registry is busy; no second claimant accepted") from exc
    try:
        if path.is_file():
            prior = _read_json(path)
            if prior.get("session") != binding.to_dict():
                raise OperatorError(
                    "session identity is claimed by a different full binding"
                )
            if prior.get("run_dir") != str(run_dir) or prior.get("nonce") != nonce:
                raise OperatorError("session identity is owned by another run/nonce")
            return path
        data = _json_text(expected).encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        return path
    finally:
        lock.release()


def _default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RUN_ROOT / f"run-{stamp}-{uuid.uuid4().hex[:8]}"


def _supervisor_fact_from_disk(supervisor: v3_lifecycle.CloreSupervisor) -> dict[str, Any]:
    record_path = Path(supervisor.record_path)
    ack_path = Path(supervisor.ack_path)
    heartbeat_path = Path(supervisor.heartbeat_path)
    if not record_path.is_file() or not ack_path.is_file() or not heartbeat_path.is_file():
        raise OperatorError("supervisor ack/record/heartbeat is incomplete")
    return v3_readiness.make_supervisor_fact(
        _read_json(record_path),
        _read_json(ack_path),
        _read_json(heartbeat_path),
    )


def _candidate_facts(candidate: v3_readiness.LiveCandidate) -> dict[str, Any]:
    return {
        "directory": str(candidate.directory),
        "manifest": str(candidate.manifest_path),
        "config": str(candidate.config_path),
        "evidence": str(candidate.evidence_path),
        "raw": str(candidate.raw_path),
        "binding": str(candidate.binding_path),
        "manifest_sha256": _sha_file(candidate.manifest_path),
        "config_sha256": _sha_file(candidate.config_path),
        "evidence_sha256": _sha_file(candidate.evidence_path),
        "raw_sha256": _sha_file(candidate.raw_path),
        "binding_sha256": _sha_file(candidate.binding_path),
        "observed_at": candidate.binding.get("evidence", {}).get("observed_at"),
        "expires_at": candidate.binding.get("evidence", {}).get("expires_at"),
        "GPU_QUALITY_VERIFIED": False,
        "GENERATION_STATE": "NOT_RUN",
    }


def _assert_candidate_bound_to_lifecycle(
    lifecycle: v3_lifecycle.LifecycleState,
    candidate: v3_readiness.LiveCandidate,
) -> None:
    pipeline = lifecycle.pipeline if isinstance(lifecycle.pipeline, Mapping) else {}
    facts = pipeline.get("candidate")
    if not isinstance(facts, Mapping):
        raise ReadinessBlocked(
            "GENERATION_GATE_BLOCKED: lifecycle has no candidate binding facts"
        )
    actual = _candidate_facts(candidate)
    for key in (
        "directory",
        "manifest",
        "config",
        "evidence",
        "raw",
        "binding",
        "manifest_sha256",
        "config_sha256",
        "evidence_sha256",
        "raw_sha256",
        "binding_sha256",
    ):
        if facts.get(key) != actual.get(key):
            raise ReadinessBlocked(
                f"GENERATION_GATE_BLOCKED: candidate lifecycle binding mismatch for {key}"
            )


def _ensure_live_candidate(
    *,
    target: Path,
    lifecycle_path: Path,
    lifecycle_value: v3_lifecycle.LifecycleState,
    bundle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    binding: v3_lifecycle.SessionBinding,
    approval: v3_readiness.SourceApproval,
    quote_observation: v3_readiness.QuoteObservation,
    host_key_fact: Mapping[str, Any],
    runner: Any,
    remote: RemoteIdentity,
    workspace: str,
    supervisor_fact: Mapping[str, Any],
) -> tuple[v3_lifecycle.LifecycleState, v3_readiness.LiveCandidate]:
    if lifecycle_value.phase == "GENERATION_READY":
        candidate = v3_readiness.load_live_candidate(
            run_dir=target,
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=lifecycle_value,
        )
        pipeline = dict(lifecycle_value.pipeline or {})
        if isinstance(pipeline.get("candidate"), Mapping):
            _assert_candidate_bound_to_lifecycle(lifecycle_value, candidate)
        else:
            pipeline["candidate"] = _candidate_facts(candidate)
            pipeline["verification"] = {
                "RUNTIME_READINESS": "VERIFIED",
                "SOURCE_STATUS": "DRAFT",
                "GPU_QUALITY_VERIFIED": False,
                "GENERATION_STATE": "NOT_RUN",
            }
            lifecycle_value = v3_lifecycle.update_phase(
                lifecycle_path,
                lifecycle_value,
                phase="GENERATION_READY",
                pipeline=pipeline,
            )
        return lifecycle_value, candidate
    if (target / "live" / "candidate-binding.json").is_file():
        candidate = v3_readiness.load_live_candidate(
            run_dir=target,
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=lifecycle_value,
        )
        pipeline = dict(lifecycle_value.pipeline or {})
        if isinstance(pipeline.get("candidate"), Mapping):
            _assert_candidate_bound_to_lifecycle(lifecycle_value, candidate)
        pipeline.update(
            {
                "candidate": _candidate_facts(candidate),
                "supervisor_fact": dict(supervisor_fact),
                "verification": {
                    "RUNTIME_READINESS": "VERIFIED",
                    "SOURCE_STATUS": "DRAFT",
                    "GPU_QUALITY_VERIFIED": False,
                    "GENERATION_STATE": "NOT_RUN",
                },
            }
        )
        lifecycle_value = v3_lifecycle.update_phase(
            lifecycle_path,
            lifecycle_value,
            phase="GENERATION_READY",
            pipeline=pipeline,
        )
        return lifecycle_value, candidate
    if lifecycle_value.phase != "READY":
        raise ReadinessBlocked(
            "GENERATION_BLOCKED: live candidate requires the READY bootstrap phase"
        )
    deadline_epoch = min(binding.planned_cancel_at, binding.hard_end_at) - 60.0
    if time.time() >= deadline_epoch:
        raise ReadinessBlocked("GENERATION_BLOCKED: deadline too close for runtime evidence")
    plan = v3_readiness.build_runtime_evidence_plan(
        bundle=bundle,
        source_hashes=source_hashes,
        approval=approval,
        quote_observation=quote_observation,
        host_key_fact=host_key_fact,
        binding=binding,
        lifecycle=lifecycle_value,
        workspace=workspace,
        owned_root=workspace,
        deadline_epoch=deadline_epoch,
    )
    try:
        collected = v3_bootstrap.collect_runtime_evidence(
            runner=runner,
            identity=remote,
            remote_root=workspace,
            plan=plan,
            timeout=v3_lifecycle.clip_remote_timeout(deadline_epoch, 900.0),
            deadline_monotonic=time.monotonic()
            + max(0.05, deadline_epoch - time.time()),
        )
        candidate = v3_readiness.build_live_candidate(
            run_dir=target,
            bundle=bundle,
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=lifecycle_value,
            approval=approval,
            quote_observation=quote_observation,
            host_key_fact=host_key_fact,
            evidence=collected["facts"],
            evidence_raw=collected["raw"],
            collector_sha256=collected["collector_sha256"],
            plan_sha256=collected["plan_sha256"],
        )
    except (v3_readiness.ReadinessError, v3_bootstrap.BootstrapError) as exc:
        raise ReadinessBlocked(f"GENERATION_BLOCKED: {exc}") from exc
    pipeline = dict(lifecycle_value.pipeline or {})
    pipeline.update(
        {
            "source_gate": dict(lifecycle_value.pipeline.get("source_gate", {}))
            if isinstance(lifecycle_value.pipeline, Mapping)
            else {},
            "supervisor_fact": dict(supervisor_fact),
            "candidate": _candidate_facts(candidate),
            "runtime_evidence": {
                "observed_at": candidate.binding.get("evidence", {}).get("observed_at"),
                "expires_at": candidate.binding.get("evidence", {}).get("expires_at"),
                "raw_sha256": collected["raw_sha256"],
                "collector_sha256": collected["collector_sha256"],
                "plan_sha256": collected["plan_sha256"],
                "source_bundle_sha256": v3_readiness.source_bundle_digest(source_hashes),
                "source_approval_sha256": approval.sha256,
                "session_binding_sha256": v3_readiness.session_binding_digest(binding),
                "host_key": dict(host_key_fact),
            },
            "verification": {
                "RUNTIME_READINESS": "VERIFIED",
                "SOURCE_STATUS": "DRAFT",
                "GPU_QUALITY_VERIFIED": False,
                "GENERATION_STATE": "NOT_RUN",
            },
        }
    )
    lifecycle_value = v3_lifecycle.update_phase(
        lifecycle_path,
        lifecycle_value,
        phase="GENERATION_READY",
        pipeline=pipeline,
    )
    return lifecycle_value, candidate


def prepare(
    *,
    workspace: str = "/workspace",
    run_dir: Path | None = None,
    execute_downloads: bool = False,
    session_state: Path | None = None,
    order_id: int | None = None,
    created_at: int | None = None,
    live: bool = False,
    source_approval: Path | None = None,
    quote_observation: Path | None = None,
    remote: RemoteIdentity | None = None,
    expected_fingerprint: str | None = None,
    resume_dead: bool = False,
    remote_runner: Any | None = None,
    tunnel_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Perform real offline preparation and write an LF PowerShell/bash runbook.

    No environment, SSH, provider or network call is made. The derived frame is
    verified, the setup/preflight plans are rendered, disk is checked when the
    workspace exists, and a durable state/lock is written.
    """

    if execute_downloads and not live:
        raise OperatorError(
            "EXECUTE_DOWNLOADS_REQUIRES_EXPLICIT_LIVE: no download was attempted"
        )
    bundle = load_bundle()
    plan = build_plan(workspace=workspace)
    bundle_hashes = _bundle_hashes(bundle)
    session_state_obj = None
    session_binding = None
    source_approval_obj = None
    quote_observation_obj = None
    host_key_fact = None
    source_gate = None
    if session_state is not None:
        if order_id is None or created_at is None:
            raise OperatorError("session state requires --order-id and --created-at")
        session_state_obj, session_binding = v3_lifecycle.load_session_binding(
            session_state,
            expected_order_id=order_id,
            expected_created_at=created_at,
        )
        v3_lifecycle.assert_deadline(session_binding)
        v3_lifecycle.assert_budget(session_binding, session_state_obj)
    if live:
        if session_binding is None or session_state_obj is None:
            raise OperatorError("live prepare requires explicit session state")
        if source_approval is None:
            raise OperatorError("live prepare requires --source-approval")
        if remote is None or expected_fingerprint is None:
            raise OperatorError(
                "live prepare requires --remote-host/--known-hosts and "
                "--expected-fingerprint"
            )
        observation_path = (
            Path(quote_observation)
            if quote_observation is not None
            else v3_readiness.quote_observation_path(session_state)
        )
        try:
            source_approval_obj = v3_readiness.load_source_approval(source_approval)
            quote_observation_obj = v3_readiness.load_quote_observation(observation_path)
        except v3_readiness.ReadinessError as exc:
            raise ReadinessBlocked(
                f"SOURCE_GATE_BLOCKED before lifecycle mutation: {exc}"
            ) from exc
        runner = remote_runner or RemoteRunner()
        try:
            v3_lifecycle.verify_host_key(
                runner=runner,
                identity=remote,
                expected_fingerprint=expected_fingerprint,
                known_hosts=remote.known_hosts,
                deadline_monotonic=time.monotonic()
                + max(0.05, min(session_binding.planned_cancel_at, session_binding.hard_end_at) - time.time() - 60.0),
            )
        except v3_lifecycle.LifecycleError as exc:
            raise OperatorError(str(exc)) from exc
        host_key_fact = v3_readiness.make_host_key_fact(remote, expected_fingerprint)
        source_gate = v3_readiness.evaluate_source_gate(
            bundle=bundle,
            source_hashes=bundle_hashes,
            approval=source_approval_obj,
            quote_observation=quote_observation_obj,
            session_state=session_state_obj,
            binding=session_binding,
            host_key_fact=host_key_fact,
            remote=remote,
            expected_fingerprint=expected_fingerprint,
        )
        if not source_gate.passed:
            raise ReadinessBlocked(
                "SOURCE_GATE_BLOCKED before lifecycle mutation: "
                + "; ".join(source_gate.issues)
            )
    target = Path(run_dir) if run_dir is not None else _default_run_dir()
    if session_binding is not None:
        owner_path = SESSION_OWNER_ROOT / f"{_session_key(session_binding)}.json"
        if owner_path.is_file():
            prior_owner = _read_json(owner_path)
            if prior_owner.get("run_dir") != str(target):
                raise OperatorError("session identity is owned by another run/nonce")
    _assert_clean_path(target)
    _prevalidate_run_root(target, bundle_hashes)
    target.mkdir(parents=True, exist_ok=True)
    lock_path = _validate_owned_path(target, target / "operator.lock", name="operator lock")
    lock = InstanceLock(lock_path).acquire()
    try:
        existing = (target / "run.json").is_file()
        prior = _ensure_run_document(target, bundle_hashes)
        suffix = f"-{uuid.uuid4().hex[:8]}" if existing else ""
        setup = _setup_plan(bundle, workspace)
        required = sum(spec.size_bytes for spec in setup.downloads)
        disk = _disk_status(workspace, required)
        setup_text = setup.render()
        if "\r\n" in setup_text:
            raise OperatorError("setup runbook unexpectedly contains CRLF")
        setup_path = _validate_owned_path(target, target / f"setup{suffix}.sh", name="setup")
        preflight_path = _validate_owned_path(target, target / f"preflight{suffix}.txt", name="preflight")
        runbook_path = _validate_owned_path(target, target / f"runbook{suffix}.ps1", name="runbook")
        plan_path = _validate_owned_path(target, target / f"plan{suffix}.json", name="plan")
        run_path = _validate_owned_path(target, target / "run.json", name="run.json")
        heartbeat_path = _validate_owned_path(target, target / "heartbeat.json", name="heartbeat")
        history_path = _validate_owned_path(target, target / "prepare-history.jsonl", name="history")
        lifecycle_path = _validate_owned_path(target, target / "lifecycle.json", name="lifecycle")
        quote_path = _validate_owned_path(target, target / "session-quote.json", name="session quote")
        supervisor_record_path = _validate_owned_path(target, target / "supervisor.json", name="supervisor record")
        supervisor_log_path = _validate_owned_path(target, target / "supervisor.log", name="supervisor log")
        download_job_path = _validate_owned_path(target, target / "download-job.json", name="download job")
        download_status_path = _validate_owned_path(target, target / "download-status.json", name="download status")
        download_progress_path = _validate_owned_path(target, target / "download-progress.json", name="download progress")
        lifecycle_phase = "UNBOUND_OFFLINE"
        lifecycle_value = None
        if session_binding is None and lifecycle_path.is_file():
            raise OperatorError(
                "existing run has a bound session; the same --session-state identity is required"
            )
        if session_binding is not None and session_state_obj is not None:
            _write_json(quote_path, session_state_obj.quote.to_dict())
            if lifecycle_path.is_file():
                lifecycle_value = v3_lifecycle.load_lifecycle(lifecycle_path)
                if (
                    lifecycle_value.run_id != target.name
                    or lifecycle_value.session != session_binding
                    or dict(lifecycle_value.bundle_hashes) != dict(bundle_hashes)
                ):
                    raise OperatorError("existing lifecycle immutable identity mismatch")
            elif existing:
                raise OperatorError("existing run has no lifecycle identity; explicit new run required")
            else:
                lifecycle_value = v3_lifecycle.new_lifecycle(
                    run_id=target.name,
                    session=session_binding,
                    bundle_hashes=bundle_hashes,
                    phase="BOUND_LIVE" if live else "BOUND_OFFLINE",
                )
                v3_lifecycle.write_lifecycle(lifecycle_path, lifecycle_value)
            _register_session_owner(
                session_binding, run_dir=target, nonce=lifecycle_value.nonce
            )
            if live and source_gate is not None:
                pipeline = dict(lifecycle_value.pipeline or {})
                pipeline.update(
                    {
                        "source_gate": dict(source_gate.facts),
                        "source_approval_path": str(source_approval_obj.path)
                        if source_approval_obj is not None
                        else None,
                        "quote_observation_path": str(quote_observation_obj.path)
                        if quote_observation_obj is not None
                        else None,
                        "host_key_fact": dict(host_key_fact) if host_key_fact else None,
                    }
                )
                lifecycle_value = v3_lifecycle.update_phase(
                    lifecycle_path,
                    lifecycle_value,
                    phase=lifecycle_value.phase,
                    pipeline=pipeline,
                )
            lifecycle_phase = lifecycle_value.phase
            supervisor_pending = False
            supervisor_fact = None
            if live:
                supervisor_deadline = time.monotonic() + max(
                    0.05,
                    min(session_binding.planned_cancel_at, session_binding.hard_end_at)
                    - time.time()
                    - 60.0,
                )
                supervisor = v3_lifecycle.CloreSupervisor(
                    session_state_path=session_state,
                    quote_path=quote_path,
                    binding=session_binding,
                    log_path=supervisor_log_path,
                    record_path=supervisor_record_path,
                    run_id=target.name,
                    nonce=lifecycle_value.nonce,
                )
                try:
                    if supervisor_record_path.is_file():
                        supervisor.readback(deadline_monotonic=supervisor_deadline)
                        lifecycle_phase = lifecycle_value.phase
                    else:
                        supervisor_record = supervisor.start(
                            deadline_monotonic=supervisor_deadline
                        )
                        lifecycle_value = v3_lifecycle.update_phase(
                            lifecycle_path,
                            lifecycle_value,
                            phase="SUPERVISOR_STARTED",
                            supervisor=supervisor_record.to_dict(),
                        )
                        lifecycle_phase = lifecycle_value.phase
                except v3_lifecycle.SupervisorPending:
                    supervisor_pending = True
                    lifecycle_value = v3_lifecycle.update_phase(
                        lifecycle_path, lifecycle_value, phase="SUPERVISOR_PENDING"
                    )
                    lifecycle_phase = lifecycle_value.phase
                if not supervisor_pending and supervisor_record_path.is_file():
                    supervisor_fact = _supervisor_fact_from_disk(supervisor)
                    assert remote is not None
                    supervisor_gate = v3_readiness.evaluate_source_gate(
                        bundle=bundle,
                        source_hashes=bundle_hashes,
                        approval=source_approval_obj,
                        quote_observation=quote_observation_obj,
                        session_state=session_state_obj,
                        binding=session_binding,
                        host_key_fact=host_key_fact,
                        remote=remote,
                        expected_fingerprint=expected_fingerprint,
                        supervisor_fact=supervisor_fact,
                        require_supervisor=True,
                        run_id=target.name,
                        nonce=lifecycle_value.nonce,
                    )
                    if not supervisor_gate.passed:
                        raise ReadinessBlocked(
                            "SUPERVISOR_GATE_BLOCKED before remote activity: "
                            + "; ".join(supervisor_gate.issues)
                        )
                    pipeline = dict(lifecycle_value.pipeline or {})
                    pipeline["source_gate"] = dict(supervisor_gate.facts)
                    pipeline["supervisor_fact"] = dict(supervisor_fact)
                    lifecycle_value = v3_lifecycle.update_phase(
                        lifecycle_path,
                        lifecycle_value,
                        phase=lifecycle_value.phase,
                        pipeline=pipeline,
                    )
                if execute_downloads and not supervisor_pending:
                    assert remote is not None
                    lifecycle_value = _advance_live_pipeline(
                        target=target,
                        lifecycle_path=lifecycle_path,
                        lifecycle_value=lifecycle_value,
                        session_binding=session_binding,
                        remote=remote,
                        expected_fingerprint=expected_fingerprint,
                        workspace=workspace,
                        bundle=bundle,
                        setup_text=setup_text,
                        preflight_text="\n".join(_preflight_lines(workspace)) + "\n",
                        runner=remote_runner,
                        execute_downloads=True,
                        resume_dead=resume_dead,
                        tunnel_factory=tunnel_factory,
                    )
                    lifecycle_phase = lifecycle_value.phase
                    pipeline = dict(lifecycle_value.pipeline or {})
                    pipeline["ssh_identity"] = {
                        "host": remote.host,
                        "user": remote.user,
                        "port": remote.port,
                        "known_hosts": str(remote.known_hosts),
                        "fingerprint": expected_fingerprint,
                    }
                    lifecycle_value = v3_lifecycle.update_phase(
                        lifecycle_path,
                        lifecycle_value,
                        phase=lifecycle_value.phase,
                        pipeline=pipeline,
                    )
                    download_status_path = _validate_owned_path(
                        target, target / "download-status.json", name="download status"
                    )
                    download_status = (
                        _read_json(download_status_path)
                        if download_status_path.is_file()
                        else {"status": lifecycle_value.phase}
                    )
                    lifecycle_value = v3_lifecycle.update_phase(
                        lifecycle_path,
                        lifecycle_value,
                        phase=lifecycle_value.phase,
                        downloader={
                            "status": download_status.get("status"),
                            "job": str(download_job_path),
                            "progress": str(download_progress_path),
                        },
                    )
                    lifecycle_phase = lifecycle_value.phase
                    if lifecycle_phase in {"READY", "GENERATION_READY"}:
                        if (
                            source_approval_obj is None
                            or quote_observation_obj is None
                            or host_key_fact is None
                            or supervisor_fact is None
                        ):
                            raise ReadinessBlocked(
                                "GENERATION_BLOCKED: candidate prerequisites are incomplete"
                            )
                        lifecycle_value, _candidate = _ensure_live_candidate(
                            target=target,
                            lifecycle_path=lifecycle_path,
                            lifecycle_value=lifecycle_value,
                            bundle=bundle,
                            source_hashes=bundle_hashes,
                            binding=session_binding,
                            approval=source_approval_obj,
                            quote_observation=quote_observation_obj,
                            host_key_fact=host_key_fact,
                            runner=remote_runner or RemoteRunner(),
                            remote=remote,
                            workspace=workspace,
                            supervisor_fact=supervisor_fact,
                        )
                        lifecycle_phase = lifecycle_value.phase
        _atomic_write(setup_path, setup_text)
        _atomic_write(preflight_path, "\n".join(_preflight_lines(workspace)) + "\n")
        runbook = (
            "# P1 v3 two-phase PowerShell runbook.\n"
            "# Run from the repository root. Remote commands are executed only by\n"
            "# the bounded wrapper; do not open a second PS/root window.\n"
            "$ErrorActionPreference = 'Stop'\n"
            "$Repo = (Get-Location).Path\n"
            "$Python = Join-Path $Repo '.venv\\Scripts\\python.exe'\n"
            "$Op = Join-Path $Repo 'scripts\\p1\\v3_operator.py'\n"
            f"$RunDir = '{target}'\n"
            "$SessionState = '<SESSION_STATE_PATH>'\n"
            "$Approval = '<SOURCE_APPROVAL_PATH>'\n"
            "$Quote = '<MACHINE_QUOTE_PATH>'\n"
            "$OrderId = '<ORDER_ID>'\n"
            "$CreatedAt = '<CREATED_AT>'\n"
            "$RemoteHost = '<REMOTE_HOST>'\n"
            "$RemoteIdentity = '<LOCAL_PRIVATE_KEY_PATH>'\n"
            "$KnownHosts = '<LOCAL_KNOWN_HOSTS_PATH>'\n"
            "$Fingerprint = 'SHA256:<TRUSTED_FINGERPRINT>'\n"
            "$Workspace = '/workspace'\n"
            "# Before Pay: source-only plan/readiness; no secrets or session values.\n"
            "& $Python $Op plan --json\n"
            "& $Python $Op readiness --json\n"
            "# Owner/Codex may create the explicit scoped source approval once.\n"
            "# & $Python $Op approve-source --approval-doc docs\\P1-session-02-approval-20261026.md --approval-out $Approval --json\n"
            "# After Pay: bind the actual order and machine quote; no hand-edited JSON.\n"
            "# & $Python $Op session-bind --session-state $SessionState --quote $Quote --order-id $OrderId --created-at $CreatedAt --json\n"
            "& $Python $Op prepare --live --execute-downloads --run-dir $RunDir --workspace $Workspace --session-state $SessionState --order-id $OrderId --created-at $CreatedAt --source-approval $Approval --remote-host $RemoteHost --remote-identity $RemoteIdentity --remote-known-hosts $KnownHosts --expected-fingerprint $Fingerprint --json\n"
            "& $Python $Op status --run-dir $RunDir --session-state $SessionState --remote-host $RemoteHost --remote-identity $RemoteIdentity --remote-known-hosts $KnownHosts --expected-fingerprint $Fingerprint --remote-workspace $Workspace --json\n"
            "# run-once uses the per-run live candidate only; source DRAFT remains DRAFT.\n"
            "& $Python $Op run-once --live --run-dir $RunDir --session-state $SessionState --order-id $OrderId --created-at $CreatedAt --remote-host $RemoteHost --remote-identity $RemoteIdentity --remote-known-hosts $KnownHosts --expected-fingerprint $Fingerprint --json\n"
            "# Collect defaults to the exact backend-fetched artifact artifacts/<job_id>/result.mp4.\n"
            "& $Python $Op collect --run-dir $RunDir --json\n"
            "# Remote collect is only allowed from the pipeline-produced remote-export.json;\n"
            "# never provide an invented remote MP4/helper path.\n"
            "# Attach/readback: rerun status/run-once with the same RunDir/SessionState;\n"
            "# never start a second backend or submit when job identity is unknown.\n"
        )
        _atomic_write(runbook_path, runbook)
        _write_json(plan_path, plan)
        now = datetime.now(timezone.utc).isoformat()
        downloads_state = (
            "DOWNLOAD_COMPLETE"
            if _pipeline_rank(lifecycle_phase) >= 7
            else "DOWNLOAD_FAILED"
            if lifecycle_phase == "DOWNLOAD_FAILED"
            else "NOT_RUN_OFFLINE"
        )
        if lifecycle_phase == "SUPERVISOR_PENDING":
            run_status = "SUPERVISOR_PENDING"
        else:
            run_status = (
                "PREPARED_READY"
                if lifecycle_phase in {"READY", "GENERATION_READY"}
                else "PREPARED_OFFLINE"
            )
        pipeline_facts = (
            dict(lifecycle_value.pipeline or {})
            if isinstance(lifecycle_value, v3_lifecycle.LifecycleState)
            else {}
        )
        candidate_facts = pipeline_facts.get("candidate")
        runtime_facts = pipeline_facts.get("runtime_evidence")
        state = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "run_id": target.name,
            "status": run_status,
            "created_at": (prior or {}).get("created_at", now),
            "heartbeat_at": now,
            "workspace": workspace,
            "disk": disk,
            "downloads": downloads_state,
            "backend": (
                "READY_FOR_BACKEND"
                if lifecycle_phase in {"READY", "GENERATION_READY"}
                else "NOT_RUN_OFFLINE"
            ),
            "ssh": "NOT_RUN_OFFLINE",
            "decoder": _decoder_status(),
            "bundle_hashes": bundle_hashes,
            "lifecycle_path": str(lifecycle_path),
            "lifecycle_phase": lifecycle_phase,
            "generation_ready": lifecycle_phase == "GENERATION_READY",
            "candidate": candidate_facts if isinstance(candidate_facts, Mapping) else None,
            "runtime_evidence": runtime_facts if isinstance(runtime_facts, Mapping) else None,
            "verification": pipeline_facts.get("verification"),
            "financial_status": _financial_status(session_binding),
            "owner_risk_admission_sha256": (
                session_binding.owner_risk_admission_sha256
                if session_binding is not None
                else None
            ),
            "provider_billing_verified": False,
            "next_action": (
                "review generation_ready candidate/evidence; run-once revalidates "
                "full facts and uses the existing backend serve gate"
                if lifecycle_phase == "GENERATION_READY"
                else "review the rendered runbook; live requires approved source/session/SSH"
            ),
        }
        _write_json(run_path, state)
        write_worker_state(heartbeat_path, {
            "run_id": target.name,
            "pid": os.getpid(),
            "updated_at": now,
            "phase": run_status,
        })
        _append_jsonl(history_path, {
            "at": now,
            "status": run_status,
            "setup": setup_path.name,
            "preflight": preflight_path.name,
            "runbook": runbook_path.name,
            "plan": plan_path.name,
            "resumed": existing,
        })
        return {
            "status": run_status,
            "run_dir": str(target),
            "runbook": str(runbook_path),
            "setup_script": str(setup_path),
            "preflight": str(preflight_path),
            "plan": str(plan_path),
            "resumed": existing,
            "disk": disk,
            "downloads": downloads_state,
            "ssh": "NOT_RUN_OFFLINE",
            "lifecycle": str(lifecycle_path),
            "lifecycle_phase": lifecycle_phase,
            "generation_ready": lifecycle_phase == "GENERATION_READY",
            "candidate": candidate_facts if isinstance(candidate_facts, Mapping) else None,
            "runtime_evidence": runtime_facts if isinstance(runtime_facts, Mapping) else None,
            "verification": pipeline_facts.get("verification"),
            "financial_status": _financial_status(session_binding),
            "owner_risk_admission_sha256": (
                session_binding.owner_risk_admission_sha256
                if session_binding is not None
                else None
            ),
            "provider_billing_verified": False,
            "next_action": state["next_action"],
        }
    finally:
        lock.release()


def status(
    *,
    run_dir: Path | str | None = None,
    session_state: Path | str | None = None,
    remote: RemoteIdentity | None = None,
    expected_fingerprint: str | None = None,
    remote_workspace: str = "/workspace",
    remote_runner: Any | None = None,
) -> dict[str, Any]:
    target = Path(run_dir) if run_dir is not None else _latest_run_dir()
    if target is None or not target.exists():
        return {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": "NO_RUN_STATE",
            "run_dir": str(target) if target is not None else None,
            "live": "NOT_RUN",
            "next_action": "run prepare or provide --run-dir",
        }
    state = _read_json(target / "run.json") if (target / "run.json").is_file() else {}
    heartbeat = (
        read_worker_state(target / "heartbeat.json")
        if (target / "heartbeat.json").is_file()
        else None
    )
    jobs = _read_jobs_state(target / "jobs.sqlite3")
    export_manifest = _read_export_manifest(target / "export")
    session = _read_session_state(Path(session_state)) if session_state else None
    lifecycle = None
    lifecycle_path = target / "lifecycle.json"
    if lifecycle_path.is_file():
        try:
            lifecycle = v3_lifecycle.load_lifecycle(lifecycle_path).to_dict()
        except v3_lifecycle.LifecycleError as exc:
            lifecycle = {"status": "NOT_VERIFIED", "reason": str(exc)}
    remote_status = None
    if remote is not None:
        if expected_fingerprint is None:
            raise OperatorError("remote status requires --expected-fingerprint")
        runner = remote_runner or RemoteRunner()
        try:
            v3_lifecycle.verify_host_key(
                runner=runner,
                identity=remote,
                expected_fingerprint=expected_fingerprint,
                known_hosts=remote.known_hosts,
                deadline_monotonic=time.monotonic() + 10.0,
            )
        except v3_lifecycle.LifecycleError as exc:
            raise OperatorError(str(exc)) from exc
        if not lifecycle_path.is_file():
            raise OperatorError("remote status requires a bound run lifecycle")
        lifecycle_value = v3_lifecycle.load_lifecycle(lifecycle_path)
        root = str(PurePosixPath(remote_workspace))
        helper = str(PurePosixPath(root) / "scripts" / "p1" / "v3_remote_fetch_helper.py")
        deadline_epoch = min(
            float(lifecycle_value.session.planned_cancel_at),
            float(lifecycle_value.session.hard_end_at),
        ) - 60.0
        read_timeout = v3_lifecycle.clip_remote_timeout(deadline_epoch, 15.0)

        def read(name: str) -> dict[str, Any]:
            return v3_lifecycle.read_remote_json(
                runner=runner,
                identity=remote,
                helper_path=helper,
                remote_path=str(PurePosixPath(root) / name),
                remote_root=root,
                timeout=read_timeout,
            )

        owner = read("worker-owner.json")
        status_value = read("status.json")
        progress = read("progress.json")
        heartbeat_value = read("heartbeat.json")
        for name, value in (
            ("owner", owner),
            ("status", status_value),
            ("progress", progress),
            ("heartbeat", heartbeat_value),
        ):
            if (
                value.get("run_id") != lifecycle_value.run_id
                or value.get("nonce") != lifecycle_value.nonce
            ):
                raise OperatorError(f"remote status {name} identity mismatch")
        for key in ("files_done", "files_total", "bytes_done", "bytes_total"):
            number = progress.get(key)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise OperatorError(f"remote status progress {key} is invalid")
        heartbeat_at = heartbeat_value.get("heartbeat_at") or heartbeat_value.get("at")
        if not isinstance(heartbeat_at, (int, float)):
            raise OperatorError("remote status heartbeat has no timestamp")
        heartbeat_age = time.time() - float(heartbeat_at)
        fresh = -30.0 <= heartbeat_age <= 120.0
        remote_status = {
            "owner": owner,
            "status": status_value,
            "progress": progress,
            "heartbeat": heartbeat_value,
            "heartbeat_age_seconds": heartbeat_age,
            "fresh": fresh,
        }
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": state.get("status", "UNKNOWN"),
        "run_dir": str(target),
        "heartbeat": heartbeat,
        "jobs": jobs,
        "export": export_manifest,
        "session": session,
        "financial_status": (
            session.get("financial_status", "NOT_BOUND")
            if isinstance(session, Mapping)
            else "NOT_BOUND"
        ),
        "provider_billing_verified": False,
        "lifecycle": lifecycle,
        "generation": {
            "phase": lifecycle.get("phase") if isinstance(lifecycle, Mapping) else None,
            "candidate": (
                lifecycle.get("pipeline", {}).get("candidate")
                if isinstance(lifecycle, Mapping) and isinstance(lifecycle.get("pipeline"), Mapping)
                else None
            ),
            "runtime_evidence": (
                lifecycle.get("pipeline", {}).get("runtime_evidence")
                if isinstance(lifecycle, Mapping) and isinstance(lifecycle.get("pipeline"), Mapping)
                else None
            ),
            "verification": (
                lifecycle.get("pipeline", {}).get("verification")
                if isinstance(lifecycle, Mapping) and isinstance(lifecycle.get("pipeline"), Mapping)
                else None
            ),
        },
        "remote_status": remote_status,
        "decoder": _decoder_status(),
        "live": "NOT_RUN",
        "not_verified": [
            "SSH/provider/GPU",
            "real video decode" if _decoder_status()["status"] != "AVAILABLE" else None,
        ],
        "next_action": state.get("next_action", "inspect state and runbook"),
    }


def _latest_run_dir() -> Path | None:
    if not RUN_ROOT.is_dir():
        return None
    candidates = [path for path in RUN_ROOT.iterdir() if path.is_dir() and path.name.startswith("run-")]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_jobs_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "NOT_RUN", "path": str(path)}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT job_id,status,prompt_id,progress,created_at,updated_at FROM jobs ORDER BY created_ms DESC LIMIT 5"
            ).fetchall()
            events = conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"status": "NOT_VERIFIED", "path": str(path), "reason": str(exc)}
    return {
        "status": "OK",
        "path": str(path),
        "jobs": [
            {
                "job_id": row[0],
                "status": row[1],
                "prompt_id": row[2],
                "progress": row[3],
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ],
        "event_count": events,
    }


def _read_export_manifest(export_dir: Path) -> dict[str, Any]:
    if not export_dir.is_dir():
        return {"status": "NOT_RUN"}
    manifests = sorted(
        export_dir.glob(".clore-export-*/manifest.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not manifests:
        return {"status": "NOT_RUN", "path": str(export_dir)}
    try:
        payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "NOT_VERIFIED", "path": str(manifests[-1]), "reason": str(exc)}
    return {"status": payload.get("status", "UNKNOWN"), "path": str(manifests[-1])}


def _read_session_state(path: Path) -> dict[str, Any]:
    try:
        state = clore_session.load_state(path, safety_mode=True)
    except Exception as exc:
        return {"status": "NOT_VERIFIED", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "status": state.status,
        "order_id": state.order_id,
        "server_id": state.server_id,
        "created_at": state.created_at,
        "hard_end_at": state.hard_end_at,
        "planned_cancel_at": state.planned_cancel_at,
        "billing_verified": state.billing_verified,
        "financial_status": (
            "OWNER_ACCEPTED_FINANCIAL_RISK"
            if state.owner_risk is not None
            else "STRICT_QUOTE"
        ),
        "owner_risk_admission_id": (
            dict(state.owner_risk).get("admission_id")
            if isinstance(state.owner_risk, Mapping)
            else None
        ),
        "owner_risk_admission_sha256": state.owner_risk_sha256,
        "provider_billing_verified": False,
    }


def _bounded_history_get(
    transport: Any,
    prompt_id: str,
    *,
    deadline_monotonic: float | None,
) -> Any:
    """Run the existing async history parser under one absolute deadline.

    No new transport and no background thread: ``asyncio.wait_for`` cancels the
    existing transport coroutine and awaits cancellation/client cleanup.  Only
    a completed history entry is accepted; absence is not verified and there is
    no queue fallback.
    """

    history_async = getattr(transport, "_history_async", None)
    status_from_entry = getattr(transport, "_status_from_entry", None)
    if not callable(history_async) or not callable(status_from_entry):
        raise OperatorError("ComfyUI transport lacks the required history API")
    if deadline_monotonic is None:
        timeout = None
    else:
        timeout = deadline_monotonic - time.monotonic()
        if timeout <= 0:
            raise OperatorError("deadline expired before ComfyUI history read")

    async def _read():
        try:
            entry = await asyncio.wait_for(history_async(prompt_id), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise OperatorError("ComfyUI history deadline expired") from exc
        if entry is None:
            return None
        return status_from_entry(prompt_id, entry)

    return asyncio.run(_read())


def _produce_remote_export(
    *,
    target: Path,
    remote: RemoteIdentity,
    workspace: str,
    config_path: Path,
    lifecycle: v3_lifecycle.LifecycleState,
    transport: Any | None = None,
    runner: Any | None = None,
    helper_path: str | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Persist an exact remote-export mapping from validated ComfyUI history.

    The prompt id/job id come from the durable backend result; the SaveVideo
    artifact reference comes from the existing ComfyHTTP history parser.  The
    remote helper then reads size/sha256 for that exact owned output path.
    Any missing/ambiguous history identity fails closed.
    """

    job_path = _validate_owned_path(target, target / "job-id.json", name="job id")
    result_path = _validate_owned_path(
        target, target / "backend-result.json", name="backend result"
    )
    if not job_path.is_file() or not result_path.is_file():
        raise OperatorError("remote export requires a successful backend result")
    job_id = _read_json(job_path).get("job_id")
    result_record = _read_json(result_path)
    if not isinstance(job_id, str) or result_record.get("job_id") != job_id:
        raise OperatorError("remote export job identity mismatch")
    payload = result_record.get("result")
    if isinstance(payload, Mapping) and isinstance(payload.get("result"), Mapping):
        payload = payload["result"]
    prompt_id = payload.get("prompt_id") if isinstance(payload, Mapping) else None
    if not isinstance(prompt_id, str) or not prompt_id:
        raise OperatorError("remote export requires the exact backend prompt id")
    if transport is None:
        from anime_studio.comfy_http import build_transport_from_config

        config = load_config(config_path)
        token = os.environ.get(config.token_env) if config.token_env else None
        transport = build_transport_from_config(config_path, token=token)
    status = _bounded_history_get(transport, prompt_id, deadline_monotonic=deadline_monotonic)
    if getattr(status, "state", None) != "succeeded":
        raise OperatorError("ComfyUI history prompt is not in a successful state")
    artifact = getattr(status, "artifact", None)
    if not isinstance(artifact, Mapping):
        raise OperatorError("ComfyUI history has no exact SaveVideo artifact")
    if (
        artifact.get("scheme") != "comfy-http"
        or artifact.get("type") != "output"
        or artifact.get("media_type") != "video/mp4"
    ):
        raise OperatorError("ComfyUI artifact reference identity is invalid")
    filename = artifact.get("filename")
    subfolder = artifact.get("subfolder", "")
    if (
        not isinstance(filename, str)
        or not filename
        or PurePosixPath(filename).name != filename
        or not filename.lower().endswith(".mp4")
        or not isinstance(subfolder, str)
        or any(part == ".." for part in PurePosixPath(subfolder).parts)
    ):
        raise OperatorError("ComfyUI artifact name/subfolder is unsafe")
    root = str(PurePosixPath(workspace))
    relative = PurePosixPath("ComfyUI", "output", subfolder, filename)
    if str(relative).startswith("/") or ".." in relative.parts:
        raise OperatorError("remote export relative path is unsafe")
    remote_artifact_path = str(PurePosixPath(root) / relative)
    helper = helper_path or str(
        PurePosixPath(root) / "scripts" / "p1" / "v3_remote_fetch_helper.py"
    )
    reader = runner or RemoteRunner()
    result = reader.run_remote(
        remote,
        ("python3", helper, "metadata", remote_artifact_path, root),
        timeout=30.0,
        deadline_monotonic=deadline_monotonic,
    )
    if getattr(result, "timed_out", False) or result.returncode != 0:
        raise OperatorError("remote artifact metadata read failed")
    try:
        metadata = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError("remote artifact metadata is invalid JSON") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != "anime-video-studio/p1-v3-metadata/v1"
        or metadata.get("path") != remote_artifact_path
        or not isinstance(metadata.get("size"), int)
        or metadata["size"] <= 0
        or not isinstance(metadata.get("sha256"), str)
        or len(metadata["sha256"]) != 64
    ):
        raise OperatorError("remote artifact metadata identity mismatch")
    ssh = dict(lifecycle.pipeline or {}).get("ssh_identity")
    export = {
        "schema": "anime-video-studio/p1-v3-remote-export/v1",
        "run_id": target.name,
        "nonce": lifecycle.nonce,
        "session": lifecycle.session.to_dict(),
        "ssh_identity": ssh,
        "job_id": job_id,
        "prompt_id": prompt_id,
        "remote_root": root,
        "remote_artifact_path": remote_artifact_path,
        "remote_helper_path": helper,
        "relative_path": str(relative),
        "filename": filename,
        "size": metadata["size"],
        "sha256": metadata["sha256"],
        "artifact": dict(artifact),
        "source": "backend-result+comfy-history",
    }
    _write_json(target / "remote-export.json", export)
    return export


def _validate_ready_runtime(
    *,
    target: Path,
    lifecycle: v3_lifecycle.LifecycleState,
    remote: RemoteIdentity,
    expected_fingerprint: str,
    runner: Any | None = None,
    deadline_monotonic: float | None = None,
) -> None:
    """Revalidate the READY Comfy/tunnel/SSH identity before any POST."""

    pipeline = dict(lifecycle.pipeline or {})
    ssh = pipeline.get("ssh_identity")
    expected_ssh = {
        "host": remote.host,
        "user": remote.user,
        "port": remote.port,
        "known_hosts": str(remote.known_hosts),
        "fingerprint": expected_fingerprint,
    }
    if not isinstance(ssh, Mapping) or dict(ssh) != expected_ssh:
        raise OperatorError("READY lifecycle SSH identity does not match this run")
    tunnel_info = pipeline.get("tunnel")
    if not isinstance(tunnel_info, Mapping):
        raise OperatorError("READY lifecycle has no owned tunnel record")
    if deadline_monotonic is not None:
        _monotonic_timeout(deadline_monotonic, 1.0)
    owner_path = _validate_owned_path(
        target, target / "tunnel-owner.json", name="tunnel owner"
    )
    tunnel = v3_bootstrap.OwnedTunnel.attach(
        identity=remote,
        owner_path=owner_path,
        run_id=lifecycle.run_id,
        nonce=lifecycle.nonce,
        expected_fingerprint=expected_fingerprint,
        deadline_monotonic=deadline_monotonic,
    )
    tunnel.verify_mapping(deadline_monotonic=deadline_monotonic)
    if deadline_monotonic is not None:
        _monotonic_timeout(deadline_monotonic, 1.0)
    if pipeline.get("comfy") is None:
        raise OperatorError("READY lifecycle has no owned ComfyUI record")
    root = pipeline.get("root")
    if not isinstance(root, str):
        raise OperatorError("READY lifecycle has no owned root")
    helper = str(PurePosixPath(root) / "scripts" / "p1" / "v3_remote_fetch_helper.py")
    reader = runner or RemoteRunner()
    owner = v3_lifecycle.read_remote_json(
        runner=reader,
        identity=remote,
        helper_path=helper,
        remote_path=str(PurePosixPath(root) / "comfy-owner.json"),
        remote_root=root,
        deadline_monotonic=deadline_monotonic,
    )
    ack = v3_lifecycle.read_remote_json(
        runner=reader,
        identity=remote,
        helper_path=helper,
        remote_path=str(PurePosixPath(root) / "comfy-ack.json"),
        remote_root=root,
        deadline_monotonic=deadline_monotonic,
    )
    if (
        owner.get("schema") != "anime-video-studio/p1-v3-process-owner/v1"
        or owner.get("kind") != "comfy"
        or owner.get("run_id") != lifecycle.run_id
        or owner.get("nonce") != lifecycle.nonce
        or ack.get("run_id") != lifecycle.run_id
        or ack.get("nonce") != lifecycle.nonce
        or ack.get("kind") != "comfy"
        or not isinstance(owner.get("command_sha256"), str)
        or owner.get("command_sha256") != ack.get("command_sha256")
    ):
        raise OperatorError("READY ComfyUI owner/ack identity mismatch")


def run_once(
    *,
    live: bool = False,
    config_path: Path = V3_CONFIG,
    session_state: Path | None = None,
    order_id: int | None = None,
    created_at: int | None = None,
    remote: RemoteIdentity | None = None,
    expected_fingerprint: str | None = None,
    run_dir: Path | None = None,
    backend_runner: Callable[..., Any] | None = None,
    remote_runner: Any | None = None,
    ready_validator: Callable[..., Any] | None = None,
    workspace: str = "/workspace",
    comfy_transport: Any | None = None,
) -> dict[str, Any]:
    """Plan or (only with explicit live gates) run one job through existing API."""

    bundle = load_bundle()
    readiness = bundle["readiness"]
    plan = build_plan()
    if not live:
        return {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": "LIVE_NOT_RUN",
            "readiness": {
                "live_ready": readiness.live_ready,
                "issues": list(readiness.issues),
            },
            "would_run": plan["v3"],
            "next_action": "explicit live approval and a satisfied readiness gate are required",
        }
    target = Path(run_dir) if run_dir is not None else _default_run_dir()
    candidate_config_path = target / "live" / "candidate-config.json"
    candidate_binding_path = target / "live" / "candidate-binding.json"
    if not candidate_config_path.is_file() or not candidate_binding_path.is_file():
        raise ReadinessBlocked(
            "READINESS_GATE_BLOCKED before mutation: source manifest is DRAFT and "
            "no validated per-run live candidate exists"
        )
    if session_state is None or order_id is None or created_at is None:
        raise OperatorError("live run requires --session-state, --order-id and --created-at")
    state = _load_session_state(session_state)
    if state.order_id != order_id or state.created_at != created_at:
        raise OperatorError("session identity does not match exact order/created_at")
    now = int(datetime.now(timezone.utc).timestamp())
    if now >= state.hard_end_at:
        raise OperatorError("session hard end has passed; no mutation")
    if remote is None or expected_fingerprint is None:
        raise OperatorError(
            "live run requires explicit SSH identity and expected fingerprint"
        )
    b_hashes = _bundle_hashes(bundle)
    try:
        candidate_binding = _read_json(candidate_binding_path)
        approval = v3_readiness.load_source_approval(
            candidate_binding["source_approval"]["path"]
        )
        quote_observation = v3_readiness.load_quote_observation(
            candidate_binding["quote_observation"]["path"]
        )
    except (KeyError, TypeError, v3_readiness.ReadinessError) as exc:
        raise ReadinessBlocked(
            f"GENERATION_GATE_BLOCKED before mutation: candidate provenance is invalid: {exc}"
        ) from exc
    runner = remote_runner or RemoteRunner()
    try:
        v3_lifecycle.verify_host_key(
            runner=runner,
            identity=remote,
            expected_fingerprint=expected_fingerprint,
            known_hosts=remote.known_hosts,
            deadline_monotonic=time.monotonic()
            + max(0.05, min(state.planned_cancel_at, state.hard_end_at) - time.time() - 60.0),
        )
    except RemoteUnavailable as exc:
        raise OperatorError(f"SSH tools unavailable: {exc}") from exc
    except v3_lifecycle.LifecycleError as exc:
        raise OperatorError(f"host key verification failed: {exc}") from exc
    except RemoteError as exc:
        raise OperatorError(f"host key check failed: {exc}") from exc
    host_key_fact = v3_readiness.make_host_key_fact(remote, expected_fingerprint)

    _assert_no_link_components(target, stop=target.parent)
    if target.exists():
        _prevalidate_run_root(target, b_hashes)
    else:
        target.mkdir(parents=True, exist_ok=False)
    lock = InstanceLock(target / "operator.lock").acquire()
    try:
        run_path = _validate_owned_path(target, target / "run.json", name="run.json")
        setup_path = _validate_owned_path(target, target / "setup.sh", name="setup")
        preflight_path = _validate_owned_path(target, target / "preflight.txt", name="preflight")
        binding = v3_lifecycle.SessionBinding.from_state(state)
        v3_lifecycle.assert_deadline(binding)
        v3_lifecycle.assert_budget(binding, state)
        quote_path = _validate_owned_path(target, target / "session-quote.json", name="session quote")
        lifecycle_path = _validate_owned_path(target, target / "lifecycle.json", name="lifecycle")
        supervisor_record_path = _validate_owned_path(target, target / "supervisor.json", name="supervisor record")
        supervisor_log_path = _validate_owned_path(target, target / "supervisor.log", name="supervisor log")
        if lifecycle_path.is_file():
            lifecycle_value = v3_lifecycle.load_lifecycle(lifecycle_path)
            if (
                lifecycle_value.run_id != target.name
                or lifecycle_value.session != binding
                or dict(lifecycle_value.bundle_hashes) != dict(b_hashes)
            ):
                raise OperatorError("existing lifecycle immutable identity mismatch")
            if lifecycle_value.phase not in {"READY", "GENERATION_READY"}:
                raise OperatorError(
                    "live run-once requires READY/GENERATION_READY preparation; current phase is "
                    + lifecycle_value.phase
                )
            try:
                candidate = v3_readiness.load_live_candidate(
                    run_dir=target,
                    source_hashes=b_hashes,
                    binding=binding,
                    lifecycle=lifecycle_value,
                )
            except v3_readiness.ReadinessError as exc:
                raise ReadinessBlocked(
                    f"GENERATION_GATE_BLOCKED before mutation: {exc}"
                ) from exc
            _assert_candidate_bound_to_lifecycle(lifecycle_value, candidate)
            validator = ready_validator or _validate_ready_runtime
            validator(
                target=target,
                lifecycle=lifecycle_value,
                remote=remote,
                expected_fingerprint=expected_fingerprint,
                runner=runner,
                deadline_monotonic=time.monotonic()
                + max(0.05, min(binding.planned_cancel_at, binding.hard_end_at) - time.time() - 60.0),
            )
        else:
            raise OperatorError(
                "live run-once requires a prepared run (prepare --live --execute-downloads)"
            )
        _register_session_owner(binding, run_dir=target, nonce=lifecycle_value.nonce)
        _write_json(quote_path, state.quote.to_dict())
        _write_json(
            run_path,
            {
                "schema": SCHEMA,
                "task_id": TASK_ID,
                "run_id": target.name,
                "status": "LIVE_STARTING",
                "order_id": order_id,
                "created_at": created_at,
                "hard_end_at": state.hard_end_at,
                "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "bundle_hashes": b_hashes,
                "lifecycle_phase": lifecycle_value.phase,
                "candidate": _candidate_facts(candidate),
                "financial_status": _financial_status(binding),
                "owner_risk_admission_sha256": binding.owner_risk_admission_sha256,
                "provider_billing_verified": False,
            },
        )
        if not setup_path.is_file():
            _atomic_write(setup_path, _setup_plan(bundle, "/workspace").render())
        if not preflight_path.is_file():
            _atomic_write(preflight_path, "\n".join(_preflight_lines("/workspace")) + "\n")
        supervisor = v3_lifecycle.CloreSupervisor(
            session_state_path=session_state,
            quote_path=quote_path,
            binding=binding,
            log_path=supervisor_log_path,
            record_path=supervisor_record_path,
            run_id=target.name,
            nonce=lifecycle_value.nonce,
        )
        supervisor_deadline = time.monotonic() + max(
            0.05, min(binding.planned_cancel_at, binding.hard_end_at) - time.time() - 60.0
        )
        try:
            if supervisor_record_path.is_file():
                supervisor.readback(deadline_monotonic=supervisor_deadline)
            else:
                supervisor_record = supervisor.start(
                    deadline_monotonic=supervisor_deadline
                )
                next_phase = (
                    "SUPERVISOR_STARTED"
                    if lifecycle_value.phase in {"BOUND_LIVE", "BOUND_OFFLINE"}
                    else lifecycle_value.phase
                )
                lifecycle_value = v3_lifecycle.update_phase(
                    lifecycle_path,
                    lifecycle_value,
                    phase=next_phase,
                    supervisor=supervisor_record.to_dict(),
                )
        except v3_lifecycle.SupervisorPending:
            lifecycle_value = v3_lifecycle.update_phase(
                lifecycle_path, lifecycle_value, phase="SUPERVISOR_PENDING"
            )
            _write_json(
                run_path,
                {
                    "schema": SCHEMA,
                    "task_id": TASK_ID,
                    "run_id": target.name,
                    "status": "SUPERVISOR_PENDING",
                    "order_id": order_id,
                    "created_at": created_at,
                    "hard_end_at": state.hard_end_at,
                    "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return {
                "status": "SUPERVISOR_PENDING",
                "run_dir": str(target),
                "no_activity": True,
            }
        supervisor_fact = _supervisor_fact_from_disk(supervisor)
        generation_gate = v3_readiness.evaluate_generation_gate(
            run_dir=target,
            bundle=bundle,
            source_hashes=b_hashes,
            approval=approval,
            quote_observation=quote_observation,
            session_state=state,
            binding=binding,
            lifecycle=lifecycle_value,
            host_key_fact=host_key_fact,
            remote=remote,
            expected_fingerprint=expected_fingerprint,
            supervisor_fact=supervisor_fact,
        )
        if not generation_gate.passed:
            raise ReadinessBlocked(
                "GENERATION_GATE_BLOCKED before any POST: "
                + "; ".join(generation_gate.issues)
            )
        config_path = candidate.config_path
        candidate_facts = _candidate_facts(candidate)

        def _pre_submit_guard() -> None:
            current_state = _load_session_state(session_state)
            current_binding = v3_lifecycle.SessionBinding.from_state(current_state)
            if current_binding != binding:
                raise OperatorError("session identity changed before submit")
            if current_state.status != "ARMED":
                raise ReadinessBlocked(
                    f"submission blocked: session is {current_state.status}"
                )
            if time.time() >= current_state.hard_end_at:
                raise ReadinessBlocked(
                    "submission blocked: session hard end has passed"
                )
            v3_lifecycle.assert_deadline(current_binding)
            v3_lifecycle.assert_budget(current_binding, current_state)

        result = _run_backend_once(
            bundle=bundle,
            target=target,
            config_path=config_path,
            binding=binding,
            lifecycle=lifecycle_value,
            backend_runner=backend_runner,
            pre_submit_guard=_pre_submit_guard,
        )
        if result.get("status") == "succeeded" and remote is not None:
            _produce_remote_export(
                target=target,
                remote=remote,
                workspace=workspace,
                config_path=config_path,
                lifecycle=lifecycle_value,
                transport=comfy_transport,
                runner=remote_runner or runner,
                deadline_monotonic=time.monotonic()
                + max(0.05, min(binding.planned_cancel_at, binding.hard_end_at) - time.time() - 60.0),
            )
        _write_json(run_path, {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "run_id": target.name,
            "status": result.get("status", "LIVE_NOT_VERIFIED"),
            "order_id": order_id,
            "created_at": created_at,
            "hard_end_at": state.hard_end_at,
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "bundle_hashes": b_hashes,
            "lifecycle_phase": lifecycle_value.phase,
            "candidate": candidate_facts,
            "verification": {
                "RUNTIME_READINESS": "VERIFIED",
                "SOURCE_STATUS": "DRAFT",
                "GPU_QUALITY_VERIFIED": False,
                "GENERATION_STATE": "SUBMITTED" if result.get("job_id") else "NOT_RUN",
            },
            "financial_status": _financial_status(binding),
            "owner_risk_admission_sha256": binding.owner_risk_admission_sha256,
            "provider_billing_verified": False,
            "result": result,
        })
        return {"status": result.get("status", "LIVE_NOT_VERIFIED"), "run_dir": str(target), "result": result}
    finally:
        lock.release()


def owner_risk_pending(
    *,
    pending_quote: Path,
    output: Path,
) -> dict[str, Any]:
    """Read a pending quote read-only and write a non-authorizing forecast."""

    try:
        payload = owner_risk.create_pending(
            pending_quote=pending_quote,
            output_path=output,
        )
    except owner_risk.OwnerRiskError as exc:
        raise OperatorError(f"owner-risk pending refused: {exc}") from exc
    return {
        "status": payload.get("status"),
        "financial_status": payload.get("financial_status"),
        "authorizes_session": False,
        "provider_billing_verified": False,
        "path": str(output),
        "quote_sha256": payload.get("quote_sha256"),
        "observed_projection_60m_usd": (
            str(payload.get("observed_projection_60m_usd"))
            if payload.get("observed_projection_60m_usd") is not None
            else None
        ),
        "policy_forecast_60m_usd": str(payload.get("policy_forecast_60m_usd")),
    }


def owner_risk_admit(
    *,
    quote: Path,
    output: Path,
    approval_doc: Path = owner_risk.RISK_DOC,
    session_approval_doc: Path = owner_risk.SESSION_APPROVAL_DOC,
    server_id: int = owner_risk.ALLOWED_SERVER_ID,
    hoster_id: int = owner_risk.ALLOWED_HOSTER_ID,
    valid_hours: int = 2,
    observation_origin: str | None = None,
    expected_quote_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the explicit scoped owner financial-risk admission; offline only."""

    try:
        admission = owner_risk.create_admission(
            quote=quote,
            output_path=output,
            approval_doc=approval_doc,
            session_approval_doc=session_approval_doc,
            server_id=server_id,
            hoster_id=hoster_id,
            valid_hours=valid_hours,
            observation_origin=observation_origin,
            expected_quote_sha256=expected_quote_sha256,
        )
        digest = owner_risk.admission_digest(admission)
    except owner_risk.OwnerRiskError as exc:
        raise OperatorError(f"owner-risk admission refused: {exc}") from exc
    return {
        "status": "OWNER_ACCEPTED_FINANCIAL_RISK",
        "admission_id": admission.admission_id,
        "admission_sha256": digest,
        "path": str(output),
        "created_at": admission.created_at.isoformat(),
        "expires_at": admission.expires_at.isoformat(),
        "projection": dict(admission.projection),
        "verification": dict(admission.verification),
        "provenance": dict(admission.provenance),
        "note": "Local owner attestation; not a signature or provider billing audit.",
    }


def fresh_quote(
    *,
    server_id: int,
    output: Path,
    currency: str = owner_risk.ALLOWED_CURRENCY,
    live_read: bool = False,
    api_key_env: str = "CLORE_API_KEY",
    timeout_seconds: float = 10.0,
    max_response_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """One explicit GET-only marketplace read; never an order or claim."""

    if not live_read:
        raise OperatorError(
            "fresh quote requires explicit --live-read; no network/env access was made"
        )
    try:
        document = asyncio.run(
            marketplace_quote.collect_marketplace_quote(
                server_id=server_id,
                currency=currency,
                output_path=output,
                allow_live=True,
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        )
    except marketplace_quote.MarketplaceQuoteError as exc:
        raise OperatorError(f"fresh marketplace quote refused: {exc}") from exc
    return {
        "status": document.get("status"),
        "path": str(output),
        "server_id": document.get("server_id"),
        "hoster_id": document.get("hoster_id"),
        "currency": document.get("currency"),
        "price_per_day": document.get("price_per_day"),
        "mrl_raw_seconds": document.get("mrl_raw_seconds"),
        "facts_sha256": document.get("facts_sha256"),
        "observed_at": document.get("observed_at"),
        "authorizes_session": False,
        "provider_billing_verified": False,
        "fits_code_cap": (document.get("proposal") or {}).get("fits_code_cap"),
        "note": (
            "Read-only marketplace fact, not VERIFIED_PROVIDER_BILLING; "
            "no order/claim/Pay was created."
        ),
    }


def owner_risk_approve_proposal(
    *,
    quote: Path,
    output: Path,
    server_id: int,
    hoster_id: int | None = None,
    currency: str = owner_risk.ALLOWED_CURRENCY,
    approval_doc: Path = owner_risk.RISK_DOC,
    session_approval_doc: Path = owner_risk.SESSION_APPROVAL_DOC,
    valid_hours: int = 2,
    approved_by: str = "owner",
    confirm_approval: str | None = None,
    expected_quote_sha256: str | None = None,
) -> dict[str, Any]:
    """Create one explicit data-driven scoped admission; never a claim/order."""

    try:
        admission = owner_risk.create_scoped_admission(
            quote=quote,
            output_path=output,
            server_id=server_id,
            hoster_id=hoster_id,
            currency=currency,
            approval_doc=approval_doc,
            session_approval_doc=session_approval_doc,
            valid_hours=valid_hours,
            approved_by=approved_by,
            approval_confirm=confirm_approval,
            expected_quote_sha256=expected_quote_sha256,
        )
        digest = owner_risk.admission_digest(admission)
    except owner_risk.OwnerRiskError as exc:
        raise OperatorError(f"scoped owner-risk admission refused: {exc}") from exc
    return {
        "status": "OWNER_ACCEPTED_FINANCIAL_RISK",
        "admission_id": admission.admission_id,
        "permission_id": owner_risk.permission_id_for_admission(admission),
        "admission_sha256": digest,
        "path": str(output),
        "server_id": admission.scope.get("server_id"),
        "hoster_id": admission.scope.get("hoster_id"),
        "currency": admission.scope.get("currency"),
        "price_per_day": admission.scope.get("price_per_day"),
        "created_at": admission.created_at.isoformat(),
        "expires_at": admission.expires_at.isoformat(),
        "projection": dict(admission.projection),
        "verification": dict(admission.verification),
        "provenance": dict(admission.provenance),
        "note": (
            "Explicit local owner attestation over the named proposal; not a "
            "signature/provider billing audit and not a claim until session-bind."
        ),
    }


def _load_session_state(path: Path) -> Any:
    try:
        return clore_session.load_state(path)
    except Exception as exc:
        raise OperatorError(f"cannot load Clore session state: {type(exc).__name__}") from exc
def bind_session(
    *,
    session_state: Path,
    quote: Path,
    order_id: int,
    created_at: int,
    cap_usd: str = "1.00",
    duration_seconds: int = 3600,
    planned_cancel_at: int | None = None,
    hard_end_at: int | None = None,
    owner_risk_admission: Path | None = None,
    now: datetime | None = None,
    server_id: int | None = None,
) -> dict[str, Any]:
    """Bind a real order and provider/owner machine quote; no network/secrets."""

    if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id <= 0:
        raise OperatorError("order_id must be a positive integer")
    if isinstance(created_at, bool) or not isinstance(created_at, int) or created_at <= 0:
        raise OperatorError("created_at must be a positive epoch integer")
    try:
        data = json.loads(quote.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorError(f"cannot read machine quote: {type(exc).__name__}") from exc
    if isinstance(data, Mapping) and isinstance(data.get("quote"), Mapping):
        data = data["quote"]
    if not isinstance(data, Mapping):
        raise OperatorError("machine quote must be an object")
    try:
        quote_obj = clore_session.CloreQuote.from_dict(data)
        cap = Decimal(str(cap_usd))
    except Exception as exc:
        raise OperatorError(f"machine quote/cap is invalid: {type(exc).__name__}") from exc
    if cap <= 0 or cap > Decimal("1.00"):
        raise OperatorError("session cap must be <= 1.00 USD")
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int):
        raise OperatorError("duration_seconds must be an integer")
    if not 1 <= duration_seconds <= 3600:
        raise OperatorError("duration_seconds must be <= 3600")
    admission = None
    if owner_risk_admission is not None:
        try:
            admission = owner_risk.load_admission(
                owner_risk_admission,
                now=now,
                allow_expired=False,
                require_fresh=True,
            )
            if Path(quote).resolve() != Path(str(admission.quote.get("path"))).resolve():
                raise owner_risk.OwnerRiskError(
                    "machine quote path differs from the admitted quote path"
                )
            if _sha_file(quote).lower() != str(admission.quote.get("sha256")).lower():
                raise owner_risk.OwnerRiskError("machine quote hash differs from admission")
            admission_issues = owner_risk.validate_admission(
                admission,
                allow_expired=False,
                quote_sha256=_sha_file(quote),
            )
            if admission_issues:
                raise owner_risk.OwnerRiskError("; ".join(admission_issues))
            if server_id is not None and server_id != int(
                admission.scope["server_id"]
            ):
                raise owner_risk.OwnerRiskError(
                    "explicit server id differs from the owner-risk admission"
                )
        except owner_risk.OwnerRiskError as exc:
            raise OperatorError(f"owner-risk session binding refused: {exc}") from exc
    try:
        state = clore_session.initialize_state(
            session_state,
            order_id=order_id,
            server_id=(
                int(admission.scope["server_id"])
                if admission is not None
                else (server_id or clore_session.EXPECTED_SERVER_ID)
            ),
            created_at=created_at,
            quote=quote_obj,
            currency=quote_obj.currency,
            cap_usd=cap,
            duration_seconds=duration_seconds,
            planned_cancel_at=planned_cancel_at,
            hard_end_at=hard_end_at,
            owner_risk_admission=admission,
            now=now,
        )
        binding = v3_lifecycle.SessionBinding.from_state(state)
    except Exception as exc:
        raise OperatorError(
            f"session binding refused: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        observation = v3_readiness.create_quote_observation(
            quote_path=quote,
            session_state=session_state,
            binding=binding,
        )
    except v3_readiness.ReadinessError as exc:
        raise OperatorError(str(exc)) from exc
    return {
        "status": "SESSION_BOUND",
        "session_state": str(session_state),
        "session": binding.to_dict(),
        "cap_usd": str(cap),
        "quote_sha256": _sha_file(quote),
        "quote_observation": str(observation.path),
        "quote_observation_sha256": observation.sha256,
        "hard_end_at": state.hard_end_at,
        "planned_cancel_at": state.planned_cancel_at,
        "financial_status": (
            "OWNER_ACCEPTED_FINANCIAL_RISK" if admission is not None else "STRICT_QUOTE"
        ),
        "owner_risk_admission_id": (
            admission.admission_id if admission is not None else None
        ),
        "owner_risk_admission_sha256": (
            state.owner_risk_sha256 if admission is not None else None
        ),
        "provider_billing_verified": False,
    }


def approve_source(
    *,
    approval_out: Path,
    approval_doc: Path = v3_readiness.CURRENT_APPROVAL_DOC,
    approved_by: str = "owner",
    cap_usd: str = "1.00",
    duration_seconds: int = 3600,
    valid_hours: int = v3_readiness.DEFAULT_APPROVAL_VALID_HOURS,
) -> dict[str, Any]:
    """Create the explicit owner/source attestation; never a check override."""

    bundle = load_bundle()
    source_hashes = _bundle_hashes(bundle)
    try:
        approval = v3_readiness.create_source_approval(
            bundle=bundle,
            source_hashes=source_hashes,
            output_path=approval_out,
            approval_doc=approval_doc,
            approved_by=approved_by,
            cap_usd=cap_usd,
            duration_seconds=duration_seconds,
            valid_hours=valid_hours,
        )
    except v3_readiness.ReadinessError as exc:
        raise OperatorError(str(exc)) from exc
    return {
        "status": "SOURCE_APPROVED",
        "approval": str(approval.path),
        "approval_sha256": approval.sha256,
        "created_at": approval.payload.get("created_at"),
        "expires_at": approval.payload.get("expires_at"),
        "source_bundle_sha256": approval.payload.get("source_bundle_sha256"),
        "scope": approval.payload.get("scope"),
        "note": "Local owner attestation; not a cryptographic signature or host attestation.",
    }


def source_readiness() -> dict[str, Any]:
    """Source-only pre-Pay readiness; no session, SSH, env, network or mutation."""

    bundle = load_bundle()
    source_hashes = _bundle_hashes(bundle)
    issues = v3_readiness.validate_source_bundle(bundle, source_hashes)
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "SOURCE_READY" if not issues else "SOURCE_BLOCKED",
        "source_bundle": source_hashes,
        "source_bundle_sha256": v3_readiness.source_bundle_digest(source_hashes),
        "manifest_status": bundle["manifest"].get("status"),
        "source_promoted": False,
        "issues": list(issues),
        "runtime_evidence": "NOT_RUN",
        "gpu_quality_verified": False,
        "financial_status": "OWNER_RISK_NOT_ADMITTED",
        "provider_billing_verified": False,
        "live_ready": False,
        "next_action": (
            "after Pay: session-bind actual order/machine quote, then prepare with "
            "the owner source approval and explicit SSH identity"
            if not issues
            else "fix source pins/licenses/dates/plan before any live preparation"
        ),
    }


def _run_backend_once(
    *,
    bundle: Mapping[str, Any],
    target: Path,
    config_path: Path,
    binding: v3_lifecycle.SessionBinding,
    lifecycle: v3_lifecycle.LifecycleState | None = None,
    backend_runner: Callable[..., Any] | None = None,
    pre_submit_guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run the existing backend once through its loopback HTTP API.

    Every startup/queue/HTTP step receives the same immutable
    ``RunDeadline`` built from the full ``SessionBinding``.  A deadline expiry
    before a request is recorded as ``DEADLINE_BEFORE_POST``; a deadline expiry
    after a POST has been sent stays ``BackendAmbiguous`` and is never retried.
    """

    if not isinstance(binding, v3_lifecycle.SessionBinding):
        raise OperatorError("backend run requires a complete SessionBinding")
    if lifecycle is not None:
        if (
            not isinstance(lifecycle, v3_lifecycle.LifecycleState)
            or lifecycle.run_id != target.name
            or lifecycle.session != binding
        ):
            raise OperatorError("backend run lifecycle identity mismatch")
    deadline = v3_backend.RunDeadline(binding)
    if backend_runner is not None:
        return backend_runner(bundle=bundle, target=target, binding=binding)

    def _guard_error() -> Exception | None:
        if pre_submit_guard is None:
            return None
        try:
            pre_submit_guard()
        except Exception as exc:
            return exc
        return None

    job = bundle.get("job")
    if not isinstance(job, Mapping):
        raise OperatorError("v3 job is missing")
    prompt = job.get("prompt")
    seed = job.get("seed")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(seed, int):
        raise OperatorError("v3 job prompt/seed is invalid")
    try:
        deadline.require_remaining()
    except v3_backend.BackendDeadlineExpired:
        return {"status": "DEADLINE_BEFORE_POST", "no_post": True}

    intent_path = _validate_owned_path(target, target / "submit-intent.json", name="submit intent")
    job_id_path = _validate_owned_path(target, target / "job-id.json", name="job id")
    result_path = _validate_owned_path(target, target / "backend-result.json", name="backend result")
    log_path = _validate_owned_path(target, target / "backend.log", name="backend log")
    prompt_hash = _sha_bytes(prompt.encode("utf-8"))

    intent: dict[str, Any] | None = None
    if intent_path.is_file():
        intent = _read_json(intent_path)
        if (
            intent.get("schema") != "anime-video-studio/p1-v3-submit-intent/v1"
            or intent.get("run_id") != target.name
            or intent.get("prompt_sha256") != prompt_hash
            or intent.get("seed") != seed
        ):
            raise OperatorError("existing submit intent identity mismatch")
        if intent.get("status") == "AMBIGUOUS_DISPATCH":
            return {"status": "AMBIGUOUS_DISPATCH", "intent": str(intent_path)}
        if intent.get("status") == "DEADLINE_BEFORE_POST":
            return {"status": "DEADLINE_BEFORE_POST", "intent": str(intent_path), "no_post": True}
        job_id = intent.get("job_id")
        if not job_id:
            # An intent without a durable job id cannot be safely re-POSTed.
            raise OperatorError("existing submit intent has no job identity; no re-POST")
        if not isinstance(intent.get("timeout_seconds"), int):
            raise OperatorError("existing submit intent timeout is missing/invalid")
        process = v3_backend.BackendProcess.build(
            config_path=config_path,
            db_path=target / "jobs.sqlite3",
            artifacts_dir=target / "artifacts",
            log_path=log_path,
            run_dir=target,
        )
        try:
            process.start(deadline=deadline)
        except v3_backend.BackendDeadlineExpired:
            return {
                "status": "DEADLINE_BEFORE_POST",
                "job_id": str(job_id),
                "no_post": True,
            }
        try:
            client = v3_backend.BackendClient(
                process.base_url, timeout=30.0, deadline=deadline
            )
            try:
                client.health(deadline=deadline)
            except v3_backend.BackendDeadlineExpired:
                return {
                    "status": "DEADLINE_BEFORE_POST",
                    "job_id": str(job_id),
                    "no_post": True,
                }
            remaining = deadline.remaining()
            if remaining <= 0:
                return {
                    "status": "DEADLINE_BEFORE_POST",
                    "job_id": str(job_id),
                    "no_post": True,
                }
            try:
                final = client.wait_terminal(
                    str(job_id),
                    deadline=time.monotonic() + max(1.0, remaining),
                    session_deadline=deadline,
                )
            except v3_backend.BackendDeadlineExpired:
                return {
                    "status": "DEADLINE_AFTER_POST",
                    "job_id": str(job_id),
                    "no_repost": True,
                }
            result = {
                "status": final.get("status", "UNKNOWN"),
                "job_id": str(job_id),
                "job": final,
            }
            if final.get("status") == "succeeded":
                try:
                    result["result"] = client.get_result(
                        str(job_id), deadline=deadline
                    )
                except v3_backend.BackendError as exc:
                    result["result_error"] = type(exc).__name__
            _write_json(result_path, result)
            return result
        finally:
            process.stop()

    try:
        timeout_seconds = deadline.job_timeout_seconds()
    except v3_backend.BackendDeadlineExpired:
        return {"status": "DEADLINE_BEFORE_POST", "no_post": True}
    guard_error = _guard_error()
    if guard_error is not None:
        return {
            "status": "SUBMIT_BLOCKED",
            "no_post": True,
            "reason": type(guard_error).__name__,
        }
    idempotency_key = f"{target.name}-{prompt_hash[:32]}"
    intent = {
        "schema": "anime-video-studio/p1-v3-submit-intent/v1",
        "run_id": target.name,
        "prompt_sha256": prompt_hash,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "idempotency_key": idempotency_key,
        "status": "INTENT_RECORDED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deadline": deadline.to_dict(),
    }
    _write_json(intent_path, intent)
    process = v3_backend.BackendProcess.build(
        config_path=config_path,
        db_path=target / "jobs.sqlite3",
        artifacts_dir=target / "artifacts",
        log_path=log_path,
        run_dir=target,
    )
    try:
        process.start(deadline=deadline)
    except v3_backend.BackendDeadlineExpired:
        intent["status"] = "DEADLINE_BEFORE_POST"
        _write_json(intent_path, intent)
        return {"status": "DEADLINE_BEFORE_POST", "intent": str(intent_path), "no_post": True}
    try:
        client = v3_backend.BackendClient(
            process.base_url,
            timeout=min(30.0, max(1.0, deadline.remaining())),
            deadline=deadline,
        )
        intent["base_url"] = process.base_url
        _write_json(intent_path, intent)
        try:
            client.health(deadline=deadline)
        except v3_backend.BackendDeadlineExpired:
            intent["status"] = "DEADLINE_BEFORE_POST"
            _write_json(intent_path, intent)
            return {
                "status": "DEADLINE_BEFORE_POST",
                "intent": str(intent_path),
                "no_post": True,
            }
        config = load_config(config_path)
        token = os.environ.get(config.token_env) if config.token_env else None
        try:
            v3_backend.check_comfy_queue(
                config.base_url, token, deadline=deadline
            )
            deadline.require_remaining()
        except v3_backend.BackendDeadlineExpired:
            intent["status"] = "DEADLINE_BEFORE_POST"
            _write_json(intent_path, intent)
            return {
                "status": "DEADLINE_BEFORE_POST",
                "intent": str(intent_path),
                "no_post": True,
            }
        # No POST has been sent yet.  Clamp the stored timeout once to the
        # remaining absolute budget; never rewrite it after a durable job id or
        # an ambiguous dispatch exists.
        try:
            allowed_timeout = deadline.job_timeout_seconds()
        except v3_backend.BackendDeadlineExpired:
            intent["status"] = "DEADLINE_BEFORE_POST"
            _write_json(intent_path, intent)
            return {
                "status": "DEADLINE_BEFORE_POST",
                "intent": str(intent_path),
                "no_post": True,
            }
        if intent["timeout_seconds"] > allowed_timeout:
            intent["timeout_seconds"] = max(1, allowed_timeout - 1)
            _write_json(intent_path, intent)
        guard_error = _guard_error()
        if guard_error is not None:
            intent["status"] = "SUBMIT_BLOCKED"
            intent["error_type"] = type(guard_error).__name__
            _write_json(intent_path, intent)
            return {
                "status": "SUBMIT_BLOCKED",
                "intent": str(intent_path),
                "no_post": True,
                "reason": type(guard_error).__name__,
            }
        try:
            response = client.post_job(
                prompt=prompt,
                seed=seed,
                timeout_seconds=intent["timeout_seconds"],
                idempotency_key=idempotency_key,
                deadline=deadline,
            )
        except v3_backend.BackendDeadlineExpired:
            # Raised before a request is sent (validation/remaining check).
            intent["status"] = "DEADLINE_BEFORE_POST"
            _write_json(intent_path, intent)
            return {
                "status": "DEADLINE_BEFORE_POST",
                "intent": str(intent_path),
                "no_post": True,
            }
        except v3_backend.BackendAmbiguous as exc:
            intent["status"] = "AMBIGUOUS_DISPATCH"
            intent["error_type"] = type(exc).__name__
            _write_json(intent_path, intent)
            return {"status": "AMBIGUOUS_DISPATCH", "intent": str(intent_path)}
        job_payload = response.get("job")
        if not isinstance(job_payload, Mapping) or not isinstance(job_payload.get("job_id"), str):
            intent["status"] = "AMBIGUOUS_DISPATCH"
            _write_json(intent_path, intent)
            return {"status": "AMBIGUOUS_DISPATCH", "intent": str(intent_path)}
        job_id = str(job_payload["job_id"])
        intent["job_id"] = job_id
        intent["status"] = "JOB_RECORDED"
        _write_json(intent_path, intent)
        _write_json(job_id_path, {"job_id": job_id, "idempotency_key": idempotency_key})
        try:
            final = client.wait_terminal(
                job_id,
                deadline=time.monotonic() + max(1.0, deadline.remaining()),
                session_deadline=deadline,
            )
        except v3_backend.BackendDeadlineExpired:
            return {
                "status": "DEADLINE_AFTER_POST",
                "job_id": job_id,
                "no_repost": True,
            }
        result = {"status": final.get("status", "UNKNOWN"), "job_id": job_id, "job": final}
        if final.get("status") == "succeeded":
            try:
                result["result"] = client.get_result(job_id, deadline=deadline)
            except v3_backend.BackendError as exc:
                result["result_error"] = type(exc).__name__
        _write_json(result_path, result)
        return result
    finally:
        process.stop()

def _parse_ratio(value: Any) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, denominator = value.split("/", 1)
    try:
        num = int(numerator)
        den = int(denominator)
    except ValueError:
        return None
    if den == 0:
        return None
    return num / den


def _remaining(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return deadline_monotonic - time.monotonic()


def _monotonic_timeout(deadline_monotonic: float, requested: float) -> float:
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise OperatorError("absolute deadline expired before a remote call")
    return max(0.05, min(float(requested), remaining))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _validate_media_payload(payload: Any) -> tuple[dict[str, Any] | None, int, float | None, list[str]]:
    problems: list[str] = []
    if not isinstance(payload, dict):
        return None, 0, None, ["ffprobe JSON root is not an object"]
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return None, 0, None, ["ffprobe streams is not a list"]
    videos: list[dict[str, Any]] = []
    audio_count = 0
    for item in streams:
        if not isinstance(item, dict):
            problems.append("ffprobe stream entry is not an object")
            continue
        codec_type = item.get("codec_type")
        if not isinstance(codec_type, str):
            problems.append("ffprobe stream codec_type is not a string")
            continue
        if codec_type == "video":
            videos.append(item)
        elif codec_type == "audio":
            audio_count += 1
    if len(videos) != 1:
        problems.append("expected exactly one video stream")
        return None, audio_count, None, problems
    video = videos[0]
    width = video.get("width")
    height = video.get("height")
    if isinstance(width, bool) or not isinstance(width, int):
        problems.append("video width is not an integer")
    if isinstance(height, bool) or not isinstance(height, int):
        problems.append("video height is not an integer")
    fps_value = video.get("avg_frame_rate") or video.get("r_frame_rate")
    fps = _parse_ratio(fps_value) if isinstance(fps_value, str) else None
    if fps is None or not math.isfinite(fps):
        problems.append("video fps is missing/non-finite")
    frame_count_raw = video.get("nb_read_frames") or video.get("nb_frames")
    try:
        frame_count = int(frame_count_raw)
    except (TypeError, ValueError):
        problems.append("video frame count is missing/invalid")
        frame_count = None
    fmt = payload.get("format")
    if not isinstance(fmt, dict):
        problems.append("ffprobe format is not an object")
        duration = None
    else:
        duration = _finite_number(fmt.get("duration"))
        if duration is None:
            problems.append("format duration is missing/non-finite")
    if problems:
        return None, audio_count, duration, problems
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "fps_raw": fps_value,
        "frame_count": frame_count,
        "duration": duration,
        "audio_streams": audio_count,
    }, audio_count, duration, []


def _inspect_media(path: Path, *, deadline_monotonic: float | None = None) -> dict[str, Any]:
    """Require both exact metadata and a real full ffmpeg decode.

    ffprobe metadata alone is never a successful decode; ffmpeg alone cannot
    prove frame count/fps/duration; PyAV is not implemented. Any missing tool,
    malformed metadata, wrong parameter, audio stream, timeout or nonzero
    decode exit returns NOT_VERIFIED.
    """

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    remaining = _remaining(deadline_monotonic)
    if remaining is not None and remaining <= 0:
        return {"status": "NOT_VERIFIED", "reason": "collect deadline expired before decode"}
    if not ffprobe or not ffmpeg:
        return {
            "status": "NOT_VERIFIED",
            "decoder": "NONE",
            "reason": (
                "full decode requires ffprobe+ffmpeg; PyAV is not implemented; "
                "structural MP4 check is not a decode"
            ),
        }
    probe_timeout = min(60.0, remaining) if remaining is not None else 60.0
    if probe_timeout <= 0:
        return {"status": "NOT_VERIFIED", "reason": "collect deadline expired before ffprobe"}
    try:
        probed = run_bounded_process(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-print_format",
                "json",
                "-show_entries",
                (
                    "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,"
                    "nb_frames,nb_read_frames:format=duration"
                ),
                str(path),
            ],
            timeout=probe_timeout,
            max_output_bytes=256 * 1024,
        )
    except (RemoteError, OSError) as exc:
        return {"status": "NOT_VERIFIED", "reason": f"ffprobe failed: {type(exc).__name__}"}
    if probed.returncode != 0 or probed.timed_out:
        return {"status": "NOT_VERIFIED", "reason": f"ffprobe exit {probed.returncode}"}
    try:
        payload = json.loads(
            probed.stdout.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {"status": "NOT_VERIFIED", "reason": "ffprobe JSON is invalid/non-finite"}
    metadata, audio_count, duration, schema_problems = _validate_media_payload(payload)
    if schema_problems or metadata is None or duration is None:
        return {
            "status": "NOT_VERIFIED",
            "decoder": "ffprobe",
            "reason": "; ".join(schema_problems or ["ffprobe schema invalid"]),
        }
    width = metadata["width"]
    height = metadata["height"]
    fps = metadata["fps"]
    fps_value = metadata["fps_raw"]
    frame_count = metadata["frame_count"]
    audio_streams = audio_count
    expected = {
        "width": EXPECTED_VIDEO_WIDTH,
        "height": EXPECTED_VIDEO_HEIGHT,
        "fps": EXPECTED_VIDEO_FPS,
        "frame_count": EXPECTED_VIDEO_FRAMES,
        "duration": EXPECTED_VIDEO_DURATION,
        "audio_streams": 0,
    }
    problems: list[str] = []
    if width != EXPECTED_VIDEO_WIDTH or height != EXPECTED_VIDEO_HEIGHT:
        problems.append(f"dimensions {width}x{height} != 704x1280")
    if fps is None or abs(fps - EXPECTED_VIDEO_FPS) > 0.001:
        problems.append(f"fps {fps_value!r} != 24")
    if frame_count != EXPECTED_VIDEO_FRAMES:
        problems.append(f"frame_count {frame_count} != 121")
    if abs(duration - EXPECTED_VIDEO_DURATION) > 0.05:
        problems.append(f"duration {duration} != {EXPECTED_VIDEO_DURATION:.6f}")
    if audio_streams:
        problems.append(f"audio_streams {audio_streams} != 0")
    if problems:
        return {
            "status": "NOT_VERIFIED",
            "decoder": "ffprobe",
            "reason": "; ".join(problems),
            "metadata": metadata,
            "expected": expected,
        }

    remaining = _remaining(deadline_monotonic)
    decode_timeout = min(DECODE_TIMEOUT_SECONDS, remaining) if remaining is not None else DECODE_TIMEOUT_SECONDS
    if decode_timeout <= 0:
        return {"status": "NOT_VERIFIED", "reason": "collect deadline expired before ffmpeg decode"}
    try:
        decoded = run_bounded_process(
            [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
            timeout=decode_timeout,
            max_output_bytes=64 * 1024,
        )
    except (RemoteError, OSError) as exc:
        return {"status": "NOT_VERIFIED", "reason": f"ffmpeg full decode failed: {type(exc).__name__}"}
    if decoded.returncode != 0 or decoded.timed_out:
        return {
            "status": "NOT_VERIFIED",
            "decoder": "ffprobe",
            "reason": f"ffmpeg full decode exit {decoded.returncode}",
            "metadata": metadata,
            "expected": expected,
        }
    return {
        "status": "VERIFIED",
        "decoder": "ffprobe+ffmpeg_full_decode",
        "full_decode": True,
        "metadata": metadata,
        "expected": expected,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "audio_streams": 0,
    }


def _copy_bounded(
    source: Path,
    target: Path,
    *,
    max_bytes: int,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise OperatorError(f"source is not a regular file: {source}")
    size = source.stat().st_size
    if size <= 0 or size > max_bytes:
        raise OperatorError(f"source size is outside the bound: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise OperatorError(f"staging target already exists: {target}")
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as reader, target.open("xb") as writer:
            while True:
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    raise OperatorError("collect deadline expired during copy")
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > max_bytes:
                    raise OperatorError("copy exceeded max_bytes")
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return {"size_bytes": copied, "sha256": digest.hexdigest(), "source": str(source), "target": str(target)}


def _ensure_run_document(
    target: Path, bundle_hashes: Mapping[str, str] | None = None
) -> dict[str, Any]:
    _assert_no_link_components(target, stop=target.parent)
    path = _validate_owned_path(target, target / "run.json", name="run.json")
    if target.exists() and not target.is_dir():
        raise OperatorError("run path exists but is not a directory")
    if target.exists() and not path.is_file():
        entries = [item.name for item in target.iterdir() if item.name != "operator.lock"]
        if entries:
            raise OperatorError("nonempty run without a valid run.json")
    if path.is_file():
        payload = _read_json(path)
        if payload.get("schema") != SCHEMA or payload.get("task_id") != TASK_ID:
            raise OperatorError("existing run.json schema/task identity does not match")
        if payload.get("run_id") != target.name:
            raise OperatorError("existing run.json run_id does not match the run directory")
        if bundle_hashes is not None:
            _validate_bundle_identity(payload.get("bundle_hashes"), bundle_hashes)
        return payload
    payload = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "run_id": target.name,
        "status": "COLLECT_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        "bundle_hashes": dict(bundle_hashes or {}),
    }
    _write_json(path, payload)
    return payload


def _validate_collect_inputs(
    *,
    target: Path,
    artifact: Path | None,
    remote: RemoteIdentity | None,
    remote_artifact_path: str | None,
    remote_helper_path: str | None,
    remote_expected_size: int | None,
    remote_expected_sha256: str | None,
    remote_evidence: Sequence[str],
) -> None:
    """Validate exact local/remote namespaces before files or network calls."""

    remote_requested = remote is not None or remote_artifact_path is not None
    if remote_requested:
        if artifact is not None:
            raise OperatorError("choose local artifact or remote exact artifact, not both")
        if remote is None or not remote_artifact_path or not remote_helper_path:
            raise OperatorError("remote collect requires identity, artifact path and helper path")
        if (
            isinstance(remote_expected_size, bool)
            or not isinstance(remote_expected_size, int)
            or remote_expected_size <= 0
            or not isinstance(remote_expected_sha256, str)
            or len(remote_expected_sha256) != 64
        ):
            raise OperatorError("remote collect requires expected size/sha256")
        _validate_remote_absolute_path(remote_artifact_path, "remote artifact path")
        _validate_remote_absolute_path(remote_helper_path, "remote helper path")
        remote_name = PurePosixPath(remote_artifact_path).name
        if not remote_name or not remote_name.lower().endswith(".mp4"):
            raise OperatorError("remote artifact filename must be an exact MP4 result")
        if PurePosixPath(remote_helper_path).name != "v3_remote_fetch_helper.py":
            raise OperatorError("remote helper filename is not the pinned helper")
        if remote_evidence:
            raise OperatorError("remote raw evidence is not imported; evidence is generated locally")
        return
    if remote_evidence:
        raise OperatorError("remote evidence requires remote exact artifact")
    if artifact is not None:
        candidate = _validate_owned_path(target, Path(artifact), name="artifact")
        if candidate.name not in {"artifact.mp4", "result.mp4"}:
            raise OperatorError("artifact must be exactly artifact.mp4/result.mp4")


def _default_backend_artifact(target: Path) -> Path:
    """Return the exact backend-fetched MP4 for the durable job id.

    The existing application writes ``artifacts/<job_id>/result.mp4``.  A
    generic ``artifacts/result.mp4`` or a manually copied file is not accepted.
    """

    job_path = _validate_owned_path(target, target / "job-id.json", name="job id")
    result_path = _validate_owned_path(
        target, target / "backend-result.json", name="backend result"
    )
    if not job_path.is_file() or not result_path.is_file():
        raise OperatorError("collect requires a durable backend job/result record")
    job_record = _read_json(job_path)
    job_id = job_record.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", job_id):
        raise OperatorError("durable job id is invalid")
    result_record = _read_json(result_path)
    if result_record.get("job_id") != job_id:
        raise OperatorError("backend result job identity mismatch")
    payload = result_record.get("result")
    if not isinstance(payload, Mapping):
        raise OperatorError("backend result payload is missing")
    if isinstance(payload.get("result"), Mapping):
        payload = payload["result"]
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise OperatorError("backend result must contain exactly one artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, Mapping):
        raise OperatorError("backend artifact record is invalid")
    relative = artifact.get("relative_path")
    expected_relative = f"{job_id}/result.mp4"
    if relative != expected_relative:
        raise OperatorError("backend artifact is not the exact job result path")
    if artifact.get("media_type") != "video/mp4" or artifact.get("kind") != "video":
        raise OperatorError("backend artifact is not the expected video/mp4 result")
    candidate = _validate_owned_path(
        target, target / "artifacts" / expected_relative, name="backend artifact"
    )
    if not candidate.is_file() or _is_link(candidate):
        raise OperatorError("backend job artifact is missing/not regular")
    return candidate


def _validated_remote_export(
    *,
    target: Path,
    lifecycle: v3_lifecycle.LifecycleState | None,
    remote: RemoteIdentity,
    remote_root: str,
    remote_artifact_path: str,
    expected_size: int,
    expected_sha256: str,
    remote_helper_path: str,
) -> dict[str, Any]:
    """Require a pipeline-produced exact remote export mapping."""

    export_path = _validate_owned_path(
        target, target / "remote-export.json", name="remote export"
    )
    if not export_path.is_file():
        raise OperatorError(
            "remote collect requires remote-export.json from a validated pipeline; "
            "arbitrary CLI remote paths are not accepted"
        )
    export = _read_json(export_path)
    if (
        export.get("schema") != "anime-video-studio/p1-v3-remote-export/v1"
        or export.get("run_id") != target.name
        or not isinstance(export.get("job_id"), str)
        or not isinstance(export.get("prompt_id"), str)
        or export.get("remote_root") != remote_root
        or export.get("remote_artifact_path") != remote_artifact_path
        or export.get("remote_helper_path") != remote_helper_path
        or export.get("size") != expected_size
        or export.get("sha256") != expected_sha256
        or not isinstance(export.get("relative_path"), str)
        or export.get("filename") != PurePosixPath(remote_artifact_path).name
    ):
        raise OperatorError("remote export mapping identity mismatch")
    relative = PurePosixPath(export["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise OperatorError("remote export relative path is unsafe")
    if str(PurePosixPath(remote_root) / relative) != remote_artifact_path:
        raise OperatorError("remote export path does not match its owned root")
    if lifecycle is None:
        raise OperatorError("remote collect requires the bound run lifecycle")
    if (
        export.get("nonce") != lifecycle.nonce
        or export.get("session") != lifecycle.session.to_dict()
    ):
        raise OperatorError("remote export session/nonce identity mismatch")
    ssh = export.get("ssh_identity")
    expected_ssh = dict(lifecycle.pipeline or {}).get("ssh_identity")
    if not isinstance(ssh, Mapping) or ssh != expected_ssh:
        raise OperatorError("remote export SSH identity does not match the lifecycle")
    if (
        ssh.get("host") != remote.host
        or ssh.get("user") != remote.user
        or ssh.get("port") != remote.port
        or ssh.get("known_hosts") != str(remote.known_hosts)
        or not isinstance(ssh.get("fingerprint"), str)
    ):
        raise OperatorError("remote export SSH identity does not match the selected remote")
    return export


def _pinned_bundle_files(bundle: Mapping[str, Any]) -> tuple[tuple[Path, str, str], ...]:
    expected = _bundle_hashes(bundle)
    if bundle.get("workflow_sha256") != expected["workflow"]:
        raise OperatorError("bundle workflow hash is not pin-matched")
    manifest_workflow = bundle.get("manifest", {}).get("workflow", {}).get("sha256")
    if manifest_workflow != expected["workflow"]:
        raise OperatorError("bundle manifest workflow hash is not pin-matched")
    config = bundle.get("config")
    config_workflow = getattr(config, "workflow_sha256", None)
    if config_workflow != expected["workflow"]:
        raise OperatorError("bundle config workflow hash is not pin-matched")
    return (
        (V3_DERIVED, "input.png", expected["derived"]),
        (V3_WORKFLOW, "workflow.json", expected["workflow"]),
        (V3_MANIFEST, "manifest.json", expected["manifest"]),
        (V3_JOB, "job.json", expected["job"]),
        (V3_POLICY, "policy.json", expected["policy"]),
    )


def collect(
    *,
    run_dir: Path | None = None,
    artifact: Path | None = None,
    remote: RemoteIdentity | None = None,
    remote_artifact_path: str | None = None,
    remote_helper_path: str | None = None,
    remote_expected_size: int | None = None,
    remote_expected_sha256: str | None = None,
    remote_offset: int = 0,
    remote_prefix_sha256: str | None = None,
    expected_fingerprint: str | None = None,
    remote_evidence: Sequence[str] = (),
    timeout_seconds: float = 600.0,
    remote_runner: Any | None = None,
    remote_root: str = "/workspace",
) -> dict[str, Any]:
    """Collect critical MP4 first and generate evidence from the pinned v3 bundle.

    User-provided structured evidence is not imported. Evidence files are
    either copied from the verified v3 package (pin-matched) or generated from
    validated fields. Remote fetch is not implemented.
    """

    if timeout_seconds <= 0 or timeout_seconds > 3600:
        raise OperatorError("collect timeout must be bounded")
    entry_deadline = time.monotonic() + float(timeout_seconds)
    deadline_monotonic = entry_deadline
    target = Path(run_dir) if run_dir is not None else _default_run_dir()
    if remote is not None:
        _validate_remote_absolute_path(remote_root, "remote_root")
    _validate_collect_inputs(
        target=target,
        artifact=Path(artifact) if artifact is not None else None,
        remote=remote,
        remote_artifact_path=remote_artifact_path,
        remote_helper_path=remote_helper_path,
        remote_expected_size=remote_expected_size,
        remote_expected_sha256=remote_expected_sha256,
        remote_evidence=remote_evidence,
    )
    bundle = load_bundle()
    hashes = _bundle_hashes(bundle)
    pinned_files = _pinned_bundle_files(bundle)
    _assert_clean_path(target)
    _prevalidate_run_root(target, hashes)
    target.mkdir(parents=True, exist_ok=True)
    lock_path = _validate_owned_path(target, target / "operator.lock", name="operator lock")
    lock = InstanceLock(lock_path).acquire()
    try:
        _ensure_run_document(target, hashes)
        lifecycle_path = target / "lifecycle.json"
        lifecycle_value = None
        if lifecycle_path.is_file():
            lifecycle_value = v3_lifecycle.load_lifecycle(lifecycle_path)
            planned_end = float(lifecycle_value.session.planned_cancel_at)
            hard_end = float(lifecycle_value.session.hard_end_at)
            clipped = min(planned_end, hard_end) - 60.0
            if time.time() >= clipped:
                raise OperatorError("collect refused: session planned/hard deadline is too close")
            deadline_monotonic = min(entry_deadline, time.monotonic() + max(0.05, clipped - time.time()))
        staging = target / f"staging-{uuid.uuid4().hex[:8]}"
        _validate_owned_path(target, staging, name="staging")
        report_path = target / f"collect-{uuid.uuid4().hex[:8]}.json"
        _validate_owned_path(target, report_path, name="collect report")
        staging.mkdir()
        if remote is not None:
            if expected_fingerprint is None:
                raise OperatorError("remote collect requires expected fingerprint")
            runner = remote_runner or RemoteRunner()
            try:
                v3_lifecycle.verify_host_key(
                    runner=runner,
                    identity=remote,
                    expected_fingerprint=expected_fingerprint,
                    known_hosts=remote.known_hosts,
                    timeout=_monotonic_timeout(deadline_monotonic, 10.0),
                    deadline_monotonic=deadline_monotonic,
                )
            except v3_lifecycle.LifecycleError as exc:
                raise OperatorError(str(exc)) from exc
            if time.monotonic() >= deadline_monotonic:
                raise OperatorError("collect deadline expired after host-key verification")
            _validated_remote_export(
                target=target,
                lifecycle=lifecycle_value,
                remote=remote,
                remote_root=remote_root,
                remote_artifact_path=str(remote_artifact_path),
                expected_size=int(remote_expected_size),
                expected_sha256=str(remote_expected_sha256),
                remote_helper_path=str(remote_helper_path),
            )
            transfer_dir = _validate_owned_path(
                target, target / "collect-transfer", name="collect transfer"
            )
            transfer_dir.mkdir(parents=True, exist_ok=True)
            partial = _validate_owned_path(
                target, transfer_dir / "artifact.mp4.part", name="collect partial"
            )
            v3_lifecycle.assert_safe_local_path(partial)
            transfer_state_path = _validate_owned_path(
                target, target / "collect-transfer.json", name="collect transfer state"
            )
            transfer_state = None
            if transfer_state_path.is_file():
                transfer_state = _read_json(transfer_state_path)
            if transfer_state is None:
                if partial.exists():
                    raise OperatorError("unknown collect partial; refusing implicit resume")
                transfer_state = {
                    "schema": "anime-video-studio/p1-v3-collect-transfer/v1",
                    "run_id": target.name,
                    "nonce": uuid.uuid4().hex,
                    "remote_host": remote.host,
                    "remote_user": remote.user,
                    "remote_port": remote.port,
                    "remote_root": remote_root,
                    "remote_path": remote_artifact_path,
                    "remote_helper_path": remote_helper_path,
                    "artifact_name": PurePosixPath(str(remote_artifact_path)).name,
                    "expected_size": int(remote_expected_size),
                    "expected_sha256": str(remote_expected_sha256),
                    "offset": 0,
                    "prefix_sha256": None,
                    "helper_staged": False,
                    "status": "STARTED",
                    "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                }
                _write_json(transfer_state_path, transfer_state)
                if int(remote_offset or 0) != 0:
                    raise OperatorError(
                        "explicit remote offset requires an existing same-run partial"
                    )
            else:
                if transfer_state.get("schema") != "anime-video-studio/p1-v3-collect-transfer/v1":
                    raise OperatorError("collect transfer schema is unknown")
                if (
                    transfer_state.get("run_id") != target.name
                    or not isinstance(transfer_state.get("nonce"), str)
                    or len(transfer_state["nonce"]) < 16
                ):
                    raise OperatorError("collect transfer run identity mismatch")
                for key, expected in (
                    ("remote_host", remote.host),
                    ("remote_user", remote.user),
                    ("remote_port", remote.port),
                    ("remote_root", remote_root),
                    ("remote_path", remote_artifact_path),
                    ("remote_helper_path", remote_helper_path),
                    ("expected_size", int(remote_expected_size)),
                    ("expected_sha256", str(remote_expected_sha256)),
                ):
                    if transfer_state.get(key) != expected:
                        raise OperatorError("collect transfer identity mismatch")
            transfer_state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < 0 or offset > int(remote_expected_size):
                raise OperatorError("collect partial size is invalid")
            prefix_sha = None
            if offset:
                prefix_sha = _sha_file(partial)
                if transfer_state.get("prefix_sha256") not in (None, prefix_sha):
                    raise OperatorError("collect partial prefix hash mismatch")
            if offset == int(remote_expected_size):
                if prefix_sha != str(remote_expected_sha256):
                    raise OperatorError("complete collect partial hash mismatch")
                transfer_state["status"] = "COMPLETE"
                transfer_state["offset"] = offset
                _write_json(transfer_state_path, transfer_state)
            else:
                if not transfer_state.get("helper_staged"):
                    runner.put_file(
                        remote,
                        remote_helper_path,
                        ROOT / "scripts" / "p1" / "v3_remote_fetch_helper.py",
                        root=remote_root,
                        allow_same_hash=True,
                        timeout=_monotonic_timeout(deadline_monotonic, 60.0),
                        deadline_monotonic=deadline_monotonic,
                    )
                    transfer_state["helper_staged"] = True
                    transfer_state["status"] = "STAGED"
                    _write_json(transfer_state_path, transfer_state)
                if time.monotonic() >= deadline_monotonic:
                    raise OperatorError("collect deadline expired before transfer")
                try:
                    v3_lifecycle.fetch_exact_stream(
                        runner=runner,
                        identity=remote,
                        remote_helper_path=remote_helper_path,
                        remote_path=remote_artifact_path,
                        expected_size=int(remote_expected_size),
                        expected_sha256=str(remote_expected_sha256),
                        local_path=partial,
                        offset=offset,
                        expected_prefix_sha256=prefix_sha,
                        timeout=_monotonic_timeout(deadline_monotonic, 3600.0),
                        remote_root=remote_root,
                        deadline_monotonic=deadline_monotonic,
                    )
                except BaseException as exc:
                    transfer_state["status"] = "FAILED"
                    transfer_state["error_type"] = type(exc).__name__
                    if partial.is_file():
                        transfer_state["offset"] = partial.stat().st_size
                        transfer_state["prefix_sha256"] = _sha_file(partial)
                    else:
                        transfer_state["offset"] = 0
                    transfer_state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
                    _write_json(transfer_state_path, transfer_state)
                    raise
                transfer_state["status"] = "COMPLETE"
                transfer_state["offset"] = int(remote_expected_size)
                transfer_state["prefix_sha256"] = None
                transfer_state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(transfer_state_path, transfer_state)
            _copy_bounded(
                partial,
                staging / "artifact.mp4",
                max_bytes=clore_export.DEFAULT_MAX_ITEM_BYTES,
                deadline_monotonic=deadline_monotonic,
            )

        else:
            artifact_source = (
                _validate_owned_path(target, Path(artifact), name="artifact")
                if artifact is not None
                else _default_backend_artifact(target)
            )
            artifact_source = _validate_owned_path(target, artifact_source, name="artifact")
            if not artifact_source.is_file() or _is_link(artifact_source):
                raise OperatorError("critical artifact.mp4 is missing/not regular; refusing success")
            _copy_bounded(
                artifact_source,
                staging / "artifact.mp4",
                max_bytes=clore_export.DEFAULT_MAX_ITEM_BYTES,
                deadline_monotonic=deadline_monotonic,
            )
        for source, name, expected_hash in pinned_files:
            if _sha_file(source) != expected_hash:
                raise OperatorError(f"pinned bundle file changed: {name}")
            _copy_bounded(
                source,
                staging / name,
                max_bytes=clore_export.DEFAULT_MAX_ITEM_BYTES,
                deadline_monotonic=deadline_monotonic,
            )
        live_dir = target / "live"
        live_binding_path = live_dir / "candidate-binding.json"
        if live_binding_path.is_file():
            live_binding = _read_json(live_binding_path)
            files = live_binding.get("files")
            if not isinstance(files, Mapping):
                raise OperatorError("live candidate binding has no file hashes")
            for live_name, staging_name, hash_key in (
                ("runtime-evidence.json", "runtime/runtime-evidence.json", "runtime_evidence"),
                ("runtime-evidence.raw.json", "runtime/runtime-evidence.raw.json", "runtime_evidence_raw"),
                ("candidate-manifest.json", "runtime/candidate-manifest.json", "candidate_manifest"),
                ("candidate-config.json", "runtime/candidate-config.json", "candidate_config"),
            ):
                source = live_dir / live_name
                expected = files.get(hash_key)
                if not source.is_file() or not isinstance(expected, str):
                    raise OperatorError(f"live candidate artifact is missing: {live_name}")
                if _sha_file(source) != expected:
                    raise OperatorError(f"live candidate artifact hash mismatch: {live_name}")
                _copy_bounded(
                    source,
                    staging / staging_name,
                    max_bytes=clore_export.DEFAULT_MAX_ITEM_BYTES,
                    deadline_monotonic=deadline_monotonic,
                )
            _copy_bounded(
                live_binding_path,
                staging / "runtime/candidate-binding.json",
                max_bytes=clore_export.DEFAULT_MAX_ITEM_BYTES,
                deadline_monotonic=deadline_monotonic,
            )
        _write_json(
            staging / "request.json",
            {
                "schema": "anime-video-studio/p1-v3-request/v1",
                "task_id": TASK_ID,
                "prompt_sha256": _sha_bytes(bundle["positive"].encode("utf-8")),
                "seed": bundle["job"].get("seed"),
                "timeout_seconds": bundle["job"].get("timeout_seconds"),
                "workflow_sha256": hashes["workflow"],
                "manifest_status": bundle["manifest"].get("status"),
                "input_sha256": hashes["derived"],
            },
        )
        _write_json(
            staging / "runtime.json",
            {
                "schema": "anime-video-studio/p1-v3-runtime/v1",
                "task_id": TASK_ID,
                "source_sha256": hashes["source"],
                "derived_sha256": hashes["derived"],
                "workflow_sha256": hashes["workflow"],
                "manifest_sha256": hashes["manifest"],
                "job_sha256": hashes["job"],
                "policy_sha256": hashes["policy"],
                "params": {
                    "width": 704,
                    "height": 1280,
                    "frames": 121,
                    "fps": 24,
                    "seed": 424242,
                    "steps": 20,
                    "cfg": 5,
                },
                "decoder": {
                    "status": _decoder_status()["status"],
                    "full_decode": _decoder_status().get("full_decode"),
                    "pyav": _decoder_status().get("pyav"),
                },
                "financial_status": _financial_status(
                    lifecycle_value.session
                    if isinstance(lifecycle_value, v3_lifecycle.LifecycleState)
                    else None
                ),
                "owner_risk_admission_sha256": (
                    lifecycle_value.session.owner_risk_admission_sha256
                    if isinstance(lifecycle_value, v3_lifecycle.LifecycleState)
                    else None
                ),
                "provider_billing_verified": False,
                "live": "NOT_RUN",
            },
        )
        export_dir = _validate_owned_path(target, target / "export", name="export")
        plan = _collect_plan(export_dir, staging)
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise OperatorError("collect deadline expired before export")
        result = asyncio.run(
            clore_export.run_staged_export(
                plan,
                deadline_at=datetime.now(timezone.utc)
                + timedelta(seconds=remaining),
                max_item_bytes=clore_export.DEFAULT_MAX_ITEM_BYTES,
            )
        )
        exported_artifact = (
            result.run_dir / "artifact.mp4" if result.run_dir is not None else None
        )
        media = _inspect_media(
            exported_artifact
            if exported_artifact is not None and exported_artifact.is_file()
            else staging / "artifact.mp4",
            deadline_monotonic=deadline_monotonic,
        )
        status = (
            "COLLECTED_DECODE_VERIFIED"
            if media["status"] == "VERIFIED"
            else "COLLECTED_DECODE_NOT_VERIFIED"
        )
        if result.status not in {"COMPLETE", "PARTIAL"}:
            status = "COLLECT_FAILED"
        payload = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": status,
            "run_dir": str(target),
            "export_status": result.status,
            "completed": list(result.completed),
            "lost": list(result.lost),
            "manifest_path": str(result.manifest_path) if result.manifest_path else None,
            "media": media,
            "not_exported": list(NOT_EXPORTED_EVIDENCE),
            "not_implemented": [
                "user-provided structured evidence import",
                "costs.json source ledger",
            ],
            "remote": "BOUNDED_STREAM" if remote is not None else "NOT_RUN",
            "report": str(report_path),
            "next_action": (
                "inspect export manifest; decode remains NOT_VERIFIED"
                if media["status"] != "VERIFIED"
                else "inspect verified metadata and manifest"
            ),
        }
        _write_json(report_path, payload)
        return payload
    finally:
        lock.release()


def _remote_identity(args: argparse.Namespace) -> RemoteIdentity | None:
    host = getattr(args, "remote_host", None)
    if not host:
        return None
    identity = getattr(args, "remote_identity", None)
    known = getattr(args, "remote_known_hosts", None)
    if not identity or not known:
        raise OperatorError(
            "--remote-host requires --remote-identity and --remote-known-hosts"
        )
    return RemoteIdentity(
        host=host,
        user=getattr(args, "remote_user", "root") or "root",
        port=getattr(args, "remote_port", 22),
        identity_file=Path(identity),
        known_hosts=Path(known),
    )


def _add_remote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--remote-port", type=int, default=22)
    parser.add_argument("--remote-identity", default=None)
    parser.add_argument("--remote-known-hosts", default=None)
    parser.add_argument("--expected-fingerprint", default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/p1/v3_operator.py",
        description="P1 v3 offline-first operator; live is explicitly gated.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="offline plan; no network or mutation")
    plan.add_argument("--workspace", default="/workspace")
    plan.add_argument("--json", action="store_true")
    readiness_p = sub.add_parser(
        "readiness", help="source-only readiness; no session/SSH/env/mutation"
    )
    readiness_p.add_argument("--json", action="store_true")
    risk_pending_p = sub.add_parser(
        "owner-risk-pending",
        help="read pending quote and write a non-authorizing forecast",
    )
    risk_pending_p.add_argument("--pending-quote", required=True)
    risk_pending_p.add_argument("--output", required=True)
    risk_pending_p.add_argument("--json", action="store_true")
    risk_admit_p = sub.add_parser(
        "owner-risk-admit",
        help="create explicit scoped owner financial-risk admission",
    )
    risk_admit_p.add_argument("--quote", required=True)
    risk_admit_p.add_argument("--risk-doc", default=str(owner_risk.RISK_DOC))
    risk_admit_p.add_argument(
        "--session-approval-doc", default=str(owner_risk.SESSION_APPROVAL_DOC)
    )
    risk_admit_p.add_argument("--output", required=True)
    risk_admit_p.add_argument(
        "--server-id", type=int, default=owner_risk.ALLOWED_SERVER_ID
    )
    risk_admit_p.add_argument(
        "--hoster-id", type=int, default=owner_risk.ALLOWED_HOSTER_ID
    )
    risk_admit_p.add_argument("--valid-hours", type=int, default=2)
    risk_admit_p.add_argument("--observation-origin", default=None)
    risk_admit_p.add_argument("--expected-quote-sha256", default=None)
    risk_admit_p.add_argument("--json", action="store_true")
    fresh_p = sub.add_parser(
        "fresh-quote",
        help="explicit GET-only fresh marketplace quote; no order/claim/Pay",
    )
    fresh_p.add_argument("--server-id", type=int, required=True)
    fresh_p.add_argument("--currency", default=owner_risk.ALLOWED_CURRENCY)
    fresh_p.add_argument("--output", required=True)
    fresh_p.add_argument("--live-read", action="store_true")
    fresh_p.add_argument("--api-key-env", default="CLORE_API_KEY")
    fresh_p.add_argument("--timeout-seconds", type=float, default=10.0)
    fresh_p.add_argument("--max-response-bytes", type=int, default=1_000_000)
    fresh_p.add_argument("--json", action="store_true")
    risk_approve_p = sub.add_parser(
        "owner-risk-approve",
        help="explicitly approve one data-driven scoped proposal admission",
    )
    risk_approve_p.add_argument("--quote", required=True)
    risk_approve_p.add_argument("--output", required=True)
    risk_approve_p.add_argument("--server-id", type=int, required=True)
    risk_approve_p.add_argument("--hoster-id", type=int, default=None)
    risk_approve_p.add_argument("--currency", default=owner_risk.ALLOWED_CURRENCY)
    risk_approve_p.add_argument("--risk-doc", default=str(owner_risk.RISK_DOC))
    risk_approve_p.add_argument(
        "--session-approval-doc", default=str(owner_risk.SESSION_APPROVAL_DOC)
    )
    risk_approve_p.add_argument("--valid-hours", type=int, default=2)
    risk_approve_p.add_argument("--approved-by", default="owner")
    risk_approve_p.add_argument("--confirm-approval", required=True)
    risk_approve_p.add_argument("--expected-quote-sha256", default=None)
    risk_approve_p.add_argument("--json", action="store_true")
    approve_p = sub.add_parser(
        "approve-source", help="create explicit scoped owner/source approval"
    )
    approve_p.add_argument("--approval-doc", default=str(v3_readiness.CURRENT_APPROVAL_DOC))
    approve_p.add_argument("--approval-out", required=True)
    approve_p.add_argument("--approved-by", default="owner")
    approve_p.add_argument("--cap-usd", default="1.00")
    approve_p.add_argument("--duration-seconds", type=int, default=3600)
    approve_p.add_argument("--valid-hours", type=int, default=v3_readiness.DEFAULT_APPROVAL_VALID_HOURS)
    approve_p.add_argument("--json", action="store_true")
    bind_p = sub.add_parser(
        "session-bind", help="bind actual order and machine quote; no network"
    )
    bind_p.add_argument("--session-state", required=True)
    bind_p.add_argument("--quote", required=True)
    bind_p.add_argument("--order-id", type=int, required=True)
    bind_p.add_argument("--created-at", type=int, required=True)
    bind_p.add_argument("--cap-usd", default="1.00")
    bind_p.add_argument("--duration-seconds", type=int, default=3600)
    bind_p.add_argument("--planned-cancel-at", type=int, default=None)
    bind_p.add_argument("--hard-end-at", type=int, default=None)
    bind_p.add_argument("--owner-risk-admission", default=None)
    bind_p.add_argument("--server-id", type=int, default=None)
    bind_p.add_argument("--json", action="store_true")
    prepare_p = sub.add_parser("prepare", help="real offline preparation and runbook")
    prepare_p.add_argument("--workspace", default="/workspace")
    prepare_p.add_argument("--run-dir", default=None)
    prepare_p.add_argument("--execute-downloads", action="store_true")
    prepare_p.add_argument("--session-state", default=None)
    prepare_p.add_argument("--quote-observation", default=None)
    prepare_p.add_argument("--source-approval", default=None)
    prepare_p.add_argument("--order-id", type=int, default=None)
    prepare_p.add_argument("--created-at", type=int, default=None)
    prepare_p.add_argument("--live", action="store_true")
    prepare_p.add_argument("--resume-dead", action="store_true")
    _add_remote_arguments(prepare_p)
    prepare_p.add_argument("--json", action="store_true")
    status_p = sub.add_parser("status", help="read durable state; remote only when explicit")
    status_p.add_argument("--run-dir", default=None)
    status_p.add_argument("--session-state", default=None)
    status_p.add_argument("--remote", action="store_true")
    status_p.add_argument("--remote-workspace", default="/workspace")
    _add_remote_arguments(status_p)
    status_p.add_argument("--json", action="store_true")
    run_p = sub.add_parser("run-once", help="plan or explicitly gated live run")
    run_p.add_argument("--live", action="store_true")
    run_p.add_argument("--session-state", default=None)
    run_p.add_argument("--order-id", type=int, default=None)
    run_p.add_argument("--created-at", type=int, default=None)
    run_p.add_argument("--run-dir", default=None)
    _add_remote_arguments(run_p)
    run_p.add_argument("--json", action="store_true")
    collect_p = sub.add_parser("collect", help="critical-first local export and decode gate")
    collect_p.add_argument("--run-dir", default=None)
    collect_p.add_argument("--artifact", default=None)
    collect_p.add_argument("--remote-artifact-path", default=None)
    collect_p.add_argument("--remote-helper-path", default=None)
    collect_p.add_argument("--remote-expected-size", type=int, default=None)
    collect_p.add_argument("--remote-expected-sha256", default=None)
    collect_p.add_argument("--remote-offset", type=int, default=0)
    collect_p.add_argument("--remote-prefix-sha256", default=None)
    collect_p.add_argument("--remote-root", default="/workspace")
    collect_p.add_argument("--remote-evidence", action="append", default=[])
    _add_remote_arguments(collect_p)
    collect_p.add_argument("--timeout-seconds", type=float, default=600.0)
    collect_p.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            payload = build_plan(workspace=args.workspace)
            print(_json_text(payload), end="")
            return 0
        if args.command == "readiness":
            payload = source_readiness()
            print(_json_text(payload), end="")
            return 0
        if args.command == "owner-risk-pending":
            payload = owner_risk_pending(
                pending_quote=Path(args.pending_quote),
                output=Path(args.output),
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "owner-risk-admit":
            payload = owner_risk_admit(
                quote=Path(args.quote),
                output=Path(args.output),
                approval_doc=Path(args.risk_doc),
                session_approval_doc=Path(args.session_approval_doc),
                server_id=args.server_id,
                hoster_id=args.hoster_id,
                valid_hours=args.valid_hours,
                observation_origin=args.observation_origin,
                expected_quote_sha256=args.expected_quote_sha256,
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "fresh-quote":
            payload = fresh_quote(
                server_id=args.server_id,
                output=Path(args.output),
                currency=args.currency,
                live_read=args.live_read,
                api_key_env=args.api_key_env,
                timeout_seconds=args.timeout_seconds,
                max_response_bytes=args.max_response_bytes,
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "owner-risk-approve":
            payload = owner_risk_approve_proposal(
                quote=Path(args.quote),
                output=Path(args.output),
                server_id=args.server_id,
                hoster_id=args.hoster_id,
                currency=args.currency,
                approval_doc=Path(args.risk_doc),
                session_approval_doc=Path(args.session_approval_doc),
                valid_hours=args.valid_hours,
                approved_by=args.approved_by,
                confirm_approval=args.confirm_approval,
                expected_quote_sha256=args.expected_quote_sha256,
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "approve-source":
            payload = approve_source(
                approval_out=Path(args.approval_out),
                approval_doc=Path(args.approval_doc),
                approved_by=args.approved_by,
                cap_usd=args.cap_usd,
                duration_seconds=args.duration_seconds,
                valid_hours=args.valid_hours,
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "session-bind":
            payload = bind_session(
                session_state=Path(args.session_state),
                quote=Path(args.quote),
                order_id=args.order_id,
                created_at=args.created_at,
                cap_usd=args.cap_usd,
                duration_seconds=args.duration_seconds,
                planned_cancel_at=args.planned_cancel_at,
                hard_end_at=args.hard_end_at,
                owner_risk_admission=(
                    Path(args.owner_risk_admission)
                    if args.owner_risk_admission
                    else None
                ),
                server_id=args.server_id,
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "prepare":
            payload = prepare(
                workspace=args.workspace,
                run_dir=Path(args.run_dir) if args.run_dir else None,
                execute_downloads=args.execute_downloads,
                session_state=Path(args.session_state) if args.session_state else None,
                order_id=args.order_id,
                created_at=args.created_at,
                live=args.live,
                source_approval=Path(args.source_approval) if args.source_approval else None,
                quote_observation=Path(args.quote_observation) if args.quote_observation else None,
                remote=_remote_identity(args),
                expected_fingerprint=args.expected_fingerprint,
                resume_dead=args.resume_dead,
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "status":
            if getattr(args, "remote", False) and not args.remote_host:
                raise OperatorError(
                    "--remote requires --remote-host, --remote-identity and --remote-known-hosts"
                )
            payload = status(
                run_dir=Path(args.run_dir) if args.run_dir else None,
                session_state=Path(args.session_state) if args.session_state else None,
                remote=_remote_identity(args),
                expected_fingerprint=args.expected_fingerprint,
                remote_workspace=args.remote_workspace,
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "run-once":
            payload = run_once(
                live=args.live,
                session_state=Path(args.session_state) if args.session_state else None,
                order_id=args.order_id,
                created_at=args.created_at,
                remote=_remote_identity(args),
                expected_fingerprint=args.expected_fingerprint,
                run_dir=Path(args.run_dir) if args.run_dir else None,
            )
            print(_json_text(payload), end="")
            return 0
        if args.command == "collect":
            payload = collect(
                run_dir=Path(args.run_dir) if args.run_dir else None,
                artifact=Path(args.artifact) if args.artifact else None,
                remote=_remote_identity(args),
                remote_artifact_path=args.remote_artifact_path,
                remote_helper_path=args.remote_helper_path,
                remote_expected_size=args.remote_expected_size,
                remote_expected_sha256=args.remote_expected_sha256,
                remote_offset=args.remote_offset,
                remote_prefix_sha256=args.remote_prefix_sha256,
                expected_fingerprint=args.expected_fingerprint,
                remote_evidence=args.remote_evidence,
                timeout_seconds=args.timeout_seconds,
                remote_root=args.remote_root,
            )
            print(_json_text(payload), end="")
            return 0
    except ReadinessBlocked as exc:
        print(_json_text({"status": "READINESS_GATE_BLOCKED", "error": str(exc)}), end="")
        return 2
    except (
        OperatorError,
        v3_lifecycle.LifecycleError,
        v3_readiness.ReadinessError,
        owner_risk.OwnerRiskError,
        RemoteError,
        clore_export.ExportError,
        setup_plan.SetupPlanError,
        OSError,
        ValueError,
        sqlite3.Error,
    ) as exc:
        print(
            _json_text(
                {
                    "status": "REFUSED",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            ),
            end="",
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
