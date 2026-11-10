"""Resolve a loopback ComfyUI URL and prove a fresh I1 runtime for this run.

A persisted ``readiness_current`` boolean is only a fast precheck.  Before an
explicit ``generate``/``retry`` connection run, KF1 reuses I1's existing
``ops.resume`` full probe over the trusted connection: real model hashes,
schema/object_info, workflow hashes and owned tunnel evidence in this run's
epoch.  Manual ``--base-url`` is explicitly a test/manual bypass of that I1
proof, but the per-bound-job actual object_info gate still applies.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .comfy import validate_base_url
from .errors import KF1Error

KF1_PROFILES = {"keyframes-qwen-edit-2511", "bundle-wan22-qwen-edit-2511"}


def _required_checks(report: Mapping[str, Any]) -> None:
    required = {
        "probe_schema",
        "profile_hash",
        "torch_cuda",
        "comfy_commit",
        "pip_check",
        "models",
        "object_info",
        "workflow_links",
        "output_schema",
        "system_stats",
        "workflow_schema",
        "tunnel_health",
    }
    checks = report.get("checks") or []
    passed = {
        str(item.get("name"))
        for item in checks
        if isinstance(item, Mapping) and item.get("passed") is True
    }
    missing = sorted(required - passed)
    if missing:
        raise KF1Error("readiness не содержит fresh passed checks: " + ", ".join(missing))


def validate_kf1_context(ctx) -> None:
    """Fast cached precheck; NEVER sufficient to authorize a job by itself."""

    profile = ctx.profile
    if profile.profile_id not in KF1_PROFILES:
        raise KF1Error(
            "KF1 execute требует opt-in keyframes/bundle профиль, не Wan-only; "
            "выберите config/i1/keyframes-qwen-edit-2511.profile.json или bundle"
        )
    state = ctx.state
    if state.get("phase") != "READY":
        raise KF1Error(
            f"I1 connection phase={state.get('phase')}; нужен свежий READY, не INSTALLED"
        )
    if state.get("readiness_current") is not True:
        raise KF1Error("cached readiness не является current; нужен свежий I1 probe")
    report = state.get("readiness") or {}
    if not isinstance(report, dict) or report.get("ready") is not True:
        raise KF1Error("KF1 readiness не READY; upload/POST запрещены")
    if report.get("profile_sha256") != profile.sha256:
        raise KF1Error("readiness profile hash не совпадает с выбранным профилем")
    if int(report.get("model_bytes_verified", 0)) != profile.model_bytes():
        raise KF1Error("readiness model bytes не совпадают; model tamper/stale evidence")
    for workflow_id, meta in profile.workflows.items():
        if report.get("workflow_hashes", {}).get(workflow_id) != meta["api_sha256"]:
            raise KF1Error(f"readiness workflow hash не совпадает: {workflow_id}")
    _required_checks(report)


def _validate_probe(probe: Mapping[str, Any], profile: Any, ctx) -> None:
    if not isinstance(probe, Mapping):
        raise KF1Error("fresh I1 probe не вернул actual evidence")
    if probe.get("schema") != "anime-video-studio/i1-runtime-evidence/v1":
        raise KF1Error("fresh I1 probe schema некорректна")
    state = ctx.state
    if probe.get("run_id") != state.get("run_id") or probe.get("nonce") != state.get("nonce"):
        raise KF1Error("fresh I1 probe принадлежит другому run/nonce")
    if probe.get("profile_sha256") != profile.sha256:
        raise KF1Error("fresh I1 probe profile hash не совпадает")
    expected_models = {
        str(item["target"]): (int(item["size_bytes"]), str(item["sha256"]).lower())
        for item in profile.models
    }
    actual_models = probe.get("models")
    if not isinstance(actual_models, list) or len(actual_models) != len(expected_models):
        raise KF1Error("fresh I1 probe model count не совпадает")
    seen = set()
    for item in actual_models:
        if not isinstance(item, Mapping):
            raise KF1Error("fresh I1 probe model entry некорректен")
        target = str(item.get("target"))
        if target not in expected_models:
            raise KF1Error(f"fresh I1 probe неизвестная model target: {target}")
        size, sha = expected_models[target]
        if (
            int(item.get("actual_size_bytes", -1)) != size
            or str(item.get("actual_sha256", "")).lower() != sha
        ):
            raise KF1Error(f"fresh I1 probe model bytes tampered: {target}")
        seen.add(target)
    if seen != set(expected_models):
        raise KF1Error("fresh I1 probe не покрывает все модели")
    workflows = probe.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        raise KF1Error("fresh I1 probe workflow evidence отсутствует")
    by_id = {
        str(item.get("workflow_id")): item
        for item in workflows
        if isinstance(item, Mapping)
    }
    for workflow_id, meta in profile.workflows.items():
        item = by_id.get(workflow_id)
        if item is None:
            raise KF1Error(f"fresh I1 probe без workflow {workflow_id}")
        if item.get("sha256") != meta["api_sha256"]:
            raise KF1Error(f"fresh I1 probe workflow hash: {workflow_id}")
        if set(item.get("classes") or []) != set(meta["required_classes"]):
            raise KF1Error(f"fresh I1 probe workflow classes: {workflow_id}")
        if any(
            check.get("present") is not True for check in item.get("input_checks") or []
        ):
            raise KF1Error(f"fresh I1 probe inputs missing: {workflow_id}")
        if any(
            check.get("compatible") is not True
            for check in item.get("link_checks") or []
        ):
            raise KF1Error(f"fresh I1 probe links incompatible: {workflow_id}")
        if any(
            check.get("present") is not True
            for check in item.get("output_checks") or []
        ):
            raise KF1Error(f"fresh I1 probe output schema missing: {workflow_id}")
    object_info = probe.get("object_info")
    if not isinstance(object_info, Mapping):
        raise KF1Error("fresh I1 probe object_info evidence отсутствует")
    if not isinstance(object_info.get("sha256"), str) or len(object_info["sha256"]) != 64:
        raise KF1Error("fresh I1 probe object_info hash отсутствует")


def verify_kf1_runtime(ctx, *, resume_fn=None) -> dict[str, Any]:
    """Run the existing I1 full probe and return a fresh current-run attestation."""

    validate_kf1_context(ctx)
    from scripts.i1 import ops

    probe_fn = resume_fn or ops.resume
    try:
        result = probe_fn(ctx)
    except Exception as exc:
        raise KF1Error(
            f"actual I1 verification не прошла: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(result, Mapping):
        raise KF1Error("actual I1 verification не вернула результат")
    state = result.get("state") or ctx.state
    try:
        ctx.state = state
    except Exception:
        pass
    probe = result.get("probe")
    report = result.get("readiness") or state.get("readiness") or {}
    # Re-validate the freshly updated state, not the stale persisted boolean.
    validate_kf1_context(ctx)
    if not isinstance(report, Mapping) or report.get("ready") is not True:
        raise KF1Error("actual I1 verification не подтвердила fresh READY")
    _required_checks(report)
    _validate_probe(probe, ctx.profile, ctx)
    profile = ctx.profile
    model_hashes = {}
    for item in ctx.profile.models:
        model_hashes[str(item["target"])] = str(item["sha256"]).lower()
    workflow_hashes = {
        workflow_id: meta["api_sha256"]
        for workflow_id, meta in profile.workflows.items()
    }
    attestation = {
        "schema": "anime-studio/kf1-runtime-attestation/v1",
        "mode": "i1_full_probe",
        "i1_verified": True,
        "verified_epoch": time.time(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "probe_observed_at": probe.get("observed_at"),
        "run_id": state.get("run_id"),
        "nonce": state.get("nonce"),
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "installation_target": {
            "workspace": (state.get("identity") or {}).get("workspace"),
            "comfy_commit": profile.comfy.get("commit"),
            "profile_path": str(ctx.profile_path),
        },
        "model_bytes_verified": sum(
            int(item["size_bytes"]) for item in profile.models
        ),
        "model_hashes": model_hashes,
        "workflow_hashes": workflow_hashes,
        "object_info_sha256": (probe.get("object_info") or {}).get("sha256"),
        "probe_sha256": hashlib.sha256(
            json.dumps(probe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "checks": [
            str(item.get("name"))
            for item in (report.get("checks") or [])
            if isinstance(item, Mapping) and item.get("passed") is True
        ],
    }
    return attestation


@dataclass(frozen=True)
class KF1Connection:
    base_url: str
    attestation: dict[str, Any]


def open_kf1_connection(
    *,
    base_url: str | None,
    connection: str | None,
    state_dir: Path | str,
    profile_path: Path | str,
) -> KF1Connection:
    if base_url:
        # Manual/test mode: loopback only, explicitly not an I1 READY claim.
        return KF1Connection(
            base_url=validate_base_url(base_url),
            attestation={
                "schema": "anime-studio/kf1-runtime-attestation/v1",
                "mode": "manual_base_url",
                "i1_verified": False,
                "verified_epoch": time.time(),
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "note": "manual --base-url test bypass; actual bound object_info still enforced",
            },
        )
    if not connection:
        raise KF1Error(
            "Для --execute нужен --connection NAME (I1 trusted tunnel) "
            "или manual/test --base-url loopback"
        )
    from scripts.i1 import ops, tunnel

    ctx = ops.load_context(
        name=connection,
        state_dir=Path(state_dir),
        profile_path=Path(profile_path),
        verify_key=True,
    )
    validate_kf1_context(ctx)
    attestation = verify_kf1_runtime(ctx)
    runtime = ctx.profile.data["runtime"]
    local_port = int(runtime["local_tunnel_port"])
    remote_port = int(runtime["comfy_remote_port"])
    tunnel.ensure(
        identity=ctx.identity,
        run_dir=ctx.run_dir,
        run_id=ctx.state["run_id"],
        nonce=ctx.state["nonce"],
        fingerprint=ctx.state["identity"]["fingerprint"],
        local_port=local_port,
        remote_port=remote_port,
    )
    ops._wait_health(local_port, timeout=90.0)
    return KF1Connection(
        base_url=validate_base_url(f"http://127.0.0.1:{local_port}"),
        attestation=attestation,
    )


def resolve_base_url(
    *,
    base_url: str | None,
    connection: str | None,
    state_dir: Path | str,
    profile_path: Path | str,
) -> str:
    return open_kf1_connection(
        base_url=base_url,
        connection=connection,
        state_dir=state_dir,
        profile_path=profile_path,
    ).base_url


__all__ = [
    "KF1Connection",
    "KF1_PROFILES",
    "open_kf1_connection",
    "resolve_base_url",
    "validate_kf1_context",
    "verify_kf1_runtime",
]
