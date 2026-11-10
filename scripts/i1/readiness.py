"""Strict READY evaluation for I1."""

from __future__ import annotations

from typing import Any, Mapping

from .profile import Profile
from .util import format_bytes

READY = "READY"
INSTALLED = "INSTALLED"
FAILED = "FAILED"
OFFLINE_VERIFIED = "OFFLINE_VERIFIED"


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    message: str,
    *,
    required: bool = True,
    facts: Mapping[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "required": bool(required),
            "message": message,
            "facts": dict(facts or {}),
        }
    )


def evaluate(
    *,
    probe: Mapping[str, Any] | None,
    profile: Profile,
    tunnel_health: Mapping[str, Any] | None,
    ui: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if profile.data.get("schema") == "anime-video-studio/i1-profile/v2":
        return _evaluate_v2(
            probe=probe, profile=profile, tunnel_health=tunnel_health, ui=ui
        )
    return _evaluate_v1(
        probe=probe, profile=profile, tunnel_health=tunnel_health, ui=ui
    )


def _evaluate_v1(
    *,
    probe: Mapping[str, Any] | None,
    profile: Profile,
    tunnel_health: Mapping[str, Any] | None,
    ui: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if not isinstance(probe, Mapping):
        _check(checks, "probe", False, "actual probe evidence is missing")
        return {
            "status": INSTALLED,
            "ready": False,
            "checks": checks,
            "blockers": ["actual probe evidence is missing"],
            "warnings": [],
        }
    _check(
        checks,
        "probe_schema",
        probe.get("schema") == "anime-video-studio/i1-runtime-evidence/v1",
        "runtime evidence schema",
    )
    _check(
        checks,
        "profile_hash",
        probe.get("profile_sha256") == profile.sha256,
        "profile hash matches the pinned profile",
    )
    _check(
        checks,
        "workflow_hash",
        probe.get("workflow_sha256") == profile.workflow["api_sha256"],
        "API workflow hash matches the pin",
    )
    torch = probe.get("torch") if isinstance(probe.get("torch"), Mapping) else {}
    _check(
        checks,
        "torch_cuda",
        bool(torch.get("available")) and bool(torch.get("version")) and bool(torch.get("cuda")),
        "torch and CUDA are actually available",
        facts=torch,
    )
    comfy = probe.get("comfyui") if isinstance(probe.get("comfyui"), Mapping) else {}
    _check(
        checks,
        "comfy_commit",
        comfy.get("commit") == profile.comfy["commit"] and comfy.get("import_ok") is True,
        "ComfyUI commit/imports actually match",
        facts=comfy,
    )
    pip = probe.get("pip") if isinstance(probe.get("pip"), Mapping) else {}
    _check(checks, "pip_check", pip.get("check_exit") == 0, "pip check is clean", facts=pip)
    models = probe.get("models")
    expected = {item["target"]: item for item in profile.models}
    actual = {}
    if isinstance(models, list):
        for item in models:
            if isinstance(item, Mapping) and isinstance(item.get("target"), str):
                actual[item["target"]] = item
    models_ok = len(actual) == 4 and set(actual) == set(expected)
    if models_ok:
        for target, item in expected.items():
            got = actual[target]
            if (
                got.get("actual_size_bytes") != item["size_bytes"]
                or str(got.get("actual_sha256", "")).lower() != item["sha256"].lower()
            ):
                models_ok = False
                break
    _check(
        checks,
        "models",
        models_ok,
        "all four model files match size and SHA256",
        facts={"files": len(actual), "bytes": sum(int(item["size_bytes"]) for item in profile.models)},
    )
    object_info = probe.get("object_info") if isinstance(probe.get("object_info"), Mapping) else {}
    input_checks = object_info.get("input_checks")
    _check(
        checks,
        "object_info",
        bool(object_info.get("classes"))
        and isinstance(input_checks, list)
        and bool(input_checks)
        and all(item.get("present") is True for item in input_checks),
        "object_info contains all workflow classes and required input types",
        facts={"classes": object_info.get("classes"), "inputs": len(input_checks or [])},
    )
    link_checks = object_info.get("link_checks")
    _check(
        checks,
        "workflow_links",
        isinstance(link_checks, list)
        and all(item.get("compatible") is True for item in link_checks),
        "workflow linked inputs/slot types are compatible",
        facts={"links": len(link_checks or [])},
    )
    savevideo = object_info.get("savevideo_dynamic") if isinstance(object_info, Mapping) else {}
    _check(
        checks,
        "savevideo",
        isinstance(savevideo, Mapping)
        and not savevideo.get("issues")
        and savevideo.get("selected") == {"format": "mp4", "codec": "h264", "encoding": "auto"},
        "SaveVideo nested mp4/h264/auto schema",
        facts=savevideo if isinstance(savevideo, Mapping) else {},
    )
    system_stats = probe.get("system_stats") if isinstance(probe.get("system_stats"), Mapping) else {}
    devices = system_stats.get("devices")
    _check(
        checks,
        "system_stats",
        isinstance(devices, list) and bool(devices),
        "ComfyUI /system_stats reports a CUDA device",
        facts={"devices": len(devices or [])},
    )
    workflow = probe.get("workflow") if isinstance(probe.get("workflow"), Mapping) else {}
    _check(
        checks,
        "workflow_schema",
        workflow.get("sha256") == profile.workflow["api_sha256"]
        and sorted(workflow.get("classes") or []) == sorted(profile.required_classes),
        "API workflow schema/classes match",
        facts={"classes": workflow.get("classes")},
    )
    _check(
        checks,
        "tunnel_health",
        isinstance(tunnel_health, Mapping) and bool(tunnel_health.get("system")),
        "ComfyUI is reachable through the owned loopback tunnel",
        facts={"reachable": isinstance(tunnel_health, Mapping)},
    )
    if tunnel_health is None:
        blockers.append("owned tunnel/ComfyUI health is unavailable")
    for item in checks:
        if item["required"] and not item["passed"]:
            blockers.append(item["message"])
    ui = ui if isinstance(ui, Mapping) else {}
    if ui.get("ui_status") == "UI_FILE_ONLY":
        warnings.append("UI file is registered but not listed by the frontend")
    elif ui.get("ui_status") in {"UI_GRAPH_NOT_VERIFIED", "DERIVED_FROM_API_NOT_LIVE_VERIFIED"}:
        warnings.append("UI graph auto-render is NOT_VERIFIED; exact API/UI file fallback is required")
    elif ui.get("ui_status") == "API_FILE_ONLY":
        warnings.append("API JSON is not a UI graph; no UI workflow was registered")
    if not ui.get("ui_graph_rendered"):
        warnings.append("UI graph open/render was not verified")
    ready = not blockers
    return {
        "status": READY if ready else INSTALLED,
        "ready": ready,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "ui_status": ui.get("ui_status"),
        "ui_graph_rendered": False,
        "model_bytes_verified": sum(int(item["size_bytes"]) for item in profile.models) if models_ok else 0,
    }


def _evaluate_v2(
    *,
    probe: Mapping[str, Any] | None,
    profile: Profile,
    tunnel_health: Mapping[str, Any] | None,
    ui: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if not isinstance(probe, Mapping):
        _check(checks, "probe", False, "actual probe evidence is missing")
        return {
            "status": INSTALLED,
            "ready": False,
            "checks": checks,
            "blockers": ["actual probe evidence is missing"],
            "warnings": [],
        }
    _check(
        checks,
        "probe_schema",
        probe.get("schema") == "anime-video-studio/i1-runtime-evidence/v1",
        "runtime evidence schema",
    )
    _check(
        checks,
        "profile_hash",
        probe.get("profile_sha256") == profile.sha256,
        "profile hash matches the pinned profile",
    )
    torch = probe.get("torch") if isinstance(probe.get("torch"), Mapping) else {}
    _check(
        checks,
        "torch_cuda",
        bool(torch.get("available")) and bool(torch.get("version")) and bool(torch.get("cuda")),
        "torch and CUDA are actually available",
        facts=torch,
    )
    comfy = probe.get("comfyui") if isinstance(probe.get("comfyui"), Mapping) else {}
    _check(
        checks,
        "comfy_commit",
        comfy.get("commit") == profile.comfy["commit"] and comfy.get("import_ok") is True,
        "ComfyUI commit/imports actually match",
        facts=comfy,
    )
    pip = probe.get("pip") if isinstance(probe.get("pip"), Mapping) else {}
    _check(checks, "pip_check", pip.get("check_exit") == 0, "pip check is clean", facts=pip)
    expected = {item["target"]: item for item in profile.models}
    actual: dict[str, Mapping[str, Any]] = {}
    raw_models = probe.get("models")
    if isinstance(raw_models, list):
        for item in raw_models:
            if isinstance(item, Mapping) and isinstance(item.get("target"), str):
                actual[item["target"]] = item
    models_ok = len(actual) == len(expected) and set(actual) == set(expected)
    if models_ok:
        for target, item in expected.items():
            got = actual[target]
            if (
                got.get("actual_size_bytes") != item["size_bytes"]
                or str(got.get("actual_sha256", "")).lower() != item["sha256"].lower()
            ):
                models_ok = False
                break
    _check(
        checks,
        "models",
        models_ok,
        f"all {len(expected)} model files match size and SHA256",
        facts={"files": len(actual), "bytes": sum(int(item["size_bytes"]) for item in profile.models)},
    )
    workflows = probe.get("workflows")
    workflow_evidence: dict[str, Mapping[str, Any]] = {}
    if isinstance(workflows, list):
        for item in workflows:
            if isinstance(item, Mapping) and isinstance(item.get("workflow_id"), str):
                workflow_evidence[item["workflow_id"]] = item
    all_classes_ok = set(workflow_evidence) == set(profile.workflows)
    input_ok = True
    link_ok = True
    output_ok = True
    for workflow_id, meta in profile.workflows.items():
        evidence = workflow_evidence.get(workflow_id)
        if not isinstance(evidence, Mapping):
            all_classes_ok = False
            continue
        if sorted(evidence.get("classes") or []) != sorted(meta["required_classes"]):
            all_classes_ok = False
        input_checks = evidence.get("input_checks")
        if not isinstance(input_checks, list) or not input_checks:
            input_ok = False
        elif not all(item.get("present") is True for item in input_checks):
            input_ok = False
        link_checks = evidence.get("link_checks")
        if not isinstance(link_checks, list):
            link_ok = False
        elif not all(item.get("compatible") is True for item in link_checks):
            link_ok = False
        output_checks = evidence.get("output_checks")
        if not isinstance(output_checks, list) or not output_checks:
            output_ok = False
        elif not all(item.get("present") is True for item in output_checks):
            output_ok = False
    _check(
        checks,
        "object_info",
        all_classes_ok and input_ok,
        "object_info contains all workflow classes and required input types",
        facts={"workflows": sorted(workflow_evidence)},
    )
    _check(
        checks,
        "workflow_links",
        link_ok,
        "workflow linked inputs/slot types are compatible",
        facts={"workflows": sorted(workflow_evidence)},
    )
    _check(
        checks,
        "output_schema",
        output_ok,
        "SaveImage/SaveVideo output schema matches the profile",
        facts={"workflows": sorted(workflow_evidence)},
    )
    system_stats = probe.get("system_stats") if isinstance(probe.get("system_stats"), Mapping) else {}
    devices = system_stats.get("devices")
    _check(
        checks,
        "system_stats",
        isinstance(devices, list) and bool(devices),
        "ComfyUI /system_stats reports a CUDA device",
        facts={"devices": len(devices or [])},
    )
    workflow_hash_ok = True
    for workflow_id, meta in profile.workflows.items():
        evidence = workflow_evidence.get(workflow_id)
        if not isinstance(evidence, Mapping):
            workflow_hash_ok = False
            continue
        if evidence.get("sha256") != meta["api_sha256"]:
            workflow_hash_ok = False
        if sorted(evidence.get("classes") or []) != sorted(meta["required_classes"]):
            workflow_hash_ok = False
    _check(
        checks,
        "workflow_schema",
        workflow_hash_ok,
        "API workflow schema/classes match",
        facts={"workflows": sorted(workflow_evidence)},
    )
    _check(
        checks,
        "tunnel_health",
        isinstance(tunnel_health, Mapping) and bool(tunnel_health.get("system")),
        "ComfyUI is reachable through the owned loopback tunnel",
        facts={"reachable": isinstance(tunnel_health, Mapping)},
    )
    if tunnel_health is None:
        blockers.append("owned tunnel/ComfyUI health is unavailable")
    for item in checks:
        if item["required"] and not item["passed"]:
            blockers.append(item["message"])
    ui = ui if isinstance(ui, Mapping) else {}
    if ui.get("ui_status") == "API_FILE_ONLY":
        warnings.append("API JSON is not a UI graph; no UI workflow was registered")
    elif ui.get("ui_status") in {"UI_FILE_ONLY", "UI_GRAPH_NOT_VERIFIED"}:
        warnings.append("UI graph auto-render is NOT_VERIFIED; exact API/UI file fallback is required")
    if not ui.get("ui_graph_rendered"):
        warnings.append("UI graph open/render was not verified")
    ready = not blockers
    return {
        "status": READY if ready else INSTALLED,
        "ready": ready,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "ui_status": ui.get("ui_status"),
        "ui_graph_rendered": False,
        "model_bytes_verified": (
            sum(int(item["size_bytes"]) for item in profile.models) if models_ok else 0
        ),
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "workflow_hashes": {
            key: meta["api_sha256"] for key, meta in profile.workflows.items()
        },
    }


def offline_report(*, note: str = "local fixture harness") -> dict[str, Any]:
    """A fake/local harness result must never be presented as remote READY."""

    return {
        "status": OFFLINE_VERIFIED,
        "ready": False,
        "checks": [],
        "blockers": [],
        "warnings": [note, "OFFLINE_VERIFIED is not a real GPU READY claim"],
        "ui_graph_rendered": False,
    }


def human_summary(report: Mapping[str, Any]) -> str:
    lines = [f"Статус: {report.get('status', FAILED)}"]
    for item in report.get("checks", []):
        mark = "OK" if item.get("passed") else "FAIL"
        lines.append(f"  [{mark}] {item.get('message')}")
    for warning in report.get("warnings", []):
        lines.append(f"  [!] {warning}")
    for blocker in report.get("blockers", []):
        lines.append(f"  [BLOCK] {blocker}")
    if report.get("status") == READY:
        lines.append("READY: actual CUDA/import/weights/object_info/workflow/tunnel checks passed.")
        lines.append("UI graph auto-open remains NOT_VERIFIED unless explicitly listed.")
    return "\n".join(lines)


__all__ = [
    "FAILED",
    "INSTALLED",
    "OFFLINE_VERIFIED",
    "READY",
    "evaluate",
    "human_summary",
    "offline_report",
]

