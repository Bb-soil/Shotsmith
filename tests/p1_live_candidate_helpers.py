"""Test-only builders for an explicit live candidate from synthetic facts.

These helpers do not bypass the production gates: they construct the same
candidate/evidence artifacts the operator builds, using actual immutable source
pins and a small deterministic fact set. Tests still exercise the full
candidate/readiness validation path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.p1 import v3_lifecycle, v3_operator, v3_readiness

ROOT = Path(__file__).resolve().parents[1]
OBJECT_INFO_FIXTURE = ROOT / "tests" / "fixtures" / "p1" / "savevideo-object_info-7a283320.json"


def fake_evidence(
    *,
    bundle: dict[str, Any],
    binding: v3_lifecycle.SessionBinding,
    lifecycle: v3_lifecycle.LifecycleState,
    approval: v3_readiness.SourceApproval,
    host_key_fact: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    models = []
    for model in bundle["manifest"]["models"]:
        models.append(
            {
                "id": model["id"],
                "target": model["target"],
                "size_bytes": model["size_bytes"],
                "sha256": model["sha256"],
                "actual_size_bytes": model["size_bytes"],
                "actual_sha256": model["sha256"],
            }
        )
    node_types = bundle["config"].binding_raw["node_types"]
    classes = sorted({str(value) for value in node_types.values()})
    fixture = json.loads(OBJECT_INFO_FIXTURE.read_text(encoding="utf-8"))
    descriptors = {name: fixture[name] for name in classes}
    for class_name, input_names in {
        class_name: {
            str(input_name)
            for node_id, bound_class in node_types.items()
            if bound_class == class_name
            for input_name in bundle["workflow"][str(node_id)]["inputs"]
            if not (
                class_name == "SaveVideo"
                and (
                    str(input_name) == "format"
                    or str(input_name).startswith("format.")
                )
            )
        }
        for class_name in classes
    }.items():
        descriptor = descriptors[class_name]
        inputs = descriptor.setdefault("input", {})
        required = inputs.setdefault("required", {})
        for input_name in input_names:
            required.setdefault(input_name, ["STRING", {}])
    graph_inputs: list[dict[str, Any]] = []
    for node_id, class_name in node_types.items():
        inputs = bundle["workflow"][str(node_id)]["inputs"]
        for input_name in inputs:
            text = str(input_name)
            if class_name == "SaveVideo" and (
                text == "format" or text.startswith("format.")
            ):
                continue
            graph_inputs.append(
                {"node_id": str(node_id), "class": class_name, "input": text}
            )
    graph_inputs.sort(key=lambda item: (item["node_id"], item["class"], item["input"]))
    return {
        "schema": v3_readiness.RUNTIME_EVIDENCE_SCHEMA,
        "run_id": lifecycle.run_id,
        "nonce": lifecycle.nonce,
        "session_binding_sha256": v3_readiness.session_binding_digest(binding),
        "source_bundle_sha256": v3_readiness.source_bundle_digest(source_hashes),
        "source_approval_sha256": approval.sha256,
        "host_key": dict(host_key_fact),
        "python": {"version": "3.12.9"},
        "torch": {
            "version": "2.9.1+cu128",
            "cuda_version": "12.8",
            "cuda_available": True,
            "device_count": 1,
            "device_name": "NVIDIA GeForce RTX 4090",
            "total_vram_bytes": 25757220864,
        },
        "comfyui": {
            "commit": bundle["manifest"]["comfyui"]["commit"],
            "import_ok": True,
            "import_error": None,
        },
        "pip": {
            "check_exit": 0,
            "freeze_sha256": "a" * 64,
            "freeze_line_count": 10,
        },
        "models": models,
        "system_stats": {
            "sha256": "e" * 64,
            "devices": [
                {
                    "name": "NVIDIA GeForce RTX 4090",
                    "type": "cuda",
                    "index": 0,
                    "vram_total": 25757220864,
                }
            ],
        },
        "object_info": {
            "sha256": "b" * 64,
            "size_bytes": 2048,
            "classes": classes,
            "descriptors": descriptors,
            "missing_classes": [],
            "validation_issues": [],
            "dynamic_combo": {
                "path": ["format", "codec", "encoding"],
                "selected": {"format": "mp4", "codec": "h264", "encoding": "auto"},
                "issues": [],
            },
        },
        "graph": {
            "node_classes": {str(key): value for key, value in node_types.items()},
            "required_classes": classes,
            "missing_graph_classes": [],
            "input_checks": [
                {**item, "present": True} for item in graph_inputs
            ],
        },
        "collector": {"sha256": "c" * 64, "plan_sha256": "d" * 64},
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


def install_live_candidate(
    *,
    run: Path,
    session_state: Path,
    quote_path: Path,
    remote: Any,
    fingerprint: str,
) -> dict[str, Any]:
    bundle = dict(v3_operator.load_bundle())
    source_hashes = v3_operator._bundle_hashes(bundle)
    approval = v3_readiness.create_source_approval(
        bundle=bundle,
        source_hashes=source_hashes,
        output_path=run / "source-approval.json",
    )
    state = v3_operator._load_session_state(session_state)
    binding = v3_lifecycle.SessionBinding.from_state(state)
    observation = v3_readiness.create_quote_observation(
        quote_path=quote_path,
        session_state=session_state,
        binding=binding,
    )
    host_key_fact = v3_readiness.make_host_key_fact(remote, fingerprint)
    lifecycle = v3_lifecycle.load_lifecycle(run / "lifecycle.json")
    evidence = fake_evidence(
        bundle=bundle,
        binding=binding,
        lifecycle=lifecycle,
        approval=approval,
        host_key_fact=host_key_fact,
        source_hashes=source_hashes,
    )
    candidate = v3_readiness.build_live_candidate(
        run_dir=run,
        bundle=bundle,
        source_hashes=source_hashes,
        binding=binding,
        lifecycle=lifecycle,
        approval=approval,
        quote_observation=observation,
        host_key_fact=host_key_fact,
        evidence=evidence,
        evidence_raw=json.dumps(evidence, sort_keys=True).encode("utf-8"),
        collector_sha256="c" * 64,
        plan_sha256="d" * 64,
    )
    pipeline = dict(lifecycle.pipeline or {})
    pipeline["candidate"] = v3_operator._candidate_facts(candidate)
    pipeline["verification"] = {
        "RUNTIME_READINESS": "VERIFIED",
        "SOURCE_STATUS": "DRAFT",
        "GPU_QUALITY_VERIFIED": False,
        "GENERATION_STATE": "NOT_RUN",
    }
    lifecycle = v3_lifecycle.update_phase(
        run / "lifecycle.json",
        lifecycle,
        phase="GENERATION_READY",
        pipeline=pipeline,
    )
    return {
        "approval": approval,
        "observation": observation,
        "host_key_fact": host_key_fact,
        "candidate": candidate,
        "binding": binding,
        "lifecycle": lifecycle,
        "source_hashes": source_hashes,
    }


def supervisor_fact_for(binding: v3_lifecycle.SessionBinding) -> dict[str, Any]:
    import time

    return {
        "run_id": "run",
        "nonce": "nonce",
        "pid": 4321,
        "status": "ARMED",
        "binding_sha256": v3_readiness.session_binding_digest(binding),
        "config_sha256": "e" * 64,
        "heartbeat_at": time.time(),
    }


def write_quote_observation_for_test(
    *,
    quote_path: Path,
    session_state: Path,
    binding: v3_lifecycle.SessionBinding,
) -> v3_readiness.QuoteObservation:
    return v3_readiness.create_quote_observation(
        quote_path=quote_path,
        session_state=session_state,
        binding=binding,
    )
