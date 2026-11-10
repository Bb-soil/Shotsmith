from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from anime_studio.p1_config import check_readiness, validate_candidate_manifest, load_config
from scripts.p1 import clore_session as cs
from scripts.p1 import v3_lifecycle, v3_operator, v3_readiness
from scripts.p1.remote import RemoteIdentity
from tests.p1_live_candidate_helpers import install_live_candidate

ROOT = Path(__file__).resolve().parents[1]
QUOTE_PATH = ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json"


def _trusted_identity(tmp_path: Path) -> tuple[RemoteIdentity, str]:
    identity = tmp_path / "id_trusted"
    known = tmp_path / "known_hosts_trusted"
    identity.write_text("not-a-key", encoding="utf-8")
    blob = b"readiness-test-key"
    known.write_text(
        f"[example.test]:22 ssh-ed25519 {base64.b64encode(blob).decode()}\n",
        encoding="utf-8",
    )
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return (
        RemoteIdentity(
            host="example.test",
            user="root",
            port=22,
            identity_file=identity,
            known_hosts=known,
        ),
        fingerprint,
    )


def _context(tmp_path: Path):
    quote = cs.CloreQuote.from_dict(json.loads(QUOTE_PATH.read_text(encoding="utf-8")))
    now = datetime.now(timezone.utc)
    state_path = tmp_path / "session.json"
    state = cs.initialize_state(
        state_path,
        order_id=97001,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        now=now,
    )
    binding = v3_lifecycle.SessionBinding.from_state(state)
    bundle = dict(v3_operator.load_bundle())
    source_hashes = v3_operator._bundle_hashes(bundle)
    approval = v3_readiness.create_source_approval(
        bundle=bundle,
        source_hashes=source_hashes,
        output_path=tmp_path / "source-approval.json",
    )
    observation = v3_readiness.create_quote_observation(
        quote_path=QUOTE_PATH,
        session_state=state_path,
        binding=binding,
    )
    remote, fingerprint = _trusted_identity(tmp_path)
    host_key_fact = v3_readiness.make_host_key_fact(remote, fingerprint)
    run = tmp_path / "run"
    run.mkdir()
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=run.name,
        session=binding,
        bundle_hashes=source_hashes,
        phase="READY",
    )
    v3_lifecycle.write_lifecycle(run / "lifecycle.json", lifecycle)
    supervisor_fact = {
        "run_id": run.name,
        "nonce": lifecycle.nonce,
        "pid": 4321,
        "status": "ARMED",
        "binding_sha256": v3_readiness.session_binding_digest(binding),
        "config_sha256": "e" * 64,
        "heartbeat_at": time.time(),
    }
    return {
        "bundle": bundle,
        "source_hashes": source_hashes,
        "approval": approval,
        "observation": observation,
        "state_path": state_path,
        "state": state,
        "binding": binding,
        "remote": remote,
        "fingerprint": fingerprint,
        "host_key_fact": host_key_fact,
        "run": run,
        "lifecycle": lifecycle,
        "supervisor_fact": supervisor_fact,
    }


def _source_gate(context, **overrides):
    values = {
        "bundle": context["bundle"],
        "source_hashes": context["source_hashes"],
        "approval": context["approval"],
        "quote_observation": context["observation"],
        "session_state": context["state"],
        "binding": context["binding"],
        "host_key_fact": context["host_key_fact"],
        "remote": context["remote"],
        "expected_fingerprint": context["fingerprint"],
        "supervisor_fact": context["supervisor_fact"],
        "require_supervisor": True,
        "run_id": context["run"].name,
        "nonce": context["lifecycle"].nonce,
    }
    values.update(overrides)
    return v3_readiness.evaluate_source_gate(**values)


def test_prepare_live_requires_source_approval_before_run_dir(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    run = tmp_path / "must-not-exist"
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(
            live=True,
            run_dir=run,
            session_state=context["state_path"],
            order_id=context["state"].order_id,
            created_at=context["state"].created_at,
            remote=context["remote"],
            expected_fingerprint=context["fingerprint"],
        )
    assert not run.exists()


def test_prepare_live_valid_source_no_downloads_does_no_remote_or_post(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    run = tmp_path / "prepared-live"
    monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)

    def fake_start(supervisor, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            to_dict=lambda: {
                "run_id": supervisor.run_id,
                "nonce": supervisor.nonce,
            }
        )

    monkeypatch.setattr(v3_lifecycle.CloreSupervisor, "start", fake_start)
    result = v3_operator.prepare(
        live=True,
        run_dir=run,
        session_state=context["state_path"],
        order_id=context["state"].order_id,
        created_at=context["state"].created_at,
        source_approval=context["approval"].path,
        remote=context["remote"],
        expected_fingerprint=context["fingerprint"],
    )
    assert result["status"] in {"PREPARED_OFFLINE", "SUPERVISOR_PENDING"}
    assert (run / "run.json").is_file()
    assert not (run / "jobs.sqlite3").exists()
    assert not (run / "live").exists()
    runbook = (run / "runbook.ps1").read_text(encoding="utf-8")
    assert "--source-approval" in runbook
    assert "session-bind" in runbook
    assert "approve-source" in runbook
    assert "StrictHostKeyChecking" not in runbook


def test_valid_draft_approved_source_allows_bootstrap_only(tmp_path: Path) -> None:
    context = _context(tmp_path)
    gate = _source_gate(context)
    assert gate.passed, gate.issues
    assert gate.facts["phase"] == "SOURCE_BOOTSTRAP"
    assert context["bundle"]["manifest"]["status"] == "DRAFT"
    assert (context["run"] / "live").exists() is False
    assert context["source_hashes"]["manifest"] == v3_readiness.FROZEN_SOURCE_BUNDLE["manifest"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda b: b["manifest"]["models"][0].update({"sha256": "0" * 64}),
        lambda b: b["manifest"]["models"][1].update({"license_reviewed_at": "2999-01-01"}),
        lambda b: b["manifest"]["models"][2].update({"size_bytes": 1}),
        lambda b: b["manifest"]["live_plan"].update({"seed": 1}),
    ],
)
def test_source_gate_rejects_missing_pin_license_or_cross_bind(
    tmp_path: Path, mutator
) -> None:
    context = _context(tmp_path)
    bundle = copy.deepcopy(context["bundle"])
    mutator(bundle)
    gate = _source_gate(context, bundle=bundle)
    assert not gate.passed
    assert gate.issues


def test_source_gate_rejects_approval_quote_age_budget_deadline_host_supervisor(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    approval_payload = copy.deepcopy(dict(context["approval"].payload))
    approval_payload["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    expired_approval = v3_readiness.SourceApproval(
        path=context["approval"].path,
        payload=approval_payload,
        sha256=context["approval"].sha256,
    )
    assert not _source_gate(context, approval=expired_approval).passed

    quote_payload = copy.deepcopy(dict(context["observation"].payload))
    quote_payload["observed_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=1)
    ).isoformat()
    stale_quote = v3_readiness.QuoteObservation(
        path=context["observation"].path,
        payload=quote_payload,
        sha256=context["observation"].sha256,
    )
    assert not _source_gate(context, quote_observation=stale_quote).passed

    wrong_host = dict(context["host_key_fact"])
    wrong_host["fingerprint"] = "SHA256:wrong"
    assert not _source_gate(context, host_key_fact=wrong_host).passed

    assert not _source_gate(context, supervisor_fact=None).passed
    assert not _source_gate(
        context,
        now=datetime.now(timezone.utc) + timedelta(hours=2),
    ).passed

    two_dollar = replace(context["binding"], cap_usd="2.00")
    assert not _source_gate(context, binding=two_dollar).passed


def test_candidate_is_explicit_and_source_draft_stays_immutable(tmp_path: Path) -> None:
    context = _context(tmp_path)
    before = {
        path: path.read_bytes()
        for path in (
            ROOT / v3_readiness.SOURCE_PATHS["manifest"],
            ROOT / v3_readiness.SOURCE_PATHS["config"],
            ROOT / v3_readiness.SOURCE_PATHS["workflow"],
        )
    }
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )
    candidate = installed["candidate"]
    after = {
        path: path.read_bytes()
        for path in (
            ROOT / v3_readiness.SOURCE_PATHS["manifest"],
            ROOT / v3_readiness.SOURCE_PATHS["config"],
            ROOT / v3_readiness.SOURCE_PATHS["workflow"],
        )
    }
    assert before == after
    source_manifest = json.loads(
        (ROOT / v3_readiness.SOURCE_PATHS["manifest"]).read_text(encoding="utf-8")
    )
    candidate_manifest = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
    assert source_manifest["status"] == "DRAFT"
    assert candidate_manifest["status"] == "VERIFIED"
    assert candidate_manifest["candidate"]["source_status"] == "DRAFT"
    assert candidate_manifest["verification"]["GPU_QUALITY_VERIFIED"] is False
    assert candidate_manifest["verification"]["GENERATION_STATE"] == "NOT_RUN"
    assert candidate.manifest_path != ROOT / v3_readiness.SOURCE_PATHS["manifest"]
    assert check_readiness(candidate.config_path).live_ready is True
    assert candidate.binding["source_bundle_sha256"] == v3_readiness.source_bundle_digest(
        installed["source_hashes"]
    )


def test_candidate_without_explicit_provenance_is_not_generation_ready(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )
    candidate = json.loads(
        installed["candidate"].manifest_path.read_text(encoding="utf-8")
    )
    candidate.pop("candidate")
    config = load_config(installed["candidate"].config_path)
    issues = validate_candidate_manifest(
        candidate,
        workflow_path=config.workflow_path,
        actual_workflow_sha256=config.workflow_sha256 or "",
        config=config,
        manifest_path=installed["candidate"].manifest_path,
    )
    assert issues


def test_stale_tampered_or_wrong_identity_evidence_is_refused(tmp_path: Path) -> None:
    context = _context(tmp_path)
    bundle = context["bundle"]
    source_hashes = context["source_hashes"]
    binding = context["binding"]
    lifecycle = context["lifecycle"]
    base = {
        "schema": v3_readiness.RUNTIME_EVIDENCE_SCHEMA,
        "run_id": context["run"].name,
        "nonce": lifecycle.nonce,
        "session_binding_sha256": v3_readiness.session_binding_digest(binding),
        "source_bundle_sha256": v3_readiness.source_bundle_digest(source_hashes),
        "source_approval_sha256": context["approval"].sha256,
        "host_key": dict(context["host_key_fact"]),
        "python": {"version": "3.12.9"},
        "torch": {"version": "2.9.1", "cuda_version": "13.0"},
        "comfyui": {
            "commit": bundle["manifest"]["comfyui"]["commit"],
            "import_ok": True,
            "import_error": None,
        },
        "pip": {"check_exit": 0, "freeze_sha256": "a" * 64, "freeze_line_count": 4},
        "models": [
            {
                "id": model["id"],
                "target": model["target"],
                "size_bytes": model["size_bytes"],
                "sha256": model["sha256"],
                "actual_size_bytes": model["size_bytes"],
                "actual_sha256": model["sha256"],
            }
            for model in bundle["manifest"]["models"]
        ],
        "object_info": {
            "sha256": "b" * 64,
            "size_bytes": 10,
            "classes": ["KSampler"],
            "missing_classes": [],
            "validation_issues": [],
            "dynamic_combo": {
                "path": ["format", "codec", "encoding"],
                "selected": {"format": "mp4", "codec": "h264", "encoding": "auto"},
                "issues": [],
            },
        },
        "graph": {
            "node_classes": {"3": "KSampler"},
            "required_classes": ["KSampler"],
            "missing_graph_classes": [],
            "input_checks": [{"class": "KSampler", "input": "seed", "present": True}],
        },
        "collector": {"sha256": "c" * 64, "plan_sha256": "d" * 64},
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    stale = copy.deepcopy(base)
    stale["observed_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with pytest.raises(v3_readiness.ReadinessError):
        v3_readiness.build_live_candidate(
            run_dir=tmp_path / "stale-run",
            bundle=bundle,
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=replace(lifecycle, run_id="stale-run"),
            approval=context["approval"],
            quote_observation=context["observation"],
            host_key_fact=context["host_key_fact"],
            evidence=stale,
            evidence_raw=json.dumps(stale, sort_keys=True).encode("utf-8"),
            collector_sha256="c" * 64,
            plan_sha256="d" * 64,
        )
    wrong = copy.deepcopy(base)
    wrong["session_binding_sha256"] = "0" * 64
    with pytest.raises(v3_readiness.ReadinessError):
        v3_readiness.build_live_candidate(
            run_dir=tmp_path / "wrong-run",
            bundle=bundle,
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=replace(lifecycle, run_id="wrong-run"),
            approval=context["approval"],
            quote_observation=context["observation"],
            host_key_fact=context["host_key_fact"],
            evidence=wrong,
            evidence_raw=json.dumps(wrong, sort_keys=True).encode("utf-8"),
            collector_sha256="c" * 64,
            plan_sha256="d" * 64,
        )
    # Tamper with an existing candidate's raw file; load must refuse it.
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )
    installed["candidate"].raw_path.write_bytes(b"tampered\n")
    with pytest.raises(v3_readiness.ReadinessError):
        v3_readiness.load_live_candidate(
            run_dir=context["run"],
            source_hashes=source_hashes,
            binding=binding,
            lifecycle=context["lifecycle"],
        )


def test_actual_probe_emits_compact_facts_and_never_leaks_full_object_info(
    tmp_path: Path,
) -> None:
    from tests.p1_probe_helpers import make_probe_workspace, model_facts_for, run_actual_probe

    target = "models/diffusion_models/fixture.safetensors"
    workspace, mods, commit = make_probe_workspace(tmp_path, [target])
    facts = model_facts_for(workspace, [target])
    plan = {
        "schema": v3_readiness.RUNTIME_PLAN_SCHEMA,
        "run_id": "run-fixture",
        "nonce": "nonce-fixture",
        "session_binding_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "source_approval_sha256": "c" * 64,
        "host_key": {
            "fingerprint": "SHA256:trusted",
            "known_hosts_sha256": "d" * 64,
            "host": "example.test",
            "user": "root",
            "port": 22,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        },
        "workspace": str(workspace),
        "owned_root": str(ROOT),
        "venv_python": sys.executable,
        "comfy_dir": str(workspace / "ComfyUI"),
        "object_info_path": str(workspace / "p1-evidence" / "object_info.json"),
        "preflight_path": str(ROOT / "scripts" / "p1" / "preflight.py"),
        "models": facts,
        "required_classes": ["KSampler", "SaveVideo"],
        "required_inputs": {"CreateVideo": ["fps"]},
        "node_classes": {"3": "KSampler", "58": "SaveVideo"},
        "dynamic_path": ["format", "codec", "encoding"],
        "expected_comfy_commit": commit,
        "deadline_epoch": time.time() + 120,
        "max_bytes": 256 * 1024,
    }
    payload, raw = run_actual_probe(tmp_path=tmp_path, plan=plan, mods=mods)
    assert payload["schema"] == v3_readiness.RUNTIME_EVIDENCE_SCHEMA
    assert payload["models"][0]["actual_sha256"] == facts[0]["sha256"]
    assert payload["object_info"]["dynamic_combo"]["selected"] == {
        "format": "mp4",
        "codec": "h264",
        "encoding": "auto",
    }
    assert "full" not in raw.decode("utf-8").lower()
    assert len(raw) < 64 * 1024

def test_cli_readiness_approve_and_bind_are_offline_no_secret_argv(tmp_path: Path, capsys) -> None:
    assert v3_operator.main(["readiness", "--json"]) == 0
    readiness_payload = json.loads(capsys.readouterr().out)
    assert readiness_payload["status"] == "SOURCE_READY"
    assert readiness_payload["manifest_status"] == "DRAFT"

    approval_path = tmp_path / "approval.json"
    assert (
        v3_operator.main(
            [
                "approve-source",
                "--approval-out",
                str(approval_path),
                "--cap-usd",
                "1.00",
                "--json",
            ]
        )
        == 0
    )
    approval_payload = json.loads(capsys.readouterr().out)
    assert approval_payload["status"] == "SOURCE_APPROVED"
    assert approval_payload["scope"]["max_total_usd"] == "1.00"

    state_path = tmp_path / "session.json"
    created_at = int(datetime.now(timezone.utc).timestamp()) - 5
    assert (
        v3_operator.main(
            [
                "session-bind",
                "--session-state",
                str(state_path),
                "--quote",
                str(QUOTE_PATH),
                "--order-id",
                "97001",
                "--created-at",
                str(created_at),
                "--json",
            ]
        )
        == 0
    )
    bind_payload = json.loads(capsys.readouterr().out)
    assert bind_payload["status"] == "SESSION_BOUND"
    assert Path(bind_payload["quote_observation"]).is_file()
    assert not (tmp_path / "jobs.sqlite3").exists()

def _candidate_validation(candidate):
    config = load_config(candidate.config_path)
    manifest = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
    return manifest, config


def _validate_mutated(manifest, config, candidate):
    return validate_candidate_manifest(
        manifest,
        workflow_path=config.workflow_path,
        actual_workflow_sha256=config.workflow_sha256 or "",
        config=config,
        manifest_path=candidate.manifest_path,
    )


def test_raw_invalid_contradictory_and_recomputed_hash_are_refused(tmp_path: Path) -> None:
    context = _context(tmp_path)
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )
    candidate = installed["candidate"]
    manifest, config = _candidate_validation(candidate)

    candidate.raw_path.write_text("{", encoding="utf-8")
    digest = hashlib.sha256(candidate.raw_path.read_bytes()).hexdigest()
    manifest["candidate"]["evidence"]["raw_sha256"] = digest
    manifest["candidate"]["runtime_evidence"]["raw_sha256"] = digest
    assert _validate_mutated(manifest, config, candidate)

    contradictory = json.loads(candidate.evidence_path.read_text(encoding="utf-8"))
    contradictory["torch"]["version"] = "9.9.9+cu999"
    candidate.raw_path.write_bytes(json.dumps(contradictory, sort_keys=True).encode("utf-8"))
    digest = hashlib.sha256(candidate.raw_path.read_bytes()).hexdigest()
    manifest["candidate"]["evidence"]["raw_sha256"] = digest
    manifest["candidate"]["runtime_evidence"]["raw_sha256"] = digest
    assert _validate_mutated(manifest, config, candidate)


def test_graph_class_input_nested_codec_torch_and_system_facts_are_refused(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )
    candidate = installed["candidate"]
    baseline_manifest, config = _candidate_validation(candidate)
    cases = []
    missing_class = copy.deepcopy(baseline_manifest)
    missing_class["candidate"]["runtime_evidence"]["object_info"]["descriptors"].pop("KSampler")
    cases.append(missing_class)
    missing_input = copy.deepcopy(baseline_manifest)
    missing_input["candidate"]["runtime_evidence"]["object_info"]["descriptors"]["KSampler"]["input"]["required"].pop("seed", None)
    missing_input["candidate"]["runtime_evidence"]["object_info"]["descriptors"]["KSampler"]["input"]["required"].pop("steps", None)
    missing_input["candidate"]["runtime_evidence"]["object_info"]["descriptors"]["KSampler"]["input"]["required"].pop("cfg", None)
    cases.append(missing_input)
    wrong_codec = copy.deepcopy(baseline_manifest)
    save = wrong_codec["candidate"]["runtime_evidence"]["object_info"]["descriptors"]["SaveVideo"]["input"]["required"]["format"][1]["options"]
    mp4 = next(item for item in save if item.get("key") == "mp4")
    codec_options = mp4["inputs"]["required"]["codec"][1]["options"]
    next(item for item in codec_options if item.get("key") == "h264")["key"] = "av1"
    cases.append(wrong_codec)
    false_cuda = copy.deepcopy(baseline_manifest)
    false_cuda["candidate"]["runtime_evidence"]["torch"]["cuda_available"] = False
    false_cuda["candidate"]["runtime_evidence"]["torch"]["device_count"] = 0
    cases.append(false_cuda)
    incompatible_torch = copy.deepcopy(baseline_manifest)
    incompatible_torch["candidate"]["runtime_evidence"]["torch"]["version"] = "2.6.0+cu126"
    incompatible_torch["candidate"]["runtime_evidence"]["torch"]["cuda_version"] = "12.6"
    cases.append(incompatible_torch)
    bad_system = copy.deepcopy(baseline_manifest)
    bad_system["candidate"]["runtime_evidence"]["system_stats"]["devices"][0]["type"] = "cpu"
    bad_system["candidate"]["runtime_evidence"]["system_stats"]["devices"][0]["vram_total"] = 0
    cases.append(bad_system)
    for manifest in cases:
        assert _validate_mutated(manifest, config, candidate)


def test_approval_cap_is_effective_and_binding_cannot_exceed_it(tmp_path: Path) -> None:
    context = _context(tmp_path)
    approval = v3_readiness.create_source_approval(
        bundle=context["bundle"],
        source_hashes=context["source_hashes"],
        output_path=tmp_path / "approval-050.json",
        cap_usd="0.50",
    )
    assert not _source_gate(context, approval=approval).passed

    quote = {
        "price_per_day": "5.00",
        "price_per_hour": "0.25",
        "price_per_minute": "0.005",
        "currency": "USD-Blockchain",
        "creation_fee": "0.05",
        "mandatory_other_fees": "0.00",
        "reserve": "0.02",
        "fresh_verified": True,
        "fees_known": True,
        "tax_fx_known": True,
        "min_rental_hours": "1",
        "provider_fee_percent": "5",
    }
    quote_path = tmp_path / "quote-050.json"
    quote_path.write_text(json.dumps(quote), encoding="utf-8")
    state_path = tmp_path / "session-050.json"
    now = datetime.now(timezone.utc)
    state = cs.initialize_state(
        state_path,
        order_id=97002,
        created_at=int(now.timestamp()) - 5,
        quote=cs.CloreQuote.from_dict(quote),
        cap_usd=Decimal("0.50"),
        now=now,
    )
    binding = v3_lifecycle.SessionBinding.from_state(state)
    observation = v3_readiness.create_quote_observation(
        quote_path=quote_path,
        session_state=state_path,
        binding=binding,
    )
    supervisor_fact = {
        "run_id": context["run"].name,
        "nonce": context["lifecycle"].nonce,
        "pid": 4321,
        "status": "ARMED",
        "binding_sha256": v3_readiness.session_binding_digest(binding),
        "config_sha256": "e" * 64,
        "heartbeat_at": time.time(),
    }
    valid_gate = v3_readiness.evaluate_source_gate(
        bundle=context["bundle"],
        source_hashes=context["source_hashes"],
        approval=approval,
        quote_observation=observation,
        session_state=state,
        binding=binding,
        host_key_fact=context["host_key_fact"],
        remote=context["remote"],
        expected_fingerprint=context["fingerprint"],
        supervisor_fact=supervisor_fact,
        require_supervisor=True,
        run_id=context["run"].name,
        nonce=context["lifecycle"].nonce,
    )
    assert valid_gate.passed, valid_gate.issues

    over_cap_state = cs.initialize_state(
        tmp_path / "session-100.json",
        order_id=97003,
        created_at=int(now.timestamp()) - 5,
        quote=cs.CloreQuote.from_dict(quote),
        cap_usd=Decimal("1.00"),
        now=now,
    )
    over_binding = v3_lifecycle.SessionBinding.from_state(over_cap_state)
    over_observation = v3_readiness.create_quote_observation(
        quote_path=quote_path,
        session_state=tmp_path / "session-100.json",
        binding=over_binding,
    )
    over_supervisor_fact = dict(supervisor_fact)
    over_supervisor_fact["binding_sha256"] = v3_readiness.session_binding_digest(over_binding)
    blocked = v3_readiness.evaluate_source_gate(
        bundle=context["bundle"],
        source_hashes=context["source_hashes"],
        approval=approval,
        quote_observation=over_observation,
        session_state=over_cap_state,
        binding=over_binding,
        host_key_fact=context["host_key_fact"],
        remote=context["remote"],
        expected_fingerprint=context["fingerprint"],
        supervisor_fact=over_supervisor_fact,
        require_supervisor=True,
        run_id=context["run"].name,
        nonce=context["lifecycle"].nonce,
    )
    assert not blocked.passed
    assert any("approval cap" in issue for issue in blocked.issues)

def _probe_plan_for(workspace, commit, facts):
    return {
        "schema": v3_readiness.RUNTIME_PLAN_SCHEMA,
        "run_id": "run-fixture",
        "nonce": "nonce-fixture",
        "session_binding_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "source_approval_sha256": "c" * 64,
        "host_key": {
            "fingerprint": "SHA256:trusted",
            "known_hosts_sha256": "d" * 64,
            "host": "example.test",
            "user": "root",
            "port": 22,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        },
        "workspace": str(workspace),
        "owned_root": str(ROOT),
        "venv_python": sys.executable,
        "comfy_dir": str(workspace / "ComfyUI"),
        "object_info_path": str(workspace / "p1-evidence" / "object_info.json"),
        "preflight_path": str(ROOT / "scripts" / "p1" / "preflight.py"),
        "models": facts,
        "required_classes": ["KSampler", "SaveVideo"],
        "required_inputs": {"CreateVideo": ["fps"]},
        "node_classes": {"3": "KSampler", "58": "SaveVideo"},
        "dynamic_path": ["format", "codec", "encoding"],
        "expected_comfy_commit": commit,
        "deadline_epoch": time.time() + 120,
        "max_bytes": 256 * 1024,
    }


def test_probe_ignores_malicious_same_name_modules_outside_checkout(tmp_path: Path) -> None:
    from tests.p1_probe_helpers import make_probe_workspace, model_facts_for, run_probe_process

    target = "models/diffusion_models/fixture.safetensors"
    workspace, mods, commit = make_probe_workspace(tmp_path, [target])
    external = tmp_path / "external-modules"
    (external / "comfy").mkdir(parents=True)
    (external / "comfy" / "__init__.py").write_text("raise RuntimeError('foreign comfy')\n", encoding="utf-8")
    (external / "nodes.py").write_text("raise RuntimeError('foreign nodes')\n", encoding="utf-8")
    plan = _probe_plan_for(workspace, commit, model_facts_for(workspace, [target]))
    result = run_probe_process(
        tmp_path=tmp_path, plan=plan, mods=mods, extra_pythonpath=[external]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert str(workspace / "ComfyUI" / "comfy" / "__init__.py").lower() in payload["comfyui"]["module"].lower()


def test_probe_rejects_external_same_name_modules_when_checkout_missing(tmp_path: Path) -> None:
    from tests.p1_probe_helpers import make_probe_workspace, model_facts_for, run_probe_process

    target = "models/diffusion_models/fixture.safetensors"
    workspace, mods, commit = make_probe_workspace(tmp_path, [target])
    import shutil

    shutil.rmtree(workspace / "ComfyUI" / "comfy")
    (workspace / "ComfyUI" / "nodes.py").unlink()
    external = tmp_path / "external-modules"
    (external / "comfy").mkdir(parents=True)
    (external / "comfy" / "__init__.py").write_text("__version__='foreign'\n", encoding="utf-8")
    (external / "nodes.py").write_text("# foreign nodes\n", encoding="utf-8")
    plan = _probe_plan_for(workspace, commit, model_facts_for(workspace, [target]))
    result = run_probe_process(
        tmp_path=tmp_path, plan=plan, mods=mods, extra_pythonpath=[external]
    )
    assert result.returncode != 0
    assert "foreign comfy" in result.stderr.lower()


def test_probe_rejects_foreign_preimported_module(tmp_path: Path) -> None:
    from tests.p1_probe_helpers import make_probe_workspace, model_facts_for, run_probe_process

    target = "models/diffusion_models/fixture.safetensors"
    workspace, mods, commit = make_probe_workspace(tmp_path, [target])
    external = tmp_path / "preimport-modules"
    (external / "comfy").mkdir(parents=True)
    (external / "comfy" / "__init__.py").write_text("__version__='foreign'\n", encoding="utf-8")
    (external / "nodes.py").write_text("# foreign nodes\n", encoding="utf-8")
    (external / "sitecustomize.py").write_text("import comfy\n", encoding="utf-8")
    plan = _probe_plan_for(workspace, commit, model_facts_for(workspace, [target]))
    result = run_probe_process(
        tmp_path=tmp_path, plan=plan, mods=mods, extra_pythonpath=[external]
    )
    assert result.returncode != 0
    assert "pre-imported" in result.stderr.lower() or "foreign" in result.stderr.lower()

def test_candidate_manifest_rejects_session_cap_above_approval_cap(tmp_path: Path) -> None:
    context = _context(tmp_path)
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )
    candidate = installed["candidate"]
    manifest, config = _candidate_validation(candidate)
    manifest["candidate"]["source_approval"]["scope"]["max_total_usd"] = "0.50"
    assert _validate_mutated(manifest, config, candidate)

def test_quote_cost_above_approval_cap_blocks_even_with_one_dollar_binding(tmp_path: Path) -> None:
    context = _context(tmp_path)
    approval = v3_readiness.create_source_approval(
        bundle=context["bundle"],
        source_hashes=context["source_hashes"],
        output_path=tmp_path / "approval-050-high.json",
        cap_usd="0.50",
    )
    high_quote = {
        "price_per_day": "10.00",
        "price_per_hour": "0.30",
        "price_per_minute": "0.005",
        "currency": "USD-Blockchain",
        "creation_fee": "0.05",
        "mandatory_other_fees": "0.00",
        "reserve": "0.02",
        "fresh_verified": True,
        "fees_known": True,
        "tax_fx_known": True,
        "min_rental_hours": "1",
        "provider_fee_percent": "5",
    }
    quote_path = tmp_path / "quote-high.json"
    quote_path.write_text(json.dumps(high_quote), encoding="utf-8")
    state_path = tmp_path / "session-high.json"
    now = datetime.now(timezone.utc)
    state = cs.initialize_state(
        state_path,
        order_id=97004,
        created_at=int(now.timestamp()) - 5,
        quote=cs.CloreQuote.from_dict(high_quote),
        cap_usd=Decimal("1.00"),
        now=now,
    )
    binding = v3_lifecycle.SessionBinding.from_state(state)
    observation = v3_readiness.create_quote_observation(
        quote_path=quote_path, session_state=state_path, binding=binding
    )
    supervisor_fact = {
        "run_id": context["run"].name,
        "nonce": context["lifecycle"].nonce,
        "pid": 4321,
        "status": "ARMED",
        "binding_sha256": v3_readiness.session_binding_digest(binding),
        "config_sha256": "e" * 64,
        "heartbeat_at": time.time(),
    }
    gate = v3_readiness.evaluate_source_gate(
        bundle=context["bundle"],
        source_hashes=context["source_hashes"],
        approval=approval,
        quote_observation=observation,
        session_state=state,
        binding=binding,
        host_key_fact=context["host_key_fact"],
        remote=context["remote"],
        expected_fingerprint=context["fingerprint"],
        supervisor_fact=supervisor_fact,
        require_supervisor=True,
        run_id=context["run"].name,
        nonce=context["lifecycle"].nonce,
    )
    assert not gate.passed
    assert any("approval cap" in issue or "projected total" in issue for issue in gate.issues)

def test_builder_refuses_raw_mismatch_or_invalid_schema(tmp_path: Path) -> None:
    from tests.p1_live_candidate_helpers import fake_evidence

    context = _context(tmp_path)
    evidence = fake_evidence(
        bundle=context["bundle"],
        binding=context["binding"],
        lifecycle=context["lifecycle"],
        approval=context["approval"],
        host_key_fact=context["host_key_fact"],
        source_hashes=context["source_hashes"],
    )
    common = dict(
        bundle=context["bundle"],
        source_hashes=context["source_hashes"],
        binding=context["binding"],
        lifecycle=context["lifecycle"],
        approval=context["approval"],
        quote_observation=context["observation"],
        host_key_fact=context["host_key_fact"],
        evidence=evidence,
        collector_sha256="c" * 64,
        plan_sha256="d" * 64,
    )
    with pytest.raises(v3_readiness.ReadinessError):
        v3_readiness.build_live_candidate(
            run_dir=tmp_path / "raw-mismatch", evidence_raw=b"{}\n", **common
        )
    with pytest.raises(v3_readiness.ReadinessError):
        v3_readiness.build_live_candidate(
            run_dir=tmp_path / "raw-invalid", evidence_raw=b"{", **common
        )

def test_probe_accepts_true_namespace_comfy_inside_checkout(tmp_path: Path) -> None:
    from tests.p1_probe_helpers import make_probe_workspace, model_facts_for, run_probe_process

    target = "models/diffusion_models/fixture.safetensors"
    workspace, mods, commit = make_probe_workspace(tmp_path, [target])
    (workspace / "ComfyUI" / "comfy" / "__init__.py").unlink()
    plan = _probe_plan_for(workspace, commit, model_facts_for(workspace, [target]))
    result = run_probe_process(tmp_path=tmp_path, plan=plan, mods=mods)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["comfyui"]["module_kind"] == "namespace"
    assert payload["comfyui"]["module"].lower().replace("\\", "/").endswith("comfyui/comfy")


def test_probe_rejects_foreign_namespace_comfy(tmp_path: Path) -> None:
    import shutil

    from tests.p1_probe_helpers import make_probe_workspace, model_facts_for, run_probe_process

    target = "models/diffusion_models/fixture.safetensors"
    workspace, mods, commit = make_probe_workspace(tmp_path, [target])
    shutil.rmtree(workspace / "ComfyUI" / "comfy")
    (workspace / "ComfyUI" / "nodes.py").unlink()
    external = tmp_path / "foreign-namespace"
    (external / "comfy").mkdir(parents=True)
    (external / "nodes.py").write_text("# foreign nodes\n", encoding="utf-8")
    plan = _probe_plan_for(workspace, commit, model_facts_for(workspace, [target]))
    result = run_probe_process(
        tmp_path=tmp_path, plan=plan, mods=mods, extra_pythonpath=[external]
    )
    assert result.returncode != 0
    assert "foreign comfy namespace" in result.stderr.lower() or "foreign comfy" in result.stderr.lower()

def _write_semantically_consistent_candidate(tmp_path: Path, installed, mutator):
    candidate = installed["candidate"]
    manifest = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
    runtime = copy.deepcopy(manifest["candidate"]["runtime_evidence"])
    runtime.pop("raw_sha256", None)
    runtime.pop("raw_size_bytes", None)
    mutator(runtime)
    raw_bytes = json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence = copy.deepcopy(runtime)
    evidence["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    evidence["raw_size_bytes"] = len(raw_bytes)
    evidence_bytes = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["candidate"]["runtime_evidence"] = evidence
    manifest["candidate"]["evidence"]["raw_sha256"] = evidence["raw_sha256"]
    manifest["candidate"]["evidence"]["raw_size_bytes"] = len(raw_bytes)
    manifest["candidate"]["evidence"]["sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    room = tmp_path / f"semantic-{len(list(tmp_path.glob('semantic-*')))}"
    room.mkdir()
    (room / "candidate-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (room / "runtime-evidence.json").write_bytes(evidence_bytes)
    (room / "runtime-evidence.raw.json").write_bytes(raw_bytes)
    config = load_config(candidate.config_path)
    return manifest, config, room / "candidate-manifest.json"


def _write_raw_identity_candidate(tmp_path: Path, installed, raw_mutator):
    candidate = installed["candidate"]
    manifest = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
    runtime = copy.deepcopy(manifest["candidate"]["runtime_evidence"])
    raw = json.loads(candidate.raw_path.read_text(encoding="utf-8"))
    raw_mutator(raw)
    raw_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    runtime["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    runtime["raw_size_bytes"] = len(raw_bytes)
    evidence_bytes = json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["candidate"]["runtime_evidence"] = runtime
    manifest["candidate"]["evidence"]["raw_sha256"] = runtime["raw_sha256"]
    manifest["candidate"]["evidence"]["raw_size_bytes"] = len(raw_bytes)
    manifest["candidate"]["evidence"]["sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    room = tmp_path / f"raw-id-{len(list(tmp_path.glob('raw-id-*')))}"
    room.mkdir()
    (room / "candidate-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (room / "runtime-evidence.json").write_bytes(evidence_bytes)
    (room / "runtime-evidence.raw.json").write_bytes(raw_bytes)
    config = load_config(candidate.config_path)
    return manifest, config, room / "candidate-manifest.json"


def test_semantically_bad_consistent_candidate_has_specific_issues(tmp_path: Path) -> None:
    context = _context(tmp_path)
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )

    def missing_class(runtime):
        runtime["object_info"]["descriptors"].pop("KSampler")

    def missing_input(runtime):
        runtime["object_info"]["descriptors"]["KSampler"]["input"]["required"].pop("seed")

    def wrong_codec(runtime):
        options = runtime["object_info"]["descriptors"]["SaveVideo"]["input"]["required"]["format"][1]["options"]
        mp4 = next(item for item in options if item.get("key") == "mp4")
        codec_options = mp4["inputs"]["required"]["codec"][1]["options"]
        next(item for item in codec_options if item.get("key") == "h264")["key"] = "av1"

    def false_cuda(runtime):
        runtime["torch"]["cuda_available"] = False

    def incompatible_torch(runtime):
        runtime["torch"]["version"] = "2.6.0+cu126"
        runtime["torch"]["cuda_version"] = "12.6"

    def cuda_12_80(runtime):
        runtime["torch"]["cuda_version"] = "12.80"

    def cpu_device(runtime):
        runtime["system_stats"]["devices"][0]["type"] = "cpu"

    cases = [
        (missing_class, "descriptor"),
        (missing_input, "missing input 'seed'"),
        (wrong_codec, "SaveVideo dynamic"),
        (false_cuda, "cuda_available must be true"),
        (incompatible_torch, "torch.version is incompatible"),
        (cuda_12_80, "cuda_version is incompatible"),
        (cpu_device, "type must be CUDA"),
    ]
    for index, (mutator, expected) in enumerate(cases):
        manifest, config, manifest_path = _write_semantically_consistent_candidate(
            tmp_path, installed, mutator
        )
        # Each helper call uses a fresh room; keep expected message-specific.
        issues = validate_candidate_manifest(
            manifest,
            workflow_path=config.workflow_path,
            actual_workflow_sha256=config.workflow_sha256 or "",
            config=config,
            manifest_path=manifest_path,
        )
        assert any(expected in issue for issue in issues), (index, expected, issues)


def test_raw_approval_and_host_identity_are_not_synthesized(tmp_path: Path) -> None:
    context = _context(tmp_path)
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )

    def missing_approval(raw):
        raw.pop("source_approval_sha256", None)

    manifest, config, path = _write_raw_identity_candidate(tmp_path, installed, missing_approval)
    issues = validate_candidate_manifest(
        manifest, workflow_path=config.workflow_path,
        actual_workflow_sha256=config.workflow_sha256 or "", config=config,
        manifest_path=path,
    )
    assert any("raw runtime_evidence source_approval identity mismatch" in issue for issue in issues)

    def conflicting_host(raw):
        raw["host_key"] = {**raw["host_key"], "fingerprint": "SHA256:conflict"}

    manifest, config, path = _write_raw_identity_candidate(tmp_path, installed, conflicting_host)
    issues = validate_candidate_manifest(
        manifest, workflow_path=config.workflow_path,
        actual_workflow_sha256=config.workflow_sha256 or "", config=config,
        manifest_path=path,
    )
    assert any("raw runtime_evidence host_key identity mismatch" in issue for issue in issues)
