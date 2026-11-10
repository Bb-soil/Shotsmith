from __future__ import annotations

import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anime_studio.p1_config import check_readiness
from scripts.p1 import clore_session as cs
from scripts.p1 import v3_lifecycle, v3_operator, v3_readiness
from scripts.p1.remote import RemoteIdentity
from tests.comfy_stub import run_stub
from tests.helpers import db_scalar
from tests.p1_live_candidate_helpers import install_live_candidate

ROOT = Path(__file__).resolve().parents[1]
QUOTE_PATH = ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json"


def _trusted_identity(tmp_path: Path) -> tuple[RemoteIdentity, str]:
    identity = tmp_path / "id_trusted"
    known = tmp_path / "known_hosts_trusted"
    identity.write_text("not-a-key", encoding="utf-8")
    blob = b"live-coldstart-test-key"
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
    run = tmp_path / "run"
    run.mkdir()
    v3_operator._ensure_run_document(run, source_hashes)
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=run.name,
        session=binding,
        bundle_hashes=source_hashes,
        phase="READY",
    )
    v3_lifecycle.write_lifecycle(run / "lifecycle.json", lifecycle)
    remote, fingerprint = _trusted_identity(tmp_path)
    return {
        "bundle": bundle,
        "source_hashes": source_hashes,
        "state_path": state_path,
        "state": state,
        "binding": binding,
        "run": run,
        "lifecycle": lifecycle,
        "remote": remote,
        "fingerprint": fingerprint,
    }


def _fake_supervisor_start(self, **kwargs):
    from scripts.p1.v3_lifecycle import SupervisorRecord

    self.config_path.write_text(json.dumps(self._config_payload()), encoding="utf-8")
    config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
    binding_sha256 = hashlib.sha256(
        json.dumps(
            self.binding.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    record = SupervisorRecord(
        run_id=self.run_id,
        nonce=self.nonce,
        pid=4321,
        state_path=str(self.session_state_path),
        quote_path=str(self.quote_path),
        log_path=str(self.log_path),
        started_at=datetime.now(timezone.utc).isoformat(),
        order_id=self.binding.order_id,
        created_at=self.binding.created_at,
        config_sha256=config_sha256,
        binding_sha256=binding_sha256,
    )
    self.record_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    self.ack_path.write_text(
        json.dumps(
            {
                "schema": "anime-video-studio/p1-v3-supervisor-ack/v1",
                "status": "ARMED",
                "run_id": self.run_id,
                "nonce": self.nonce,
                "order_id": self.binding.order_id,
                "created_at": self.binding.created_at,
                "ack_at": time.time(),
                "pid": 4321,
                "config_sha256": config_sha256,
                "binding_sha256": binding_sha256,
            }
        ),
        encoding="utf-8",
    )
    self.heartbeat_path.write_text(
        json.dumps(
            {
                "schema": "anime-video-studio/p1-v3-supervisor-heartbeat/v1",
                "run_id": self.run_id,
                "nonce": self.nonce,
                "at": time.time(),
                "pid": 4321,
                "config_sha256": config_sha256,
                "binding_sha256": binding_sha256,
            }
        ),
        encoding="utf-8",
    )
    return record


def test_full_fake_coldstart_actual_candidate_real_backend_one_post_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    with run_stub() as (base_url, stub):
        original_candidate_config = v3_readiness._candidate_config

        def candidate_config_with_stub(*, live_dir: Path, source_hashes: dict[str, str]):
            config = original_candidate_config(
                live_dir=live_dir, source_hashes=source_hashes
            )
            config["runtime"]["base_url"] = base_url
            return config

        monkeypatch.setattr(
            v3_readiness, "_candidate_config", candidate_config_with_stub
        )
        installed = install_live_candidate(
            run=context["run"],
            session_state=context["state_path"],
            quote_path=QUOTE_PATH,
            remote=context["remote"],
            fingerprint=context["fingerprint"],
        )
        assert check_readiness(installed["candidate"].config_path).live_ready is True
        stub.set_success("result.mp4")
        stub.history[stub.prompt_id]["outputs"] = {
            "58": {
                "images": [
                    {"filename": "result.mp4", "subfolder": "", "type": "output"}
                ]
            }
        }

        monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
        monkeypatch.setattr(v3_lifecycle, "_pid_live", lambda pid: True)
        monkeypatch.setattr(v3_lifecycle.CloreSupervisor, "start", _fake_supervisor_start)
        monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
        monkeypatch.setattr(v3_operator, "_produce_remote_export", lambda **kwargs: None)

        result = v3_operator.run_once(
            live=True,
            session_state=context["state_path"],
            order_id=context["state"].order_id,
            created_at=context["state"].created_at,
            remote=context["remote"],
            expected_fingerprint=context["fingerprint"],
            run_dir=context["run"],
            ready_validator=lambda **kwargs: None,
        )
        assert result["status"] == "succeeded", result
        assert stub.prompt_requests == 1
        assert db_scalar(context["run"] / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 1
        deadline_before = db_scalar(
            context["run"] / "jobs.sqlite3",
            "SELECT deadline_ts FROM jobs LIMIT 1",
        )
        assert (context["run"] / "live" / "candidate-binding.json").is_file()
        assert json.loads((context["run"] / "live" / "candidate-manifest.json").read_text())[
            "verification"
        ]["GENERATION_STATE"] == "NOT_RUN"

        second = v3_operator.run_once(
            live=True,
            session_state=context["state_path"],
            order_id=context["state"].order_id,
            created_at=context["state"].created_at,
            remote=context["remote"],
            expected_fingerprint=context["fingerprint"],
            run_dir=context["run"],
            ready_validator=lambda **kwargs: None,
        )
        assert second["status"] == "succeeded", second
        assert stub.prompt_requests == 1
        assert db_scalar(context["run"] / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 1
        assert (
            db_scalar(
                context["run"] / "jobs.sqlite3",
                "SELECT deadline_ts FROM jobs LIMIT 1",
            )
            == deadline_before
        )
        collected = v3_operator.collect(run_dir=context["run"])
        assert collected["status"].startswith("COLLECTED")

def test_run_once_refuses_tampered_candidate_before_backend_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    installed = install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )
    installed["candidate"].raw_path.write_bytes(b"tampered\n")
    monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
    monkeypatch.setattr(v3_lifecycle, "_pid_live", lambda pid: True)
    monkeypatch.setattr(v3_lifecycle.CloreSupervisor, "start", _fake_supervisor_start)
    monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
    calls: list[str] = []

    def backend_runner(**kwargs):
        calls.append(str(kwargs["target"]))
        return {"status": "SHOULD_NOT_RUN"}

    with pytest.raises(v3_operator.ReadinessBlocked):
        v3_operator.run_once(
            live=True,
            session_state=context["state_path"],
            order_id=context["state"].order_id,
            created_at=context["state"].created_at,
            remote=context["remote"],
            expected_fingerprint=context["fingerprint"],
            run_dir=context["run"],
            backend_runner=backend_runner,
            ready_validator=lambda **kwargs: None,
        )
    assert calls == []


def test_run_once_refuses_candidate_lifecycle_binding_tamper_before_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    install_live_candidate(
        run=context["run"],
        session_state=context["state_path"],
        quote_path=QUOTE_PATH,
        remote=context["remote"],
        fingerprint=context["fingerprint"],
    )
    lifecycle = v3_lifecycle.load_lifecycle(context["run"] / "lifecycle.json")
    pipeline = dict(lifecycle.pipeline or {})
    candidate_facts = dict(pipeline["candidate"])
    candidate_facts["binding_sha256"] = "0" * 64
    pipeline["candidate"] = candidate_facts
    v3_lifecycle.update_phase(
        context["run"] / "lifecycle.json",
        lifecycle,
        phase="GENERATION_READY",
        pipeline=pipeline,
    )
    monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
    monkeypatch.setattr(v3_lifecycle, "_pid_live", lambda pid: True)
    monkeypatch.setattr(v3_lifecycle.CloreSupervisor, "start", _fake_supervisor_start)
    monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
    calls: list[str] = []

    def backend_runner(**kwargs):
        calls.append(str(kwargs["target"]))
        return {"status": "SHOULD_NOT_RUN"}

    with pytest.raises(v3_operator.ReadinessBlocked):
        v3_operator.run_once(
            live=True,
            session_state=context["state_path"],
            order_id=context["state"].order_id,
            created_at=context["state"].created_at,
            remote=context["remote"],
            expected_fingerprint=context["fingerprint"],
            run_dir=context["run"],
            backend_runner=backend_runner,
            ready_validator=lambda **kwargs: None,
        )
    assert calls == []

def test_full_fake_coldstart_actual_probe_synthetic_source_real_backend_one_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy
    import sys

    from tests.p1_probe_helpers import make_probe_workspace, model_facts_for, run_actual_probe

    context = _context(tmp_path)
    targets = [model["target"] for model in context["bundle"]["manifest"]["models"]]
    extra_inputs: dict[str, set[str]] = {}
    for node_id, class_name in context["bundle"]["config"].binding_raw["node_types"].items():
        for input_name in context["bundle"]["workflow"][str(node_id)]["inputs"]:
            text = str(input_name)
            if class_name == "SaveVideo" and (
                text == "format" or text.startswith("format.")
            ):
                continue
            extra_inputs.setdefault(class_name, set()).add(text)
    workspace, mods, commit = make_probe_workspace(
        tmp_path,
        targets,
        extra_inputs={name: sorted(values) for name, values in extra_inputs.items()},
    )
    facts = model_facts_for(workspace, targets)
    manifest = copy.deepcopy(context["bundle"]["manifest"])
    for model, fact in zip(manifest["models"], facts):
        model["size_bytes"] = fact["size_bytes"]
        model["sha256"] = fact["sha256"]
    manifest["comfyui"]["commit"] = commit
    expected_models = {
        model["id"]: {
            "role": model["role"],
            "size_bytes": model["size_bytes"],
            "sha256": model["sha256"],
            "filename": Path(model["filename"]).name,
            "target": model["target"],
        }
        for model in manifest["models"]
    }
    monkeypatch.setattr(v3_readiness, "EXPECTED_V3_MODELS", expected_models)
    manifest_path = tmp_path / "synthetic-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_hashes = dict(context["source_hashes"])
    source_hashes["manifest"] = manifest_hash
    monkeypatch.setitem(v3_readiness.FROZEN_SOURCE_BUNDLE, "manifest", manifest_hash)
    monkeypatch.setitem(v3_readiness.SOURCE_PATHS, "manifest", str(manifest_path))
    monkeypatch.setattr(v3_operator, "V3_MANIFEST", manifest_path)
    bundle = dict(context["bundle"])
    bundle["manifest"] = manifest
    monkeypatch.setattr(v3_operator, "load_bundle", lambda: bundle)
    monkeypatch.setattr(v3_operator, "_bundle_hashes", lambda bundle=None: source_hashes)
    run = tmp_path / "actual-probe-run"
    run.mkdir()
    v3_operator._ensure_run_document(run, source_hashes)
    lifecycle = v3_lifecycle.new_lifecycle(
        run_id=run.name,
        session=context["binding"],
        bundle_hashes=source_hashes,
        phase="READY",
    )
    v3_lifecycle.write_lifecycle(run / "lifecycle.json", lifecycle)
    approval = v3_readiness.create_source_approval(
        bundle=bundle,
        source_hashes=source_hashes,
        output_path=tmp_path / "synthetic-approval.json",
    )
    observation = v3_readiness.create_quote_observation(
        quote_path=QUOTE_PATH,
        session_state=context["state_path"],
        binding=context["binding"],
    )
    host = v3_readiness.make_host_key_fact(context["remote"], context["fingerprint"])
    plan = v3_readiness.build_runtime_evidence_plan(
        bundle=bundle,
        source_hashes=source_hashes,
        approval=approval,
        quote_observation=observation,
        host_key_fact=host,
        binding=context["binding"],
        lifecycle=lifecycle,
        workspace=str(workspace),
        owned_root=str(ROOT),
        venv_python=sys.executable,
        comfy_dir=str(workspace / "ComfyUI"),
        object_info_path=str(workspace / "p1-evidence" / "object_info.json"),
        preflight_path=str(ROOT / "scripts" / "p1" / "preflight.py"),
        deadline_epoch=time.time() + 120,
    )
    evidence, raw = run_actual_probe(tmp_path=tmp_path, plan=plan, mods=mods)
    with run_stub() as (base_url, stub):
        original_candidate_config = v3_readiness._candidate_config

        def candidate_config_with_stub(*, live_dir: Path, source_hashes: dict[str, str]):
            config = original_candidate_config(
                live_dir=live_dir, source_hashes=source_hashes
            )
            config["runtime"]["base_url"] = base_url
            return config

        monkeypatch.setattr(
            v3_readiness, "_candidate_config", candidate_config_with_stub
        )
        candidate = v3_readiness.build_live_candidate(
            run_dir=run,
            bundle=bundle,
            source_hashes=source_hashes,
            binding=context["binding"],
            lifecycle=lifecycle,
            approval=approval,
            quote_observation=observation,
            host_key_fact=host,
            evidence=evidence,
            evidence_raw=raw,
            collector_sha256=evidence["collector"]["sha256"],
            plan_sha256=evidence["collector"]["plan_sha256"],
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
        assert check_readiness(candidate.config_path).live_ready is True
        stub.set_success("result.mp4")
        stub.history[stub.prompt_id]["outputs"] = {
            "58": {"images": [{"filename": "result.mp4", "subfolder": "", "type": "output"}]}
        }
        monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
        monkeypatch.setattr(v3_lifecycle, "_pid_live", lambda pid: True)
        monkeypatch.setattr(v3_lifecycle.CloreSupervisor, "start", _fake_supervisor_start)
        monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
        monkeypatch.setattr(v3_operator, "_produce_remote_export", lambda **kwargs: None)
        result = v3_operator.run_once(
            live=True,
            session_state=context["state_path"],
            order_id=context["state"].order_id,
            created_at=context["state"].created_at,
            remote=context["remote"],
            expected_fingerprint=context["fingerprint"],
            run_dir=run,
            ready_validator=lambda **kwargs: None,
        )
        assert result["status"] == "succeeded", result
        assert stub.prompt_requests == 1
        assert db_scalar(run / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 1
        candidate_payload = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
        assert candidate_payload["candidate"]["runtime_evidence"]["models"][0]["actual_sha256"]
        assert candidate_payload["candidate"]["runtime_evidence"]["object_info"][
            "dynamic_combo"
        ]["selected"] == {
            "format": "mp4",
            "codec": "h264",
            "encoding": "auto",
        }
