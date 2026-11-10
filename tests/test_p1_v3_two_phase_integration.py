from __future__ import annotations

import copy
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx

from scripts.p1 import clore_session as cs
from scripts.p1 import marketplace_quote, owner_risk, v3_lifecycle, v3_operator, v3_readiness, v3_remote_launcher, v3_supervisor_child

from scripts.p1.clore_bridge import CloreCancelAck, CloreOrder
from scripts.p1.remote import RemoteIdentity
from tests.comfy_stub import MP4_FIXTURE, run_stub
from tests.helpers import db_scalar
from tests.p1_probe_helpers import make_probe_workspace, model_facts_for
from tests.test_p1_v3_bootstrap import _FakeTunnel
from scripts.p1 import v3_bootstrap

ROOT = Path(__file__).resolve().parents[1]
QUOTE_PATH = ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "quote-ready-example.json"
SUPERVISOR_STARTS: list[str] = []


def _identity(tmp_path: Path) -> tuple[RemoteIdentity, str]:
    identity = tmp_path / "id"
    known = tmp_path / "known_hosts"
    identity.write_text("not-a-key", encoding="utf-8")
    blob = b"integration-fake-boundary"
    known.write_text(
        f"[example.test]:22 ssh-ed25519 {__import__('base64').b64encode(blob).decode()}\n",
        encoding="utf-8",
    )
    fingerprint = "SHA256:" + __import__("base64").b64encode(
        hashlib.sha256(blob).digest()
    ).decode().rstrip("=")
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


def _owner_risk_quote(path: Path, now: datetime) -> Path:
    payload = {
        "observed_at": now.astimezone(timezone.utc).isoformat(),
        "observation_origin": "owner_forwarded_official_response_not_direct_account_audit",
        "provenance": {
            "kind": "owner_forwarded_official_response",
            "not_provider_api_audit": True,
        },
        "quote": {
            "price_per_day": "12",
            "price_per_hour": "0.50",
            "price_per_minute": "0.008333333333333333333333333333",
            "currency": "USD-Blockchain",
            "creation_fee": "0.10",
            "mandatory_other_fees": None,
            "reserve": "0.10",
            "fresh_verified": True,
            "fees_known": False,
            "tax_fx_known": False,
            "min_rental_hours": None,
            "provider_fee_percent": "5",
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _scoped_quote(path: Path, now: datetime) -> Path:
    payload = json.loads(
        (ROOT / "tests" / "fixtures" / "p1" / "clore-v1" / "marketplace.json").read_text(
            encoding="utf-8"
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://api.clore.ai/v1/marketplace"
        return httpx.Response(200, json=payload)

    asyncio.run(
        marketplace_quote.collect_marketplace_quote(
            server_id=120777,
            output_path=path,
            transport=httpx.MockTransport(handler),
            now=now,
            clock=lambda: 0.0,
            sleep=lambda seconds: asyncio.sleep(0),
        )
    )
    return path


class LocalRemoteRunner:

    """Fake SSH boundary executing only allowlisted local fixture actions."""

    def __init__(self, *, remote_root: str, local_root: Path, artifact_path: Path):
        self.remote_root = remote_root
        self.local_root = local_root
        self.artifact_path = artifact_path
        self.commands: list[tuple[str, ...]] = []
        self.puts: list[str] = []

    def _local(self, remote_path: str) -> Path:
        text = str(remote_path).replace("\\", "/")
        root = self.remote_root.rstrip("/")
        if text == root:
            return self.local_root
        prefix = root + "/"
        if not text.startswith(prefix):
            raise AssertionError(f"remote path outside owned root: {remote_path}")
        return self.local_root / Path(text[len(prefix):])

    def put_file(self, identity, remote_path, local_path, **kwargs):
        self.puts.append(str(remote_path))
        target = self._local(str(remote_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != Path(local_path).read_bytes():
            if target.name == ".p1-owned-root.json":
                raise RuntimeError("foreign root")
        target.write_bytes(Path(local_path).read_bytes())

    def read_remote_exact_json(self, identity, *, helper_path, remote_path, root, max_bytes, timeout, **kwargs):
        return json.loads(self._local(str(remote_path)).read_text(encoding="utf-8"))

    def run_remote(self, identity, args, timeout=None, **kwargs):
        args = tuple(str(item) for item in args)
        self.commands.append(args)
        if args[0] == "mkdir":
            return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)
        if args[0] == "df":
            return SimpleNamespace(
                returncode=0,
                stdout="Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 100000000 1 99999999 1% /workspace\n",
                stderr="",
                timed_out=False,
            )
        if args[0] == "python3" and "os.execvp" in args[2]:
            return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)
        if "v3_remote_launcher.py" in args[1]:
            launch = json.loads(self._local(args[3]).read_text(encoding="utf-8"))
            command_hash = v3_remote_launcher.manifest_hash(launch)
            root = self._local(launch["root"])
            if launch["kind"] == "download_worker":
                owner = {
                    "schema": "anime-video-studio/p1-v3-process-owner/v1",
                    "kind": "download_worker",
                    "run_id": launch["run_id"],
                    "nonce": launch["nonce"],
                    "command_sha256": command_hash,
                    "pid": 5151,
                }
                for name, value in (
                    ("worker-owner.json", owner),
                    ("worker-ack.json", owner),
                    ("status.json", {"run_id": launch["run_id"], "nonce": launch["nonce"], "status": "COMPLETE"}),
                    (
                        "progress.json",
                        {
                            "run_id": launch["run_id"],
                            "nonce": launch["nonce"],
                            "status": "COMPLETE",
                            "files_done": 3,
                            "files_total": 3,
                            "bytes_done": 10,
                            "bytes_total": 10,
                        },
                    ),
                    (
                        "heartbeat.json",
                        {
                            "run_id": launch["run_id"],
                            "nonce": launch["nonce"],
                            "status": "COMPLETE",
                            "heartbeat_at": time.time(),
                        },
                    ),
                ):
                    (root / name).write_text(json.dumps(value), encoding="utf-8")
            else:
                owner = {
                    "schema": "anime-video-studio/p1-v3-process-owner/v1",
                    "kind": "comfy",
                    "run_id": launch["run_id"],
                    "nonce": launch["nonce"],
                    "command_sha256": command_hash,
                    "pid": 6161,
                }
                (root / "comfy-owner.json").write_text(json.dumps(owner), encoding="utf-8")
                (root / "comfy-ack.json").write_text(json.dumps(owner), encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema": "anime-video-studio/p1-v3-launch-result/v1",
                        "kind": launch["kind"],
                        "run_id": launch["run_id"],
                        "nonce": launch["nonce"],
                        "status": "STARTED",
                        "command_sha256": command_hash,
                        "pid": 5151 if launch["kind"] == "download_worker" else 6161,
                    }
                ),
                stderr="",
                timed_out=False,
            )
        if "v3_preflight_runner.py" in args[1]:
            return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)
        if "v3_runtime_probe.py" in args[1]:
            plan_path = self._local(args[-1])
            probe_path = self._local(args[1])
            venv_path = self._local(args[0])
            env = dict(__import__("os").environ)
            env["PYTHONPATH"] = str(self.local_root / "ComfyUI" / "site-packages")
            result = subprocess.run(
                [str(venv_path), str(probe_path), "--plan", str(plan_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=120,
                check=False,
                cwd=str(self.local_root),
            )
            if result.returncode != 0:
                raise AssertionError(result.stderr or result.stdout)
            return SimpleNamespace(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
            )
        if "v3_remote_fetch_helper.py" in args[1] and "metadata" in args:
            data = self.artifact_path.read_bytes()
            payload = {
                "schema": "anime-video-studio/p1-v3-metadata/v1",
                "path": str(args[3]),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
                timed_out=False,
            )
        raise AssertionError(f"unknown fake SSH command: {args}")


def _fake_supervisor_start(self, **kwargs):
    from scripts.p1.v3_lifecycle import SupervisorRecord

    SUPERVISOR_STARTS.append(self.nonce)
    self.config_path.write_text(json.dumps(self._config_payload()), encoding="utf-8")
    config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
    binding_sha256 = hashlib.sha256(
        json.dumps(self.binding.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
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


@pytest.mark.parametrize("risk_mode", ["strict", "legacy_risk", "scoped_risk"])
def test_two_phase_prepare_runonce_collect_real_coordinator_fake_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_mode: str
) -> None:
    name = f"p1int-{uuid.uuid4().hex[:10]}"
    workspace = f"/{name}"
    local_root = Path(workspace)
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True)
    try:
        context_bundle = dict(v3_operator.load_bundle())
        targets = [model["target"] for model in context_bundle["manifest"]["models"]]
        extra_inputs: dict[str, set[str]] = {}
        for node_id, class_name in context_bundle["config"].binding_raw["node_types"].items():
            for input_name in context_bundle["workflow"][str(node_id)]["inputs"]:
                text = str(input_name)
                if class_name == "SaveVideo" and (text == "format" or text.startswith("format.")):
                    continue
                extra_inputs.setdefault(class_name, set()).add(text)
        probe_workspace, _site_packages, commit = make_probe_workspace(
            local_root,
            targets,
            extra_inputs={key: sorted(value) for key, value in extra_inputs.items()},
        )
        # The helper creates <root>/probe-workspace; make it the actual owned root.
        inner = local_root / "probe-workspace"
        for child in list(inner.iterdir()):
            shutil.move(str(child), str(local_root / child.name))
        inner.rmdir()
        venv_dir = local_root / "p1-venv" / "bin"
        venv_dir.mkdir(parents=True)
        (venv_dir / "python").symlink_to(sys.executable)
        model_facts = model_facts_for(local_root, targets)
        manifest = copy.deepcopy(context_bundle["manifest"])
        for model, fact in zip(manifest["models"], model_facts):
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
        source_hashes = dict(v3_operator._bundle_hashes(context_bundle))
        source_hashes["manifest"] = manifest_hash
        monkeypatch.setitem(v3_readiness.FROZEN_SOURCE_BUNDLE, "manifest", manifest_hash)
        monkeypatch.setitem(v3_readiness.SOURCE_PATHS, "manifest", str(manifest_path))
        monkeypatch.setattr(v3_operator, "V3_MANIFEST", manifest_path)
        bundle = dict(context_bundle)
        bundle["manifest"] = manifest
        monkeypatch.setattr(v3_operator, "load_bundle", lambda: bundle)
        monkeypatch.setattr(v3_operator, "_bundle_hashes", lambda bundle=None: source_hashes)

        now = datetime.now(timezone.utc)
        state_path = tmp_path / "session.json"
        if risk_mode == "legacy_risk":
            quote_path = _owner_risk_quote(tmp_path / "owner-risk-quote.json", now)
            quote = cs.CloreQuote.from_dict(
                json.loads(quote_path.read_text(encoding="utf-8"))["quote"]
            )
            monkeypatch.setattr(
                owner_risk, "DEFAULT_REGISTRY_ROOT", tmp_path / "owner-risk-registry"
            )
            admission = owner_risk.create_admission(
                quote=quote_path,
                output_path=tmp_path / "owner-risk-admission.json",
                now=now,
            )
            state = cs.initialize_state(
                state_path,
                order_id=99001,
                server_id=111947,
                created_at=int(now.timestamp()) - 5,
                quote=quote,
                owner_risk_admission=admission,
                now=now,
            )
        elif risk_mode == "scoped_risk":
            quote_path = _scoped_quote(tmp_path / "scoped-quote.json", now)
            quote = cs.CloreQuote.from_dict(
                json.loads(quote_path.read_text(encoding="utf-8"))["quote"]
            )
            monkeypatch.setattr(
                owner_risk, "DEFAULT_REGISTRY_ROOT", tmp_path / "scoped-risk-registry"
            )
            admission = owner_risk.create_scoped_admission(
                quote=quote_path,
                output_path=tmp_path / "scoped-admission.json",
                server_id=120777,
                hoster_id=47111,
                approval_confirm=owner_risk.APPROVAL_CONFIRMATION,
                now=now,
            )
            state = cs.initialize_state(
                state_path,
                order_id=99001,
                server_id=120777,
                created_at=int(now.timestamp()) - 5,
                quote=quote,
                owner_risk_admission=admission,
                now=now,
            )
        else:
            quote_path = QUOTE_PATH
            quote = cs.CloreQuote.from_dict(
                json.loads(QUOTE_PATH.read_text(encoding="utf-8"))
            )
            state = cs.initialize_state(
                state_path,
                order_id=99001,
                created_at=int(now.timestamp()) - 5,
                quote=quote,
                now=now,
            )
        binding = v3_lifecycle.SessionBinding.from_state(state)
        approval = v3_readiness.create_source_approval(
            bundle=bundle,
            source_hashes=source_hashes,
            output_path=tmp_path / "source-approval.json",
        )
        observation = v3_readiness.create_quote_observation(
            quote_path=quote_path, session_state=state_path, binding=binding
        )
        remote, fingerprint = _identity(tmp_path)
        host_key_fact = v3_readiness.make_host_key_fact(remote, fingerprint)
        artifact = local_root / "ComfyUI" / "output" / "result.mp4"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(MP4_FIXTURE)
        source_snapshot_files = {
            path: path.read_bytes()
            for path in (
                manifest_path,
                ROOT / "config/p1.wan22-5b.shot01-v3.draft.json",
                ROOT / "workflows/P1/wan22-5b-shot01-v3.api.draft.json",
            )
        }
        runner = LocalRemoteRunner(
            remote_root=workspace, local_root=local_root, artifact_path=artifact
        )
        run = tmp_path / "run"
        with run_stub() as (base_url, stub):
            stub.set_success("result.mp4")
            stub.history[stub.prompt_id]["outputs"] = {
                "58": {"images": [{"filename": "result.mp4", "subfolder": "", "type": "output"}]}
            }
            original_candidate_config = v3_readiness._candidate_config

            def candidate_config_with_stub(*, live_dir: Path, source_hashes: dict[str, str]):
                config = original_candidate_config(live_dir=live_dir, source_hashes=source_hashes)
                config["runtime"]["base_url"] = base_url
                return config

            monkeypatch.setattr(v3_readiness, "_candidate_config", candidate_config_with_stub)
            monkeypatch.setattr(v3_lifecycle, "verify_host_key", lambda **kwargs: None)
            monkeypatch.setattr(v3_lifecycle, "_pid_live", lambda pid: True)
            monkeypatch.setattr(v3_lifecycle.CloreSupervisor, "start", _fake_supervisor_start)
            monkeypatch.setattr(v3_operator, "SESSION_OWNER_ROOT", tmp_path / "owners")
            SUPERVISOR_STARTS.clear()
            prepared = v3_operator.prepare(
                live=True,
                execute_downloads=True,
                workspace=workspace,
                run_dir=run,
                session_state=state_path,
                order_id=state.order_id,
                created_at=state.created_at,
                source_approval=approval.path,
                remote=remote,
                expected_fingerprint=fingerprint,
                remote_runner=runner,
                tunnel_factory=_FakeTunnel,
            )
            assert prepared["status"] == "PREPARED_READY", prepared
            lifecycle = v3_lifecycle.load_lifecycle(run / "lifecycle.json")
            assert lifecycle.phase == "GENERATION_READY"
            if risk_mode in ("legacy_risk", "scoped_risk"):
                assert (
                    prepared["financial_status"]
                    == "OWNER_ACCEPTED_FINANCIAL_RISK"
                )
                risk_state = cs.load_state(state_path)
                assert risk_state.owner_risk is not None
                assert (
                    lifecycle.session.owner_risk_admission_sha256
                    == risk_state.owner_risk_sha256
                )
            nonce = lifecycle.nonce
            deadline = (lifecycle.session.planned_cancel_at, lifecycle.session.hard_end_at)
            candidate_path = run / "live" / "candidate-binding.json"
            candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            source_snapshot = dict(lifecycle.bundle_hashes)
            probe_commands = [c for c in runner.commands if "v3_runtime_probe.py" in c[1]]
            assert len(probe_commands) == 1
            assert len([p for p in runner.puts if p.endswith("/launch.json")]) == 1
            assert len(SUPERVISOR_STARTS) == 1

            # Use the existing fake tunnel class for READY revalidation.
            monkeypatch.setattr(
                v3_bootstrap.OwnedTunnel,
                "attach",
                classmethod(lambda cls, **kwargs: _FakeTunnel(**kwargs)),
            )
            result = v3_operator.run_once(
                live=True,
                session_state=state_path,
                order_id=state.order_id,
                created_at=state.created_at,
                remote=remote,
                expected_fingerprint=fingerprint,
                run_dir=run,
                remote_runner=runner,
            )
            assert result["status"] == "succeeded", result
            assert stub.prompt_requests == 1
            assert db_scalar(run / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 1
            deadline_before = db_scalar(run / "jobs.sqlite3", "SELECT deadline_ts FROM jobs LIMIT 1")
            second = v3_operator.run_once(
                live=True,
                session_state=state_path,
                order_id=state.order_id,
                created_at=state.created_at,
                remote=remote,
                expected_fingerprint=fingerprint,
                run_dir=run,
                remote_runner=runner,
            )
            assert second["status"] == "succeeded"
            assert stub.prompt_requests == 1
            assert db_scalar(run / "jobs.sqlite3", "SELECT COUNT(*) FROM jobs") == 1
            assert db_scalar(run / "jobs.sqlite3", "SELECT deadline_ts FROM jobs LIMIT 1") == deadline_before
            assert len([c for c in runner.commands if "v3_runtime_probe.py" in c[1]]) == 1
            assert len([p for p in runner.puts if p.endswith("/launch.json")]) == 1
            assert len(SUPERVISOR_STARTS) == 1
            assert {
                path: path.read_bytes() for path in source_snapshot_files
            } == source_snapshot_files
            lifecycle_after = v3_lifecycle.load_lifecycle(run / "lifecycle.json")
            assert lifecycle_after.nonce == nonce
            assert (lifecycle_after.session.planned_cancel_at, lifecycle_after.session.hard_end_at) == deadline
            assert dict(lifecycle_after.bundle_hashes) == source_snapshot
            assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == candidate_hash
            assert cs.load_state(state_path).billing_verified is False
            collected = v3_operator.collect(run_dir=run)
            assert collected["status"].startswith("COLLECTED")
            if risk_mode in ("legacy_risk", "scoped_risk"):
                runtime_payloads = []
                for candidate in run.rglob("runtime.json"):
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    if payload.get("schema") == "anime-video-studio/p1-v3-runtime/v1":
                        runtime_payloads.append(payload)
                assert runtime_payloads
                assert (
                    runtime_payloads[-1]["financial_status"]
                    == "OWNER_ACCEPTED_FINANCIAL_RISK"
                )
                assert runtime_payloads[-1]["provider_billing_verified"] is False

            live_state = cs.load_state(state_path)
            quote = live_state.quote
            order = CloreOrder(
                order_id=live_state.order_id,
                server_id=live_state.server_id,
                created_at=live_state.created_at,
                    price_per_day=quote.price_per_day or quote.rental_per_minute() * Decimal("1440"),
                    currency=live_state.currency,
                    expired=False,
                    fee_percent=quote.provider_fee_percent or Decimal("5"),
                    creation_fee=quote.creation_fee or Decimal("0.10"),

                spend=Decimal("0"),
                spot=False,
                mrl_seconds=3600,
            )
            expired_order = replace(order, expired=True)
            provider = _TypedFakeProvider(order, expired_order)
            clock = _FakeClock(
                datetime.fromtimestamp(
                    live_state.planned_cancel_at - 1, tz=timezone.utc
                )
            )
            real_session_class = cs.CloreSession

            def controlled_session(state_file, *, bridge, **kwargs):
                return real_session_class(
                    state_file,
                    bridge=bridge,
                    clock=clock.clock,
                    monotonic=clock.monotonic,
                    sleep=clock.sleep,
                    readback_timeout_seconds=30.0,
                    readback_poll_seconds=1.0,
                )

            supervisor = v3_lifecycle.CloreSupervisor(
                session_state_path=state_path,
                quote_path=run / "session-quote.json",
                binding=binding,
                log_path=run / "supervisor.log",
                record_path=run / "supervisor.json",
                run_id=run.name,
                nonce=nonce,
            )
            config = supervisor._config_payload()
            monkeypatch.setattr(
                v3_supervisor_child,
                "_load_bridge",
                lambda expected_server_id=None: provider,
            )
            monkeypatch.setattr(
                v3_supervisor_child.clore_session, "CloreSession", controlled_session
            )
            monkeypatch.setattr(v3_supervisor_child.asyncio, "sleep", clock.sleep)
            assert asyncio.run(v3_supervisor_child._watch(config)) == 0
            terminal_ack = json.loads(supervisor.ack_path.read_text(encoding="utf-8"))
            assert terminal_ack["status"] == "ENDED_REPORTED"
            assert not supervisor.heartbeat_path.exists()
            terminal_state = cs.load_state(state_path)
            assert terminal_state.status == "ENDED_REPORTED"
            assert terminal_state.billing_verified is False
            assert terminal_state.planned_cancel_at == live_state.planned_cancel_at
            assert terminal_state.hard_end_at == live_state.hard_end_at
            assert provider.cancel_calls == 1
            assert provider.cancel_requests == [
                (live_state.order_id, live_state.server_id, live_state.created_at)
            ]
            assert provider.read_ids and all(
                order_id == live_state.order_id for order_id in provider.read_ids
            )
            with pytest.raises(v3_lifecycle.LifecycleError):
                supervisor.readback(deadline_monotonic=time.monotonic() + 5.0)
            read_count = len(provider.read_ids)
            recovery = real_session_class(
                state_path,
                bridge=provider,
                clock=clock.clock,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                readback_timeout_seconds=30.0,
                readback_poll_seconds=1.0,
            )
            recovered = asyncio.run(recovery.evaluate())
            assert recovered.status == "ENDED_REPORTED"
            assert recovered.billing_verified is False
            assert provider.cancel_calls == 1
            assert len(provider.read_ids) == read_count


    finally:
        if local_root.exists():
            shutil.rmtree(local_root, ignore_errors=True)



def test_prepare_missing_approval_has_zero_external_actions(tmp_path: Path) -> None:
    quote = cs.CloreQuote.from_dict(json.loads(QUOTE_PATH.read_text(encoding="utf-8")))
    now = datetime.now(timezone.utc)
    state_path = tmp_path / "session.json"
    state = cs.initialize_state(
        state_path,
        order_id=99002,
        created_at=int(now.timestamp()) - 5,
        quote=quote,
        now=now,
    )
    remote, fingerprint = _identity(tmp_path)

    class CountingRunner:
        def __init__(self):
            self.commands = []
            self.puts = []

        def run_remote(self, *args, **kwargs):
            self.commands.append(args)
            raise AssertionError("external action is not allowed")

        def put_file(self, *args, **kwargs):
            self.puts.append(args)
            raise AssertionError("external action is not allowed")

    runner = CountingRunner()
    run = tmp_path / "must-not-exist"
    with pytest.raises(v3_operator.OperatorError):
        v3_operator.prepare(
            live=True,
            execute_downloads=True,
            run_dir=run,
            session_state=state_path,
            order_id=state.order_id,
            created_at=state.created_at,
            remote=remote,
            expected_fingerprint=fingerprint,
            remote_runner=runner,
        )
    assert not run.exists()
    assert runner.commands == []
    assert runner.puts == []

class _FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start
        self.mono = 0.0

    def clock(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.mono

    async def sleep(self, seconds: float) -> None:
        self.now = self.now + __import__("datetime").timedelta(seconds=seconds)
        self.mono += seconds


class _TypedFakeProvider:
    def __init__(self, order: CloreOrder, expired: CloreOrder) -> None:
        self.order = order
        self.expired = expired
        self.read_ids: list[int] = []
        self.read_identities: list[tuple[int, int, int]] = []
        self.cancel_requests: list[tuple[int, int, int]] = []
        self.cancel_calls = 0
        self.ack = CloreCancelAck(0, True, False, "cancel accepted")

    async def read_order(self, order_id: int, *, deadline_monotonic: float | None = None):
        self.read_ids.append(order_id)
        result = self.expired if self.cancel_calls else self.order
        self.read_identities.append((result.order_id, result.server_id, result.created_at))
        if result.order_id != order_id:
            raise AssertionError("provider read used a foreign order id")
        return result

    async def cancel_order(self, order_id: int, *, deadline_monotonic: float | None = None):
        self.cancel_calls += 1
        self.cancel_requests.append(
            (order_id, self.order.server_id, self.order.created_at)
        )
        if order_id != self.order.order_id:
            raise AssertionError("provider cancel used a foreign order id")
        return self.ack

    async def aclose(self) -> None:
        return None
