from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.kf1.comfy import ComfyClient
from scripts.kf1.errors import AmbiguousDispatch, TerminalRemoteFailure
from scripts.kf1.plan import resolve_plan
from scripts.kf1.runner import Runner
from scripts.kf1.store import PlanStore
from tests.kf1_fake_comfy import FakeComfy

ROOT = Path(__file__).resolve().parents[1]


def _plan():
    return resolve_plan(ROOT / "config/keyframes/two-scenes-v2.json")


def test_full_fake_comfy_pipeline_preserves_a1_and_resume_is_zero_post(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path, plan.plan_sha256)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 3
        assert fake.upload_count == 5
        assert fake.view_count == 3
        state = store.load(plan=plan)
        a1 = state["frames"]["shot-01/A"]["versions"]["1"]
        assert a1["status"] == "PRESERVED"
        assert a1["output"]["sha256"] == plan.refs["first_frame"].sha256
        assert Path(a1["output"]["path"]).read_bytes() == plan.refs["first_frame"].path.read_bytes()
        assert result["status"] == "COMPLETE"
        repeat = runner.run(execute=True, deadline_seconds=10)
        assert repeat["status"] == "COMPLETE"
        assert fake.prompt_count == 3
        assert all(
            item["status"] in {"SKIPPED", "COMPLETE", "PRESERVED"}
            for item in repeat["state"]["frames"]
        )
        client.close()
    finally:
        fake.stop()


def test_ambiguous_no_history_stays_ambiguous_and_never_resubmits(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy(mode="ambiguous_unknown").start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path, plan.plan_sha256)
        runner = Runner(plan, store, client)
        with pytest.raises(AmbiguousDispatch):
            runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 1
        state = store.load(plan=plan)
        assert state["attempts"][0]["status"] == "AMBIGUOUS"
        with pytest.raises(AmbiguousDispatch):
            runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 1
        with pytest.raises(AmbiguousDispatch):
            runner.retry_frame("shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=10)
        assert fake.prompt_count == 2
        state = store.load(plan=plan)
        assert len(store.attempts_for_frame(state, "shot-01/B")) == 2
        client.close()
    finally:
        fake.stop()


def test_ambiguous_but_history_match_recovers_same_prompt_without_resubmit(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy(mode="ambiguous").start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path, plan.plan_sha256)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10)
        # The first POST response was ambiguous, but exact correlation/history
        # recovered the already accepted prompt; no second POST was sent.
        assert result["status"] == "COMPLETE"
        assert fake.prompt_count == 3
        state = store.load(plan=plan)
        assert not [a for a in state["attempts"] if a["status"] == "AMBIGUOUS"]
        client.close()
    finally:
        fake.stop()


def test_terminal_failure_never_auto_retries(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy(mode="terminal_failure").start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path, plan.plan_sha256)
        runner = Runner(plan, store, client)
        with pytest.raises(TerminalRemoteFailure):
            runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 1
        with pytest.raises(TerminalRemoteFailure):
            runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 1
        client.close()
    finally:
        fake.stop()


def test_retry_new_version_invalidates_downstream_and_preserves_old_output(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path, plan.plan_sha256)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10)
        state = store.load(plan=plan)
        old_path = Path(state["frames"]["shot-01/B"]["versions"]["1"]["output"]["path"])
        old_bytes = old_path.read_bytes()
        result = runner.retry_frame("shot-01/B", owner_ack=True, raise_limit=True, deadline_seconds=10)
        assert result["status"] == "COMPLETE"
        assert fake.prompt_count == 4
        state = store.load(plan=plan)
        assert state["frames"]["shot-01/B"]["current_version"] == 2
        assert old_path.read_bytes() == old_bytes
        assert state["frames"]["shot-02/A"]["invalidated"] is True
        assert state["frames"]["shot-02/B"]["invalidated"] is True
        client.close()
    finally:
        fake.stop()


def test_corrupt_png_does_not_complete_and_resume_uses_same_prompt(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy().start()
    fake.output_bytes = b"not-a-png"
    fake.fixed_output = True
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path, plan.plan_sha256)
        runner = Runner(plan, store, client)
        with pytest.raises(Exception):
            runner.run(execute=True, deadline_seconds=10)
        assert fake.prompt_count == 1
        state = store.load(plan=plan)
        assert state["attempts"][0]["status"] == "SUBMITTED"
        fake.output_bytes = __import__("tests.kf1_fake_comfy", fromlist=["_png"])._png(704, 1280)
        result = runner.run(execute=True, deadline_seconds=10)
        assert result["status"] == "COMPLETE"
        state = store.load(plan=plan)
        assert len(store.attempts_for_frame(state, "shot-01/B")) == 1
        assert fake.prompt_count == 3
        client.close()
    finally:
        fake.stop()

def test_generation_limit_is_not_reset_by_plan_edit_or_new_plan_hash(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        (ROOT / "config/keyframes/two-scenes-v2.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plan = resolve_plan(plan_path)
    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path / "state", plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        runner.run(execute=True, deadline_seconds=10)
        assert store.ledger_count() == 3
        from scripts.kf1.errors import GenerationLimitError
        with pytest.raises(GenerationLimitError):
            runner.retry_frame("shot-01/B", owner_ack=True, raise_limit=False, deadline_seconds=10)
        # Editing the plan file (new plan hash) must not reset the attempt ledger.
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        data["seed"] = 123
        plan_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        edited = resolve_plan(plan_path)
        assert edited.plan_sha256 != plan.plan_sha256
        edited_store = PlanStore(tmp_path / "state", edited.plan_sha256, edited.path)
        assert edited_store.ledger_count() == 3
        client.close()
    finally:
        fake.stop()

def test_plan_lock_blocks_concurrent_generate(tmp_path: Path) -> None:
    plan = _plan()
    store = PlanStore(tmp_path, plan.plan_sha256, plan.path)
    with store.lock():
        with pytest.raises(Exception):
            store.lock().acquire()

def test_poll_transient_retries_same_prompt_without_another_post(tmp_path: Path) -> None:
    plan = _plan()
    fake = FakeComfy().start()
    fake.history_failures = 2
    try:
        client = ComfyClient(fake.base_url)
        store = PlanStore(tmp_path, plan.plan_sha256, plan.path)
        runner = Runner(plan, store, client)
        result = runner.run(execute=True, deadline_seconds=10)
        assert result["status"] == "COMPLETE"
        assert fake.prompt_count == 3
        state = store.load(plan=plan)
        assert len(store.attempts_for_frame(state, "shot-01/B")) == 1
        client.close()
    finally:
        fake.stop()
