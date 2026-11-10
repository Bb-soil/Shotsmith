from __future__ import annotations

import json

from pathlib import Path
from types import SimpleNamespace

from scripts.kf1.comfy import ComfyClient
from scripts.kf1.plan import resolve_plan
from scripts.kf1.runner import Runner
from scripts.kf1.store import PlanStore
from scripts.kf2.cli import _add_image_to_project, _cmd_prepare, _save
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import FakeComfy, _png


def test_prepare_keeps_existing_approval_semantics_and_creates_no_approval(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("p1", title="Approval", orientation="portrait")
    (tmp_path / "first.png").write_bytes(_png(704, 1280))
    (tmp_path / "object.png").write_bytes(_png(64, 64))
    first = _add_image_to_project(
        project, project_dir, role="provided_keyframe", path=str(tmp_path / "first.png")
    )
    identity = _add_image_to_project(
        project,
        project_dir,
        role="object",
        path=str(tmp_path / "object.png"),
        subject_id="car",
        subject_label="car",
    )
    project["scenes"] = [
        {
            "id": "shot-01",
            "description": "A car on a road.",
            "action": "",
            "camera": "Fixed.",
            "subjects_visible": ["car"],
            "subjects_absent": [],
            "shared_invariants": ["identity"],
            "shared_allowed_changes": ["position"],
            "initial": {
                "kind": "first_frame",
                "asset_id": first["id"],
                "instruction": "",
                "references": [],
                "invariants": [],
                "allowed_changes": [],
                "use_previous_anchor": False,
                "require_anchor_approval": True,
            },
            "final": {
                "kind": "continue",
                "asset_id": None,
                "instruction": "The car approaches.",
                "references": [{"asset_id": identity["id"], "role": "identity"}],
                "invariants": [],
                "allowed_changes": [],
                "require_anchor_approval": False,
            },
        }
    ]
    scene2 = json.loads(json.dumps(project["scenes"][0]))
    scene2["id"] = "shot-02"
    scene2["initial"] = {
        "kind": "new_shot",
        "asset_id": None,
        "instruction": "New camera.",
        "references": [{"asset_id": identity["id"], "role": "identity"}],
        "invariants": ["identity"],
        "allowed_changes": ["camera", "position"],
        "use_previous_anchor": False,
        "require_anchor_approval": False,
    }
    scene2["final"] = {
        "kind": "continue",
        "asset_id": None,
        "instruction": "Keep the new camera.",
        "references": [{"asset_id": identity["id"], "role": "identity"}],
        "invariants": [],
        "allowed_changes": [],
        "require_anchor_approval": True,
    }
    project["scenes"].append(scene2)
    project = _save(store, project)
    state = tmp_path / "kf1-state"
    code, payload = _cmd_prepare(
        SimpleNamespace(
            id="p1",
            projects_dir=str(tmp_path / "projects"),
            kf1_state_dir=str(state),
            json=True,
        )
    )
    assert code == 0
    assert payload["prepared"]["approval_required_frames"] == ["shot-02/B"]
    plan = resolve_plan(project_dir / "plan.json")
    assert plan.frames[1].require_anchor_approval is False
    assert plan.frames[3].require_anchor_approval is True
    assert not list(plan.path.parent.rglob("approval.json"))

    fake = FakeComfy().start()
    try:
        client = ComfyClient(fake.base_url)
        store_kf1 = PlanStore(state, plan.plan_sha256, plan.path)
        first = Runner(plan, store_kf1, client).run(
            execute=True, deadline_seconds=10, stop_after="shot-02/A"
        )
        assert first["status"] == "STOPPED_AFTER"
        posted = fake.prompt_count
        assert posted == 2
        second = Runner(plan, store_kf1, client).run(execute=True, deadline_seconds=10)
        assert second["status"] == "WAITING_APPROVAL"
        assert fake.prompt_count == posted
        client.close()
    finally:
        fake.stop()
