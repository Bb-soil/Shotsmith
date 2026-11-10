from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.kf2.cli import _add_image_to_project, _cmd_prepare, _menu, _save
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import _png


def test_local_wizard_resume_edit_prepare_and_kf1_dryrun(tmp_path: Path) -> None:
    """Actual local wizard path: images, add scene, resume, preview, prepare."""

    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("demo", title="Локальная история", orientation="portrait")
    (tmp_path / "start frame.png").write_bytes(_png(704, 1280, (20, 30, 40)))
    (tmp_path / "car view.png").write_bytes(_png(64, 64, (80, 90, 100)))
    first = _add_image_to_project(
        project,
        project_dir,
        role="provided_keyframe",
        path=f'"{tmp_path / "start frame.png"}"',
    )
    identity = _add_image_to_project(
        project,
        project_dir,
        role="object",
        path=f'"{tmp_path / "car view.png"}"',
        subject_id="car",
        subject_label="машина",
        view="front",
    )
    project = _save(store, project)

    answers = iter(
        [
            "2",
            "shot-01",
            "1",
            "Машина стоит на дороге.",
            "---",
            "2",
            "Машина медленно приближается.",
            "---",
            "3",
            "Камера неподвижна.",
            "---",
            "4",
            "car",
            "5",
            "",
            "6",
            "identity,style",
            "7",
            "position,apparent_scale",
            "8",
            first["id"],
            "",
            "9",
            "continue",
            "Та же машина приближается к камере.",
            "---",
            "",
            "",
            "n",
            f"{identity['id']}:identity",
            "0",
            "6",
        ]
    )

    def fake_input(prompt: str) -> str:
        return next(answers)

    printed: list[str] = []
    project = _menu(store, project, project_dir, input_fn=fake_input, print_fn=printed.append)
    assert any(scene["id"] == "shot-01" for scene in project["scenes"])
    assert project["scenes"][0]["final"]["instruction"] == "Та же машина приближается к камере."

    # Resume/edit path must be able to save again without forcing prepare.
    project = store.load("demo")
    project["title"] = "Локальная история — edit"
    project = store.save("demo", project, expected_revision=project["revision"])
    assert project["status"] == "DRAFT"

    args = SimpleNamespace(
        id="demo",
        projects_dir=str(tmp_path / "projects"),
        kf1_state_dir=str(tmp_path / "kf1-state"),
        json=True,
    )
    code, payload = _cmd_prepare(args)
    assert code == 0
    assert payload["status"] == "PREPARED"
    assert payload["dryrun"]["status"] == "PLAN_ONLY"
    plan_path = project_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == "anime-studio/keyframe-plan/v2"
    assert plan["scenes"][0]["b"]["require_anchor_approval"] is False
    assert plan["output_policy"]["candidate_geometry"] == [704, 1280]
