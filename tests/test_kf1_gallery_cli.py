from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.kf1 import cli
from scripts.kf1.comfy import ComfyClient
from scripts.kf1.errors import ReleaseDeferred
from scripts.kf1.gallery import export_storyboard
from scripts.kf1.plan import resolve_plan
from scripts.kf1.release import release_plan
from scripts.kf1.runner import Runner
from scripts.kf1.store import PlanStore
from tests.kf1_fake_comfy import FakeComfy

ROOT = Path(__file__).resolve().parents[1]


def _plan():
    return resolve_plan(ROOT / "config/keyframes/two-scenes-v2.json")


def _run(tmp_path: Path):
    plan = _plan()
    fake = FakeComfy().start()
    client = ComfyClient(fake.base_url)
    store = PlanStore(tmp_path, plan.plan_sha256)
    runner = Runner(plan, store, client)
    runner.run(execute=True, deadline_seconds=10)
    return plan, fake, client, store, runner


def test_gallery_is_local_escaped_and_marks_not_reviewed(tmp_path: Path) -> None:
    plan, fake, client, store, _runner = _run(tmp_path)
    try:
        plan.data["scenes"][0]["description"] = "<script>alert('x')</script>"
        manifest = export_storyboard(plan, store.load(plan=plan), tmp_path / "export")
        assert manifest["visual_status"] == "NOT_REVIEWED"
        assert manifest["video_submitted"] is False
        html = (tmp_path / "export" / "index.html").read_text(encoding="utf-8")
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
        assert "http://" not in html and "https://" not in html
        assert "assets/shot-01/A-v0001.png" in html
        role = manifest["scenes"][0]["roles"][0]
        assert role["wan_width"] == 448 and role["wan_height"] == 816
        assert (tmp_path / "export" / role["wan_path"]).is_file()
        assert (tmp_path / "export" / "storyboard.json").is_file()
        # no remote JS/service worker
        assert "serviceWorker" not in html
    finally:
        client.close()
        fake.stop()


def test_release_requires_empty_queue_and_terminal_owned_prompts(tmp_path: Path) -> None:
    plan, fake, client, store, _runner = _run(tmp_path)
    try:
        state = store.load(plan=plan)
        fake.queue_pending = [["id", "foreign"]]
        with pytest.raises(ReleaseDeferred):
            release_plan(plan, state, client, execute=True)
        assert fake.free_count == 0
        fake.queue_pending = []
        result = release_plan(plan, state, client, execute=True)
        assert result["status"] == "RELEASE_REQUESTED"
        assert result["vram_free_verified"] is False
        assert fake.free_count == 1
    finally:
        client.close()
        fake.stop()


def test_cli_generate_default_is_offline(tmp_path: Path, capsys) -> None:
    code = cli.main(
        [
            "generate",
            "--plan",
            str(ROOT / "config/keyframes/two-scenes-v2.json"),
            "--state-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "PLAN_ONLY"
    assert payload["post_plan"] == ["shot-01/B", "shot-02/A", "shot-02/B"]


def test_cli_plan_example_and_text_no_json_requirement(tmp_path: Path, capsys) -> None:
    code = cli.main(["plan", "--example", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "PLAN_READY"
    text = tmp_path / "plan with spaces.txt"
    text.write_text(
        "\n".join(
            [
                "title: План с пробелами",
                "first_frame: assets/frames/derived/shot-01-start-v3-704x1280.png",
                "character_reference: assets/references/character-sheet.png",
                "scene: shot-01",
                "description: Русское описание",
                "a_source: first_frame",
                "b_source: generate_from_scene_a",
                "b_edit: Keep the exact camera.",
            ]
        ),
        encoding="utf-8",
    )
    saved = tmp_path / "saved plan.json"
    code = cli.main(
        ["plan", "--from-text", str(text), "--save", str(saved), "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "PLAN_READY"
    assert saved.is_file()

def test_cli_execute_uses_fake_http_and_argv_safe_path(tmp_path: Path, capsys) -> None:
    fake = FakeComfy().start()
    try:
        state_dir = tmp_path / "state with spaces"
        plan = tmp_path / "plan with spaces.json"
        plan.write_text(
            (ROOT / "config/keyframes/two-scenes-v2.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        code = cli.main(
            [
                "generate",
                "--plan",
                str(plan),
                "--state-dir",
                str(state_dir),
                "--execute",
                "--base-url",
                fake.base_url,
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["status"] == "COMPLETE"
        assert fake.prompt_count == 3
        code = cli.main(
            [
                "status",
                "--plan",
                str(plan),
                "--state-dir",
                str(state_dir),
                "--json",
            ]
        )
        assert code == 0
        status = json.loads(capsys.readouterr().out.strip())
        assert status["attempts_used"] == 3
        assert status["offline"] is True
    finally:
        fake.stop()
