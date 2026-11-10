from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kf2 import assets as assets_mod
from scripts.kf2 import cli
from scripts.kf2.assets import decode_image_bytes, MAX_PIXELS
from scripts.kf2.errors import AssetError, ProjectError
from scripts.kf2.paths import resolve_local_source
from scripts.kf2.store import ProjectStore
from tests.kf1_fake_comfy import _png


def _project(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("p1", title="Проект", orientation="portrait")
    return store, project, project_dir


def _valid_project(tmp_path: Path):
    store, project, project_dir = _project(tmp_path)
    (tmp_path / "first.png").write_bytes(_png(704, 1280))
    (tmp_path / "id.png").write_bytes(_png(64, 64))
    first = cli._add_image_to_project(
        project, project_dir, role="provided_keyframe", path=str(tmp_path / "first.png")
    )
    identity = cli._add_image_to_project(
        project, project_dir, role="object", path=str(tmp_path / "id.png"),
        subject_id="car", subject_label="машина", view="front",
    )
    project["scenes"] = [cli._blank_scene(0, "shot-01")]
    project["scenes"][0]["description"] = "Машина на дороге."
    project["scenes"][0]["final"]["instruction"] = "Машина приближается."
    project["scenes"][0]["initial"]["asset_id"] = first["id"]
    project["scenes"][0]["final"]["references"] = [{"asset_id": identity["id"], "role": "identity"}]
    project = cli._save(store, project)
    return store, project, project_dir


def _junction(link: Path, target: Path) -> bool:
    if os.name != "nt":
        return False
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0


def test_history_junction_is_rejected_without_outside_write(tmp_path: Path) -> None:
    store, project, project_dir = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    history = project_dir / "history"
    shutil.rmtree(history)
    if not _junction(history, outside):
        pytest.skip("junction unavailable")
    with pytest.raises(ProjectError):
        store.save("p1", project, expected_revision=project["revision"])
    assert list(outside.iterdir()) == []


def test_provenance_parent_junction_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    provenance = outside / "p.json"
    provenance.write_text("{}", encoding="utf-8")
    link = tmp_path / "linked-dir"
    if not _junction(link, outside):
        pytest.skip("junction unavailable")
    try:
        with pytest.raises(AssetError):
            resolve_local_source(str(link / "p.json"), label="provenance", allowed_suffixes={".json"})
    finally:
        subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True, timeout=30)


def test_mapped_remote_drive_gate_is_mockable(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "image.png"
    source.write_bytes(_png(8, 8))
    monkeypatch.setattr("scripts.kf2.paths.drive_type_is_remote", lambda _path: True)
    with pytest.raises(AssetError):
        resolve_local_source(str(source), label="image", allowed_suffixes={".png"})


def test_journal_recovery_restores_old_project_and_plan(tmp_path: Path) -> None:
    store, project, project_dir = _valid_project(tmp_path)
    code, payload = cli._cmd_prepare(
        SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state"), json=True)
    )
    assert code == 0
    old_project = (project_dir / "project.json").read_bytes()
    old_plan = (project_dir / "plan.json").read_bytes()
    from scripts.kf1.plan import resolve_plan

    old_canonical = resolve_plan(project_dir / "plan.json").plan_sha256
    lock = store.lock("p1")
    try:
        store._write_prepare_journal_locked(
            "p1",
            old_project_bytes=old_project,
            old_plan_bytes=old_plan,
            old_plan_canonical_sha256=old_canonical,
        )
    finally:
        lock.release()
    (project_dir / "project.json").write_bytes(b"broken")
    (project_dir / "plan.json").write_bytes(b"new")
    assert store.recover_prepare_journal("p1") is True
    assert (project_dir / "project.json").read_bytes() == old_project
    assert (project_dir / "plan.json").read_bytes() == old_plan
    assert not (project_dir / "prepare.journal.json").exists()

def test_png_boundary_cap_accepts_exact_limit_and_rejects_over(monkeypatch) -> None:
    import struct

    def png_header(width: int, height: int) -> bytes:
        return (
            b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">II", width, height)
            + bytes([8, 2, 0, 0, 0])
        )

    monkeypatch.setattr(assets_mod, "Image", None)
    monkeypatch.setattr(
        assets_mod,
        "validate_png_bytes",
        lambda data: {
            "width": 10000, "height": 5000, "orientation": 1, "mode": "RGB",
            "sha256": "x", "size_bytes": len(data),
        },
    )
    exact = decode_image_bytes(png_header(10000, 5000), label="exact")
    assert exact["width"] * exact["height"] == MAX_PIXELS
    with pytest.raises(AssetError):
        decode_image_bytes(png_header(10000, 5001), label="over")


def test_public_non_json_preview_and_prepare_are_readable(tmp_path: Path, capsys) -> None:
    _valid_project(tmp_path)
    args = SimpleNamespace(
        id="p1", projects_dir=str(tmp_path / "projects"),
        kf1_state_dir=str(tmp_path / "state"), json=False,
    )
    # Preview human output contains real Russian labels and scene details.
    code, payload = cli._cmd_preview(SimpleNamespace(id="p1", projects_dir=args.projects_dir, json=False))
    assert code == 0
    cli._print_summary(payload)
    out = capsys.readouterr().out
    assert "Проект:" in out and "Библиотека:" in out and "Сцены:" in out

    code, payload = cli._cmd_prepare(args)
    assert code == 0
    cli._print_prepare(payload)
    out = capsys.readouterr().out
    assert "Подготовка:" in out and "Plan SHA256:" in out and "Dry-run: PLAN_ONLY" in out
    assert "--execute" not in out


def test_apply_proposal_orientation_and_bom_numbered(tmp_path: Path) -> None:
    from scripts.kf2.text_import import apply_proposal, parse_text_document

    data = "\ufeff1. Первая сцена\n2. Вторая сцена\n".encode("utf-8")
    proposal = parse_text_document(data, numbered=True)
    project = _project(tmp_path)[1]
    project = apply_proposal(project, proposal)
    assert project["scenes"][0]["final"]["require_anchor_approval"] is False
    assert project["scenes"][1]["final"]["require_anchor_approval"] is True


def test_prepare_rolls_back_if_metadata_save_fails(tmp_path: Path, monkeypatch) -> None:
    store, project, project_dir = _valid_project(tmp_path)
    code, _ = cli._cmd_prepare(
        SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state1"), json=True)
    )
    assert code == 0
    old_project = (project_dir / "project.json").read_bytes()
    old_plan = (project_dir / "plan.json").read_bytes()

    def fail_save(self, *args, **kwargs):
        raise RuntimeError("injected metadata save failure")

    monkeypatch.setattr(ProjectStore, "_save_locked", fail_save)
    with pytest.raises(RuntimeError):
        cli._cmd_prepare(
            SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state2"), json=True)
        )
    assert (project_dir / "project.json").read_bytes() == old_project
    assert (project_dir / "plan.json").read_bytes() == old_plan
    assert not (project_dir / "prepare.journal.json").exists()


def test_asset_subject_view_association_edits_and_saves(tmp_path: Path) -> None:
    store, project, project_dir = _project(tmp_path)
    img = tmp_path / "car.png"
    img.write_bytes(_png(32, 32))
    asset = cli._add_image_to_project(project, project_dir, role="object", path=str(img), subject_id="car", view="front")
    project = cli._save(store, project)

    def fake_input(prompt: str) -> str:
        values = {
            "Asset (номер или ID): ": "1",
            "Subject ID (для object можно -): ": "car",
            "Subject label: ": "машина",
                "View label (пусто/- = убрать) [front]: ": "side",
        }
        return values.get(prompt, "")

    cli._edit_asset(store, project, input_fn=fake_input, print_fn=lambda _x: None)
    reloaded = store.load("p1")
    assert reloaded["library"][0]["subject_id"] == "car"
    assert reloaded["library"][0]["view"] == "side"
    assert reloaded["library"][0]["id"] == asset["id"]


def test_jpeg_without_pillow_is_unsupported_without_install(monkeypatch) -> None:
    monkeypatch.setattr(assets_mod, "Image", None)
    with pytest.raises(AssetError):
        decode_image_bytes(b"\xff\xd8fakejpeg", label="jpeg")


def test_non_json_prepare_blocked_is_stderr_exit2(tmp_path: Path, capsys) -> None:
    _project(tmp_path)
    code = cli.main(["prepare", "p1", "--projects-dir", str(tmp_path / "projects")])
    captured = capsys.readouterr()
    assert code == 2
    assert "MISSING" in captured.err or "BLOCK" in captured.err
    assert captured.out == ""


def test_reimport_updates_provenance_only_after_validation(tmp_path: Path) -> None:
    store, project, project_dir = _project(tmp_path)
    (tmp_path / "a.png").write_bytes(_png(704, 1280))
    (tmp_path / "b.png").write_bytes(_png(64, 64))
    first = cli._add_image_to_project(project, project_dir, role="provided_keyframe", path=str(tmp_path / "a.png"))
    provided = cli._add_image_to_project(project, project_dir, role="provided_keyframe", path=str(tmp_path / "b.png"))
    project = cli._save(store, project)
    provenance = {
        "content_sha256": provided["sha256"],
        "frame_id": "shot-02/A",
        "plan_sha256": "1" * 64,
        "revision_id": "2" * 64,
        "version": 1,
    }
    updated = cli._add_image_to_project(
        project, project_dir, role="provided_keyframe", path=str(tmp_path / "b.png"), provenance=provenance
    )
    assert updated["id"] == provided["id"]
    assert updated["provenance"]["frame_id"] == "shot-02/A"
    project = cli._save(store, project)
    project = store.load("p1")
    project["scenes"] = [
        cli._blank_scene(0, "shot-01"),
        cli._blank_scene(1, "shot-02"),
    ]
    project["scenes"][0]["description"] = "A1"
    project["scenes"][0]["initial"]["asset_id"] = first["id"]
    project["scenes"][0]["final"]["kind"] = "provided_keyframe"
    project["scenes"][0]["final"]["asset_id"] = provided["id"]
    project["scenes"][1]["description"] = "B provided"
    project["scenes"][1]["initial"]["kind"] = "provided_keyframe"
    project["scenes"][1]["initial"]["asset_id"] = provided["id"]
    project["scenes"][1]["final"]["kind"] = "provided_keyframe"
    project["scenes"][1]["final"]["asset_id"] = provided["id"]
    project = cli._save(store, project)
    code, payload = cli._cmd_prepare(
        SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state"), json=True)
    )
    assert code == 0, payload

    bad = dict(provenance)
    bad["content_sha256"] = "0" * 64
    before = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))["library"]
    with pytest.raises(AssetError):
        cli._add_image_to_project(
            project, project_dir, role="provided_keyframe", path=str(tmp_path / "b.png"), provenance=bad
        )
    assert json.loads((project_dir / "project.json").read_text(encoding="utf-8"))["library"] == before


def test_clearing_object_subject_removes_all_views(tmp_path: Path) -> None:
    store, project, project_dir = _project(tmp_path)
    img = tmp_path / "car.png"
    img.write_bytes(_png(32, 32))
    asset = cli._add_image_to_project(project, project_dir, role="object", path=str(img), subject_id="car", view="front")
    project = cli._save(store, project)

    def fake_input(prompt: str) -> str:
        values = {
            "Asset (номер или ID): ": "1",
            "Existing Subject ID (для object можно -) [car]: ": "-",
            "View label (пусто/- = убрать) [front]: ": "-",
            "Provenance JSON (пусто=оставить, - = очистить): ": "",
        }
        return values.get(prompt, "")

    cli._edit_asset(store, project, input_fn=fake_input, print_fn=lambda _x: None)
    reloaded = store.load("p1")
    assert reloaded["library"][0]["subject_id"] is None
    assert reloaded["library"][0]["view"] is None
    assert all(not (s.get("views") or []) for s in reloaded["subjects"])
    assert reloaded["library"][0]["id"] == asset["id"]
