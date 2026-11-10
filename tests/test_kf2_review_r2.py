from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kf2 import cli
from scripts.kf2.errors import AssetError, ProjectConflict, ProjectError, ProjectNotFound
from scripts.kf2.paths import validate_local_destination
from scripts.kf2.paths import MAX_PROJECT_JSON_BYTES
from scripts.kf2.store import ProjectStore, authoring_digest, new_project
from tests.kf1_fake_comfy import _png


def _valid(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    project, project_dir = store.create("p1", title="Проект", orientation="portrait")
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


def _prepare(store, project, project_dir, tmp_path, *, name="state"):
    return cli._cmd_prepare(
        SimpleNamespace(
            id=str(project["id"]),
            projects_dir=str(tmp_path / "projects"),
            kf1_state_dir=str(tmp_path / name),
            json=True,
        )
    )


def _journal(store, project_dir, *, canonical=None):
    project_bytes = (project_dir / "project.json").read_bytes()
    plan_bytes = (project_dir / "plan.json").read_bytes()
    if canonical is None:
        from scripts.kf1.plan import resolve_plan

        canonical = resolve_plan(project_dir / "plan.json").plan_sha256
    lock = store.lock("p1")
    try:
        store._write_prepare_journal_locked(
            "p1",
            old_project_bytes=project_bytes,
            old_plan_bytes=plan_bytes,
            old_plan_canonical_sha256=canonical,
        )
    finally:
        lock.release()
    return project_bytes, plan_bytes


def test_public_load_recovers_journal_before_edit_and_next_prepare_keeps_new_fields(tmp_path: Path) -> None:
    store, project, project_dir = _valid(tmp_path)
    assert _prepare(store, project, project_dir, tmp_path)[0] == 0
    _journal(store, project_dir)
    old_plan = (project_dir / "plan.json").read_bytes()
    (project_dir / "plan.json").write_bytes(b'{"changed":true}\n')
    project = store.load("p1")  # public load must recover before reading
    assert (project_dir / "plan.json").read_bytes() == old_plan
    assert not (project_dir / "prepare.journal.json").exists()
    project = store.load("p1")
    project["title"] = "Новое название"
    project["scenes"][0]["action"] = "Новое действие"
    project = store.save("p1", project, expected_revision=project["revision"])
    assert _prepare(store, project, project_dir, tmp_path, name="state2")[0] == 0
    reloaded = store.load("p1")
    assert reloaded["title"] == "Новое название"
    assert reloaded["scenes"][0]["action"] == "Новое действие"


def test_public_preview_list_and_import_recover_journal(tmp_path: Path) -> None:
    store, project, project_dir = _valid(tmp_path)
    assert _prepare(store, project, project_dir, tmp_path)[0] == 0
    image = tmp_path / "add.png"
    image.write_bytes(_png(8, 8))
    for operation in ("preview", "list", "resume", "add"):
        _journal(store, project_dir)
        if operation == "preview":
            code, _ = cli._cmd_preview(
                SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), json=True)
            )
        elif operation == "list":
            code, _ = cli._cmd_list(SimpleNamespace(projects_dir=str(tmp_path / "projects"), json=True))
        elif operation == "resume":
            code, _ = cli._cmd_resume(
                SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), json=True)
            )
        else:
            code, _ = cli._cmd_add_image(
                SimpleNamespace(
                    id="p1", projects_dir=str(tmp_path / "projects"), json=True,
                    role="style", path=str(image), subject=None, subject_label=None,
                    view=None, provenance=None,
                )
            )
        assert code == 0
        assert not (project_dir / "prepare.journal.json").exists()
    text = tmp_path / "text.txt"
    text.write_bytes("scene: shot-01\ndescription: Текст\n".encode("utf-8"))
    _journal(store, project_dir)
    code, _ = cli._cmd_import_text(
        SimpleNamespace(
            id="p1", projects_dir=str(tmp_path / "projects"), json=True,
            file=str(text), numbered=False, confirm=True,
        )
    )
    assert code == 0
    assert not (project_dir / "prepare.journal.json").exists()


def _mutate_journal(project_dir: Path, mutate) -> bytes:
    path = project_dir / "prepare.journal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return data


def test_invalid_journals_fail_closed_without_side_effects(tmp_path: Path) -> None:
    store, project, project_dir = _valid(tmp_path)
    assert _prepare(store, project, project_dir, tmp_path)[0] == 0
    _journal(store, project_dir)
    journal_path = project_dir / "prepare.journal.json"
    valid_journal = journal_path.read_bytes()
    project_before = (project_dir / "project.json").read_bytes()
    plan_before = (project_dir / "plan.json").read_bytes()

    foreign = new_project("other", title="foreign", orientation="portrait")
    cases = [
        lambda p: p.__setitem__("old_project_b64", "!!!!"),
        lambda p: p.__setitem__("old_project_b64", base64.b64encode(b"").decode("ascii")),
        lambda p: p.__setitem__("old_project_b64", base64.b64encode(json.dumps(foreign).encode("utf-8")).decode("ascii")),
        lambda p: p.__setitem__("pair_sha256", "0" * 64),
        lambda p: p.__setitem__("old_plan_exists", "true"),
        lambda p: p.__setitem__("old_plan_b64", None),
    ]
    for mutate in cases:
        journal_path.write_bytes(valid_journal)
        bad = _mutate_journal(project_dir, mutate)
        with pytest.raises(ProjectError):
            store.load("p1")
        assert (project_dir / "project.json").read_bytes() == project_before
        assert (project_dir / "plan.json").read_bytes() == plan_before
        assert journal_path.read_bytes() == bad
    journal_path.unlink()


def test_project_and_plan_and_journal_write_caps(tmp_path: Path, monkeypatch) -> None:
    store, project, project_dir = _valid(tmp_path)
    before = (project_dir / "project.json").read_bytes()
    project["style"] = "x" * MAX_PROJECT_JSON_BYTES
    with pytest.raises(ProjectError):
        store.save("p1", project, expected_revision=project["revision"])
    assert (project_dir / "project.json").read_bytes() == before

    project = store.load("p1")
    assert _prepare(store, project, project_dir, tmp_path)[0] == 0
    old_project = (project_dir / "project.json").read_bytes()
    old_plan = (project_dir / "plan.json").read_bytes()
    monkeypatch.setattr("scripts.kf2.cli.MAX_PLAN_BYTES", 50)
    with pytest.raises(ProjectError):
        _prepare(store, project, project_dir, tmp_path, name="state2")
    assert (project_dir / "project.json").read_bytes() == old_project
    assert (project_dir / "plan.json").read_bytes() == old_plan
    assert not (project_dir / "prepare.journal.json").exists()


def test_unc_and_mapped_destinations_are_gated_before_filesystem(tmp_path: Path, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("filesystem stat/exists/mkdir must not run before lexical/drive gate")

    monkeypatch.setattr(Path, "exists", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    with pytest.raises(AssetError):
        ProjectStore(r"\\server\share\projects")
    with pytest.raises(AssetError):
        ProjectStore(r"\\?\C:\projects")
    monkeypatch.undo()

    monkeypatch.setattr("scripts.kf2.paths.drive_type_is_remote", lambda _path: True)
    with pytest.raises(AssetError):
        ProjectStore(tmp_path / "remote-projects")


def test_edit_asset_updates_subject_views_and_clear_association(tmp_path: Path) -> None:
    store, project, project_dir = _valid(tmp_path)
    asset = next(item for item in project["library"] if item["role"] == "object")
    subject = next(item for item in project["subjects"] if item["id"] == "car")

    def set_side(prompt: str) -> str:
        values = {
            "Asset (номер или ID): ": "2",
            "Existing Subject ID (для object можно -) [car]: ": "car",
            "View label (пусто/- = убрать) [front]: ": "side",
            "Provenance JSON (пусто=оставить, - = очистить): ": "",
        }
        return values.get(prompt, "")

    cli._edit_asset(store, project, input_fn=set_side, print_fn=lambda _x: None)
    project = store.load("p1")
    subject = next(item for item in project["subjects"] if item["id"] == "car")
    assert subject["views"] == [{"asset_id": asset["id"], "label": "side"}]
    result = cli._summary(project, project_dir)
    assert not any(item["code"] == "SUBJECT_VIEW" for item in result["blockers"])

    def clear_view(prompt: str) -> str:
        values = {
            "Asset (номер или ID): ": "2",
            "Existing Subject ID (для object можно -) [car]: ": "car",
            "View label (пусто/- = убрать) [side]: ": "-",
            "Provenance JSON (пусто=оставить, - = очистить): ": "",
        }
        return values.get(prompt, "")

    cli._edit_asset(store, project, input_fn=clear_view, print_fn=lambda _x: None)
    project = store.load("p1")
    subject = next(item for item in project["subjects"] if item["id"] == "car")
    assert subject["views"] == []
    assert next(item for item in project["library"] if item["id"] == asset["id"])["view"] is None


def test_threaded_prepare_holds_lock_and_edit_cannot_write(tmp_path: Path, monkeypatch) -> None:
    store, project, project_dir = _valid(tmp_path)
    stale = store.load("p1")
    started = threading.Event()
    release = threading.Event()
    original_dryrun = cli._dryrun

    def slow_dryrun(plan_path, state_dir):
        started.set()
        if not release.wait(10):
            raise RuntimeError("test release timeout")
        return original_dryrun(plan_path, state_dir)

    monkeypatch.setattr(cli, "_dryrun", slow_dryrun)
    errors: list[BaseException] = []

    def run_prepare() -> None:
        try:
            cli._cmd_prepare(
                SimpleNamespace(
                    id="p1", projects_dir=str(tmp_path / "projects"),
                    kf1_state_dir=str(tmp_path / "thread-state"), json=True,
                )
            )
        except BaseException as exc:  # pragma: no cover - reported by assertion
            errors.append(exc)

    thread = threading.Thread(target=run_prepare)
    thread.start()
    assert started.wait(10)
    second_errors: list[BaseException] = []

    def run_second_prepare() -> None:
        try:
            cli._cmd_prepare(
                SimpleNamespace(
                    id="p1", projects_dir=str(tmp_path / "projects"),
                    kf1_state_dir=str(tmp_path / "thread-state-2"), json=True,
                )
            )
        except BaseException as exc:
            second_errors.append(exc)

    second = threading.Thread(target=run_second_prepare)
    second.start()
    second.join(10)
    assert not second.is_alive()
    assert second_errors and isinstance(second_errors[0], (ProjectConflict, ProjectError))
    stale["title"] = "Не должно сохраниться"
    with pytest.raises((ProjectConflict, ProjectError)):
        store.save("p1", stale, expected_revision=stale["revision"])
    release.set()
    thread.join(20)
    assert not thread.is_alive()
    assert errors == []
    assert store.load("p1")["title"] == "Проект"


def test_journal_larger_than_4mib_recovers_with_bounded_32mib(tmp_path: Path) -> None:
    store, project, project_dir = _valid(tmp_path)
    project["style"] = "x" * (3 * 1024 * 1024)
    project = store.save("p1", project, expected_revision=project["revision"])
    assert _prepare(store, project, project_dir, tmp_path)[0] == 0
    _journal(store, project_dir)
    assert (project_dir / "prepare.journal.json").stat().st_size > 4 * 1024 * 1024
    loaded = store.load("p1")
    assert loaded["style"] == "x" * (3 * 1024 * 1024)
    assert not (project_dir / "prepare.journal.json").exists()


def test_unc_kf1_state_dir_is_rejected_before_prepare_writes(tmp_path: Path) -> None:
    store, project, project_dir = _valid(tmp_path)
    before = (project_dir / "project.json").read_bytes()
    with pytest.raises(AssetError):
        cli._cmd_prepare(
            SimpleNamespace(
                id="p1", projects_dir=str(tmp_path / "projects"),
                kf1_state_dir=r"\\server\share\kf1", json=True,
            )
        )
    assert (project_dir / "project.json").read_bytes() == before
    assert not (project_dir / "plan.json").exists()
    assert not (project_dir / "prepare.journal.json").exists()


def test_drive_gate_is_lexical_and_fails_closed(monkeypatch) -> None:
    import ctypes
    from pathlib import Path as P

    seen: list[str] = []

    def fake_drive(root: str) -> int:
        seen.append(root)
        return 4

    monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", fake_drive)
    for method in ("resolve", "stat", "lstat", "exists", "mkdir"):
        monkeypatch.setattr(P, method, lambda *a, **k: (_ for _ in ()).throw(AssertionError("filesystem touched before drive gate")))
    with pytest.raises(AssetError):
        validate_local_destination(P(r"C:\managed\projects"))
    assert seen == ["C:\\"]

    seen.clear()
    monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", lambda root: (seen.append(root) or 0))
    with pytest.raises(AssetError):
        validate_local_destination(P(r"D:\managed\projects"))
    assert seen == ["D:\\"]


def test_prepared_plan_semantic_mismatch_is_rejected_but_stale_draft_allowed(tmp_path: Path) -> None:
    from scripts.kf2.store import authoring_digest

    store, project, project_dir = _valid(tmp_path)
    assert _prepare(store, project, project_dir, tmp_path)[0] == 0
    project = store.load("p1")
    old_plan = (project_dir / "plan.json").read_bytes()
    from scripts.kf1.plan import resolve_plan

    canonical = resolve_plan(project_dir / "plan.json").plan_sha256
    bad = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    bad["prepared"]["plan_sha256"] = "1" * 64
    bad["content_sha256"] = authoring_digest(bad)
    bad_bytes = json.dumps(bad, ensure_ascii=False, sort_keys=True).encode("utf-8")
    lock = store.lock("p1")
    try:
        store._write_prepare_journal_locked(
            "p1", old_project_bytes=bad_bytes, old_plan_bytes=old_plan,
            old_plan_canonical_sha256=canonical,
        )
    finally:
        lock.release()
    with pytest.raises(ProjectError):
        store.load("p1")
    assert (project_dir / "plan.json").read_bytes() == old_plan

    stale = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    stale["title"] = "Stale draft title"
    stale["content_sha256"] = authoring_digest(stale)
    stale_bytes = json.dumps(stale, ensure_ascii=False, sort_keys=True).encode("utf-8")
    (project_dir / "prepare.journal.json").unlink()
    lock = store.lock("p1")
    try:
        store._write_prepare_journal_locked(
            "p1", old_project_bytes=stale_bytes, old_plan_bytes=old_plan,
            old_plan_canonical_sha256=canonical,
        )
    finally:
        lock.release()
    recovered = store.load("p1")
    assert recovered["title"] == "Stale draft title"
    assert recovered["prepared"]["plan_sha256"] == canonical


def test_changed_plan_dryrun_and_save_faults_restore_old_pair_and_keep_edit(tmp_path: Path, monkeypatch) -> None:
    from scripts.kf1.plan import resolve_plan

    store, project, project_dir = _valid(tmp_path)
    original_dryrun = cli._dryrun
    assert _prepare(store, project, project_dir, tmp_path, name="state1")[0] == 0
    project = store.load("p1")
    project["title"] = "Edited before failed prepare"
    project["scenes"][0]["final"]["instruction"] = "Explicit edited frame instruction."
    project = store.save("p1", project, expected_revision=project["revision"])
    old_plan = (project_dir / "plan.json").read_bytes()
    edited_project = (project_dir / "project.json").read_bytes()
    old_sha = resolve_plan(project_dir / "plan.json").plan_sha256
    seen: list[str] = []

    def fail_dryrun(plan_path, state_dir):
        seen.append(resolve_plan(plan_path).plan_sha256)
        assert "Explicit edited frame instruction." in plan_path.read_text(encoding="utf-8")
        raise RuntimeError("changed-plan dryrun fault")

    monkeypatch.setattr(cli, "_dryrun", fail_dryrun)
    with pytest.raises(RuntimeError):
        _prepare(store, project, project_dir, tmp_path, name="state2")
    assert seen and seen[0] != old_sha
    assert (project_dir / "plan.json").read_bytes() == old_plan
    assert (project_dir / "project.json").read_bytes() == edited_project
    assert store.load("p1")["title"] == "Edited before failed prepare"
    assert store.load("p1")["prepared"]["plan_sha256"] == old_sha

    # Metadata-save fault must also restore old plan but preserve edited draft.
    monkeypatch.setattr(cli, "_dryrun", lambda plan_path, state_dir: original_dryrun(plan_path, state_dir))
    original_save = ProjectStore._save_locked

    def fail_save(self, *args, **kwargs):
        raise RuntimeError("changed-plan metadata fault")

    monkeypatch.setattr(ProjectStore, "_save_locked", fail_save)
    with pytest.raises(RuntimeError):
        _prepare(store, project, project_dir, tmp_path, name="state3")
    assert (project_dir / "plan.json").read_bytes() == old_plan
    assert (project_dir / "project.json").read_bytes() == edited_project
    monkeypatch.setattr(ProjectStore, "_save_locked", original_save)
    assert store.load("p1")["title"] == "Edited before failed prepare"


def test_unknown_public_ops_do_not_reserve_project_id(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    typo = "p1-typ"
    for call in (
        lambda: store.load(typo),
        lambda: store.save(typo, new_project(typo)),
        lambda: cli._cmd_preview(SimpleNamespace(id=typo, projects_dir=str(tmp_path / "projects"), json=True)),
        lambda: cli._cmd_resume(SimpleNamespace(id=typo, projects_dir=str(tmp_path / "projects"), json=True)),
        lambda: cli._cmd_prepare(SimpleNamespace(id=typo, projects_dir=str(tmp_path / "projects"), kf1_state_dir=str(tmp_path / "state"), json=True)),
    ):
        with pytest.raises(ProjectNotFound):
            call()
        assert not (tmp_path / "projects" / typo).exists()
    created, _ = store.create(typo, title="later", orientation="portrait")
    assert created["id"] == typo
    assert (tmp_path / "projects" / typo).is_dir()


def test_oversized_touched_asset_rejected_before_resolver(tmp_path: Path, monkeypatch) -> None:
    store, project, project_dir = _valid(tmp_path)
    assert _prepare(store, project, project_dir, tmp_path)[0] == 0
    _journal(store, project_dir)
    project_now = store.load("p1")
    # Restore journal after the public load consumed it.
    _journal(store, project_dir)
    asset = next(item for item in project_now["library"] if item["role"] == "provided_keyframe")
    (project_dir / asset["stored_path"]).write_bytes(b"x" * (32 * 1024 * 1024 + 1))
    project_before = (project_dir / "project.json").read_bytes()
    plan_before = (project_dir / "plan.json").read_bytes()
    journal_before = (project_dir / "prepare.journal.json").read_bytes()
    called = {"resolver": False}

    def fail_resolver(*args, **kwargs):
        called["resolver"] = True
        raise AssertionError("resolver must not be called before bounded asset validation")

    monkeypatch.setattr("scripts.kf1.plan.resolve_plan", fail_resolver)
    with pytest.raises(ProjectError):
        store.load("p1")
    assert called["resolver"] is False
    assert (project_dir / "project.json").read_bytes() == project_before
    assert (project_dir / "plan.json").read_bytes() == plan_before
    assert (project_dir / "prepare.journal.json").read_bytes() == journal_before


def test_ancestor_junction_stops_before_descendant_probe(tmp_path: Path, monkeypatch) -> None:
    import os
    import stat
    from types import SimpleNamespace

    junction = tmp_path / "junction"
    child = junction / "child.png"
    original_lstat = os.lstat
    attempts: list[str] = []

    def fake_lstat(value):
        text = os.path.normcase(os.path.abspath(str(value)))
        attempts.append(text)
        if text == os.path.normcase(os.path.abspath(str(junction))):
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        if text.startswith(os.path.normcase(os.path.abspath(str(junction)) + os.sep)):
            raise AssertionError("descendant lstat attempted through junction")
        return original_lstat(value)

    monkeypatch.setattr("scripts.kf2.paths.os.lstat", fake_lstat)
    original_exists = Path.exists

    def guarded_exists(self):
        text = os.path.normcase(os.path.abspath(str(self)))
        if text == os.path.normcase(os.path.abspath(str(junction))) or text.startswith(
            os.path.normcase(os.path.abspath(str(junction)) + os.sep)
        ):
            raise AssertionError("Path.exists on descendant")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    from scripts.kf2.paths import atomic_write_bytes, checked_exists, resolve_local_source

    for value, label, suffixes in (
        (str(child), "image", {".png"}),
        (str(child), "text", {".txt"}),
        (str(child), "provenance", {".json"}),
    ):
        with pytest.raises(AssetError):
            resolve_local_source(value, label=label, allowed_suffixes=suffixes)
    with pytest.raises(AssetError):
        checked_exists(child, label="managed child")
    with pytest.raises((AssetError, ProjectError)):
        atomic_write_bytes(child, b"x")
    with pytest.raises(ProjectError):
        ProjectStore(junction / "projects")
    descendant_prefix = os.path.normcase(os.path.abspath(str(junction))) + os.sep
    assert not any(item.startswith(descendant_prefix) for item in attempts)


def test_public_load_save_do_not_exists_probe_targets_before_gate(tmp_path: Path, monkeypatch) -> None:
    store, project, project_dir = _valid(tmp_path)
    original_exists = Path.exists
    forbidden = {"project.json", "prepare.journal.json", "plan.json"}

    def guarded_exists(self):
        if Path(self).name in forbidden:
            raise AssertionError(f"Path.exists before managed gate: {self}")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    loaded = store.load("p1")
    assert loaded["id"] == "p1"
    loaded["title"] = "after gate"
    saved = store.save("p1", loaded, expected_revision=loaded["revision"])
    assert saved["title"] == "after gate"


def test_final_project_json_symlink_is_rejected_before_public_ops(tmp_path: Path) -> None:
    store, project, project_dir = _valid(tmp_path)
    target = tmp_path / "real-project.json"
    target.write_bytes((project_dir / "project.json").read_bytes())
    try:
        (project_dir / "project.json").unlink()
        os.symlink(target, project_dir / "project.json")
    except OSError:
        pytest.skip("symlink privilege unavailable")
    for call in (
        lambda: store.load("p1"),
        lambda: store.save("p1", project, expected_revision=project["revision"]),
        lambda: cli._cmd_preview(SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), json=True)),
    ):
        with pytest.raises((AssetError, ProjectError)):
            call()


def test_old_plan_reference_junction_is_rejected_before_resolver(monkeypatch) -> None:
    import hashlib
    import os
    import stat
    from types import SimpleNamespace
    from scripts.kf2.store import _validate_old_plan_bytes

    junction = Path("C:/kf2-junction-placeholder")
    child = junction / "a.png"
    original_lstat = os.lstat
    attempts: list[str] = []

    def fake_lstat(value):
        text = os.path.normcase(os.path.abspath(str(value)))
        attempts.append(text)
        if text == os.path.normcase(os.path.abspath(str(junction))):
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        if text.startswith(os.path.normcase(os.path.abspath(str(junction)) + os.sep)):
            raise AssertionError("descendant lstat")
        return original_lstat(value)

    monkeypatch.setattr("scripts.kf2.paths.os.lstat", fake_lstat)
    plan = {
        "schema": "anime-studio/keyframe-plan/v2",
        "title": "t",
        "first_frame": str(child),
        "scenes": [],
    }
    data = json.dumps(plan).encode("utf-8")
    called = {"resolver": False}

    def fail(*args, **kwargs):
        called["resolver"] = True
        raise AssertionError("resolver must not run")

    monkeypatch.setattr("scripts.kf1.plan.resolve_plan", fail)
    with pytest.raises(ProjectError):
        _validate_old_plan_bytes(
            data,
            project_dir=Path("C:/unused"),
            project={},
            expected_raw_sha=hashlib.sha256(data).hexdigest(),
            expected_canonical_sha="0" * 64,
        )
    assert called["resolver"] is False
    assert not any(item.startswith(os.path.normcase(os.path.abspath(str(junction)) + os.sep)) for item in attempts)


def test_validate_library_asset_and_public_preview_gate_asset_parent_junction(tmp_path: Path, monkeypatch) -> None:
    import os
    import stat
    from types import SimpleNamespace
    from scripts.kf2.assets import validate_library_asset

    store, project, project_dir = _valid(tmp_path)
    project["library"][0]["stored_path"] = "assets/junction/child.png"
    project = store.save("p1", project, expected_revision=project["revision"])
    junction = project_dir / "assets" / "junction"
    original_lstat = os.lstat
    attempts: list[str] = []

    def fake_lstat(value):
        text = os.path.normcase(os.path.abspath(str(value)))
        attempts.append(text)
        if text == os.path.normcase(os.path.abspath(str(junction))):
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        if text.startswith(os.path.normcase(os.path.abspath(str(junction)) + os.sep)):
            raise AssertionError("descendant lstat through asset junction")
        return original_lstat(value)

    monkeypatch.setattr("scripts.kf2.paths.os.lstat", fake_lstat)
    with pytest.raises(AssetError):
        validate_library_asset(project_dir, project["library"][0])
    code, payload = cli._cmd_preview(
        SimpleNamespace(id="p1", projects_dir=str(tmp_path / "projects"), json=True)
    )
    assert code == 0
    assert any(item["code"] == "ASSET_INVALID" for item in payload["blockers"])
    descendant_prefix = os.path.normcase(os.path.abspath(str(junction)) + os.sep)
    assert not any(item.startswith(descendant_prefix) for item in attempts)
