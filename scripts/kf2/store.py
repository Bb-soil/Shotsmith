"""Versioned, optimistic-lock persistence for KF2 authoring projects."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from anime_studio.errors import InstanceAlreadyRunning
from anime_studio.lock import InstanceLock

from .errors import (
    ProjectConflict,
    ProjectError,
    ProjectExists,
    ProjectNotFound,
    ProjectTamper,
)
from .paths import (
    DEFAULT_PROJECTS_DIR,
    MAX_IMAGE_BYTES,
    MAX_JOURNAL_BYTES,
    MAX_PLAN_BYTES,
    MAX_PROJECT_JSON_BYTES,
    ROOT,
    assert_contained,
    assert_no_reparse_ancestors,
    atomic_write_bytes,
    checked_exists,
    checked_is_dir,
    checked_unlink,
    atomic_write_json,
    canonical_json_bytes,
    ensure_managed_dir,
    is_reparse,
    read_bytes_limited,
    sha256_bytes,
    validate_local_destination,
    validate_project_id,
)

PROJECT_SCHEMA = "anime-studio/kf2-project/v1"
PLAN_SCHEMA = "anime-studio/keyframe-plan/v2"
JOURNAL_SCHEMA = "anime-studio/kf2-prepare-journal/v2"
PORTRAIT = ([704, 1280], [448, 816])
LANDSCAPE = ([1280, 704], [816, 448])
AUTHORING_EXCLUDE = {
    "content_sha256",
    "prepared",
    "revision",
    "created_utc",
    "updated_utc",
    "status",
}
PREPARE_JOURNAL = "prepare.journal.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_strict(data: bytes, *, label: str) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(text, object_pairs_hook=_no_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectError(f"{label}: некорректный JSON") from exc


def authoring_digest(project: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in project.items() if key not in AUTHORING_EXCLUDE}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _geometry(orientation: str) -> tuple[list[int], list[int]]:
    if orientation == "portrait":
        return PORTRAIT
    if orientation == "landscape":
        return LANDSCAPE
    raise ProjectError("orientation должен быть portrait или landscape")


def new_project(
    project_id: str,
    *,
    title: str = "Новый проект",
    orientation: str = "portrait",
) -> dict[str, Any]:
    clean_id = validate_project_id(project_id)
    candidate, video = _geometry(orientation)
    created = now_utc()
    project: dict[str, Any] = {
        "schema": PROJECT_SCHEMA,
        "id": clean_id,
        "revision": 1,
        "status": "DRAFT",
        "created_utc": created,
        "updated_utc": created,
        "language": "ru",
        "title": title.strip() or "Новый проект",
        "orientation": orientation,
        "candidate_geometry": list(candidate),
        "video_geometry": list(video),
        "style": "",
        "subject_label": "",
        "subjects": [],
        "library": [],
        "scripts": [],
        "scenes": [],
        "generation_limit": 3,
        "seed": 424242,
        "notes": "",
        "import": None,
        "prepared": None,
    }
    project["content_sha256"] = authoring_digest(project)
    return project


def _validate_project_bytes(data: bytes, *, project_id: str) -> dict[str, Any]:
    if len(data) > MAX_PROJECT_JSON_BYTES:
        raise ProjectTamper(f"project.json превышает {MAX_PROJECT_JSON_BYTES} bytes")
    project = _parse_json_strict(data, label="project.json")
    if not isinstance(project, dict):
        raise ProjectError("project.json должен быть объектом")
    if project.get("schema") != PROJECT_SCHEMA:
        raise ProjectError("project.json schema неверна")
    if project.get("id") != project_id:
        raise ProjectError("project.json id не совпадает с каталогом")
    if not isinstance(project.get("revision"), int) or isinstance(project.get("revision"), bool) or project["revision"] < 1:
        raise ProjectError("project.revision должен быть >=1")
    if project.get("content_sha256") != authoring_digest(project):
        raise ProjectTamper("project.json content_sha256 mismatch")
    return project


def _validate_project_payload(project: Mapping[str, Any]) -> bytes:
    data = canonical_json_bytes(project)
    if len(data) > MAX_PROJECT_JSON_BYTES:
        raise ProjectError(f"project JSON превышает {MAX_PROJECT_JSON_BYTES} bytes")
    # Validate serialization and identity before any durable change.
    _validate_project_bytes(data, project_id=str(project.get("id")))
    return data


def _plan_paths(plan: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for key in ("first_frame", "character_reference", "background_reference"):
        value = plan.get(key)
        if isinstance(value, str) and value:
            result.append(value)
    for ref in plan.get("references") or []:
        if isinstance(ref, Mapping) and isinstance(ref.get("path"), str):
            result.append(ref["path"])
    for scene in plan.get("scenes") or []:
        if not isinstance(scene, Mapping):
            continue
        for role in ("a", "b"):
            frame = scene.get(role)
            if isinstance(frame, Mapping):
                provided = frame.get("provided_frame")
                if isinstance(provided, Mapping) and isinstance(provided.get("path"), str):
                    result.append(provided["path"])
    return result


def _validate_old_plan_bytes(
    data: bytes,
    *,
    project_dir: Path,
    project: Mapping[str, Any],
    expected_raw_sha: str,
    expected_canonical_sha: str,
) -> None:
    if len(data) > MAX_PLAN_BYTES:
        raise ProjectError(f"old plan превышает {MAX_PLAN_BYTES} bytes")
    if sha256_bytes(data) != expected_raw_sha:
        raise ProjectError("old plan raw SHA256 mismatch")
    plan = _parse_json_strict(data, label="old plan")
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise ProjectError("old plan не KF1 v2")
    from .assets import decode_image_bytes
    from .paths import resolve_local_source

    for value in _plan_paths(plan):
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        try:
            source = resolve_local_source(
                str(path), label="old plan image", allowed_suffixes=IMAGE_SUFFIXES
            )
        except Exception as exc:
            raise ProjectError(f"old plan image validation failed: {exc}") from exc
        lexical = Path(os.path.abspath(str(source)))
        try:
            assert_contained(ROOT, lexical)
        except ProjectError:
            assert_contained(project_dir, lexical)
        image_bytes = read_bytes_limited(
            source, max_bytes=MAX_IMAGE_BYTES, label="old plan image"
        )
        image_info = decode_image_bytes(image_bytes, label="old plan image")
        for asset in project.get("library") or []:
            if not isinstance(asset, Mapping):
                continue
            stored = asset.get("stored_path")
            if not isinstance(stored, str):
                continue
            asset_path = os.path.normcase(os.path.abspath(str(project_dir / stored)))
            if asset_path != os.path.normcase(os.path.abspath(str(source))):
                continue
            expected = {
                "sha256": asset.get("sha256"),
                "size_bytes": asset.get("size_bytes"),
                "width": asset.get("width"),
                "height": asset.get("height"),
                "format": asset.get("format"),
            }
            actual = {
                "sha256": image_info["sha256"],
                "size_bytes": image_info["size_bytes"],
                "width": image_info["width"],
                "height": image_info["height"],
                "format": image_info["format"],
            }
            if expected != actual:
                raise ProjectError(f"old plan asset metadata mismatch: {stored}")
    # The existing resolver is intentionally called only after every path is
    # local, bounded and metadata-crosschecked. It re-reads; this is not claimed
    # as protection against a same-account adversary racing the file.
    temp = None
    try:
        handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False)
        temp = Path(handle.name)
        with handle:
            handle.write(data)
            handle.flush()
        from scripts.kf1.plan import resolve_plan

        resolved_plan = resolve_plan(temp)
    except ProjectError:
        raise
    except Exception as exc:
        raise ProjectError(f"old plan не проходит resolve_plan: {exc}") from exc
    finally:
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
    if resolved_plan.plan_sha256 != expected_canonical_sha:
        raise ProjectError("old plan canonical SHA256 mismatch")

class ProjectStore:
    def __init__(self, projects_dir: Path | str | None = None) -> None:
        self.projects_dir = validate_local_destination(
            Path(projects_dir) if projects_dir else DEFAULT_PROJECTS_DIR,
            label="projects-dir",
        )
        assert_no_reparse_ancestors(self.projects_dir)
        if self.projects_dir.exists():
            if is_reparse(self.projects_dir) or not self.projects_dir.is_dir():
                raise ProjectError(f"unsafe projects directory: {self.projects_dir}")
        else:
            self.projects_dir.mkdir(parents=True, exist_ok=True)
            assert_no_reparse_ancestors(self.projects_dir)
        if is_reparse(self.projects_dir):
            raise ProjectError(f"unsafe projects directory: {self.projects_dir}")

    def directory(self, project_id: str) -> Path:
        clean_id = validate_project_id(project_id)
        path = validate_local_destination(self.projects_dir / clean_id, label="project-dir")
        assert_contained(self.projects_dir, path)
        assert_no_reparse_ancestors(path)
        if checked_exists(path, label="project-dir") and not checked_is_dir(path, label="project-dir"):
            raise ProjectError(f"unsafe project directory: {path}")
        return path

    def project_file(self, project_id: str) -> Path:
        return self.directory(project_id) / "project.json"

    def journal_path(self, project_id: str) -> Path:
        return self.directory(project_id) / PREPARE_JOURNAL

    def _read_project_bytes(self, project_id: str) -> bytes:
        path = self.project_file(project_id)
        if not checked_exists(path, label="project.json"):
            raise ProjectNotFound(f"проект не найден: {project_id}")
        return read_bytes_limited(path, max_bytes=MAX_PROJECT_JSON_BYTES, label="project.json")

    def _load_locked(self, project_id: str) -> dict[str, Any]:
        return _validate_project_bytes(self._read_project_bytes(project_id), project_id=project_id)

    def lock(self, project_id: str) -> InstanceLock:
        project_dir = self.directory(project_id)
        if not checked_is_dir(project_dir, label="project-dir"):
            raise ProjectNotFound(f"проект не найден: {project_id}")
        project_file = project_dir / "project.json"
        journal = project_dir / PREPARE_JOURNAL
        if not checked_exists(project_file, label="project.json") and not checked_exists(journal, label="prepare journal"):
            raise ProjectNotFound(f"проект не найден: {project_id}")
        path = project_dir / "project.lock"
        assert_no_reparse_ancestors(path.parent)
        if is_reparse(path):
            raise ProjectError(f"unsafe project lock: {path}")
        try:
            return InstanceLock(path).acquire()
        except InstanceAlreadyRunning as exc:
            raise ProjectConflict("project занят другим процессом") from exc

    def load(self, project_id: str) -> dict[str, Any]:
        lock = self.lock(project_id)
        try:
            self.recover_prepare_journal_locked(project_id)
            return self._load_locked(project_id)
        finally:
            lock.release()

    def create(
        self,
        project_id: str,
        *,
        title: str = "Новый проект",
        orientation: str = "portrait",
    ) -> tuple[dict[str, Any], Path]:
        clean_id = validate_project_id(project_id)
        project = new_project(clean_id, title=title, orientation=orientation)
        project_bytes = _validate_project_payload(project)
        path = self.directory(clean_id)
        assert_no_reparse_ancestors(path.parent)
        try:
            path.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise ProjectExists(f"проект уже существует (включая Windows case collision): {clean_id}") from exc
        except OSError as exc:
            raise ProjectError(f"не удалось создать project dir: {exc}") from exc
        try:
            ensure_managed_dir(path, path / "history", label="history directory")
            atomic_write_bytes(path / "project.json", project_bytes, exclusive=True)
        except Exception:
            try:
                if not checked_exists(path / "project.json", label="project.json"):
                    shutil.rmtree(path)
            except OSError:
                pass
            raise
        return project, path

    def _save_locked(
        self,
        project_id: str,
        project: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        clean_id = validate_project_id(project_id)
        if project.get("id") != clean_id:
            raise ProjectError("project.id не совпадает с project_id")
        current_bytes = self._read_project_bytes(clean_id)
        current = _validate_project_bytes(current_bytes, project_id=clean_id)
        if expected_revision is None:
            expected_revision = int(current["revision"])
        if int(current.get("revision", -1)) != int(expected_revision):
            raise ProjectConflict(
                f"stale project revision: expected {expected_revision}, actual {current.get('revision')}"
            )
        updated = dict(project)
        updated["revision"] = int(current["revision"]) + 1
        updated["updated_utc"] = now_utc()
        updated["content_sha256"] = authoring_digest(updated)
        updated_bytes = _validate_project_payload(updated)
        history = ensure_managed_dir(
            self.directory(clean_id), self.directory(clean_id) / "history", label="history directory"
        )
        archive = history / f"project-v{int(current['revision']):04d}.json"
        if checked_exists(archive, label="history"):
            if read_bytes_limited(archive, max_bytes=MAX_PROJECT_JSON_BYTES, label="history") != current_bytes:
                raise ProjectTamper(f"history collision/tamper: {archive}")
        else:
            atomic_write_bytes(archive, current_bytes, exclusive=True)
        atomic_write_bytes(self.project_file(clean_id), updated_bytes)
        return updated

    def save(
        self,
        project_id: str,
        project: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        lock = self.lock(project_id)
        try:
            self.recover_prepare_journal_locked(project_id)
            return self._save_locked(project_id, project, expected_revision=expected_revision)
        finally:
            lock.release()

    def _write_prepare_journal_locked(
        self,
        project_id: str,
        *,
        old_project_bytes: bytes,
        old_plan_bytes: bytes | None,
        old_plan_canonical_sha256: str | None,
    ) -> Path:
        if len(old_project_bytes) > MAX_PROJECT_JSON_BYTES:
            raise ProjectError("old project превышает 4 MiB")
        if old_plan_bytes is not None and len(old_plan_bytes) > MAX_PLAN_BYTES:
            raise ProjectError("old plan превышает 8 MiB")
        if (old_plan_bytes is None) != (old_plan_canonical_sha256 is None):
            raise ProjectError("old plan/canonical pair несогласован")
        pair = old_project_bytes + b"\0" + (old_plan_bytes or b"")
        payload = {
            "schema": JOURNAL_SCHEMA,
            "created_utc": now_utc(),
            "old_project_b64": base64.b64encode(old_project_bytes).decode("ascii"),
            "old_project_sha256": sha256_bytes(old_project_bytes),
            "old_plan_exists": old_plan_bytes is not None,
            "old_plan_b64": base64.b64encode(old_plan_bytes).decode("ascii") if old_plan_bytes is not None else None,
            "old_plan_sha256": sha256_bytes(old_plan_bytes) if old_plan_bytes is not None else None,
            "old_plan_canonical_sha256": old_plan_canonical_sha256,
            "pair_sha256": sha256_bytes(pair),
        }
        data = canonical_json_bytes(payload)
        if len(data) > MAX_JOURNAL_BYTES:
            raise ProjectError("prepare journal превышает 32 MiB")
        path = self.journal_path(project_id)
        atomic_write_bytes(path, data)
        return path

    def _clear_prepare_journal_locked(self, project_id: str) -> None:
        path = self.journal_path(project_id)
        checked_unlink(path, label="prepare journal")

    def recover_prepare_journal_locked(self, project_id: str) -> bool:
        path = self.journal_path(project_id)
        if not checked_exists(path, label="prepare journal"):
            return False
        raw = read_bytes_limited(path, max_bytes=MAX_JOURNAL_BYTES, label="prepare journal")
        payload = _parse_json_strict(raw, label="prepare journal")
        if not isinstance(payload, Mapping) or payload.get("schema") != JOURNAL_SCHEMA:
            raise ProjectError("prepare journal schema/format не поддержан; recovery требуется вручную")
        required = {
            "old_project_b64",
            "old_project_sha256",
            "old_plan_exists",
            "old_plan_b64",
            "old_plan_sha256",
            "old_plan_canonical_sha256",
            "pair_sha256",
        }
        if not required.issubset(payload):
            raise ProjectError("prepare journal incomplete; current project/plan/journal не изменены")
        if not isinstance(payload["old_plan_exists"], bool):
            raise ProjectError("prepare journal old_plan_exists должен быть bool")
        old_plan_exists = payload["old_plan_exists"]
        try:
            old_project = base64.b64decode(str(payload["old_project_b64"]), validate=True)
            old_plan = (
                base64.b64decode(str(payload["old_plan_b64"]), validate=True)
                if old_plan_exists
                else None
            )
        except Exception as exc:
            raise ProjectError("prepare journal base64 невалиден; ничего не изменено") from exc
        if len(old_project) > MAX_PROJECT_JSON_BYTES:
            raise ProjectError("prepare journal old project превышает 4 MiB")
        if old_plan is not None and len(old_plan) > MAX_PLAN_BYTES:
            raise ProjectError("prepare journal old plan превышает 8 MiB")
        if old_plan_exists:
            if payload.get("old_plan_b64") is None or not isinstance(payload.get("old_plan_sha256"), str) or not isinstance(payload.get("old_plan_canonical_sha256"), str):
                raise ProjectError("prepare journal plan metadata несогласован")
        else:
            if payload.get("old_plan_b64") is not None or payload.get("old_plan_sha256") is not None or payload.get("old_plan_canonical_sha256") is not None:
                raise ProjectError("prepare journal plan absence несогласован")
        if sha256_bytes(old_project) != payload.get("old_project_sha256"):
            raise ProjectError("prepare journal old project SHA256 mismatch")
        pair = old_project + b"\0" + (old_plan or b"")
        if sha256_bytes(pair) != payload.get("pair_sha256"):
            raise ProjectError("prepare journal pair digest mismatch")
        project = _validate_project_bytes(old_project, project_id=project_id)
        project_dir = self.directory(project_id)
        if project.get("id") != project_id:
            raise ProjectError("prepare journal project id mismatch")
        prepared = project.get("prepared")
        if old_plan_exists:
            assert old_plan is not None
            _validate_old_plan_bytes(
                old_plan,
                project_dir=project_dir,
                project=project,
                expected_raw_sha=str(payload["old_plan_sha256"]),
                expected_canonical_sha=str(payload["old_plan_canonical_sha256"]),
            )
            if not isinstance(prepared, Mapping):
                raise ProjectError("old plan exists без prepared metadata; recovery небезопасен")
            plan_sha = prepared.get("plan_sha256")
            if (
                not isinstance(plan_sha, str)
                or len(plan_sha) != 64
                or any(char not in "0123456789abcdef" for char in plan_sha.lower())
            ):
                raise ProjectError("prepared.plan_sha256 должен быть 64 hex")
            if plan_sha != str(payload["old_plan_canonical_sha256"]):
                raise ProjectError("prepared.plan_sha256 не совпадает с old canonical plan SHA")
            if prepared.get("plan_path") != "plan.json":
                raise ProjectError("prepared.plan_path должен быть stable plan.json")
        elif prepared is not None:
            raise ProjectError("prepared metadata exists без old plan")
        # All validation passed: only now touch the project pair.
        atomic_write_bytes(project_dir / "project.json", old_project)
        plan_path = project_dir / "plan.json"
        if old_plan_exists and old_plan is not None:
            atomic_write_bytes(plan_path, old_plan)
        elif checked_exists(plan_path, label="plan during recovery"):
            checked_unlink(plan_path, label="plan during recovery")
        self._clear_prepare_journal_locked(project_id)
        return True

    def recover_prepare_journal(self, project_id: str) -> bool:
        lock = self.lock(project_id)
        try:
            return self.recover_prepare_journal_locked(project_id)
        finally:
            lock.release()

    def list(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if not checked_exists(self.projects_dir, label="projects-dir"):
            return result
        for entry in sorted(self.projects_dir.iterdir(), key=lambda item: item.name):
            if not checked_is_dir(entry, label="project-dir"):
                continue
            try:
                lock = self.lock(entry.name)
                try:
                    self.recover_prepare_journal_locked(entry.name)
                    project = self._load_locked(entry.name)
                finally:
                    lock.release()
            except ProjectError as exc:
                result.append({"id": entry.name, "status": "INVALID", "error": str(exc)})
                continue
            result.append(
                {
                    "id": project.get("id"),
                    "title": project.get("title"),
                    "status": project.get("status"),
                    "revision": project.get("revision"),
                    "prepared": bool(project.get("prepared")),
                }
            )
        return result


__all__ = [
    "AUTHORING_EXCLUDE",
    "JOURNAL_SCHEMA",
    "LANDSCAPE",
    "PORTRAIT",
    "PREPARE_JOURNAL",
    "PROJECT_SCHEMA",
    "ProjectStore",
    "authoring_digest",
    "new_project",
    "now_utc",
]
