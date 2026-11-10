"""Durable per-plan/per-frame state for KF1.

Stable identity is the canonical plan PATH, not the editable content hash.  One
OS lock covers all revisions of that path; a global limits lock serializes the
shared generation ledger.  Every revision retains its own immutable plan
snapshot, frames and attempts so status/release can see old unresolved work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from anime_studio.lock import InstanceLock

from .errors import GenerationLimitError, StoreError

SCHEMA = "anime-studio/kf1-state/v1"
SAFE = re.compile(r"[^A-Za-z0-9._-]+")
CONSUMING = {"INTENT", "POSTING", "SUBMITTED", "AMBIGUOUS", "COMPLETE", "FAILED_TERMINAL"}
UNRESOLVED = {"INTENT", "POSTING", "SUBMITTED", "AMBIGUOUS"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: str) -> str:
    text = SAFE.sub("_", value)
    return text or "frame"


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def _validate_state(data: Mapping[str, Any], plan_sha256: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
        raise StoreError("state schema неверна")
    if data.get("plan_sha256") != plan_sha256:
        raise StoreError("state принадлежит другому revision")
    if not isinstance(data.get("plan_path"), str):
        raise StoreError("state.plan_path некорректен")
    if not isinstance(data.get("frames"), dict):
        raise StoreError("state.frames некорректен")
    for key in ("attempts", "invalidations"):
        if not isinstance(data.get(key), list):
            raise StoreError(f"state.{key} некорректен")
    return dict(data)


@contextmanager
def _blocking_lock(path: Path, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    lock = None
    while True:
        candidate = InstanceLock(path)
        try:
            candidate.acquire()
            lock = candidate
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise StoreError(f"global store lock timeout: {path}")
            time.sleep(0.01)
    try:
        yield lock
    finally:
        lock.release()


class PlanStore:
    def __init__(
        self, root: Path | str, plan_sha256: str, plan_path: Path | str | None = None
    ) -> None:
        if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
            raise StoreError("plan_sha256 некорректен")
        self.base_root = Path(root)
        self.base_root.mkdir(parents=True, exist_ok=True)
        canonical = (
            os.path.normcase(str(Path(plan_path).resolve(strict=False)))
            if plan_path is not None
            else plan_sha256
        )
        self.plan_path = canonical
        self.plan_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.limits_path = self.base_root / "limits.json"
        self.limits_lock_path = self.base_root / ".limits.lock"
        self.root = self.base_root / "plans" / self.plan_key
        self.root.mkdir(parents=True, exist_ok=True)
        self.revisions_root = self.root / "revisions"
        self.revisions_root.mkdir(parents=True, exist_ok=True)
        self.current_path = self.root / "current.json"
        self.lock_path = self.base_root / ".generate.lock"
        self.plan_sha256 = plan_sha256
        self.state_path = self.revisions_root / plan_sha256 / "state.json"
        self.frames_root = self.revisions_root / plan_sha256 / "frames"

    # ------------------------------------------------------------- revisions
    def _revision_state_path(self, plan_sha256: str) -> Path:
        return self.revisions_root / plan_sha256 / "state.json"

    def _read_state_file(self, plan_sha256: str) -> dict[str, Any] | None:
        path = self._revision_state_path(plan_sha256)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"state повреждён: {path}") from exc
        return _validate_state(data, plan_sha256)

    def _current_state(self) -> dict[str, Any] | None:
        if not self.current_path.is_file():
            return None
        try:
            current = json.loads(self.current_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        sha = current.get("plan_sha256") if isinstance(current, Mapping) else None
        if not isinstance(sha, str):
            return None
        return self._read_state_file(sha)

    def _write_current(self, plan_sha256: str) -> None:
        _atomic_json(
            self.current_path,
            {
                "plan_key": self.plan_key,
                "plan_path": self.plan_path,
                "plan_sha256": plan_sha256,
            },
        )

    def _snapshot_plan(self, plan: Any) -> None:
        revision_dir = self.revisions_root / plan.plan_sha256
        revision_dir.mkdir(parents=True, exist_ok=True)
        snapshot = revision_dir / "plan-snapshot.json"
        if not snapshot.is_file():
            _atomic_json(snapshot, plan.data)
        source = revision_dir / "plan-source.json"
        if not source.is_file():
            try:
                _atomic_bytes(source, Path(plan.path).read_bytes())
            except OSError:
                pass
        resolved = revision_dir / "resolved-plan.json"
        if not resolved.is_file():
            _atomic_json(
                resolved,
                {
                    "plan_path": str(plan.path),
                    "plan_sha256": plan.plan_sha256,
                    "refs": {
                        key: {
                            "path": str(value.path),
                            "sha256": value.sha256,
                            "size_bytes": value.size_bytes,
                            "width": value.width,
                            "height": value.height,
                            "format": value.format,
                        }
                        for key, value in plan.refs.items()
                    },
                    "settings": plan.settings,
                    "output_policy": plan.output_policy,
                    "pins": plan.pins,
                },
            )

    def _activate_paths(self, plan: Any) -> None:
        self.plan_sha256 = plan.plan_sha256
        self.state_path = self._revision_state_path(plan.plan_sha256)
        self.frames_root = self.revisions_root / plan.plan_sha256 / "frames"

    def assert_stable_plan_identity(self, plan: Any) -> None:
        target = os.path.normcase(str(Path(plan.path).resolve(strict=False)))
        for state in self.iter_states():
            stored = state.get("plan_path")
            if not isinstance(stored, str):
                raise StoreError("revision без plan_path")
            if os.path.normcase(str(Path(stored).resolve(strict=False))) != target:
                raise StoreError(
                    "revision plan_path не совпадает со stable plan identity"
                )

    def _unresolved_details(self, exclude_sha256: str | None) -> list[str]:
        """Scan ALL revisions, never only the current pointer."""

        details: list[str] = []
        for state in self.iter_states():
            if state.get("plan_sha256") == exclude_sha256:
                continue
            for attempt in state.get("attempts", []):
                if (
                    isinstance(attempt, Mapping)
                    and attempt.get("status") in UNRESOLVED
                ):
                    details.append(
                        f"{state.get('plan_sha256')}:{attempt.get('frame_id')}:"
                        f"{attempt.get('status')}:{attempt.get('prompt_id') or '-'}"
                    )
        return details

    def load(self, *, plan: Any) -> dict[str, Any]:
        """Activate a revision for mutation.

        The current pointer is written only after the global unresolved
        scan, so loading/exporting an older revision can never hide a newer
        unresolved attempt.
        """

        self._activate_paths(plan)
        self.assert_stable_plan_identity(plan)
        unresolved = self._unresolved_details(plan.plan_sha256)
        if unresolved:
            raise StoreError(
                "Нельзя начать новую revision плана: есть unresolved intent старой "
                f"revision ({', '.join(unresolved)}). "
                "Восстановите exact prompt или сделайте explicit retry/owner решение; "
                "старые данные не забываются."
            )
        existing = self._read_state_file(plan.plan_sha256)
        if existing is not None:
            self._snapshot_plan(plan)
            self._write_current(plan.plan_sha256)
            return existing
        self._snapshot_plan(plan)
        state = self.new_state(plan=plan)
        self.save(state)
        return state

    def load_any(self, *, plan: Any) -> dict[str, Any]:
        """Read-only current revision for status; never creates/mutates state."""

        self._activate_paths(plan)
        current = self._current_state()
        if current is None:
            return self.new_state(plan=plan)
        return current

    def load_revision_readonly(self, *, plan: Any) -> dict[str, Any]:
        """Read a specific historical revision without activating it."""

        self._activate_paths(plan)
        self.assert_stable_plan_identity(plan)
        state = self._read_state_file(plan.plan_sha256)
        if state is not None:
            return state
        return self.new_state(plan=plan)

    def current_revision_sha256(self) -> str | None:
        current = self._current_state()
        return str(current.get("plan_sha256")) if current is not None else None

    def iter_states(self, *, plan: Any = None) -> list[dict[str, Any]]:
        states = []
        if not self.revisions_root.is_dir():
            return states
        for folder in sorted(self.revisions_root.iterdir()):
            if not folder.is_dir():
                continue
            state = self._read_state_file(folder.name)
            if state is not None:
                states.append(state)
        return states

    def all_prompt_ids_all(self, *, plan: Any) -> list[str]:
        result = set()
        for state in self.iter_states(plan=plan):
            result.update(self.all_prompt_ids(state))
        return sorted(result)

    def lock(self) -> InstanceLock:
        return InstanceLock(self.lock_path)

    def new_state(self, *, plan: Any) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "plan_sha256": plan.plan_sha256,
            "plan_path": str(plan.path),
            "revision_id": plan.plan_sha256,
            "plan_snapshot": str(
                (self.revisions_root / plan.plan_sha256 / "plan-source.json").resolve()
            ),
            "profile_id": plan.pins["profile_id"],
            "generation_limit": plan.generation_limit,
            "extra_limit": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "frames": {},
            "attempts": [],
            "invalidations": [],
        }

    def save(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["updated_at"] = _now()
        _atomic_json(self.state_path, payload)
        self._write_current(str(payload["plan_sha256"]))

    # ------------------------------------------------------------- ledger
    @staticmethod
    def _validate_ledger_entry(plan_key: str, entry: Any) -> dict[str, Any]:
        if not isinstance(plan_key, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_key):
            raise StoreError("limits.json содержит некорректный plan_key")
        if not isinstance(entry, Mapping):
            raise StoreError(f"limits.json entry {plan_key} не объект")
        count = entry.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StoreError(f"limits.json count {plan_key} должен быть неотрицательным int")
        attempt_ids = entry.get("attempt_ids")
        if not isinstance(attempt_ids, list):
            raise StoreError(f"limits.json attempt_ids {plan_key} должен быть списком")
        if any(not isinstance(item, str) or not item for item in attempt_ids):
            raise StoreError(f"limits.json attempt_ids {plan_key} содержит нестроку")
        if len(attempt_ids) != len(set(attempt_ids)):
            raise StoreError(f"limits.json attempt_ids {plan_key} содержит дубликаты")
        if count != len(attempt_ids):
            raise StoreError(
                f"limits.json count/attempt_ids {plan_key} не совпадают: {count} != {len(attempt_ids)}"
            )
        for key in ("plan_sha256", "plan_path", "updated_at"):
            if key in entry and not isinstance(entry[key], str):
                raise StoreError(f"limits.json {plan_key}.{key} должен быть строкой")
        if "plan_sha256" in entry and not re.fullmatch(
            r"[0-9a-f]{64}", str(entry["plan_sha256"])
        ):
            raise StoreError(f"limits.json {plan_key}.plan_sha256 некорректен")
        return dict(entry)

    def _ledger(self) -> dict[str, dict[str, Any]]:
        if not self.limits_path.is_file():
            return {}
        try:
            data = json.loads(self.limits_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise StoreError("limits.json повреждён")
        if not isinstance(data, dict):
            raise StoreError("limits.json должен быть JSON-объектом")
        return {
            str(plan_key): self._validate_ledger_entry(str(plan_key), entry)
            for plan_key, entry in data.items()
        }

    def ledger_count(self) -> int:
        with _blocking_lock(self.limits_lock_path):
            entry = self._ledger().get(self.plan_key)
        if entry is None:
            return 0
        return int(entry["count"])

    def _ledger_reserve(self, attempt_id: str, plan_sha256: str) -> None:
        with _blocking_lock(self.limits_lock_path):
            data = self._ledger()
            entry = data.get(self.plan_key)
            if entry is None:
                entry = {"count": 0, "attempt_ids": []}
            entry = dict(entry)
            entry["count"] = int(entry["count"]) + 1
            entry["attempt_ids"] = list(entry["attempt_ids"]) + [attempt_id]
            entry["plan_sha256"] = plan_sha256
            entry["plan_path"] = self.plan_path
            entry["updated_at"] = _now()
            data[self.plan_key] = self._validate_ledger_entry(self.plan_key, entry)
            _atomic_json(self.limits_path, data)

    # ------------------------------------------------------------- frames
    def frame_dir(self, frame_id: str, version: int) -> Path:
        scene, _, role = frame_id.partition("/")
        path = self.frames_root / _safe(scene) / _safe(role or "A") / f"v{int(version):04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def existing_versions(self, frame_id: str) -> list[int]:
        scene, _, role = frame_id.partition("/")
        folder = self.frames_root / _safe(scene) / _safe(role or "A")
        if not folder.is_dir():
            return []
        values = []
        for child in folder.iterdir():
            match = re.fullmatch(r"v(\d{4})", child.name)
            if match and child.is_dir():
                values.append(int(match.group(1)))
        return sorted(values)

    def next_version(self, frame_id: str) -> int:
        versions = self.existing_versions(frame_id)
        return (max(versions) + 1) if versions else 1

    def attempt_count(self, state: Mapping[str, Any]) -> int:
        local = sum(
            1
            for attempt in state.get("attempts", [])
            if isinstance(attempt, Mapping) and attempt.get("status") in CONSUMING
        )
        return max(local, self.ledger_count())

    def begin_attempt(
        self,
        state: dict[str, Any],
        *,
        frame_id: str,
        version: int,
        intent: Mapping[str, Any],
        allow_global_attempt_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        limit = int(state.get("generation_limit", 0)) + int(state.get("extra_limit", 0))
        if self.attempt_count(state) >= limit:
            raise GenerationLimitError(
                f"Достигнут лимит генераций {limit}; нужен explicit owner --raise-limit"
            )
        self.assert_global_clean_for_new_work(
            allow_attempt_ids=allow_global_attempt_ids
        )
        attempt_id = uuid.uuid4().hex
        record = {
            "attempt_id": attempt_id,
            "frame_id": frame_id,
            "version": int(version),
            "status": "INTENT",
            "created_at": _now(),
            "updated_at": _now(),
            **dict(intent),
        }
        state.setdefault("attempts", []).append(record)
        self._ledger_reserve(attempt_id, str(state.get("plan_sha256", "")))
        state.setdefault("frames", {}).setdefault(
            frame_id, {"current_version": 0, "versions": {}}
        )
        self.save(state)
        return record

    def update_attempt(
        self, state: dict[str, Any], attempt_id: str, *, status: str, **updates: Any
    ) -> dict[str, Any]:
        for attempt in state.get("attempts", []):
            if attempt.get("attempt_id") == attempt_id:
                attempt["status"] = status
                attempt["updated_at"] = _now()
                attempt.update(updates)
                self.save(state)
                return attempt
        raise StoreError("attempt не найден")

    def complete_attempt(
        self,
        state: dict[str, Any],
        attempt_id: str,
        *,
        output: Mapping[str, Any],
        binding_sha256: str,
        source_hashes: Mapping[str, Any],
        receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One atomic mutation for attempt COMPLETE + frame current + receipt."""
        attempt = None
        for item in state.get("attempts", []):
            if item.get("attempt_id") == attempt_id:
                attempt = item
                break
        if attempt is None:
            raise StoreError("attempt не найден")
        frame_id = str(attempt["frame_id"])
        version = int(attempt["version"])
        now = _now()
        attempt.update(
            {
                "status": "COMPLETE",
                "updated_at": now,
                "completed_at": now,
                "output": dict(output),
                "binding_sha256": binding_sha256,
                "source_hashes": dict(source_hashes),
            }
        )
        frame = state.setdefault("frames", {}).setdefault(
            frame_id, {"current_version": 0, "versions": {}}
        )
        frame["current_version"] = max(int(frame.get("current_version", 0)), version)
        frame.setdefault("versions", {})[str(version)] = {
            "status": "COMPLETE",
            "attempt_id": attempt_id,
            "output": dict(output),
            "binding_sha256": binding_sha256,
            "source_hashes": dict(source_hashes),
            "completed_at": now,
            "plan_sha256": state.get("plan_sha256"),
            "revision_id": state.get("revision_id"),
            "receipt": dict(receipt or {}),
        }
        self.save(state)
        return attempt

    def mark_invalidated(self, state: dict[str, Any], frame_ids: list[str], reason: str) -> None:
        if not frame_ids:
            return
        state.setdefault("invalidations", []).append(
            {"at": _now(), "frames": sorted(set(frame_ids)), "reason": reason}
        )
        for frame_id in frame_ids:
            frame = state.setdefault("frames", {}).setdefault(
                frame_id, {"current_version": 0, "versions": {}}
            )
            frame["invalidated"] = True
            frame["invalid_reason"] = reason
        self.save(state)

    def clear_invalidated(self, state: dict[str, Any], frame_id: str) -> None:
        frame = state.setdefault("frames", {}).setdefault(
            frame_id, {"current_version": 0, "versions": {}}
        )
        frame.pop("invalidated", None)
        frame.pop("invalid_reason", None)
        self.save(state)

    def all_prompt_ids(self, state: Mapping[str, Any]) -> list[str]:
        result = []
        for attempt in state.get("attempts", []):
            if isinstance(attempt, Mapping) and isinstance(attempt.get("prompt_id"), str):
                result.append(attempt["prompt_id"])
        return sorted(set(result))

    def attempt_by_id(self, state: Mapping[str, Any], attempt_id: str) -> dict[str, Any] | None:
        for attempt in state.get("attempts", []):
            if isinstance(attempt, Mapping) and attempt.get("attempt_id") == attempt_id:
                return dict(attempt)
        return None

    def attempts_for_frame(self, state: Mapping[str, Any], frame_id: str) -> list[dict[str, Any]]:
        return [
            dict(attempt)
            for attempt in state.get("attempts", [])
            if isinstance(attempt, Mapping) and attempt.get("frame_id") == frame_id
        ]

    def version_dir(self, frame_id: str, version: int) -> Path:
        scene, _, role = frame_id.partition("/")
        return self.frames_root / _safe(scene) / _safe(role or "A") / f"v{int(version):04d}"

    def ensure_version_dir(self, frame_id: str, version: int) -> Path:
        path = self.version_dir(frame_id, version)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------- global safety scan
    @staticmethod
    def _state_at(path: Path, expected_sha: str) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"state повреждён: {path}") from exc
        return _validate_state(data, expected_sha)

    def _iter_all_states(self) -> list[tuple[str, dict[str, Any]]]:
        """Read every known same-root plan/revision state, not only self.

        Only 64-hex plan_key/revision directory names are treated as our format;
        arbitrary extraneous files are ignored.  A malformed known state fails
        closed.
        """

        plans_root = self.base_root / "plans"
        if not plans_root.is_dir():
            return []
        result: list[tuple[str, dict[str, Any]]] = []
        for plan_dir in sorted(plans_root.iterdir()):
            if not plan_dir.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", plan_dir.name):
                continue
            current_path = plan_dir / "current.json"
            if current_path.is_file():
                try:
                    current = json.loads(current_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise StoreError(f"current.json повреждён: {current_path}") from exc
                sha = current.get("plan_sha256") if isinstance(current, Mapping) else None
                if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                    raise StoreError(f"current.json содержит некорректный plan_sha256: {current_path}")
                if not (plan_dir / "revisions" / sha / "state.json").is_file():
                    raise StoreError(f"current revision state отсутствует: {current_path}")
            revisions = plan_dir / "revisions"
            if not revisions.is_dir():
                continue
            for revision_dir in sorted(revisions.iterdir()):
                if not revision_dir.is_dir() or not re.fullmatch(
                    r"[0-9a-f]{64}", revision_dir.name
                ):
                    continue
                state_path = revision_dir / "state.json"
                if not state_path.is_file():
                    raise StoreError(f"revision без state.json: {state_path}")
                result.append((plan_dir.name, self._state_at(state_path, revision_dir.name)))
        return result

    def global_unresolved_report(self) -> dict[str, Any]:
        states = self._iter_all_states()
        ledger = self._ledger()
        unresolved: list[dict[str, Any]] = []
        ledger_issues: list[str] = []
        attempts_by_plan: dict[str, dict[str, Mapping[str, Any]]] = {}
        for plan_key, state in states:
            bucket = attempts_by_plan.setdefault(plan_key, {})
            for attempt in state.get("attempts", []):
                if not isinstance(attempt, Mapping):
                    raise StoreError("state attempt не объект")
                attempt_id = attempt.get("attempt_id")
                if not isinstance(attempt_id, str) or not attempt_id:
                    raise StoreError("state attempt без attempt_id")
                if attempt_id in bucket:
                    raise StoreError(
                        f"state {plan_key} содержит duplicate attempt_id {attempt_id}"
                    )
                bucket[attempt_id] = attempt
                if attempt.get("status") not in CONSUMING:
                    continue
                if attempt.get("status") in UNRESOLVED:
                    unresolved.append(
                        {
                            "plan_key": plan_key,
                            "plan_sha256": state.get("plan_sha256"),
                            "frame_id": attempt.get("frame_id"),
                            "attempt_id": attempt_id,
                            "status": attempt.get("status"),
                            "prompt_id": attempt.get("prompt_id"),
                        }
                    )
                entry = ledger.get(plan_key)
                if not isinstance(entry, Mapping) or attempt_id not in set(
                    entry.get("attempt_ids") or []
                ):
                    ledger_issues.append(
                        f"{plan_key}:{attempt_id}:{attempt.get('status')}: "
                        "budget-consuming attempt без exact ledger reservation"
                    )
        for plan_key, entry in ledger.items():
            entry = self._validate_ledger_entry(plan_key, entry)
            known = attempts_by_plan.get(plan_key, {})
            for attempt_id in entry.get("attempt_ids") or []:
                if attempt_id not in known:
                    ledger_issues.append(
                        f"{plan_key}:{attempt_id}: orphan reservation без state intent"
                    )
            if entry.get("attempt_ids") and plan_key not in attempts_by_plan:
                ledger_issues.append(f"{plan_key}: ledger без state directory")
        for plan_key, attempts in attempts_by_plan.items():
            entry = ledger.get(plan_key)
            reserved = set(entry.get("attempt_ids") or []) if isinstance(entry, Mapping) else set()
            for attempt_id, attempt in attempts.items():
                if attempt.get("status") in CONSUMING and attempt_id not in reserved:
                    ledger_issues.append(
                        f"{plan_key}:{attempt_id}:{attempt.get('status')}: "
                        "budget-consuming attempt без ledger reservation"
                    )
        return {
            "unresolved": unresolved,
            "ledger_issues": sorted(set(ledger_issues)),
            "state_count": len(states),
        }

    def assert_global_clean_for_new_work(
        self, *, allow_attempt_ids: set[str] | None = None
    ) -> None:
        """Fail closed before new upload/reserve/POST anywhere in this state root."""
        allow = set(allow_attempt_ids or set())
        report = self.global_unresolved_report()
        remaining = [
            item for item in report["unresolved"] if item["attempt_id"] not in allow
        ]
        if remaining or report["ledger_issues"]:
            details = [
                f"{item['plan_sha256']}:{item['frame_id']}:{item['status']}:{item['attempt_id']}"
                for item in remaining
            ] + list(report["ledger_issues"])
            raise StoreError(
                "Глобальный safety scan state root не пуст; новый upload/POST запрещён: "
                + "; ".join(details)
            )

    # ------------------------------------------------------------- approval
    def approval_path(self, frame_id: str, version: int) -> Path:
        return self.version_dir(frame_id, version) / "approval.json"

    def read_approval(self, frame_id: str, version: int) -> dict[str, Any] | None:
        path = self.approval_path(frame_id, version)
        if not path.is_file():
            return None
        if path.is_symlink():
            raise StoreError("approval.json не должен быть symlink")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreError(f"approval.json повреждён: {path}") from exc
        if not isinstance(data, dict):
            raise StoreError("approval.json должен быть объектом")
        if data.get("schema") != "anime-studio/kf1-approval/v1":
            raise StoreError("approval.json schema неверна")
        return data

    def write_approval(
        self, frame_id: str, version: int, payload: Mapping[str, Any]
    ) -> Path:
        path = self.approval_path(frame_id, version)
        if path.is_symlink():
            raise StoreError("approval.json не должен быть symlink")
        _atomic_json(path, payload)
        return path

    def assert_artifact_contained(self, path: Path | str) -> Path:
        raw = Path(path)
        if raw.is_symlink():
            raise StoreError("artifact не должен быть symlink")
        try:
            resolved = raw.resolve(strict=True)
        except OSError as exc:
            raise StoreError(f"artifact отсутствует: {raw}") from exc
        root = self.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise StoreError("artifact path escapes plan store")
        return resolved


__all__ = ["CONSUMING", "PlanStore", "SCHEMA", "UNRESOLVED"]

