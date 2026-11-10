"""Contained, race-aware filesystem helpers for KF2 project authoring.

These helpers are deliberately offline. They never create network clients and
never read credentials.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .errors import AssetError, ProjectError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTS_DIR = ROOT / "data" / "projects"
MAX_PROJECT_JSON_BYTES = 4 * 1024 * 1024
MAX_PLAN_BYTES = 8 * 1024 * 1024
MAX_JOURNAL_BYTES = 32 * 1024 * 1024
MAX_SCRIPT_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PROJECT_BYTES = 512 * 1024 * 1024
MAX_ASSETS = 1000
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
SENSITIVE_NAME = re.compile(
    r"^(credentials?|secrets?|private|id_rsa|id_ed25519|token|password|passwd|key)$",
    re.IGNORECASE,
)
SENSITIVE_EXTENSIONS = {".pem", ".key", ".pfx", ".p12", ".crt", ".der"}
FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
DRIVE_REMOTE = 4
_WINDOWS_BAD_CHARS = set('*?"<>|')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    assert_no_reparse_ancestors(path)
    return sha256_bytes(read_bytes_limited(path, max_bytes=MAX_PROJECT_BYTES, label="file"))


def strip_wrapping_quotes(value: str) -> str:
    """Remove one matching pair of surrounding quotes, never inner quotes."""

    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def is_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)


def _lexical_absolute(path: Path) -> str:
    return os.path.abspath(str(path))


def assert_no_reparse_ancestors(path: Path, *, stop: Path | None = None) -> None:
    """Reject links component-by-component from lexical root to target.

    A junction must be detected before any descendant is stat'ed. No
    Path.resolve/exists is used here; only lstat on already-safe prefixes.
    """

    current = _lexical_absolute(path)
    stop_abs = os.path.normcase(_lexical_absolute(stop)) if stop is not None else None
    drive, tail = os.path.splitdrive(current)
    prefix = (drive + os.sep) if drive else os.sep
    if is_reparse(Path(prefix)):
        raise ProjectError(f"unsafe symlink/junction in path: {prefix}")
    for component in tail.replace("\\", "/").split("/"):
        if not component:
            continue
        prefix = os.path.join(prefix, component)
        if is_reparse(Path(prefix)):
            raise ProjectError(f"unsafe symlink/junction in path: {prefix}")
        if stop_abs is not None and os.path.normcase(prefix) == stop_abs:
            break


def assert_contained(root: Path, path: Path) -> Path:
    """Lexical managed-root containment; no filesystem resolution."""

    root_abs = os.path.normcase(_lexical_absolute(root))
    path_abs = os.path.normcase(_lexical_absolute(path))
    try:
        common = os.path.commonpath([root_abs, path_abs])
    except ValueError as exc:
        raise ProjectError(f"path escapes managed root: {path}") from exc
    if common != root_abs:
        raise ProjectError(f"path escapes managed root: {path}")
    return Path(_lexical_absolute(path))


def _gate(path: Path, *, label: str) -> None:
    try:
        assert_no_reparse_ancestors(path)
    except ProjectError as exc:
        raise AssetError(f"{label}: {exc}") from exc


def checked_exists(path: Path, *, label: str = "path") -> bool:
    _gate(path, label=label)
    try:
        os.lstat(path)
        return True
    except OSError:
        return False


def checked_is_dir(path: Path, *, label: str = "path") -> bool:
    _gate(path, label=label)
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if is_reparse(path):
        raise AssetError(f"{label}: symlink/junction запрещён: {path}")
    return stat.S_ISDIR(info.st_mode)


def checked_is_file(path: Path, *, label: str = "path") -> bool:
    _gate(path, label=label)
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if is_reparse(path):
        raise AssetError(f"{label}: symlink/junction запрещён: {path}")
    return stat.S_ISREG(info.st_mode)


def checked_unlink(path: Path, *, label: str = "path") -> bool:
    _gate(path, label=label)
    try:
        os.lstat(path)
    except OSError:
        return False
    if is_reparse(path):
        raise AssetError(f"{label}: symlink/junction запрещён: {path}")
    os.unlink(path)
    return True


def ensure_managed_dir(root: Path, path: Path, *, label: str = "directory") -> Path:
    """Create/check a directory below a managed root, rejecting link ancestry."""

    path = validate_local_destination(path, label=label)
    assert_contained(root, path)
    assert_no_reparse_ancestors(path.parent)
    path.mkdir(parents=True, exist_ok=True)
    assert_no_reparse_ancestors(path)
    if is_reparse(path) or not path.is_dir():
        raise ProjectError(f"unsafe {label}: {path}")
    return path


def validate_project_id(value: str) -> str:
    text = strip_wrapping_quotes(value)
    if not isinstance(text, str) or not SAFE_ID.fullmatch(text):
        raise ProjectError(
            "project id должен быть lower-case [a-z0-9][a-z0-9._-]{0,63}"
        )
    if text in {".", ".."} or text.endswith((".", " ")):
        raise ProjectError(f"недопустимый project id: {text!r}")
    if text.split(".", 1)[0].lower() in RESERVED_NAMES:
        raise ProjectError(f"зарезервированное Windows-имя project id: {text!r}")
    return text


def validate_local_path_text(value: str, *, label: str = "path") -> str:
    text = strip_wrapping_quotes(value)
    if not text:
        raise AssetError(f"{label} пуст")
    if "\x00" in text:
        raise AssetError(f"{label} содержит NUL")
    for char in text:
        if ord(char) < 32 or ord(char) == 127:
            raise AssetError(f"{label} содержит control character")
    if text.startswith(("\\\\", "//")):
        raise AssetError(f"{label}: UNC/сетевые пути запрещены")
    if text.startswith(("\\\\.\\", "\\\\?\\")):
        raise AssetError(f"{label}: device paths запрещены")
    if os.name == "nt":
        if any(char in text for char in _WINDOWS_BAD_CHARS):
            raise AssetError(f"{label}: недопустимый Windows-символ")
        colon_positions = [index for index, char in enumerate(text) if char == ":"]
        for index in colon_positions:
            if index != 1 or len(text) < 3 or text[2] not in "\\/":
                raise AssetError(f"{label}: ADS/drive-relative путь запрещён")
        for component in re.split(r"[\\/]", text):
            if not component:
                continue
            if component.endswith((".", " ")):
                raise AssetError(f"{label}: компонент заканчивается точкой/пробелом")
            if component.split(".", 1)[0].lower() in RESERVED_NAMES:
                raise AssetError(f"{label}: зарезервированное Windows-имя")
    return text


def validate_local_destination(path: Path, *, label: str = "path") -> Path:
    """Lexical/local-drive gate for a destination before any stat/mkdir."""

    text = validate_local_path_text(str(path), label=label)
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if drive_type_is_remote(candidate):
        raise AssetError(f"{label}: сетевой/mapped drive запрещён")
    try:
        assert_no_reparse_ancestors(candidate)
    except ProjectError as exc:
        raise AssetError(f"{label}: {exc}") from exc
    if is_reparse(candidate):
        raise AssetError(f"{label}: symlink/junction запрещён")
    return candidate


def validate_source_name(path: Path, *, label: str = "file") -> None:
    name = path.name.lower()
    if name in {".env", ".env.local"} or name.startswith(".env."):
        raise AssetError(f"{label}: .env/секреты запрещены")
    if path.suffix.lower() in SENSITIVE_EXTENSIONS or SENSITIVE_NAME.search(path.stem):
        raise AssetError(f"{label}: имя похоже на credentials/секрет")


def drive_type_is_remote(path: Path) -> bool:
    if os.name != "nt":
        return False
    text = _lexical_absolute(path)
    drive, rest = os.path.splitdrive(text)
    if (
        len(drive) != 2
        or drive[1] != ":"
        or not drive[0].isalpha()
        or not rest.startswith(("\\", "/"))
    ):
        raise AssetError(f"invalid drive/root: {path}")
    root = drive + os.sep
    try:
        value = int(ctypes.windll.kernel32.GetDriveTypeW(root))
    except Exception as exc:
        raise AssetError(f"GetDriveTypeW failed for {root}") from exc
    if value == DRIVE_REMOTE:
        return True
    if value in {0, 1} or value not in {2, 3, 5, 6}:
        raise AssetError(f"unknown/invalid drive type {value} for {root}")
    return False


def resolve_local_source(
    value: str,
    *,
    label: str,
    allowed_suffixes: set[str] | None = None,
) -> Path:
    """Resolve one local regular file with the same gate for image/text/provenance."""

    text = validate_local_path_text(value, label=label)
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if drive_type_is_remote(candidate):
        raise AssetError(f"{label}: сетевой/mapped drive запрещён")
    # Root-to-leaf gate must run before the final is_reparse/lstat probe.
    try:
        assert_no_reparse_ancestors(candidate)
    except ProjectError as exc:
        raise AssetError(f"{label}: {exc}") from exc
    if is_reparse(candidate):
        raise AssetError(f"{label}: symlink/junction запрещён")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AssetError(f"{label} не найден: {text}") from exc
    try:
        assert_no_reparse_ancestors(resolved)
    except ProjectError as exc:
        raise AssetError(f"{label}: {exc}") from exc
    if is_reparse(resolved):
        raise AssetError(f"{label}: symlink/junction запрещён")
    try:
        info = os.lstat(resolved)
    except OSError as exc:
        raise AssetError(f"{label} не найден: {text}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise AssetError(f"{label} должен быть обычным файлом")
    if drive_type_is_remote(resolved):
        raise AssetError(f"{label}: сетевой/mapped drive запрещён")
    validate_source_name(resolved, label=label)
    if allowed_suffixes is not None and resolved.suffix.lower() not in allowed_suffixes:
        raise AssetError(f"{label}: неподдерживаемое расширение {resolved.suffix or '<none>'}")
    return resolved

def read_bytes_limited(
    path: Path, *, max_bytes: int, label: str = "file"
) -> bytes:
    try:
        assert_no_reparse_ancestors(path)
    except ProjectError as exc:
        raise AssetError(f"{label}: {exc}") from exc
    if is_reparse(path):
        raise AssetError(f"{label} не должен быть symlink/junction: {path}")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise AssetError(f"{label} не найден: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise AssetError(f"{label} должен быть обычным файлом: {path}")
    if before.st_size < 1 or before.st_size > max_bytes:
        raise AssetError(f"{label} вне допустимого размера 1..{max_bytes} bytes")
    try:
        with open(path, "rb") as handle:
            start = os.fstat(handle.fileno())
            data = handle.read(max_bytes + 1)
            end = os.fstat(handle.fileno())
    except OSError as exc:
        raise AssetError(f"{label} не читается: {path}") from exc
    if len(data) > max_bytes:
        raise AssetError(f"{label} превышает {max_bytes} bytes")
    if len(data) != start.st_size or end.st_size != start.st_size:
        raise AssetError(f"{label} изменён во время чтения: {path}")
    if end.st_mtime_ns != start.st_mtime_ns:
        raise AssetError(f"{label} изменён во время чтения: {path}")
    try:
        after = os.lstat(path)
    except OSError as exc:
        raise AssetError(f"{label} исчез во время чтения: {path}") from exc
    if is_reparse(path) or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        raise AssetError(f"{label} подменён symlink/junction во время чтения: {path}")
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise AssetError(f"{label} изменён во время чтения: {path}")
    return data


def _temp_in_parent(path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    return Path(handle.name)


def atomic_write_bytes(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    path = validate_local_destination(path, label="destination")
    assert_no_reparse_ancestors(path)
    parent = path.parent
    assert_no_reparse_ancestors(parent)
    parent.mkdir(parents=True, exist_ok=True)
    assert_no_reparse_ancestors(parent)
    if is_reparse(parent):
        raise ProjectError(f"unsafe parent symlink/junction: {parent}")
    if is_reparse(path):
        raise ProjectError(f"refusing to overwrite symlink/junction: {path}")
    if exclusive:
        if path.exists():
            raise FileExistsError(str(path))
        try:
            with open(path, "xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise
        except OSError as exc:
            raise ProjectError(f"cannot create {path}: {exc}") from exc
        return
    temp = _temp_in_parent(path)
    try:
        if is_reparse(temp):
            raise ProjectError(f"unsafe temporary path: {temp}")
        with open(temp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def read_json_limited(path: Path, *, max_bytes: int = MAX_PROJECT_JSON_BYTES) -> Any:
    data = read_bytes_limited(path, max_bytes=max_bytes, label="JSON")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectError(f"JSON не читается: {path}: {type(exc).__name__}") from exc


__all__ = [
    "DEFAULT_PROJECTS_DIR",
    "MAX_ASSETS",
    "MAX_IMAGE_BYTES",
    "MAX_PROJECT_BYTES",
    "MAX_PROJECT_JSON_BYTES",
    "MAX_SCRIPT_BYTES",
    "ROOT",
    "SAFE_ID",
    "assert_contained",
    "checked_exists",
    "checked_is_dir",
    "checked_is_file",
    "checked_unlink",
    "assert_no_reparse_ancestors",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "drive_type_is_remote",
    "ensure_managed_dir",
    "is_reparse",
    "read_bytes_limited",
    "read_json_limited",
    "resolve_local_source",
    "sha256_bytes",
    "sha256_file",
    "strip_wrapping_quotes",
    "validate_local_path_text",
    "validate_project_id",
    "validate_source_name",
]
