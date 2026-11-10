"""Shared deterministic helpers for I1.

The module intentionally depends only on the standard library so the same files
can be staged into the owned remote root and executed there.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(dict(payload)))


def json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def atomic_write(path: Path | str, data: bytes, *, mode: int = 0o600) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        try:
            os.chmod(target, mode)
        except OSError:
            pass
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    atomic_write(path, json_text(payload).encode("utf-8"))


def read_json(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON {path} must be an object")
    return value


def validate_name(value: Any, name: str = "name") -> str:
    if not isinstance(value, str) or not NAME_RE.fullmatch(value):
        raise ValueError(f"{name} must match [A-Za-z0-9._-] and start alphanumeric")
    return value


def validate_posix_absolute(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty path")
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) != value:
        raise ValueError(f"{name} must be a normalized absolute POSIX path")
    if any(part in ("", ".", "..") for part in path.parts[1:]):
        raise ValueError(f"{name} contains traversal/empty segments")
    return str(path)


def validate_relative_posix(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{name} must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{name} is not normalized/contained")
    return str(path)


def validate_workspace(value: Any) -> str:
    text = validate_posix_absolute(value, "workspace")
    if text == "/" or text.startswith("/proc") or text.startswith("/sys"):
        raise ValueError("workspace must be a normal writable directory")
    return text


def redact(text: str, *, limit: int = 2000) -> str:
    """Bound diagnostics and remove secret *values*, not just their labels."""

    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", "")
    text = re.sub(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
        "[REDACTED]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(
        r"(?i)\b(token|api[_-]?key|password|passphrase|secret|access[_-]?token|client[_-]?secret)\b(\s*[=:]\s*)([\"']?)[^\s\"'&;,]+",
        r"\1\2\3[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)([?&](?:token|api[_-]?key|access[_-]?token|password|secret)=)[^&\s]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(https?://)[^/@\s]+@", r"\1[REDACTED]@", text)
    for marker in ("BEGIN OPENSSH PRIVATE KEY", "PRIVATE KEY", "Authorization:"):
        if marker.lower() in text.lower():
            text = re.sub(re.escape(marker), "[REDACTED]", text, flags=re.IGNORECASE)
    if len(text) > limit:
        text = text[:limit] + "...[truncated]"
    return text


def format_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(number)} {unit}"
            return f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{number:.1f} TiB"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
