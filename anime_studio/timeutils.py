"""Small UTC/time helpers. Deadlines are persisted as epoch milliseconds."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def now_ms() -> int:
    return int(time.time() * 1000)


def now_iso() -> str:
    return ms_to_iso(now_ms())


def ms_to_iso(value_ms: int) -> str:
    return (
        datetime.fromtimestamp(value_ms / 1000.0, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def iso_to_ms(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    return int(datetime.fromisoformat(normalized).timestamp() * 1000)
