"""Runtime settings. No .env loading and no secrets are read here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Settings:
    db_path: Path
    artifacts_dir: Path
    lock_path: Path | None = None
    poll_interval: float = 0.2
    enable_worker: bool = True
    lease_seconds: float = 5.0
    adapter_mode: Literal["mock", "comfy-http"] = "mock"
    allow_live_transport: bool = False
    comfy_config_path: Path | None = None

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.adapter_mode not in {"mock", "comfy-http"}:
            raise ValueError("adapter_mode must be 'mock' or 'comfy-http'")
        if self.comfy_config_path is not None:
            self.comfy_config_path = Path(self.comfy_config_path)
        if self.adapter_mode == "comfy-http":
            if not self.allow_live_transport:
                raise ValueError(
                    "comfy-http requires an explicit allow_live_transport=True opt-in"
                )
            if self.comfy_config_path is None:
                raise ValueError("comfy-http requires comfy_config_path")
        if self.lock_path is None:
            self.lock_path = Path(str(self.db_path) + ".lock")
        else:
            self.lock_path = Path(self.lock_path)
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
