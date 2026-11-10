"""Validated HTTP input models and canonical payload hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MockOptions(BaseModel):
    """Explicit mock-only controls. They never cause a GPU or network call."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    duration_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    fail: bool = False
    fail_at_progress: int = Field(default=50, ge=1, le=100)
    mode: Literal["mock"] = "mock"


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    prompt: str = Field(min_length=1, max_length=4000)
    timeout_seconds: int = Field(default=60, ge=1, le=3600)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    mock_options: MockOptions = Field(default_factory=MockOptions)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class CancelResponseModel(BaseModel):
    """Documentation-only shape; routes return JSON dictionaries."""

    job_id: str
    status: str


def canonical_payload(request: JobRequest) -> tuple[str, str]:
    """Return canonical JSON and its sha256.

    The idempotency key is deliberately not part of this value. Fields are
    sorted, omitted optional None values are removed, and the schema version is
    included so a future payload format can be distinguished.
    """

    body: dict[str, Any] = request.model_dump(mode="json", exclude_none=True)
    wrapped = {"payload_version": 1, **body}
    text = json.dumps(
        wrapped,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest
