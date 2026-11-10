"""Explicit, non-interfering ComfyUI /free policy for KF1."""

from __future__ import annotations

from typing import Any, Mapping

from .comfy import ComfyClient, ComfyError, history_status
from .errors import ReleaseDeferred, StoreError
from .plan import ResolvedPlan


def _terminal(entry: Mapping[str, Any] | None) -> bool:
    if entry is None:
        return False
    status, error = history_status(entry)
    return status in {"success", "error", "failed", "cancelled"} or bool(error)


def release_plan(
    plan: ResolvedPlan,
    state: Mapping[str, Any],
    client: ComfyClient | None,
    *,
    execute: bool = False,
) -> dict[str, Any]:
    owned = sorted(
        {
            str(item["prompt_id"])
            for item in state.get("attempts", [])
            if isinstance(item, Mapping) and isinstance(item.get("prompt_id"), str)
        }
    )
    if not execute:
        return {
            "status": "PLAN_ONLY",
            "owned_prompts": owned,
            "policy": "queue must be empty and every owned prompt terminal",
            "vram_free_verified": False,
            "video_submitted": False,
        }
    if client is None:
        raise StoreError("release --execute требует ComfyClient")
    running, pending = client.queue_ids()
    if running or pending:
        raise ReleaseDeferred(
            "ComfyUI queue не пуст; /free не отправлен, foreign work не затрагивается"
        )
    for prompt_id in owned:
        entry = client.history(prompt_id)
        if not _terminal(entry):
            raise ReleaseDeferred(
                f"owned prompt {prompt_id} не terminal; /free не отправлен"
            )
    # The queue is stateless between this check and POST /free.  We cannot
    # guarantee exclusivity on a shared runtime; document that limitation and
    # never call /interrupt or kill a process.
    accepted = client.free()
    return {
        "status": "RELEASE_REQUESTED" if accepted else "RELEASE_NOT_ACCEPTED",
        "request_accepted": bool(accepted),
        "vram_free_verified": False,
        "owned_prompts": owned,
        "queue_empty_at_check": True,
        "video_submitted": False,
        "limitation": (
            "ComfyUI /free is a global queue flag; a foreign prompt can arrive "
            "between the empty-queue check and the flag being consumed. No "
            "exclusive ownership is claimed."
        ),
    }


__all__ = ["release_plan"]
