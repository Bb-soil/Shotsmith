"""Job state machine for P0.

Terminal states are immutable. Every transition that changes status must go
through Orchestrator._transition(), which checks this table inside a SQLite
write transaction.
"""

from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        JobStatus.SUCCEEDED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
        JobStatus.TIMED_OUT.value,
    }
)

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    JobStatus.QUEUED.value: frozenset(
        {
            JobStatus.DISPATCHING.value,
            JobStatus.CANCELLED.value,
            JobStatus.TIMED_OUT.value,
            JobStatus.FAILED.value,
        }
    ),
    JobStatus.DISPATCHING.value: frozenset(
        {
            JobStatus.SUBMITTED.value,
            JobStatus.RUNNING.value,
            JobStatus.CANCELLED.value,
            JobStatus.TIMED_OUT.value,
            JobStatus.FAILED.value,
        }
    ),
    JobStatus.SUBMITTED.value: frozenset(
        {
            JobStatus.RUNNING.value,
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.TIMED_OUT.value,
        }
    ),
    JobStatus.RUNNING.value: frozenset(
        {
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.TIMED_OUT.value,
        }
    ),
    JobStatus.SUCCEEDED.value: frozenset(),
    JobStatus.FAILED.value: frozenset(),
    JobStatus.CANCELLED.value: frozenset(),
    JobStatus.TIMED_OUT.value: frozenset(),
}


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def can_transition(current: str, target: str) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())


class IllegalTransition(RuntimeError):
    pass
