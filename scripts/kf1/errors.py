"""KF1 error types."""

from __future__ import annotations


class KF1Error(Exception):
    exit_code = 2

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class PlanError(KF1Error):
    pass


class NormalizationError(KF1Error):
    pass


class BindingError(KF1Error):
    pass


class StoreError(KF1Error):
    pass


class GenerationLimitError(KF1Error):
    pass


class ApprovalRequired(KF1Error):
    code = "WAITING_APPROVAL"


class V1MigrationRequired(KF1Error):
    code = "V1_MIGRATION_REQUIRED"


class AmbiguousDispatch(KF1Error):
    code = "AMBIGUOUS_DISPATCH"


class TerminalRemoteFailure(KF1Error):
    code = "REMOTE_TERMINAL_FAILURE"


class TransientRemoteError(KF1Error):
    code = "REMOTE_TRANSIENT"


class ReleaseDeferred(KF1Error):
    code = "DEFERRED_FOREIGN_WORK"


__all__ = [
    "AmbiguousDispatch",
    "ApprovalRequired",
    "BindingError",
    "GenerationLimitError",
    "KF1Error",
    "NormalizationError",
    "PlanError",
    "ReleaseDeferred",
    "StoreError",
    "TerminalRemoteFailure",
    "TransientRemoteError",
    "V1MigrationRequired",
]
