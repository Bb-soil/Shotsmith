"""Domain errors mapped to explicit HTTP responses by the API layer."""

from __future__ import annotations


class StudioError(Exception):
    code = "STUDIO_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class JobNotFound(StudioError):
    code = "JOB_NOT_FOUND"


class IdempotencyConflict(StudioError):
    code = "IDEMPOTENCY_CONFLICT"

    def __init__(self, existing_job_id: str) -> None:
        super().__init__(
            "Idempotency-Key already belongs to a different canonical payload.",
            code=self.code,
        )
        self.existing_job_id = existing_job_id


class JobAlreadyTerminal(StudioError):
    code = "JOB_ALREADY_TERMINAL"

    def __init__(self, status: str) -> None:
        super().__init__(f"Job is already terminal: {status}", code=self.code)
        self.status = status


class ResultNotReady(StudioError):
    code = "RESULT_NOT_READY"

    def __init__(self, status: str) -> None:
        super().__init__(
            f"Job result is unavailable while status is {status}.", code=self.code
        )
        self.status = status


class AdapterError(StudioError):
    code = "ADAPTER_ERROR"


class RemoteValidationRejected(AdapterError):
    """The remote rejected the prompt before accepting it.

    This is a confirmed negative result: no remote prompt_id was created and a
    retry would only fail again until the workflow/input is fixed.
    """

    code = "REMOTE_VALIDATION_REJECTED"


class RemoteTransientError(AdapterError):
    """A network failure that happened before the request could be accepted.

    Callers must not treat this as a confirmed remote absence when there is any
    doubt. P1 never retries automatically; the job records the classification
    and a human/Codex decides whether a new idempotency key may be used.
    """

    code = "REMOTE_TRANSIENT_ERROR"


class RemoteDeadlineExpired(AdapterError):
    """The persisted job deadline expired before the remote prompt was sent.

    This is a *confirmed not sent* result: no ``/prompt`` request was made, so
    it must become a local timeout rather than an ambiguous dispatch or a
    generic remote failure. It intentionally does not inherit from
    ``RemoteTransientError`` or ``RemoteAmbiguousSubmit``.
    """

    code = "DEADLINE_EXPIRED_BEFORE_SEND"


class RemoteAmbiguousSubmit(AdapterError):
    """The request may or may not have reached/accepted by ComfyUI."""

    code = "AMBIGUOUS_DISPATCH"


class RemoteAuthError(AdapterError):
    code = "REMOTE_AUTH_ERROR"


class RemoteProtocolError(AdapterError):
    """Malformed/unsupported remote response, as opposed to a network blip."""

    code = "REMOTE_PROTOCOL_ERROR"


class RemotePromptLost(AdapterError):
    code = "REMOTE_PROMPT_LOST"


class RemoteCancelUnsupported(AdapterError):
    """Cancellation cannot be confirmed safely for this worker topology."""

    code = "CANCEL_REQUIRES_EXCLUSIVE_WORKER"


class WorkflowBindingError(StudioError):
    code = "WORKFLOW_BINDING_INVALID"


class P1ConfigError(StudioError):
    code = "P1_CONFIG_INVALID"


class P1ManifestError(StudioError):
    code = "P1_MANIFEST_INVALID"


class ExecutionBindingMismatch(StudioError):
    """The database and current adapter execution identity disagree."""

    code = "EXECUTION_BINDING_MISMATCH"


class ArtifactError(StudioError):
    code = "ARTIFACT_ERROR"


class InstanceAlreadyRunning(StudioError):
    code = "INSTANCE_ALREADY_RUNNING"
