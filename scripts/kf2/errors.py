"""KF2 project-authoring error types."""

from __future__ import annotations


class KF2Error(Exception):
    exit_code = 2

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ProjectError(KF2Error):
    pass


class ProjectConflict(ProjectError):
    code = "PROJECT_REVISION_CONFLICT"


class ProjectExists(ProjectError):
    pass


class ProjectNotFound(ProjectError):
    pass


class ProjectTamper(ProjectError):
    pass


class ProjectIncomplete(ProjectError):
    code = "PROJECT_INCOMPLETE"


class ProjectBlocked(ProjectError):
    code = "PROJECT_BLOCKED"


class AssetError(ProjectError):
    pass


class TextImportError(ProjectError):
    pass


class CompileError(ProjectError):
    pass


__all__ = [
    "AssetError",
    "CompileError",
    "KF2Error",
    "ProjectBlocked",
    "ProjectConflict",
    "ProjectError",
    "ProjectExists",
    "ProjectIncomplete",
    "ProjectNotFound",
    "ProjectTamper",
    "TextImportError",
]
