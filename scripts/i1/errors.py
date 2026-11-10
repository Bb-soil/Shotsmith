"""Small exception hierarchy used by the I1 console installer."""

from __future__ import annotations


class I1Error(Exception):
    """Base I1 failure; the message is safe to show to the user."""

    exit_code = 4


class UsageError(I1Error):
    exit_code = 2


class BlockedError(I1Error):
    """A pre-mutation compatibility/safety check refused the operation."""

    exit_code = 2


class HostTrustError(I1Error):
    exit_code = 3


class RemoteFailure(I1Error):
    exit_code = 4


class Interrupted(I1Error):
    exit_code = 5


class NotReady(I1Error):
    exit_code = 6


class StateError(I1Error):
    exit_code = 4
