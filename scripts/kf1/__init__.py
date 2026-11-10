"""KF1 — keyframe generation slice.

This package is intentionally separate from the legacy P1 live-candidate and
video publication paths.  It reuses bounded transport/readiness/lock primitives
from I1 and a small HTTP client for fake/real ComfyUI endpoints, but it never
submits a video job.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "kf1-2026-10-29"
