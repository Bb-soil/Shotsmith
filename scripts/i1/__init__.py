"""Provider-independent SSH installer for one pinned ComfyUI profile.

I1 is deliberately separate from the experimental P1 provider/session code.
It reuses bounded SSH transport, the verified downloader and the owned tunnel,
but never imports Clore/owner-risk policy through a side dependency.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "i1-2026-10-28"
