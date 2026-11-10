"""Staged as ``.avs-i1/comfy-entry/main.py``; attests birth then runs real ComfyUI."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    import i1_child_entry  # staged next to this shim

    entry = Path(__file__).resolve().parent
    workspace = entry.parent.parent
    manifest = i1_child_entry.write_birth(
        kind="comfy", manifest_path=workspace / "comfy-launch.json"
    )
    comfy_dir = Path(str(manifest.get("comfy_dir", ""))).resolve()
    actual = comfy_dir / "main.py"
    if not actual.is_file() or actual.is_symlink():
        print("ComfyUI main.py is missing", file=sys.stderr)
        return 2
    # Python executes this shim from .avs-i1/comfy-entry, so sys.path[0] would
    # otherwise point at the shim directory.  Replace it with the verified real
    # checkout directory (not just append) so sibling modules/packages resolve
    # exactly as if Python had executed the real main.py directly.  Keep only
    # stdlib/site-packages and other trusted paths; never rely on PYTHONPATH.
    shim_dir = str(entry)
    trusted = [
        item
        for item in sys.path
        if item and item not in (shim_dir, str(comfy_dir), os.getcwd())
    ]
    sys.path[:] = [str(comfy_dir)] + trusted
    sys.argv = [str(actual), *sys.argv[1:]]
    runpy.run_path(str(actual), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
