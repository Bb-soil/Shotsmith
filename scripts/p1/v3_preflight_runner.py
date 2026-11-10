"""Staged preflight subset runner for the owned P1 v3 pipeline.

Offline prerequisites run after downloads but before ComfyUI starts.  The full
API preflight (system stats, object_info and validation) runs only after the
owned ComfyUI process and tunnel are ready.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import preflight  # noqa: E402

OFFLINE = {
    "nvidia-smi",
    "ffmpeg-version",
    "ffprobe-version",
    "pip-check",
    "pip-freeze",
    "core-imports",
}
API = {"system-stats", "object-info", "object-info-validation"}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in {"offline", "api"}:
        print("usage: v3_preflight_runner.py offline|api <workspace>", file=sys.stderr)
        return 2
    mode, workspace = args
    os.chdir(workspace)
    checks = preflight.build_preflight_checks(workspace=workspace)
    selected = OFFLINE if mode == "offline" else API
    try:
        results = preflight.run_preflight_checks(
            tuple(check for check in checks if check.name in selected)
        )
    except (preflight.PreflightError, preflight.PreflightExecutionError, OSError) as exc:
        print(f"preflight-{mode} failed: {exc}", file=sys.stderr)
        return 2
    for result in results:
        print(f"{result.check.name}: exit {result.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
