"""Render-only Clore startup/autoSSH readiness plan.

No subprocess, SSH, Docker pull, environment read, or provider call is made.
The public Clore contract documents ``autossh_entrypoint`` and
``/root/onstart.sh`` but not the exact entrypoint override, UID/HOME/PATH,
Python/torch layout, signal behavior, or image-digest handling. Those unknowns
therefore keep readiness false; this module only prints manual checks and
placeholders.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.p1.clore_bridge import CloreConfigError

IMAGE_CANDIDATE = "runpod/pytorch:1.4.1-cu1281-torch291-ubuntu2404"
EXPECTED_IMAGE_DIGEST = (
    "sha256:8ab44dd05aad6ecc18e77814a20f7fb0ee706c4696663cf1bf912b0f5b457e9f"
)
COMFYUI_REMOTE_PORT = 8188
LOCAL_TUNNEL_PORT = 8189
SSH_PORT = 22


@dataclass(frozen=True)
class StartupEvidence:
    """Observed facts required before startup can be called ready."""

    base_entrypoint_inspected: bool = False
    actual_image_digest: str | None = None
    uid_home_python_torch_verified: bool = False
    onstart_script_verified: bool = False
    ssh_public_key_only: bool = False
    only_tcp22: bool = False
    comfyui_loopback: bool = False
    no_public_jupyter: bool = False


@dataclass(frozen=True)
class StartupReadiness:
    live_ready: bool
    blockers: tuple[str, ...]
    plan: str


def assess_startup(evidence: StartupEvidence | None = None) -> StartupReadiness:
    evidence = evidence or StartupEvidence()
    blockers: list[str] = []
    if not evidence.base_entrypoint_inspected:
        blockers.append(
            "base entrypoint vs autoSSH override sequence is not inspected"
        )
    if evidence.actual_image_digest is None:
        blockers.append("actual image digest is unknown")
    elif evidence.actual_image_digest != EXPECTED_IMAGE_DIGEST:
        blockers.append("actual image digest does not match the pinned candidate")
    if not evidence.uid_home_python_torch_verified:
        blockers.append(
            "UID/HOME/PATH/python3/torch layout is not verified on the actual image"
        )
    if not evidence.onstart_script_verified:
        blockers.append("/root/onstart.sh contract and lifetime are not verified")
    if not evidence.ssh_public_key_only:
        blockers.append("public-key-only SSH is not verified")
    if not evidence.only_tcp22:
        blockers.append("only TCP 22 is not verified; public HTTP ports may exist")
    if not evidence.comfyui_loopback:
        blockers.append("ComfyUI loopback 127.0.0.1:8188 is not verified")
    if not evidence.no_public_jupyter:
        blockers.append("public Jupyter/8188/8888 exposure is not ruled out")
    return StartupReadiness(
        live_ready=not blockers,
        blockers=tuple(blockers),
        plan=render_startup_plan(),
    )


def render_startup_plan() -> str:
    """Return a review-only plan; it is never executed by this module."""

    return "\n".join(
        (
            "# Clore startup/autoSSH review plan only. Do not execute from the",
            "# offline adaptation task. No image pull, SSH, secret read or provider call.",
            f"# Candidate image (tag): {IMAGE_CANDIDATE}",
            f"# Expected digest (manifest pin): {EXPECTED_IMAGE_DIGEST}",
            "# Digest support on Clore is NOT documented; do not silently retag.",
            "# Pin the manifest DRAFT only after actual runtime/object_info evidence.",
            "",
            "# Manual inspection before any dependency/model download:",
            f"#   ports: only {SSH_PORT}/tcp; no public 8188/8888/Jupyter/VS Code",
            "#   command: id; printf '%s\\n' \"$HOME\"; command -v python3",
            "#   command: python3 -c 'import torch; print(torch.__version__)'",
            "#   inspect /root/onstart.sh and the effective base entrypoint",
            "#   verify actual image digest equals the pinned candidate",
            "",
            "# SSH is public-key-only; no password and no private key value in argv/logs.",
            f"# Tunnel example: ssh -N -o ExitOnForwardFailure=yes \\",
            f"#   -L 127.0.0.1:{LOCAL_TUNNEL_PORT}:127.0.0.1:{COMFYUI_REMOTE_PORT} \\",
            "#   -p <SSH_PORT> -i <LOCAL_PRIVATE_KEY_PATH> root@<ORDER_HOST>",
            f"# ComfyUI must bind 127.0.0.1:{COMFYUI_REMOTE_PORT}, never 0.0.0.0.",
            "# All setup/download/render/export time counts from provider ct;",
            "# hard end is ct+60m and planned cancel is at or before ct+55m.",
            "",
            "# Unknowns kept explicit: autoSSH entrypoint override, /root/onstart.sh",
            "# lifecycle/signals, UID/HOME/PATH, Python/torch, image digest support.",
            "# Startup readiness is false until those are observed in a separately",
            "# approved live session.",
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/p1/clore_startup.py",
        description="Print the offline Clore startup review plan; no execution.",
    )
    parser.add_argument(
        "--json", action="store_true", help="print structured readiness JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        readiness = assess_startup()
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "CLORE_STARTUP_PLAN",
                        "live_ready": readiness.live_ready,
                        "blockers": list(readiness.blockers),
                        "plan": readiness.plan,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
        else:
            print(readiness.plan)
            if readiness.blockers:
                print("\n# Startup blockers:")
                for blocker in readiness.blockers:
                    print(f"# - {blocker}")
        return 0
    except CloreConfigError as exc:
        print(f"startup plan refused: {exc}")
        return 2


__all__ = [
    "COMFYUI_REMOTE_PORT",
    "EXPECTED_IMAGE_DIGEST",
    "IMAGE_CANDIDATE",
    "LOCAL_TUNNEL_PORT",
    "SSH_PORT",
    "StartupEvidence",
    "StartupReadiness",
    "assess_startup",
    "render_startup_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
