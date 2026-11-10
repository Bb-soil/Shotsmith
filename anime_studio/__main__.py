"""CLI entry point: ``python -m anime_studio``.

Defaults are deliberately localhost-only. P0 has no authentication, so binding
to a non-loopback address requires an explicit --allow-remote flag.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from pathlib import Path

import uvicorn

from .api import create_app
from .demo import run_demo
from .errors import P1ConfigError, P1ManifestError, StudioError
from .p1_config import check_readiness
from .settings import Settings


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anime_studio")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the localhost HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--db", default="data/p0.sqlite3")
    serve.add_argument("--artifacts", default="data/artifacts")
    serve.add_argument("--poll-interval", type=float, default=0.2)
    serve.add_argument("--lease-seconds", type=float, default=5.0)
    serve.add_argument("--no-worker", action="store_true")
    serve.add_argument("--allow-remote", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.add_argument(
        "--adapter",
        choices=("mock", "comfy-http"),
        default="mock",
        help="MOCK is the default; comfy-http additionally requires explicit live opt-in",
    )
    serve.add_argument("--allow-live-transport", action="store_true")
    serve.add_argument("--comfy-config", default=None)

    demo = sub.add_parser("demo", help="run the offline MOCK demonstration")
    demo.add_argument("--evidence", default="docs/p0-evidence/demo-run.json")
    demo.add_argument("--data-dir", default="data/demo")

    check = sub.add_parser(
        "p1-check",
        help="offline P1 config/workflow/manifest readiness check; no network",
    )
    check.add_argument("--config", required=True)
    check.add_argument("--evidence", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "serve":
        if not _is_loopback(args.host) and not args.allow_remote:
            print(
                "Refusing to bind a non-loopback address without --allow-remote. "
                "P0 has no authentication.",
                file=sys.stderr,
            )
            return 2
        if args.adapter == "comfy-http":
            if not args.allow_live_transport:
                print(
                    "Refusing --adapter comfy-http without --allow-live-transport. "
                    "Default is MOCK and no network is allowed.",
                    file=sys.stderr,
                )
                return 2
            if not args.comfy_config:
                print(
                    "Refusing --adapter comfy-http without --comfy-config.",
                    file=sys.stderr,
                )
                return 2
            try:
                readiness = check_readiness(Path(args.comfy_config))
            except (P1ConfigError, P1ManifestError, StudioError) as exc:
                print(f"P1 readiness check failed: {exc}", file=sys.stderr)
                return 2
            if not readiness.live_ready:
                print(
                    "P1 readiness gate blocked live ComfyUI; live_ready=false: "
                    + "; ".join(readiness.issues),
                    file=sys.stderr,
                )
                return 2
        try:
            settings = Settings(
                db_path=Path(args.db),
                artifacts_dir=Path(args.artifacts),
                poll_interval=args.poll_interval,
                enable_worker=not args.no_worker,
                lease_seconds=args.lease_seconds,
                adapter_mode=args.adapter,
                allow_live_transport=args.allow_live_transport,
                comfy_config_path=Path(args.comfy_config) if args.comfy_config else None,
            )
        except ValueError as exc:
            print(f"Invalid settings: {exc}", file=sys.stderr)
            return 2
        try:
            app = create_app(settings)
            uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        except (P1ConfigError, P1ManifestError, StudioError) as exc:
            print(f"Startup refused: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "demo":
        return run_demo(
            evidence_path=Path(args.evidence),
            data_dir=Path(args.data_dir),
        )

    if args.command == "p1-check":
        try:
            readiness = check_readiness(Path(args.config))
        except (P1ConfigError, P1ManifestError, StudioError) as exc:
            print(f"P1 readiness check failed: {exc}", file=sys.stderr)
            return 2
        payload = readiness.to_dict()
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        print(text, end="")
        if args.evidence:
            evidence_path = Path(args.evidence)
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(text, encoding="utf-8")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
