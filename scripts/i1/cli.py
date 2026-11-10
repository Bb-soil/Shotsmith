"""Russian console entry point for the I1 SSH installer."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from . import ops
from .errors import BlockedError, I1Error, Interrupted, NotReady, UsageError
from .util import format_bytes, format_duration, redact

DEFAULT_STATE = str(ops.DEFAULT_STATE_DIR)
DEFAULT_PROFILE = str(ops.DEFAULT_PROFILE)


class ProgressPrinter:
    def __init__(self, sink=None) -> None:
        self._emit = sink or print
        self.last_at = 0.0
        self.last_bytes = 0
        self.last_progress_at = time.time()
        self.last_stage = None
        self.last_file = None
        self.last_heartbeat_print = 0.0
        self.last_connection = None

    def __call__(self, state: Mapping[str, Any]) -> None:
        now = time.time()
        if now - self.last_at < 0.7:
            return
        self.last_at = now
        observation = state.get("observation") or {}
        connection = observation.get("connection_state")
        worker_state = observation.get("worker_state")
        progress_fresh = bool(observation.get("progress_fresh"))
        rate_fresh = bool(observation.get("rate_fresh"))
        progress = state.get("progress") or {}
        stage = progress.get("stage") or state.get("stage") or state.get("phase")
        if connection not in (None, "CURRENT"):
            if self.last_connection != connection:
                as_of = (
                    observation.get("last_confirmed_at")
                    or observation.get("cached_as_of")
                    or "unknown"
                )
                self._emit(
                    f"[{stage}] CONNECTION_LOST; cached progress as of {as_of}; "
                    "worker UNKNOWN",
                    flush=True,
                )
                self.last_connection = connection
                self.last_progress_at = now
                self.last_bytes = 0
            return
        if self.last_connection not in (None, "CURRENT"):
            # Reconnected: the first fresh sample must not derive current speed
            # from bytes that were cached during the outage.
            self.last_bytes = int(progress.get("bytes_done") or 0)
            self.last_progress_at = now
        self.last_connection = connection
        if connection == "CURRENT" and not progress_fresh:
            if now - self.last_heartbeat_print > 10:
                self._emit(
                    f"[{stage}] connection CURRENT; worker liveness UNKNOWN; "
                    "cached progress only",
                    flush=True,
                )
                self.last_heartbeat_print = now
            return
        if worker_state == "LIVE" and observation.get("heartbeat_fresh") is False:
            if now - self.last_heartbeat_print > 10:
                self._emit(
                    f"[{stage}] process confirmed live; heartbeat STALE; "
                    "responsiveness unknown",
                    flush=True,
                )
                self.last_heartbeat_print = now
            return
        current_file = progress.get("current_file")
        files_done = int(progress.get("files_done") or 0)
        files_total = int(progress.get("files_total") or 1)
        bytes_done = int(progress.get("bytes_done") or 0)
        bytes_total = int(progress.get("bytes_total") or 0)
        percent = (100.0 * bytes_done / bytes_total) if bytes_total else 0.0
        delta = max(0, bytes_done - self.last_bytes)
        elapsed = max(0.001, now - self.last_progress_at)
        client_speed = delta / elapsed if delta else 0.0
        client_eta = (bytes_total - bytes_done) / client_speed if client_speed > 0 and bytes_total else None
        worker_speed = progress.get("speed_bytes_per_sec")
        worker_eta = progress.get("eta_seconds")
        if not rate_fresh:
            worker_speed = None
            worker_eta = None
            client_speed = 0.0
            client_eta = None
        speed = (
            float(worker_speed)
            if isinstance(worker_speed, (int, float))
            and not isinstance(worker_speed, bool)
            and math.isfinite(float(worker_speed))
            else client_speed
        )
        if (
            isinstance(worker_eta, (int, float))
            and not isinstance(worker_eta, bool)
            and math.isfinite(float(worker_eta))
            and worker_eta >= 0
        ):
            eta = float(worker_eta)
        else:
            eta = client_eta
        eta_known = progress.get("eta_known")
        if eta_known is False or eta is None or speed <= 0:
            eta_text = "неизвестно"
        else:
            eta_text = format_duration(eta)
        heartbeat = progress.get("heartbeat_at")
        hb = ""
        if isinstance(heartbeat, (int, float)):
            age = max(0.0, now - float(heartbeat))
            hb = f" heartbeat {age:.0f}s"
        if stage != self.last_stage or current_file != self.last_file or delta > 0 or bytes_done == 0:
            file_text = f" {current_file}" if current_file else ""
            rate_text = (
                f"{format_bytes(int(speed))}/s ETA {eta_text}"
                if rate_fresh
                else "rate unknown (ETA unknown)"
            )
            self._emit(
                f"[{stage}] файл {files_done}/{files_total}{file_text} "
                f"{percent:5.1f}% {format_bytes(bytes_done)}/{format_bytes(bytes_total)} "
                f"{rate_text}{hb}",
                flush=True,
            )
            self.last_progress_at = now
            self.last_bytes = bytes_done
            self.last_stage = stage
            self.last_file = current_file
        elif now - self.last_progress_at > 10 and now - self.last_heartbeat_print > 10:
            self._emit(
                f"[{stage}] process confirmed live; no new bytes; rate "
                f"{'current' if rate_fresh else 'unknown'}{hb}",
                flush=True,
            )
            self.last_heartbeat_print = now


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", default=DEFAULT_STATE)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--name", default=None)
    parser.add_argument("--json", action="store_true")


def _add_connect(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--host", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--identity", default=None)
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--accept-fingerprint", default=None)
    parser.add_argument("--replace", action="store_true")


def _add_simple(parser: argparse.ArgumentParser) -> None:
    _common(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avs.ps1 worker",
        description="I1: provider-independent SSH установщик ComfyUI 14B. "
        "Все команды работают флагами без диалога; --json не печатает лишнего.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    connect_p = sub.add_parser("connect", help="подключить существующий GPU-сервер")
    _add_connect(connect_p)
    inspect_p = sub.add_parser("inspect", help="только чтение: диагностика сервера")
    _add_simple(inspect_p)
    install_p = sub.add_parser("install", help="план или установка после --execute")
    _add_simple(install_p)
    install_p.add_argument("--execute", action="store_true")
    install_p.add_argument("--yes", action="store_true")
    status_p = sub.add_parser("status", help="локальный и фактический статус worker")
    _add_simple(status_p)
    status_p.add_argument("--offline", action="store_true")
    resume_p = sub.add_parser("resume", help="продолжить прерванную установку")
    _add_simple(resume_p)
    resume_p.add_argument(
        "--new-attempt",
        action="store_true",
        help="явно начать новый download attempt только после dead/stale старого",
    )
    open_p = sub.add_parser("open", help="открыть localhost URL через owned tunnel")
    _add_simple(open_p)
    open_p.add_argument("--browser", action="store_true")
    disconnect_p = sub.add_parser("disconnect", help="закрыть только owned tunnel")
    _add_simple(disconnect_p)
    return parser


def _interactive_connect(args: argparse.Namespace) -> None:
    if args.json or not sys.stdin.isatty():
        return
    if not args.host:
        args.host = input("Host: ").strip()
    if not args.user:
        args.user = input("User [root]: ").strip() or "root"
    if args.port is None:
        raw = input("Port [22]: ").strip() or "22"
        try:
            args.port = int(raw)
        except ValueError as exc:
            raise UsageError("port должен быть числом") from exc
    if not args.workspace:
        args.workspace = input("Workspace [/workspace]: ").strip() or "/workspace"
    if not args.identity and not args.agent:
        raw = input("Путь к ключу (пусто = ssh-agent): ").strip()
        if raw:
            args.identity = raw
        else:
            args.agent = True


def _require_connection_args(args: argparse.Namespace) -> None:
    if not args.host or not args.user or args.port is None or not args.workspace:
        raise UsageError("connect требует --host --user --port --workspace или интерактивный TTY")


def _ctx(args: argparse.Namespace, *, verify_key: bool = True) -> ops.Context:
    return ops.load_context(
        name=args.name,
        state_dir=Path(args.state_dir),
        profile_path=Path(args.profile),
        verify_key=verify_key,
    )


def _human_inspection(payload: Mapping[str, Any]) -> None:
    inspection = payload.get("inspection", payload)
    facts = inspection.get("facts", {})
    print(f"Профиль: {payload.get('profile_id', '?')}")
    print(f"OS/arch: {facts.get('os')} / {facts.get('arch')}")
    print(f"Workspace: {facts.get('workspace')}")
    print(f"GPU: {facts.get('gpu')}")
    print(f"Torch: {facts.get('torch')}")
    print(f"Disk free: {format_bytes(int(facts.get('disk_free_bytes') or 0))}")
    print(f"ComfyUI: {facts.get('comfy_commit') or 'not installed'}")
    for item in inspection.get("warnings", []):
        print(f"[!] {item.get('message')}")
    for item in inspection.get("blockers", []):
        print(f"[BLOCK] {item.get('message')}")
    print("Совместимость: " + ("OK" if inspection.get("compatible") else "BLOCKED"))


def _human_plan(plan: Mapping[str, Any]) -> None:
    print(f"Профиль: {plan.get('profile_id')}  hash {plan.get('profile_sha256')}")
    print(f"ComfyUI commit: {plan.get('comfyui_commit')}")
    print("Модели:")
    for item in plan.get("models", []):
        print(f"  - {item.get('target')}  {format_bytes(int(item.get('size_bytes') or 0))}")
    print(f"Итого моделей: {format_bytes(int(plan.get('model_bytes') or 0))}")
    print(f"Требуется диска: {format_bytes(int(plan.get('disk_required_bytes') or 0))}")
    print(f"Свободно сейчас: {format_bytes(int(plan.get('disk_free_bytes') or 0))}")
    print(f"Workflow API: {plan.get('workflow_api', {}).get('path')}")
    print(f"UI: {plan.get('workflow_ui', {}).get('status')} (graph_rendered=false)")
    print(f"[!] {plan.get('billing_warning')}")
    for item in plan.get("warnings", []):
        print(f"[!] {item}")
    for item in plan.get("blockers", []):
        print(f"[BLOCK] {item}")


def _human_status(payload: Mapping[str, Any]) -> None:
    print(f"Фаза: {payload.get('phase') or payload.get('status')}  stage: {payload.get('stage')}")
    observation = payload.get("observation") or {}
    connection = payload.get("effective_connection_state") or observation.get(
        "connection_state"
    )
    worker_state = payload.get("worker_state") or observation.get("worker_state")
    responsiveness = payload.get("worker_responsiveness") or observation.get(
        "worker_responsiveness"
    )
    print(
        f"Соединение: {connection or 'UNKNOWN'}  "
        f"Worker: {worker_state or payload.get('worker_status') or 'UNKNOWN'}  "
        f"Responsiveness: {responsiveness or 'UNKNOWN'}"
    )
    last_confirmed = payload.get("last_confirmed_at") or observation.get(
        "last_confirmed_at"
    )
    if last_confirmed:
        print(f"Последнее подтверждение: {last_confirmed}")
    bytes_observed_at = payload.get("bytes_observed_at") or observation.get(
        "bytes_observed_at"
    )
    if bytes_observed_at:
        print(f"Байты наблюдены: {bytes_observed_at}")
    diagnostic = payload.get("diagnostic") or observation.get("diagnostic")
    if diagnostic:
        if isinstance(diagnostic, Mapping):
            print(f"[!] {diagnostic.get('code')}: {diagnostic.get('message')}")
        else:
            print(f"[!] {diagnostic}")
    print(f"Workspace: {payload.get('workspace')}")
    if payload.get("progress_fresh"):
        progress = payload.get("progress") or {}
        if progress:
            print(
                f"Прогресс: {progress.get('files_done', 0)}/{progress.get('files_total', 4)} "
                f"{format_bytes(int(progress.get('bytes_done') or 0))}/"
                f"{format_bytes(int(progress.get('bytes_total') or 0))}"
            )
            if progress.get("stage"):
                print(f"Этап: {progress.get('stage')}")
            speed = progress.get("speed_bytes_per_sec")
            eta = progress.get("eta_seconds")
            if payload.get("rate_fresh") and isinstance(speed, (int, float)) and not isinstance(speed, bool):
                if progress.get("eta_known") is True and isinstance(eta, (int, float)):
                    print(
                        f"Скорость: {format_bytes(int(speed))}/s, "
                        f"ETA {format_duration(float(eta))}"
                    )
                else:
                    print(f"Скорость: {format_bytes(int(speed))}/s, ETA неизвестно")
            else:
                print("Скорость: неизвестно (rate не подтверждён текущим окном)")
    else:
        cached = payload.get("cached_progress") or payload.get("local", {}).get(
            "progress"
        ) or {}
        if cached:
            print(
                f"Cached progress as of {payload.get('cached_as_of') or 'unknown'}: "
                f"{cached.get('files_done', 0)}/{cached.get('files_total', 4)} "
                f"{format_bytes(int(cached.get('bytes_done') or 0))}/"
                f"{format_bytes(int(cached.get('bytes_total') or 0))}"
            )
        print("[!] Speed/ETA are not current; worker liveness is unknown")
    readiness = payload.get("readiness") or {}
    if readiness:
        print(f"Readiness: {readiness.get('status')}")
        if payload.get("readiness_current") is False:
            print("[!] Readiness не подтверждён текущей online-проверкой")
        for warning in readiness.get("warnings", []):
            print(f"[!] {warning}")
    if payload.get("worker_error"):
        print(f"[worker] {payload['worker_error']}")


def _dispatch(args: argparse.Namespace, *, progress_sink=None) -> tuple[int, Any]:
    if args.command == "connect":
        _interactive_connect(args)
        _require_connection_args(args)
        name = args.name
        if not name:
            name = args.host.replace(".", "-")
        payload = ops.connect(
            args=ops.ConnectArgs(
                name=name,
                host=args.host,
                user=args.user,
                port=args.port,
                workspace=args.workspace,
                identity=args.identity,
                agent=args.agent,
                accept_fingerprint=args.accept_fingerprint,
                replace=args.replace,
                json=args.json,
            ),
            state_dir=Path(args.state_dir),
            profile_path=Path(args.profile),
        )
        return 0, payload
    if args.command == "inspect":
        ctx = _ctx(args)
        payload = ops.inspect(ctx)
        return 0, payload
    if args.command == "install":
        ctx = _ctx(args)
        planned = ops.plan_install(ctx)
        inspection = planned["inspection"]
        plan = planned["plan"]
        if not args.json:
            _human_inspection({"inspection": inspection, "profile_id": ctx.profile.profile_id})
            _human_plan(plan)
        if inspection.get("blockers"):
            if args.json:
                return 2, {"status": "BLOCKED", **planned}
            return 2, planned
        if not args.execute:
            return 0, {"status": "PLAN_ONLY", **planned}
        if not args.yes and sys.stdin.isatty() and not args.json:
            answer = input("Выполнить установку? [y/N] ").strip().lower()
            if answer not in {"y", "yes", "д", "да"}:
                return 2, {"status": "CANCELLED", **planned}
        printer = (ProgressPrinter() if progress_sink is None else ProgressPrinter(progress_sink)) if not args.json else None
        result = ops.execute_install(ctx, progress=printer)
        report = result.get("readiness", {})
        if args.json:
            return (0 if report.get("ready") else 6), result
        if report:
            from .readiness import human_summary

            print(human_summary(report))
        if report.get("ready"):
            return 0, result
        return 6, result
    if args.command == "status":
        ctx = _ctx(args, verify_key=False)
        payload = ops.status(ctx, offline=args.offline)
        return 0, payload
    if args.command == "resume":
        ctx = _ctx(args)
        printer = (ProgressPrinter() if progress_sink is None else ProgressPrinter(progress_sink)) if not args.json else None
        result = ops.resume(
            ctx, progress=printer, new_attempt=getattr(args, "new_attempt", False)
        )
        report = result.get("readiness", {})
        if args.json:
            return (0 if report.get("ready") else 6), result
        if report:
            from .readiness import human_summary

            print(human_summary(report))
        return (0 if report.get("ready") else 6), result
    if args.command == "open":
        ctx = _ctx(args)
        payload = ops.open_comfy(ctx, browser=args.browser)
        if not args.json:
            print(f"ComfyUI: {payload['url']}")
            print(f"UI: {payload['ui_status']} (graph_rendered=false)")
            print(f"Точный API JSON: {payload.get('local_api_file')}")
            if payload.get("local_ui_file"):
                print(f"UI-файл: {payload['local_ui_file']}")
            print(f"[!] {payload['billing_warning']}")
        return 0, payload
    if args.command == "disconnect":
        ctx = _ctx(args, verify_key=False)
        payload = ops.disconnect(ctx)
        if not args.json:
            print(f"Статус: {payload['status']}")
            print(f"[!] {payload['billing_warning']}")
        return 0 if payload["status"] == "DISCONNECTED" else 4, payload
    raise UsageError("unknown command")


def main(argv: list[str] | None = None, *, progress_sink=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        code, payload = _dispatch(args) if progress_sink is None else _dispatch(args, progress_sink=progress_sink)
        if getattr(args, "json", False) and payload is not None:
            _print_json(payload)
        elif args.command == "connect" and not args.json and payload is not None:
            print(f"Подключено: {payload['name']} {payload['user']}@{payload['host']}:{payload['port']}")
            print(f"Fingerprint: {payload['fingerprint']}")
            print(f"Workspace: {payload['workspace']}")
        elif args.command == "status" and not args.json and payload is not None:
            _human_status(payload)
        elif args.command == "inspect" and payload is not None:
            _human_inspection(payload)
        return code
    except KeyboardInterrupt:
        print("Прервано. Используйте resume; remote worker может продолжать.", file=sys.stderr)
        return 5
    except I1Error as exc:
        message = redact(str(exc), limit=1200)
        if getattr(args, "json", False):
            _print_json(
                {
                    "status": "ERROR",
                    "error": type(exc).__name__,
                    "message": message,
                    "exit_code": exc.exit_code,
                }
            )
        else:
            print(f"Ошибка: {message}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        message = redact(str(exc), limit=1200)
        if getattr(args, "json", False):
            _print_json(
                {"status": "ERROR", "error": type(exc).__name__, "message": message, "exit_code": 4}
            )
        else:
            print(f"Непредвиденная ошибка: {message}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
