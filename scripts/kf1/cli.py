"""Russian CLI for the KF1 keyframe slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from .approval import approve_frame
from .comfy import ComfyClient
from .connection import open_kf1_connection
from .errors import KF1Error, V1MigrationRequired
from .gallery import export_storyboard
from .runtime_gate import validate_runtime_object_info
from .plan import (
    EXAMPLE_PLAN,
    PLAN_SCHEMA,
    interactive_plan,
    load_example,
    parse_text_plan,
    plan_to_json,
    preview_lines,
    resolve_plan,
)
from .release import release_plan
from .runner import Runner
from .store import PlanStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "data" / "kf1"
DEFAULT_PLANS = ROOT / "data" / "kf1" / "plans"
DEFAULT_PROFILE = ROOT / "config" / "i1" / "keyframes-qwen-edit-2511.profile.json"


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _write_plan(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise KF1Error(f"План уже существует: {path}; используйте другое имя")
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_or_create_plan(args: argparse.Namespace) -> Path:
    if args.from_json:
        return Path(args.from_json)
    if args.from_text:
        text = Path(args.from_text).read_text(encoding="utf-8")
        data = parse_text_plan(text)
    elif args.interactive:
        data = interactive_plan()
    elif args.example:
        if not args.save:
            return EXAMPLE_PLAN
        data = load_example()
    else:
        raise KF1Error("Укажите --from-json, --from-text, --interactive или --example")
    data.setdefault("schema", PLAN_SCHEMA)
    target = Path(args.save) if args.save else DEFAULT_PLANS / f"plan-{uuid.uuid4().hex}.json"
    _write_plan(target, data)
    return target


def _human_progress(line: str) -> None:
    print(line, flush=True)


def _client(args: argparse.Namespace) -> tuple[ComfyClient, dict[str, Any]]:
    opened = open_kf1_connection(
        base_url=args.base_url,
        connection=args.connection,
        state_dir=Path(args.i1_state_dir),
        profile_path=Path(args.profile),
    )
    token = None
    if getattr(args, "token_env", None):
        token = os.environ.get(args.token_env)
        if not token:
            raise KF1Error(f"Переменная окружения {args.token_env} не задана")
    return ComfyClient(opened.base_url, token=token), opened.attestation


def _runner(args: argparse.Namespace, plan: Any) -> Runner:
    store = PlanStore(args.state_dir, plan.plan_sha256, plan.path)
    return Runner(plan, store, None)


def _cmd_plan(args: argparse.Namespace) -> tuple[int, Any]:
    path = _load_or_create_plan(args)
    plan = resolve_plan(path)
    lines = preview_lines(plan)
    if args.json:
        return 0, {"status": "PLAN_READY", "plan": plan_to_json(plan), "preview": lines}
    for line in lines:
        print(line)
    return 0, None


def _validate_stop_after(plan: Any, stop_after: str | None) -> None:
    if stop_after is None:
        return
    if stop_after not in {frame.frame_id for frame in plan.frames}:
        raise KF1Error(f"unknown --until-frame: {stop_after}")


def _cmd_generate(args: argparse.Namespace) -> tuple[int, Any]:
    plan = resolve_plan(args.plan)
    _validate_stop_after(plan, args.until_frame)
    store = PlanStore(args.state_dir, plan.plan_sha256, plan.path)
    if not args.execute:
        state = store.load_any(plan=plan)
        runner = Runner(plan, store, None)
        payload = {
            "status": "PLAN_ONLY",
            "plan": plan_to_json(plan),
            "preview": preview_lines(plan),
            "state": runner.status(state),
            "post_plan": [frame.frame_id for frame in plan.generated_frames],
            "until_frame": args.until_frame,
        }
        if args.json:
            return 0, payload
        for line in payload["preview"]:
            print(line)
        print("Офлайн: upload/POST не выполнялись. Для запуска нужен явный --execute.")
        return 0, None
    client, attestation = _client(args)
    try:
        from scripts.i1.profile import load_profile
        validate_runtime_object_info(load_profile(args.profile), client.object_info())
        runner = Runner(plan, store, client, attestation=attestation)
        result = runner.run(
            execute=True,
            deadline_seconds=float(args.deadline),
            progress=None if args.json else _human_progress,
            raise_limit=bool(args.raise_limit),
            stop_after=args.until_frame,
        )
        return 0, result
    finally:
        client.close()


def _cmd_status(args: argparse.Namespace) -> tuple[int, Any]:
    plan = resolve_plan(args.plan)
    store = PlanStore(args.state_dir, plan.plan_sha256, plan.path)
    state = store.load_any(plan=plan)
    runner = Runner(plan, store, None)
    payload = runner.status(state)
    payload["offline"] = True
    payload["note"] = (
        "Кэшированные durable facts; после потери SSH это единственный "
        "достоверный локальный статус."
    )
    if args.json:
        return 0, payload
    print(f"План: {plan.title}")
    print(f"Plan SHA256: {plan.plan_sha256}")
    print(f"Использовано попыток: {payload['attempts_used']} / "
          f"{payload['generation_limit'] + payload['extra_limit']}")
    for frame in payload["frames"]:
        output = frame.get("output") or {}
        approval = frame.get("approval_status") or "NOT_APPLICABLE"
        print(
            f"  {frame['frame_id']}: {frame['status']} v{frame.get('current_version')} "
            f"approval={approval} sha256={frame.get('content_sha256') or '-'} "
            f"binding={frame.get('binding_sha256') or '-'} "
            f"{output.get('path') or ''}"
        )
        if frame.get("approval_command"):
            print(f"    approve: {frame['approval_command']}")
    return 0, None


def _cmd_retry(args: argparse.Namespace) -> tuple[int, Any]:
    plan = resolve_plan(args.plan)
    store = PlanStore(args.state_dir, plan.plan_sha256, plan.path)
    if not args.owner_ack:
        raise KF1Error("retry-frame требует explicit --owner-ack")
    if plan.schema_version == 1:
        raise V1MigrationRequired(
            "v1 retry-frame запрещён: создайте v2-копию плана; "
            "старый план/state не перезаписываются"
        )
    if not args.execute:
        state = store.load_any(plan=plan)
        runner = Runner(plan, store, None)
        payload = {
            "status": "PLAN_ONLY",
            "frame": args.frame,
            "owner_ack": True,
            "state": runner.status(state),
            "note": "Новая версия не создана; POST не выполнялся.",
        }
        if args.json:
            return 0, payload
        print(f"Retry plan: {args.frame}; POST не выполнялся.")
        return 0, None
    client, attestation = _client(args)
    try:
        from scripts.i1.profile import load_profile
        validate_runtime_object_info(load_profile(args.profile), client.object_info())
        runner = Runner(plan, store, client, attestation=attestation)
        result = runner.retry_frame(
            args.frame,
            owner_ack=True,
            raise_limit=bool(args.raise_limit),
            deadline_seconds=float(args.deadline),
            progress=None if args.json else _human_progress,
        )
        return 0, result
    finally:
        client.close()


def _cmd_approve(args: argparse.Namespace) -> tuple[int, Any]:
    plan = resolve_plan(args.plan)
    store = PlanStore(args.state_dir, plan.plan_sha256, plan.path)
    result = approve_frame(
        plan,
        store,
        frame_id=args.frame,
        version=int(args.version),
        sha256=args.sha256,
        owner_ack=bool(args.owner_ack),
        note=args.note,
    )
    if args.json:
        return 0, result
    print(f"Approval: {result['status']} {args.frame} v{int(args.version)}")
    print(f"SHA256: {args.sha256}")
    return 0, result


def _cmd_export(args: argparse.Namespace) -> tuple[int, Any]:
    plan = resolve_plan(args.plan)
    store = PlanStore(args.state_dir, plan.plan_sha256, plan.path)
    # Read-only historical load: export must never create state or move the
    # current pointer while another revision is unresolved.
    state = store.load_revision_readonly(plan=plan)
    output = Path(args.output) if args.output else store.root / "storyboard"
    manifest = export_storyboard(plan, state, output, store=store)
    payload = {"status": "EXPORTED", "output": str(output), "manifest": manifest}
    if args.json:
        return 0, payload
    print(f"Storyboard: {output / 'index.html'}")
    print("visual_status=NOT_REVIEWED; quality=NOT_RUN; video не отправлялось.")
    return 0, payload


def _cmd_release(args: argparse.Namespace) -> tuple[int, Any]:
    plan = resolve_plan(args.plan)
    store = PlanStore(args.state_dir, plan.plan_sha256, plan.path)
    states = store.iter_states(plan=plan)
    combined_state = {
        "attempts": [
            attempt
            for revision in states
            for attempt in revision.get("attempts", [])
            if isinstance(attempt, Mapping)
        ]
    }
    client = None
    try:
        if args.execute:
            client, _attestation = _client(args)
        payload = release_plan(plan, combined_state, client, execute=bool(args.execute))
        if args.json:
            return 0, payload
        print(f"Статус: {payload['status']}")
        if payload.get("request_accepted"):
            print("POST /free принят как запрос; VRAM free не подтверждён.")
        print(payload.get("limitation") or payload.get("policy") or "")
        return 0, None
    finally:
        if client is not None:
            client.close()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE))
    parser.add_argument("--json", action="store_true")


def _net(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--connection", default=None)
    parser.add_argument("--i1-state-dir", default=str(ROOT / "data" / "i1"))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--token-env", default=None)
    parser.add_argument("--deadline", type=float, default=900.0)


def _cmd_masked_prepare(args):
    from .masked_edit import prepare
    payload = prepare(args.spec, args.output)
    if args.json:
        return 0, payload
    print("Подготовлено offline: " + payload["output"])
    print("Генерация не запускалась. Preview guide и mask находятся в пакете.")
    return 0, None


def _cmd_masked_verify(args):
    from .masked_edit import verify
    payload = verify(args.package, args.image)
    code = 0 if payload["status"] == "PASS" else 2
    if args.json:
        return code, payload
    print(f"{payload['status']}: изменённых защищённых пикселей {payload['changed_protected_pixels']}")
    print("Качество позы этой проверкой не оценивается.")
    return code, None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avs.ps1 keyframes",
        description="KF1: опорные кадры по референсам. По умолчанию только offline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    plan_p = sub.add_parser("plan", help="создать/проверить план без сети")
    _common(plan_p)
    source = plan_p.add_mutually_exclusive_group()
    source.add_argument("--from-json", default=None)
    source.add_argument("--from-text", default=None)
    source.add_argument("--interactive", action="store_true")
    source.add_argument("--example", action="store_true", help="использовать существующий example JSON")
    plan_p.add_argument("--save", default=None)
    generate_p = sub.add_parser("generate", help="offline preview или --execute POST")
    _common(generate_p)
    _net(generate_p)
    generate_p.add_argument("--plan", required=True)
    generate_p.add_argument("--raise-limit", action="store_true")
    generate_p.add_argument(
        "--until-frame",
        default=None,
        help="остановиться после COMPLETE/PRESERVED указанного frame_id включительно",
    )
    status_p = sub.add_parser("status", help="durable local status, без сети")
    _common(status_p)
    status_p.add_argument("--plan", required=True)
    status_p.add_argument("--offline", action="store_true", default=True)
    retry_p = sub.add_parser("retry-frame", help="explicit owner retry новой версией")
    _common(retry_p)
    _net(retry_p)
    retry_p.add_argument("--plan", required=True)
    retry_p.add_argument("--frame", required=True)
    retry_p.add_argument("--owner-ack", action="store_true")
    retry_p.add_argument("--raise-limit", action="store_true")
    approve_p = sub.add_parser(
        "approve", help="offline owner approval точной version/content SHA"
    )
    _common(approve_p)
    approve_p.add_argument("--plan", required=True)
    approve_p.add_argument("--frame", required=True)
    approve_p.add_argument("--version", required=True, type=int)
    approve_p.add_argument("--sha256", required=True)
    approve_p.add_argument("--owner-ack", action="store_true")
    approve_p.add_argument("--note", default=None)
    export_p = sub.add_parser("export", help="локальная storyboard gallery, без video")
    _common(export_p)
    export_p.add_argument("--plan", required=True)
    export_p.add_argument("--output", default=None)
    release_p = sub.add_parser("release", help="explicit /free только при empty queue")
    _common(release_p)
    _net(release_p)
    release_p.add_argument("--plan", required=True)
    masked_p = sub.add_parser("masked-prepare", help="offline пакет локальной правки позы по маске")
    _common(masked_p)
    masked_p.add_argument("--spec", required=True)
    masked_p.add_argument("--output", required=True)
    verify_p = sub.add_parser("masked-verify", help="проверить RGB результата вне маски, без сети")
    _common(verify_p)
    verify_p.add_argument("--package", required=True)
    verify_p.add_argument("--image", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        code, payload = {
            "masked-prepare": _cmd_masked_prepare,
            "masked-verify": _cmd_masked_verify,
            "plan": _cmd_plan,
            "generate": _cmd_generate,
            "status": _cmd_status,
            "retry-frame": _cmd_retry,
            "approve": _cmd_approve,
            "export": _cmd_export,
            "release": _cmd_release,
        }[args.command](args)
        if args.json and payload is not None:
            _print_json(payload)
        return code
    except KF1Error as exc:
        message = str(exc)
        if args.json:
            _print_json({"status": "ERROR", "error": type(exc).__name__, "message": message})
        else:
            print(f"Ошибка: {message}", file=sys.stderr)
        return getattr(exc, "exit_code", 2)
    except KeyboardInterrupt:
        print("Прервано; durable state сохранён. Используйте status/resume.", file=sys.stderr)
        return 5
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if args.json:
            _print_json({"status": "ERROR", "error": type(exc).__name__, "message": message})
        else:
            print(f"Непредвиденная ошибка: {message}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())


