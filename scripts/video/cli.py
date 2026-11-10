"""CLI and terminal wizard for local video plans and explicit GPU execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from scripts.kf1.connection import open_kf1_connection
from scripts.kf1.comfy import ComfyClient
from . import core, media

PROFILE = core.ROOT/'config/i1/bundle-wan22-qwen-edit-2511.profile.json'


def parser():
    result = argparse.ArgumentParser(prog='avs.ps1 video', description='A/B → Wan → MP4. Без --execute сеть не используется.')
    result.add_argument('--root', type=Path, default=core.DEFAULT_ROOT)
    result.add_argument('--json', action='store_true')
    sub = result.add_subparsers(dest='command', required=True)
    make = sub.add_parser('create')
    make.add_argument('--id', required=True)
    make.add_argument('--a', type=Path, required=True)
    make.add_argument('--b', type=Path, required=True)
    make.add_argument('--prompt', required=True)
    make.add_argument('--negative', default='')
    make.add_argument('--portrait', action='store_true')
    make.add_argument('--frames', type=int, default=121)
    make.add_argument('--seed', type=int)
    clone = sub.add_parser('clone', help='Новая версия с новым ID и seed; старая остаётся неизменной')
    clone.add_argument('id')
    clone.add_argument('--new-id', required=True)
    clone.add_argument('--prompt')
    clone.add_argument('--seed', type=int)
    story = sub.add_parser('from-storyboard')
    story.add_argument('--storyboard', type=Path, required=True)
    story.add_argument('--scene', required=True)
    story.add_argument('--id', required=True)
    story.add_argument('--prompt', required=True)
    for name in ('status', 'show', 'approve', 'generate', 'resume', 'accept', 'cancel'):
        child = sub.add_parser(name)
        child.add_argument('id')
        if name in {'approve', 'accept'}:
            child.add_argument('--sha256', required=True)
        if name in {'generate', 'resume', 'cancel'}:
            child.add_argument('--execute', action='store_true')
            child.add_argument('--connection')
            child.add_argument('--profile', default=str(PROFILE))
            child.add_argument('--base-url', help='Явный manual/test loopback, не подтверждение I1')
            child.add_argument('--deadline', type=float, default=1800)
    sub.add_parser('list')
    sub.add_parser('doctor')
    assembly = sub.add_parser('assemble')
    assembly.add_argument('--scenes', nargs='+', required=True)
    assembly.add_argument('--output', type=Path, required=True)
    assembly.add_argument('--execute', action='store_true')
    return result


def from_storyboard(args):
    manifest = core.read_json(args.storyboard)
    if manifest.get('schema') != 'anime-studio/kf1-storyboard/v1':
        raise ValueError('Нужен экспорт storyboard.json из keyframes export')
    scene = next((s for s in manifest['scenes'] if s['scene_id'] == args.scene), None)
    if scene is None:
        raise ValueError('Сцена не найдена')
    files = {}
    for role in ('A', 'B'):
        item = next((r for r in scene['roles'] if r['role'] == role), None)
        if not item or item.get('status') != 'COMPLETE':
            raise ValueError(f'{role}: кадр не готов')
        base = args.storyboard.parent.resolve()
        path = core.no_links(base/item['path'])
        if base not in path.resolve().parents or core.sha(path.read_bytes()) != item['sha256']:
            raise ValueError(f'{role}: неверный путь или SHA экспортированного кадра')
        files[role] = path
    info = __import__('scripts.kf1.normalize', fromlist=['inspect_image']).inspect_image(files['A'])
    return core.create(args.id, files['A'], files['B'], args.prompt,
                       landscape=info['width'] > info['height'], root=args.root)


def dispatch(args, emit=print):
    root = args.root
    if args.command == 'create':
        return core.create(args.id, args.a, args.b, args.prompt, negative=args.negative,
            landscape=not args.portrait, frames=args.frames, seed=args.seed, root=root)
    if args.command == 'from-storyboard':
        return from_storyboard(args)
    if args.command == 'clone':
        folder, plan, _ = core.load(args.id, root)
        return core.create(args.new_id, folder/'A.png', folder/'B.png',
            args.prompt or plan['prompt'], negative=plan['negative'],
            landscape=plan['width'] > plan['height'], frames=plan['frames'], seed=args.seed, root=root)
    if args.command == 'list':
        return {'videos': [core.status(p.name, root=root) for p in sorted(root.glob('*'))
                           if p.is_dir() and (p/'plan.json').exists()]}
    if args.command == 'doctor':
        try:
            executable = media.ffmpeg()
            version = media.command([executable, '-version']).stdout.splitlines()[0]
            return {'status': 'LOCAL_READY', 'ffmpeg': executable, 'version': version,
                    'gpu': 'NOT_CHECKED', 'template_sha256': core.sha(core.TEMPLATE.read_bytes())}
        except ValueError as exc:
            return {'status': 'LOCAL_BLOCKED', 'error': str(exc), 'gpu': 'NOT_CHECKED'}
    if args.command == 'approve':
        return core.approve(args.id, args.sha256, root=root)
    if args.command == 'accept':
        return core.accept(args.id, args.sha256, root=root)
    if args.command == 'show':
        folder, _, _ = core.load(args.id, root)
        path = folder/'video.mp4' if (folder/'video.mp4').exists() else folder
        if sys.platform == 'win32':
            os.startfile(str(path))
        return {'status': 'OPENED', 'path': str(path)}
    if args.command == 'status':
        return core.status(args.id, root=root)
    if args.command == 'assemble':
        if not args.execute:
            return {'status': 'PLAN_ONLY', 'ordered_scenes': args.scenes, 'output': str(args.output), 'network': False}
        return core.assemble(args.scenes, args.output, root=root)
    folder, _, digest = core.load(args.id, root)
    if not args.execute:
        return {'status': 'PLAN_ONLY', 'id': args.id, 'plan_sha256': digest,
                'note': 'Требуются просмотр A/B, approve и явный --execute. Повторных POST нет.'}
    if args.command == 'resume' and not (folder/'state.json').exists():
        raise ValueError('Нет существующего задания для resume; сначала generate')
    core.approved(folder, digest)
    media.ffmpeg()  # fail locally before network or submission
    if not args.base_url and 'bundle' not in Path(args.profile).name:
        raise ValueError('Видео требует bundle профиль с Wan, не Qwen-only')
    opened = open_kf1_connection(base_url=args.base_url, connection=args.connection,
        state_dir=core.ROOT/'data/i1', profile_path=args.profile)
    with ComfyClient(opened.base_url) as client:
        if args.command == 'cancel':
            # Never issue a global /interrupt: it could stop another user's job.
            from anime_studio.lock import InstanceLock
            with InstanceLock(folder/'.run.lock'):
                state = core.status(args.id, root=root)
                prompt = state.get('prompt_id')
                running, pending = client.queue_ids()
                if not prompt or prompt in running:
                    raise ValueError('Задача выполняется или prompt_id неизвестен; глобальный interrupt запрещён')
                if prompt not in pending:
                    raise ValueError('Точный prompt отсутствует в pending; используйте resume для readback')
                submitted = core.read_json(folder/'submitted.api.json')
                candidates = [item for item in client.queue()['queue_pending']
                              if core.exact_queue_entry(item, state, submitted) and item[1] == prompt]
                if len(candidates) != 1 or core.sha(core.canonical(submitted)) != state['submitted_graph_sha256']:
                    raise ValueError('Отмена запрещена: queue не подтверждает exact graph/correlation/nonce')
                code, _, _ = client._request('POST', '/queue', json_body={'delete': [prompt]})
                if code != 200 or any(prompt in group for group in client.queue_ids()):
                    raise ValueError('Отмена не подтверждена')
                state['status'] = 'CANCELLED'
                core._atomic_json(folder/'state.json', state)
                return state
        return core.execute(args.id, client, root=root, attestation=opened.attestation,
                            wait=args.deadline, emit=emit)


def main(argv=None, *, progress_sink=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args, emit=progress_sink or print)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get('status') in {'FAILED', 'LOCAL_BLOCKED'} else 0
    except KeyboardInterrupt:
        print('Прервано наблюдение. Генерация может продолжаться; используйте video resume. Повторный POST запрещён.')
        return 5
    except Exception as exc:
        from scripts.i1.util import redact
        print('Ошибка: '+redact(str(exc), limit=1500), file=sys.stderr)
        return 2
