"""Human-facing Shotsmith workflows; delegates to the existing command handlers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

from scripts.video import core, media
from scripts.video.cli import main as video_main

ROOT = core.ROOT


def yes(console, text):
    return console.ask(text+' [y/N]').lower() in {'y', 'yes', 'д', 'да'}


def path(console, text):
    return Path(console.ask(text).strip('"'))


def open_local(value):
    if os.name == 'nt':
        os.startfile(str(value))
    print('  '+str(value))


def projects():
    found = []
    for entry in sorted((ROOT/'data/projects').glob('*/project.json')):
        try:
            # Use the project store's canonical identity/tamper validation.
            from scripts.kf2.store import ProjectStore
            project = ProjectStore().load(entry.parent.name)
            found.append({'id': entry.parent.name, 'title': project.get('title'),
                          'status': project.get('status'), 'revision': project.get('revision')})
        except Exception as exc:
            found.append({'id': entry.parent.name, 'error': str(exc)})
    return found


def keyframes(console, command, args):
    from scripts.kf1.cli import main
    from scripts.kf1.plan import resolve_plan
    from scripts.kf1.store import PlanStore
    from scripts.kf1.runner import Runner
    if args:
        return main(args)
    plan_path = path(console, 'Путь к подготовленному plan.json')
    plan = resolve_plan(plan_path)
    if command == 'storyboard':
        output = ROOT/'data/storyboards'/plan.plan_sha256[:16]
        code = main(['export', '--plan', str(plan.path), '--output', str(output)])
        if code == 0:
            open_local(output/'index.html')
        return code
    if command == 'approve':
        store = PlanStore(ROOT/'data/kf1', plan.plan_sha256, plan.path)
        runner = Runner(plan, store, None)
        state = store.load_any(plan=plan)
        payload = runner.status(state)
        for item in payload['frames']:
            print(f"  {item['frame_id']} v{item.get('current_version')} {item.get('status')} SHA={item.get('content_sha256')}")
        frame = console.ask('Какой кадр одобрить (например shot-01/B)')
        item = next((v for v in payload['frames'] if v['frame_id'] == frame), None)
        if not item or not item.get('content_sha256') or not item.get('current_version'):
            raise ValueError('Нет завершённого кадра для одобрения')
        output = item.get('output') or {}
        if output.get('path'):
            open_local(Path(output['path']))
        if not yes(console, 'Просмотрели и принимаете именно эту версию и SHA?'):
            return 0
        return main(['approve', '--plan', str(plan.path), '--frame', frame,
                     '--version', str(item['current_version']), '--sha256', item['content_sha256'], '--owner-ack'])
    action = console.ask('Кадры: status / generate / retry / export', 'status')
    if action == 'export':
        output = ROOT/'data/storyboards'/plan.plan_sha256[:16]
        code = main(['export', '--plan', str(plan.path), '--output', str(output)])
        if code == 0: open_local(output/'index.html')
        return code
    if action == 'status':
        return main(['status', '--plan', str(plan.path)])
    if action not in {'generate', 'retry'}:
        raise ValueError('Неизвестная операция кадров')
    if not console.name:
        raise ValueError('Сначала /use или /connect')
    frame = console.ask('Один целевой кадр, например shot-01/B')
    if action == 'generate':
        code = main(['generate', '--plan', str(plan.path), '--until-frame', frame])
        argv = ['generate', '--plan', str(plan.path), '--until-frame', frame]
    else:
        code = main(['status', '--plan', str(plan.path)])
        argv = ['retry-frame', '--plan', str(plan.path), '--frame', frame, '--owner-ack']
    if code or not yes(console, 'Отправить на выбранный GPU? Генерация может быть платной; готовые кадры сохраняются.'):
        return code
    return main([*argv, '--connection', console.name, '--profile', console.profile, '--execute'])


def video(console, args):
    if args:
        if args[0] in {'generate', 'resume', 'cancel'} and '--base-url' not in args:
            if '--connection' not in args and console.name:
                args += ['--connection', console.name]
            if '--profile' not in args:
                args += ['--profile', console.profile]
        return video_main(args, progress_sink=console.ui.progress)
    print('  Видео: create — выбрать A/B; storyboard — из раскадровки; clone — новая версия; approve; generate; resume; status; show; accept; cancel; list')
    action = console.ask('Действие', 'create')
    if action == 'list':
        return video_main(['list'])
    if action == 'clone':
        old = console.ask('ID исходной видеосцены')
        _, plan, _ = core.load(old)
        new = console.ask('Новый ID', old+'-'+uuid.uuid4().hex[:6])
        prompt = console.ask('Описание движения', plan['prompt'])
        return video_main(['clone', old, '--new-id', new, '--prompt', prompt])
    if action in {'create', 'storyboard'}:
        job = console.ask('ID новой видеосцены', 'scene-'+uuid.uuid4().hex[:8])
        if action == 'create':
            a = path(console, 'Кадр A — путь к изображению')
            b = path(console, 'Кадр B — путь к изображению')
            prompt = console.ask('Опишите движение между A и B (русский поддерживается)')
            negative = console.ask('Чего не должно быть (можно пусто)')
            portrait = console.ask('Формат landscape / portrait', 'landscape')
            if portrait not in {'landscape', 'portrait'}:
                raise ValueError('Формат: landscape или portrait')
            seconds = console.ask('Длительность 3 / 5 / 10 секунд', '5')
            if seconds not in {'3', '5', '10'}:
                raise ValueError('Выберите 3, 5 или 10 секунд')
            result = core.create(job, a, b, prompt, negative=negative,
                                 landscape=portrait == 'landscape', frames={'3':73, '5':121, '10':241}[seconds])
        else:
            from argparse import Namespace
            from scripts.video.cli import from_storyboard
            manifest = path(console, 'Путь к storyboard.json после /storyboard')
            content = core.read_json(manifest)
            for scene in content.get('scenes', []):
                print('  '+scene['scene_id']+' — '+scene.get('description', ''))
            scene = console.ask('ID сцены')
            prompt = console.ask('Точное описание движения для видео')
            result = from_storyboard(Namespace(storyboard=manifest, scene=scene, id=job, prompt=prompt, root=core.DEFAULT_ROOT))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        folder, plan, digest = core.load(job)
        open_local(folder)
        print(f"  {plan['frames']/24:.2f} с; {plan['width']}×{plan['height']}; 24 fps; seed={plan['seed']}")
        print('  Просмотрите A.png и B.png. Это одобрение входов, НЕ запуск GPU.')
        if yes(console, 'Принимаете эту пару и показанные параметры?'):
            print(core.approve(job, digest))
        print('  Следующее: /video → generate → '+job)
        return 0
    if action not in {'approve', 'generate', 'resume', 'status', 'show', 'accept', 'cancel'}:
        raise ValueError('Неизвестная операция видео')
    job = console.ask('ID видеосцены')
    if action == 'approve':
        folder, plan, digest = core.load(job)
        print(json.dumps(plan, ensure_ascii=False, indent=2)); open_local(folder)
        if yes(console, 'Принимаете A/B и эти параметры?'):
            print(core.approve(job, digest))
        return 0
    if action == 'accept':
        current = core.status(job)
        if current.get('status') != 'COMPLETE':
            raise ValueError('Сначала получите COMPLETE')
        open_local(Path(current['path']))
        print('  SHA256: '+current['sha256'])
        if yes(console, 'Принимаете именно этот ролик для сборки?'):
            print(core.accept(job, current['sha256']))
        return 0
    argv = [action, job]
    if action in {'generate', 'resume', 'cancel'}:
        if not console.name:
            raise ValueError('Сначала /use или /connect')
        print(json.dumps(core.status(job), ensure_ascii=False, indent=2))
        question = {'generate':'Запустить одну GPU-генерацию?', 'resume':'Получить существующее задание без повторного POST?', 'cancel':'Удалить только своё ожидающее задание из очереди? Работающее задание не прерывается.'}[action]
        if not yes(console, question):
            return 0
        argv += ['--execute', '--connection', console.name, '--profile', console.profile]
    return video_main(argv, progress_sink=console.ui.progress)


def handle(console, command, args):
    if command == 'video':
        return video(console, args)
    if command in {'frames', 'approve', 'storyboard'}:
        return keyframes(console, command, args)
    if command == 'assemble':
        if args:
            return video_main(['assemble', *args])
        video_main(['list'])
        jobs = console.ask('ID принятых видеосцен по порядку, через запятую').split(',')
        jobs = [j.strip() for j in jobs if j.strip()]
        output = path(console, 'Новый выходной файл .mp4')
        if yes(console, 'Собрать локально в этом порядке: '+', '.join(jobs)+'?'):
            print(json.dumps(core.assemble(jobs, output), ensure_ascii=False, indent=2))
            open_local(output)
        return 0
    if command in {'doctor', 'outputs'}:
        return video_main(['doctor' if command == 'doctor' else 'list'])
    if command == 'projects':
        print(json.dumps(projects(), ensure_ascii=False, indent=2))
        return 0
    from scripts.kf2.cli import main
    if command == 'new':
        return main(['create', *args])
    if command in {'edit', 'prepare'}:
        project = args[0] if args else console.ask('ID проекта')
        return main(['resume' if command == 'edit' else 'prepare', project])
    raise ValueError('Неизвестная команда')
