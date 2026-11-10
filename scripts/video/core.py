"""Immutable local plans, exact approvals, one-POST execution and recovery."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid

from anime_studio.lock import InstanceLock
from anime_studio.workflow import safe_relative_subfolder, safe_remote_filename
from scripts.kf1.comfy import ComfyClient
from scripts.kf1.normalize import inspect_image, normalize_image
from scripts.kf1.runtime_gate import validate_bound_workflow
from scripts.kf1.store import _atomic_bytes, _atomic_json
from . import media

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / 'data/videos'
TEMPLATE = ROOT / 'workflows/P1/wan22-14b-shot02-alley-v1.api.json'
TEMPLATE_SHA = '9b1967e71da48263c53acc226a0a51daa39378a5cbcfbaa7ceb3988b2969c55d'
SAFE_ID = re.compile(r'^[a-z0-9][a-z0-9_-]{0,63}$')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def no_links(path: Path) -> Path:
    path = path.absolute()
    for part in [path, *path.parents]:
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError(f'Ссылка/junction запрещена: {part}')
    return path


def read_json(path: Path, limit: int = 8*1024*1024):
    no_links(path)
    if path.stat().st_size > limit:
        raise ValueError('JSON превышает лимит')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def directory(job: str, root: Path = DEFAULT_ROOT) -> Path:
    if not SAFE_ID.fullmatch(job):
        raise ValueError('ID: латинские строчные буквы, цифры, _ и -, до 64 символов')
    return no_links(Path(root) / job)


def load(job: str, root: Path = DEFAULT_ROOT) -> tuple[Path, dict, str]:
    folder = directory(job, root)
    plan = read_json(folder / 'plan.json')
    if plan.get('schema') != 'shotsmith/video-plan/v1' or plan.get('id') != job:
        raise ValueError('Неверная схема/ID видеоплана')
    expected = (folder / 'plan.sha256').read_text(encoding='ascii').strip()
    digest = sha(canonical(plan))
    if digest != expected:
        raise ValueError('Видеоплан изменён: создайте новый ID вместо перезаписи')
    for role in ('A', 'B'):
        for suffix, field in (('.png', 'normalized_sha256'), ('.source', 'source_sha256')):
            path = no_links(folder / (role + suffix))
            if sha(path.read_bytes()) != plan['inputs'][role][field]:
                raise ValueError(f'{role}: вход изменён')
    return folder, plan, digest


def exact_queue_entry(entry, state: dict, graph: dict) -> bool:
    return (isinstance(entry, list) and len(entry) >= 4 and isinstance(entry[1], str)
            and entry[2] == graph and isinstance(entry[3], dict)
            and entry[3].get('correlation_key') == state['correlation']
            and entry[3].get('nonce') == state['nonce'])


def create(job: str, a: Path, b: Path, prompt: str, *, negative: str = '',
           landscape: bool = True, frames: int = 121, seed: int | None = None,
           root: Path = DEFAULT_ROOT) -> dict:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
        raise ValueError('Нужно описание движения длиной 1..16000 символов')
    if len(negative) > 8000 or type(frames) is not int or not 17 <= frames <= 241 or (frames-1)%8:
        raise ValueError('Кадров должно быть 8n+1, от 17 до 241; negative до 8000 символов')
    if seed is None:
        seed = uuid.uuid4().int % 2**31
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Seed: целое 0..2147483647')
    sources = {r: no_links(Path(p)) for r, p in [('A', a), ('B', b)]}
    for path in sources.values():
        if not path.is_file() or path.stat().st_size > 64*1024*1024:
            raise ValueError('Изображение должно быть обычным файлом до 64 MiB')
        inspect_image(path)
    folder = directory(job, root)
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    width, height = (816, 448) if landscape else (448, 816)
    inputs = {}
    for role, path in sources.items():
        data = path.read_bytes()
        _atomic_bytes(folder / (role + '.source'), data)
        normalized = normalize_image(path, folder / (role + '.png'), policy='contain',
                                     target=[width, height], pad=True, label='video-'+role)
        inputs[role] = {'source_path': str(path.resolve()), 'source_sha256': sha(data),
                        'normalized_sha256': normalized['output']['sha256']}
    plan = {'schema': 'shotsmith/video-plan/v1', 'id': job, 'inputs': inputs,
            'prompt': prompt, 'negative': negative, 'width': width, 'height': height,
            'frames': frames, 'fps': 24, 'seed': seed, 'steps': 20, 'cfg': 4,
            'workflow_sha256': TEMPLATE_SHA, 'max_posts': 1, 'automatic_retry': False}
    _atomic_json(folder / 'plan.json', plan)
    _atomic_bytes(folder / 'plan.sha256', (sha(canonical(plan))+'\n').encode('ascii'))
    return {'status': 'PLAN_ONLY', 'id': job, 'path': str(folder),
            'plan_sha256': sha(canonical(plan)), 'seconds': frames/24,
            'geometry': [width, height], 'posts': 1}


def approve(job: str, expected: str, *, root: Path = DEFAULT_ROOT) -> dict:
    folder, _, digest = load(job, root)
    if expected != digest:
        raise ValueError('Одобрение требует точный SHA256 текущего плана')
    _atomic_json(folder / 'approval.json', {'plan_sha256': digest, 'approved_at': time.time()})
    return {'status': 'APPROVED', 'id': job, 'plan_sha256': digest}


def status(job: str, *, root: Path = DEFAULT_ROOT) -> dict:
    folder, plan, digest = load(job, root)
    state = read_json(folder / 'state.json') if (folder / 'state.json').exists() else {'status': 'PLANNED'}
    return {'id': job, 'plan_sha256': digest, 'path': str(folder), 'seconds': plan['frames']/24,
            **state}


def graph_for(plan: dict, names: dict[str, str], prefix: str) -> dict:
    if sha(TEMPLATE.read_bytes()) != TEMPLATE_SHA or plan['workflow_sha256'] != TEMPLATE_SHA:
        raise ValueError('Исходный Wan workflow не совпадает с закреплённым SHA256')
    graph = read_json(TEMPLATE)
    graph['80']['inputs']['image'] = names['A']
    graph['89']['inputs']['image'] = names['B']
    graph['81']['inputs'].update(width=plan['width'], height=plan['height'], length=plan['frames'])
    graph['90']['inputs']['text'] = plan['prompt']
    graph['78']['inputs']['text'] = plan['negative']
    graph['84']['inputs']['noise_seed'] = plan['seed']
    graph['83']['inputs']['filename_prefix'] = prefix
    return graph


def approved(folder: Path, digest: str):
    if not (folder / 'approval.json').exists() or read_json(folder / 'approval.json').get('plan_sha256') != digest:
        raise ValueError('Сначала просмотрите A/B и одобрите точный SHA256 видеоплана')


def execute(job: str, client: ComfyClient, *, root: Path = DEFAULT_ROOT,
            attestation: dict, wait: float = 1800, emit=print) -> dict:
    if not math.isfinite(wait) or not 1 <= wait <= 3600:
        raise ValueError('Ожидание: 1..3600 секунд')
    folder, plan, digest = load(job, root)
    lock = InstanceLock(folder / '.run.lock')
    with lock:
        approved(folder, digest)
        state_path = folder / 'state.json'
        state = read_json(state_path) if state_path.exists() else None
        if state and state.get('plan_sha256') != digest:
            raise ValueError('Состояние другого плана')
        if state and state.get('status') == 'COMPLETE':
            if sha((folder / 'video.mp4').read_bytes()) != state.get('sha256'):
                raise ValueError('Сохранённое видео изменено')
            return state
        if state and state.get('status') in {'FAILED', 'CANCELLED'}:
            raise ValueError('Задание завершено с ошибкой/отменено; повтор требует нового плана и одобрения')
        if state and not state.get('prompt_id'):
            # Reads only: an INTENT/AMBIGUOUS run must never submit a second POST.
            found = client.find_prompt_by_correlation(state['correlation'], state['nonce'])
            if not found and hasattr(client, 'queue'):
                submitted = read_json(folder/'submitted.api.json')
                if sha(canonical(submitted)) != state['submitted_graph_sha256']:
                    raise ValueError('Сохранённый граф изменён')
                queue = client.queue()
                matches = [item[1] for group in ('queue_running', 'queue_pending')
                           for item in queue[group] if exact_queue_entry(item, state, submitted)]
                if len(matches) > 1:
                    raise ValueError('Несколько заданий с одной identity; повтор запрещён')
                found = matches[0] if matches else None
            if not found:
                raise ValueError('Приём предыдущего POST неизвестен. Повтор запрещён; дождитесь history и используйте resume.')
            state['prompt_id'] = found
            state['status'] = 'SUBMITTED'
            _atomic_json(state_path, state)
        if state is None:
            running, pending = client.queue_ids()
            if running or pending:
                raise ValueError('Очередь ComfyUI занята. Новое задание не отправлено.')
            run = uuid.uuid4().hex
            receipts, names = {}, {}
            for role in ('A', 'B'):
                emit(f'[UPLOAD] {role}')
                receipt = client.upload_png(folder / (role+'.png'), remote_name=f'shotsmith-{run}-{role}.png')
                if receipt['source_sha256'] != plan['inputs'][role]['normalized_sha256']:
                    raise ValueError('Upload SHA256 не совпадает')
                names[role] = receipt['name']
                receipts[receipt['name']] = receipt
            graph = graph_for(plan, names, f'shotsmith/video/{job}/{run}')
            validate_bound_workflow(graph, client.object_info(), upload_receipts=receipts)
            _atomic_json(folder / 'submitted.api.json', graph)
            _atomic_json(folder / 'runtime.json', attestation)
            state = {'status': 'INTENT', 'plan_sha256': digest, 'correlation': run,
                     'nonce': uuid.uuid4().hex, 'prompt_id': None,
                     'submitted_graph_sha256': sha(canonical(graph)), 'posts_reserved': 1}
            _atomic_json(state_path, state)  # durable intent BEFORE the one POST
            try:
                result = client.submit(graph, client_id=run, correlation_key=run, nonce=state['nonce'])
            except Exception:
                state['status'] = 'AMBIGUOUS'
                _atomic_json(state_path, state)
                raise
            state.update(status='SUBMITTED', prompt_id=result.prompt_id)
            _atomic_json(state_path, state)
        prompt_id = state['prompt_id']
        deadline = time.monotonic()+wait
        last_emit = 0.0
        while time.monotonic() < deadline:
            entry = client.history(prompt_id)
            if entry:
                identity = entry.get('prompt')
                graph = read_json(folder / 'submitted.api.json')
                if (not isinstance(identity, list) or len(identity) < 4 or identity[1] != prompt_id
                        or identity[2] != graph or sha(canonical(graph)) != state['submitted_graph_sha256']
                        or identity[3].get('correlation_key') != state['correlation']
                        or identity[3].get('nonce') != state['nonce']):
                    raise ValueError('History не принадлежит точному отправленному графу')
                _atomic_json(folder / 'history.json', entry)
                if entry.get('status', {}).get('status_str') == 'error':
                    state.update(status='FAILED', error='ComfyUI execution error; см. history.json')
                    _atomic_json(state_path, state)
                    return state
                outputs = entry.get('outputs', {}).get('83', {})
                found = [item for items in outputs.values() if isinstance(items, list)
                         for item in items if isinstance(item, dict) and str(item.get('filename', '')).lower().endswith('.mp4')]
                if len(found) != 1:
                    raise ValueError('History не содержит ровно один MP4 выхода SaveVideo 83')
                item = found[0]
                filename = safe_remote_filename(item['filename'])
                subfolder = safe_relative_subfolder(item.get('subfolder', ''))
                prefix = graph['83']['inputs']['filename_prefix']
                expected_dir, expected_stem = prefix.rsplit('/', 1)
                if item.get('type') != 'output' or subfolder != expected_dir or not filename.startswith(expected_stem+'_'):
                    raise ValueError('MP4 output mapping не совпадает с exact prefix')
                emit('[COLLECT] Получение MP4')
                code, _, data = client._request('GET', '/view', params={'filename': filename,
                    'subfolder': subfolder, 'type': 'output'}, max_bytes=256*1024*1024, timeout=180)
                if code != 200 or b'ftyp' not in data[:32]:
                    raise ValueError('Ответ не является MP4')
                _atomic_bytes(folder / 'video.part.mp4', data)
                facts = media.probe(folder / 'video.part.mp4')
                if (facts['width'], facts['height'], facts['fps']) != (plan['width'], plan['height'], plan['fps']):
                    raise ValueError('Geometry/FPS MP4 не совпадает с планом')
                if abs(facts['duration_seconds']-plan['frames']/24) > .12:
                    raise ValueError('Длительность MP4 не совпадает с планом')
                (folder / 'video.part.mp4').replace(folder / 'video.mp4')
                state.update(status='COMPLETE', sha256=sha(data), path=str(folder/'video.mp4'), media=facts)
                _atomic_json(state_path, state)
                emit('[COMPLETE] '+str(folder/'video.mp4'))
                return state
            if time.monotonic()-last_emit >= 10:
                emit('[SAMPLING] '+prompt_id+' — расчёт, повторных POST нет')
                last_emit = time.monotonic()
            time.sleep(min(3, max(0, deadline-time.monotonic())))
        return {**state, 'status': 'WAITING', 'note': 'Используйте resume: только тот же prompt_id'}


def accept(job: str, expected: str, *, root: Path = DEFAULT_ROOT) -> dict:
    folder, _, digest = load(job, root)
    state = status(job, root=root)
    path = folder/'video.mp4'
    if state.get('status') != 'COMPLETE' or expected != state.get('sha256') or sha(path.read_bytes()) != expected:
        raise ValueError('Требуется точный SHA256 завершённого MP4')
    _atomic_json(folder/'accepted.json', {'sha256': expected, 'plan_sha256': digest})
    return {'status': 'ACCEPTED_BY_OWNER', 'id': job}


def assemble(jobs: list[str], output: Path, *, root: Path = DEFAULT_ROOT) -> dict:
    if not jobs or len(jobs) > 100:
        raise ValueError('Выберите 1..100 принятых сцен в нужном порядке')
    output = no_links(output)
    if output.exists() or output.with_suffix('.manifest.json').exists() or output.suffix.lower() != '.mp4':
        raise ValueError('Выход должен быть новым файлом .mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    from tempfile import TemporaryDirectory
    with InstanceLock(output.with_suffix('.assembly.lock')), TemporaryDirectory(prefix='shotsmith-', dir=output.parent) as temp:
        temp = Path(temp)
        facts, provenance = [], []
        # Snapshot exact accepted bytes before ffmpeg; independent from live ledger edits.
        for index, job in enumerate(jobs):
            folder, _, digest = load(job, root)
            approved_video = read_json(folder/'accepted.json')
            state = status(job, root=root)
            data = (folder/'video.mp4').read_bytes()
            if (state.get('status') != 'COMPLETE' or sha(data) != approved_video['sha256']
                    or approved_video['plan_sha256'] != digest):
                raise ValueError(f'{job}: требуется актуальное принятие MP4')
            snapshot = temp/f'{index:04d}.mp4'
            snapshot.write_bytes(data)
            facts.append(media.probe(snapshot))
            provenance.append({'id': job, 'sha256': sha(data), 'plan_sha256': digest})
        geometry = {(f['width'], f['height'], f['fps']) for f in facts}
        if len(geometry) != 1:
            raise ValueError('Сцены имеют разные geometry/FPS; молчаливого растягивания нет')
        if sum(f['duration_seconds'] for f in facts) > 600:
            raise ValueError('Локальная сборка ограничена 10 минутами')
        listing = temp/'concat.txt'
        listing.write_text('\n'.join(f"file '{n:04d}.mp4'" for n in range(len(jobs))), encoding='utf-8')
        result = temp/'result.mp4'
        media.command([media.ffmpeg(), '-v', 'error', '-f', 'concat', '-safe', '1', '-i', str(listing),
                       '-map', '0:v:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                       '-metadata', 'comment=AI-generated with Shotsmith', '-movflags', '+faststart', str(result)])
        result_facts = media.probe(result)
        if abs(result_facts['duration_seconds']-sum(f['duration_seconds'] for f in facts)) > .15:
            raise ValueError('Длительность сборки не совпадает с суммой сцен')
        data = result.read_bytes()
        with output.open('xb') as stream:
            stream.write(data)
        evidence = {'status': 'ASSEMBLED', 'path': str(output), 'sha256': sha(data),
                    'media': result_facts, 'scenes': provenance}
        _atomic_json(output.with_suffix('.manifest.json'), evidence)
        return evidence
