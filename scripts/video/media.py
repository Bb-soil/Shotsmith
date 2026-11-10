"""Local FFmpeg tools; no downloads, shell interpolation or hidden GPU calls."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def ffmpeg() -> str:
    configured = os.environ.get('SHOTSMITH_FFMPEG')
    candidates = [configured] if configured else [
        str(ROOT / 'data/tools/ffmpeg.exe'), shutil.which('ffmpeg'),
    ]
    try:
        import imageio_ffmpeg
        if not configured:
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise ValueError('FFmpeg не найден. Укажите SHOTSMITH_FFMPEG или data/tools/ffmpeg.exe; автоматического скачивания нет.')


def command(argv: list[str], timeout: float = 300) -> subprocess.CompletedProcess:
    # Capture to disk rather than an unbounded PIPE during media decode.
    with tempfile.TemporaryFile() as log:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        log.seek(0, 2)
        length = log.tell()
        log.seek(max(0, length - 16384))
        text = log.read().decode('utf-8', errors='replace')
    if result.returncode:
        raise ValueError(f'FFmpeg exit {result.returncode}: {text[-3000:]}')
    return subprocess.CompletedProcess(argv, result.returncode, stdout=text)


def probe(path: Path, *, decode: bool = True) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**3:
        raise ValueError('Нет обычного MP4 в пределах 1 GiB')
    # -frames:v 1 succeeds, unlike ffmpeg -i alone (which intentionally exits 1).
    report = command([ffmpeg(), '-hide_banner', '-i', str(path), '-map', '0:v:0',
                      '-frames:v', '1', '-f', 'null', '-']).stdout
    size = re.search(r'Video:.*?\b(\d{2,5})x(\d{2,5})\b', report)
    fps = re.search(r'(\d+(?:\.\d+)?) fps', report)
    duration = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', report)
    if not (size and fps and duration):
        raise ValueError('Не удалось проверить geometry/fps/duration MP4')
    seconds = int(duration[1])*3600 + int(duration[2])*60 + float(duration[3])
    if seconds <= 0 or seconds > 600:
        raise ValueError('Длительность вне диапазона 0..600 секунд')
    if decode:
        command([ffmpeg(), '-v', 'error', '-xerror', '-i', str(path), '-map', '0:v:0',
                 '-f', 'null', '-'], timeout=max(120, seconds*10))
    return {'width': int(size[1]), 'height': int(size[2]), 'fps': float(fps[1]),
            'duration_seconds': seconds, 'decoded': decode}
