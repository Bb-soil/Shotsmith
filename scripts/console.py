"""Local slash-command shell over the existing AVS command handlers."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'bundle': 'config/i1/bundle-wan22-qwen-edit-2511.profile.json',
    'qwen': 'config/i1/keyframes-qwen-edit-2511.profile.json',
}
WORKER = {'connect', 'inspect', 'install', 'status', 'resume', 'open', 'disconnect'}


def split_command(line: str) -> list[str]:
    # posix=False preserves Windows backslashes. Quotes must enclose the whole
    # argument: --identity "C:\\My keys\\key" (the same convention as help).
    return [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p
            for p in shlex.split(line, posix=False)]


def clean(text: str) -> str:
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', str(text))


def enable_color() -> bool:
    if not sys.stdout.isatty() or 'NO_COLOR' in os.environ:
        return False
    if os.name == 'nt':
        try:
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetStdHandle.argtypes = [wintypes.DWORD]
            kernel.GetStdHandle.restype = wintypes.HANDLE
            kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            handle = kernel.GetStdHandle(-11 & 0xffffffff)
            mode = wintypes.DWORD()
            if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel.SetConsoleMode(handle, mode.value | 4))
        except (OSError, AttributeError):
            return False
    return True


class Terminal:
    def __init__(self, stream=None, color: bool | None = None):
        self.stream = stream or sys.stdout
        self.color = enable_color() if color is None else color
        self.tty = self.stream.isatty()
        self.active = False
        self.last_stage = None

    def tint(self, text: str, code='36') -> str:
        return f'\033[{code}m{text}\033[0m' if self.color else text

    def finish(self):
        if self.active:
            self.stream.write('\n')
            self.stream.flush()
            self.active = False

    def progress(self, line: str, **kwargs):
        line = clean(line)
        match = re.search(r'\s(\d+(?:\.\d+)?)%\s', line)
        stage = line.split(']', 1)[0] + ']'
        if match:
            percent = max(0, min(100, float(match.group(1))))
            filled = int(24 * percent / 100)
            bar = '█' * filled + '░' * (24 - filled)
            # Keep bytes/rate/ETA visible even with long model filenames.
            count = re.search(r'файл (\d+/\d+)', line[:match.start()])
            files = f' {count.group(1)} файлов' if count else ''
            line = f'{bar} {percent:5.1f}%{files}  ' + line[match.end():]
        width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
        if self.tty:
            if self.active and stage != self.last_stage:
                self.finish()
            rendered = line[:width]
            self.stream.write('\r' + self.tint(rendered) + ' ' * max(0, width-len(rendered)))
            self.stream.flush()
            self.active = True
            if not match or stage in {'[COMPLETE]', '[FAILED]', '[INTERRUPTED]'}:
                self.finish()
        else:
            print(line, file=self.stream, flush=True)
        self.last_stage = stage


HELP = '''
  /connect                 Подключить GPU: мастер адреса, порта и ключа
  /connections             Список сохранённых подключений
  /use имя                 Выбрать подключение; профиль читается из записи
  /profile bundle|qwen     Выбрать набор моделей (bundle = Wan + Qwen)
  /inspect                 Проверить совместимость сервера
  /install                 Показать план установки
  /install --execute       Установить после штатного подтверждения
  /status [--offline]      Состояние загрузки и готовности
  /resume                  Продолжить существующую установку
  /open                    Адрес ComfyUI
  /disconnect              Закрыть туннель; аренда продолжает тарифицироваться
  /project preview ID      Обычные команды проекта
  /keyframes status ...    Обычные команды кадров; --execute остаётся явным
  /new                     Создать проект: текст, референсы и сцены
  /projects                Список проектов
  /edit [ID]               Продолжить редактирование проекта
  /prepare [ID]            Проверить и подготовить план кадров
  /frames                  Статус / генерация / пересоздание кадров
  /approve                 Просмотреть и одобрить точную версию кадра
  /storyboard              Экспорт и просмотр раскадровки
  /video                   A/B → видеоплан → одобрение → Wan → MP4
  /assemble                Сборка принятых сцен в выбранном порядке
  /outputs                 Сохранённые видео и их состояния
  /doctor                  Проверить локальный FFmpeg без GPU
  /help                    Показать команды
  /exit                    Выйти из оболочки (аренда не останавливается)

  Дополнительные флаги доступны: /connect --help, /install --help.
  Пути с пробелами заключайте в кавычки. Команды не исполняются через shell.
'''


class Console:
    def __init__(self, terminal=None, input_fn=None):
        self.ui = terminal or Terminal()
        self.input = input_fn or input
        self.name = None
        self.profile = PROFILES['bundle']

    def ask(self, label, default=''):
        value = self.input(f'  {label}' + (f' [{default}]' if default else '') + ': ').strip()
        return value or default

    def use(self, name):
        from scripts.i1.ops import DEFAULT_STATE_DIR
        # The shell only lists/selects names; actual trust/containment checks
        # remain in I1. Never load a caller-selected arbitrary JSON path here.
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}', name):
            raise ValueError('Неверное имя подключения')
        from scripts.i1.profile import load_profile
        import json
        record = json.loads((Path(DEFAULT_STATE_DIR) / 'profiles' / f'{name}.json').read_text(encoding='utf-8-sig'))
        profile_id = record['identity']['profile_id']
        matches = [p for p in (ROOT / 'config/i1').glob('*.profile.json')
                   if load_profile(p).profile_id == profile_id]
        if len(matches) != 1:
            raise ValueError('Не удалось однозначно определить профиль; выберите /profile')
        self.name, self.profile = name, str(matches[0])
        print(f'  Выбрано: {name}')

    def execute(self, line):
        parts = split_command(line.strip())
        if not parts:
            return True
        command, args = parts[0], parts[1:]
        if not command.startswith('/'):
            print('  Введите /help — команды начинаются с /.')
            return True
        command = command[1:].lower()
        if command in {'exit', 'quit'}:
            print('  До встречи. Аренда и сервер не остановлены.')
            return False
        if command == 'help':
            print(HELP)
            return True
        if command == 'use':
            if len(args) != 1:
                raise ValueError('Использование: /use имя')
            self.use(args[0])
            return True
        if command == 'profile':
            if len(args) != 1 or args[0] not in PROFILES:
                raise ValueError('Использование: /profile bundle|qwen')
            self.profile = PROFILES[args[0]]
            print(f'  Профиль: {args[0]}')
            return True
        if command == 'connections':
            from scripts.i1.ops import DEFAULT_STATE_DIR
            names = sorted(p.stem for p in (Path(DEFAULT_STATE_DIR)/'profiles').glob('*.json'))
            print('\n'.join('  /use ' + clean(n) for n in names) or '  Подключений нет. Введите /connect.')
            return True
        if command in {'new', 'projects', 'edit', 'prepare', 'frames', 'approve', 'storyboard', 'video', 'assemble', 'outputs', 'doctor'}:
            from scripts.app_workflow import handle
            code = handle(self, command, args)
            print(f'  Завершено · код {code}')
            return True
        if command in {'project', 'keyframes'}:
            if command == 'project':
                from scripts.kf2.cli import main
            else:
                from scripts.kf1.cli import main
            code = main(args)
            print(f'  Завершено · код {code}')
            return True
        if command not in WORKER:
            raise ValueError('Неизвестная команда. Введите /help.')
        from scripts.i1 import cli
        selected = self.name
        if command == 'connect' and not args:
            selected = self.ask('Имя подключения')
            host = self.ask('Адрес сервера')
            port = self.ask('SSH-порт', '22')
            user = self.ask('Пользователь', 'root')
            identity = self.ask('Файл SSH-ключа', str(Path.home()/'.ssh/anime-video-studio'))
            workspace = self.ask('Рабочая папка сервера', '/workspace')
            args = ['--name', selected, '--host', host, '--port', port,
                    '--user', user, '--identity', identity.strip('"'), '--workspace', workspace]
        if '--help' not in args:
            if '--name' in args:
                selected = args[args.index('--name')+1]
            elif selected:
                args += ['--name', selected]
            else:
                raise ValueError('Сначала /connect или /use имя.')
            if '--profile' not in args:
                args += ['--profile', self.profile]
        try:
            code = cli.main([command, *args], progress_sink=self.ui.progress)
        finally:
            self.ui.finish()
        if command == 'connect' and code == 0:
            self.name = selected
            if '--profile' in args:
                self.profile = args[args.index('--profile')+1]
        print(self.ui.tint(f'  Завершено · код {code}', '32' if code == 0 else '33'))
        return True

    def run(self):
        print(self.ui.tint('\n  SHOTSMITH', '1;36'))
        print('  GPU · установка · проекты · опорные кадры')
        print('  /connect — подключение   /connections — сохранённые   /help — команды\n')
        while True:
            try:
                prompt = f'  shotsmith [{clean(self.name or "локально")}] › '
                if not self.execute(self.input(self.ui.tint(prompt))):
                    return 0
            except EOFError:
                self.ui.finish()
                return 0
            except KeyboardInterrupt:
                self.ui.finish()
                print('\n  Команда прервана. Состояние сервера не предполагается; проверьте /status.')
            except SystemExit as exc:
                self.ui.finish()
                if exc.code:
                    print(f'  Ошибка аргументов · код {exc.code}')
            except (ValueError, OSError, KeyError, IndexError) as exc:
                self.ui.finish()
                from scripts.i1.util import redact
                print('  ' + self.ui.tint(redact(str(exc), limit=500), '33'))


def main():
    return Console().run()


if __name__ == '__main__':
    raise SystemExit(main())
