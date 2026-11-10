import io
import pytest
from scripts.console import Console, Terminal, split_command
from scripts.i1 import cli


def test_windows_paths_are_literal():
    assert split_command('/connect --identity "C:\\Мои ключи\\key"') == [
        '/connect', '--identity', 'C:\\Мои ключи\\key']


def test_shell_exit_does_not_disconnect(monkeypatch):
    monkeypatch.setattr(cli, 'main', lambda *a, **k: pytest.fail('backend called'))
    assert Console().execute('/exit') is False


def test_help_does_not_call_backend(monkeypatch):
    monkeypatch.setattr(cli, 'main', lambda *a, **k: pytest.fail('backend called'))
    assert Console().execute('/help') is True


def test_no_connection_does_not_call_backend(monkeypatch):
    monkeypatch.setattr(cli, 'main', lambda *a, **k: pytest.fail('backend called'))
    with pytest.raises(ValueError, match='connect'):
        Console().execute('/resume')


def test_install_does_not_add_execute_or_yes(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'main', lambda argv, **kw: calls.append(argv) or 0)
    console = Console()
    console.name = 'test'
    console.execute('/install')
    assert calls[0] == ['install', '--name', 'test', '--profile', console.profile]
    console.execute('/install --execute')
    assert '--execute' in calls[1] and '--yes' not in calls[1]


def test_connect_guided_preserves_trust_confirmation(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'main', lambda argv, **kw: calls.append(argv) or 0)
    answers = iter(['test', 'example.test', '2222', '', '', ''])
    console = Console(input_fn=lambda _: next(answers))
    console.execute('/connect')
    assert console.name == 'test'
    assert '--accept-fingerprint' not in calls[0]
    assert '--replace' not in calls[0]
    assert calls[0][calls[0].index('--port')+1] == '2222'


def test_failed_connect_does_not_change_selection(monkeypatch):
    monkeypatch.setattr(cli, 'main', lambda *a, **kw: 2)
    console = Console()
    console.name = 'old'
    console.execute('/connect --name new --host example.test')
    assert console.name == 'old'


@pytest.mark.parametrize('line', ['/garbage', '/use ../bad', '/profile mystery'])
def test_bad_commands_are_rejected(line):
    with pytest.raises(ValueError):
        Console().execute(line)


def test_non_tty_progress_has_no_escape_and_keeps_unknown():
    stream = io.StringIO()
    ui = Terminal(stream, color=False)
    ui.progress('[DOWNLOADING] файл 1/7 model 25.0% 15 GiB/61 GiB rate unknown (ETA unknown)')
    ui.progress('[DOWNLOADING] CONNECTION_LOST; cached progress worker UNKNOWN')
    out = stream.getvalue()
    assert '░' in out and '25.0%' in out
    assert 'rate unknown' in out and 'CONNECTION_LOST' in out
    assert '\x1b' not in out and '\r' not in out


def test_tty_progress_replaces_line_and_finishes():
    class TTY(io.StringIO):
        def isatty(self):
            return True
    stream = TTY()
    ui = Terminal(stream, color=False)
    ui.progress('[DOWNLOADING] файл 0/7 model 10.0% 6 GiB/61 GiB 10 MiB/s ETA 1h')
    ui.progress('[DOWNLOADING] файл 0/7 model 20.0% 12 GiB/61 GiB 10 MiB/s ETA 1h')
    ui.finish()
    assert stream.getvalue().count('\r') == 2
    assert stream.getvalue().endswith('\n')


def test_progress_uses_existing_freshness_gate():
    emitted = []
    printer = cli.ProgressPrinter(lambda text, **kw: emitted.append(text))
    printer({'stage': 'DOWNLOADING',
             'observation': {'connection_state': 'CONNECTION_LOST'},
             'progress': {'bytes_done': 50, 'bytes_total': 100}})
    assert len(emitted) == 1 and 'CONNECTION_LOST' in emitted[0]
    assert '%' not in emitted[0]


def test_exit_and_eof_keep_shell_local(monkeypatch):
    monkeypatch.setattr(cli, 'main', lambda *a, **k: pytest.fail('backend called'))
    def end(_):
        raise EOFError
    assert Console(input_fn=end).run() == 0


def test_explicit_profile_and_name_not_overwritten(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'main', lambda argv, **kw: calls.append(argv) or 0)
    console = Console()
    console.name = 'old'
    console.execute('/status --name other --profile "C:\\Profiles folder\\profile.json" --offline')
    assert calls[0].count('--profile') == 1 and calls[0].count('--name') == 1
    assert calls[0][calls[0].index('--profile')+1] == 'C:\\Profiles folder\\profile.json'
