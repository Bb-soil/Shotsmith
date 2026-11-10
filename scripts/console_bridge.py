"""JSON-lines bridge for the Terminal.Gui frontend; no operations on startup."""
import builtins
import contextlib
import io
import json
import re
import sys

from scripts.console import Console, Terminal, clean


def answer_text(prompt, answer):
    """Keep non-secret wizard answers visible after the input field is cleared."""
    from scripts.i1.util import redact
    sensitive = re.search(r'password|passphrase|парол|секрет|secret|token|токен|api.?key|api.?ключ', prompt, re.I)
    if sensitive:
        return '  ↳ [скрыто]'
    if not answer:
        default = re.search(r'\[([^\[\]]+)\]\s*:\s*$', prompt)
        return '  ↳ ' + (clean(default.group(1)) + ' (по умолчанию)' if default else '(пустой ответ)')
    return '  ↳ ' + redact(clean(answer), limit=2000)


def main():
    incoming, outgoing = sys.stdin, sys.stdout

    def emit(kind, text='', **fields):
        outgoing.write(json.dumps({'type': kind, 'text': text, **fields}, ensure_ascii=False)+'\n')
        outgoing.flush()

    def read():
        raw = incoming.readline()
        if not raw:
            raise EOFError
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('Expected object')
        return value

    def ask(prompt=''):
        emit('prompt', clean(prompt))
        value = read()
        if set(value) != {'input'} or not isinstance(value['input'], str):
            raise ValueError('Expected input response')
        emit('log', answer_text(prompt, value['input']))
        return value['input']

    class Input(io.TextIOBase):
        def isatty(self):
            return True  # Actual confirmations are routed through ask(), never auto-yes.
        def readline(self, *args):
            return incoming.readline(*args)

    class Output(io.TextIOBase):
        pending = ''
        def write(self, text):
            self.pending += text
            while '\n' in self.pending:
                line,self.pending=self.pending.split('\n',1)
                if line:emit('log',clean(line))
            return len(text)
        def flush(self):
            if self.pending:
                emit('log',clean(self.pending));self.pending=''
        def isatty(self):
            return False

    class UI(Terminal):
        def __init__(self):
            self.color=False;self.active=False
        def progress(self,line,**kwargs):
            line=clean(line)
            match=re.search(r'\s(\d+(?:\.\d+)?)%\s',line)
            emit('progress',line,ratio=min(1,max(0,float(match.group(1))/100)) if match else None)
        def finish(self):
            pass

    console=Console(terminal=UI(),input_fn=ask)
    output=Output()
    original_input=builtins.input
    try:
        builtins.input=ask
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            sys.stdin=Input()
            while True:
                try:
                    message=read()
                except EOFError:
                    break
                try:
                    if set(message)!={'command'} or not isinstance(message['command'],str):
                        raise ValueError('Expected command')
                    keep=console.execute(message['command'])
                    output.flush()
                    emit('context', f'{console.name or "локально"}  ·  {console.profile.split("/")[-1]}')
                    emit('done','Готово · следующая команда')
                    if not keep:break
                except (EOFError,KeyboardInterrupt):
                    break
                except SystemExit as exc:
                    output.flush();emit('done',f'Команда завершена · код {exc.code or 0}')
                except Exception as exc:
                    from scripts.i1.util import redact
                    output.flush();emit('log',redact(str(exc),limit=1000));emit('done','Ошибка · подробности выше')
    finally:
        builtins.input=original_input
        sys.stdin=incoming


if __name__=='__main__':
    main()
