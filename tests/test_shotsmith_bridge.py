import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_bridge_commands_are_json_and_do_not_contact_gpu():
    payload='\n'.join(json.dumps({'command':c}) for c in ['/help','/profile qwen','/exit'])+'\n'
    result=subprocess.run([sys.executable,'-B','-m','scripts.console_bridge'],cwd=ROOT,
        input=payload.encode('utf-8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=dict(os.environ,PYTHONIOENCODING='utf-8'),timeout=15)
    assert result.returncode==0,result.stderr
    messages=[json.loads(line) for line in result.stdout.decode('utf-8').splitlines()]
    assert len([m for m in messages if m['type']=='done'])==3
    assert any('/connect' in m['text'] for m in messages)
    assert any('qwen' in m['text'] for m in messages if m['type']=='context')
    assert not result.stderr


def test_bridge_unknown_command_recovers():
    payload='\n'.join(json.dumps({'command':c}) for c in ['/not-real','/exit'])+'\n'
    result=subprocess.run([sys.executable,'-B','-m','scripts.console_bridge'],cwd=ROOT,
        input=payload.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15,
        env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    messages=[json.loads(line) for line in result.stdout.decode('utf-8').splitlines()]
    assert result.returncode==0
    assert len([m for m in messages if m['type']=='done'])==2
    assert any('Неизвестная' in m['text'] for m in messages)


def test_bridge_prompt_requires_explicit_reply(monkeypatch,capsys):
    import io
    from scripts import console_bridge
    replies='\n'.join(json.dumps(v) for v in [
        {'command':'/test'}, {'input':'n'}, {'command':'/exit'}])+'\n'
    monkeypatch.setattr(sys,'stdin',io.StringIO(replies))
    seen=[]
    def execute(self,line):
        if line=='/exit':return False
        seen.append(input('Доверять ключу? [y/N]'))
        return True
    monkeypatch.setattr(console_bridge.Console,'execute',execute)
    console_bridge.main()
    messages=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert seen==['n']
    assert any(m['type']=='log' and m['text']=='  ↳ n' for m in messages)
    assert any(m['type']=='prompt' and 'ключу' in m['text'] for m in messages)


def test_bridge_malformed_message_has_no_dispatch(monkeypatch,capsys):
    import io
    from scripts import console_bridge
    monkeypatch.setattr(sys,'stdin',io.StringIO('{"input":"wrong"}\n'))
    monkeypatch.setattr(console_bridge.Console,'execute',lambda *a: (_ for _ in ()).throw(AssertionError('called')))
    console_bridge.main()
    messages=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages[-1]['type']=='done'


def test_wizard_answer_history_and_defaults():
    from scripts.console_bridge import answer_text
    assert answer_text('Имя подключения: ', 'river-02') == '  ↳ river-02'
    assert answer_text('Рабочая папка сервера [/workspace]: ', '') == '  ↳ /workspace (по умолчанию)'
    assert answer_text('Адрес сервера: ', '') == '  ↳ (пустой ответ)'
    for label in ['Password:', 'Пароль:', 'API-ключ:', 'Passphrase:']:
        assert answer_text(label, 'do-not-display') == '  ↳ [скрыто]'
