from scripts.i1 import cli


def test_inspect_prints_facts_and_blockers(monkeypatch,capsys):
    monkeypatch.setattr(cli,'_dispatch',lambda args:(0,{
        'profile_id':'fixture','inspection':{'compatible':False,
        'facts':{'os':'Linux','arch':'x86_64','workspace':'/workspace'},
        'warnings':[],'blockers':[{'code':'fixture','message':'test blocker'}]}}))
    assert cli.main(['inspect','--name','fixture'])==0
    output=capsys.readouterr().out
    assert 'Linux' in output and '/workspace' in output and 'test blocker' in output
