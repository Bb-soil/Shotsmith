import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kf1.errors import AmbiguousDispatch
from scripts.kf1.png import write_png
from scripts.video import core, cli, media


@pytest.fixture
def job(tmp_path):
    a=tmp_path/'a.png';b=tmp_path/'b.png'
    a.write_bytes(write_png(16,8,bytes([200,100,50,255])*128))
    b.write_bytes(write_png(16,8,bytes([100,200,50,255])*128))
    root=tmp_path/'videos'
    created=core.create('scene-1',a,b,'Девушка садится.',root=root,seed=123)
    core.approve('scene-1',created['plan_sha256'],root=root)
    return root,created


class Remote:
    def __init__(self, *, ambiguous=False, bad_graph=False, bad_output=False, terminal=False):
        self.posts=0;self.uploads=0;self.ambiguous=ambiguous
        self.bad_graph=bad_graph;self.bad_output=bad_output;self.terminal=terminal
        self.graph=None;self.extra=None
    def queue_ids(self): return [],[]
    def upload_png(self,path,remote_name):
        self.uploads+=1
        return {'name':remote_name,'source_sha256':core.sha(path.read_bytes())}
    def object_info(self): return {'fixture':True}
    def submit(self,workflow,*,client_id,correlation_key,nonce):
        self.posts+=1;self.graph=workflow;self.extra={'correlation_key':correlation_key,'nonce':nonce}
        if self.ambiguous: raise AmbiguousDispatch('response lost')
        return SimpleNamespace(prompt_id='exact-prompt')
    def find_prompt_by_correlation(self,correlation_key,nonce):
        assert self.extra=={'correlation_key':correlation_key,'nonce':nonce}
        return 'exact-prompt'
    def history(self,prompt_id):
        graph=copy.deepcopy(self.graph)
        if self.bad_graph: graph['90']['inputs']['text']='tampered'
        prefix=self.graph['83']['inputs']['filename_prefix']
        directory,stem=prefix.rsplit('/',1)
        return {'prompt':[0,'exact-prompt',graph,self.extra],
                'status':{'status_str':'error' if self.terminal else 'success'},
                'outputs':{'83':{'images':[{'filename':stem+'_00001_.mp4',
                    'subfolder':'foreign' if self.bad_output else directory,'type':'output'}]}}}
    def _request(self,*args,**kwargs):return 200,{},b'0000ftypfake-mp4'


@pytest.fixture
def execution(monkeypatch):
    # This fixture tests durable transport ordering, not runtime schema/codec correctness.
    monkeypatch.setattr(core,'validate_bound_workflow',lambda *a,**kw: {'status':'fixture'})
    monkeypatch.setattr(media,'probe',lambda *a,**kw: {'width':816,'height':448,'fps':24,'duration_seconds':121/24,'decoded':True})


def test_one_post_and_repeat_zero_posts(job,execution):
    root,_=job;r=Remote()
    first=core.execute('scene-1',r,root=root,attestation={})
    second=core.execute('scene-1',r,root=root,attestation={})
    assert first==second and r.posts==1 and r.uploads==2
    assert first['status']=='COMPLETE'


def test_ambiguous_recovery_never_second_post(job,execution):
    root,_=job;r=Remote(ambiguous=True)
    with pytest.raises(AmbiguousDispatch):core.execute('scene-1',r,root=root,attestation={})
    assert core.status('scene-1',root=root)['status']=='AMBIGUOUS'
    assert core.execute('scene-1',r,root=root,attestation={})['status']=='COMPLETE'
    assert r.posts==1


def test_unknown_ambiguous_has_no_repost(job,execution):
    root,_=job;r=Remote(ambiguous=True)
    with pytest.raises(AmbiguousDispatch):core.execute('scene-1',r,root=root,attestation={})
    r.find_prompt_by_correlation=lambda *a:None
    with pytest.raises(ValueError,match='неизвестен'):core.execute('scene-1',r,root=root,attestation={})
    assert r.posts==1


@pytest.mark.parametrize('option',['bad_graph','bad_output'])
def test_foreign_history_is_not_collected(job,execution,option):
    root,_=job;r=Remote(**{option:True})
    with pytest.raises(ValueError):core.execute('scene-1',r,root=root,attestation={})
    assert not (root/'scene-1/video.mp4').exists()
    assert r.posts==1


def test_terminal_failure_cannot_repeat(job,execution):
    root,_=job;r=Remote(terminal=True)
    assert core.execute('scene-1',r,root=root,attestation={})['status']=='FAILED'
    with pytest.raises(ValueError):core.execute('scene-1',r,root=root,attestation={})
    assert r.posts==1


def test_input_tamper_blocks_before_upload(job,execution):
    root,_=job;(root/'scene-1/A.png').write_bytes(b'changed')
    r=Remote()
    with pytest.raises(ValueError):core.execute('scene-1',r,root=root,attestation={})
    assert r.uploads==r.posts==0


def test_exact_approval_required(job,execution):
    root,_=job;(root/'scene-1/approval.json').unlink()
    r=Remote()
    with pytest.raises(ValueError):core.execute('scene-1',r,root=root,attestation={})
    assert r.posts==r.uploads==0


def test_accept_binds_exact_video(job,execution):
    root,_=job;r=Remote();state=core.execute('scene-1',r,root=root,attestation={})
    with pytest.raises(ValueError):core.accept('scene-1','0'*64,root=root)
    core.accept('scene-1',state['sha256'],root=root)
    (root/'scene-1/video.mp4').write_bytes(b'tampered')
    with pytest.raises(ValueError):core.accept('scene-1',state['sha256'],root=root)


def test_busy_queue_does_not_submit(job,execution):
    root,_=job;r=Remote();r.queue_ids=lambda: (['foreign'],[])
    with pytest.raises(ValueError,match='занята'):core.execute('scene-1',r,root=root,attestation={})
    assert r.posts==r.uploads==0


def test_plan_geometry_and_no_mutation(job):
    root,created=job;folder,plan,_=core.load('scene-1',root)
    names={'A':'a.png','B':'b.png'}
    graph=core.graph_for(plan,names,'shotsmith/video/test')
    assert graph['81']['inputs']['width']==816 and graph['81']['inputs']['length']==121
    assert graph['90']['inputs']['text']=='Девушка садится.'
    assert graph['84']['inputs']['noise_seed']==123
    with pytest.raises(FileExistsError):core.create('scene-1',folder/'A.png',folder/'B.png','new',root=root)


@pytest.mark.parametrize('identifier',['../escape','A','x/y','x'*65])
def test_id_containment(identifier,tmp_path):
    with pytest.raises(ValueError):core.directory(identifier,tmp_path)


def test_seed_policy_keeps_legacy_and_separates_frames():
    from scripts.kf1.runner import Runner
    from scripts.kf1.plan import resolve_plan
    plan=resolve_plan(core.ROOT/'config/keyframes/two-scenes-v2.json')
    runner=Runner(plan,None,None,plan_root=core.ROOT)
    base=runner._frame_seed(plan.frames[1])
    assert runner._frame_seed(plan.frames[2])==base
    plan.data['seed_policy']='per_frame'
    values=[runner._frame_seed(frame) for frame in plan.frames]
    assert len(set(values))==len(values)
    assert all(0<=v<2147483648 for v in values)


def test_console_video_plan_only_no_network(monkeypatch,job,capsys):
    from scripts.console import Console
    root,_=job
    monkeypatch.setattr(cli,'open_kf1_connection',lambda **kw:pytest.fail('network'))
    code=cli.main(['--root',str(root),'generate','scene-1'])
    assert code==0 and 'PLAN_ONLY' in capsys.readouterr().out
    assert Console().execute('/help') is True


def test_resume_without_intent_cannot_submit(monkeypatch,job):
    root,_=job
    monkeypatch.setattr(cli,'open_kf1_connection',lambda **kw:pytest.fail('network'))
    assert cli.main(['--root',str(root),'resume','scene-1','--execute','--base-url','http://127.0.0.1:1'])==2


def test_actual_ffmpeg_assemble_requires_acceptance(tmp_path):
    try:executable=media.ffmpeg()
    except ValueError:pytest.skip('local ffmpeg not installed')
    source=tmp_path/'source.mp4'
    media.command([executable,'-v','error','-f','lavfi','-i','color=c=blue:s=816x448:r=24',
                   '-t','0.25','-c:v','libx264','-pix_fmt','yuv420p',str(source)])
    facts=media.probe(source)
    assert facts['decoded'] and facts['width']==816 and facts['fps']==24
    root=tmp_path/'jobs'
    a=tmp_path/'a.png';a.write_bytes(write_png(16,8,bytes([1,2,3,255])*128))
    for job in ('one','two'):
        core.create(job,a,a,'test',root=root)
        folder,_,digest=core.load(job,root)
        video=source.read_bytes();(folder/'video.mp4').write_bytes(video)
        core._atomic_json(folder/'state.json',{'status':'COMPLETE','sha256':core.sha(video),'plan_sha256':digest})
    output=tmp_path/'assembled.mp4'
    with pytest.raises(FileNotFoundError):core.assemble(['one','two'],output,root=root)
    for job in ('one','two'):core.accept(job,core.sha(source.read_bytes()),root=root)
    result=core.assemble(['two','one'],output,root=root)
    assert result['media']['duration_seconds']==pytest.approx(.5,abs=.02)
    assert [s['id'] for s in result['scenes']]==['two','one']
    with pytest.raises(ValueError):core.assemble(['one'],output,root=root)


def test_http_contract_real_codec_and_exact_schema(job,tmp_path):
    """Mock HTTP transport, finite pinned node schema, real FFmpeg MP4 decode."""
    import httpx
    import re
    from scripts.kf1.comfy import ComfyClient
    from tests.kf1_object_info import build_object_info
    root,_=job
    try: executable=media.ffmpeg()
    except ValueError: pytest.skip('local ffmpeg unavailable')
    fixture=tmp_path/'fixture.mp4'
    media.command([executable,'-v','error','-f','lavfi','-i','color=c=green:s=816x448:r=24',
                   '-frames:v','121','-c:v','libx264','-pix_fmt','yuv420p',str(fixture)])
    state={'posts':0}
    def transport(request):
        route=request.url.path
        if route=='/queue':return httpx.Response(200,json={'queue_running':[],'queue_pending':[]})
        if route=='/object_info':return httpx.Response(200,json=build_object_info(include_wan=True))
        if route=='/upload/image':
            name=re.search(br'filename="([^"]+)"',request.content)[1].decode()
            return httpx.Response(200,json={'name':name,'subfolder':'','type':'input'})
        if route=='/prompt':
            state['posts']+=1;state['request']=json.loads(request.content)
            return httpx.Response(200,json={'prompt_id':'real-contract','node_errors':{}})
        if route=='/history/real-contract':
            sent=state['request'];prefix=sent['prompt']['83']['inputs']['filename_prefix'];directory,stem=prefix.rsplit('/',1)
            return httpx.Response(200,json={'real-contract':{'prompt':[0,'real-contract',sent['prompt'],sent['extra_data']],
                'status':{'status_str':'success'},'outputs':{'83':{'images':[{'filename':stem+'_00001_.mp4','subfolder':directory,'type':'output'}]}}}})
        if route=='/view':return httpx.Response(200,content=fixture.read_bytes())
        pytest.fail(route)
    http=httpx.Client(base_url='http://127.0.0.1:8189',transport=httpx.MockTransport(transport))
    client=ComfyClient('http://127.0.0.1:8189',client=http)
    result=core.execute('scene-1',client,root=root,attestation={'mode':'offline_http_fixture'})
    assert result['status']=='COMPLETE' and result['media']['decoded']
    assert core.execute('scene-1',client,root=root,attestation={})==result
    assert state['posts']==1
    http.close()


def test_single_instance_blocks_duplicate_submit(job,execution):
    from anime_studio.lock import InstanceLock
    from anime_studio.errors import InstanceAlreadyRunning
    root,_=job;r=Remote()
    with InstanceLock(root/'scene-1/.run.lock'):
        with pytest.raises(InstanceAlreadyRunning):core.execute('scene-1',r,root=root,attestation={})
    assert r.posts==0


def test_clone_is_new_unapproved_version(job):
    root,_=job
    assert cli.main(['--root',str(root),'clone','scene-1','--new-id','scene-2','--seed','9'])==0
    folder,plan,_=core.load('scene-2',root)
    assert plan['seed']==9 and not (folder/'approval.json').exists()
    assert core.status('scene-1',root=root)['status']=='PLANNED'


def test_storyboard_uses_verified_pair_and_rejects_escape(job,tmp_path):
    root,_=job
    folder,plan,_=core.load('scene-1',root)
    roles=[{'role':role,'status':'COMPLETE','path':role+'.png','sha256':plan['inputs'][role]['normalized_sha256']} for role in ('A','B')]
    manifest={'schema':'anime-studio/kf1-storyboard/v1','scenes':[{'scene_id':'shot-01','roles':roles}]}
    core._atomic_json(folder/'storyboard.json',manifest)
    args=SimpleNamespace(storyboard=folder/'storyboard.json',scene='shot-01',id='from-story',prompt='Течёт река',root=root)
    assert cli.from_storyboard(args)['status']=='PLAN_ONLY'
    manifest['scenes'][0]['roles'][0]['path']='../../escape.png'
    core._atomic_json(folder/'storyboard.json',manifest)
    with pytest.raises(ValueError):cli.from_storyboard(args)
