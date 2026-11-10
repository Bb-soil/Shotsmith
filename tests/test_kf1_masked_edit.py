import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.kf1 import masked_edit as m, png
from scripts.kf1.errors import KF1Error


def write_image(path, w=64, h=64, color=(40, 90, 130, 255)):
    data = png.write_png(w, h, bytes(color) * (w*h))
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def spec(tmp_path):
    data = {"schema": m.SCHEMA, "source": write_image(tmp_path/'source.png'),
            "pose": write_image(tmp_path/'pose.png', color=(220, 30, 40, 255)),
            "edit_regions": [[8, 8, 48, 48]], "feather": 2,
            "pose_polygon": [[8, 8], [40, 8], [40, 40], [8, 40]],
            "pose_placement": [16, 16, 24, 24], "flip_pose": True,
            "prompt": "Keep the seated pose and fill gray with consistent background."}
    p=tmp_path/'spec.json'
    p.write_text(json.dumps(data), encoding='utf-8')
    return p, data


def test_prepare_guide_and_graph_protect_background(spec, tmp_path):
    path, data = spec
    before={k: Path(data[k]['path']).read_bytes() for k in ('source','pose')}
    result=m.prepare(path,tmp_path/'bundle')
    manifest=json.loads((tmp_path/'bundle/manifest.json').read_text())
    graph=json.loads((tmp_path/'bundle/workflow.api.json').read_text())
    original=png.read_png(before['source'])
    guide=png.read_png(Path(result['guide']).read_bytes())
    mask=png.read_png(Path(result['mask']).read_bytes())
    assert guide.rgba != original.rgba
    assert any(mask.rgba[i] == 255 for i in range(0,len(mask.rgba),4))
    assert any(0 < mask.rgba[i] < 255 for i in range(0,len(mask.rgba),4))
    for i in range(0,len(mask.rgba),4):
        if mask.rgba[i] == 0: assert guide.rgba[i:i+4] == original.rgba[i:i+4]
    assert graph['30']['inputs']['image1'] == ['4',0]
    assert graph['31']['inputs']['image1'] == ['4',0]
    assert graph['40']['inputs']['pixels'] == ['4',0]
    assert graph['50']['inputs']['latent_image'] == ['41',0]
    assert graph['41']['inputs']['mask'] == ['6',0]
    assert graph['6']['inputs'] == {'image':['5',0], 'channel':'red'}
    assert graph['65']['inputs'] == {'destination':['1',0],'source':['60',0],'mask':['6',0], 'x':0,'y':0,'resize_source':False}
    assert graph['70']['inputs']['images'] == ['65',0]
    # Original image must only be consumed by final background restore.
    uses=[nid for nid,node in graph.items() for value in node['inputs'].values() if value == ['1',0]]
    assert uses == ['65']
    assert manifest['gpu_verified'] is False and manifest['automatic_submit'] is False
    assert manifest['protected_pixels'] > 0
    for key in before: assert Path(data[key]['path']).read_bytes() == before[key]
    assert m.verify(tmp_path/'bundle',result['guide'])['status'] == 'PASS'


def test_protected_pixel_change_is_detected(spec,tmp_path):
    path,_=spec
    result=m.prepare(path,tmp_path/'bundle')
    image=png.read_png(Path(result['guide']).read_bytes())
    changed=bytearray(image.rgba); changed[0] ^= 255
    p=tmp_path/'changed.png'; p.write_bytes(png.write_png(image.width,image.height,bytes(changed)))
    report=m.verify(tmp_path/'bundle',p)
    assert report['status']=='FAIL' and report['changed_protected_pixels']==1
    assert report['pose_quality']=='NOT_VERIFIED'


@pytest.mark.parametrize('mutation', ['hash','out_of_bounds','empty_mask','full_mask','pose_outside','polygon','flip','steps'])
def test_invalid_specs_create_no_bundle(spec,tmp_path,mutation):
    path,data=spec
    if mutation=='hash': data['source']['sha256']='0'*64
    elif mutation=='out_of_bounds': data['edit_regions']=[[63,0,10,10]]
    elif mutation=='empty_mask': data['edit_regions']=[[0,0,1,1]]
    elif mutation=='full_mask': data['edit_regions']=[[0,0,64,64]]; data['feather']=0
    elif mutation=='pose_outside': data['pose_placement']=[0,0,24,24]
    elif mutation=='polygon': data['pose_polygon']=[[0,0],[64,0],[10,10]]
    elif mutation=='flip': data['flip_pose']='yes'
    elif mutation=='steps': data['settings']={'steps':2.5}
    path.write_text(json.dumps(data))
    with pytest.raises(KF1Error): m.prepare(path,tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


def test_prepare_refuses_overwrite_and_verifier_detects_tampering(spec,tmp_path):
    path,_=spec
    result=m.prepare(path,tmp_path/'bundle')
    with pytest.raises(KF1Error,match='существует'): m.prepare(path,tmp_path/'bundle')
    Path(result['mask']).write_bytes(b'broken')
    with pytest.raises(KF1Error,match='изменён'): m.verify(tmp_path/'bundle',result['guide'])


def test_flip_preserves_pose_and_changes_direction():
    buf=bytearray(bytes((0,0,0,255))*64)
    for y in range(2,6):
        for x in range(2,6): buf[(y*8+x)*4]=x*30
    img=png.PNGImage(8,8,6,8,0,bytes(buf))
    points=[[2,2],[6,2],[6,6],[2,6]]
    normal=m.polygon_cutout(img,points,False); flipped=m.polygon_cutout(img,points,True)
    for y in range(normal.height):
        for x in range(normal.width):
            a=(y*normal.width+x)*4; b=(y*normal.width+(normal.width-1-x))*4
            assert normal.rgba[a:a+4] == flipped.rgba[b:b+4]


def test_mask_orientation_and_exact_composite():
    mask=m.rectangle_mask(8,8,[[2,2,4,4]],0)
    assert mask[0]==0 and mask[3*8+3]==255
    original=bytes((10,20,30,255))*64
    raw=bytes((200,210,220,255))*64
    out=m.composite_rgba(original,raw,mask)
    assert out[:4] == original[:4]
    assert out[(3*8+3)*4:(3*8+3)*4+4] == raw[(3*8+3)*4:(3*8+3)*4+4]


def test_prepare_is_deterministic_without_optional_pillow(spec,tmp_path):
    path,_=spec
    m.prepare(path,tmp_path/'one'); m.prepare(path,tmp_path/'two')
    assert (tmp_path/'one/manifest.json').read_bytes() == (tmp_path/'two/manifest.json').read_bytes()
    for f in (tmp_path/'one').iterdir(): assert f.read_bytes() == (tmp_path/'two'/f.name).read_bytes()


def test_mask_does_not_feather_at_canvas_border():
    mask=m.rectangle_mask(16,16,[[0,0,8,16]],2)
    assert mask[0] == 255 and mask[15*16] == 255
    assert mask[7] == 0 and mask[8] == 0


def test_generic_graph_has_no_subject_specific_pose():
    graph=m.build_graph('user edit', 'test')
    assert 'standing' not in graph['31']['inputs']['prompt']


def test_cli_offline_prepare_and_verify(spec,tmp_path,capsys):
    from scripts.kf1.cli import main
    path,_=spec
    assert main(['masked-prepare','--spec',str(path),'--output',str(tmp_path/'cli'),'--json'])==0
    payload=json.loads(capsys.readouterr().out)
    assert payload['gpu_verified'] is False
    assert main(['masked-verify','--package',str(tmp_path/'cli'),'--image',payload['guide'],'--json'])==0
    assert json.loads(capsys.readouterr().out)['status']=='PASS'


def test_malformed_spec_is_clean_error(tmp_path):
    p=tmp_path/'bad.json'; p.write_text('[]')
    with pytest.raises(KF1Error,match='JSON-объектом'): m.prepare(p,tmp_path/'out')
    assert not (tmp_path/'out').exists()
