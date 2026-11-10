"""Execute pinned stock mask math on CPU using a minimal numpy tensor adapter.

This is NOT a ComfyUI import, sampler execution, or GPU test.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.kf1.masked_edit import build_graph

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/masked-edit-evidence-20261107/sources'


def tree(name, expected):
    data=(SOURCE/name).read_bytes()
    assert hashlib.sha256(data).hexdigest()==expected
    return ast.parse(data)


def test_new_stock_node_schema_names_match_pinned_source():
    t=tree('nodes_mask.py','08171f430ddd9a327cec92293ac25fd1bc39d5c9079187a51463fb42d2a9d65b')
    graph=build_graph('pose','test')
    for cls_name,nid in [('ImageCompositeMasked','65'),('ImageToMask','6')]:
        cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls_name)
        schema=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='define_schema')
        names={n.args[0].value for n in ast.walk(schema) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='Input'
               and n.args and isinstance(n.args[0],ast.Constant)}
        assert set(graph[nid]['inputs']) == names
    t=tree('nodes.py','4cfd9be3754f9dc25213febdf96b690d6e32007c2e539d37e799d82226d758c5')
    cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='SetLatentNoiseMask')
    declaration=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='INPUT_TYPES')
    raw=ast.literal_eval(declaration.body[0].value)
    assert set(raw['required']) == set(graph['41']['inputs'])


def test_actual_stock_composite_keeps_pixels_outside_mask():
    np=pytest.importorskip('numpy')
    t=tree('nodes_mask.py','08171f430ddd9a327cec92293ac25fd1bc39d5c9079187a51463fb42d2a9d65b')
    composite=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='composite')
    cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='ImageCompositeMasked')
    execute=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    execute.decorator_list=[]

    class Tensor(np.ndarray):
        @property
        def device(self): return 'cpu'
        def to(self,*args,copy=False,**kwargs): return self.copy() if copy else self
        def clone(self): return self.copy()
        def movedim(self,a,b): return np.moveaxis(self,a,b)
        def unsqueeze(self,n): return np.expand_dims(self,n)
    def tensor(a): return np.asarray(a,dtype=np.float32).view(Tensor)
    def interpolate(value,*,size,**kwargs):
        assert value.shape[-2:] == tuple(size)  # graph uses matched geometry and resize_source=False
        return value
    namespace={'torch':NS(nn=NS(functional=NS(interpolate=interpolate)),ones_like=np.ones_like),
               'comfy':NS(utils=NS(repeat_to_batch_size=lambda x,n:x if x.shape[0]==n else pytest.fail('unexpected batch'))),
               'node_helpers':NS(image_alpha_fix=lambda a,b:(a,b)), 'IO':NS(NodeOutput=lambda *values:values)}
    exec(compile(ast.Module(body=[composite,execute],type_ignores=[]),str(SOURCE/'nodes_mask.py'),'exec'),namespace)
    rng=np.random.default_rng(42)
    src_u8=rng.integers(0,256,(1,8,12,3),dtype=np.uint8)
    original=tensor(src_u8/255)
    generated=tensor(rng.random((1,8,12,3)))
    mask=tensor(np.zeros((1,8,12)));mask[:,2:6,3:9]=1;mask[:,2,3:9]=0.5
    out=namespace['execute'](None,original,generated,0,0,False,mask)[0]
    # Test actual source formula, including a soft seam and original nonmutation.
    expected=generated*mask[...,None]+original*(1-mask[...,None])
    np.testing.assert_allclose(out,expected,rtol=0,atol=0)
    q=np.clip(out*255,0,255).astype(np.uint8)
    np.testing.assert_array_equal(q[mask==0],src_u8[mask==0])
    np.testing.assert_array_equal(original,tensor(src_u8/255))
