import os,warnings,functools,sys
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference; from onnx2torch import convert
import coremltools as ct
ONNX_DIR="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic"
B,T,L,STEPS=2,92,80,8
def _p():
    def q(m):
        if isinstance(m,str): m=onnx.load(m)
        try: return shape_inference.infer_shapes(m)
        except Exception: return m
    onnx2torch.converter.safe_shape_inference=q
_p()
sl=onnxslim.slim(os.path.join(ONNX_DIR,"vector_estimator.onnx"))
for o in sl.opset_import:
    if o.domain in ("","ai.onnx"): o.version=17
net=convert(sl); net.eval()
for p in net.parameters(): p.requires_grad_(False)

rng=np.random.RandomState(0)
x0=rng.randn(B,144,L).astype(np.float32)
te=rng.randn(B,256,T).astype(np.float32); st=rng.randn(B,50,256).astype(np.float32)
lm=np.ones((B,1,L),np.float32); tm=np.ones((B,1,T),np.float32)
ml=ct.models.MLModel(f"models/ve_L{L}.mlpackage",compute_units=ct.ComputeUnit.CPU_AND_NE)

def euler(fwd):
    x=x0.copy()
    for k in range(STEPS):
        v=fwd(x,np.full(B,float(k),np.float32),np.full(B,float(STEPS),np.float32))
        vc,vu=v[0:1],v[1:2]
        vv=np.concatenate([4.0*vc-3.0*vu]*2,0)
        x=x+vv/STEPS
    return x

def f_torch(x,cs,ts_):
    with torch.no_grad():
        return net(*[torch.from_numpy(a) for a in (x,te,st,lm,tm,cs,ts_)]).numpy()
def f_ane(x,cs,ts_):
    return ml.predict({"noisy_latent":x,"text_emb":te,"style_ttl":st,"latent_mask":lm,
                       "text_mask":tm,"current_step":cs,"total_step":ts_})["denoised_latent"]

a=euler(f_torch); b=euler(f_ane)
d=np.abs(a-b)
print(f"\nAfter {STEPS} chained CFG-Euler steps (L={L}):")
print(f"  torch fp32 latent: mean {a.mean():+.5f}  std {a.std():.5f}")
print(f"  ANE   fp16 latent: mean {b.mean():+.5f}  std {b.std():.5f}")
print(f"  abs diff: max {d.max():.5e}  mean {d.mean():.5e}")
print(f"  relative to signal std: mean {d.mean()/a.std():.5e}   max {d.max()/a.std():.5e}")
print(f"  cosine similarity: {float((a.ravel()@b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b))):.8f}")
