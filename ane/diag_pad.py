import os,warnings,functools
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference; from onnx2torch import convert
SRC="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
def _p():
    def q(m):
        if isinstance(m,str): m=onnx.load(m)
        try: return shape_inference.infer_shapes(m)
        except Exception: return m
    onnx2torch.converter.safe_shape_inference=q
_p()
sl=onnxslim.slim(SRC)
for o in sl.opset_import:
    if o.domain in ("","ai.onnx"): o.version=17
net=convert(sl); net.eval()
for p in net.parameters(): p.requires_grad_(False)
B,T,STEPS=1,92,8
rng=np.random.RandomState(0)
Ltrue=73; BK=80
te=rng.randn(B,256,T).astype(np.float32); st=rng.randn(B,50,256).astype(np.float32)
tm=np.ones((B,1,T),np.float32); x0=rng.randn(B,144,Ltrue).astype(np.float32)
def fwd(x,lm):
    with torch.no_grad():
        return net(torch.from_numpy(x),torch.from_numpy(te),torch.from_numpy(st),torch.from_numpy(lm),
                   torch.from_numpy(tm),torch.zeros(B),torch.full((B,),float(STEPS))).numpy()
ref=fwd(x0,np.ones((B,1,Ltrue),np.float32))

def trial(name, fill):
    xp=np.zeros((B,144,BK),np.float32); xp[:,:,:Ltrue]=x0; fill(xp)
    lm=np.zeros((B,1,BK),np.float32); lm[:,:,:Ltrue]=1
    out=fwd(xp,lm)[:,:,:Ltrue]
    e=np.abs(out-ref).mean(axis=(0,1))
    print(f"\n  {name}: max {np.abs(out-ref).max():.4e} mean {np.abs(out-ref).mean():.4e}")
    print(f"    per-pos mean err, first 8: {np.array2string(e[:8],precision=4)}")
    print(f"    per-pos mean err, last 16: {np.array2string(e[-16:],precision=4)}")
    return e
e1=trial("zero pad",        lambda x: None)
e2=trial("edge-replicate",  lambda x: x.__setitem__((slice(None),slice(None),slice(Ltrue,None)), x[:,:,Ltrue-1:Ltrue]))

# also: does mask matter at all? all-ones mask on padded input
xp=np.zeros((B,144,BK),np.float32); xp[:,:,:Ltrue]=x0
out=fwd(xp,np.ones((B,1,BK),np.float32))[:,:,:Ltrue]
print(f"\n  zero pad + ALL-ONES mask: mean {np.abs(out-ref).mean():.4e}  (vs masked {np.abs(e1).mean():.4e})")
