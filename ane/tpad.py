import warnings,functools
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
B,S=1,8; Lt=73; Tt=92; TB=128
rng=np.random.RandomState(0)
te=rng.randn(B,256,Tt).astype(np.float32); st=rng.randn(B,50,256).astype(np.float32)
z=rng.randn(B,144,Lt).astype(np.float32)
def fwd(x,te_,tm,lm,k):
    with torch.no_grad():
        return net(torch.from_numpy(x),torch.from_numpy(te_),torch.from_numpy(st),torch.from_numpy(lm),
                   torch.from_numpy(tm),torch.full((B,),float(k)),torch.full((B,),float(S))).numpy()
def ode(te_,tm,lm,x):
    for k in range(S): x=fwd(x,te_,tm,lm,k)
    return x
lm=np.ones((B,1,Lt),np.float32)
REF=ode(te,np.ones((B,1,Tt),np.float32),lm,z.copy())
tep=np.zeros((B,256,TB),np.float32); tep[:,:,:Tt]=te
tmp=np.zeros((B,1,TB),np.float32); tmp[:,:,:Tt]=1
PAD=ode(tep,tmp,lm,z.copy())
z2=rng.randn(B,144,Lt).astype(np.float32)
SEED=ode(te,np.ones((B,1,Tt),np.float32),lm,z2.copy())
def s(n,p,q):
    d=np.abs(p-q); c=float((p.ravel()@q.ravel())/(np.linalg.norm(p)*np.linalg.norm(q)))
    print(f"  {n:36s} mean|d| {d.mean():8.5f} rel {100*d.mean()/p.std():7.3f}%  cos {c:.7f}"); return d.mean()
print(f"text padding T={Tt} -> {TB}, latent exact L={Lt}, std {REF.std():.4f}\n")
a=s("text-padding delta",REF,PAD); b=s("different-seed baseline",REF,SEED)
print(f"\n  ratio {100*a/b:.2f}% of seed variation -> {'PASS' if a/b<0.05 else 'FAIL'}")
