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
B,T,S=1,92,8; Lt=73; BK=80
rng=np.random.RandomState(0)
te=rng.randn(B,256,T).astype(np.float32); st=rng.randn(B,50,256).astype(np.float32)
tm=np.ones((B,1,T),np.float32)
def fwd(x,lm,k):
    with torch.no_grad():
        return net(torch.from_numpy(x),torch.from_numpy(te),torch.from_numpy(st),torch.from_numpy(lm),
                   torch.from_numpy(tm),torch.full((B,),float(k)),torch.full((B,),float(S))).numpy()
def euler(x,lm):
    for k in range(S): x = x + fwd(x,lm,k)/S
    return x
# A: exact L, seed 1     B: bucketed L=80 masked, same seed 1     C: exact L, seed 2
z1=np.random.RandomState(1).randn(B,144,Lt).astype(np.float32)
z2=np.random.RandomState(2).randn(B,144,Lt).astype(np.float32)
A=euler(z1.copy(), np.ones((B,1,Lt),np.float32))
zp=np.zeros((B,144,BK),np.float32); zp[:,:,:Lt]=z1
lmp=np.zeros((B,1,BK),np.float32); lmp[:,:,:Lt]=1
Bb=euler(zp.copy(), lmp)[:,:,:Lt]
C=euler(z2.copy(), np.ones((B,1,Lt),np.float32))
def cmp(n,p,q):
    d=np.abs(p-q); cos=float((p.ravel()@q.ravel())/(np.linalg.norm(p)*np.linalg.norm(q)))
    print(f"  {n:38s} mean|d| {d.mean():8.4f}  rel {100*d.mean()/p.std():6.2f}%  cos {cos:.6f}")
print(f"final latent std {A.std():.4f}\n")
cmp("bucketing delta (same seed)", A, Bb)
cmp("different-noise-seed delta  (baseline)", A, C)
r=np.abs(A-Bb).mean()/np.abs(A-C).mean()
print(f"\n  bucketing delta is {100*r:.1f}% of the run-to-run seed variation")
print(f"  -> {'PASS: bucketing is far below inherent stochastic variation' if r<0.25 else 'MARGINAL/FAIL'}")
