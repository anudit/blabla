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
B,T,S=1,92,8
rng=np.random.RandomState(0); Lt=73
te=rng.randn(B,256,T).astype(np.float32); st=rng.randn(B,50,256).astype(np.float32)
tm=np.ones((B,1,T),np.float32); x0=rng.randn(B,144,Lt).astype(np.float32)
def fwd(x,lm):
    with torch.no_grad():
        return net(torch.from_numpy(x),torch.from_numpy(te),torch.from_numpy(st),torch.from_numpy(lm),
                   torch.from_numpy(tm),torch.zeros(B),torch.full((B,),float(S))).numpy()
ref=fwd(x0,np.ones((B,1,Lt),np.float32)); sd=ref.std()
print(f"reference L={Lt}, signal std {sd:.4f}\n")
print(f"  {'pad':>4} {'bucket':>7} {'mean abs':>10} {'rel %':>7} {'max abs':>10}")
for pad in [1,2,4,8,16,32]:
    Bk=Lt+pad
    xp=np.zeros((B,144,Bk),np.float32); xp[:,:,:Lt]=x0
    lm=np.zeros((B,1,Bk),np.float32); lm[:,:,:Lt]=1
    o=fwd(xp,lm)[:,:,:Lt]; d=np.abs(o-ref)
    print(f"  {pad:>4} {Bk:>7} {d.mean():>10.4e} {100*d.mean()/sd:>6.2f}% {d.max():>10.4e}")
