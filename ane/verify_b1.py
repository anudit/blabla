import os,warnings,functools
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference; from onnx2torch import convert
import coremltools as ct
SRC="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
OUT="/Users/anudit/Documents/GitHub/tts-metal/ane/models2"
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
ORDER=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
B,T,STEPS=1,92,8
rng=np.random.RandomState(0)
Ltrue=73                     # deliberately not a bucket boundary
te=rng.randn(B,256,T).astype(np.float32); st=rng.randn(B,50,256).astype(np.float32)
tm=np.ones((B,1,T),np.float32)
x0=rng.randn(B,144,Ltrue).astype(np.float32)

def torch_fwd(x,lm,cs):
    L=x.shape[2]
    with torch.no_grad():
        return net(torch.from_numpy(x),torch.from_numpy(te),torch.from_numpy(st),
                   torch.from_numpy(lm),torch.from_numpy(tm),
                   torch.full((B,),float(cs)),torch.full((B,),float(STEPS))).numpy()

# ---------- TEST 1: mask-padding equivalence (torch fp32, isolates the model not fp16) ----------
lm_t=np.ones((B,1,Ltrue),np.float32)
ref=torch_fwd(x0,lm_t,0)
BK=80
xp=np.zeros((B,144,BK),np.float32); xp[:,:,:Ltrue]=x0
lm_p=np.zeros((B,1,BK),np.float32); lm_p[:,:,:Ltrue]=1
pad=torch_fwd(xp,lm_p,0)[:,:,:Ltrue]
d=np.abs(ref-pad)
print(f"TEST 1  mask-padding equivalence (L={Ltrue} -> bucket {BK}), torch fp32")
print(f"  max abs diff {d.max():.4e}   mean {d.mean():.4e}   (signal std {ref.std():.4f})")
print(f"  -> {'PASS' if d.max() < 1e-3 else 'FAIL — padding changes the result'}")

# ---------- TEST 2: fp16 ANE drift over 8 chained CFG-Euler steps, B=1, bucketed ----------
ml=ct.models.MLModel(f"{OUT}/plain_L{BK}.mlpackage",compute_units=ct.ComputeUnit.CPU_AND_NE)
def ane_fwd(x,lm,cs):
    return ml.predict({"noisy_latent":x,"text_emb":te,"style_ttl":st,"latent_mask":lm,"text_mask":tm,
                       "current_step":np.full(B,float(cs),np.float32),
                       "total_step":np.full(B,float(STEPS),np.float32)})["denoised_latent"]
def euler(fwd,x,lm):
    x=x.copy()
    for k in range(STEPS): x = x + fwd(x,lm,k)/STEPS
    return x
a=euler(torch_fwd,xp,lm_p)[:,:,:Ltrue]
b=euler(ane_fwd,  xp,lm_p)[:,:,:Ltrue]
d=np.abs(a-b)
cos=float((a.ravel()@b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)))
print(f"\nTEST 2  ANE fp16 vs torch fp32, {STEPS} chained Euler steps, B=1 bucket {BK}")
print(f"  torch std {a.std():.5f}   ANE std {b.std():.5f}")
print(f"  max abs {d.max():.4e}  mean {d.mean():.4e}  rel-to-std {d.mean()/a.std():.4e}")
print(f"  cosine  {cos:.8f}")
print(f"  -> {'PASS' if cos > 0.9999 else 'FAIL'}")
