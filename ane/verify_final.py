import warnings,functools
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference; from onnx2torch import convert
import coremltools as ct
SRC="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
MULTI="/Users/anudit/Documents/GitHub/tts-metal/ane/models3/ve_multi.mlpackage"
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
B,T,S=1,92,8; Lt=73; BK=96
rng=np.random.RandomState(0)
te=rng.randn(B,256,T).astype(np.float32); st=rng.randn(B,50,256).astype(np.float32)
tm=np.ones((B,1,T),np.float32)
def tfwd(x,lm,k):
    with torch.no_grad():
        return net(torch.from_numpy(x),torch.from_numpy(te),torch.from_numpy(st),torch.from_numpy(lm),
                   torch.from_numpy(tm),torch.full((B,),float(k)),torch.full((B,),float(S))).numpy()
ml=ct.models.MLModel(MULTI,compute_units=ct.ComputeUnit.CPU_AND_NE,function_name=f"L{BK}")
def afwd(x,lm,k):
    return ml.predict({"noisy_latent":x,"text_emb":te,"style_ttl":st,"latent_mask":lm,"text_mask":tm,
                       "current_step":np.full(B,float(k),np.float32),
                       "total_step":np.full(B,float(S),np.float32)})["denoised_latent"]
def ode(fwd,x,lm):                      # CORRECT recurrence: model returns x_{k+1}
    for k in range(S): x = fwd(x,lm,k)
    return x
def stats(n,p,q):
    d=np.abs(p-q); cos=float((p.ravel()@q.ravel())/(np.linalg.norm(p)*np.linalg.norm(q)))
    print(f"  {n:44s} mean|d| {d.mean():9.5f}  rel {100*d.mean()/p.std():7.3f}%  cos {cos:.7f}")
    return d.mean()

z1=np.random.RandomState(1).randn(B,144,Lt).astype(np.float32)
z2=np.random.RandomState(2).randn(B,144,Lt).astype(np.float32)
lm_t=np.ones((B,1,Lt),np.float32)
zp=np.zeros((B,144,BK),np.float32); zp[:,:,:Lt]=z1
lmp=np.zeros((B,1,BK),np.float32); lmp[:,:,:Lt]=1

REF  = ode(tfwd, z1.copy(), lm_t)                 # torch fp32, exact length
PADT = ode(tfwd, zp.copy(), lmp)[:,:,:Lt]         # torch fp32, bucketed
PADA = ode(afwd, zp.copy(), lmp)[:,:,:Lt]         # ANE  fp16, bucketed
SEED = ode(tfwd, z2.copy(), lm_t)                 # torch fp32, different seed

print(f"final latent std {REF.std():.4f}   (8 steps, model does CFG+Euler internally)\n")
a=stats("A  bucketing only    (fp32, L73->96)", REF, PADT)
b=stats("B  bucketing + ANE fp16  (end-to-end)", REF, PADA)
c=stats("C  ANE fp16 only     (both bucketed)", PADT, PADA)
d=stats("D  different noise seed  (BASELINE)", REF, SEED)
print(f"\n  A/D = {100*a/d:.2f}%   B/D = {100*b/d:.2f}%   C/D = {100*c/d:.2f}%")
print(f"  -> {'PASS' if b/d < 0.05 else 'FAIL'}: total ANE+bucketing deviation is {100*b/d:.2f}% of inherent seed variation")
