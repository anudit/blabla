import os,sys,time,json,warnings,functools
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference
from onnx2torch import convert
import coremltools as ct
sys.path.insert(0,os.path.dirname(__file__))
ONNX_DIR="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic"
OUT="/Users/anudit/Documents/GitHub/tts-metal/ane/models"; os.makedirs(OUT,exist_ok=True)
B,T=2,92
def _p():
    def patched(m):
        if isinstance(m,str): m=onnx.load(m)
        try: return shape_inference.infer_shapes(m)
        except Exception: return m
    onnx2torch.converter.safe_shape_inference=patched
_p()
sl=onnxslim.slim(os.path.join(ONNX_DIR,"vector_estimator.onnx"))
for o in sl.opset_import:
    if o.domain in ("","ai.onnx"): o.version=17
net=convert(sl); net.eval()
for p in net.parameters(): p.requires_grad_(False)
class W(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,a,b,c,d,e,f,g): return s.m(a,b,c,d,e,f,g)
net=W(net)
ORDER=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
def ex(L): return [torch.randn(B,144,L),torch.randn(B,256,T),torch.randn(B,50,256),
                   torch.ones(B,1,L),torch.ones(B,1,T),torch.zeros(B),torch.full((B,),8.0)]
BUCKETS=[int(x) for x in sys.argv[1].split(",")]
res={}
for L in BUCKETS:
    p=f"{OUT}/ve_L{L}.mlpackage"
    if not os.path.exists(p):
        e=ex(L)
        with torch.no_grad():
            ts=torch.jit.freeze(torch.jit.trace(net,tuple(e),check_trace=False).eval())
        m=ct.convert(ts,inputs=[ct.TensorType(name=n,shape=tuple(t.shape),dtype=np.float32) for n,t in zip(ORDER,e)],
            outputs=[ct.TensorType(name="denoised_latent",dtype=np.float32)],convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15)
        m.save(p)
    x={"noisy_latent":np.random.randn(B,144,L).astype(np.float32),"text_emb":np.random.randn(B,256,T).astype(np.float32),
       "style_ttl":np.random.randn(B,50,256).astype(np.float32),"latent_mask":np.ones((B,1,L),np.float32),
       "text_mask":np.ones((B,1,T),np.float32),"current_step":np.zeros(B,np.float32),"total_step":np.full(B,8.,np.float32)}
    row={}
    for tag,u in [("ANE",ct.ComputeUnit.CPU_AND_NE),("GPU",ct.ComputeUnit.CPU_AND_GPU)]:
        mm=ct.models.MLModel(p,compute_units=u)
        for _ in range(5): mm.predict(x)
        t=[]
        for _ in range(25):
            t0=time.perf_counter(); mm.predict(x); t.append((time.perf_counter()-t0)*1000)
        row[tag]=float(np.median(t)); del mm
    res[L]=row
    sp=row["GPU"]/row["ANE"]
    print(f"  L={L:<5} ANE {row['ANE']:7.2f} ms  | GPU {row['GPU']:7.2f} ms  | ANE speedup {sp:5.2f}x  | 8 steps: ANE {row['ANE']*8:6.1f} ms")
json.dump(res,open(f"{OUT}/sweep.json","w"),indent=1)
