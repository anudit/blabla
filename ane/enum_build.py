import os,time,warnings,functools,json
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference; from onnx2torch import convert
import coremltools as ct
from collections import Counter
SRC="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
OUT="/Users/anudit/Documents/GitHub/tts-metal/ane/models3"; os.makedirs(OUT,exist_ok=True)
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
class W(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,a,b,c,d,e,f,g): return s.m(a,b,c,d,e,f,g)
net=W(net)
ORDER=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
B,T=1,92
BUCKETS=[64,96,128,160,192,256]
ex=[torch.randn(B,144,BUCKETS[0]),torch.randn(B,256,T),torch.randn(B,50,256),
    torch.ones(B,1,BUCKETS[0]),torch.ones(B,1,T),torch.zeros(B),torch.full((B,),8.0)]
with torch.no_grad():
    ts=torch.jit.freeze(torch.jit.trace(net,tuple(ex),check_trace=False).eval())
lat=ct.EnumeratedShapes(shapes=[(B,144,L) for L in BUCKETS], default=(B,144,BUCKETS[0]))
msk=ct.EnumeratedShapes(shapes=[(B,1,L)   for L in BUCKETS], default=(B,1,BUCKETS[0]))
t0=time.time()
m=ct.convert(ts,
    inputs=[ct.TensorType(name="noisy_latent",shape=lat,dtype=np.float32),
            ct.TensorType(name="text_emb",shape=(B,256,T),dtype=np.float32),
            ct.TensorType(name="style_ttl",shape=(B,50,256),dtype=np.float32),
            ct.TensorType(name="latent_mask",shape=msk,dtype=np.float32),
            ct.TensorType(name="text_mask",shape=(B,1,T),dtype=np.float32),
            ct.TensorType(name="current_step",shape=(B,),dtype=np.float32),
            ct.TensorType(name="total_step",shape=(B,),dtype=np.float32)],
    outputs=[ct.TensorType(name="denoised_latent",dtype=np.float32)],
    convert_to="mlprogram",compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,minimum_deployment_target=ct.target.macOS15)
p=f"{OUT}/ve_enum.mlpackage"; m.save(p)
print(f"converted in {time.time()-t0:.1f}s -> {p}")
os.system(f"du -sh {p}")

mo=ct.models.MLModel(p,compute_units=ct.ComputeUnit.CPU_AND_NE)
print("\n  L      ANE ms   8-steps")
for L in BUCKETS:
    x={"noisy_latent":np.random.randn(B,144,L).astype(np.float32),"text_emb":np.random.randn(B,256,T).astype(np.float32),
       "style_ttl":np.random.randn(B,50,256).astype(np.float32),"latent_mask":np.ones((B,1,L),np.float32),
       "text_mask":np.ones((B,1,T),np.float32),"current_step":np.zeros(B,np.float32),"total_step":np.full(B,8.,np.float32)}
    for _ in range(5): mo.predict(x)
    t=[]
    for _ in range(25):
        t0=time.perf_counter(); mo.predict(x); t.append((time.perf_counter()-t0)*1000)
    print(f"  {L:<5}  {np.median(t):7.2f}  {np.median(t)*8:7.1f} ms")
try:
    plan=ct.models.compute_plan.MLComputePlan.load_from_path(mo.get_compiled_model_path(),compute_units=ct.ComputeUnit.CPU_AND_NE)
    c=Counter(); bad=Counter()
    for f in plan.model_structure.program.functions.values():
        for op in f.block.operations:
            u=plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is None: continue
            d=type(u.preferred_compute_device).__name__.replace("ML","").replace("ComputeDevice","")
            c[d]+=1
            if d!="NeuralEngine": bad[op.operator_name]+=1
    print(f"\n  residency: {c['NeuralEngine']}/{sum(c.values())} on ANE; non-ANE: {dict(bad)}")
except Exception as e: print("  plan:",type(e).__name__,str(e)[:120])
