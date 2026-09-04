import os,sys,time,json,warnings,functools
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference
from onnx2torch import convert
import coremltools as ct
from collections import Counter

BASE="/Users/anudit/Documents/GitHub/tts-metal/ane"
SRC_PLAIN="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
SRC_HOIST=f"{BASE}/ve_hoisted.onnx"
OUT=f"{BASE}/models2"; os.makedirs(OUT,exist_ok=True)
B,T=1,92
def _p():
    def q(m):
        if isinstance(m,str): m=onnx.load(m)
        try: return shape_inference.infer_shapes(m)
        except Exception: return m
    onnx2torch.converter.safe_shape_inference=q
_p()

VARIANTS={
 "plain": (SRC_PLAIN, ["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]),
 "hoist": (SRC_HOIST, ["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","total_step","time_emb"]),
}
def mk(order,L):
    d={"noisy_latent":torch.randn(B,144,L),"text_emb":torch.randn(B,256,T),"style_ttl":torch.randn(B,50,256),
       "latent_mask":torch.ones(B,1,L),"text_mask":torch.ones(B,1,T),
       "current_step":torch.zeros(B),"total_step":torch.full((B,),8.0),"time_emb":torch.randn(2*B,64,1)}
    return [d[k] for k in order]

nets={}
for v,(src,order) in VARIANTS.items():
    sl=onnxslim.slim(src)
    for o in sl.opset_import:
        if o.domain in ("","ai.onnx"): o.version=17
    n=convert(sl); n.eval()
    for p in n.parameters(): p.requires_grad_(False)
    nets[v]=n
    print(f"loaded {v}")

class W(torch.nn.Module):
    def __init__(s,m,order): super().__init__(); s.m=m; s.order=order
    def forward(s,*a): return s.m(*a)

def residency(path):
    m=ct.models.MLModel(path,compute_units=ct.ComputeUnit.CPU_AND_NE)
    try: plan=ct.models.compute_plan.MLComputePlan.load_from_path(m.get_compiled_model_path(),compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e: return None,f"{type(e).__name__}"
    c=Counter(); bad=Counter()
    for f in plan.model_structure.program.functions.values():
        for op in f.block.operations:
            u=plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is None: continue
            d=type(u.preferred_compute_device).__name__.replace("ML","").replace("ComputeDevice","")
            c[d]+=1
            if d!="NeuralEngine": bad[op.operator_name]+=1
    del m; return c,bad

LS=[int(x) for x in sys.argv[1].split(",")]
report={}
for v,(src,order) in VARIANTS.items():
    net=W(nets[v],order)
    print(f"\n{'#'*72}\n#  variant = {v}   (B={B}, T={T})\n{'#'*72}")
    for L in LS:
        p=f"{OUT}/{v}_L{L}.mlpackage"
        ex=mk(order,L)
        if not os.path.exists(p):
            with torch.no_grad():
                ts=torch.jit.freeze(torch.jit.trace(net,tuple(ex),check_trace=False).eval())
            mm=ct.convert(ts,inputs=[ct.TensorType(name=n,shape=tuple(t.shape),dtype=np.float32) for n,t in zip(order,ex)],
                outputs=[ct.TensorType(name="denoised_latent",dtype=np.float32)],convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16,compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.macOS15)
            mm.save(p)
        x={n:t.numpy() for n,t in zip(order,ex)}
        row={}
        for tag,u in [("ANE",ct.ComputeUnit.CPU_AND_NE),("GPU",ct.ComputeUnit.CPU_AND_GPU)]:
            mo=ct.models.MLModel(p,compute_units=u)
            for _ in range(5): mo.predict(x)
            t=[]
            for _ in range(25):
                t0=time.perf_counter(); mo.predict(x); t.append((time.perf_counter()-t0)*1000)
            row[tag]=float(np.median(t)); del mo
        c,bad=residency(p)
        nres = f"{c['NeuralEngine']}/{sum(c.values())} ANE" if c else "n/a"
        badstr = ", ".join(f"{k}x{n}" for k,n in bad.most_common(6)) if c and bad else ("clean" if c else bad)
        print(f"  L={L:<5} ANE {row['ANE']:7.2f} ms | GPU {row['GPU']:7.2f} ms | ANEx{row['GPU']/row['ANE']:4.2f} | 8st {row['ANE']*8:6.1f} ms | {nres} | {badstr}")
        report[f"{v}|{L}"]={**row,"ane_ops":c['NeuralEngine'] if c else None,"tot_ops":sum(c.values()) if c else None,"non_ane":dict(bad) if c else None}
json.dump(report,open(f"{BASE}/report_b1.json","w"),indent=1)
