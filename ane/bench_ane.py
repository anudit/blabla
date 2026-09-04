import numpy as np, time, json, warnings, sys
warnings.filterwarnings("ignore")
import coremltools as ct
import functools; print = functools.partial(print, flush=True)
from collections import Counter

OUT="/Users/anudit/Documents/GitHub/tts-metal/ane/models"
B,T=2,92
ORDER=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
def inp(L):
    return {"noisy_latent":np.random.randn(B,144,L).astype(np.float32),
            "text_emb":np.random.randn(B,256,T).astype(np.float32),
            "style_ttl":np.random.randn(B,50,256).astype(np.float32),
            "latent_mask":np.ones((B,1,L),np.float32),"text_mask":np.ones((B,1,T),np.float32),
            "current_step":np.zeros(B,np.float32),"total_step":np.full(B,8.0,np.float32)}

UNITS={"ANE (CPU+NE)":ct.ComputeUnit.CPU_AND_NE,"GPU (CPU+GPU)":ct.ComputeUnit.CPU_AND_GPU,
       "ALL":ct.ComputeUnit.ALL,"CPU only":ct.ComputeUnit.CPU_ONLY}

def residency(mlpackage, unit):
    """Per-op device assignment from the CoreML compute plan (needs a compiled .mlmodelc)."""
    try:
        cpath = ct.models.MLModel(mlpackage, compute_units=unit).get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(cpath, compute_units=unit)
    except Exception as e:
        return f"compute-plan unavailable: {type(e).__name__}: {str(e)[:100]}", None
    prog = plan.model_structure.program
    if prog is None: return "no mlprogram structure", None
    c=Counter(); est=Counter()
    for fname, func in prog.functions.items():
        for op in func.block.operations:
            d = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if d is None: c["<unassigned>"]+=1; continue
            c[type(d.preferred_compute_device).__name__.replace("ML","").replace("ComputeDevice","")]+=1
            e = plan.get_estimated_cost_for_mlprogram_operation(op)
            if e is not None: est[type(d.preferred_compute_device).__name__.replace("ML","").replace("ComputeDevice","")]+=e.weight
    return c, est

res={}
for L in [64,128,256]:
    path=f"{OUT}/ve_L{L}.mlpackage"
    sys.stdout.flush(); print(f"\n{'='*68}\n L={L}  (B={B}, T={T})  — one ODE step\n{'='*68}")
    for tag,u in UNITS.items():
        try:
            m=ct.models.MLModel(path, compute_units=u)
        except Exception as e:
            print(f"  {tag:16s} LOAD FAILED {str(e)[:90]}"); continue
        x=inp(L)
        for _ in range(5): m.predict(x)
        ts=[]
        for _ in range(30):
            t0=time.perf_counter(); m.predict(x); ts.append((time.perf_counter()-t0)*1000)
        ts=np.array(ts); med=float(np.median(ts))
        res[f"{tag}|{L}"]=med
        print(flush=True) if False else None; print(f"  {tag:16s} median {med:7.2f} ms   min {ts.min():7.2f}   p90 {np.percentile(ts,90):7.2f}   |  x8 steps = {med*8:7.1f} ms")
        del m
    c,est = residency(path, ct.ComputeUnit.CPU_AND_NE)
    if isinstance(c,str): print(f"  residency(CPU+NE): {c}")
    else:
        tot=sum(c.values())
        print(f"  residency(CPU+NE): {tot} ops -> " + ", ".join(f"{k}={v} ({100*v/tot:.1f}%)" for k,v in c.most_common()))
        if est: 
            te=sum(est.values())
            print(f"  cost-weighted:     " + ", ".join(f"{k}={100*v/te:.1f}%" for k,v in est.most_common()))
json.dump(res, open("/Users/anudit/Documents/GitHub/tts-metal/ane/ane_results.json","w"), indent=1)
