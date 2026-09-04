import os,time,warnings,functools
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
from collections import Counter
M2="/Users/anudit/Documents/GitHub/tts-metal/ane/models2"
OUT="/Users/anudit/Documents/GitHub/tts-metal/ane/models3"; os.makedirs(OUT,exist_ok=True)
BUCKETS=[64,96,128,192,256]
d=MultiFunctionDescriptor()
for L in BUCKETS:
    d.add_function(f"{M2}/plain_L{L}.mlpackage", src_function_name="main", target_function_name=f"L{L}")
d.default_function_name=f"L{BUCKETS[0]}"
p=f"{OUT}/ve_multi.mlpackage"
t0=time.time(); save_multifunction(d,p); print(f"merged {len(BUCKETS)} functions in {time.time()-t0:.1f}s")
os.system(f"du -sh {p}; echo 'sum of separate:'; du -shc "+" ".join(f"{M2}/plain_L{L}.mlpackage" for L in BUCKETS)+" | tail -1")
B,T=1,92
print("\n  L      ANE ms   8-steps    residency")
for L in BUCKETS:
    mo=ct.models.MLModel(p,compute_units=ct.ComputeUnit.CPU_AND_NE,function_name=f"L{L}")
    x={"noisy_latent":np.random.randn(B,144,L).astype(np.float32),"text_emb":np.random.randn(B,256,T).astype(np.float32),
       "style_ttl":np.random.randn(B,50,256).astype(np.float32),"latent_mask":np.ones((B,1,L),np.float32),
       "text_mask":np.ones((B,1,T),np.float32),"current_step":np.zeros(B,np.float32),"total_step":np.full(B,8.,np.float32)}
    for _ in range(5): mo.predict(x)
    t=[]
    for _ in range(25):
        t0=time.perf_counter(); mo.predict(x); t.append((time.perf_counter()-t0)*1000)
    res=""
    try:
        plan=ct.models.compute_plan.MLComputePlan.load_from_path(mo.get_compiled_model_path(),
              compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"L{L}")
        c=Counter()
        for f in plan.model_structure.program.functions.values():
            for op in f.block.operations:
                u=plan.get_compute_device_usage_for_mlprogram_operation(op)
                if u: c[type(u.preferred_compute_device).__name__]+=1
        res=f"{c['MLNeuralEngineComputeDevice']}/{sum(c.values())} ANE"
    except Exception as e: res=type(e).__name__
    print(f"  {L:<5}  {np.median(t):7.2f}  {np.median(t)*8:7.1f} ms   {res}")
    del mo
