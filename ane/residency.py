import numpy as np, warnings, functools, sys
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import coremltools as ct
from collections import Counter
OUT="/Users/anudit/Documents/GitHub/tts-metal/ane/models"

for L in [64,128,256]:
    p=f"{OUT}/ve_L{L}.mlpackage"
    print(f"\n===== L={L} =====")
    m = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_NE)   # keep alive
    cpath = m.get_compiled_model_path()
    try:
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(cpath, compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        print("  plan failed:", type(e).__name__, str(e)[:200]); del m; continue
    prog = plan.model_structure.program
    c=Counter(); cost=Counter(); unsupported=[]
    for fname, func in prog.functions.items():
        for op in func.block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is None:
                c["<none>"]+=1; continue
            dev = type(u.preferred_compute_device).__name__
            dev = dev.replace("ML","").replace("ComputeDevice","")
            c[dev]+=1
            e = plan.get_estimated_cost_for_mlprogram_operation(op)
            if e is not None: cost[dev]+=e.weight
            if dev!="NeuralEngine" and op.operator_name not in ("const","cast"):
                unsupported.append((op.operator_name, dev))
    tot=sum(c.values())
    print(f"  ops: {tot} -> " + ", ".join(f"{k}={v} ({100*v/tot:.1f}%)" for k,v in c.most_common()))
    if cost:
        tc=sum(cost.values())
        print(f"  cost-weighted: " + ", ".join(f"{k}={100*v/tc:.1f}%" for k,v in cost.most_common()))
    if unsupported:
        print(f"  NON-ANE ops ({len(unsupported)}):")
        for name,cnt in Counter(unsupported).most_common(15): print(f"      {name[0]:28s} -> {name[1]:14s} x{cnt}")
    else:
        print("  NON-ANE ops: none (excluding const/cast)")
    del m
