import os,time,warnings,functools
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference; from onnx2torch import convert
import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
SRC="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
TMP="/Users/anudit/Documents/GitHub/tts-metal/ane/grid_tmp"; os.makedirs(TMP,exist_ok=True)
OUT="/Users/anudit/Documents/GitHub/tts-metal/ane/models3"
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
B=1
TS=[96,192,304]; LS=[64,96,128,192,288]
ORDER=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
d=MultiFunctionDescriptor(); n=0; t0=time.time()
for T in TS:
    for L in LS:
        p=f"{TMP}/T{T}_L{L}.mlpackage"
        if not os.path.exists(p):
            ex=[torch.randn(B,144,L),torch.randn(B,256,T),torch.randn(B,50,256),
                torch.ones(B,1,L),torch.ones(B,1,T),torch.zeros(B),torch.full((B,),8.0)]
            with torch.no_grad():
                ts=torch.jit.freeze(torch.jit.trace(net,tuple(ex),check_trace=False).eval())
            m=ct.convert(ts,inputs=[ct.TensorType(name=nm,shape=tuple(t.shape),dtype=np.float32) for nm,t in zip(ORDER,ex)],
                outputs=[ct.TensorType(name="denoised_latent",dtype=np.float32)],convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16,compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.macOS15)
            m.save(p)
        d.add_function(p, src_function_name="main", target_function_name=f"T{T}_L{L}")
        n+=1
d.default_function_name=f"T{TS[0]}_L{LS[0]}"
pkg=f"{OUT}/ve_grid.mlpackage"
save_multifunction(d,pkg)
print(f"\nbuilt {n} functions in {time.time()-t0:.0f}s")
os.system(f"du -sh {pkg}; du -shc {TMP}/*.mlpackage | tail -1")
