import os,warnings,functools,time
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference; from onnx2torch import convert
import coremltools as ct
SRC="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
OUT="/Users/anudit/Documents/GitHub/tts-metal/ane/single_T96_L96.mlpackage"
if not os.path.exists(OUT):
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
    net=W(net); B,T,L=1,96,96
    ORDER=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
    ex=[torch.randn(B,144,L),torch.randn(B,256,T),torch.randn(B,50,256),torch.ones(B,1,L),
        torch.ones(B,1,T),torch.zeros(B),torch.full((B,),8.0)]
    with torch.no_grad():
        ts=torch.jit.freeze(torch.jit.trace(net,tuple(ex),check_trace=False).eval())
    m=ct.convert(ts,inputs=[ct.TensorType(name=n,shape=tuple(t.shape),dtype=np.float32) for n,t in zip(ORDER,ex)],
        outputs=[ct.TensorType(name="denoised_latent",dtype=np.float32)],convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15)
    m.save(OUT); print("built")
B,T,L=1,96,96
t0=time.time()
m=ct.models.MLModel(OUT,compute_units=ct.ComputeUnit.CPU_AND_NE)
x={"noisy_latent":np.random.randn(B,144,L).astype(np.float32),"text_emb":np.random.randn(B,256,T).astype(np.float32),
   "style_ttl":np.random.randn(B,50,256).astype(np.float32),"latent_mask":np.ones((B,1,L),np.float32),
   "text_mask":np.ones((B,1,T),np.float32),"current_step":np.zeros(B,np.float32),"total_step":np.full(B,8.,np.float32)}
m.predict(x)
print(f"SINGLE-FUNCTION T96_L96: first-load+compile {time.time()-t0:.2f}s")
