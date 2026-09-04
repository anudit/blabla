import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
import numpy as np, torch, onnx, onnxslim, onnx2torch
from onnx import shape_inference
from onnx2torch import convert
import coremltools as ct

ONNX_DIR = "/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic"
OUT      = "/Users/anudit/Documents/GitHub/tts-metal/ane/models"
os.makedirs(OUT, exist_ok=True)
B, T = 2, 92

def _patch():
    def patched(m):
        if isinstance(m, str): m = onnx.load(m)
        try: return shape_inference.infer_shapes(m)
        except Exception: return m
    onnx2torch.converter.safe_shape_inference = patched
_patch()

def load_pt(name):
    slimmed = onnxslim.slim(os.path.join(ONNX_DIR, name))
    for o in slimmed.opset_import:
        if o.domain in ("", "ai.onnx"): o.version = 17
    m = convert(slimmed); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

ORDER = ["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]

def sample(L):
    return [torch.randn(B,144,L), torch.randn(B,256,T), torch.randn(B,50,256),
            torch.ones(B,1,L), torch.ones(B,1,T),
            torch.zeros(B), torch.full((B,), 8.0)]

class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step):
        return self.m(noisy_latent, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step)

print("loading vector_estimator.onnx -> torch ...", flush=True)
t0=time.time(); net = Wrap(load_pt("vector_estimator.onnx")); print(f"  ok {time.time()-t0:.1f}s", flush=True)

results={}
for L in [64,128,256]:
    ex = sample(L)
    print(f"\n=== L={L} ===", flush=True)
    with torch.no_grad():
        ref = net(*ex)
    print("  torch out", tuple(ref.shape), float(ref.mean()), float(ref.std()), flush=True)
    with torch.no_grad():
        ts = torch.jit.trace(net, tuple(ex), check_trace=False); ts = torch.jit.freeze(ts.eval())
    mlm = ct.convert(
        ts,
        inputs=[ct.TensorType(name=n, shape=tuple(e.shape), dtype=np.float32) for n,e in zip(ORDER, ex)],
        outputs=[ct.TensorType(name="denoised_latent", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    p = f"{OUT}/ve_L{L}.mlpackage"; mlm.save(p); print("  saved", p, flush=True)
    # numerical check
    out = mlm.predict({n: e.numpy() for n,e in zip(ORDER, ex)})["denoised_latent"]
    err = np.abs(out - ref.numpy()); den = np.abs(ref.numpy()).mean()
    print(f"  fp16 vs torch: max_abs={err.max():.4e}  mean_abs={err.mean():.4e}  rel={err.mean()/den:.4e}", flush=True)
    results[L] = {"max_abs": float(err.max()), "rel": float(err.mean()/den)}
json.dump(results, open(f"{OUT}/convert_report.json","w"), indent=1)
print("\nDONE")
