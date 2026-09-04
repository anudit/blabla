import numpy as np, onnxruntime as ort, time, sys, json

MODEL = "/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
B, T = 2, 92

def make_inputs(L):
    return {
        "noisy_latent": np.random.randn(B,144,L).astype(np.float32),
        "text_emb":     np.random.randn(B,256,T).astype(np.float32),
        "style_ttl":    np.random.randn(B,50,256).astype(np.float32),
        "latent_mask":  np.ones((B,1,L), np.float32),
        "text_mask":    np.ones((B,1,T), np.float32),
        "current_step": np.array([0,0], np.float32),
        "total_step":   np.array([8,8], np.float32),
    }

def bench(providers, L, tag, popts=None, warmup=3, iters=15):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_free_dimension_override_by_name("batch_size", B)
    so.add_free_dimension_override_by_name("latent_length", L)
    so.add_free_dimension_override_by_name("text_length", T)
    try:
        sess = ort.InferenceSession(MODEL, so, providers=providers,
                                    provider_options=popts) if popts else \
               ort.InferenceSession(MODEL, so, providers=providers)
    except Exception as e:
        print(f"  {tag:22s} L={L:<4} SESSION FAILED: {str(e)[:160]}"); return None
    x = make_inputs(L)
    for _ in range(warmup): sess.run(None, x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); sess.run(None, x); ts.append((time.perf_counter()-t0)*1000)
    ts = np.array(ts)
    print(f"  {tag:22s} L={L:<4} median {np.median(ts):7.1f} ms   min {ts.min():7.1f}   p90 {np.percentile(ts,90):7.1f}")
    return float(np.median(ts))

print("providers available:", ort.get_available_providers())
res = {}
for L in [64, 128, 256]:
    print(f"\n--- L={L} (B={B}, T={T}) ---")
    res[f"cpu_{L}"]  = bench(["CPUExecutionProvider"], L, "CPU EP")
    for name, opts in [
        ("CoreML ALL",  {"MLComputeUnits":"ALL"}),
        ("CoreML ANE-only", {"MLComputeUnits":"CPUAndNeuralEngine"}),
        ("CoreML GPU",  {"MLComputeUnits":"CPUAndGPU"}),
    ]:
        res[f"{name}_{L}"] = bench(["CoreMLExecutionProvider","CPUExecutionProvider"], L, name,
                                   popts=[opts, {}])
json.dump(res, open("/Users/anudit/Documents/GitHub/tts-metal/ane/ort_results.json","w"), indent=1)
