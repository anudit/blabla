#!/usr/bin/env python3
"""
Supertonic 3 reference runner (onnxruntime) — the ground truth the Metal port
targets. Replicates supertonic/core.py's pipeline exactly, times each stage
end-to-end, and writes a correct WAV.

Usage:
  /tmp/st_venv/bin/python tools/supertonic_reference.py \
      --onnx supertonic-3/onnx --voice supertonic-3/voice_styles/M1.json \
      --steps 8 --out /tmp/supertonic_reference.wav
"""
import argparse, json, time, wave, struct
import numpy as np
import onnxruntime as ort


def load_voice(path):
    d = json.load(open(path))
    ttl = np.array(d["style_ttl"]["data"], dtype=np.float32)   # [1,50,256]
    dp = np.array(d["style_dp"]["data"], dtype=np.float32)     # [1,8,16]
    return ttl, dp


def length_to_mask(lengths, max_len=None):
    max_len = max_len or int(lengths.max())
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, 1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def preprocess(text, lang="en"):
    # minimal: strip, ensure ending punctuation, wrap language token (matches core.py)
    import unicodedata, re
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="supertonic-3/onnx")
    ap.add_argument("--voice", default="supertonic-3/voice_styles/M1.json")
    ap.add_argument("--text", default="A gentle breeze moved through the open window while everyone listened to the story.")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--out", default="/tmp/supertonic_reference.wav")
    ap.add_argument("--sr", type=int, default=44100)
    a = ap.parse_args()

    so = ort.SessionOptions()
    def sess(n): return ort.InferenceSession(f"{a.onnx}/{n}.onnx", so, providers=["CPUExecutionProvider"])
    t0 = time.time()
    dp = sess("duration_predictor"); te = sess("text_encoder")
    ve = sess("vector_estimator"); vo = sess("vocoder")
    load_ms = (time.time() - t0) * 1000

    indexer = json.load(open(f"{a.onnx}/unicode_indexer.json"))
    ttl, dpstyle = load_voice(a.voice)

    text = preprocess(a.text, "en")
    ids = np.array([[indexer[ord(c)] for c in text]], dtype=np.int64)
    text_mask = length_to_mask(np.array([ids.shape[1]], dtype=np.int64))
    print(f"text='{text}' tokens={ids.shape[1]}")

    times = {}
    def timed(name, fn):
        s = time.time(); r = fn(); times[name] = (time.time() - s) * 1000; return r

    dur = timed("duration_predictor", lambda: dp.run(None, {"text_ids": ids, "style_dp": dpstyle, "text_mask": text_mask})[0])
    dur = dur / a.speed
    text_emb = timed("text_encoder", lambda: te.run(None, {"text_ids": ids, "style_ttl": ttl, "text_mask": text_mask})[0])

    # sample_noisy_latent
    sr, base_chunk, ccf, ldim = a.sr, 512, 6, 24
    wav_len_max = dur.max() * sr
    wav_lengths = (dur * sr).astype(np.int64)
    chunk = base_chunk * ccf
    latent_len = int(np.ceil(wav_len_max / chunk))
    latent_dim = ldim * ccf
    xt = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
    latent_lengths = (wav_lengths + chunk - 1) // chunk
    latent_mask = length_to_mask(latent_lengths)
    xt = xt * latent_mask
    total_step = np.array([a.steps], dtype=np.float32)
    print(f"duration={dur.ravel()} s  latent_len={latent_len}")

    def flow():
        x = xt
        for step in range(a.steps):
            cs = np.array([step], dtype=np.float32)
            x = ve.run(None, {"noisy_latent": x, "text_emb": text_emb, "style_ttl": ttl,
                              "text_mask": text_mask, "latent_mask": latent_mask,
                              "current_step": cs, "total_step": total_step})[0]
        return x
    xt = timed("flow_matching", flow)
    wav = timed("vocoder", lambda: vo.run(None, {"latent": xt})[0])

    total = sum(times.values())
    audio_s = wav.shape[-1] / sr
    print(f"\n[reference] load {load_ms:.0f} ms")
    for k in ["duration_predictor", "text_encoder", "flow_matching", "vocoder"]:
        print(f"[reference]   {k:20s} {times[k]:8.1f} ms")
    print(f"[reference]   {'TOTAL':20s} {total:8.1f} ms  → {audio_s:.2f}s audio  RTF {audio_s/(total/1000):.2f}x  (steps={a.steps})")

    # write wav
    w = wav.squeeze()
    w = np.clip(w, -1, 1)
    with wave.open(a.out, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((w * 32767).astype("<i2").tobytes())
    print(f"[reference] wrote {a.out} ({audio_s:.2f}s)")


if __name__ == "__main__":
    main()
