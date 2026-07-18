# tts-metal

A native macOS **menu-bar text-to-speech reader** that runs entirely on-device in
**Metal compute shaders** — no Python, no ONNX Runtime, no Core ML. Select text
anywhere in the system, press **⌥⌘R**, and it speaks the selection. Synthesized
speech is optionally cleaned and upsampled to 48 kHz by a second neural network
(**LavaSR v2**), also implemented from scratch in Metal.

```
selection / text ──▶ Phonemizer ──▶ Kitten TTS (Metal) ──▶ LavaSR v2 (Metal) ──▶ AVAudioEngine
                     eSpeak rules     24 kHz speech          48 kHz enhanced       gapless queue
```

---

## Stack

| Layer | Implementation |
|---|---|
| **UI** | SwiftUI `MenuBarExtra` (agent app, no dock icon) |
| **Global hotkey** | Carbon `RegisterEventHotKey` (⌥⌘R) |
| **Selection capture** | Accessibility API (`AXUIElement`) |
| **Phonemization** | eSpeak-style rules + 234k-entry dictionary (`Phonemizer`, `EspeakRules`) |
| **TTS model** | **Kitten TTS mini** (80 M params) — ALBERT text encoder + prosody predictor + HiFi-GAN-style decoder, parsed from ONNX and run in Metal (`MetalTtsEngine`, `OnnxParser`) |
| **Enhancer** | **LavaSR v2** (Vocos, 50 MB) — mel → ConvNeXt backbone → iSTFT, run in Metal (`LavaEnhancer`) |
| **Compute** | ~40 hand-written Metal kernels (`Shaders.metal`) |
| **Audio** | `AVAudioEngine` + `AVAudioPlayerNode`, fixed 48 kHz, gapless queue with 5-chunk look-ahead backpressure (`AudioPlayer`) |
| **Weights** | safetensors (LavaSR) + ONNX (Kitten), bundled, dequantized/uploaded at launch |

Everything runs on the GPU via a single `MTLCommandQueue`; heavy work is off the
main thread, and chunk generation pipelines ahead of playback so the next sentence
is ready before the current one finishes.

---

## LavaSR v2 enhancer (the Metal port)

LavaSR v2 is a **Vocos**-based universal bandwidth-extension / super-resolution
model. It takes the 24 kHz TTS output, computes a mel spectrogram, and a ConvNeXt
network regresses a full-band complex STFT that is inverted to a **48 kHz** waveform.

**Forward pass** (all in Metal — see the `lava_*` kernels plus reused
`reuse_matmul` / iSTFT / conv1d / layernorm / gelu kernels):

```
resample 24k → 44.1k                      (Lanczos, CPU)
STFT magnitude        n_fft=2048, hop=512  → [1025, T]      lava_stft_mag
mel = fbᵀ · mag       80 mels, slaney      → [80, T]        reuse_matmul
safe_log                                                    lava_safe_log
Conv1d embed          80→512, k=7                           conv1d_tiled
LayerNorm
8 × ConvNeXt block:
    depthwise Conv1d  512, k=7                              lava_dwconv1d
    LayerNorm → Linear 512→1536 → GELU → Linear 1536→512    reuse_matmul, gelu
    γ · x + residual                                        lava_gamma_residual
LayerNorm
Linear head           512 → 2050 (mag + phase)             reuse_matmul
mag = clip(exp(·)), S = mag·e^{iϕ}                          lava_head_to_complex
iSTFT                 n_fft=2048, hop=512, "same"           reuse_irfft, reuse_ola
resample 44.1k → 48k                       (Lanczos, CPU)
```

The model runs at **44.1 kHz internally** (the rate its mel filterbank was built
for) and the output is resampled to 48 kHz. Config: `dim=512`, `intermediate=1536`,
`8` ConvNeXt layers, `n_mels=80`, `n_fft=2048`, `hop=512`.

---

## Performance

Measured on Apple Silicon (Release build), steady state after warm-up, using the
built-in profiler (`REUSE_SELFTEST=1`, `REUSE_SECS=<n>`). Times are the **LavaSR v2
enhancement stage only** (the Kitten TTS synthesis runs before it).

### Stage breakdown (3.5 s of audio)

| Stage | Time | Notes |
|---|---:|---|
| mel (resample + STFT + mel + log) | ~75 ms | dominated by CPU resampling + the 2048-pt STFT |
| Conv1d embed | ~2 ms | |
| 8 × ConvNeXt backbone | ~10 ms | the actual "model" — tiny |
| Linear head | ~2 ms | |
| iSTFT | ~3 ms | |
| **total** | **~152 ms** | |

### Realtime factor (enhancement)

| Input audio | Enhance time | Realtime factor |
|---:|---:|---:|
| 1.5 s | 0.074 s | **20.2×** |
| 3.5 s | 0.152 s | **23.1×** |
| 10 s | 0.414 s | **24.1×** |

> **Realtime factor** = (seconds of audio) ÷ (seconds to process). 20× means one
> second of speech is enhanced in ~50 ms, so enhancement is effectively free
> relative to playback — the next sentence is always ready before the current one
> ends. First-audio latency for a typical sentence (~1.5 s) is ~75 ms.

The heaviest remaining cost is the CPU Lanczos resampling and the STFT; the neural
backbone itself is ~10 ms. This replaced the previous RE-USE (SEMamba) enhancer,
which ran at ~0.7× realtime (slower than playback) and caused multi-second stalls
between sentences.

---

## Build & run

Open `tts-metal/tts-metal.xcodeproj` in Xcode and run (Release recommended), or:

```bash
cd tts-metal
xcodebuild -project tts-metal.xcodeproj -scheme tts-metal -configuration Release build
```

The model weights (`kitten_tts_mini_v0_8.onnx`, `voices.npz`, `lavasr_v2.safetensors`)
are bundled as resources and git-ignored (fetched/placed at build time).

**Usage:** grant Accessibility permission (to read the system selection), pick a
voice/speed in the menu-bar popover, then press **⌥⌘R** on any highlighted text, or
type into the "Type to speak" box. The **Enhance audio (LavaSR v2)** toggle
(default on) controls the 48 kHz enhancement layer.

**Profiling the enhancer:**

```bash
REUSE_SELFTEST=1 REUSE_SECS=3.5 /path/to/tts-metal.app/Contents/MacOS/tts-metal
```

---

## Credits

- **Kitten TTS** — KittenML (`kitten-tts-mini`).
- **LavaSR v2** — Yatharth Sharma, Apache-2.0 ([repo](https://github.com/ysharma3501/LavaSR)),
  built on **Vocos** (Siuzdak et al.).
- The UL-UNAS denoiser shipped alongside LavaSR is not (yet) ported; the enhancer
  path is the LavaSR v2 BWE model.
