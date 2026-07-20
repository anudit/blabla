# tts-metal

A native macOS **menu-bar text-to-speech reader** that runs the **Supertonic 3** latent flow-matching model entirely on-device using **Metal compute shaders** — no Python, no ONNX Runtime, no Core ML. 

Select text anywhere in the system, press **⌥⌘R**, and it speaks the selection in a high-fidelity voice.

```
selection / text ──▶ Tokenizer ──▶ Supertonic 3 (Metal) ──▶ Resampler (Lanczos) ──▶ AVAudioEngine (48 kHz)
                     65k table        44.1 kHz audio            upsample               gapless queue
```

---

## Features

- **Pure Metal Compute**: Executes the full 99.2M parameter Supertonic 3 architecture directly on Apple Silicon GPUs using custom Metal kernels.
- **Multilingual Support**: Synthesizes speech across 31 languages (including English, Korean, Japanese, Arabic, German, Spanish, French, Hindi, and more).
- **Zero-Shot Voice Cloning**: Dynamically estimates and applies speaker embeddings from reference voice styles defined in JSON configs.
- **Expression Tags**: Supports expressive synthesis using prompt tags like `<laugh>`, `<breath>`, and `<sigh>`.
- **High-Fidelity Output**: Generates native 44.1 kHz full-band audio, which is resampled on-the-fly to 48 kHz for gapless playback using a high-quality Lanczos sinc filter.

---

## Stack

| Layer | Implementation |
|---|---|
| **UI** | SwiftUI `MenuBarExtra` (agent app, running entirely in the macOS menu bar) |
| **Global hotkey** | Carbon `RegisterEventHotKey` (⌥⌘R) |
| **Selection capture** | Accessibility API (`AXUIElement`) |
| **Tokenizer** | Fast, local unicode codepoint index mapping using a `65536`-entry table (`unicode_indexer.json`) |
| **TTS Engine** | **Supertonic 3** (99.2M parameters) — parsed from ONNX model definitions at startup and executed via custom Metal wrappers (`SupertonicEngine.swift`) |
| **Compute** | Hand-written Metal compute shaders (`Shaders.metal`) optimized for attention, Rotary Position Embeddings (RoPE), relative-position biases, and causal dilated convolutions |
| **Audio Playback** | `AVAudioEngine` + `AVAudioPlayerNode` queue with a 5-chunk sliding-window backpressure system for gapless streaming |
| **Weights** | ONNX weights (`duration_predictor`, `text_encoder`, `vector_estimator`, `vocoder`) and voice style configs bundled directly in app resources |

---

## Supertonic 3 TTS pipeline

Supertonic 3 is an iterative flow-matching ODE-based text-to-speech model. Its execution flow consists of the following custom-implemented stages:

1. **Text Preprocessing & Tokenization**: Normalizes incoming unicode text, strips excess whitespace, inserts punctuation, wraps sections with language tags (e.g. `<lang>...</lang>`), and maps characters to indices.
2. **Duration Predictor**: Runs the `duration_predictor` model using text token IDs, text mask, and a `style_dp` (1x8x16) voice style tensor to determine phoneme-level frame durations.
3. **Text Encoder**: Runs the `text_encoder` model using text token IDs, text mask, and a `style_ttl` (1x50x256) voice style tensor to produce row-major phoneme embeddings.
4. **Flow Matching ODE (Vector Estimator)**: 
   - Generates a Gaussian noise latent tensor matching the length computed by the duration predictor.
   - Solves the probability flow ODE by running the `vector_estimator` model (64M parameters) iteratively (typically **8 steps** of an Euler ODE solver) to estimate the vector field, transforming the noise into a clean speech latent representation.
5. **Vocoder**: Runs the `vocoder` model (25M parameters) to synthesize the final 44.1 kHz raw audio waveform.

---

## Performance

Benchmarks measured on an **Apple M2 Max** (Release build, 8 ODE steps) for a 92-token
English test sentence producing **5.5 s** of 44.1 kHz audio.

### End-to-End Latency

| Pipeline | End-to-End Time | Audio Generated | Realtime Factor (RTF) |
|---|---:|---:|---:|
| **Supertonic 3 (Metal)** | **~950 ms** | 5.5 s @ 44.1 kHz | **~5.8×** (at 8 steps) |

### Stage Breakdown

| Stage | Latency | Notes |
|---|---:|---|
| **Duration Predictor** | ~18 ms | Lightweight MLP |
| **Text Encoder** | ~114 ms | ConvNeXt + Relative-Position Attention |
| **Flow Matching Loop** | **~800 ms** | 8 steps of the 64M parameter vector field (~50 ms/step, ×2 for CFG) |
| **Vocoder** | ~47 ms | Causal Convolutions & ISTFT head |

> **Realtime Factor (RTF)** = (seconds of audio generated) ÷ (synthesis latency). RTF ~5.8×
> means 1 second of speech is synthesized in ~170 ms. The flow-matching ODE steps are a
> speed-to-quality control knob (fewer steps = faster).

### Optimization history

The flow-matching loop originally ran at **~8,670 ms (RTF 0.62×, slower than realtime)** — a
severe CPU stall. Three changes brought it to ~800 ms (**~9× faster**) with **bit-identical**
output (verified by the `ST_VALIDATE` harness: `vfield[0] corr=1.0000`):

1. **BLAS cross-attention** — the two per-block cross-attentions ran as scalar single-threaded
   Swift triple-loops (~10 GMAC of serial work per generation, the actual hang). Now Accelerate
   `cblas_sgemm` GEMMs (Q·Kᵀ, softmax, P·V). *Largest win.*
2. **Register-blocked fused GEMM** — ConvNeXt pointwise convs moved from a naive
   one-output-per-thread kernel to a 64×64-tile / 4×4-per-thread GEMM (`reuse_matmul_{gelu,relu}`)
   with the activation fused into the epilogue.
3. **Transposed-weight cache** — constant weights were re-transposed on every matmul call
   (~900 redundant dispatches/generation); now transposed once and cached.

### Next optimization opportunities (ranked by expected win)

1. **Move cross-attention Q/K/V/out projections onto the GPU (~150–300 ms).** The attentions
   are BLAS now, but each `vfVelocity` still does `readBuf`/`makeBuf` round-trips that stall the
   GPU pipeline (CPU↔GPU sync per block). Keeping projections on-device (they're plain matmuls,
   weights already resident) removes the flushes and the largest remaining serial gap.
2. **Batch conditional + unconditional CFG passes (~up to 2× on flow matching).** Each step runs
   `vfVelocity` twice (cond + uncond) sequentially. Stacking them into one batch-of-2 doubles GPU
   occupancy for the small `L≈79` sequence and halves dispatch/launch overhead.
3. **fp16 storage + `simdgroup_matrix` GEMM (~1.5–2× on matmuls).** The GEMM is fp32 scalar FMA;
   Apple GPUs have hardware `simdgroup_float8x8` / fp16 matrix units. Store weights and feature
   maps as `half` and accumulate in fp32 for a large matmul throughput gain.
4. **Persistent command buffer / fewer encoders (~30–80 ms).** `flushAndWait` is called on every
   `readBuf`; batching more work per command buffer and reading back only at stage boundaries cuts
   encoder churn and driver overhead.
5. **Fuse LayerNorm + the dwconv/residual epilogue (~20–40 ms).** ConvNeXt does dwconv → transpose
   → LayerNorm → matmul → gamma·residual as separate dispatches with intermediate buffers; fusing
   the norm and residual passes removes several elementwise round-trips per layer.
6. **Text-encoder CPU `speechPromptedEncoder` → BLAS/GPU (~50–90 ms).** Still a scalar port; the
   same BLAS treatment as the vector-field attentions would shrink the ~114 ms text-encoder stage.

---

## Build & run

Open `tts-metal/tts-metal.xcodeproj` in Xcode and build in **Release** mode, or build from the command line:

```bash
cd tts-metal
xcodebuild -project tts-metal.xcodeproj -scheme tts-metal -configuration Release build
```

The model weights are git-ignored and bundled into the app resources under `supertonic/`:
- `duration_predictor.onnx`
- `text_encoder.onnx`
- `vector_estimator.onnx`
- `vocoder.onnx`
- `unicode_indexer.json`
- `voice_styles/` (JSON reference configurations)

### Usage
1. Open the application.
2. Grant Accessibility permissions (required to read the system selection).
3. Select a voice (presets like `M1`..`M5`, `F1`..`F5`, or a cloned voice style like `david-deep`) and speed in the menu-bar popover.
4. Press **⌥⌘R** on any highlighted text, or type text directly into the "Type to speak" box to synthesize speech.

### Developer Smoke Testing

To run the on-device self-test logic at startup:
```bash
SUPERTONIC_SELFTEST=1 /path/to/tts-metal.app/Contents/MacOS/tts-metal
```

To run the stage-by-stage numerical validation against a reference dump:
```bash
ST_VALIDATE=1 /path/to/tts-metal.app/Contents/MacOS/tts-metal
```

---

## Credits

- **Supertonic 3** — Supertone (Supertone/supertonic-3 on Hugging Face).
- **tts-metal** is a custom, lightweight, dependency-free Metal adaptation of the Supertonic 3 model architecture.
