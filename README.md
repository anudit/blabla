# tts-metal

A native macOS **text-to-speech suite** that runs the **Supertonic 3** latent flow-matching model entirely on-device using **Metal compute shaders** — no Python, no ONNX Runtime, no Core ML.

Two surfaces, one engine:

1. **Menu-bar selection reader** — select text anywhere in the system, press **⌥⌘R**, and it speaks the selection.
2. **BlaBla document reader** — a full private AI reading window (ported from the BlaBla web app): drop a PDF / EPUB / MOBI / DOCX / Markdown / TXT file or paste a URL, and it reads the document aloud with sentence karaoke highlighting, auto-scroll, TOC navigation, auto-resume bookmarks, a floating mini player, and media-key control. Everything stays on-device.

```
selection / document ──▶ Loaders ──▶ Sentence stream ──▶ Normalizer ──▶ Supertonic 3 (Metal) ──▶ Resampler ──▶ AVAudioEngine
                        (PDF/EPUB/   (click-to-jump ids)   (numbers,        44.1 kHz audio          (Lanczos)    gapless queue
                         MOBI/DOCX/                        money, dates,
                         MD/URL)                           abbreviations)
```

---

## Features

- **Pure Metal Compute**: Executes the full 99.2M parameter Supertonic 3 architecture directly on Apple Silicon GPUs using custom Metal kernels.
- **Multilingual Support**: Synthesizes speech across 31 languages (including English, Korean, Japanese, Arabic, German, Spanish, French, Hindi, and more).
- **Zero-Shot Voice Cloning**: Dynamically estimates and applies speaker embeddings from reference voice styles defined in JSON configs.
- **Expression Tags**: Supports expressive synthesis using prompt tags like `<laugh>`, `<breath>`, and `<sigh>`.
- **High-Fidelity Output**: Generates native 44.1 kHz full-band audio, which is resampled on-the-fly to 48 kHz for gapless playback using a high-quality Lanczos sinc filter.

### BlaBla document reader (ported from the BlaBla web app)

| Feature | Details |
|---|---|
| **Formats** | PDF (PDFKit text layer), EPUB (spine + nav/NCX TOC), MOBI/AZW (PalmDOC LZ77), DOCX, Markdown (frontmatter, code, tables, lists), TXT, URL fetch, clipboard paste |
| **Reader UI** | Rendered blocks, click any sentence to jump playback, sentence + word karaoke highlighting, auto-scroll tracking, scroll-to-current button, TOC outline sidebar |
| **Playback** | 3-sentence look-ahead prefetch, sliding-window backpressure, gapless tagged scheduling, punctuation-aware pauses (`. , : ; ! ?`) |
| **Text frontend** | Abbreviations, money, dates, times, phone numbers, versions, ordinals, dotted acronyms, big-number expansion |
| **Resume** | Auto-save bookmarks (≤20 history entries, progress bars) keyed by file identity/URL |
| **System integration** | Now Playing + media keys (play/pause, ±1 sentence) via MPRemoteCommandCenter, floating always-on-top mini player |
| **Settings** | Voice (M1–M5/F1–F5/david-deep), speed 1×–2×, volume, font size 0.8–1.6, 6 reader themes, test voice, reset document |

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
| **Supertonic 3 (Metal)** | **~720 ms** | 5.5 s @ 44.1 kHz | **~7.6×** (at 8 steps) |

### Stage Breakdown

| Stage | Latency | Notes |
|---|---:|---|
| **Duration Predictor** | ~18 ms | Lightweight MLP |
| **Text Encoder** | ~114 ms | ConvNeXt + Relative-Position Attention |
| **Flow Matching Loop** | **~570 ms** | 8 steps of the 64M parameter vector field, cond+uncond batched |
| **Vocoder** | ~47 ms | Causal Convolutions & ISTFT head |

> **Realtime Factor (RTF)** = (seconds of audio generated) ÷ (synthesis latency). RTF ~7.6×
> means 1 second of speech is synthesized in ~130 ms. The flow-matching ODE steps are a
> speed-to-quality control knob (fewer steps = faster).

The flow-matching loop is **entirely on-GPU**: the ConvNeXt backbone, both cross-attentions
(RoPE→text, tanh→style — projections and softmax included), the time conditioning, and the
CFG Euler step all run as Metal kernels, and the conditional/unconditional passes are batched
together (`M = 2·L`) so the small latent length still fills the GPU. The ODE state stays
resident across all 8 steps — no per-step CPU round-trip. The pointwise matmuls use a
register-blocked, activation-fused GEMM. Correctness is checked by the `ST_VALIDATE` harness
(`vfield[0] corr=1.0000`, bit-identical to the reference).

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
