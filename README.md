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

Benchmarks measured on Apple Silicon (Release build) for an 84-character English test sentence:

### End-to-End Latency

| Pipeline | End-to-End Time | Audio Generated | Realtime Factor (RTF) |
|---|---:|---:|---:|
| **Supertonic 3 (Metal)** | **1,205 ms** | 5.43 s @ 44.1 kHz | **4.5×** (at 8 steps) |

### Stage Breakdown

| Stage | Latency | Notes |
|---|---:|---|
| **Duration Predictor** | 1.2 ms | Lightweight MLP |
| **Text Encoder** | 8.0 ms | ConvNeXt + Relative-Position Attention |
| **Flow Matching Loop** | **1,094 ms** | 8 steps of the 64M parameter vector field (137 ms/step) |
| **Vocoder** | 76.0 ms | Causal Convolutions & ISTFT head |
| **Total** | **1,179 ms** | (Excluding audio enqueue overhead) |

> **Realtime Factor (RTF)** = (seconds of audio generated) ÷ (synthesis latency). An RTF of 4.5× means 1 second of speech is synthesized in ~220 ms. The flow-matching ODE steps function as a speed-to-quality control knob (fewer steps = faster).

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
