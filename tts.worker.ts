// src/tts.worker.ts
import { KokoroTTS } from "kokoro-js";

// Model files are cached automatically by transformers.js using Cache Storage —
// no custom IndexedDB interceptor needed.

// --- HELPER: WebGPU Detection ---
async function detectWebGPU() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (e) {
    return false;
  }
}

// --- HELPER: Environment Checks ---
function getSystemCapabilities() {
  const ua = navigator.userAgent.toLowerCase();

  // 1. Detect Mobile/Tablet
  const isMobile = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(ua);

  // 2. Detect RAM (Note: Browser caps this at 8GB for privacy)
  // If undefined, we assume 4GB to be safe.
  const memory = (navigator as any).deviceMemory || 4;

  return { isMobile, memory };
}

// --- Worker Logic ---

type WorkerMessage =
  | { type: 'init' }
  | { type: 'generate', text: string, lineIndex?: number, voice?: string, speed?: number };

const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
let tts: KokoroTTS | null = null;

self.addEventListener("message", async (e: MessageEvent<WorkerMessage>) => {
  const { type } = e.data;
  try {
    if (type === 'init') {
      if (tts) return;

      const hasWebGPU = await detectWebGPU();
      const { isMobile, memory } = getSystemCapabilities();

      // DECISION LOGIC:
      // We force WASM/q8 if:
      // 1. WebGPU is missing.
      // 2. OR Device is Mobile (Pixel 8 has WebGPU but crashes on fp32 models due to driver instability).
      // 3. OR RAM is < 8GB (Low spec desktop/laptop).
      const forceWasm = !hasWebGPU || isMobile || memory < 8;

      const device = "webgpu";

      // q8 is standard for wasm. fp32 is standard for webgpu.
      const dtype =  "fp32";

      console.log(`[Worker] Init: Mobile=${isMobile}, RAM=${memory}GB, WebGPU=${hasWebGPU}`);
      console.log(`[Worker] Selected: Device=${device}, Dtype=${dtype}`);

      tts = await KokoroTTS.from_pretrained(model_id, {
        dtype: dtype,
        device: device,
      });

      self.postMessage({ status: 'ready', device, dtype });
    }
    else if (type === 'generate') {
      const { text, lineIndex, voice, speed } = e.data;
      if (!tts) throw new Error("TTS not initialized");

      const audio = await tts.generate(text, {
        voice: voice || "af_bella",
        // Use native model speed so pitch stays natural at all rates.
        // Raising playbackRate in Web Audio shifts pitch up; passing speed
        // here lets the model handle tempo while preserving pitch.
        speed: speed ?? 1.0,
      });

      self.postMessage({
        status: 'complete',
        audio: audio.audio,
        text,
        lineIndex
      }, [audio.audio.buffer]);
    }
  } catch (err: any) {
    console.error(err);
    self.postMessage({
      status: 'error',
      error: err.message
    });
  }
});
