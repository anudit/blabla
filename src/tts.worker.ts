// src/tts.worker.ts
import { KokoroTTS } from "kokoro-js";

// We define a simple message protocol
type WorkerMessage =
  | { type: 'init' }
  | { type: 'generate', text: string };

const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
let tts: KokoroTTS | null = null;

// Listen for messages from the main thread
self.addEventListener("message", async (e: MessageEvent<WorkerMessage>) => {
  const { type } = e.data;

  try {
    if (type === 'init') {
      if (tts) return; // Already initialized

      // Load the model
      // We force "wasm" to ensure it runs in the browser securely
      tts = await KokoroTTS.from_pretrained(model_id, {
        dtype: "q8", // q8 is faster for browser/wasm
        device: "webgpu",
      });

      self.postMessage({ status: 'ready' });
    }
    else if (type === 'generate') {
      const { text } = e.data as { type: 'generate', text: string };

      if (!tts) {
        throw new Error("TTS not initialized");
      }

      // Generate audio
      const audio = await tts.generate(text, {
        voice: "af_bella", // You can make this dynamic if needed
        speed: 1.0,
      });

      // Send the raw audio buffer back to the main thread
      // We use 'audio.audio' to get the Float32Array
      self.postMessage({
        status: 'complete',
        audio: audio.audio,
        text
      });
    }
  } catch (err: any) {
    self.postMessage({
      status: 'error',
      error: err.message
    });
  }
});
