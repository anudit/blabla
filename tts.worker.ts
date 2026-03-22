// tts.worker.ts — Kitten TTS WebGPU (kitten-tts-micro-0.8, 40M)
import { KittenTTSEngine, textToInputIds } from 'kitten-tts-webgpu';

const ONNX_URL = "https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/kitten_tts_micro_v0_8.onnx";
const VOICES_URL = "https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/voices.npz";

let engine: KittenTTSEngine | null = null;

self.addEventListener('message', async (e: MessageEvent<any>) => {
  const { type } = e.data;
  try {
    if (type === 'init') {
      if (engine) return;
      
      if (!navigator.gpu) {
        throw new Error("WebGPU is not supported in this browser.");
      }

      engine = new KittenTTSEngine();
      self.postMessage({ status: 'loading', message: 'Initializing WebGPU...' });
      await engine.init();
      
      self.postMessage({ status: 'loading', message: 'Downloading 40M Model...' });
      await engine.loadModel(ONNX_URL, VOICES_URL);
      
      console.log("[KittenTTS Worker] Ready: device=webgpu, dtype=fp32");
      self.postMessage({ status: 'ready', device: 'webgpu', dtype: 'fp32' });
    }

    else if (type === 'generate') {
      const { text, lineIndex, voice = 'Bella', speed = 1.0 } = e.data;
      if (!engine) throw new Error('TTS not initialized');

      // 1. Text to phonemes/ids
      const { ids } = await textToInputIds(text);
      
      // 2. Generate waveform
      const { waveform } = await engine.generate(ids, voice, speed, text.length);

      // Post-process: trim last 5000 samples for better sentence boundaries (same as original code)
      const audio = waveform.slice(0, Math.max(0, waveform.length - 5000));

      self.postMessage({ status: 'complete', audio, text, lineIndex }, [audio.buffer]);
    }
  } catch (err: any) {
    console.error('[KittenTTS Worker]', err);
    self.postMessage({ status: 'error', error: err?.message || String(err) || 'Unknown error' });
  }
});
