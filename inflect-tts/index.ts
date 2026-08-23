/**
 * inflect-tts — Run Inflect-Micro-v2 in the browser with pure WebGPU kernels.
 *
 * One function is all you need:
 *
 * ```typescript
 * import { textToSpeech } from './inflect-tts/index.ts';
 *
 * const wav = await textToSpeech('Hello, world!');
 * const audio = new Audio(URL.createObjectURL(wav));
 * audio.play();
 * ```
 */
import { InflectTTSEngine, splitText, boundaryPauseSeconds, edgeFade } from './engine.ts';
import { textToInputIds } from './frontend.ts';
import { float32ToWav } from './wav.ts';

export { InflectTTSEngine, splitText, boundaryPauseSeconds, edgeFade } from './engine.ts';
export { textToInputIds, normalizeText, textToPhonemes, cleanForTts, SYMBOLS } from './frontend.ts';
export { float32ToWav } from './wav.ts';
export { GraphExecutor } from './executor.ts';

const SAMPLE_RATE = 24000;

/** Per-engine singleton shared by all textToSpeech() calls. */
let sharedEngine: InflectTTSEngine | null = null;
let initPromise: Promise<InflectTTSEngine> | null = null;

/** Default graph URLs (official ONNX export of Inflect-Micro-v2). */
export const DEFAULT_DURATION_URL =
  'https://huggingface.co/owensong/Inflect-Micro-v2-ONNX/resolve/main/onnx/duration.onnx';
export const DEFAULT_DECODE_URL =
  'https://huggingface.co/owensong/Inflect-Micro-v2-ONNX/resolve/main/onnx/decode.onnx';

async function getEngine(
  durationUrl: string,
  decodeUrl: string,
  onProgress?: (stage: string) => void,
): Promise<InflectTTSEngine> {
  if (sharedEngine) return sharedEngine;
  if (initPromise) return initPromise;

  const promise = (async () => {
    const engine = new InflectTTSEngine();
    onProgress?.('Initializing WebGPU…');
    await engine.init();
    onProgress?.('Loading model (~38 MB)…');
    await engine.loadModel(durationUrl, decodeUrl);
    sharedEngine = engine;
    return engine;
  })();

  initPromise = promise;
  try {
    return await promise;
  } catch (err) {
    initPromise = null;
    throw err;
  }
}

/** Options for the `textToSpeech` function. */
export interface TextToSpeechOptions {
  /** Speaking speed multiplier. Default `1.0`. Range: 0.5 – 2.0. */
  speed?: number;
  /** Noise variation. Default `0.667`. Range: 0.0 – 1.0. */
  variation?: number;
  /** Seed for deterministic synthesis. Default `0`. */
  seed?: number;
  /** Duration graph URL. Defaults to the official HuggingFace export. */
  durationUrl?: string;
  /** Decode graph URL. */
  decodeUrl?: string;
  /** Progress callback with stage descriptions like `'Loading model…'`. */
  onProgress?: (stage: string) => void;
}

/**
 * Convert text to a WAV audio blob using Inflect-Micro-v2.
 * Long text is chunked at sentence boundaries with natural pauses.
 *
 * @example
 * ```typescript
 * import { textToSpeech } from './inflect-tts/index.ts';
 * const wav = await textToSpeech('Slow and steady wins the race.', { speed: 0.9 });
 * new Audio(URL.createObjectURL(wav)).play();
 * ```
 */
export async function textToSpeech(text: string, options: TextToSpeechOptions = {}): Promise<Blob> {
  if (!text?.trim()) throw new Error('Text must be a non-empty string.');
  const { speed = 1.0, variation = 0.667, seed = 0, durationUrl = DEFAULT_DURATION_URL, decodeUrl = DEFAULT_DECODE_URL, onProgress } = options;

  const engine = await getEngine(durationUrl, decodeUrl, onProgress);

  const chunks = splitText(text);
  const pieces: Float32Array[] = [];

  for (let index = 0; index < chunks.length; index++) {
    if (index) {
      pieces.push(new Float32Array(Math.round(SAMPLE_RATE * boundaryPauseSeconds(chunks[index - 1]))));
    }
    onProgress?.(`Generating speech (${index + 1}/${chunks.length})…`);
    const { ids } = await textToInputIds(chunks[index]);
    const { waveform } = await engine.generate(ids, { speed, variation, seed: seed + index });
    pieces.push(edgeFade(waveform, SAMPLE_RATE));
  }

  let total = 0;
  for (const p of pieces) total += p.length;
  const waveform = new Float32Array(total);
  let offset = 0;
  for (const p of pieces) {
    waveform.set(p, offset);
    offset += p.length;
  }
  for (let j = 0; j < waveform.length; j++) {
    waveform[j] = Math.max(-1, Math.min(1, waveform[j]));
  }

  onProgress?.('Encoding WAV…');
  return float32ToWav(waveform, SAMPLE_RATE);
}
