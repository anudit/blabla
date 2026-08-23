// tts.worker.ts — Inflect-Micro-v2 TTS via pure WebGPU kernels (~38M, 24 kHz)
import { InflectTTSEngine, textToInputIds, edgeFade, splitText, boundaryPauseSeconds } from './inflect-tts/index.ts';

const ONNX_BASE = 'https://huggingface.co/owensong/Inflect-Micro-v2-ONNX/resolve/main/onnx';
const DURATION_URL = `${ONNX_BASE}/duration.onnx`;
const DECODE_URL = `${ONNX_BASE}/decode.onnx`;
const MODEL_DB = 'inflect-tts-models-v1';

// IDB helpers — IndexedDB stores raw ArrayBuffer so there are no CORS or
// redirect restrictions (unlike Cache Storage, which rejects cross-origin
// redirected responses from dedicated workers).
function openModelDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(MODEL_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore('models');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function idbGet(db: IDBDatabase, key: string): Promise<ArrayBuffer | undefined> {
  return new Promise((resolve, reject) => {
    const req = db.transaction('models', 'readonly').objectStore('models').get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function idbPut(db: IDBDatabase, key: string, buf: ArrayBuffer): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('models', 'readwrite');
    tx.objectStore('models').put(buf, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function fetchCached(db: IDBDatabase, remoteUrl: string, cacheKey: string): Promise<string> {
  const cached = await idbGet(db, cacheKey);
  if (cached) return URL.createObjectURL(new Blob([cached], { type: 'application/octet-stream' }));
  const res = await fetch(remoteUrl);
  if (!res.ok) throw new Error(`Failed to fetch ${remoteUrl}: ${res.status}`);
  const buf = await res.arrayBuffer();
  await idbPut(db, cacheKey, buf);
  return URL.createObjectURL(new Blob([buf], { type: 'application/octet-stream' }));
}

let engine: InflectTTSEngine | null = null;
let generateChain: Promise<void> = Promise.resolve();

async function runGenerate(text: string, lineIndex: number | undefined, speed: number, notify: boolean) {
  if (!engine) throw new Error('TTS not initialized');
  const sr = engine.sampleRateValue;
  const chunks = splitText(text);
  const pieces: Float32Array[] = [];
  for (let i = 0; i < chunks.length; i++) {
    if (i) pieces.push(new Float32Array(Math.round(sr * boundaryPauseSeconds(chunks[i - 1]))));
    let ids: number[];
    try {
      ({ ids } = await textToInputIds(chunks[i]));
    } catch {
      continue;
    }
    if (ids.length < 3) continue;
    const { waveform } = await engine.generate(ids, { speed });
    pieces.push(edgeFade(waveform, sr));
  }
  if (!notify) return;
  if (!pieces.length) throw new Error('The text frontend produced no speakable tokens.');
  let total = 0;
  for (const p of pieces) total += p.length;
  const audio = new Float32Array(total);
  let off = 0;
  for (const p of pieces) { audio.set(p, off); off += p.length; }
  self.postMessage({ status: 'complete', audio, text, lineIndex }, [audio.buffer]);
}

self.addEventListener('message', (e: MessageEvent<any>) => {
  const { type } = e.data;
  generateChain = generateChain.then(async () => {
    try {
      if (type === 'init') {
        if (engine) return;

        engine = new InflectTTSEngine();
        self.postMessage({ status: 'loading', message: 'Initializing WebGPU...' });
        await engine.init();

        const db = await openModelDB();
        const alreadyCached = !!(await idbGet(db, 'duration.onnx'));
        self.postMessage({ status: 'loading', message: alreadyCached ? 'Loading Model...' : 'Downloading 38M Model...' });

        const [durationUrl, decodeUrl] = await Promise.all([
          fetchCached(db, DURATION_URL, 'duration.onnx'),
          fetchCached(db, DECODE_URL, 'decode.onnx'),
        ]);

        self.postMessage({ status: 'models_cached' });

        await engine.loadModel(durationUrl, decodeUrl);

        // Compile shaders, JIT the duration graph, and load espeak WASM
        // before the UI can post real lines — overlapping generates share
        // GPU buffers and garbled the first spoken sentence.
        self.postMessage({ status: 'loading', message: 'Warming up…' });
        await runGenerate('Warm up.', -1, 1.0, false);

        console.log('[InflectTTS Worker] Ready');
        self.postMessage({ status: 'ready', device: navigator.gpu ? 'webgpu' : 'cpu', dtype: 'fp32' });
      } else if (type === 'generate') {
        const { text, lineIndex, speed = 1.0 } = e.data;
        try {
          await runGenerate(text, lineIndex, speed, true);
        } catch (err) {
          // A bad line can leave GPU activations in the buffer pool; rebuild
          // once so the rest of the book is not permanently garbled.
          console.warn('[InflectTTS Worker] generate failed, recovering GPU', err);
          await engine?.recoverGpu();
          await runGenerate(text, lineIndex, speed, true);
        }
      }
    } catch (err: any) {
      console.error('[InflectTTS Worker]', err);
      if (type === 'init') engine = null;
      const isFetchError = err instanceof TypeError && err.message.toLowerCase().includes('fetch');
      self.postMessage({
        status: 'error',
        error: err?.message || String(err) || 'Unknown error',
        fetchFailed: isFetchError,
      });
    }
  });
});
