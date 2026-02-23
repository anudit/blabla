// src/tts.worker.ts
import { KokoroTTS } from "kokoro-js";

// --- IndexedDB Cache Logic (Kept mostly the same) ---
const DB_NAME = 'KokoroModelCache';
const DB_VERSION = 1;
const STORE_NAME = 'modelFiles';

let _dbPromise: Promise<IDBDatabase> | null = null;
function openDatabase(): Promise<IDBDatabase> {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise((resolve, reject) => {
    const request = self.indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'url' });
      }
    };
    request.onsuccess = (event) => resolve((event.target as IDBOpenDBRequest).result);
    request.onerror = (event) => { _dbPromise = null; reject((event.target as IDBOpenDBRequest).error); };
  });
  return _dbPromise;
}

const originalFetch = self.fetch.bind(self);
self.fetch = async function (input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  if (typeof input !== 'string' || !input.includes('huggingface.co')) {
    return originalFetch(input, init);
  }
  const url = input;
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    // Prevent hanging promises if the transaction aborts unexpectedly
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(new Error('IDB transaction aborted'));

    const store = transaction.objectStore(STORE_NAME);
    const getRequest = store.get(url);

    getRequest.onsuccess = () => {
      const cached = getRequest.result;
      if (cached) {
        resolve(new Response(cached.blob, {
          status: 200,
          statusText: 'OK',
          headers: { 'Content-Type': cached.contentType || '' },
        }));
      } else {
        const fetchPromise = originalFetch(url, init)
          .then(async (response) => {
            if (!response.ok) return response;
            const blob = await response.clone().blob();
            const contentType = response.headers.get('Content-Type') || '';
            const putDb = await openDatabase();
            const putTransaction = putDb.transaction(STORE_NAME, 'readwrite');
            putTransaction.onerror = (err) => console.error('[Cache] IDB write failed', err);
            putTransaction.objectStore(STORE_NAME).put({ url, blob, contentType });
            return response;
          })
          .catch(reject);
        resolve(fetchPromise);
      }
    };
    getRequest.onerror = () => reject(getRequest.error);
  });
};
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
