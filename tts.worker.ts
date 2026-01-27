// src/tts.worker.ts
import { KokoroTTS } from "kokoro-js";

// --- IndexedDB Cache Logic (Kept mostly the same) ---
const DB_NAME = 'KokoroModelCache';
const DB_VERSION = 1;
const STORE_NAME = 'modelFiles';

async function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = self.indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'url' });
      }
    };
    request.onsuccess = (event) => resolve((event.target as IDBOpenDBRequest).result);
    request.onerror = (event) => reject((event.target as IDBOpenDBRequest).error);
  });
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
    const store = transaction.objectStore(STORE_NAME);
    const getRequest = store.get(url);

    getRequest.onsuccess = () => {
      const cached = getRequest.result;
      if (cached) {
        // console.log(`[Cache] Serving ${url} from IndexedDB`);
        resolve(new Response(cached.blob, {
          status: 200,
          statusText: 'OK',
          headers: { 'Content-Type': cached.contentType || '' },
        }));
      } else {
        // console.log(`[Cache] Fetching ${url} from network`);
        const fetchPromise = originalFetch(url, init)
          .then(async (response) => {
            if (!response.ok) return response;
            const blob = await response.clone().blob();
            const contentType = response.headers.get('Content-Type') || '';
            const putDb = await openDatabase();
            const putTransaction = putDb.transaction(STORE_NAME, 'readwrite');
            const putStore = putTransaction.objectStore(STORE_NAME);
            putStore.put({ url, blob, contentType });
            return response;
          })
          .catch(reject);
        resolve(fetchPromise);
      }
    };
    getRequest.onerror = () => reject(getRequest.error);
  });
};

// --- OPTIMIZATION: WebGPU Detection ---
async function detectWebGPU() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (e) {
    return false;
  }
}

// --- Worker Logic ---

type WorkerMessage =
  | { type: 'init' }
  | { type: 'generate', text: string, lineIndex?: number, voice?: string };

const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
let tts: KokoroTTS | null = null;

self.addEventListener("message", async (e: MessageEvent<WorkerMessage>) => {
  const { type } = e.data;
  try {
    if (type === 'init') {
      if (tts) return;

      // 1. Detect environment
      const hasWebGPU = await detectWebGPU();
      const device = hasWebGPU ? "webgpu" : "wasm";

      // 2. Select optimization level
      // Mobile/CPU benefits significantly from q8 (8-bit quantization)
      // instead of fp32, reducing memory usage and calculation time.
      const dtype = device === "wasm" ? "q8" : "fp32";

      console.log(`[Worker] Initializing with device: ${device}, dtype: ${dtype}`);

      tts = await KokoroTTS.from_pretrained(model_id, {
        dtype: dtype,
        device: device,
      });

      self.postMessage({ status: 'ready', device, dtype });
    }
    else if (type === 'generate') {
      const { text, lineIndex, voice } = e.data;
      if (!tts) throw new Error("TTS not initialized");

      const audio = await tts.generate(text, {
        voice: voice || "af_bella",
        speed: 1.0,
      });

      self.postMessage({
        status: 'complete',
        audio: audio.audio, // Float32Array
        text,
        lineIndex
      });
    }
  } catch (err: any) {
    console.error(err);
    self.postMessage({
      status: 'error',
      error: err.message
    });
  }
});
