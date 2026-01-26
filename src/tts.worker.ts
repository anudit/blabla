// src/tts.worker.ts
import { KokoroTTS } from "kokoro-js";
// IndexedDB setup
const DB_NAME = 'KokoroModelCache';
const DB_VERSION = 1;
const STORE_NAME = 'modelFiles';

// Function to open the database (creates if not exists)
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

// Override global fetch to cache Hugging Face model files in IndexedDB
const originalFetch = self.fetch.bind(self);
self.fetch = async function (input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  if (typeof input !== 'string' || !input.includes('huggingface.co')) {
    // Non-HF URLs: use normal fetch
    return originalFetch(input, init);
  }

  const url = input;
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly'); // Use 'readonly' for get to avoid issues
    const store = transaction.objectStore(STORE_NAME);
    const getRequest = store.get(url);

    getRequest.onsuccess = () => {
      const cached = getRequest.result;
      if (cached) {
        console.log(`[Cache] Serving ${url} from IndexedDB`);
        resolve(new Response(cached.blob, {
          status: 200,
          statusText: 'OK',
          headers: { 'Content-Type': cached.contentType || '' },
        }));
      } else {
        // No cache: resolve with fetch + cache chain
        console.log(`[Cache] Fetching ${url} from network`);
        const fetchPromise = originalFetch(url, init)
          .then(async (response) => {
            if (!response.ok) {
              return response;
            }

            const blob = await response.clone().blob();
            const contentType = response.headers.get('Content-Type') || '';

            // Cache in a new transaction
            const putDb = await openDatabase();
            const putTransaction = putDb.transaction(STORE_NAME, 'readwrite');
            const putStore = putTransaction.objectStore(STORE_NAME);
            const putRequest = putStore.put({ url, blob, contentType });

            // Wait for put to complete (optional but ensures error handling)
            return new Promise<Response>((putResolve, putReject) => {
              putRequest.onsuccess = () => {
                console.log(`[Cache] Stored ${url} in IndexedDB`);
                putResolve(response);
              };
              putRequest.onerror = () => putReject(putRequest.error);
              putTransaction.onerror = () => putReject(putTransaction.error);
            });
          })
          .catch(reject);

        resolve(fetchPromise);
      }
    };

    getRequest.onerror = () => reject(getRequest.error);
    transaction.onerror = () => reject(transaction.error);
  });
};


// We define a simple message protocol
type WorkerMessage =
  | { type: 'init' }
  | { type: 'generate', text: string, lineIndex?: number, voice?: string };
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
        dtype: "fp32", // q8 is faster for browser/wasm
        device: "webgpu",
      });
      self.postMessage({ status: 'ready' });
    }
    else if (type === 'generate') {
      const { text, lineIndex, voice } = e.data;
      if (!tts) {
        throw new Error("TTS not initialized");
      }
      // Generate audio
      const audio = await tts.generate(text, {
        voice: voice || "af_bella", // You can make this dynamic if needed
        speed: 1.0,
      });
      // Send the raw audio buffer back to the main thread
      // We use 'audio.audio' to get the Float32Array
      self.postMessage({
        status: 'complete',
        audio: audio.audio,
        text,
        lineIndex
      });
    }
  } catch (err: any) {
    self.postMessage({
      status: 'error',
      error: err.message
    });
  }
});
