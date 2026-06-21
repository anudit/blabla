/**
 * OCR Web Worker — runs PaddleOCR-WebGPU text detection + recognition off the
 * main thread so heavy GPU command submission, readback, and CTC decoding never
 * hang the UI.
 *
 * The paddleocr-webgpu package was written for a DOM context: it uses
 * `document.createElement('canvas')` and `instanceof HTMLCanvasElement`. Workers
 * have neither, so we provide minimal shims that map those onto `OffscreenCanvas`
 * before any engine method runs.
 */

// Must be first: installs OffscreenCanvas-based DOM shims before the engine
// module (imported below) is evaluated.
import './ocr-shims';
import { PaddleOcrEngine, TextDetectionEngine } from 'paddleocr-webgpu';
import { segmentTextLines } from './ocr-segment';

const MODEL_CACHE = 'paddleocr-cache-v1';

const REC_BASE = 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main';
const DET_BASE = 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main';

type ProgressFn = (pct: number) => void;

async function fetchBlob(url: string, onProgress: ProgressFn): Promise<Blob> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);

  const totalBytes = parseInt(response.headers.get('Content-Length') ?? '0', 10);
  const reader = response.body?.getReader();
  if (!reader) {
    onProgress(1);
    return await response.blob();
  }

  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.byteLength;
      if (totalBytes > 0) onProgress(received / totalBytes);
    }
  }
  onProgress(1);
  return new Blob(chunks as BlobPart[]);
}

/**
 * Fetch a model, caching it in the CacheStorage when available. If the Cache
 * API is unavailable or throws (private mode, quota, insecure context, …) we
 * fall back to a plain network fetch instead of failing the whole OCR flow.
 */
async function loadWithCache(url: string, onProgress: ProgressFn): Promise<string> {
  try {
    const cache = await caches.open(MODEL_CACHE);
    const cached = await cache.match(url);
    if (cached) {
      onProgress(1);
      return URL.createObjectURL(await cached.blob());
    }
    const blob = await fetchBlob(url, onProgress);
    try {
      await cache.put(url, new Response(blob, {
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': String(blob.size),
        },
      }));
    } catch (e) {
      console.warn('[OCR worker] cache.put failed, continuing without cache:', e);
    }
    return URL.createObjectURL(blob);
  } catch (e) {
    console.warn('[OCR worker] Cache API unavailable, fetching without cache:', e);
    const blob = await fetchBlob(url, onProgress);
    return URL.createObjectURL(blob);
  }
}

// ── Engine state ──────────────────────────────────────────────────────────
let recEngine: PaddleOcrEngine | null = null;
let detEngine: TextDetectionEngine | null = null;
let initPromise: Promise<void> | null = null;

function post(msg: any, transfer?: Transferable[]) {
  (self as any).postMessage(msg, transfer ?? []);
}

async function initEngines() {
  post({ type: 'progress', stage: 'Downloading OCR recognition model (20MB)...', pct: 0 });
  const recModelUrl = await loadWithCache(`${REC_BASE}/inference.onnx`, (pct) => {
    post({ type: 'progress', stage: 'Downloading OCR recognition model (20MB)...', pct: pct * 0.25 });
  });

  post({ type: 'progress', stage: 'Downloading OCR detection model...', pct: 0.25 });
  const detModelUrl = await loadWithCache(`${DET_BASE}/inference.onnx`, (pct) => {
    post({ type: 'progress', stage: 'Downloading OCR detection model...', pct: 0.25 + pct * 0.6 });
  });

  post({ type: 'progress', stage: 'Downloading OCR character dictionary...', pct: 0.85 });
  const vocabUrl = await loadWithCache(`${REC_BASE}/inference.yml`, () => {});

  post({ type: 'progress', stage: 'Initializing WebGPU pipeline and compiling WGSL shaders...', pct: 0.95 });

  const rec = new PaddleOcrEngine();
  await rec.init({
    modelUrl: recModelUrl,
    vocabUrl,
    channelOrder: 'bgr',
    maxWidth: 2048,
  });
  recEngine = rec;

  const det = new TextDetectionEngine();
  await det.init({ modelUrl: detModelUrl });
  detEngine = det;
}

function ensureInit(): Promise<void> {
  if (!initPromise) {
    initPromise = initEngines().catch((err) => {
      initPromise = null;
      throw err;
    });
  }
  return initPromise;
}

// ── Detection box helpers ──────────────────────────────────────────────────
interface LineBox { x0: number; y0: number; x1: number; y1: number; score: number; }

function iou(a: LineBox, b: LineBox): number {
  const ix0 = Math.max(a.x0, b.x0);
  const iy0 = Math.max(a.y0, b.y0);
  const ix1 = Math.min(a.x1, b.x1);
  const iy1 = Math.min(a.y1, b.y1);
  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  const areaA = Math.max(0, a.x1 - a.x0) * Math.max(0, a.y1 - a.y0);
  const areaB = Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

/** Suppress overlapping detection boxes so the same line isn't recognized twice. */
function dedupeBoxes(boxes: LineBox[], iouThresh = 0.5): LineBox[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: LineBox[] = [];
  for (const box of sorted) {
    if (kept.some((k) => iou(box, k) > iouThresh)) continue;
    kept.push(box);
  }
  return kept;
}

// ── Page processing ────────────────────────────────────────────────────────
async function processPage(bitmap: ImageBitmap): Promise<string> {
  await ensureInit();
  const W = bitmap.width;
  const H = bitmap.height;

  const canvas = new OffscreenCanvas(W, H);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  let lines: LineBox[] = [];
  try {
    const detResult = await detEngine!.detect(canvas as any);
    lines = detResult.boxes.map((box: any) => {
      const xs = box.points.map((p: number[]) => p[0]);
      const ys = box.points.map((p: number[]) => p[1]);
      return {
        x0: Math.max(0, Math.floor(Math.min(...xs))),
        x1: Math.min(W, Math.ceil(Math.max(...xs))),
        y0: Math.max(0, Math.floor(Math.min(...ys))),
        y1: Math.min(H, Math.ceil(Math.max(...ys))),
        score: typeof box.score === 'number' ? box.score : 1,
      };
    });
    lines = dedupeBoxes(lines);
    console.log(`[OCR worker] Detection found ${lines.length} boxes (after dedup) in ${detResult.inferenceTimeMs.toFixed(1)}ms`);
  } catch (detErr) {
    console.warn('[OCR worker] detection failed, falling back to segmentTextLines:', detErr);
    lines = segmentTextLines(canvas as any).map((l) => ({ ...l, score: 1 }));
  }

  if (lines.length === 0) {
    const res = await recEngine!.recognize(canvas as any);
    return res.text || '';
  }

  const crops = lines.map(({ x0, y0, x1, y1 }) => {
    const clW = Math.max(1, x1 - x0);
    const chH = Math.max(1, y1 - y0);
    const lc = new OffscreenCanvas(clW, chH);
    const lctx = lc.getContext('2d')!;
    lctx.fillStyle = '#FFFFFF';
    lctx.fillRect(0, 0, clW, chH);
    const srcW = Math.min(clW, W - x0);
    const srcH = Math.min(chH, H - y0);
    if (srcW > 0 && srcH > 0) {
      lctx.drawImage(canvas, x0, y0, srcW, srcH, 0, 0, srcW, srcH);
    }
    return lc;
  });

  const results = await recEngine!.recognizePipelined(crops as any);
  const mapped = results
    .map((r: any, idx: number) => ({ text: r.text.trim(), y0: lines[idx]?.y0 ?? 0 }))
    .filter((item) => item.text);
  mapped.sort((a, b) => a.y0 - b.y0);
  return mapped.map((item) => item.text).join('\n');
}

// ── Message handling ────────────────────────────────────────────────────────
self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  if (msg.type === 'init') {
    try {
      await ensureInit();
      post({ type: 'ready' });
    } catch (err: any) {
      post({ type: 'error', message: err?.message || String(err) });
    }
  } else if (msg.type === 'process') {
    const { reqId, pageNum, bitmap } = msg;
    try {
      const pageText = await processPage(bitmap);
      post({ type: 'result', reqId, pageNum, pageText });
    } catch (err: any) {
      try { bitmap?.close?.(); } catch {}
      post({ type: 'error', reqId, pageNum, message: err?.message || String(err) });
    }
  }
};
