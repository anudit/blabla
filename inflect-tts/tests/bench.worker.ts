/**
 * TEMPORARY browser-only benchmark + correctness harness for InflectFastEngine.
 *
 * Two jobs in one run, because bun/Node has no WebGPU so this has to live in a
 * page worker (see fast-test.worker.ts, same trick):
 *
 *  1. correctness — a short token sequence decoded by both the validated CPU
 *     GraphExecutor and the GPU fast path, diffed sample by sample.
 *  2. timing — a realistic sentence-length sequence decoded on the GPU only,
 *     repeated so the numbers aren't a single noisy sample.
 *
 * Usage: served via the /bench.worker.js dev route, driven from /bench.
 */
import { GraphExecutor } from '../executor.ts';
import { InflectFastEngine } from '../fast-engine.ts';
import { InflectTTSEngine } from '../engine.ts';
import type { CpuTensor } from '../tensor.ts';

function gaussianNoise(n: number, seed: number): Float32Array {
  let s = seed >>> 0 || 1;
  const next = () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const u1 = Math.max(next(), 1e-12);
    const u2 = next();
    const mag = Math.sqrt(-2 * Math.log(u1));
    out[j] = mag * Math.cos(2 * Math.PI * u2);
    if (++j < n) out[j] = mag * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

const DUR_URL = 'https://huggingface.co/owensong/Inflect-Micro-v2-ONNX/resolve/main/onnx/duration.onnx';
const DEC_URL = 'https://huggingface.co/owensong/Inflect-Micro-v2-ONNX/resolve/main/onnx/decode.onnx';

const SHORT_TOKENS = [3, 46, 66, 70, 70, 73, 3, 20, 89, 3];
/**
 * "The cat sat on the mat, quietly, and then it wandered off into the rain."
 * through textToInputIds() — a realistic sentence length for the reader.
 */
const LONG_TOKENS = [
  0, 81, 0, 83, 0, 16, 0, 53, 0, 156, 0, 72, 0, 62, 0, 16, 0, 61, 0, 156,
  0, 72, 0, 62, 0, 16, 0, 76, 0, 56, 0, 81, 0, 83, 0, 16, 0, 55, 0, 156,
  0, 72, 0, 62, 0, 3, 0, 16, 0, 53, 0, 65, 0, 156, 0, 43, 0, 102, 0, 83,
  0, 62, 0, 54, 0, 51, 0, 3, 0, 16, 0, 72, 0, 56, 0, 46, 0, 16, 0, 81,
  0, 156, 0, 86, 0, 56, 0, 16, 0, 102, 0, 62, 0, 16, 0, 65, 0, 156, 0, 76,
  0, 56, 0, 46, 0, 85, 0, 46, 0, 16, 0, 156, 0, 76, 0, 48, 0, 16, 0, 157,
  0, 102, 0, 56, 0, 62, 0, 135, 0, 16, 0, 81, 0, 83, 0, 16, 0, 123, 0, 156,
  0, 47, 0, 102, 0, 56, 0, 4, 0,
];

/** A short clause: "It was raining outside." */
const MID_TOKENS = [
  0, 60, 0, 88, 0, 16, 0, 90, 0, 86, 0, 100, 0, 16, 0, 123, 0, 156, 0, 61,
  0, 60, 0, 62, 0, 133, 0, 16, 0, 72, 0, 90, 0, 88, 0, 88, 0, 156, 0, 72,
  0, 61, 0, 4, 0,
];

function maxDiff(a: Float32Array, b: Float32Array): number {
  let mx = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(a[i] - b[i]));
  return mx;
}

/** Cheap order-sensitive checksum so two runs can be compared across reloads. */
function checksum(a: Float32Array): number {
  let h = 2166136261;
  for (let i = 0; i < a.length; i++) {
    h ^= Math.round(a[i] * 1e6) | 0;
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

async function durationOutputs(durBuf: ArrayBuffer, ids: number[]) {
  const tokens = new BigInt64Array(ids.length);
  ids.forEach((id, j) => { tokens[j] = BigInt(id); });
  const ex = new GraphExecutor(durBuf);
  await ex.init(null);
  return ex.run(
    new Map<string, CpuTensor>([
      ['tokens', { dims: [1, ids.length], type: 'i64', data: tokens }],
      ['lengths', { dims: [1], type: 'i64', data: new BigInt64Array([BigInt(ids.length)]) }],
      ['length_scale', { dims: [], type: 'f32', data: new Float32Array([1.0]) }],
    ]),
    ['m_p_exp', 'logs_p_exp', 'y_mask'],
  );
}

self.addEventListener('message', async (e: MessageEvent<any>) => {
  if (e.data?.type !== 'run') return;
  const post = (msg: any) => self.postMessage(msg);
  const runs: number = e.data.runs ?? 5;
  const skipCpu: boolean = !!e.data.skipCpu;
  try {
    post({ status: 'fetching' });
    const [durBuf, decBuf] = await Promise.all([
      fetch(DUR_URL).then((r) => r.arrayBuffer()),
      fetch(DEC_URL).then((r) => r.arrayBuffer()),
    ]);

    const adapter = await (navigator as any).gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const fast = new InflectFastEngine();
    await fast.init(device);
    await fast.loadWeights(decBuf);

    // ── 1. correctness vs the CPU interpreter ──
    let correctness: any = { skipped: true };
    if (!skipCpu) {
      post({ status: 'correctness' });
      const dur = await durationOutputs(durBuf, SHORT_TOKENS);
      const mP = dur.get('m_p_exp')!, logsP = dur.get('logs_p_exp')!, yMask = dur.get('y_mask')!;
      const T = mP.dims[2];
      const noise = gaussianNoise((mP.data as Float32Array).length, 0);

      const cpuEx = new GraphExecutor(decBuf);
      await cpuEx.init(null);
      const cpuOut = await cpuEx.run(
        new Map<string, CpuTensor>([
          ['m_p_exp', mP], ['logs_p_exp', logsP], ['y_mask', yMask],
          ['zp_noise', { dims: mP.dims, type: 'f32', data: noise }],
          ['noise_scale', { dims: [], type: 'f32', data: new Float32Array([0.667]) }],
        ]),
        ['waveform'],
      );
      const cpuWav = cpuOut.get('waveform')!.data as Float32Array;
      const gpuWav = await fast.generate(
        mP.data as Float32Array, logsP.data as Float32Array, yMask.data as Float32Array,
        noise, 0.667, T,
      );
      correctness = {
        T, samples: cpuWav.length,
        maxDiffVsCpu: maxDiff(cpuWav, gpuWav),
        gpuChecksum: checksum(gpuWav),
      };
      post({ status: 'correctness_done', correctness });
    }

    // ── 2. timing on a sentence-length sequence ──
    post({ status: 'timing' });
    const dur = await durationOutputs(durBuf, LONG_TOKENS);
    const mP = dur.get('m_p_exp')!, logsP = dur.get('logs_p_exp')!, yMask = dur.get('y_mask')!;
    const T = mP.dims[2];
    const noise = gaussianNoise((mP.data as Float32Array).length, 7);

    const times: number[] = [];
    let checksumLong = 0;
    let samples = 0;
    for (let i = 0; i <= runs; i++) {
      const t0 = performance.now();
      const wav = await fast.generate(
        mP.data as Float32Array, logsP.data as Float32Array, yMask.data as Float32Array,
        noise, 0.667, T,
      );
      const ms = performance.now() - t0;
      if (i === 0) { checksumLong = checksum(wav); samples = wav.length; } // discard warm-up
      else times.push(ms);
    }
    times.sort((a, b) => a - b);

    // ── 3. end-to-end InflectTTSEngine.generate() on alternating lengths ──
    // The decode-only loop above reuses one latent length, which is the best
    // case for the buffer pool. Real reading feeds a different length every
    // sentence, so alternate two of them here.
    post({ status: 'end_to_end' });
    const engine = new InflectTTSEngine();
    await engine.init();
    await engine.loadModelBuffers(durBuf.slice(0), decBuf.slice(0));
    const seqs = [LONG_TOKENS, SHORT_TOKENS, LONG_TOKENS.slice(0, 101), MID_TOKENS];
    const e2e: number[] = [];
    for (let i = 0; i < seqs.length * 3; i++) {
      const seq = seqs[i % seqs.length];
      const t0 = performance.now();
      await engine.generate(seq, { speed: 1.0, seed: i });
      const ms = performance.now() - t0;
      if (i >= seqs.length) e2e.push(ms); // discard the first pass over the set
    }
    const e2eSorted = e2e.slice().sort((a, b) => a - b);

    post({
      status: 'done',
      correctness,
      endToEnd: {
        perGenerate: e2e.map((t) => +t.toFixed(1)),
        median: +e2eSorted[e2eSorted.length >> 1].toFixed(1),
        total: +e2e.reduce((a, b) => a + b, 0).toFixed(1),
      },
      timing: {
        T, samples, audioSeconds: +(samples / 24000).toFixed(2),
        runs: times.map((t) => +t.toFixed(1)),
        median: +times[times.length >> 1].toFixed(1),
        min: +times[0].toFixed(1),
        checksum: checksumLong,
      },
    });
  } catch (err: any) {
    post({ status: 'error', error: err?.message || String(err), stack: err?.stack });
  }
});
