/**
 * WebGPU inference engine for Inflect-Micro-v2 (VITS, single speaker, 24 kHz).
 *
 * Same API surface as KittenTTSEngine: init() / loadModel() / generate().
 *
 * duration.onnx (text encoder + duration predictor + length regulation)
 * always runs through the generic ONNX-graph interpreter (executor.ts) —
 * it's a small fraction of total cost (~0.5s of a multi-second generate())
 * so hand-specializing it isn't worth the correctness risk.
 *
 * decode.onnx (flow + HiFi-GAN vocoder) — where ~98% of the cost lives —
 * runs through the hand-specialized InflectFastEngine (fast-engine.ts) when
 * WebGPU is available, falling back to the generic CPU interpreter otherwise.
 * See /Users/anudit/.claude/plans/floofy-mapping-allen.md for the rationale.
 */
import { GraphExecutor } from './executor.ts';
import { InflectFastEngine } from './fast-engine.ts';
import type { CpuTensor } from './tensor.ts';

// See executor.ts — `process` is undefined in the browser worker.
const TTS_DEBUG =
  typeof process !== 'undefined' && !!process.env?.TTS_DEBUG;

export interface GenerateOptions {
  /** Speaking speed multiplier. Default `1.0`. length_scale = 1/speed. */
  speed?: number;
  /** Noise variation 0..1. Default `0.667` (matches reference implementation). */
  variation?: number;
  /** Seed for deterministic synthesis. Default `0`. */
  seed?: number;
}

export class InflectTTSEngine {
  private device: any = null;
  private duration!: GraphExecutor;
  private decode: GraphExecutor | null = null;
  private fastDecode: InflectFastEngine | null = null;
  private durationBuf: ArrayBuffer | null = null;
  private decodeBuf: ArrayBuffer | null = null;
  private sampleRate = 24000;

  /** Request a WebGPU device. Must be called before loadModel(). */
  async init(): Promise<void> {
    if (!navigator.gpu) {
      console.warn('[InflectTTS] WebGPU not available — falling back to CPU inference');
      this.device = null;
      return;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU not available');
    this.device = await adapter.requestDevice();
    this.device.lost?.then?.((info: any) => {
      console.error(`[InflectTTS] Device lost: ${info.reason} — ${info.message}`);
      window.dispatchEvent(new CustomEvent('webgpu-device-lost', { detail: info }));
    });
    console.log('[InflectTTS] WebGPU device initialized');
  }

  /** Fetch and load both ONNX graphs. Accepts URLs or blob URLs. */
  async loadModel(durationUrl: string, decodeUrl: string): Promise<void> {
    const [durationBuf, decodeBuf] = await Promise.all([
      fetch(durationUrl).then((r) => r.arrayBuffer()),
      fetch(decodeUrl).then((r) => r.arrayBuffer()),
    ]);
    await this.loadModelBuffers(durationBuf, decodeBuf);
  }

  /** Load pre-fetched graph buffers. */
  async loadModelBuffers(durationBuf: ArrayBuffer, decodeBuf: ArrayBuffer): Promise<void> {
    this.durationBuf = durationBuf;
    this.decodeBuf = decodeBuf;
    this.duration = new GraphExecutor(durationBuf);
    await this.duration.init(this.device);
    if (this.device) {
      this.fastDecode = new InflectFastEngine();
      await this.fastDecode.init(this.device);
      await this.fastDecode.loadWeights(decodeBuf);
    } else {
      this.decode = new GraphExecutor(decodeBuf);
      await this.decode.init(this.device);
    }
    console.log(`[InflectTTS] Models loaded (${this.device ? 'webgpu' : 'cpu'} execution)`);
  }

  /**
   * Deterministic standard-normal noise matching numpy's default_rng(seed)
   * closely enough for natural variation (not bit-exact with numpy PCG64).
   */
  private gaussianNoise(n: number, seed: number): Float32Array {
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
      // Box-Muller pair
      out[j] = mag * Math.cos(2 * Math.PI * u2);
      if (++j < n) out[j] = mag * Math.sin(2 * Math.PI * u2);
    }
    return out;
  }

  /** Run TTS inference for one token sequence. Returns a 24 kHz waveform. */
  async generate(
    inputIds: number[],
    options: GenerateOptions = {},
  ): Promise<{ waveform: Float32Array }> {
    if (!this.duration || !(this.decode || this.fastDecode)) throw new Error('Engine not loaded — call loadModel() first');
    this.duration.purgePool();
    this.fastDecode?.purgePool();
    const { speed = 1.0, variation = 0.667, seed = 0 } = options;
    if (!inputIds.length) throw new Error('Text must be a non-empty string.');

    const tokensData = new BigInt64Array(inputIds.length);
    inputIds.forEach((id, j) => { tokensData[j] = BigInt(Math.round(id)); });

    const inputs = new Map<string, CpuTensor>([
      ['tokens', { dims: [1, inputIds.length], type: 'i64', data: tokensData }],
      ['lengths', { dims: [1], type: 'i64', data: new BigInt64Array([BigInt(inputIds.length)]) }],
      ['length_scale', { dims: [], type: 'f32', data: new Float32Array([1 / speed]) }],
    ]);

    let durOut: Map<string, CpuTensor>;
    try {
      durOut = await this.duration.run(inputs, ['m_p_exp', 'logs_p_exp', 'y_mask']);
    } catch (err) {
      this.duration.purgePool();
      this.fastDecode?.purgePool();
      throw err;
    }
    const mP = durOut.get('m_p_exp')!;
    const logsP = durOut.get('logs_p_exp')!;
    const yMask = durOut.get('y_mask')!;

    if (TTS_DEBUG) {
      const stat = (name: string, t: CpuTensor) => {
        const a = t.data as Float32Array;
        let mx = 0;
        for (let j = 0; j < a.length; j++) mx = Math.max(mx, Math.abs(a[j]));
        console.log(`[TTS_DEBUG] ${name} dims=${JSON.stringify(t.dims)} maxAbs=${mx}`);
      };
      stat('m_p_exp', mP); stat('logs_p_exp', logsP); stat('y_mask', yMask);
    }
    const zpNoise = this.gaussianNoise(mP.data.length, seed);
    const T = mP.dims[2];

    let waveform: Float32Array;
    if (this.fastDecode) {
      waveform = await this.fastDecode.generate(
        mP.data as Float32Array, logsP.data as Float32Array, yMask.data as Float32Array,
        zpNoise, variation, T,
      );
    } else {
      const decOut = await this.decode!.run(
        new Map<string, CpuTensor>([
          ['m_p_exp', mP],
          ['logs_p_exp', logsP],
          ['y_mask', yMask],
          ['zp_noise', { dims: mP.dims, type: 'f32', data: zpNoise }],
          ['noise_scale', { dims: [], type: 'f32', data: new Float32Array([variation]) }],
        ]),
        ['waveform'],
      );
      const wavFull = decOut.get('waveform')!;
      waveform = new Float32Array((wavFull.data as Float32Array).slice().buffer);
    }
    for (let j = 0; j < waveform.length; j++) {
      if (!Number.isFinite(waveform[j])) {
        this.duration.purgePool();
        this.fastDecode?.purgePool();
        if (this.fastDecode) this.fastDecode.poisoned = true;
        throw new Error('TTS produced invalid audio');
      }
    }
    return { waveform };
  }

  /** Rebuild GPU executors after a poisoned generate (device still valid). */
  async recoverGpu(): Promise<void> {
    if (!this.durationBuf || !this.decodeBuf) return;
    this.duration?.purgePool();
    this.fastDecode?.purgePool();
    await this.loadModelBuffers(this.durationBuf, this.decodeBuf);
  }

  get sampleRateValue(): number {
    return this.sampleRate;
  }
}

/** Split long text into <=280-char chunks at sentence/punctuation boundaries. */
export function splitText(text: string, limit = 280): string[] {
  const normalized = text.replace(/\s+/g, ' ').trim();
  const sentences = normalized
    .split(/(?<=[.!?;:])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  for (const sentence of sentences.length ? sentences : [normalized]) {
    let rest = sentence;
    while (rest.length > limit) {
      const search = rest.slice(0, limit + 1);
      let punctuation = -1;
      for (const mark of [',', ';', ':']) punctuation = Math.max(punctuation, search.lastIndexOf(mark));
      let splitAt = punctuation >= limit / 2 ? punctuation + 1 : rest.lastIndexOf(' ', limit + 1);
      if (splitAt < limit / 2) splitAt = limit;
      chunks.push(rest.slice(0, splitAt).trim());
      rest = rest.slice(splitAt).trim();
    }
    if (rest) chunks.push(rest);
  }
  return chunks;
}

/** Inter-chunk pause lengths by trailing punctuation of previous chunk. */
export function boundaryPauseSeconds(chunk: string): number {
  const ending = chunk.trim().slice(-1);
  switch (ending) {
    case '?': return 0.28;
    case '!': return 0.24;
    case '.': return 0.22;
    case ';': return 0.16;
    case ':': return 0.13;
    case ',': return 0.09;
    default: return 0.08;
  }
}

/** 5 ms linear edge fade to avoid clicks between chunks. */
export function edgeFade(waveform: Float32Array, sampleRate: number, milliseconds = 5): Float32Array {
  const frames = Math.min(Math.round((sampleRate * milliseconds) / 1000), Math.floor(waveform.length / 2));
  if (frames <= 0) return waveform;
  const out = waveform.slice();
  for (let j = 0; j < frames; j++) {
    const ramp = j / (frames - 1 || 1);
    out[j] *= ramp;
    out[out.length - 1 - j] *= ramp;
  }
  return out;
}
