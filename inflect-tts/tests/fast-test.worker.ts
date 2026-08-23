/**
 * TEMPORARY browser-only validation harness for InflectFastEngine.
 * Runs the existing (validated) CPU GraphExecutor decode path and the new
 * GPU InflectFastEngine on identical inputs (real duration.onnx CPU output +
 * deterministic noise) and diffs the two waveforms. bun/Node has no WebGPU,
 * so this must run as a page worker — not part of the production build.
 *
 * Usage: served via a temporary dev route, driven by javascript_tool:
 *   new Worker('/fast-test.worker.js', { type: 'module' }).postMessage({ type: 'run' })
 */
import { GraphExecutor } from '../executor.ts';
import { InflectFastEngine } from '../fast-engine.ts';
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

self.addEventListener('message', async (e: MessageEvent<any>) => {
  if (e.data?.type !== 'run') return;
  try {
    const post = (msg: any) => self.postMessage(msg);
    const DUR_URL = 'https://huggingface.co/owensong/Inflect-Micro-v2-ONNX/resolve/main/onnx/duration.onnx';
    const DEC_URL = 'https://huggingface.co/owensong/Inflect-Micro-v2-ONNX/resolve/main/onnx/decode.onnx';

    post({ status: 'fetching' });
    const [durBuf, decBuf] = await Promise.all([
      fetch(DUR_URL).then((r) => r.arrayBuffer()),
      fetch(DEC_URL).then((r) => r.arrayBuffer()),
    ]);

    // A short, simple token sequence (space + a few letters via the model's
    // vocab range) — exact ids don't matter for this test, just needs
    // duration.onnx to run and produce plausible-length m_p/logs_p/mask.
    const tokens = new BigInt64Array([3n, 46n, 66n, 70n, 70n, 73n, 3n, 20n, 89n, 3n]);

    post({ status: 'duration_cpu' });
    const durEx = new GraphExecutor(durBuf);
    await durEx.init(null);
    const durOut = await durEx.run(
      new Map<string, CpuTensor>([
        ['tokens', { dims: [1, tokens.length], type: 'i64', data: tokens }],
        ['lengths', { dims: [1], type: 'i64', data: new BigInt64Array([BigInt(tokens.length)]) }],
        ['length_scale', { dims: [], type: 'f32', data: new Float32Array([1.0]) }],
      ]),
      ['m_p_exp', 'logs_p_exp', 'y_mask'],
    );
    const mP = durOut.get('m_p_exp')!;
    const logsP = durOut.get('logs_p_exp')!;
    const yMask = durOut.get('y_mask')!;
    const T = mP.dims[2];
    const noiseScale = 0.667;
    const zpNoise = gaussianNoise((mP.data as Float32Array).length, 0);

    post({ status: 'decode_cpu_reference', T });
    const t0 = performance.now();
    const decExCpu = new GraphExecutor(decBuf);
    await decExCpu.init(null);
    const cpuOut = await decExCpu.run(
      new Map<string, CpuTensor>([
        ['m_p_exp', mP], ['logs_p_exp', logsP], ['y_mask', yMask],
        ['zp_noise', { dims: mP.dims, type: 'f32', data: zpNoise }],
        ['noise_scale', { dims: [], type: 'f32', data: new Float32Array([noiseScale]) }],
      ]),
      ['waveform', '/Mul_2_output_0', '/Add_output_0', '/flow/flows.6/Concat_output_0'],
    );
    const cpuMs = performance.now() - t0;
    const cpuWav = cpuOut.get('waveform')!.data as Float32Array;
    const cpuPostFlow = cpuOut.get('/Mul_2_output_0')!.data as Float32Array;
    const cpuZ0 = cpuOut.get('/Add_output_0')!.data as Float32Array;
    const cpuAfterFirstCoupling = cpuOut.get('/flow/flows.6/Concat_output_0')!.data as Float32Array;

    post({ status: 'decode_gpu_fast' });
    const adapter = await (navigator as any).gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const fast = new InflectFastEngine();
    await fast.init(device);
    const t1 = performance.now();
    await fast.loadWeights(decBuf);
    const loadMs = performance.now() - t1;
    const t2 = performance.now();
    const debug: { z0: Float32Array[]; postFlow: Float32Array[] } = { z0: [], postFlow: [] };
    const gpuWav = await fast.generate(
      mP.data as Float32Array, logsP.data as Float32Array, yMask.data as Float32Array,
      zpNoise, noiseScale, T, debug,
    );
    const gpuMs = performance.now() - t2;

    const maxDiffOf = (a: Float32Array, b: Float32Array) => {
      let mx = 0;
      const n = Math.min(a.length, b.length);
      for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(a[i] - b[i]));
      return mx;
    };

    post({
      status: 'done',
      cpuLen: cpuWav.length, gpuLen: gpuWav.length,
      maxDiffWav: maxDiffOf(cpuWav, gpuWav),
      maxDiffPostFlow: maxDiffOf(cpuPostFlow, debug.postFlow[4]),
      maxDiffZ0: maxDiffOf(cpuZ0, debug.z0[0]),
      maxDiffAfterFirstCoupling: maxDiffOf(cpuAfterFirstCoupling, debug.postFlow[0]),
      z0Sample: { cpu: Array.from(cpuZ0.slice(0, 8)), gpu: Array.from(debug.z0[0].slice(0, 8)) },
      z0SampleT: { cpu: Array.from(cpuZ0.slice(T - 2, T + 6)), gpu: Array.from(debug.z0[0].slice(T - 2, T + 6)) },
      cpuMs: Math.round(cpuMs), loadMs: Math.round(loadMs), gpuMs: Math.round(gpuMs),
    });
  } catch (err: any) {
    self.postMessage({ status: 'error', error: err?.message || String(err), stack: err?.stack });
  }
});
