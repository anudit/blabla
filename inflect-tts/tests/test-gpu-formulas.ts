/**
 * CPU-side checks that the WGSL formulas match ops-cpu / ONNX.
 * Run: bun run inflect-tts/tests/test-gpu-formulas.ts
 */
import { conv1d, convTranspose1d, layerNorm, transpose, slice } from '../ops-cpu.ts';
import { contiguousStrides, makeTensor, numel } from '../tensor.ts';

function f32(arr: number[], dims: number[]) {
  const t = makeTensor('f32', dims);
  (t.data as Float32Array).set(arr);
  return t;
}

function assertClose(name: string, a: Float32Array, b: Float32Array, tol = 1e-5) {
  if (a.length !== b.length) throw new Error(`${name}: length ${a.length} vs ${b.length}`);
  let mx = 0;
  for (let i = 0; i < a.length; i++) mx = Math.max(mx, Math.abs(a[i] - b[i]));
  if (mx > tol) throw new Error(`${name}: maxDiff ${mx} > ${tol}`);
  console.log(`${name}: OK (maxDiff=${mx.toExponential(2)})`);
}

function gpuConv1d(
  x: Float32Array, w: Float32Array, bias: Float32Array | null,
  N: number, C: number, Win: number, M: number, k: number, dil: number, pad0: number,
) {
  const Wout = Win + pad0 + pad0 - dil * (k - 1);
  const out = new Float32Array(N * M * Wout);
  for (let n = 0; n < N; n++) {
    for (let row = 0; row < M; row++) {
      for (let t = 0; t < Wout; t++) {
        let acc = 0;
        for (let c = 0; c < C; c++) {
          const xb = (n * C + c) * Win;
          const wb = (row * C + c) * k;
          for (let kk = 0; kk < k; kk++) {
            const s = t - pad0 + kk * dil;
            if (s >= 0 && s < Win) acc += x[xb + s] * w[wb + kk];
          }
        }
        if (bias) acc += bias[row];
        out[(n * M + row) * Wout + t] = acc;
      }
    }
  }
  return out;
}

function gpuLayerNorm(x: Float32Array, gamma: Float32Array, beta: Float32Array, inner: number, eps: number) {
  const rows = x.length / inner;
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    const base = r * inner;
    let sum = 0;
    for (let j = 0; j < inner; j++) sum += x[base + j];
    const mean = sum / inner;
    let vsum = 0;
    for (let j = 0; j < inner; j++) {
      const dv = x[base + j] - mean;
      vsum += dv * dv;
    }
    const inv = 1 / Math.sqrt(vsum / inner + eps);
    for (let j = 0; j < inner; j++) out[base + j] = (x[base + j] - mean) * inv * gamma[j] + beta[j];
  }
  return out;
}

{
  const N = 1, C = 3, Win = 8, M = 2, k = 3, dil = 1, pad = 1;
  const x = new Float32Array(N * C * Win).map((_, i) => (i + 1) * 0.1);
  const w = new Float32Array(M * C * k).map((_, i) => (i % 5) * 0.2 - 0.3);
  const b = new Float32Array([0.05, -0.02]);
  const cpu = conv1d(f32([...x], [N, C, Win]), f32([...w], [M, C, k]), f32([...b], [M]), [pad, pad], [dil]);
  const gpu = gpuConv1d(x, w, b, N, C, Win, M, k, dil, pad);
  assertClose('conv1d', cpu.data as Float32Array, gpu);
}

{
  const N = 1, C = 2, Win = 5, M = 2, k = 4, stride = 2, pad = 1;
  const x = new Float32Array(N * C * Win).map((_, i) => (i + 1) * 0.15);
  const w = new Float32Array(C * M * k).map((_, i) => (i % 4) * 0.1 - 0.15);
  const b = new Float32Array([0.01, 0.02]);
  const cpu = convTranspose1d(
    f32([...x], [N, C, Win]), f32([...w], [C, M, k]), f32([...b], [M]), [pad, pad], [stride],
  );
  // GPU gather formula: u = (t + pad - kk) / stride
  const Wout = (Win - 1) * stride - pad - pad + k;
  const gpu = new Float32Array(N * M * Wout);
  for (let n = 0; n < N; n++) {
    for (let row = 0; row < M; row++) {
      for (let t = 0; t < Wout; t++) {
        let acc = b[row];
        for (let c = 0; c < C; c++) {
          const xb = (n * C + c) * Win;
          const wb = (c * M + row) * k;
          for (let kk = 0; kk < k; kk++) {
            const num = t + pad - kk;
            if (num >= 0 && num % stride === 0) {
              const u = num / stride;
              if (u < Win) acc += x[xb + u] * w[wb + kk];
            }
          }
        }
        gpu[(n * M + row) * Wout + t] = acc;
      }
    }
  }
  assertClose('convTranspose1d', cpu.data as Float32Array, gpu);
}

{
  const inner = 96;
  const rows = 4;
  const x = new Float32Array(rows * inner).map((_, i) => Math.sin(i * 0.17));
  const g = new Float32Array(inner).map((_, i) => 0.8 + (i % 7) * 0.01);
  const b = new Float32Array(inner).map((_, i) => (i % 5) * 0.02 - 0.04);
  const cpu = layerNorm(f32([...x], [rows, inner]), f32([...g], [inner]), f32([...b], [inner]), 1e-5);
  const gpu = gpuLayerNorm(x, g, b, inner, 1e-5);
  assertClose('layerNorm', cpu.data as Float32Array, gpu);
}

function pad4(dims: number[]): number[] {
  const out = dims.slice();
  while (out.length < 4) out.unshift(1);
  return out.slice(0, 4);
}

function gpuTranspose(x: Float32Array, inDims: number[], perm: number[]): Float32Array {
  const r = inDims.length;
  const pad = 4 - r;
  const permAdj = [...Array(pad).keys()].concat(perm.map((p) => p + pad));
  const dimsP = pad4(inDims);
  const outDimsP = permAdj.map((p) => dimsP[p]);
  const inStrides = contiguousStrides(dimsP);
  const n = numel(outDimsP);
  const out = new Float32Array(n);
  for (let lin = 0; lin < n; lin++) {
    const d0 = outDimsP[0], d1 = outDimsP[1], d2 = outDimsP[2], d3 = outDimsP[3];
    const p1 = d1 * d2 * d3;
    const c0 = Math.floor(lin / p1);
    let rem = lin % p1;
    const p2 = d2 * d3;
    const c1 = Math.floor(rem / p2); rem = rem % p2;
    const c2 = Math.floor(rem / d3); const c3 = rem % d3;
    const coords = [c0, c1, c2, c3];
    let off = 0;
    for (let d = 0; d < 4; d++) off += coords[d] * inStrides[permAdj[d]];
    out[lin] = x[off];
  }
  return out;
}

{
  const dims = [1, 4, 6];
  const x = new Float32Array(numel(dims)).map((_, i) => i + 0.5);
  const cpu = transpose(f32([...x], dims), [0, 2, 1]);
  const gpu = gpuTranspose(x, dims, [0, 2, 1]);
  assertClose('transpose [0,2,1]', cpu.data as Float32Array, gpu);
}

{
  const dims = [2, 3, 4, 5];
  const x = new Float32Array(numel(dims)).map((_, i) => (i % 11) * 0.3);
  const cpu = transpose(f32([...x], dims), [0, 1, 3, 2]);
  const gpu = gpuTranspose(x, dims, [0, 1, 3, 2]);
  assertClose('transpose [0,1,3,2]', cpu.data as Float32Array, gpu);
}

function padLeading(arr: number[], fill: number): number[] {
  const out = arr.slice();
  while (out.length < 4) out.unshift(fill);
  return out;
}

function gpuSlice(x: Float32Array, inDims: number[], starts: number[], steps: number[], outDims: number[]): Float32Array {
  const startsP = padLeading(starts, 0);
  const stepsP = padLeading(steps, 1);
  const dimsP = pad4(inDims);
  const outP = pad4(outDims);
  const inStrides = contiguousStrides(dimsP);
  const n = numel(outP);
  const out = new Float32Array(n);
  for (let lin = 0; lin < n; lin++) {
    const d1 = outP[1], d2 = outP[2], d3 = outP[3];
    const p1 = d1 * d2 * d3;
    const c0 = Math.floor(lin / p1);
    let rem = lin % p1;
    const p2 = d2 * d3;
    const c1 = Math.floor(rem / p2); rem = rem % p2;
    const c2 = Math.floor(rem / d3); const c3 = rem % d3;
    const coords = [c0, c1, c2, c3];
    let off = 0;
    for (let d = 0; d < 4; d++) off += (startsP[d] + coords[d] * stepsP[d]) * inStrides[d];
    out[lin] = x[off];
  }
  return out;
}

{
  const dims = [1, 192, 8];
  const x = new Float32Array(numel(dims)).map((_, i) => i + 1);
  const cpu = slice(f32([...x], dims), [-1], [-1e18], [1], [-1]);
  const gpu = gpuSlice(x, dims, [0, 191, 0], [1, -1, 1], cpu.dims);
  assertClose('slice reverse channels', cpu.data as Float32Array, gpu);
}

{
  const dims = [2, 3, 5];
  const x = new Float32Array(numel(dims)).map((_, i) => i * 0.25);
  const cpu = slice(f32([...x], dims), [1], [4], [2], [1]);
  const gpu = gpuSlice(x, dims, [0, 0, 1], [1, 1, 1], cpu.dims);
  assertClose('slice last-dim window', cpu.data as Float32Array, gpu);
}

console.log('all formula checks passed');
