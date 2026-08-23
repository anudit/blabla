/**
 * Pure-TypeScript implementations of every ONNX op used by the
 * Inflect-Micro-v2 graphs. These are the semantic reference for the
 * WebGPU kernels and double as a CPU-only execution path.
 */
import {
  type CpuTensor, type DType, allocData, broadcastShapes, broadcastStrides,
  contiguousStrides, makeTensor, numel,
} from './tensor.ts';

type Num = number | bigint;

function f(x: CpuTensor): Float32Array { return x.data as Float32Array; }
function i(x: CpuTensor): Int32Array | BigInt64Array { return x.data as Int32Array | BigInt64Array; }


/** Byte view over a tensor's data without copying. */
function asBytes(d: CpuTensor['data']): Uint8Array {
  return new Uint8Array((d as any).buffer, (d as any).byteOffset, d.byteLength);
}

export function asNumbers(t: CpuTensor): number[] {
  const out: number[] = [];
  const d = t.data;
  if (d instanceof BigInt64Array) { for (let n = 0; n < d.length; n++) out.push(Number(d[n])); }
  else { for (let n = 0; n < d.length; n++) out.push(d[n] as number); }
  return out;
}

// ── Elementwise binary with numpy broadcasting ─────────────────────────────

export function binaryOp(
  op: 'add' | 'sub' | 'mul' | 'div', A: CpuTensor, B: CpuTensor,
): CpuTensor {
  const dims = broadcastShapes(A.dims, B.dims);
  const isFloat = A.type === 'f32' || B.type === 'f32';
  const outType: DType = isFloat ? 'f32' : A.type;
  const out = makeTensor(outType, dims);
  const o = out.data as Float32Array;
  const a = A.data as unknown as Float32Array;
  const b = B.data as unknown as Float32Array;
  const sa = broadcastStrides(A.dims, dims);
  const sb = broadcastStrides(B.dims, dims);
  const strides = contiguousStrides(dims);
  const rank = dims.length;
  const idx = new Array(rank).fill(0);
  const n = numel(dims);
  let ia = 0, ib = 0;
  const an = asNumbers(A), bn = asNumbers(B);
  for (let lin = 0; lin < n; lin++) {
    let av = an[ia], bv = bn[ib];
    let r: number;
    switch (op) {
      case 'add': r = av + bv; break;
      case 'sub': r = av - bv; break;
      case 'mul': r = av * bv; break;
      case 'div': r = av / bv; break;
    }
    if (outType === 'f32') (o as Float32Array)[lin] = r;
    else if (outType === 'i64') (out.data as BigInt64Array)[lin] = BigInt(Math.trunc(r));
    else (out.data as Int32Array)[lin] = Math.trunc(r);
    // increment multi-dim index
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      ia += sa[d]; ib += sb[d];
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      ia -= sa[d] * dims[d]; ib -= sb[d] * dims[d];
    }
  }
  return out;
}

const UNARY: Record<string, (x: number) => number> = {
  exp: Math.exp,
  tanh: Math.tanh,
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  relu: (x) => (x > 0 ? x : 0),
  neg: (x) => -x,
  ceil: Math.ceil,
};

export function unaryOp(kind: string, X: CpuTensor, alpha = 0): CpuTensor {
  const out = makeTensor(X.type as DType === 'f32' ? 'f32' : 'f32', X.dims);
  const src = f(X), dst = f(out);
  const fn = UNARY[kind];
  if (kind === 'leakyrelu') {
    for (let j = 0; j < src.length; j++) dst[j] = src[j] > 0 ? src[j] : alpha * src[j];
  } else {
    for (let j = 0; j < src.length; j++) dst[j] = fn(src[j]);
  }
  return out;
}

/** Elementwise power with broadcasting. Type follows integer promotion rules. */
export function powOp(A: CpuTensor, B: CpuTensor): CpuTensor {
  const dims = broadcastShapes(A.dims, B.dims);
  const isFloat = A.type === 'f32' || B.type === 'f32';
  const outType: DType = isFloat ? 'f32' : A.type;
  const out = makeTensor(outType, dims);
  const o = out.data as Float32Array;
  const an = asNumbers(A), bn = asNumbers(B);
  const sa = broadcastStrides(A.dims, dims);
  const sb = broadcastStrides(B.dims, dims);
  const strides = contiguousStrides(dims);
  const rank = dims.length;
  const idx = new Array(rank).fill(0);
  let ia = 0, ib = 0;
  for (let lin = 0; lin < o.length; lin++) {
    const r = Math.pow(an[ia], bn[ib]);
    if (outType === 'f32') o[lin] = r;
    else if (outType === 'i64') (out.data as BigInt64Array)[lin] = BigInt(Math.trunc(r));
    else (out.data as Int32Array)[lin] = Math.trunc(r);
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      ia += sa[d]; ib += sb[d];
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      ia -= sa[d] * dims[d]; ib -= sb[d] * dims[d];
    }
  }
  void strides;
  return out;
}

export function clipOp(X: CpuTensor, min?: number, max?: number): CpuTensor {
  const out = makeTensor('f32', X.dims);
  const s = f(X), d = f(out);
  for (let j = 0; j < s.length; j++) {
    let v = s[j];
    if (min !== undefined && v < min) v = min;
    if (max !== undefined && v > max) v = max;
    d[j] = v;
  }
  return out;
}

/** Batched matmul with batch-dim broadcasting: A[..., M, K] @ B[..., K, N]. */
export function matmul(A: CpuTensor, B: CpuTensor): CpuTensor {
  const M = A.dims[A.dims.length - 2];
  const K = A.dims[A.dims.length - 1];
  const K2 = B.dims[B.dims.length - 2];
  const N = B.dims[B.dims.length - 1];
  if (K !== K2) throw new Error(`MatMul shape mismatch ${A.dims} @ ${B.dims}`);
  const batchA = A.dims.slice(0, -2);
  const batchB = B.dims.slice(0, -2);
  const batch = broadcastShapes(batchA.length ? batchA : [1], batchB.length ? batchB : [1]);
  const dims = [...batch, M, N];

  const outT = makeTensor('f32', dims);
  const a = f(A), b = f(B), o = f(outT);
  const sa = contiguousStrides([...(batchA.length ? batchA : [1]), M, K]);
  const sb = contiguousStrides([...(batchB.length ? batchB : [1]), K, N]);
  const ba = broadcastStrides(batchA.length ? batchA : [1], batch);
  const bb = broadcastStrides(batchB.length ? batchB : [1], batch);

  const rank = batch.length;
  const idx = new Array(rank).fill(0);
  const total = batch.reduce((x, y) => x * y, 1);
  for (let bi = 0; bi < total; bi++) {
    let offA = 0, offB = 0;
    for (let d = 0; d < rank; d++) { offA += idx[d] * ba[d] * M * K; offB += idx[d] * bb[d] * K * N; }
    const base = bi * M * N;
    for (let m = 0; m < M; m++) {
      for (let nn = 0; nn < N; nn++) {
        let acc = 0;
        for (let k = 0; k < K; k++) acc += a[offA + m * K + k] * b[offB + k * N + nn];
        o[base + m * N + nn] = acc;
      }
    }
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      if (idx[d] < batch[d]) break;
      idx[d] = 0;
    }
  }
  return outT;
}

/** Conv1d, group=1. X [N,C,W], W [M,C,k], stride=1. */
export function conv1d(
  X: CpuTensor, Wt: CpuTensor, bias: CpuTensor | null,
  pads: number[], dilations: number[],
): CpuTensor {
  const [N, C, Win] = X.dims;
  const [M, Cw, k] = Wt.dims;
  const dil = dilations[0] ?? 1;
  const p0 = pads[0] ?? 0, p1 = pads[1] ?? p0;
  const Wout = Win + p0 + p1 - dil * (k - 1);
  const out = makeTensor('f32', [N, M, Wout]);
  const x = f(X), w = f(Wt), b = bias ? f(bias) : null, o = f(out);
  for (let n = 0; n < N; n++) {
    for (let m = 0; m < M; m++) {
      for (let t = 0; t < Wout; t++) {
        let acc = b ? b[m] : 0;
        for (let c = 0; c < C; c++) {
          const xb = (n * C + c) * Win;
          const wb = (m * C + c) * k;
          for (let kk = 0; kk < k; kk++) {
            const u = t - p0 + kk * dil;
            if (u >= 0 && u < Win) acc += x[xb + u] * w[wb + kk];
          }
        }
        o[(n * M + m) * Wout + t] = acc;
      }
    }
  }
  return out;
}

/** ConvTranspose1d, group=1. X [N,C,W], W [C,M,k]. */
export function convTranspose1d(
  X: CpuTensor, Wt: CpuTensor, bias: CpuTensor | null,
  pads: number[], strides: number[],
): CpuTensor {
  const [N, C, Win] = X.dims;
  const [, M, k] = Wt.dims;
  const s = strides[0] ?? 1;
  const p0 = pads[0] ?? 0, p1 = pads[1] ?? p0;
  const Wout = (Win - 1) * s - p0 - p1 + k;
  const out = makeTensor('f32', [N, M, Wout]);
  const x = f(X), w = f(Wt), b = bias ? f(bias) : null, o = f(out);
  for (let n = 0; n < N; n++) {
    for (let m = 0; m < M; m++) {
      for (let t = 0; t < Wout; t++) o[(n * M + m) * Wout + t] = b ? b[m] : 0;
    }
    for (let c = 0; c < C; c++) {
      for (let u = 0; u < Win; u++) {
        const xv = x[(n * C + c) * Win + u];
        if (xv === 0) continue;
        for (let kk = 0; kk < k; kk++) {
          const t = u * s - p0 + kk;
          if (t < 0 || t >= Wout) continue;
          for (let m = 0; m < M; m++) {
            o[(n * M + m) * Wout + t] += xv * w[(c * M + m) * k + kk];
          }
        }
      }
    }
  }
  return out;
}

/** LayerNormalization over last dim. */
export function layerNorm(X: CpuTensor, gamma: CpuTensor, beta: CpuTensor, eps: number): CpuTensor {
  const inner = X.dims[X.dims.length - 1];
  const rows = X.data.length / inner;
  const out = makeTensor('f32', X.dims);
  const x = f(X), g = f(gamma), bta = f(beta), o = f(out);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let j = 0; j < inner; j++) mean += x[r * inner + j];
    mean /= inner;
    let varr = 0;
    for (let j = 0; j < inner; j++) { const dv = x[r * inner + j] - mean; varr += dv * dv; }
    varr /= inner;
    const inv = 1 / Math.sqrt(varr + eps);
    for (let j = 0; j < inner; j++) o[r * inner + j] = (x[r * inner + j] - mean) * inv * g[j] + bta[j];
  }
  return out;
}

/** Softmax along the last dim. */
export function softmax(X: CpuTensor): CpuTensor {
  const inner = X.dims[X.dims.length - 1];
  const rows = X.data.length / inner;
  const out = makeTensor('f32', X.dims);
  const x = f(X), o = f(out);
  for (let r = 0; r < rows; r++) {
    let mx = -Infinity;
    for (let j = 0; j < inner; j++) mx = Math.max(mx, x[r * inner + j]);
    let sum = 0;
    for (let j = 0; j < inner; j++) { const ev = Math.exp(x[r * inner + j] - mx); o[r * inner + j] = ev; sum += ev; }
    for (let j = 0; j < inner; j++) o[r * inner + j] /= sum;
  }
  return out;
}

/** Generic transpose by perm. */
export function transpose(X: CpuTensor, perm?: number[]): CpuTensor {
  const rank = X.dims.length;
  const p = perm ?? [...Array(rank).keys()].reverse();
  const outDims = p.map((i) => X.dims[i]);
  const out = makeTensor(X.type, outDims);
  const inStrides = contiguousStrides(X.dims);
  const outStridesInPermOrder = contiguousStrides(outDims);
  // For each output linear index -> source offset
  const coords = new Array(rank).fill(0);
  const n = numel(outDims);
  const data = asBytes(X.data);
  const odata = asBytes(out.data);
  const elemSize = X.type === 'f32' || X.type === 'i32' ? 4 : X.type === 'i64' ? 8 : 1;
  for (let lin = 0; lin < n; lin++) {
    let src = 0;
    for (let d = 0; d < rank; d++) src += coords[d] * inStrides[p[d]];
    odata.set(data.subarray(src * elemSize, (src + 1) * elemSize), lin * elemSize);
    for (let d = rank - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < outDims[d]) break;
      coords[d] = 0;
    }
  }
  void outStridesInPermOrder;
  return out;
}

export function concat(tensors: CpuTensor[], axis: number): CpuTensor {
  if (tensors.length === 1) return tensors[0];
  const rank = tensors[0].dims.length;
  const ax = ((axis % rank) + rank) % rank;
  const outDims = tensors[0].dims.slice();
  outDims[ax] = tensors.reduce((s, t) => s + t.dims[ax], 0);
  const out = makeTensor(tensors[0].type, outDims);
  const od = asBytes(out.data);
  const elemSize = out.type === 'f32' || out.type === 'i32' ? 4 : out.type === 'i64' ? 8 : 1;
  const outerBefore = outDims.slice(0, ax).reduce((a, b) => a * b, 1);
  const innerAfter = outDims.slice(ax + 1).reduce((a, b) => a * b, 1);
  let axisOffset = 0;
  for (const t of tensors) {
    const td = asBytes(t.data);
    const chunkAxis = t.dims[ax];
    for (let ob = 0; ob < outerBefore; ob++) {
      for (let a = 0; a < chunkAxis; a++) {
        const srcStart = ((ob * chunkAxis) + a) * innerAfter * elemSize;
        const dstStart = ((ob * outDims[ax]) + axisOffset + a) * innerAfter * elemSize;
        od.set(td.subarray(srcStart, srcStart + innerAfter * elemSize), dstStart);
      }
    }
    axisOffset += chunkAxis;
  }
  return out;
}

/** Slice with starts/ends/axes/steps inputs (opset 13 semantics). */
export function slice(
  X: CpuTensor, starts: number[], ends: number[], axes: number[], steps: number[],
): CpuTensor {
  const rank = X.dims.length;
  const resolvedAxes = axes.length ? axes : [...Array(rank).keys()];
  const st = new Array(rank).fill(0);
  const en = new Array(rank).fill(0);
  const sp = new Array(rank).fill(1);
  for (let a = 0; a < rank; a++) { st[a] = 0; en[a] = X.dims[a]; }
  for (let i = 0; i < resolvedAxes.length; i++) {
    const a = ((resolvedAxes[i] % rank) + rank) % rank;
    const d = X.dims[a];
    const s0 = starts[i], e = ends[i];
    sp[a] = steps.length ? steps[i] : 1;
    if (sp[a] > 0) {
      st[a] = Math.max(0, Math.min(s0 < 0 ? s0 + d : s0, d));
      en[a] = Math.max(st[a], Math.min(e < 0 ? e + d : e, d));
    } else {
      st[a] = s0 <= -d ? -1 : Math.max(-1, Math.min(s0 < 0 ? s0 + d : s0, d - 1));
      en[a] = e <= -d ? -1 : Math.max(-1, Math.min(e < 0 ? e + d : e, d - 1));
    }
  }
  const outDims = X.dims.map((d, a) => Math.max(0, Math.ceil((en[a] - st[a]) / sp[a])));
  const out = makeTensor(X.type, outDims);
  const src = asBytes(X.data);
  const dst = asBytes(out.data);
  const elemSize = X.type === 'f32' || X.type === 'i32' ? 4 : X.type === 'i64' ? 8 : 1;
  const inStrides = contiguousStrides(X.dims);
  const n = numel(outDims);
  const coords = new Array(rank).fill(0);
  for (let lin = 0; lin < n; lin++) {
    let sIdx = 0;
    for (let d = 0; d < rank; d++) sIdx += (st[d] + coords[d] * sp[d]) * inStrides[d];
    dst.set(src.subarray(sIdx * elemSize, (sIdx + 1) * elemSize), lin * elemSize);
    for (let d = rank - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < outDims[d]) break;
      coords[d] = 0;
    }
  }
  return out;
}

export function reshapeMeta(dims: number[], shape: number[]): number[] {
  const total = dims.reduce((a, b) => a * b, 1);
  const out = shape.slice();
  const negIdx = out.indexOf(-1);
  let known = 1;
  for (const s of out) if (s !== -1 && s !== 0) known *= s;
  for (let j = 0; j < out.length; j++) {
    if (out[j] === 0) out[j] = dims[j]; // allowzero=0
  }
  if (negIdx >= 0) out[negIdx] = total / known;
  return out;
}

export function squeezeMeta(dims: number[], axesRaw: number[]): number[] {
  const axes = axesRaw.map((a) => (a < 0 ? a + dims.length : a));
  return dims.filter((_, j) => !axes.includes(j));
}

export function unsqueezeMeta(dims: number[], axesRaw: number[]): number[] {
  const rank = dims.length + axesRaw.length;
  const axes = axesRaw.map((a) => (a < 0 ? a + rank : a));
  const out = new Array(rank).fill(1);
  let di = 0;
  for (let j = 0; j < rank; j++) {
    if (!axes.includes(j)) out[j] = dims[di++];
  }
  return out;
}

export function castTensor(X: CpuTensor, to: DType): CpuTensor {
  if (X.type === to) return X;
  const out = makeTensor(to, X.dims);
  const s = X.data;
  if (to === 'bool') {
    const d = out.data as Uint8Array;
    for (let j = 0; j < s.length; j++) d[j] = Number(s[j]) !== 0 ? 1 : 0;
  } else if (s instanceof BigInt64Array) {
    const d = out.data as Exclude<CpuTensor['data'], BigInt64Array>;
    for (let j = 0; j < s.length; j++) d[j] = Number(s[j]);
  } else if (to === 'i64') {
    const d = out.data as BigInt64Array;
    for (let j = 0; j < s.length; j++) d[j] = BigInt(Math.trunc(Number(s[j])));
  } else {
    const d = out.data as Exclude<CpuTensor['data'], BigInt64Array>;
    for (let j = 0; j < s.length; j++) d[j] = Number(s[j]);
  }
  return out;
}

export function gather(X: CpuTensor, indices: CpuTensor, axis: number): CpuTensor {
  const rank = X.dims.length;
  const ax = ((axis % rank) + rank) % rank;
  const idxNums = asNumbers(indices);
  const outDims = [
    ...X.dims.slice(0, ax),
    ...indices.dims,
    ...X.dims.slice(ax + 1),
  ];
  const out = makeTensor(X.type, outDims);
  const elemSize = X.type === 'f32' || X.type === 'i32' ? 4 : X.type === 'i64' ? 8 : 1;
  const src = asBytes(X.data);
  const dst = asBytes(out.data);
  const outer = X.dims.slice(0, ax).reduce((a, b) => a * b, 1);
  const inner = X.dims.slice(ax + 1).reduce((a, b) => a * b, 1);
  const axisSize = X.dims[ax];
  for (let ob = 0; ob < outer; ob++) {
    for (let j = 0; j < idxNums.length; j++) {
      let ix = idxNums[j];
      if (ix < 0) ix += axisSize;
      const srcOff = (ob * axisSize + ix) * inner;
      const dstOff = (ob * idxNums.length + j) * inner;
      dst.set(src.subarray(srcOff * elemSize, (srcOff + inner) * elemSize), dstOff * elemSize);
    }
  }
  return out;
}

export function padConstant(X: CpuTensor, pads: number[], value = 0): CpuTensor {
  const rank = X.dims.length;
  const outDims = X.dims.map((d, j) => d + pads[j] + pads[j + rank]);
  const out = makeTensor(X.type, outDims);
  const elemSize = X.type === 'f32' || X.type === 'i32' ? 4 : X.type === 'i64' ? 8 : 1;
  (asBytes(out.data)).fill(0);
  if (value !== 0 && X.type === 'f32') (out.data as Float32Array).fill(value);
  const inStrides = contiguousStrides(X.dims);
  const outStrides = contiguousStrides(outDims);
  const n = numel(X.dims);
  const coords = new Array(rank).fill(0);
  const src = asBytes(X.data);
  const dst = asBytes(out.data);
  for (let lin = 0; lin < n; lin++) {
    let oIdx = 0;
    for (let d = 0; d < rank; d++) oIdx += (coords[d] + pads[d]) * outStrides[d];
    dst.set(src.subarray(lin * elemSize, (lin + 1) * elemSize), oIdx * elemSize);
    for (let d = rank - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < X.dims[d]) break;
      coords[d] = 0;
    }
  }
  void inStrides;
  return out;
}

export function range(start: Num, limit: Num, delta: Num, type: DType): CpuTensor {
  const s = Number(start), l = Number(limit), dd = Number(delta);
  const count = Math.max(0, Math.ceil((l - s) / dd));
  const out = makeTensor(type, [count]);
  for (let j = 0; j < count; j++) {
    const v = s + j * dd;
    if (type === 'i64') (out.data as BigInt64Array)[j] = BigInt(Math.trunc(v));
    else if (type === 'i32') (out.data as Int32Array)[j] = Math.trunc(v);
    else (out.data as Float32Array)[j] = v;
  }
  return out;
}

/** Comparison producing bool tensor, broadcast. */
export function compare(op: 'less' | 'equal', A: CpuTensor, B: CpuTensor): CpuTensor {
  const dims = broadcastShapes(A.dims, B.dims);
  const out = makeTensor('bool', dims);
  const o = out.data as Uint8Array;
  const sa = broadcastStrides(A.dims, dims);
  const sb = broadcastStrides(B.dims, dims);
  const rank = dims.length;
  const idx = new Array(rank).fill(0);
  const n = numel(dims);
  let ia = 0, ib = 0;
  const an = asNumbers(A), bn = asNumbers(B);
  for (let lin = 0; lin < n; lin++) {
    const av = an[ia], bv = bn[ib];
    o[lin] = op === 'less' ? (av < bv ? 1 : 0) : (av === bv ? 1 : 0);
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      ia += sa[d]; ib += sb[d];
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      ia -= sa[d] * dims[d]; ib -= sb[d] * dims[d];
    }
  }
  void i;
  return out;
}

export function where(cond: CpuTensor, X: CpuTensor, Y: CpuTensor): CpuTensor {
  const dims = broadcastShapes(broadcastShapes(cond.dims, X.dims), Y.dims);
  const out = makeTensor('f32', dims);
  const sc = broadcastStrides(cond.dims, dims);
  const sx = broadcastStrides(X.dims, dims);
  const sy = broadcastStrides(Y.dims, dims);
  const o = f(out), xv = f(X), yv = f(Y);
  const cd = cond.data;
  const rank = dims.length;
  const idx = new Array(rank).fill(0);
  const n = numel(dims);
  let ic = 0, ix = 0, iy = 0;
  for (let lin = 0; lin < n; lin++) {
    o[lin] = Number(cd[ic]) !== 0 ? xv[ix] : yv[iy];
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      ic += sc[d]; ix += sx[d]; iy += sy[d];
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      ic -= sc[d] * dims[d]; ix -= sx[d] * dims[d]; iy -= sy[d] * dims[d];
    }
  }
  return out;
}

export function cumSum(X: CpuTensor, axis: number, exclusive: boolean, reverse: boolean): CpuTensor {
  const rank = X.dims.length;
  const ax = ((axis % rank) + rank) % rank;
  const out = makeTensor(X.type, X.dims);
  const outer = X.dims.slice(0, ax).reduce((a, b) => a * b, 1);
  const inner = X.dims.slice(ax + 1).reduce((a, b) => a * b, 1);
  const len = X.dims[ax];
  const src = X.data as unknown as Float32Array;
  const dst = out.data as unknown as Float32Array;
  for (let ob = 0; ob < outer; ob++) {
    for (let inn = 0; inn < inner; inn++) {
      let acc = 0;
      for (let j = 0; j < len; j++) {
        const jj = reverse ? len - 1 - j : j;
        const off = (ob * len + jj) * inner + inn;
        if (exclusive) { dst[off] = acc; acc += src[off]; }
        else { acc += src[off]; dst[off] = acc; }
      }
    }
  }
  return out;
}

export function reduceSum(X: CpuTensor, axes: number[] | null, keepdims: boolean): CpuTensor {
  const rank = X.dims.length;
  const ax = axes ?? [...Array(rank).keys()];
  const set = new Set(ax.map((a) => ((a % rank) + rank) % rank));
  const kept = X.dims.filter((_, j) => !set.has(j));
  const outDims = keepdims ? X.dims.map((d, j) => (set.has(j) ? 1 : d)) : kept;
  const reduced = X.dims.filter((_, j) => set.has(j));
  const out = makeTensor('f32', outDims);
  const o = f(out);
  const x = f(X);
  const n = numel(X.dims);
  // Full-strides decomposition: kept dims map to output, reduced dims to inner block.
  const keptStrides = contiguousStrides(kept.length ? kept : [1]);
  const redStrides = contiguousStrides(reduced.length ? reduced : [1]);
  o.fill(0);
  const fullCoords = new Array(rank).fill(0);
  for (let lin = 0; lin < n; lin++) {
    let ko = 0, ro = 0;
    let ki = 0, ri = 0;
    for (let d = 0; d < rank; d++) {
      if (set.has(d)) ro += fullCoords[d] * redStrides[ri++];
      else ko += fullCoords[d] * keptStrides[ki++];
    }
    o[ko] += x[lin];
    for (let d = rank - 1; d >= 0; d--) {
      fullCoords[d]++;
      if (fullCoords[d] < X.dims[d]) break;
      fullCoords[d] = 0;
    }
  }
  void reduced;
  return out;
}

export function reduceMaxAll(X: CpuTensor): CpuTensor {
  const out = makeTensor(X.type, []);
  let mx = -Infinity;
  const d = X.data;
  for (let j = 0; j < d.length; j++) mx = Math.max(mx, Number(d[j]));
  if (X.type === 'i64') (out.data as BigInt64Array)[0] = BigInt(Math.round(mx));
  else if (X.type === 'i32') (out.data as Int32Array)[0] = Math.round(mx);
  else (out.data as Float32Array)[0] = mx;
  return out;
}
