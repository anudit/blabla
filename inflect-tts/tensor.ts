/** Tensor primitives shared by the CPU and WebGPU execution paths. */

export type DType = 'f32' | 'i64' | 'i32' | 'bool';

export interface CpuTensor {
  dims: number[];
  type: DType;
  data: Float32Array | BigInt64Array | Int32Array | Uint8Array;
}

export function numel(dims: number[]): number {
  let n = 1;
  for (const d of dims) n *= d;
  return n;
}

export const DTYPES: Record<number, DType> = { 1: 'f32', 6: 'i32', 7: 'i64', 9: 'bool' };

export function allocData(type: DType, n: number): CpuTensor['data'] {
  switch (type) {
    case 'f32': return new Float32Array(n);
    case 'i64': return new BigInt64Array(n);
    case 'i32': return new Int32Array(n);
    case 'bool': return new Uint8Array(n);
  }
}

export function makeTensor(type: DType, dims: number[]): CpuTensor {
  return { dims, type, data: allocData(type, numel(dims)) };
}

/** Convert raw initializer bytes to a typed array. */
export function rawDataToTyped(type: DType, raw: Uint8Array): CpuTensor['data'] {
  const aligned = new Uint8Array(raw.byteLength);
  aligned.set(raw);
  switch (type) {
    case 'f32': return new Float32Array(aligned.buffer);
    case 'i64': return new BigInt64Array(aligned.buffer);
    case 'i32': return new Int32Array(aligned.buffer);
    case 'bool': return aligned;
  }
}

/** numpy-style broadcast of two shapes */
export function broadcastShapes(a: number[], b: number[]): number[] {
  const rank = Math.max(a.length, b.length);
  const out: number[] = new Array(rank);
  for (let i = 0; i < rank; i++) {
    const da = i >= rank - a.length ? a[i - (rank - a.length)] : 1;
    const db = i >= rank - b.length ? b[i - (rank - b.length)] : 1;
    if (da !== db && da !== 1 && db !== 1) throw new Error(`Cannot broadcast ${a} with ${b}`);
    out[i] = Math.max(da, db);
  }
  return out;
}

/** Strides for a contiguous row-major tensor. */
export function contiguousStrides(dims: number[]): number[] {
  const strides = new Array(dims.length);
  let acc = 1;
  for (let i = dims.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= dims[i];
  }
  return strides;
}

/**
 * Strides that broadcast `dims` into `target` (stride 0 on size-1 dims).
 */
export function broadcastStrides(dims: number[], target: number[]): number[] {
  const strides = contiguousStrides(dims);
  const rankDiff = target.length - dims.length;
  const out: number[] = new Array(target.length).fill(0);
  for (let i = 0; i < dims.length; i++) {
    out[rankDiff + i] = dims[i] === 1 && target[rankDiff + i] !== 1 ? 0 : strides[i];
  }
  return out;
}
