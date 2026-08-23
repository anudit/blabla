/**
 * WGSL compute kernels for the Inflect-Micro-v2 WebGPU execution path.
 * All tensors are contiguous row-major float32 storage buffers;
 * integer tensors are stored as int32 (values < 2^31).
 * Per-dispatch shape metadata travels through a small uniform buffer.
 */

/** Pack an f64 number into a u32 word (f32 bits). */
export function packF32(v: number): number {
  const b = new ArrayBuffer(4);
  new Float32Array(b)[0] = v;
  return new Uint32Array(b)[0];
}

const META_BIND = /* wgsl */ `
@group(0) @binding(0) var<uniform> meta: array<u32, 48>;
`;

// ── Broadcast binary op ─────────────────────────────────────────────────────
// meta layout: [0]=n [1]=rank [2]=opCode, then dims(6) @4, sa(6) @10, sb(6) @16
export function binaryShader(opName: string): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn srcOffset(coords: vec3<u32>, stridesPtr: u32) -> u32 {
  var off = 0u;
  let r = meta[1];
  if (r > 0u) { off += coords.x * meta[stridesPtr + r - 1u]; }
  if (r > 1u) { off += coords.y * meta[stridesPtr + r - 2u]; }
  if (r > 2u) { off += coords.z * meta[stridesPtr + r - 3u]; }
  return off;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lin = gid.x;
  if (lin >= meta[0]) { return; }
  let rank = meta[1];
  var c = vec3<u32>(0u, 0u, 0u);
  if (rank >= 1u) { c.x = lin % meta[4 + rank - 1u]; }
  if (rank >= 2u) { c.y = (lin / meta[4 + rank - 1u]) % meta[4 + rank - 2u]; }
  if (rank == 3u || rank == 4u) { c.z = lin / (meta[4] * meta[5]); if (rank == 3u) { c.z = 0u; } }
  if (rank == 3u) { c.z = 0u; c.y = (lin / meta[6]) % meta[5]; c.x = lin % meta[6]; }
  else if (rank == 2u) { c.y = lin / meta[5]; c.x = lin % meta[5]; }
  else if (rank == 1u) { c.x = lin; }

  let av = a[srcOffset(c, 10u)];
  let bv = b[srcOffset(c, 16u)];
  ${opName === 'add' ? 'out[lin] = av + bv;' : ''}
  ${opName === 'sub' ? 'out[lin] = av - bv;' : ''}
  ${opName === 'mul' ? 'out[lin] = av * bv;' : ''}
  ${opName === 'div' ? 'out[lin] = av / bv;' : ''}
}`;
}

// ── Unary op ────────────────────────────────────────────────────────────────
// meta: [0]=n [1]=opCode [2]=bitcast(alpha)
export function unaryShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= meta[0]) { return; }
  let v = x[i];
  let alpha = bitcast<f32>(meta[2]);
  switch meta[1] {
    case 0u: { out[i] = exp(v); }
    case 1u: { out[i] = tanh(v); }
    case 2u: { out[i] = select(1.0 / (1.0 + exp(-v)), 0.0, v != v); }
    case 3u: { out[i] = max(v, 0.0); }
    case 4u: { out[i] = select(alpha * v, v, v > 0.0); }
    case 5u: { out[i] = -v; }
    case 6u: { out[i] = ceil(v); }
    default: { out[i] = v; }
  }
}`;
}

// ── Batched matmul with batch broadcast: A[...,M,K] @ B[...,K,N] ───────────
// meta: [0]=totalOut [1]=rank(batch) [2..]=batchDims(3) [5]=M [6]=N [7]=K
//       [8]=aBatchStrides(3) [11]=bBatchStrides(3)
export function matmulShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let M = meta[5];
  let N = meta[6];
  let K = meta[7];
  if (gid.x >= N || gid.y >= M) { return; }
  let bl = gid.z;
  var ao = 0u;
  var bo = 0u;
  if (bl > 0u) { ao += (bl % meta[2]) * meta[8];
    if (bl / meta[2] > 0u) { ao += ((bl / meta[2]) % meta[3]) * meta[9]; }
    if (bl / (meta[2] * max(meta[3], 1u)) > 0u) { ao += ((bl / (meta[2] * max(meta[3], 1u))) % meta[4]) * meta[10]; }
  }
  if (bl > 0u) { bo += (bl % meta[2]) * meta[11];
    if (bl / meta[2] > 0u) { bo += ((bl / meta[2]) % meta[3]) * meta[12]; }
    if (bl / (meta[2] * max(meta[3], 1u)) > 0u) { bo += ((bl / (meta[2] * max(meta[3], 1u))) % meta[4]) * meta[13]; }
  }
  let m = gid.y;
  let n = gid.x;
  var acc = 0.0;
  for (var k = 0u; k < K; k++) {
    acc += a[ao + m * K + k] * b[bo + k * N + n];
  }
  out[(bl * M + m) * N + n] = acc;
}`;
}

// ── Conv1d (stride=1, group=1) ──────────────────────────────────────────────
// X[N,C,W] W[M,C,k] -> Y[N,M,Wout]
// meta: [0]=Wout [1]=C [2]=M [3]=Win [4]=k [5]=dil [6]=pad0 [7]=biasFlag
export function conv1dShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let Wout = meta[0];
  let C = meta[1];
  let M = meta[2];
  let Win = meta[3];
  let k = meta[4];
  let dil = meta[5];
  let pad0 = meta[6];
  if (gid.x >= Wout || gid.y >= M) { return; }
  let t = gid.x;
  let m = gid.y;
  let n = gid.z;
  var acc = 0.0;
  for (var c = 0u; c < C; c++) {
    let xb = (n * C + c) * Win;
    let wb = (m * C + c) * k;
    for (var kk = 0u; kk < k; kk++) {
      let s = i32(t) - i32(pad0) + i32(kk * dil);
      if (s >= 0 && s < i32(Win)) {
        acc += x[xb + u32(s)] * w[wb + kk];
      }
    }
  }
  if (meta[7] == 1u) { acc += bias[m]; }
  out[(n * M + m) * Wout + t] = acc;
}`;
}

// ── ConvTranspose1d (group=1) ───────────────────────────────────────────────
// X[N,C,W] W[C,M,k] -> Y[N,M,Wout]
// meta: [0]=Wout [1]=C [2]=M [3]=Win [4]=k [5]=stride [6]=pad0 [7]=biasFlag
export function convTranspose1dShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let Wout = meta[0];
  let C = meta[1];
  let M = meta[2];
  let Win = meta[3];
  let k = meta[4];
  let stride = meta[5];
  let pad0 = meta[6];
  if (gid.x >= Wout || gid.y >= M) { return; }
  let t = gid.x;
  let m = gid.y;
  let n = gid.z;
  var acc = 0.0;
  for (var c = 0u; c < C; c++) {
    let xb = (n * C + c) * Win;
    let wb = (c * M + m) * k;
    for (var kk = 0u; kk < k; kk++) {
      // t == u*stride - pad0 + kk  =>  u = (t + pad0 - kk)/stride
      let num = i32(t) + i32(pad0) - i32(kk);
      if (num >= 0 && u32(num) % stride == 0u) {
        let u = u32(num) / stride;
        if (u < Win) {
          acc += x[xb + u] * w[wb + kk];
        }
      }
    }
  }
  if (meta[7] == 1u) { acc += bias[m]; }
  out[(n * M + m) * Wout + t] = acc;
}`;
}

// ── LayerNorm over last dim ─────────────────────────────────────────────────
// meta: [0]=inner [1]=bitcast(eps)
@__PURE__ void 0;
export function layerNormShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let inner = meta[0];
  let eps = bitcast<f32>(meta[1]);
  let base = wid.x * inner;
  var sum = 0.0;
  var j = lid.x;
  loop {
    if (j >= inner) { break; }
    tile[j] = x[base + j];
    sum += tile[j];
    j += 256u;
  }
  workgroupBarrier();
  let mean = sum / f32(inner);
  var vsum = 0.0;
  j = lid.x;
  loop {
    if (j >= inner) { break; }
    let dv = tile[j] - mean;
    vsum += dv * dv;
    j += 256u;
  }
  workgroupBarrier();
  let inv = 1.0 / sqrt(vsum / f32(inner) + eps);
  j = lid.x;
  loop {
    if (j >= inner) { break; }
    out[base + j] = (tile[j] - mean) * inv * gamma[j] + beta[j];
    j += 256u;
  }
}`;
}

// ── Softmax along last dim ──────────────────────────────────────────────────
export function softmaxShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let inner = meta[0];
  let base = wid.x * inner;
  var mx = -3.4e38;
  var j = lid.x;
  loop {
    if (j >= inner) { break; }
    mx = max(mx, x[base + j]);
    j += 256u;
  }
  red[lid.x] = mx;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { red[lid.x] = max(red[lid.x], red[lid.x + stride]); }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let rowMax = red[0];
  var sum = 0.0;
  j = lid.x;
  loop {
    if (j >= inner) { break; }
    let ev = exp(x[base + j] - rowMax);
    out[base + j] = ev;
    sum += ev;
    j += 256u;
  }
  red[lid.x] = sum;
  workgroupBarrier();
  stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let total = red[0];
  j = lid.x;
  loop {
    if (j >= inner) { break; }
    out[base + j] = out[base + j] / total;
    j += 256u;
  }
}`;
}

// ── Transpose (permute) ─────────────────────────────────────────────────────
// meta: [0]=n [1]=rank, dims @2(6), perm @8(6), elemSizeLog2 @14
// Supports 4-byte elements only here (all our transposes are f32/i32).
export function transposeShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lin = gid.x;
  let rank = meta[1];
  if (lin >= meta[0]) { return; }
  var c0 = 0u; var c1 = 0u; var c2 = 0u; var c3 = 0u;
  var acc = lin;
  if (rank >= 1u) { c0 = acc % meta[2 + rank - 1u]; acc = acc / meta[2 + rank - 1u]; }
  if (rank >= 2u) { c1 = acc % meta[2 + rank - 2u]; acc = acc / meta[2 + rank - 2u]; }
  if (rank >= 3u) { c2 = acc % meta[2 + rank - 3u]; acc = acc / meta[2 + rank - 3u]; }
  if (rank >= 4u) { c3 = acc % meta[2]; }
  var coords: array<u32, 4> = array<u32, 4>(c0, c1, c2, c3);
  var off = 0u;
  var lastSrc = 0u;
  for (var d = 0u; d < rank; d++) {
    let p = meta[8 + d];
    var cs = 0u;
    if (p == 0u) { cs = coords[0]; }
    else if (p == 1u) { cs = coords[1]; }
    else if (p == 2u) { cs = coords[2]; }
    else { cs = coords[3]; }
    // accumulate using source strides stored in meta[14 + d]
    off = off * 1u;
    lastSrc = lastSrc;
    // multiply-add
    off += cs * meta[20 + d];
  }
  out[lin] = x[off];
}`;
}

// ── Slice ───────────────────────────────────────────────────────────────────
// meta: [0]=n [1]=rank, starts @2(6), steps @8(6), inStrides @14(6)
export function sliceShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lin = gid.x;
  let rank = meta[1];
  if (lin >= meta[0]) { return; }
  var c0 = 0u; var c1 = 0u; var c2 = 0u; var c3 = 0u;
  var acc = lin;
  if (rank >= 1u) { c0 = acc % meta[40 + rank - 1u]; acc = acc / meta[40 + rank - 1u]; }
  if (rank >= 2u) { c1 = acc % meta[40 + rank - 2u]; acc = acc / meta[40 + rank - 2u]; }
  if (rank >= 3u) { c2 = acc % meta[40 + rank - 3u]; acc = acc / meta[40 + rank - 3u]; }
  if (rank >= 4u) { c3 = acc % meta[40]; }
  var coords: array<u32, 4> = array<u32, 4>(c0, c1, c2, c3);
  var off = 0u;
  for (var d = 0u; d < rank; d++) {
    let st = bitcast<i32>(meta[8 + d]);   // signed step
    var cs = 0u;
    if (d == 0u) { cs = coords[0]; }
    else if (d == 1u) { cs = coords[1]; }
    else if (d == 2u) { cs = coords[2]; }
    else { cs = coords[3]; }
    let start = bitcast<i32>(meta[2 + d]);
    off += u32((i32(cs) * st)) * meta[14 + d];
    off += 0u;
    let _ = start;
  }
  out[lin] = x[u32(i32(off))];
}`;
}

// ── Concat (two inputs, any axis) ───────────────────────────────────────────
// meta: [0]=n [1]=outer [2]=dimA [3]=dimB [4]=inner
export function concat2Shader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lin = gid.x;
  if (lin >= meta[0]) { return; }
  let outer = meta[1];
  let dimA = meta[2];
  let inner = meta[4];
  let axDim = meta[3] + dimA;
  // decompose: outer, axis, inner
  let inn = lin % inner;
  let rest = lin / inner;
  let ax = rest % axDim;
  let ob = rest / axDim;
  if (ax < dimA) {
    out[lin] = a[(ob * dimA + ax) * inner + inn];
  } else {
    out[lin] = b[(ob * meta[3] + (ax - dimA)) * inner + inn];
  }
}`;
}

// ── Pad constant ────────────────────────────────────────────────────────────
// meta: [0]=outN [1]=rank [2]=bitcast(value), inDims @4(6), pads_pre @10(6),
//       outStrides @16(6), inStrides @22(6)
export function padShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lin = gid.x;
  if (lin >= meta[0]) { return; }
  let rank = meta[1];
  let value = bitcast<f32>(meta[2]);
  // decompose out coords (rank <= 3 handled; higher ranks go CPU path)
  var ok = true;
  var srcOff = 0u;
  var acc = lin;
  // iterate from last dim
  for (var step = 0u; step < rank; step++) {
    let d = rank - 1u - step;
    let od = meta[16 + d];
    let c = acc % od;
    acc = acc / od;
    let pre = bitcast<i32>(meta[10 + d]);
    let ic = i32(c) - pre;
    if (ic < 0 || u32(ic) >= meta[4 + d]) { ok = false; break; }
    srcOff += u32(ic) * meta[22 + d];
  }
  if (acc != 0u) { ok = false; }
  if (ok) { out[lin] = x[srcOff]; } else { out[lin] = value; }
}`;
}

// ── Gather along axis ───────────────────────────────────────────────────────
// meta: [0]=outN [1]=axisSize [2]=inner [3]=idxCount [4]=outer
@__PURE__ void 0;
export function gatherShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lin = gid.x;
  if (lin >= meta[0]) { return; }
  let inner = meta[2];
  let inn = lin % inner;
  let rest = lin / inner;
  let j = rest % meta[3];
  let ob = rest / meta[3];
  var ix = idx[j];
  if (ix < 0) { ix = ix + i32(meta[1]); }
  out[lin] = x[(ob * meta[1] + u32(ix)) * inner + inn];
}`;
}

// ── Compare broadcast -> bool(u32 0/1) ──────────────────────────────────────
// meta like binary: [0]=n [1]=rank [2]=op(0=less 1=equal), dims @4, sa @10, sb @16
export function compareShader(opCode: number): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

fn srcOff(coords: vec3<u32>, stridesPtr: u32, rank: u32) -> u32 {
  var off = 0u;
  if (rank == 1u) { off = coords.x * meta[stridesPtr]; }
  else if (rank == 2u) { off = coords.x * meta[stridesPtr] + coords.y * meta[stridesPtr + 1u]; }
  else if (rank == 3u) { off = coords.x * meta[stridesPtr] + coords.y * meta[stridesPtr + 1u] + coords.z * meta[stridesPtr + 2u]; }
  return off;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lin = gid.x;
  if (lin >= meta[0]) { return; }
  let rank = meta[1];
  var c = vec3<u32>(0u, 0u, 0u);
  if (rank == 1u) { c.x = lin; }
  else if (rank == 2u) { c.x = lin % meta[5]; c.y = lin / meta[5]; }
  else if (rank == 3u) { c.x = lin % meta[6]; c.y = (lin / meta[6]) % meta[5]; c.z = lin / (meta[5] * meta[6]); }
  let av = a[srcOff(c, 10u, rank)];
  let bv = b[srcOff(c, 16u, rank)];
  if (${opCode} == 0u) { out[lin] = select(0u, 1u, av < bv); }
  else { out[lin] = select(0u, 1u, av == bv); }
}`;
}

// ── Where broadcast ─────────────────────────────────────────────────────────
// cond/x/y strides at 10/16/22; dims @4; rank [1]; n [0]
export function whereShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> cond: array<u32>;
@group(0) @binding(2) var<storage, read> xv: array<f32>;
@group(0) @binding(3) var<storage, read> yv: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lin = gid.x;
  if (lin >= meta[0]) { return; }
  let rank = meta[1];
  var cc = 0u; var cx = 0u; var cy = 0u;
  if (rank == 1u) { cc = lin * meta[10]; cx = lin * meta[16]; cy = lin * meta[22]; }
  else if (rank == 2u) {
    let c1 = lin % meta[5]; let c0 = lin / meta[5];
    cc = c0 * meta[10] + c1 * meta[11];
    cx = c0 * meta[16] + c1 * meta[17];
    cy = c0 * meta[22] + c1 * meta[23];
  } else if (rank == 3u) {
    let c2 = lin % meta[6]; let tmp = lin / meta[6];
    let c1 = tmp % meta[5]; let c0 = tmp / meta[5];
    cc = c0 * meta[10] + c1 * meta[11] + c2 * meta[12];
    cx = c0 * meta[16] + c1 * meta[17] + c2 * meta[18];
    cy = c0 * meta[22] + c1 * meta[23] + c2 * meta[24];
  }
  out[lin] = select(yv[cy], xv[cx], cond[cc] != 0u);
}`;
}

// ── Cast ────────────────────────────────────────────────────────────────────
// opCodes: 0=f32->i32 1=i32->f32 2=f32->bool 3=bool->f32 4=f32->f32(copy)
export function castShader(opCode: number): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<${opCode === 1 || opCode === 3 ? 'f32' : opCode === 2 ? 'f32' : opCode === 3 ? 'u32' : 'u32'}>;
@group(0) @binding(2) var<storage, read_write> out: array<${opCode === 0 ? 'i32' : opCode === 2 ? 'u32' : opCode === 3 ? 'f32' : 'f32'}>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= meta[0]) { return; }
  ${opCode === 0 ? 'out[i] = i32(x[i]);' : ''}
  ${opCode === 1 ? 'out[i] = x[i];' : ''}
  ${opCode === 2 ? 'out[i] = select(0u, 1u, x[i] != 0.0);' : ''}
  ${opCode === 3 ? 'out[i] = select(0.0, 1.0, x[i] != 0u);' : ''}
  ${opCode === 4 ? 'out[i] = x[i];' : ''}
}`;
}

// ── CumSum (exclusive/reverse attrs baked into opCode) ─────────────────────
export function cumSumShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mode = meta[1]; // 0=inclusive-fwd 1=exclusive-fwd 2=inclusive-rev 3=exclusive-rev
  let n = meta[0];
  var acc = 0.0;
  if (mode <= 1u) {
    for (var j = 0u; j < n; j++) {
      if (mode == 1u) { out[j] = acc; acc = acc + x[j]; }
      else { acc = acc + x[j]; out[j] = acc; }
    }
  } else {
    for (var jj = 0u; jj < n; jj++) {
      let j = n - 1u - jj;
      if (mode == 3u) { out[j] = acc; acc = acc + x[j]; }
      else { acc = acc + x[j]; out[j] = acc; }
    }
  }
}`;
}

// ── Fill ────────────────────────────────────────────────────────────────────
export function fillShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= meta[0]) { return; }
  out[gid.x] = bitcast<f32>(meta[1]);
}`;
}
