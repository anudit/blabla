/**
 * WGSL compute kernels for Inflect-Micro-v2 (core math ops).
 * Tensors are contiguous row-major float32 storage buffers.
 * Per-dispatch metadata travels through a uniform buffer of u32 words;
 * floats are bit-packed into u32 words on the CPU side.
 */

/** Pack a number into a u32 word (f32 bits). */
export function packF32(v: number): number {
  const b = new ArrayBuffer(4);
  new Float32Array(b)[0] = v;
  return new Uint32Array(b)[0];
}

export const META_BIND = 'struct Meta { data: array<vec4<u32>, 16> };\n@group(0) @binding(0) var<uniform> meta_buf: Meta;\nfn getMeta(i: u32) -> u32 { return meta_buf.data[i / 4u][i % 4u]; }';

/** 1D element index when dispatching (min(65535, n/64), ceil((n/64)/65535)). workgroup_size(64). */
export const LIN_IDX = 'let i = gid.y * 4194240u + gid.x;';

/** WebGPU max workgroups per dimension. */
export const MAX_WG = 65535;

// Broadcast binary op. meta: [0]=n [1]=rank(max4), outDims @8(4), sa @16(4), sb @24(4)
export function binaryShader(opName: 'add' | 'sub' | 'mul' | 'div'): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn srcOff(c0: u32, c1: u32, c2: u32, c3: u32, ptr: u32) -> u32 {
  let rank = getMeta(1u);
  if (rank == 1u) { return c0 * getMeta(ptr); }
  if (rank == 2u) { return c1 * getMeta(ptr) + c0 * getMeta(ptr + 1u); }
  if (rank == 3u) { return c2 * getMeta(ptr) + c1 * getMeta(ptr + 1u) + c0 * getMeta(ptr + 2u); }
  return c3 * getMeta(ptr) + c2 * getMeta(ptr + 1u) + c1 * getMeta(ptr + 2u) + c0 * getMeta(ptr + 3u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let lin = i;
  let rank = getMeta(1u);
  if (lin >= getMeta(0u)) { return; }
  var rem = lin;
  var c3 = 0u; var c2 = 0u; var c1 = 0u; var c0 = 0u;
  if (rank >= 4u) {
    c3 = lin / (getMeta(9u) * getMeta(10u) * getMeta(11u));
    rem = lin % (getMeta(9u) * getMeta(10u) * getMeta(11u));
  }
  if (rank >= 3u) { c2 = rem / (getMeta(10u) * getMeta(11u)); rem = rem % (getMeta(10u) * getMeta(11u)); }
  if (rank >= 2u) { c1 = rem / getMeta(11u); }
  c0 = rem % getMeta(11u);
  let av = a[srcOff(c0, c1, c2, c3, 16u)];
  let bv = b[srcOff(c0, c1, c2, c3, 24u)];
  ${opName === 'add' ? 'out[lin] = av + bv;' : ''}
  ${opName === 'sub' ? 'out[lin] = av - bv;' : ''}
  ${opName === 'mul' ? 'out[lin] = av * bv;' : ''}
  ${opName === 'div' ? 'out[lin] = av / bv;' : ''}
}`;
}

// Unary op. meta: [0]=n [1]=opCode [2]=bitcast(alpha)
// codes: 0 exp, 1 tanh, 2 sigmoid, 3 relu, 4 leakyrelu, 5 neg, 6 ceil
export function unaryShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  if (i >= getMeta(0u)) { return; }
  let v = x[i];
  let alpha = bitcast<f32>(getMeta(2u));
  switch getMeta(1u) {
    case 0u: { out[i] = exp(v); }
    case 1u: { out[i] = tanh(v); }
    case 2u: { out[i] = 1.0 / (1.0 + exp(-v)); }
    case 3u: { out[i] = max(v, 0.0); }
    case 4u: { out[i] = select(alpha * v, v, v > 0.0); }
    case 5u: { out[i] = -v; }
    case 6u: { out[i] = ceil(v); }
    default: { out[i] = v; }
  }
}`;
}

// Batched matmul with batch broadcast: A[...,M,K] @ B[...,K,N], batch rank <= 2.
// meta: [0]=M [1]=N [2]=K [3]=batchTotal
//   [4]=batchA0 [5]=batchA1 [6]=batchB0 [7]=batchB1
//   [8]=aStride0 [9]=aStride1 [10]=bStride0 [11]=bStride1
export function matmulShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let M = getMeta(0u);
  let N = getMeta(1u);
  let K = getMeta(2u);
  if (gid.x >= N || gid.y >= M || gid.z >= getMeta(3u)) { return; }
  let bl = gid.z;
  var ao = 0u;
  var bo = 0u;
  if (bl > 0u) {
    let dA0 = max(getMeta(4u), 1u);
    let dB0 = max(getMeta(6u), 1u);
    let i0a = bl % dA0;
    let i0b = bl % dB0;
    ao += i0a * getMeta(8u) + ((bl / dA0) % max(getMeta(5u), 1u)) * getMeta(9u);
    bo += i0b * getMeta(10u) + ((bl / dB0) % max(getMeta(7u), 1u)) * getMeta(11u);
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

/**
 * Time steps computed per conv thread. Each thread keeps one weight value in a
 * register and applies it to `CONV_TIME_PER_THREAD` consecutive outputs, so the
 * kernel reads the weight tensor 4x less often and launches 4x fewer threads.
 * Both dispatch sites (fast-engine's dispatchConv and the executor's generic
 * Conv case) must divide their time extent by this.
 */
export const CONV_TIME_PER_THREAD = 4;

/** Time steps covered by one conv workgroup (workgroup_size(8,8,1) on x). */
export const CONV_TIME_PER_GROUP = 8 * CONV_TIME_PER_THREAD;

// Conv1d (stride=1, group=1). X[N,C,W] W[M,C,k] -> Y[N,M,Wout]
// meta: [0]=Wout [1]=C [2]=M [3]=Win [4]=k [5]=dil [6]=pad0 [7]=biasFlag
//       [8]=time offset of this tile, in output elements
export function conv1dShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let Wout = getMeta(0u);
  let C = getMeta(1u);
  let M = getMeta(2u);
  let Win = getMeta(3u);
  let k = getMeta(4u);
  let dil = getMeta(5u);
  let pad0 = getMeta(6u);
  let biasFlag = getMeta(7u);
  let row = gid.y;
  let n = gid.z;
  let t0 = getMeta(8u) + gid.x * ${CONV_TIME_PER_THREAD}u;
  if (t0 >= Wout || row >= M) { return; }
  let iWin = i32(Win);
  var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
  for (var c = 0u; c < C; c++) {
    let xb = (n * C + c) * Win;
    let wb = (row * C + c) * k;
    for (var kk = 0u; kk < k; kk++) {
      // ONNX Conv: in = t * stride + kk * dilation - pad  (stride=1 here)
      let wv = w[wb + kk];
      let s = i32(t0) - i32(pad0) + i32(kk * dil);
      if (s >= 0 && s + 3 < iWin) {
        // Interior: all four taps are in range, so skip the per-tap bounds test.
        let xo = xb + u32(s);
        a0 += x[xo] * wv;
        a1 += x[xo + 1u] * wv;
        a2 += x[xo + 2u] * wv;
        a3 += x[xo + 3u] * wv;
      } else {
        if (s >= 0 && s < iWin) { a0 += x[xb + u32(s)] * wv; }
        let s1 = s + 1; if (s1 >= 0 && s1 < iWin) { a1 += x[xb + u32(s1)] * wv; }
        let s2 = s + 2; if (s2 >= 0 && s2 < iWin) { a2 += x[xb + u32(s2)] * wv; }
        let s3 = s + 3; if (s3 >= 0 && s3 < iWin) { a3 += x[xb + u32(s3)] * wv; }
      }
    }
  }
  var b = 0.0;
  if (biasFlag == 1u) { b = bias[row]; }
  let ob = (n * M + row) * Wout + t0;
  out[ob] = a0 + b;
  if (t0 + 1u < Wout) { out[ob + 1u] = a1 + b; }
  if (t0 + 2u < Wout) { out[ob + 2u] = a2 + b; }
  if (t0 + 3u < Wout) { out[ob + 3u] = a3 + b; }
}`;
}

// ConvTranspose1d (group=1). X[N,C,W] W[C,M,k] -> Y[N,M,Wout]
// meta: [0]=Wout [1]=C [2]=M [3]=Win [4]=k [5]=stride [6]=pad0 [7]=biasFlag
//       [8]=time offset of this tile, in output elements
export function convTranspose1dShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let Wout = getMeta(0u);
  let C = getMeta(1u);
  let M = getMeta(2u);
  let Win = getMeta(3u);
  let k = getMeta(4u);
  let stride = getMeta(5u);
  let pad0 = getMeta(6u);
  let biasFlag = getMeta(7u);
  let t = gid.x + getMeta(8u);
  let row = gid.y;
  let n = gid.z;
  if (t >= Wout || row >= M) { return; }
  // A tap contributes only when (t + pad - kk) is a non-negative multiple of
  // stride, i.e. kk = p (mod stride). Walking kk in stride-sized steps from
  // that residue visits exactly the contributing taps — the naive loop over all
  // k tested and discarded (stride - 1) of every stride taps, which for the two
  // 8x upsample stages meant 7 of every 8 iterations did nothing.
  let p = i32(t) + i32(pad0);
  let istride = i32(stride);
  let ik = i32(k);
  let iWin = i32(Win);
  let kk0 = p % istride;
  var acc = 0.0;
  for (var c = 0u; c < C; c++) {
    let xb = (n * C + c) * Win;
    let wb = (c * M + row) * k;
    var kk = kk0;
    var u = (p - kk0) / istride; // drops by 1 for every stride added to kk
    while (kk < ik) {
      if (u >= 0 && u < iWin) {
        acc += x[xb + u32(u)] * w[wb + u32(kk)];
      }
      kk += istride;
      u -= 1;
    }
  }
  if (biasFlag == 1u) { acc += bias[row]; }
  out[(n * M + row) * Wout + t] = acc;
}`;
}

// LayerNorm over last dim. meta: [0]=inner [1]=bitcast(eps)
export function layerNormShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let inner = getMeta(0u);
  let eps = bitcast<f32>(getMeta(1u));
  let base = wid.x * inner;

  var sum = 0.0;
  var j = lid.x;
  loop {
    if (j >= inner) { break; }
    sum += x[base + j];
    j += 256u;
  }
  red[lid.x] = sum;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let mean = red[0] / f32(inner);

  var vsum = 0.0;
  j = lid.x;
  loop {
    if (j >= inner) { break; }
    let dv = x[base + j] - mean;
    vsum += dv * dv;
    j += 256u;
  }
  red[lid.x] = vsum;
  workgroupBarrier();
  stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = 1.0 / sqrt(red[0] / f32(inner) + eps);
  j = lid.x;
  loop {
    if (j >= inner) { break; }
    out[base + j] = (x[base + j] - mean) * inv * gamma[j] + beta[j];
    j += 256u;
  }
}`;
}

// Softmax along last dim. meta: [0]=inner
export function softmaxShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let inner = getMeta(0u);
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
