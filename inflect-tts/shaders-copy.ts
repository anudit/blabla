/**
 * WGSL kernels for data-movement and small-op kernels (Inflect-Micro-v2).
 * Same conventions as shaders-core.ts.
 */
import { META_BIND, LIN_IDX } from './shaders-core.ts';

// Transpose/permute, rank<=4, f32.
// meta: [0]=n [1]=rank, perm @8(4), outDims @12(4) first→last, inStrides @16(4)
// Output coords are first-dim-slowest (same as CPU transpose).
export function transposeShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let lin = i;
  let rank = getMeta(1u);
  if (lin >= getMeta(0u)) { return; }
  let d0 = getMeta(12u); let d1 = getMeta(13u); let d2 = getMeta(14u); let d3 = getMeta(15u);
  // coords[d] = output index along dim d (0 = first / slowest)
  var c0 = 0u; var c1 = 0u; var c2 = 0u; var c3 = 0u;
  if (rank >= 4u) {
    let p1 = max(d1, 1u) * max(d2, 1u) * max(d3, 1u);
    c0 = lin / p1;
    var rem = lin % p1;
    let p2 = max(d2, 1u) * max(d3, 1u);
    c1 = rem / p2; rem = rem % p2;
    c2 = rem / max(d3, 1u); c3 = rem % max(d3, 1u);
  } else if (rank == 3u) {
    let p2 = max(d1, 1u) * max(d2, 1u);
    c0 = lin / p2; var rem = lin % p2;
    c1 = rem / max(d2, 1u); c2 = rem % max(d2, 1u);
  } else if (rank == 2u) {
    c0 = lin / max(d1, 1u); c1 = lin % max(d1, 1u);
  } else { c0 = lin; }
  var coords: array<u32, 4> = array<u32, 4>(c0, c1, c2, c3);
  var off = 0u;
  for (var d = 0u; d < rank; d++) {
    // src += outCoord[d] * inStride[perm[d]]
    let p = getMeta(8u + d);
    off += coords[d] * getMeta(16u + p);
  }
  out[lin] = x[off];
}`;
}

// Slice with per-dim absolute start (i32 bits) @8(4), step(i32 bits) @12(4),
// inStrides @16(4), outDims @20(4) first→last. meta: [0]=n [1]=rank
export function sliceShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let lin = i;
  let rank = getMeta(1u);
  if (lin >= getMeta(0u)) { return; }
  let d0 = getMeta(20u); let d1 = getMeta(21u); let d2 = getMeta(22u); let d3 = getMeta(23u);
  var c0 = 0u; var c1 = 0u; var c2 = 0u; var c3 = 0u;
  if (rank >= 4u) {
    let p1 = max(d1, 1u) * max(d2, 1u) * max(d3, 1u);
    c0 = lin / p1;
    var rem = lin % p1;
    let p2 = max(d2, 1u) * max(d3, 1u);
    c1 = rem / p2; rem = rem % p2;
    c2 = rem / max(d3, 1u); c3 = rem % max(d3, 1u);
  } else if (rank == 3u) {
    let p2 = max(d1, 1u) * max(d2, 1u);
    c0 = lin / p2; var rem = lin % p2;
    c1 = rem / max(d2, 1u); c2 = rem % max(d2, 1u);
  } else if (rank == 2u) {
    c0 = lin / max(d1, 1u); c1 = lin % max(d1, 1u);
  } else { c0 = lin; }
  var coords: array<u32, 4> = array<u32, 4>(c0, c1, c2, c3);
  var off = 0i;
  for (var d = 0u; d < rank; d++) {
    let strideD = i32(getMeta(16u + d));
    off += bitcast<i32>(getMeta(8u + d)) * strideD + i32(coords[d]) * bitcast<i32>(getMeta(12u + d)) * strideD;
  }
  out[lin] = x[u32(off)];
}`;
}

// Concat of two inputs along an axis. meta: [0]=n [1]=outer [2]=dimA [3]=dimB [4]=inner
export function concat2Shader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let lin = i;
  if (lin >= getMeta(0u)) { return; }
  let dimA = getMeta(2u);
  let inner = getMeta(4u);
  let axDim = getMeta(3u) + dimA;
  let inn = lin % inner;
  let rest = lin / inner;
  let ax = rest % axDim;
  let ob = rest / axDim;
  if (ax < dimA) {
    out[lin] = a[(ob * dimA + ax) * inner + inn];
  } else {
    out[lin] = b[(ob * getMeta(3u) + (ax - dimA)) * inner + inn];
  }
}`;
}

// Pad constant. meta: [0]=outN [1]=rank [2]=bitcast(value),
//   inDims @8(4), padsPre(i32) @12(4), outDims @16(4)
export function padShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let lin = i;
  if (lin >= getMeta(0u)) { return; }
  let rank = getMeta(1u);
  let value = bitcast<f32>(getMeta(2u));
  var ok = true;
  var srcOff = 0u;
  var rem = lin;
  for (var step = 0u; step < rank; step++) {
    let d = rank - 1u - step;
    let od = getMeta(16u + d);
    let c = rem % od;
    rem = rem / od;
    let ic = i32(c) - bitcast<i32>(getMeta(12u + d));
    if (ic < 0 || u32(ic) >= getMeta(8u + d)) { ok = false; break; }
    // accumulate input offset using input strides derived on CPU into m[20+d]
    srcOff += u32(ic) * getMeta(20u + d);
  }
  if (rem != 0u) { ok = false; }
  if (ok) { out[lin] = x[srcOff]; } else { out[lin] = value; }
}`;
}

// Gather along axis. meta: [0]=outN [1]=axisSize [2]=inner [3]=idxCount
export function gatherShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let lin = i;
  if (lin >= getMeta(0u)) { return; }
  let inner = getMeta(2u);
  let inn = lin % inner;
  let rest = lin / inner;
  let j = rest % getMeta(3u);
  let ob = rest / getMeta(3u);
  var ix = idx[j];
  if (ix < 0) { ix = ix + i32(getMeta(1u)); }
  out[lin] = x[(ob * getMeta(1u) + u32(ix)) * inner + inn];
}`;
}

// Compare broadcast -> bool(u32). meta like binary: [0]=n [1]=rank,
// dims @8(4), sa @16(4), sb @24(4). opCode baked at build time.
export function compareShader(opCode: number): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

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
  if (${opCode} == 0u) { out[lin] = select(0u, 1u, av < bv); }
  else { out[lin] = select(0u, 1u, av == bv); }
}`;
}

// Where(cond,x,y) broadcast select, fixed 4-D.
// dims @8(4), cond strides @16(4), x strides @24(4), y strides @32(4).
// meta: [0]=n [1]=rank(=4)
export function whereShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> cond: array<u32>;
@group(0) @binding(2) var<storage, read> xv: array<f32>;
@group(0) @binding(3) var<storage, read> yv: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn srcOff(c0: u32, c1: u32, c2: u32, c3: u32, ptr: u32) -> u32 {
  return c3 * getMeta(ptr) + c2 * getMeta(ptr + 1u) + c1 * getMeta(ptr + 2u) + c0 * getMeta(ptr + 3u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let lin = i;
  if (lin >= getMeta(0u)) { return; }
  var rem = lin;
  let c3 = lin / (getMeta(9u) * getMeta(10u) * getMeta(11u));
  rem = lin % (getMeta(9u) * getMeta(10u) * getMeta(11u));
  let c2 = rem / (getMeta(10u) * getMeta(11u));
  rem = rem % (getMeta(10u) * getMeta(11u));
  let c1 = rem / getMeta(11u);
  let c0 = rem % getMeta(11u);
  let cv = cond[srcOff(c0, c1, c2, c3, 16u)];
  let xvv = xv[srcOff(c0, c1, c2, c3, 24u)];
  let yvv = yv[srcOff(c0, c1, c2, c3, 32u)];
  out[lin] = select(yvv, xvv, cv != 0u);
}`;
}

// CumSum over flattened tensor. mode m[1]: 0 incl-fwd, 1 excl-fwd, 2 incl-rev, 3 excl-rev
export function cumSumShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mode = getMeta(1u);
  let n = getMeta(0u);
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

// Fill constant. meta: [0]=n [1]=bitcast(value)
export function fillShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  if (i >= getMeta(0u)) { return; }
  out[i] = bitcast<f32>(getMeta(1u));
}`;
}
