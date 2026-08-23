/**
 * WGSL kernels for the hand-specialized Inflect-Micro-v2 decode path
 * (flow + HiFi-GAN vocoder). Complements shaders-core.ts, which already
 * covers conv1d/convTranspose1d/unary(leakyrelu,tanh)/binary ops reused here.
 */

import { META_BIND, LIN_IDX } from './shaders-core.ts';
export { META_BIND };

/**
 * WaveNet gated activation unit: out[c,t] = tanh(x[c,t]) * sigmoid(x[C+c,t]).
 * Fuses what would otherwise be 2 unary dispatches + 1 mul into one.
 * meta: [0]=C (half-channels) [1]=T
 */
export function gatedTanhSigmoidShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let idx = i;
  let C = getMeta(0u);
  let T = getMeta(1u);
  if (idx >= C * T) { return; }
  let c = idx / T;
  let t = idx % T;
  let a = x[c * T + t];
  let b = x[(C + c) * T + t];
  out[idx] = tanh(a) * (1.0 / (1.0 + exp(-b)));
}`;
}

/**
 * VITS flow's Flip: reverses the entire channel dimension (torch.flip(x,[1])),
 * NOT a two-half swap — out[c,t] = x[C-1-c, t].
 * meta: [0]=C [1]=T
 */
export function channelReverseShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let idx = i;
  let C = getMeta(0u);
  let T = getMeta(1u);
  if (idx >= C * T) { return; }
  let c = idx / T;
  let t = idx % T;
  out[idx] = x[(C - 1u - c) * T + t];
}`;
}

/**
 * Fused elementwise op over [C,T] channel-first tensors with optional
 * per-timestep mask broadcast ([T], multiplied across all channels).
 * meta: [0]=C [1]=T [2]=op [3]=useMask [4]=bitcast(scale)
 * op: 0=a+b  1=a-b  2=a (copy)  3=(a+b)*scale  4=a*b
 */
export function elemMaskShader(): string {
  return /* wgsl */ `
${META_BIND}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${LIN_IDX}
  let idx = i;
  let C = getMeta(0u);
  let T = getMeta(1u);
  let op = getMeta(2u);
  let useMask = getMeta(3u);
  let scale = bitcast<f32>(getMeta(4u));
  if (idx >= C * T) { return; }
  let t = idx % T;
  var v: f32;
  if (op == 0u) { v = a[idx] + b[idx]; }
  else if (op == 1u) { v = a[idx] - b[idx]; }
  else if (op == 2u) { v = a[idx]; }
  else if (op == 3u) { v = (a[idx] + b[idx]) * scale; }
  else { v = a[idx] * b[idx]; }
  if (useMask == 1u) { v = v * mask[t]; }
  out[idx] = v;
}`;
}
