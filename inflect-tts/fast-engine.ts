/**
 * Hand-specialized WebGPU decode path for Inflect-Micro-v2 (flow + HiFi-GAN
 * vocoder), replacing the generic ONNX-graph interpreter (executor.ts) for
 * this graph only. Mirrors kitten-tts-webgpu's engine.ts structure: weights
 * are uploaded once by name and looked up directly by hand-written dispatch
 * code in a fixed sequence, instead of interpreting the graph node-by-node.
 *
 * duration.onnx (text encoder + duration predictor + length regulation)
 * stays on the generic CPU GraphExecutor path — profiling showed it costs
 * ~0.5s vs the decode graph's ~35s, so hand-specializing it isn't worth the
 * correctness risk (relative-position attention, in particular).
 *
 * Architecture (see /Users/anudit/.claude/plans/floofy-mapping-allen.md):
 *  z0 = m_p + exp(logs_p) * zp_noise * noise_scale
 *  4x reverse flow blocks (WaveNet-gated affine coupling, mean_only=True)
 *  HiFi-GAN vocoder: conv_pre -> 4x {upsample ConvTranspose -> 3-resblock MRF} -> conv_post -> tanh
 */
import { parseOnnxModel } from './onnx-proto.ts';
import { rawDataToTyped, DTYPES } from './tensor.ts';
import { conv1dShader, convTranspose1dShader, unaryShader, packF32, MAX_WG } from './shaders-core.ts';
import { gatedTanhSigmoidShader, elemMaskShader, channelReverseShader } from './shaders-fast.ts';

interface FTensor { buffer: any; C: number; T: number }

const UPSAMPLE_STAGES = [
  { stride: 8, kernel: 16, inC: 320, outC: 160 },
  { stride: 8, kernel: 16, inC: 160, outC: 80 },
  { stride: 2, kernel: 4, inC: 80, outC: 40 },
  { stride: 2, kernel: 4, inC: 40, outC: 20 },
];
const RESBLOCK_KERNELS = [3, 7, 11];
const DILATIONS = [1, 3, 5];
const FLOW_INDICES = [6, 4, 2, 0]; // reverse execution order (see plan)
/** decode.onnx slices latent time to 4000 before the vocoder (`/Slice`). */
const MAX_LATENT_T = 4000;

function trimChannelFirst(src: Float32Array, C: number, T: number, T0: number): Float32Array {
  if (T <= T0) return src;
  const out = new Float32Array(C * T0);
  for (let c = 0; c < C; c++) out.set(src.subarray(c * T, c * T + T0), c * T0);
  return out;
}

export class InflectFastEngine {
  private device: any = null;
  private weights = new Map<string, { buffer: any; dims: number[] }>();
  private modules = new Map<string, any>();
  private pipelines = new Map<string, any>();
  private dummy: any = null;
  private enc: any = null;
  private tempBuffers: any[] = [];
  private deadBuffers: any[] = [];
  private bufferPool = new Map<number, any[]>();
  poisoned = false;

  async init(device: any): Promise<void> {
    this.device = device;
    // STORAGE (128) is required: this buffer stands in for unused bindings
    // (e.g. elemOp's null mask/b operand) whose shader declares them as
    // storage bindings — without it, bind group creation is rejected and
    // the dispatch silently no-ops, corrupting the output.
    this.dummy = device.createBuffer({ size: 16, usage: 128 | 8 | 4 });
    device.addEventListener('uncapturederror', (event: any) => {
      console.error('[InflectFast] GPU error:', event.error?.message ?? event.error);
      this.poisoned = true;
    });
  }

  async loadWeights(decodeOnnxBuffer: ArrayBuffer): Promise<void> {
    const g = parseOnnxModel(decodeOnnxBuffer);
    for (const [name, t] of g.initializers) {
      const type = DTYPES[t.dataType] ?? 'f32';
      const data = rawDataToTyped(type, t.rawData) as Float32Array;
      const size = Math.max(16, data.length * 4);
      const buffer = this.device.createBuffer({ size, usage: 128 | 8 | 4 });
      this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
      this.weights.set(name, { buffer, dims: t.dims });
    }
    // Compile the handful of decode kernels once so the first spoken line
    // isn't stalled (and doesn't race) on pipeline creation.
    this.pipeline('conv1d', conv1dShader());
    this.pipeline('convT1d', convTranspose1dShader());
    this.pipeline('unary', unaryShader());
    this.pipeline('gts', gatedTanhSigmoidShader());
    this.pipeline('elem', elemMaskShader());
    this.pipeline('chrev', channelReverseShader());
  }

  private w(name: string): { buffer: any; dims: number[] } {
    const t = this.weights.get(name);
    if (!t) throw new Error(`[InflectFast] missing weight ${name}`);
    return t;
  }

  // ── device plumbing (pooled allocation, batched submit — see executor.ts) ──

  private flush(): void {
    if (this.device && this.enc) {
      this.device.queue.submit([this.enc.finish()]);
      this.enc = null;
      for (const b of this.tempBuffers) b.destroy();
      this.tempBuffers = [];
    }
    for (const b of this.deadBuffers) {
      let pool = this.bufferPool.get(b.size);
      if (!pool) { pool = []; this.bufferPool.set(b.size, pool); }
      pool.push(b);
    }
    this.deadBuffers = [];
  }

  private scheduleDestroy(buffer: any): void {
    this.deadBuffers.push(buffer);
  }

  /** Destroy pooled activations so a bad line cannot leak into the next one. */
  purgePool(): void {
    try {
      if (this.device && this.enc) {
        this.device.queue.submit([this.enc.finish()]);
        this.enc = null;
      }
    } catch {}
    for (const b of this.tempBuffers) try { b.destroy(); } catch {}
    this.tempBuffers = [];
    for (const b of this.deadBuffers) try { b.destroy(); } catch {}
    this.deadBuffers = [];
    for (const [, pool] of this.bufferPool) {
      for (const b of pool) try { b.destroy(); } catch {}
    }
    this.bufferPool.clear();
    this.poisoned = false;
  }

  private uniform(words: Uint32Array): any {
    const buf = this.device.createBuffer({
      size: Math.max(256, Math.ceil(words.byteLength / 256) * 256),
      usage: 64 | 8,
    });
    this.device.queue.writeBuffer(buf, 0, words);
    this.tempBuffers.push(buf);
    return buf;
  }

  private pipeline(key: string, code: string): any {
    let p = this.pipelines.get(key);
    if (p) return p;
    let mod = this.modules.get(key);
    if (!mod) { mod = this.device.createShaderModule({ code }); this.modules.set(key, mod); }
    p = this.device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    this.pipelines.set(key, p);
    return p;
  }

  private bind(pipeline: any, buffers: any[], meta: Uint32Array): any {
    const entries: any[] = [{ binding: 0, resource: { buffer: this.uniform(meta) } }];
    for (let j = 0; j < buffers.length; j++) {
      entries.push({ binding: j + 1, resource: { buffer: buffers[j] ?? this.dummy } });
    }
    return this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  }

  private allocGpu(C: number, T: number): FTensor {
    const size = Math.max(16, Math.ceil((C * T * 4) / 4) * 4);
    const pool = this.bufferPool.get(size);
    const buffer = pool && pool.length > 0
      ? pool.pop()
      : this.device.createBuffer({ size, usage: 128 | 8 | 4 });
    return { buffer, C, T };
  }

  private uploadTensor(data: Float32Array, C: number, T: number): FTensor {
    const t = this.allocGpu(C, T);
    this.device.queue.writeBuffer(t.buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return t;
  }

  /** Debug-only: read a tensor back to CPU without disturbing its buffer. */
  async debugRead(t: FTensor): Promise<Float32Array> {
    this.flush();
    const n = t.C * t.T;
    const staging = this.device.createBuffer({ size: n * 4, usage: 1 | 8 });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(t.buffer, 0, staging, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  private enc_(): any {
    return this.enc ?? (this.enc = this.device.createCommandEncoder());
  }

  private dispatch(key: string, code: string, buffers: any[], meta: Uint32Array, n: number): void {
    const p = this.pipeline(key, code);
    const bg = this.bind(p, buffers, meta);
    const enc = this.enc_();
    const pass = enc.beginComputePass();
    pass.setPipeline(p); pass.setBindGroup(0, bg);
    const groups = Math.max(1, Math.ceil(n / 64));
    pass.dispatchWorkgroups(Math.min(MAX_WG, groups), Math.ceil(groups / MAX_WG), 1);
    pass.end();
  }

  private dispatchConv(p: any, buffers: any[], meta: Uint32Array, time: number, channels: number): void {
    const tile = MAX_WG * 8;
    const ty = Math.ceil(channels / 8);
    for (let tOff = 0; tOff < time; tOff += tile) {
      meta[8] = tOff;
      const bg = this.bind(p, buffers, meta);
      const enc = this.enc_();
      const pass = enc.beginComputePass();
      pass.setPipeline(p); pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(Math.min(tile, time - tOff) / 8), ty, 1);
      pass.end();
    }
  }

  // ── op helpers ───────────────────────────────────────────────────────────
  // None of these destroy their input buffers — callers own that decision
  // (via scheduleDestroy) since several inputs here are read more than once
  // (e.g. a residual branch), and destroying-on-read caused a real bug where
  // a buffer got queued for reclaim while still needed later in the same op.

  private conv1d(x: FTensor, weightName: string, biasName: string | null, Cout: number, k: number, dil: number): FTensor {
    const pad = Math.floor(((k - 1) * dil) / 2);
    const out = this.allocGpu(Cout, x.T);
    const wt = this.w(weightName);
    const bias = biasName ? this.w(biasName).buffer : null;
    const meta = new Uint32Array(64);
    meta[0] = x.T; meta[1] = x.C; meta[2] = Cout; meta[3] = x.T; meta[4] = k; meta[5] = dil; meta[6] = pad; meta[7] = biasName ? 1 : 0;
    const p = this.pipeline('conv1d', conv1dShader());
    this.dispatchConv(p, [x.buffer, wt.buffer, out.buffer, bias], meta, x.T, Cout);
    return out;
  }

  private convTranspose1d(x: FTensor, weightName: string, biasName: string | null, Cout: number, k: number, stride: number): FTensor {
    const pad = (k - stride) / 2;
    const Wout = x.T * stride;
    const out = this.allocGpu(Cout, Wout);
    const wt = this.w(weightName);
    const bias = biasName ? this.w(biasName).buffer : null;
    const meta = new Uint32Array(64);
    meta[0] = Wout; meta[1] = x.C; meta[2] = Cout; meta[3] = x.T; meta[4] = k; meta[5] = stride; meta[6] = pad; meta[7] = biasName ? 1 : 0;
    const p = this.pipeline('convT1d', convTranspose1dShader());
    this.dispatchConv(p, [x.buffer, wt.buffer, out.buffer, bias], meta, Wout, Cout);
    return out;
  }

  private unary(x: FTensor, opCode: number, alpha: number): FTensor {
    const out = this.allocGpu(x.C, x.T);
    const meta = new Uint32Array(64);
    meta[0] = x.C * x.T; meta[1] = opCode; meta[2] = packF32(alpha);
    this.dispatch('unary', unaryShader(), [x.buffer, out.buffer], meta, x.C * x.T);
    return out;
  }

  private leakyRelu(x: FTensor, alpha = 0.1): FTensor { return this.unary(x, 4, alpha); }
  private tanhOp(x: FTensor): FTensor { return this.unary(x, 1, 0); }

  private gatedTanhSigmoid(x: FTensor): FTensor {
    const C = x.C / 2;
    const out = this.allocGpu(C, x.T);
    const meta = new Uint32Array(64);
    meta[0] = C; meta[1] = x.T;
    this.dispatch('gts', gatedTanhSigmoidShader(), [x.buffer, out.buffer], meta, C * x.T);
    return out;
  }

  /** op: 0=a+b 1=a-b 2=copy(a) 3=(a+b)*scale 4=a*b. mask (per-timestep [T]) optional.
   *  Consumes (destroys) both `a` and `b` — callers needing to keep an input
   *  alive past this call must pass a copyChannelRange() copy instead. */
  private elemOp(a: FTensor, b: FTensor | null, mask: FTensor | null, op: number, scale = 0): FTensor {
    const out = this.allocGpu(a.C, a.T);
    const meta = new Uint32Array(64);
    meta[0] = a.C; meta[1] = a.T; meta[2] = op; meta[3] = mask ? 1 : 0; meta[4] = packF32(scale);
    this.dispatch('elem', elemMaskShader(), [a.buffer, b ? b.buffer : null, mask ? mask.buffer : null, out.buffer], meta, a.C * a.T);
    this.scheduleDestroy(a.buffer);
    if (b) this.scheduleDestroy(b.buffer);
    return out;
  }

  private copyChannelRange(x: FTensor, chanOffset: number, chanCount: number): FTensor {
    const out = this.allocGpu(chanCount, x.T);
    const enc = this.enc_();
    enc.copyBufferToBuffer(x.buffer, chanOffset * x.T * 4, out.buffer, 0, chanCount * x.T * 4);
    return out;
  }

  /** VITS flow's Flip: reverses the full channel dimension (torch.flip(x,[1])). */
  private flipChannels(x: FTensor): FTensor {
    const out = this.allocGpu(x.C, x.T);
    const meta = new Uint32Array(64);
    meta[0] = x.C; meta[1] = x.T;
    this.dispatch('chrev', channelReverseShader(), [x.buffer, out.buffer], meta, x.C * x.T);
    return out;
  }

  // ── flow (4x reverse WaveNet-gated affine coupling, mean_only) ─────────────

  private waveNet(prefix: string, h0: FTensor, mask: FTensor): FTensor {
    let h = h0;
    let skipAcc: FTensor | null = null;
    for (let i = 0; i < 4; i++) {
      const inAct = this.conv1d(h, `${prefix}.enc.in_layers.${i}.weight`, `${prefix}.enc.in_layers.${i}.bias`, 192, 5, 1);
      const gated = this.gatedTanhSigmoid(inAct);
      this.scheduleDestroy(inAct.buffer);
      const last = i === 3;
      const rs = this.conv1d(gated, `${prefix}.enc.res_skip_layers.${i}.weight`, `${prefix}.enc.res_skip_layers.${i}.bias`, last ? 96 : 192, 1, 1);
      this.scheduleDestroy(gated.buffer);
      if (!last) {
        const res = this.copyChannelRange(rs, 0, 96);
        const skip = this.copyChannelRange(rs, 96, 96);
        this.scheduleDestroy(rs.buffer);
        h = this.elemOp(h, res, mask, 0); // h = (h + res) * mask; consumes old h + res
        skipAcc = skipAcc ? this.elemOp(skipAcc, skip, null, 0) : skip;
      } else {
        this.scheduleDestroy(h.buffer); // last layer: h's residual state is never read again
        skipAcc = skipAcc ? this.elemOp(skipAcc, rs, null, 0) : rs;
      }
    }
    return this.elemOp(skipAcc!, null, mask, 2); // *mask (op=2 copy(a), b unused)
  }

  private couplingLayerReverse(x: FTensor, flowIdx: number, mask: FTensor): FTensor {
    const x0 = this.copyChannelRange(x, 0, 96);
    const x1 = this.copyChannelRange(x, 96, 96);
    this.scheduleDestroy(x.buffer);

    const prefix = `model.flow.flows.${flowIdx}`;
    let h = this.conv1d(x0, `${prefix}.pre.weight`, `${prefix}.pre.bias`, 96, 1, 1);
    h = this.elemOp(h, null, mask, 2); // consumes pre-mask h
    const wnOut = this.waveNet(prefix, h, mask);
    let m = this.conv1d(wnOut, `${prefix}.post.weight`, `${prefix}.post.bias`, 96, 1, 1);
    this.scheduleDestroy(wnOut.buffer);
    m = this.elemOp(m, null, mask, 2); // consumes pre-mask m
    const x1new = this.elemOp(x1, m, mask, 1); // (x1 - m) * mask   [logs=0 since mean_only]; consumes x1 + m

    const out = this.allocGpu(192, x.T);
    const enc = this.enc_();
    enc.copyBufferToBuffer(x0.buffer, 0, out.buffer, 0, 96 * x.T * 4);
    enc.copyBufferToBuffer(x1new.buffer, 0, out.buffer, 96 * x.T * 4, 96 * x.T * 4);
    this.scheduleDestroy(x0.buffer);
    this.scheduleDestroy(x1new.buffer);
    return out;
  }

  // ── HiFi-GAN vocoder ─────────────────────────────────────────────────────

  private resBlock(x: FTensor, resblockIdx: number, kernel: number): FTensor {
    let y = x;
    for (let d = 0; d < DILATIONS.length; d++) {
      const dil = DILATIONS[d];
      const yCopy = this.copyChannelRange(y, 0, y.C); // y also needed later for the residual add
      let xt = this.leakyRelu(yCopy);
      this.scheduleDestroy(yCopy.buffer);
      let xt2 = this.conv1d(xt, `model.dec.resblocks.${resblockIdx}.convs1.${d}.weight`, `model.dec.resblocks.${resblockIdx}.convs1.${d}.bias`, y.C, kernel, dil);
      this.scheduleDestroy(xt.buffer);
      xt = this.leakyRelu(xt2);
      this.scheduleDestroy(xt2.buffer);
      xt2 = this.conv1d(xt, `model.dec.resblocks.${resblockIdx}.convs2.${d}.weight`, `model.dec.resblocks.${resblockIdx}.convs2.${d}.bias`, y.C, kernel, 1);
      this.scheduleDestroy(xt.buffer);
      y = this.elemOp(xt2, y, null, 0); // consumes xt2 + old y
    }
    return y;
  }

  private vocoder(z: FTensor): FTensor {
    let x = this.conv1d(z, 'model.dec.conv_pre.weight', 'model.dec.conv_pre.bias', 320, 7, 1);
    this.scheduleDestroy(z.buffer);
    for (let stage = 0; stage < UPSAMPLE_STAGES.length; stage++) {
      const s = UPSAMPLE_STAGES[stage];
      const xAct = this.leakyRelu(x);
      this.scheduleDestroy(x.buffer);
      x = this.convTranspose1d(xAct, `model.dec.ups.${stage}.weight`, `model.dec.ups.${stage}.bias`, s.outC, s.kernel, s.stride);
      this.scheduleDestroy(xAct.buffer);
      const outs: FTensor[] = [];
      for (let r = 0; r < RESBLOCK_KERNELS.length; r++) {
        const xCopy = this.copyChannelRange(x, 0, x.C);
        outs.push(this.resBlock(xCopy, stage * 3 + r, RESBLOCK_KERNELS[r]));
      }
      this.scheduleDestroy(x.buffer);
      // HiFi-GAN MRF: average the 3 parallel resblocks (xs / num_kernels)
      const sum01 = this.elemOp(outs[0], outs[1], null, 0);
      x = this.elemOp(sum01, outs[2], null, 3, 1 / RESBLOCK_KERNELS.length);
    }
    // Final pre-conv_post activation is F.leaky_relu(x) — PyTorch default 0.01,
    // not LRELU_SLOPE=0.1 used inside the upsample/resblock loop. Matches ONNX
    // /dec/LeakyRelu_4.
    const xAct = this.leakyRelu(x, 0.01);
    this.scheduleDestroy(x.buffer);
    let wav = this.conv1d(xAct, 'model.dec.conv_post.weight', null, 1, 7, 1);
    this.scheduleDestroy(xAct.buffer);
    const wavT = this.tanhOp(wav);
    this.scheduleDestroy(wav.buffer);
    return wavT;
  }

  // ── entry point ──────────────────────────────────────────────────────────

  async generate(
    mP: Float32Array, logsP: Float32Array, yMaskArr: Float32Array,
    zpNoise: Float32Array, noiseScale: number, T: number,
    debug?: { z0?: Float32Array[]; postFlow?: Float32Array[] },
  ): Promise<Float32Array> {
    if (T > MAX_LATENT_T) {
      mP = trimChannelFirst(mP, 192, T, MAX_LATENT_T);
      logsP = trimChannelFirst(logsP, 192, T, MAX_LATENT_T);
      yMaskArr = yMaskArr.subarray(0, MAX_LATENT_T);
      zpNoise = trimChannelFirst(zpNoise, 192, T, MAX_LATENT_T);
      T = MAX_LATENT_T;
    }
    const scaledNoise = new Float32Array(zpNoise.length);
    for (let i = 0; i < zpNoise.length; i++) scaledNoise[i] = zpNoise[i] * noiseScale;

    const mPBuf = this.uploadTensor(mP, 192, T);
    const logsPBuf = this.uploadTensor(logsP, 192, T);
    const noiseBuf = this.uploadTensor(scaledNoise, 192, T);
    const maskBuf = this.uploadTensor(yMaskArr, 1, T);

    const expLogs = this.unary(logsPBuf, 0, 0); // exp
    this.scheduleDestroy(logsPBuf.buffer);
    const noiseTerm = this.elemOp(expLogs, noiseBuf, null, 4); // exp(logs)*scaledNoise
    let z: FTensor = this.elemOp(mPBuf, noiseTerm, null, 0); // z0 = m_p + noiseTerm

    if (debug?.z0) debug.z0.push(await this.debugRead(z));

    for (const idx of FLOW_INDICES) {
      const flipped = this.flipChannels(z);
      this.scheduleDestroy(z.buffer); // pre-flip z no longer needed
      z = this.couplingLayerReverse(flipped, idx, maskBuf); // consumes `flipped`
      if (debug?.postFlow) debug.postFlow.push(await this.debugRead(z));
    }

    z = this.elemOp(z, null, maskBuf, 2); // final z = z * y_mask (top-level Mul after the flow, before the vocoder)

    if (debug?.postFlow) debug.postFlow.push(await this.debugRead(z));

    const wav = this.vocoder(z);
    this.scheduleDestroy(maskBuf.buffer);

    this.flush();
    const n = wav.C * wav.T;
    const staging = this.device.createBuffer({ size: n * 4, usage: 1 | 8 });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(wav.buffer, 0, staging, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    wav.buffer.destroy(); // final output, safe to destroy now that the copy above has completed
    return out;
  }
}
