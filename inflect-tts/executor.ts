/**
 * ONNX graph executor for Inflect-Micro-v2.
 *
 * Hybrid execution: every op runs through a pure-TypeScript CPU path
 * (ops-cpu.ts); large activations are accelerated with WebGPU compute
 * kernels when a device is available. Small/control-flow tensors stay on
 * the CPU to avoid round-trips. Falls back to 100% CPU when no WebGPU.
 */
import {
  parseOnnxModel,
  DT_FLOAT, DT_INT32, DT_INT64, DT_BOOL,
  type ParsedGraph, type ParsedNode, type ParsedTensor,
} from './onnx-proto.ts';
import {
  type CpuTensor, type DType, DTYPES, allocData, makeTensor, numel,
  rawDataToTyped, broadcastShapes, broadcastStrides, contiguousStrides,
} from './tensor.ts';
import * as cpu from './ops-cpu.ts';
import {
  binaryShader, unaryShader, matmulShader, conv1dShader,
  convTranspose1dShader, layerNormShader, softmaxShader, packF32, MAX_WG,
} from './shaders-core.ts';
import {
  transposeShader, sliceShader, concat2Shader, padShader, gatherShader,
  compareShader, whereShader, fillShader,
} from './shaders-copy.ts';

export { parseOnnxModel };
export type { ParsedGraph };

/** Tensors smaller than this many elements run on the CPU. */
const SMALL = 4096;

// `process` does not exist in browsers/workers — read env flags defensively so a
// debug check never throws a ReferenceError mid-graph.
const env: Record<string, string | undefined> =
  typeof process !== 'undefined' && process.env ? process.env : {};
const TTS_DEBUG = !!env.TTS_DEBUG;

export interface GpuTensor {
  buffer: any; // GPUBuffer
  dims: number[];
  type: DType;
}

type Cell =
  | { loc: 'cpu'; t: CpuTensor; initName?: string }
  | { loc: 'gpu'; t: GpuTensor };

function cellNumel(c: Cell): number {
  return numel(c.t.dims);
}

function pad4(dims: number[]): number[] {
  const out = dims.slice();
  while (out.length < 4) out.unshift(1);
  return out.slice(0, 4);
}

/** Left-pad a per-axis vector (starts, pads, steps) to rank 4 with `fill`. */
function padLeading(arr: number[], fill: number, rank = 4): number[] {
  const out = arr.slice();
  while (out.length < rank) out.unshift(fill);
  return out.slice(0, rank);
}

export class GraphExecutor {
  graph: ParsedGraph;
  device: any = null;
  private initCache = new Map<string, CpuTensor>();
  private weightGpu = new Map<string, GpuTensor>();
  private modules = new Map<string, any>();
  private pipelines = new Map<string, any>();
  private dummy: any = null;
  private enc: any = null;
  private tempBuffers: any[] = [];
  private deadBuffers: any[] = [];
  private protectedBuffers = new Set<any>();
  // Buffer pool: reuse GPU buffers by byte size instead of destroy+reallocate.
  // Avoids per-node createBuffer/destroy churn, which is what made frequent
  // flushing (needed to bound peak memory) prohibitively slow.
  private bufferPool = new Map<number, any[]>();
  poisoned = false;
  debugTargets: string[] | null = null;

  constructor(modelBuffer: ArrayBuffer) {
    this.graph = parseOnnxModel(modelBuffer);
    const dbg = env.TTS_DEBUG_NODES;
    this.debugTargets = dbg ? dbg.split('|') : null;
  }

  /** Provide a GPUDevice to enable the WebGPU path, or null for pure CPU. */
  async init(device: any): Promise<void> {
    this.device = device;
    if (device) {
      // STORAGE (128) included: this stands in for unused storage bindings
      // in bind() — without it, a caller passing a null buffer entry would
      // fail bind-group validation (currently unexercised, but unsafe).
      this.dummy = device.createBuffer({ size: 16, usage: 128 | 8 | 4 });
      device.addEventListener('uncapturederror', (event: any) => {
        console.error('[InflectTTS] GPU error:', event.error?.message ?? event.error);
        this.poisoned = true;
      });
    }
  }

  // ── device plumbing ───────────────────────────────────────────────────────

  private flush(): void {
    if (this.device && this.enc) {
      this.device.queue.submit([this.enc.finish()]);
      this.enc = null;
      for (const b of this.tempBuffers) b.destroy();
      this.tempBuffers = [];
    }
    // GPU buffers whose last consumer already ran can only be reclaimed once
    // the commands that reference them have actually been submitted above.
    // Return them to the size-keyed pool instead of destroying — reuse avoids
    // GPU driver overhead from constant alloc/free churn, so flush() stays
    // cheap enough to call often (bounding peak memory without hurting speed).
    for (const b of this.deadBuffers) {
      let pool = this.bufferPool.get(b.size);
      if (!pool) { pool = []; this.bufferPool.set(b.size, pool); }
      pool.push(b);
    }
    this.deadBuffers = [];
  }

  /** Reclaim a GPU buffer once any commands referencing it are flushed. */
  private scheduleDestroy(buffer: any): void {
    this.deadBuffers.push(buffer);
  }

  /** Drop pooled activations so a failed/overflow generate cannot leak into the next line. */
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

  private dispatch1d(p: any, bg: any, n: number): void {
    const enc = this.enc ?? (this.enc = this.device.createCommandEncoder());
    const pass = enc.beginComputePass();
    pass.setPipeline(p); pass.setBindGroup(0, bg);
    const groups = Math.max(1, Math.ceil(n / 64));
    pass.dispatchWorkgroups(Math.min(MAX_WG, groups), Math.ceil(groups / MAX_WG), 1);
    pass.end();
  }

  private uniform(words: Uint32Array): any {
    const buf = this.device.createBuffer({
      size: Math.max(256, Math.ceil(words.byteLength / 256) * 256),
      usage: 64 | 8, // UNIFORM | COPY_DST
    });
    this.device.queue.writeBuffer(buf, 0, words);
    this.tempBuffers.push(buf);
    return buf;
  }

  private pipeline(key: string, code: string, nBindings: number): any {
    let p = this.pipelines.get(key);
    if (p) return p;
    let mod = this.modules.get(key);
    if (!mod) {
      mod = this.device.createShaderModule({ code });
      this.modules.set(key, mod);
    }
    // Auto layout: each shader declares its own per-binding access mode
    // (read vs read_write), so let WebGPU derive the matching bind group
    // layout instead of forcing every binding to read_write.
    p = this.device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    this.pipelines.set(key, p);
    return p;
  }

  private bind(pipeline: any, buffers: any[], meta: Uint32Array | null): any {
    const entries: any[] = [{ binding: 0, resource: { buffer: meta ? this.uniform(meta) : this.uniform(new Uint32Array(64)) } }];
    for (let j = 0; j < buffers.length; j++) {
      entries.push({ binding: j + 1, resource: { buffer: buffers[j] ?? this.dummy } });
    }
    return this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  }

  private allocGpu(type: DType, dims: number[]): GpuTensor {
    const bytes = Math.max(16, numel(dims) * 4);
    const size = Math.ceil(bytes / 4) * 4;
    const pool = this.bufferPool.get(size);
    const buffer = pool && pool.length > 0
      ? pool.pop()
      : this.device.createBuffer({ size, usage: 128 | 8 | 4 }); // GPUBufferUsage.STORAGE | COPY_DST | COPY_SRC
    return { buffer, dims, type };
  }

  async download(g: GpuTensor): Promise<CpuTensor> {
    this.flush();
    const n = Math.max(4, numel(g.dims));
    const staging = this.device.createBuffer({ size: n * 4, usage: 1 | 8 }); // MAP_READ | COPY_DST
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(g.buffer, 0, staging, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const raw = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return this.u32ToTyped(raw, g.type, g.dims);
  }

  private u32ToTyped(raw: Uint32Array, type: DType, dims: number[]): CpuTensor {
    const n = numel(dims);
    if (type === 'f32') {
      return { dims, type, data: new Float32Array(raw.buffer.slice(0, n * 4)) };
    }
    if (type === 'i64') {
      const i32 = new Int32Array(raw.buffer, 0, n);
      const out = new BigInt64Array(n);
      for (let j = 0; j < n; j++) out[j] = BigInt(i32[j]);
      return { dims, type, data: out };
    }
    if (type === 'i32') {
      return { dims, type, data: new Int32Array(raw.buffer.slice(0, n * 4)) };
    }
    const u8 = new Uint8Array(n);
    for (let j = 0; j < n; j++) u8[j] = raw[j] ? 1 : 0;
    return { dims, type, data: u8 };
  }

  /** Convert CPU data to a flat u32 word array for GPU upload. */
  private static typedToU32(t: CpuTensor): Uint32Array {
    const n = t.data.length;
    const out = new Uint32Array(n);
    if (t.type === 'f32') {
      new Float32Array(out.buffer).set(t.data as Float32Array);
    } else if (t.type === 'i64') {
      const src = t.data as BigInt64Array;
      for (let j = 0; j < n; j++) out[j] = Number(src[j]) >>> 0;
    } else if (t.type === 'i32') {
      out.set(new Uint32Array((t.data as Int32Array).buffer, 0, n));
    } else {
      const src = t.data as Uint8Array;
      for (let j = 0; j < n; j++) out[j] = src[j];
    }
    return out;
  }

  private upload(t: CpuTensor): GpuTensor {
    const g = this.allocGpu(t.type, t.dims);
    const words = GraphExecutor.typedToU32(t);
    this.device.queue.writeBuffer(g.buffer, 0, words.buffer, words.byteOffset, words.byteLength);
    return g;
  }

  // ── value access ──────────────────────────────────────────────────────────

  private initializerCell(name: string): Cell {
    let t = this.initCache.get(name);
    if (!t) {
      const raw = this.graph.initializers.get(name)!;
      const type = DTYPES[raw.dataType] ?? 'f32';
      t = { dims: raw.dims, type, data: rawDataToTyped(type, raw.rawData) };
      this.initCache.set(name, t);
    }
    return { loc: 'cpu', t, initName: name };
  }

  getCell(cells: Map<string, Cell>, name: string): Cell | null {
    if (cells.has(name)) return cells.get(name)!;
    if (this.graph.initializers.has(name)) return this.initializerCell(name);
    return null;
  }

  private async toCpu(cell: Cell): Promise<CpuTensor> {
    if (cell.loc === 'cpu') return cell.t;
    return await this.download(cell.t);
  }

  private toGpu(cell: Cell): GpuTensor {
    if (cell.loc === 'gpu') return cell.t;
    // Weights/biases (initializers) are immutable across every generate()
    // call on this executor — upload once and reuse the buffer instead of
    // re-uploading the same tensor from CPU on every single call.
    if (cell.initName) {
      const cached = this.weightGpu.get(cell.initName);
      if (cached) return cached;
      const g = this.upload(cell.t);
      this.weightGpu.set(cell.initName, g);
      this.protectedBuffers.add(g.buffer);
      // Drop the CPU copy now that the weight lives on GPU — halves the
      // memory footprint for every uploaded weight tensor (re-decoded from
      // the raw ONNX bytes on the rare path that still needs it CPU-side).
      this.initCache.delete(cell.initName);
      return g;
    }
    return this.upload(cell.t);
  }

  // ── main loop ─────────────────────────────────────────────────────────────

  async run(
    inputs: Map<string, CpuTensor>,
    outputNames: string[],
  ): Promise<Map<string, CpuTensor>> {
    if (this.poisoned) this.purgePool();
    const cells = new Map<string, Cell>();
    for (const [name, t] of inputs) cells.set(name, { loc: 'cpu', t });

    const nodes = this.graph.nodes;
    // remaining uses per value (for freeing GPU memory mid-run)
    const uses = new Map<string, number>();
    const bump = (name: string, by = 1) => uses.set(name, (uses.get(name) ?? 0) + by);
    for (const n of nodes) for (const inp of n.inputs) if (inp) bump(inp);
    for (const o of outputNames) bump(o);

    for (let ni = 0; ni < nodes.length; ni++) {
      const node = nodes[ni];
      try {
        await this.execNode(node, ni, cells, outputNames);
        if (TTS_DEBUG && this.debugTargets?.length) {
          const hit = node.outputs.find((o) => this.debugTargets!.includes(o));
          if (hit && cells.has(hit)) {
            const t = await this.toCpu(cells.get(hit)!);
            let mx = 0;
            const d = t.data;
            for (let j = 0; j < d.length; j++) mx = Math.max(mx, Math.abs(Number(d[j])));
            console.log(`[TTS_DEBUG] #${ni} ${node.opType} ${hit} dims=${JSON.stringify(t.dims)} maxAbs=${mx.toExponential(3)}`);
          }
        }
      } catch (err: any) {
        throw new Error(`[${node.opType} "${node.name}"] ${err?.message ?? err}`, { cause: err });
      }
      // free dead inputs
      for (const inp of node.inputs) {
        if (!inp) continue;
        const left = (uses.get(inp) ?? 1) - 1;
        uses.set(inp, left);
        if (left <= 0 && cells.has(inp) && !this.graph.initializers.has(inp)) {
          const c = cells.get(inp)!;
          if (c.loc === 'gpu' && !this.protectedBuffers.has(c.t.buffer)) this.scheduleDestroy(c.t.buffer);
          cells.delete(inp);
        }
      }
    }

    this.flush();
    const out = new Map<string, CpuTensor>();
    for (const name of outputNames) {
      const c = cells.get(name);
      if (!c) throw new Error(`Output ${name} was not produced`);
      out.set(name, await this.toCpu(c));
    }
    return out;
  }

  /** Decide whether this node runs on the GPU. */
  private wantGpu(node: ParsedNode, cells: Map<string, Cell>, outElems: number): boolean {
    if (!this.device) return false;
    if (outElems <= SMALL) return false;
    switch (node.opType) {
      case 'Add': case 'Sub': case 'Mul': case 'Div':
      case 'Exp': case 'Tanh': case 'Sigmoid': case 'Relu':
      case 'LeakyRelu': case 'Neg': case 'Ceil':
      case 'MatMul': case 'Conv': case 'ConvTranspose':
      case 'LayerNormalization': case 'Softmax':
      case 'Transpose': case 'Slice': case 'Pad': case 'Gather':
        return true;
      case 'Concat':
        return node.outputs.length === 1;
      case 'Where': case 'Less': case 'Equal': {
        // float-only fast path
        const xName = node.opType === 'Where' ? node.inputs[1] : node.inputs[0];
        const yName = node.opType === 'Where' ? node.inputs[2] : node.inputs[1];
        const xc = xName ? this.getCell(cells, xName) : null;
        const yc = yName ? this.getCell(cells, yName) : null;
        return !!xc && !!yc && xc.t.type !== 'i64' && xc.t.type !== 'i32'
          && yc.t.type !== 'i64' && yc.t.type !== 'i32';
      }
      default:
        return false;
    }
  }

  private async input(cell: Cell, wantGpu: boolean): Promise<{ buf: any; cell: Cell }> {
    if (wantGpu) {
      const g = this.toGpu(cell);
      return { buf: g.buffer, cell: { loc: 'gpu', t: g } };
    }
    const t = await this.toCpu(cell);
    return { buf: null, cell: { loc: 'cpu', t } };
  }

  private store(cells: Map<string, Cell>, name: string, cell: Cell): void {
    const prev = cells.get(name);
    if (prev && prev.loc === 'gpu' && prev.t.buffer !== (cell as any).t?.buffer) this.scheduleDestroy(prev.t.buffer);
    cells.set(name, cell);
  }

  // ── per-op execution ──────────────────────────────────────────────────────

  private async execNode(
    node: ParsedNode, _ni: number, cells: Map<string, Cell>, outputs: string[],
  ): Promise<void> {
    const op = node.opType;
    const A = (j: number) => (node.inputs[j] ? this.getCell(cells, node.inputs[j]) : null);

    // Metadata-only ops first (work regardless of device, never move data)
    switch (op) {
      case 'Constant': {
        const attr = node.attrs.get('value');
        const t = attr!.tensor!;
        const type = DTYPES[t.dataType] ?? 'f32';
        this.store(cells, node.outputs[0], { loc: 'cpu', t: { dims: t.dims, type, data: rawDataToTyped(type, t.rawData) } });
        return;
      }
      case 'Shape': {
        const x = A(0)!;
        const dims = x.t.dims;
        const data = new BigInt64Array(dims.length);
        dims.forEach((d, j) => { data[j] = BigInt(d); });
        this.store(cells, node.outputs[0], { loc: 'cpu', t: { dims: [dims.length], type: 'i64', data } });
        return;
      }
      case 'Reshape': {
        const x = A(0)!;
        const shapeT = (await this.toCpu(A(1)!));
        const shape = cpu.asNumbers(shapeT);
        const ndims = cpu.reshapeMeta(x.t.dims, shape);
        if (x.loc === 'gpu') this.protectedBuffers.add((x.t as any).buffer);
        this.store(cells, node.outputs[0], { loc: x.loc, t: { ...(x.t as any), dims: ndims } });
        return;
      }
      case 'Squeeze': {
        const x = A(0)!;
        let axes: number[] = [];
        if (node.attrs.has('axes')) axes = node.attrs.get('axes')!.ints;
        else if (node.inputs[1]) axes = cpu.asNumbers(await this.toCpu(A(1)!));
        const ndims = cpu.squeezeMeta(x.t.dims, axes);
        if (x.loc === 'gpu') this.protectedBuffers.add((x.t as any).buffer);
        this.store(cells, node.outputs[0], { loc: x.loc, t: { ...(x.t as any), dims: ndims } });
        return;
      }
      case 'Unsqueeze': {
        const x = A(0)!;
        let axes: number[] = [];
        if (node.attrs.has('axes')) axes = node.attrs.get('axes')!.ints;
        else if (node.inputs[1]) axes = cpu.asNumbers(await this.toCpu(A(1)!));
        const ndims = cpu.unsqueezeMeta(x.t.dims, axes);
        if (x.loc === 'gpu') this.protectedBuffers.add((x.t as any).buffer);
        this.store(cells, node.outputs[0], { loc: x.loc, t: { ...(x.t as any), dims: ndims } });
        return;
      }
      case 'Identity': {
        this.store(cells, node.outputs[0], A(0)!);
        return;
      }
      case 'ConstantOfShape': {
        const shape = cpu.asNumbers(await this.toCpu(A(0)!));
        const valAttr = node.attrs.get('value');
        let v = 0;
        if (valAttr?.tensor && valAttr.tensor.dataType === DT_FLOAT) v = new Float32Array(rawDataToTyped('f32', valAttr.tensor.rawData).buffer)[0];
        const total = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(total).fill(v);
        this.store(cells, node.outputs[0], { loc: 'cpu', t: { dims: shape.map(Number), type: 'f32', data } });
        return;
      }
      case 'Range': {
        const s = await this.toCpu(A(0)!), l = await this.toCpu(A(1)!), d = await this.toCpu(A(2)!);
        const type = s.type === 'f32' ? 'f32' : 'i64';
        this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.range(s.data[0], l.data[0], d.data[0], type) });
        return;
      }
      case 'Cast': {
        const to = node.attrs.get('to')!.i;
        const type: DType = to === DT_FLOAT ? 'f32' : to === DT_INT64 ? 'i64' : to === DT_INT32 ? 'i32' : 'bool';
        const x = await this.toCpu(A(0)!);
        this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.castTensor(x, type) });
        return;
      }
      case 'CumSum': {
        const x = await this.toCpu(A(0)!);
        const axis = Number((await this.toCpu(A(1)!)).data[0]);
        const exclusive = node.attrs.get('exclusive')?.i === 1;
        const reverse = node.attrs.get('reverse')?.i === 1;
        this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.cumSum(x, axis, exclusive, reverse) });
        return;
      }
      case 'ReduceSum': {
        const x = await this.toCpu(A(0)!);
        let axes: number[] | null = null;
        if (node.inputs[1]) axes = cpu.asNumbers(await this.toCpu(A(1)!));
        const keepdims = (node.attrs.get('keepdims')?.i ?? 1) === 1;
        this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.reduceSum(x, axes, keepdims) });
        return;
      }
      case 'ReduceMax': {
        const x = await this.toCpu(A(0)!);
        this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.reduceMaxAll(x) });
        return;
      }
    }

    const gpu = this.wantGpu(node, cells, this.estOutElems(node, cells));

    switch (op) {
      case 'Add': case 'Sub': case 'Mul': case 'Div': {
        const a = A(0)!, b = A(1)!;
        const opName = op.toLowerCase() as 'add';
        if (gpu) {
          const dims = broadcastShapes(a.t.dims, b.t.dims);
          const ga = this.toGpu(a), gb = this.toGpu(b);
          const go = this.allocGpu('f32', dims);
          const d4 = pad4(dims);
          const sa = broadcastStrides(pad4(a.t.dims), d4);
          const sb = broadcastStrides(pad4(b.t.dims), d4);
          const meta = new Uint32Array(64);
          meta[0] = numel(dims); meta[1] = 4;
          d4.forEach((d, j) => { meta[8 + j] = d; });
          sa.forEach((s, j) => { meta[16 + j] = s; });
          sb.forEach((s, j) => { meta[24 + j] = s; });
          const p = this.pipeline(`bin_${opName}`, binaryShader(opName), 3);
          const bg = this.bind(p, [ga.buffer, gb.buffer, go.buffer], meta);
          this.dispatch1d(p, bg, numel(dims));
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          const ta = await this.toCpu(a), tb = await this.toCpu(b);
          this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.binaryOp(opName, ta, tb) });
        }
        return;
      }

      case 'Exp': case 'Tanh': case 'Sigmoid': case 'Relu': case 'LeakyRelu': case 'Neg': case 'Ceil': {
        const x = A(0)!;
        const kindMap: Record<string, string> = {
          Exp: 'exp', Tanh: 'tanh', Sigmoid: 'sigmoid', Relu: 'relu',
          LeakyRelu: 'leakyrelu', Neg: 'neg', Ceil: 'ceil',
        };
        const kind = kindMap[op];
        const alpha = op === 'LeakyRelu' ? (node.attrs.get('alpha')?.f ?? 0.01) : 0;
        if (gpu) {
          const gx = this.toGpu(x);
          const go = this.allocGpu('f32', gx.dims);
          const meta = new Uint32Array(64);
          meta[0] = numel(gx.dims);
          meta[1] = { exp: 0, tanh: 1, sigmoid: 2, relu: 3, leakyrelu: 4, neg: 5, ceil: 6 }[kind]!;
          meta[2] = packF32(alpha);
          const p = this.pipeline(`unary`, unaryShader(), 2);
          const bg = this.bind(p, [gx.buffer, go.buffer], meta);
          this.dispatch1d(p, bg, numel(go.dims));
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          const t = await this.toCpu(x);
          this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.unaryOp(kind, t, alpha) });
        }
        return;
      }

      case 'Pow': {
        this.store(cells, node.outputs[0], {
          loc: 'cpu',
          t: cpu.powOp(await this.toCpu(A(0)!), await this.toCpu(A(1)!)),
        });
        return;
      }
      case 'Clip': {
        const x = await this.toCpu(A(0)!);
        const mn = node.inputs[1] ? Number((await this.toCpu(A(1)!)).data[0]) : undefined;
        const mx = node.inputs[2] ? Number((await this.toCpu(A(2)!)).data[0]) : undefined;
        this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.clipOp(x, mn, mx) });
        return;
      }

      case 'MatMul': {
        const a = A(0)!, b = A(1)!;
        if (gpu) {
          const ga = this.toGpu(a), gb = this.toGpu(b);
          const M = a.t.dims[a.t.dims.length - 2];
          const K = a.t.dims[a.t.dims.length - 1];
          const N = b.t.dims[b.t.dims.length - 1];
          const batchA = a.t.dims.slice(0, -2);
          const batchB = b.t.dims.slice(0, -2);
          const batch = broadcastShapes(batchA.length ? batchA : [1], batchB.length ? batchB : [1]);
          const b0 = batch.length > 1 ? batch[batch.length - 2] : 1;
          const b1 = batch[batch.length - 1];
          const dims = [...batch, M, N];
          const go = this.allocGpu('f32', dims);
          const sA = contiguousStrides(batchA.length ? batchA : [1]);
          const sB = contiguousStrides(batchB.length ? batchB : [1]);
          const bsA = batchB.length || batchA.length ? broadcastStrides(batchA.length ? batchA : [1], batch) : [0, 0];
          const bsB = batchA.length || batchB.length ? broadcastStrides(batchB.length ? batchB : [1], batch) : [0, 0];
          const meta = new Uint32Array(64);
          meta[0] = M; meta[1] = N; meta[2] = K; meta[3] = b0 * b1;
          meta[4] = batch.length > 1 ? batch[0] : 1; meta[5] = b1;
          meta[6] = batch.length > 1 ? batch[0] : 1; meta[7] = b1;
          meta[8] = batch.length > 1 ? (bsA[0] * M * K) : 0; meta[9] = bsA[batch.length - 1] * M * K;
          meta[10] = batch.length > 1 ? (bsB[0] * K * N) : 0; meta[11] = bsB[batch.length - 1] * K * N;
          const p = this.pipeline(`matmul`, matmulShader(), 3);
          const bg = this.bind(p, [ga.buffer, gb.buffer, go.buffer], meta);
          const enc = this.enc ?? (this.enc = this.device.createCommandEncoder());
          const pass = enc.beginComputePass();
          pass.setPipeline(p); pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16), Math.min(65535, b0 * b1)); pass.end();
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          const ta = await this.toCpu(a), tb = await this.toCpu(b);
          this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.matmul(ta, tb) });
        }
        return;
      }

      case 'Conv': case 'ConvTranspose': {
        const x = A(0)!, w = A(1)!;
        const biasCell = node.inputs.length > 2 && node.inputs[2] ? A(2) : null;
        const pads = node.attrs.get('pads')?.ints ?? [0, 0];
        const dil = node.attrs.get('dilations')?.ints ?? [1];
        const strides = node.attrs.get('strides')?.ints ?? [1];
        if (op === 'Conv' && dil[0] !== 1) {
          // dilation path: CPU reference (rare, correctness first)
          const tx = await this.toCpu(x), tw = await this.toCpu(w);
          const tb = biasCell ? await this.toCpu(biasCell) : null;
          this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.conv1d(tx, tw, tb, pads, dil) });
          return;
        }
        if (gpu) {
          const gx = this.toGpu(x), gw = this.toGpu(w);
          const gb = biasCell ? this.toGpu(biasCell) : this.toGpu(this.zeroScalar());
          const [, C, Win] = x.t.dims;
          const Wdims = w.t.dims;
          let M: number, Wout: number;
          if (op === 'Conv') {
            M = Wdims[0];
            Wout = Win + pads[0] + (pads[1] ?? pads[0]) - dil[0] * (Wdims[2] - 1);
          } else {
            M = Wdims[1];
            Wout = (Win - 1) * strides[0] - pads[0] - (pads[1] ?? pads[0]) + Wdims[2];
          }
          const dims = [x.t.dims[0], M, Wout];
          const go = this.allocGpu('f32', dims);
          const meta = new Uint32Array(64);
          meta[0] = Wout; meta[1] = C; meta[2] = M; meta[3] = Win;
          meta[4] = Wdims[2];
          meta[5] = op === 'Conv' ? dil[0] : strides[0];
          meta[6] = pads[0];
          meta[7] = biasCell ? 1 : 0;
          const key = op === 'Conv' ? 'conv1d' : 'convT1d';
          const p = this.pipeline(key, op === 'Conv' ? conv1dShader() : convTranspose1dShader(), 4);
          const bg = this.bind(p, [gx.buffer, gw.buffer, go.buffer, gb.buffer], meta);
          const enc = this.enc ?? (this.enc = this.device.createCommandEncoder());
          const pass = enc.beginComputePass();
          pass.setPipeline(p); pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(Wout / 8), Math.ceil(M / 8), Math.min(65535, x.t.dims[0])); pass.end();
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          const tx = await this.toCpu(x), tw = await this.toCpu(w);
          const tb = biasCell ? await this.toCpu(biasCell) : null;
          const t = op === 'Conv'
            ? cpu.conv1d(tx, tw, tb, pads, dil)
            : cpu.convTranspose1d(tx, tw, tb, pads, strides);
          this.store(cells, node.outputs[0], { loc: 'cpu', t });
        }
        return;
      }

      case 'LayerNormalization': {
        const x = A(0)!;
        const gamma = A(node.inputs.length > 1 && node.inputs[1] ? 1 : 0);
        const beta = A(2);
        const parsedEps = node.attrs.get('epsilon')?.f;
        const eps = parsedEps && parsedEps > 0 ? parsedEps : 1e-5;
        const g = gamma ?? this.oneScalar(), b = beta ?? this.zeroScalar();
        if (gpu) {
          const gx = this.toGpu(x), gg = this.toGpu(g), gb = this.toGpu(b);
          const inner = x.t.dims[x.t.dims.length - 1];
          const go = this.allocGpu('f32', x.t.dims);
          const meta = new Uint32Array(64);
          meta[0] = inner; meta[1] = packF32(eps);
          const p = this.pipeline(`ln`, layerNormShader(), 4);
          const bg = this.bind(p, [gx.buffer, gg.buffer, gb.buffer, go.buffer], meta);
          const enc = this.enc ?? (this.enc = this.device.createCommandEncoder());
          const pass = enc.beginComputePass();
          pass.setPipeline(p); pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.min(65535, numel(x.t.dims) / inner)); pass.end();
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          const tx = await this.toCpu(x);
          const tg = await this.toCpu(g), tb = await this.toCpu(b);
          const out = cpu.layerNorm(tx, tg, tb, eps);
          this.store(cells, node.outputs[0], { loc: 'cpu', t: out });
        }
        return;
      }

      case 'Softmax': {
        const x = A(0)!;
        if (gpu) {
          const gx = this.toGpu(x);
          const inner = x.t.dims[x.t.dims.length - 1];
          const go = this.allocGpu('f32', x.t.dims);
          const meta = new Uint32Array(64);
          meta[0] = inner;
          const p = this.pipeline(`sm`, softmaxShader(), 2);
          const bg = this.bind(p, [gx.buffer, go.buffer], meta);
          const enc = this.enc ?? (this.enc = this.device.createCommandEncoder());
          const pass = enc.beginComputePass();
          pass.setPipeline(p); pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.min(65535, numel(x.t.dims) / inner)); pass.end();
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.softmax(await this.toCpu(x)) });
        }
        return;
      }

      case 'Transpose': {
        const x = A(0)!;
        const perm = node.attrs.has('perm')
          ? node.attrs.get('perm')!.ints
          : [...Array(x.t.dims.length).keys()].reverse();
        if (gpu && x.t.dims.length <= 4) {
          const gx = this.toGpu(x);
          const r = x.t.dims.length;
          const pad = 4 - r;
          const permAdj = [...Array(pad).keys()].concat(perm.map((p) => p + pad));
          const dimsPadded = pad4(x.t.dims);
          const outDims = permAdj.map((p) => dimsPadded[p]);
          const inStrides = contiguousStrides(dimsPadded);
          const go = this.allocGpu('f32', outDims);
          const meta = new Uint32Array(64);
          meta[0] = numel(outDims); meta[1] = 4;
          permAdj.forEach((p, j) => { meta[8 + j] = p; });
          outDims.forEach((d, j) => { meta[12 + j] = d; });
          inStrides.forEach((s, j) => { meta[16 + j] = s; });
          const p2 = this.pipeline(`tr`, transposeShader(), 2);
          const bg = this.bind(p2, [gx.buffer, go.buffer], meta);
          this.dispatch1d(p2, bg, numel(outDims));
          this.store(cells, node.outputs[0], { loc: 'gpu', t: { dims: outDims.slice(pad), type: 'f32', buffer: go.buffer } });
        } else {
          this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.transpose(await this.toCpu(x), perm) });
        }
        return;
      }

      case 'Slice': {
        const x = A(0)!;
        const starts = cpu.asNumbers(await this.toCpu(A(1)!));
        const ends = cpu.asNumbers(await this.toCpu(A(2)!));
        const axes = node.inputs[3] ? cpu.asNumbers(await this.toCpu(A(3)!)) : [];
        const steps = node.inputs[4] ? cpu.asNumbers(await this.toCpu(A(4)!)) : [];
        const resolved = this.resolveSlice(x.t.dims, starts, ends, axes, steps);
        if (gpu && x.t.dims.length <= 4 && numel(resolved.outDims) > 0) {
          const gx = this.toGpu(x);
          const r = x.t.dims.length;
          const pad = 4 - r;
          const dimsP = pad4(x.t.dims);
          const outDimsP = pad4(resolved.outDims);
          const inStrides = contiguousStrides(dimsP);
          const go = this.allocGpu('f32', resolved.outDims);
          const meta = new Uint32Array(64);
          meta[0] = numel(resolved.outDims); meta[1] = 4;
          padLeading(resolved.startsAbs, 0).forEach((s, j) => { meta[8 + j] = s >>> 0; });
          padLeading(resolved.stepsArr, 1).forEach((s, j) => { meta[12 + j] = s >>> 0; });
          inStrides.forEach((s, j) => { meta[16 + j] = s; });
          outDimsP.forEach((d, j) => { meta[20 + j] = d; });
          const p = this.pipeline(`sl`, sliceShader(), 2);
          const bg = this.bind(p, [gx.buffer, go.buffer], meta);
          this.dispatch1d(p, bg, numel(resolved.outDims));
          this.store(cells, node.outputs[0], { loc: 'gpu', t: { dims: resolved.outDims, type: 'f32', buffer: go.buffer } });
        } else {
          this.store(cells, node.outputs[0], {
            loc: 'cpu',
            t: cpu.slice(await this.toCpu(x), starts, ends, axes, steps),
          });
        }
        return;
      }

      case 'Concat': {
        const axis = node.attrs.get('axis')?.i ?? 1;
        const parts: Cell[] = [];
        for (const nm of node.inputs) if (nm) parts.push(A(parts.length)!);
        // parts indexes shift because A() uses positional lookup; rebuild explicitly
        parts.length = 0;
        let pi = 0;
        for (const nm of node.inputs) { if (nm) parts.push(this.getCell(cells, nm)!); else void pi++; }
        if (parts.length === 1) { this.store(cells, node.outputs[0], parts[0]); return; }
        let acc = parts[0];
        let accIsTemp = false;
        for (let j = 1; j < parts.length; j++) {
          const merged = await this.concatTwo(acc, acc.t.dims, parts[j], axis);
          if (accIsTemp && acc.loc === 'gpu') this.scheduleDestroy(acc.t.buffer);
          acc = merged;
          accIsTemp = true;
        }
        this.store(cells, node.outputs[0], acc);
        return;
      }

      case 'Split': {
        const x = A(0)!;
        let sizes: number[];
        if (node.inputs[1]) sizes = cpu.asNumbers(await this.toCpu(A(1)!));
        else if (node.attrs.has('split')) sizes = node.attrs.get('split')!.ints;
        else {
          const axisSize = x.t.dims[node.attrs.get('axis')?.i ?? 0];
          const nOut = node.outputs.length;
          const part = axisSize / nOut;
          sizes = Array(nOut).fill(part);
        }
        const axis = node.attrs.get('axis')?.i ?? 0;
        let offset = 0;
        for (let j = 0; j < node.outputs.length; j++) {
          const rank = x.t.dims.length;
          const starts = Array(rank).fill(0);
          const ends = x.t.dims.slice();
          starts[axis] = offset;
          ends[axis] = offset + sizes[j];
          const piece = cpu.slice(await this.toCpu(x), starts, ends, [], []);
          this.store(cells, node.outputs[j], { loc: 'cpu', t: piece });
          offset += sizes[j];
        }
        return;
      }

      case 'Pad': {
        const x = A(0)!;
        const pads = cpu.asNumbers(await this.toCpu(A(1)!)).map(Number);
        let value = 0;
        if (node.inputs[2]) value = Number((await this.toCpu(A(2)!)).data[0]);
        if (gpu && x.t.dims.length <= 4 && value === 0) {
          const gx = this.toGpu(x);
          const r = x.t.dims.length;
          const dimsP = pad4(x.t.dims);
          const outDims = x.t.dims.map((d, j) => d + pads[j] + pads[j + r]);
          const outDimsP = pad4(outDims);
          const padsPre = padLeading(pads.slice(0, r), 0);
          const inStrides = contiguousStrides(dimsP);
          const go = this.allocGpu('f32', outDims);
          const meta = new Uint32Array(64);
          meta[0] = numel(outDims); meta[1] = 4; meta[2] = packF32(value);
          dimsP.forEach((d, j) => { meta[8 + j] = d; });
          padsPre.forEach((p, j) => { meta[12 + j] = p >>> 0; });
          outDimsP.forEach((d, j) => { meta[16 + j] = d; });
          inStrides.forEach((s, j) => { meta[20 + j] = s; });
          const p = this.pipeline(`pad`, padShader(), 2);
          const bg = this.bind(p, [gx.buffer, go.buffer], meta);
          this.dispatch1d(p, bg, numel(outDims));
          this.store(cells, node.outputs[0], { loc: 'gpu', t: { dims: outDims, type: 'f32', buffer: go.buffer } });
        } else {
          this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.padConstant(await this.toCpu(x), pads, value) });
        }
        return;
      }

      case 'Gather': {
        const x = A(0)!, idx = A(1)!;
        const axis = node.attrs.get('axis')?.i ?? 0;
        if (gpu && x.t.type === 'f32' && x.t.dims.length <= 4) {
          const gx = this.toGpu(x);
          const gi = this.toGpu(idx);
          const idxN = numel(idx.t.dims);
          const outer = x.t.dims.slice(0, axis).reduce((a, b) => a * b, 1);
          const inner = x.t.dims.slice(axis + 1).reduce((a, b) => a * b, 1);
          const outDims = [...x.t.dims.slice(0, axis), ...idx.t.dims, ...x.t.dims.slice(axis + 1)];
          const go = this.allocGpu('f32', outDims);
          const meta = new Uint32Array(64);
          meta[0] = numel(outDims); meta[1] = x.t.dims[axis];
          meta[2] = inner; meta[3] = idxN;
          const p = this.pipeline(`ga`, gatherShader(), 3);
          const bg = this.bind(p, [gx.buffer, gi.buffer, go.buffer], meta);
          this.dispatch1d(p, bg, numel(outDims));
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          this.store(cells, node.outputs[0], {
            loc: 'cpu',
            t: cpu.gather(await this.toCpu(x), await this.toCpu(idx), axis),
          });
        }
        return;
      }

      case 'Less': case 'Equal': {
        const a = A(0)!, b = A(1)!;
        const opCode = op === 'Less' ? 0 : 1;
        if (gpu) {
          const dims = broadcastShapes(a.t.dims, b.t.dims);
          const ga = this.toGpu(a), gb = this.toGpu(b);
          const go = this.allocGpu('bool', dims);
          const d4 = pad4(dims);
          const sa = broadcastStrides(pad4(a.t.dims), d4);
          const sb = broadcastStrides(pad4(b.t.dims), d4);
          const meta = new Uint32Array(64);
          meta[0] = numel(dims); meta[1] = 4;
          d4.forEach((d, j) => { meta[8 + j] = d; });
          sa.forEach((s, j) => { meta[16 + j] = s; });
          sb.forEach((s, j) => { meta[24 + j] = s; });
          const p = this.pipeline(`cmp_${opCode}`, compareShader(opCode), 3);
          const bg = this.bind(p, [ga.buffer, gb.buffer, go.buffer], meta);
          this.dispatch1d(p, bg, numel(dims));
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          this.store(cells, node.outputs[0], { loc: 'cpu', t: cpu.compare(opCode === 0 ? 'less' : 'equal', await this.toCpu(a), await this.toCpu(b)) });
        }
        return;
      }

      case 'Where': {
        const c = A(0)!, x = A(1)!, y = A(2)!;
        if (gpu) {
          const dims = broadcastShapes(broadcastShapes(c.t.dims, x.t.dims), y.t.dims);
          const gc = this.toGpu(c), gx = this.toGpu(x), gy = this.toGpu(y);
          const go = this.allocGpu('f32', dims);
          const d4 = pad4(dims);
          const sc = broadcastStrides(pad4(c.t.dims), d4);
          const sx = broadcastStrides(pad4(x.t.dims), d4);
          const sy = broadcastStrides(pad4(y.t.dims), d4);
          const meta = new Uint32Array(64);
          meta[0] = numel(dims); meta[1] = 4;
          d4.forEach((d, j) => { meta[8 + j] = d; });
          sc.forEach((s, j) => { meta[16 + j] = s; });
          sx.forEach((s, j) => { meta[24 + j] = s; });
          sy.forEach((s, j) => { meta[32 + j] = s; });
          const p = this.pipeline(`wh`, whereShader(), 4);
          const bg = this.bind(p, [gc.buffer, gx.buffer, gy.buffer, go.buffer], meta);
          this.dispatch1d(p, bg, numel(dims));
          this.store(cells, node.outputs[0], { loc: 'gpu', t: go });
        } else {
          this.store(cells, node.outputs[0], {
            loc: 'cpu',
            t: cpu.where(await this.toCpu(c), await this.toCpu(x), await this.toCpu(y)),
          });
        }
        return;
      }

      default:
        throw new Error(`Unsupported op ${op}`);
    }
  }

  // ── helpers ───────────────────────────────────────────────────────────────

  private zeroScalar(): Cell {
    return { loc: 'cpu', t: { dims: [1], type: 'f32', data: new Float32Array(1) } };
  }
  private oneScalar(): Cell {
    return { loc: 'cpu', t: { dims: [1], type: 'f32', data: new Float32Array([1]) } };
  }

  private estOutElems(node: ParsedNode, cells: Map<string, Cell>): number {
    // cheap estimate: max input size (output usually >= largest relevant input)
    let mx = 0;
    for (const inp of node.inputs) {
      if (!inp) continue;
      const c = this.getCell(cells, inp);
      if (!c) continue;
      const n = cellNumel(c);
      if (n > 1e7) return n; // weights are huge; skip precise est
      mx = Math.max(mx, n);
    }
    return mx;
  }

  private resolveSlice(
    dims: number[], starts: number[], ends: number[], axes: number[], steps: number[],
  ): { outDims: number[]; startsAbs: number[]; stepsArr: number[] } {
    const rank = dims.length;
    const ax = axes.length ? axes : [...Array(rank).keys()];
    const st = Array(rank).fill(0);
    const en = Array(rank).fill(0);
    const sp = Array(rank).fill(1);
    for (let a = 0; a < rank; a++) { st[a] = 0; en[a] = dims[a]; }
    for (let j = 0; j < ax.length; j++) {
      const a = ((ax[j] % rank) + rank) % rank;
      const d = dims[a];
      const s0 = starts[j], e = ends[j];
      sp[a] = steps.length ? steps[j] : 1;
      if (sp[a] > 0) {
        st[a] = Math.max(0, Math.min(s0 < 0 ? s0 + d : s0, d));
        en[a] = Math.max(st[a], Math.min(e < 0 ? e + d : e, d));
      } else {
        st[a] = s0 <= -d ? -1 : Math.max(-1, Math.min(s0 < 0 ? s0 + d : s0, d - 1));
        en[a] = e <= -d ? -1 : Math.max(-1, Math.min(e < 0 ? e + d : e, d - 1));
      }
    }
    const outDims = dims.map((d, a) => Math.max(0, Math.ceil((en[a] - st[a]) / sp[a])));
    return { outDims, startsAbs: st, stepsArr: sp };
  }

  private async concatTwo(a: Cell, adims: number[], b: Cell, axis: number): Promise<Cell> {
    const rank = adims.length;
    const ax = ((axis % rank) + rank) % rank;
    const outer = adims.slice(0, ax).reduce((x, y) => x * y, 1);
    const inner = adims.slice(ax + 1).reduce((x, y) => x * y, 1);
    const dimA = adims[ax];
    const dimB = b.t.dims[ax];
    const outDims = adims.slice();
    outDims[ax] = dimA + dimB;
    if (a.loc === 'gpu' && b.loc === 'gpu' && this.device) {
      const go = this.allocGpu('f32', outDims);
      const meta = new Uint32Array(64);
      meta[0] = numel(outDims); meta[1] = outer;
      meta[2] = dimA; meta[3] = dimB; meta[4] = inner;
      const p = this.pipeline(`cat`, concat2Shader(), 3);
      const bg = this.bind(p, [(a.t as GpuTensor).buffer, (b.t as GpuTensor).buffer, go.buffer], meta);
      this.dispatch1d(p, bg, numel(outDims));
      return { loc: 'gpu', t: go };
    }
    const t = cpu.concat([await this.toCpu(a), await this.toCpu(b)], ax);
    return { loc: 'cpu', t };
  }
}
