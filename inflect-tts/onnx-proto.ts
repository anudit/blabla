/**
 * Minimal ONNX protobuf parser for Inflect-Micro-v2.
 * Parses ModelProto -> GraphProto: nodes (op_type/inputs/outputs/attributes),
 * initializers and graph I/O. No external protobuf dependency.
 */

export interface ParsedTensor {
  name: string;
  dims: number[];
  dataType: number;
  rawData: Uint8Array;
}

export interface ParsedAttribute {
  name: string;
  type: number;
  i: number;
  f: number;
  s: Uint8Array | null;
  ints: number[];
  floats: number[];
  tensor: ParsedTensor | null;
}

export interface ParsedNode {
  opType: string;
  name: string;
  inputs: string[];
  outputs: string[];
  attrs: Map<string, ParsedAttribute>;
}

export interface ParsedGraph {
  nodes: ParsedNode[];
  initializers: Map<string, ParsedTensor>;
  inputs: string[];
  outputs: string[];
}

// TensorProto data types
export const DT_FLOAT = 1;
export const DT_INT32 = 6;
export const DT_INT64 = 7;
export const DT_BOOL = 9;

class Reader {
  buf: Uint8Array;
  view: DataView;
  base: number; // absolute byteOffset of buf within view's buffer
  pos = 0;
  end: number;

  constructor(buf: Uint8Array, view: DataView, start: number, end: number) {
    this.buf = buf; this.view = view; this.base = buf.byteOffset; this.pos = start; this.end = end;
  }

  f32(pos: number): number {
    return this.view.getFloat32(this.base + pos, true);
  }

  get atEnd(): boolean { return this.pos >= this.end; }

  readTag(): { field: number; wire: number } {
    const tag = this.varint();
    return { field: tag >>> 3, wire: tag & 7 };
  }

  varint(): number {
    let value = 0, shift = 0;
    while (this.pos < this.end) {
      const b = this.buf[this.pos++];
      if (shift < 28) value |= (b & 0x7f) << shift;
      else value += (b & 0x7f) * Math.pow(2, shift);
      if ((b & 0x80) === 0) break;
      shift += 7;
    }
    return value;
  }

  skip(wire: number): void {
    switch (wire) {
      case 0: this.varint(); break;
      case 1: this.pos += 8; break;
      case 2: {
        const len = this.varint();
        this.pos += len;
        break;
      }
      case 5: this.pos += 4; break;
      default: throw new Error(`Cannot skip wire type ${wire}`);
    }
  }

  bytes(): Uint8Array {
    const len = this.varint();
    const out = this.buf.subarray(this.pos, this.pos + len);
    this.pos += len;
    return out;
  }
}

function parseTensor(r: Reader): ParsedTensor {
  let name = '';
  const dims: number[] = [];
  let dataType = 0;
  let rawData: Uint8Array | null = null;
  let floatData: Float32Array | null = null;
  let int64Data: BigInt64Array | null = null;
  let int32Data: Int32Array | null = null;

  while (!r.atEnd) {
    const { field, wire } = r.readTag();
    switch (field) {
      case 1: {
        if (wire === 0) dims.push(r.varint());
        else if (wire === 2) {
          const b = r.bytes();
          const sub = new Reader(b, r.view, 0, b.length);
          while (!sub.atEnd) dims.push(sub.varint());
        } else r.skip(wire);
        break;
      }
      case 2: dataType = r.varint(); break;
      case 4: { // float_data (fixed32)
        if (wire === 5) {
          floatData = Float32Array.from([r.f32(r.pos)]);
          r.pos += 4;
        } else {
          const b = r.bytes();
          const vals: number[] = [];
          for (let i = 0; i + 4 <= b.length; i += 4) vals.push(new DataView(b.buffer).getFloat32(b.byteOffset + i, true));
          floatData = Float32Array.from(vals);
        }
        break;
      }
      case 5: { // int32_data (varint, packed or single)
        const vals: number[] = [];
        if (wire === 0) vals.push(r.varint() | 0);
        else {
          const b = r.bytes();
          const sub = new Reader(b, r.view, 0, b.length);
          while (!sub.atEnd) vals.push(sub.varint() | 0);
        }
        int32Data = Int32Array.from(vals);
        break;
      }
      case 7: { // int64_data (varint, packed or single)
        const decodeI64 = (sub: Reader): bigint => {
          let v = 0n;
          let shift = 0n;
          while (!sub.atEnd) {
            const byte = sub.buf[sub.pos++];
            v += BigInt(byte & 0x7f) << shift;
            shift += 7n;
            if ((byte & 0x80) === 0) break;
          }
          if (v >= 1n << 63n) v -= 1n << 64n;
          return v;
        };
        const vals: bigint[] = [];
        if (wire === 0) vals.push(decodeI64(r));
        else {
          const b = r.bytes();
          const sub = new Reader(b, r.view, 0, b.length);
          while (!sub.atEnd) vals.push(decodeI64(sub));
        }
        int64Data = BigInt64Array.from(vals);
        break;
      }
      case 8: name = new TextDecoder().decode(r.bytes()); break;
      case 9: rawData = r.bytes(); break;
      default: r.skip(wire);
    }
  }

  let data: Uint8Array;
  if (rawData) data = rawData;
  else if (floatData) data = new Uint8Array(floatData.buffer, floatData.byteOffset, floatData.byteLength);
  else if (int64Data) data = new Uint8Array(int64Data.buffer, int64Data.byteOffset, int64Data.byteLength);
  else if (int32Data) data = new Uint8Array(int32Data.buffer, int32Data.byteOffset, int32Data.byteLength);
  else data = new Uint8Array(0);

  return { name, dims, dataType, rawData: data };
}

function parseAttribute(r: Reader): ParsedAttribute {
  const attr: ParsedAttribute = { name: '', type: 0, i: 0, f: 0, s: null, ints: [], floats: [], tensor: null };
  while (!r.atEnd) {
    const { field, wire } = r.readTag();
        switch (field) {
      case 1: attr.name = new TextDecoder().decode(r.bytes()); break;
      case 2:
        if (wire === 5) { attr.f = r.f32(r.pos); r.pos += 4; }
        else { const b = r.bytes(); for (let i = 0; i + 4 <= b.length; i += 4) attr.floats.push(new DataView(b.buffer).getFloat32(b.byteOffset + i, true)); }
        break;
      case 3: attr.i = r.varint(); break;
      case 4: attr.s = r.bytes(); break;
      case 5: {
        const b = r.bytes();
        attr.tensor = parseTensor(new Reader(b, r.view, 0, b.length));
        break;
      }
      case 7: {
        if (wire === 5) { attr.floats.push(r.f32(r.pos)); r.pos += 4; }
        else {
          const b = r.bytes();
          for (let i = 0; i + 4 <= b.length; i += 4) attr.floats.push(new DataView(b.buffer).getFloat32(b.byteOffset + i, true));
        }
        break;
      }
      case 8: {
        if (wire === 0) attr.ints.push(r.varint());
        else {
          const b = r.bytes();
          const sub = new Reader(b, r.view, 0, b.length);
          while (!sub.atEnd) attr.ints.push(sub.varint());
        }
        break;
      }
      case 20: attr.type = r.varint(); break;
      default: r.skip(wire);
    }
  }
  return attr;
}

function parseNode(r: Reader): ParsedNode {
  const node: ParsedNode = { opType: '', name: '', inputs: [], outputs: [], attrs: new Map() };
  while (!r.atEnd) {
    const { field, wire } = r.readTag();
    switch (field) {
      case 1: node.inputs.push(new TextDecoder().decode(r.bytes())); break;
      case 2: node.outputs.push(new TextDecoder().decode(r.bytes())); break;
      case 3: node.name = new TextDecoder().decode(r.bytes()); break;
      case 4: node.opType = new TextDecoder().decode(r.bytes()); break;
      case 5: {
        const b = r.bytes();
        const a = parseAttribute(new Reader(b, r.view, 0, b.length));
        node.attrs.set(a.name, a);
        break;
      }
      default: r.skip(wire);
    }
  }
  return node;
}

/** Parse an ONNX model buffer into its graph. */
export function parseOnnxModel(buffer: ArrayBuffer): ParsedGraph {
  const buf = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const root = new Reader(buf, view, 0, buf.length);

  let graphStart = -1, graphLen = -1;
  while (!root.atEnd) {
    const { field, wire } = root.readTag();
    if (field === 7 && wire === 2) {
      graphLen = root.varint();
      graphStart = root.pos;
      break;
    }
    root.skip(wire);
  }
  if (graphStart < 0) throw new Error('No graph in ONNX model');

  const g = new Reader(buf, view, graphStart, graphStart + graphLen);
  const graph: ParsedGraph = { nodes: [], initializers: new Map(), inputs: [], outputs: [] };

  while (!g.atEnd) {
    const { field, wire } = g.readTag();
    if (field === 1 && wire === 2) {
      const b = g.bytes();
      graph.nodes.push(parseNode(new Reader(b, view, 0, b.length)));
    } else if (field === 5 && wire === 2) {
      const b = g.bytes();
      const t = parseTensor(new Reader(b, view, 0, b.length));
      if (t.name) graph.initializers.set(t.name, t);
    } else if (field === 11 && wire === 2) {
      // ValueInfoProto input: field 1 = name
      const b = g.bytes();
      const vi = new Reader(b, view, 0, b.length);
      while (!vi.atEnd) {
        const t2 = vi.readTag();
        if (t2.field === 1 && t2.wire === 2) { graph.inputs.push(new TextDecoder().decode(vi.bytes())); break; }
        vi.skip(t2.wire);
      }
    } else if (field === 12 && wire === 2) {
      const b = g.bytes();
      const vi = new Reader(b, view, 0, b.length);
      while (!vi.atEnd) {
        const t2 = vi.readTag();
        if (t2.field === 1 && t2.wire === 2) { graph.outputs.push(new TextDecoder().decode(vi.bytes())); break; }
        vi.skip(t2.wire);
      }
    } else {
      g.skip(wire);
    }
  }

  return graph;
}
