import { readFileSync } from 'node:fs';
import { GraphExecutor } from '../executor.ts';

const buf = readFileSync(new URL('../testdata/duration.onnx', import.meta.url));
const ex = new GraphExecutor(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
await ex.init(null);

const orig = ex.execNode.bind(ex);
let started = false;
ex.execNode = async function (node: any, ni: number, cells: Map<any, any>, outs: string[]) {
  await orig(node, ni, cells, outs);
  if (!started && /attn_layers\.0\/Pad_4/.test(node.name)) started = true;
  if (started && !node.opType.startsWith('Constant')) {
    const c: any = cells.get(node.outputs[0]);
    if (!c) return;
    let preview = '';
    if (c.t.data.length <= 10) preview = ' vals=[' + Array.from({ length: c.t.data.length }, (_, j) => Number(c.t.data[j])) + ']';
    console.log('#' + ni, node.opType, node.name.replace('/enc_p/encoder/attn_layers.0/', ''), JSON.stringify(c.t.dims) + preview);
    if (/MatMul$/.test(node.name)) throw new Error('stop');
  }
};

// same tokens as reference
const refTokens = BigInt64Array.from(JSON.parse(readFileSync(new URL('./tokens.json', import.meta.url), 'utf8')), (v) => BigInt(v));
try {
  await ex.run(new Map([
    ['tokens', { dims: [1, refTokens.length], type: 'i64', data: refTokens }],
    ['lengths', { dims: [1], type: 'i64', data: new BigInt64Array([BigInt(refTokens.length)]) }],
    ['length_scale', { dims: [], type: 'f32', data: new Float32Array([0.8]) }],
  ]), ['m_p_exp']);
} catch (e: any) {
  console.log('ERR', e.message);
}
