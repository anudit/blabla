import { readFileSync } from 'node:fs';
import { GraphExecutor } from '../executor.ts';

const buf = readFileSync(new URL('../testdata/duration.onnx', import.meta.url));
const ex = new GraphExecutor(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
await ex.init(null);

const orig = ex.execNode.bind(ex);
ex.execNode = async function (node: any, ni: number, cells: Map<any, any>, outs: string[]) {
  await orig(node, ni, cells, outs);
  const c: any = cells.get(node.outputs[0]);
  if (!c || c.t.type !== 'f32') return;
  const arr = c.t.data as Float32Array;
  let nan = 0;
  for (let j = 0; j < arr.length; j++) if (!Number.isFinite(arr[j])) nan++;
  if (nan > 0) {
    console.log('FIRST NONFINITE #' + ni, node.opType, node.name, 'dims', JSON.stringify(c.t.dims), 'bad', nan, '/', arr.length);
    for (const inp of node.inputs) {
      if (!inp) continue;
      const ic: any = ex.getCell(cells, inp);
      if (!ic || ic.t.type !== 'f32') { console.log('   in', inp, ic ? ic.t.type : 'missing'); continue; }
      const a = ic.t.data as Float32Array;
      let bad = 0;
      for (let j = 0; j < a.length; j++) if (!Number.isFinite(a[j])) bad++;
      let preview = '';
      if (a.length <= 8) preview = ' vals=[' + Array.from(a, (v) => v.toPrecision(4)) + ']';
      console.log('   in', inp, JSON.stringify(ic.t.dims), 'bad', bad, preview);
    }
    throw new Error('stop');
  }
};

const refTokens = BigInt64Array.from(JSON.parse(readFileSync(new URL('./tokens.json', import.meta.url), 'utf8')), (v) => BigInt(v));
const t = refTokens;
try {
  await ex.run(new Map([
    ['tokens', { dims: [1, t.length], type: 'i64', data: t }],
    ['lengths', { dims: [1], type: 'i64', data: new BigInt64Array([BigInt(t.length)]) }],
    ['length_scale', { dims: [], type: 'f32', data: new Float32Array([1]) }],
  ]), ['m_p_exp']);
} catch (e: any) {
  console.log('ERR', e.message);
}
