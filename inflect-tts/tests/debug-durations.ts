import { readFileSync } from 'node:fs';
import { GraphExecutor } from '../executor.ts';

const buf = readFileSync(new URL('../testdata/duration.onnx', import.meta.url));
const ex = new GraphExecutor(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
await ex.init(null);

const targets: Record<string, string> = {
  '/dp/proj/Conv_output_0': 'dbg__dp_proj_Conv_output_0.npy',
  '/dp/Mul_3_output_0': 'dbg__dp_Mul_3_output_0.npy',
  '/Ceil_output_0': 'dbg__Ceil_output_0.npy',
};

function loadNpy(path: string): Float32Array {
  const b = readFileSync(path);
  const bytes = new Uint8Array(b);
  const dv = new DataView(b.buffer, b.byteOffset);
  const headerLen = dv.getUint16(8, true);
  const off = 10 + headerLen;
  const n = (bytes.length - off) / 4;
  const tmp = new Uint8Array(n * 4);
  tmp.set(bytes.subarray(off));
  return new Float32Array(tmp.buffer);
}

const orig = ex.execNode.bind(ex);
ex.execNode = async function (node: any, ni: number, cells: Map<any, any>, outs: string[]) {
  await orig(node, ni, cells, outs);
  const outName = node.outputs[0];
  if (targets[outName]) {
    const c: any = cells.get(outName);
    const t = await ex.toCpu(c);
    const ref = loadNpy(new URL('../testdata/' + targets[outName], import.meta.url).pathname);
    let mx = 0;
    for (let j = 0; j < ref.length; j++) mx = Math.max(mx, Math.abs(Number(t.data[j]) - ref[j]));
    console.log(outName, 'dims', JSON.stringify(t.dims), 'maxDiff vs py', mx.toExponential(3));
    if (outName === '/Ceil_output_0') {
      console.log('ceil vals:', Array.from({ length: Math.min(20, (t.data as any).length) }, (_, j) => Number(t.data[j])));
    }
    if (outName === '/Ceil_output_0') throw new Error('stop');
  }
};

const refTokens = BigInt64Array.from(JSON.parse(readFileSync(new URL('./tokens.json', import.meta.url), 'utf8')), (v) => BigInt(v));
try {
  await ex.run(new Map([
    ['tokens', { dims: [1, refTokens.length], type: 'i64', data: refTokens }],
    ['lengths', { dims: [1], type: 'i64', data: new BigInt64Array([BigInt(refTokens.length)]) }],
    ['length_scale', { dims: [], type: 'f32', data: new Float32Array([0.8]) }],
  ]), ['m_p_exp']);
} catch (e: any) {
  if (e.message !== 'stop') console.log('ERR', e.message);
}
