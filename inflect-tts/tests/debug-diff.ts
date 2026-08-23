import { readFileSync } from 'node:fs';
import { GraphExecutor } from '../executor.ts';

const buf = readFileSync(new URL('../testdata/duration.onnx', import.meta.url));
const ex = new GraphExecutor(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
await ex.init(null);

const targets: Record<string, string> = {
  '/dp/norm_1/LayerNormalization_output_0': 'dbg3__dp_norm_1_LayerNormalization_output_0.npy',
  '/dp/Mul_1_output_0': 'dbg3__dp_Mul_1_output_0.npy',
  '/dp/conv_2/Conv_output_0': 'dbg3__dp_conv_2_Conv_output_0.npy',
  '/dp/norm_2/LayerNormalization_output_0': 'dbg3__dp_norm_2_LayerNormalization_output_0.npy',
  '/dp/Mul_2_output_0': 'dbg3__dp_Mul_2_output_0.npy',
  '/dp/norm_1/Transpose_output_0': 'dbg4__dp_transposed.npy',
  '/enc_p/encoder/Mul_2_output_0': 'dbg2__enc_p_Mul_2_output_0.npy',

  '/enc_p/emb/Gather_output_0': 'dbg2__enc_p_emb_Gather_output_0.npy',
  '/enc_p/encoder/attn_layers.0/Add_3_output_0': 'dbg2__enc_p_encoder_attn_layers.0_Add_3_output_0.npy',
  '/enc_p/encoder/norm_layers_1.0/LayerNormalization_output_0': 'dbg2__enc_p_encoder_norm_layers_1.0_LayerNormalization_output_0.npy',
  '/enc_p/encoder/ffn_layers.0/conv_1/Conv_output_0': 'dbg2__enc_p_encoder_ffn_layers.0_conv_1_Conv_output_0.npy',
  '/enc_p/encoder/ffn_layers.0/Relu_output_0': 'dbg2__enc_p_encoder_ffn_layers.0_Relu_output_0.npy',
  '/enc_p/encoder/ffn_layers.0/conv_2/Conv_output_0': 'dbg2__enc_p_encoder_ffn_layers.0_conv_2_Conv_output_0.npy',
  '/enc_p/encoder/norm_layers_2.0/LayerNormalization_output_0': 'dbg2__enc_p_encoder_norm_layers_2.0_LayerNormalization_output_0.npy',
  '/enc_p/encoder/attn_layers.1/Add_3_output_0': 'dbg2__enc_p_encoder_attn_layers.1_Add_3_output_0.npy',
  '/enc_p/encoder/norm_layers_2.1/LayerNormalization_output_0': 'dbg2__enc_p_encoder_norm_layers_2.1_LayerNormalization_output_0.npy',
  '/enc_p/Mul_2_output_0': 'dbg2__enc_p_Mul_2_output_0.npy',
  '/enc_p/Split_output_0': 'dbg2__enc_p_Split_output_0.npy',
  '/dp/conv_1/Conv_output_0': 'dbg2__dp_conv_1_Conv_output_0.npy',
  '/dp/proj/Conv_output_0': 'dbg2__dp_proj_Conv_output_0.npy',
};

function loadNpy(path: string): { dims: number[]; data: Float32Array } {
  const b = readFileSync(path);
  const bytes = new Uint8Array(b);
  const dv = new DataView(b.buffer, b.byteOffset);
  const headerLen = dv.getUint16(8, true);
  const off = 10 + headerLen;
  const n = (bytes.length - off) / 4;
  const tmp = new Uint8Array(n * 4);
  tmp.set(bytes.subarray(off));
  return { dims: [], data: new Float32Array(tmp.buffer) };
}

const orig = ex.execNode.bind(ex);
ex.execNode = async function (node: any, ni: number, cells: Map<any, any>, outs: string[]) {
  await orig(node, ni, cells, outs);
  const outName = node.outputs[0];
  if (targets[outName]) {
    const c: any = cells.get(outName);
    const t = await ex.toCpu(c);
    const ref = loadNpy(new URL('../testdata/' + targets[outName], import.meta.url).pathname);
    if (t.data.length !== ref.data.length) console.log('  LENGTH MISMATCH mine', t.data.length, 'ref', ref.data.length, 'dims', JSON.stringify(t.dims));
    let mx = 0, nanMine = 0, nanRef = 0;
    const n = Math.min(t.data.length, ref.data.length);
    for (let j = 0; j < n; j++) {
      const a = Number(t.data[j]), b = ref.data[j];
      if (!Number.isFinite(a)) nanMine++;
      if (!Number.isFinite(b)) nanRef++;
      if (Number.isFinite(a) && Number.isFinite(b)) mx = Math.max(mx, Math.abs(a - b));
    }
    if (outName === '/dp/norm_1/LayerNormalization_output_0') {
      console.log('  mine[256..259]=', Array.from({length:4},(_,j)=>Number(t.data[256+j]).toPrecision(6)), 'ref=', Array.from(ref.data.slice(256,260), v=>v.toPrecision(6)));
      console.log('  mine[row1][0..4]=', Array.from({length:4},(_,j)=>Number(t.data[256+j]).toPrecision(6)));
    }
    if (nanMine || nanRef) console.log('  nonfinite: mine', nanMine, 'ref', nanRef);
    const mine = Number(t.data[0]), theirs = ref.data[0];
    console.log(outName.replace('/enc_p/', ''), 'maxDiff', mx.toExponential(3), 'mine[0..3]=', Array.from({length:4},(_,j)=>Number(t.data[j])).map(v=>v.toPrecision(5)).join(','), 'ref=', Array.from(ref.data.slice(0,4), v=>v.toPrecision(5)).join(','), mx > 1e-3 ? '<-- DIVERGES' : '');
    delete targets[outName];
    if (!Object.keys(targets).length) throw new Error('stop');
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
