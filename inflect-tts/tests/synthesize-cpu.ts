/** End-to-end CPU synthesis smoke test: text -> WAV. */
import { readFileSync, writeFileSync } from 'node:fs';
import { InflectTTSEngine } from '../engine.ts';
import { textToInputIds } from '../frontend.ts';
import { float32ToWav } from '../wav.ts';

const buf = readFileSync(new URL('../testdata/duration.onnx', import.meta.url));
const decBuf = readFileSync(new URL('../testdata/decode.onnx', import.meta.url));

const engine = new InflectTTSEngine();
await engine.init(); // no WebGPU in bun -> pure CPU
await engine.loadModelBuffers(
  buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  decBuf.buffer.slice(decBuf.byteOffset, decBuf.byteOffset + decBuf.byteLength),
);

const text = process.argv[2] ?? 'Hello, world.';
console.log('text:', text);
const t0 = performance.now();
const { ids } = await textToInputIds(text);
const { waveform } = await engine.generate(ids, { speed: 1.0 });
console.log(`generated ${waveform.length} samples (${(waveform.length / 24000).toFixed(2)}s) in ${(performance.now() - t0).toFixed(0)} ms`);

let peak = 0;
for (let j = 0; j < waveform.length; j++) peak = Math.max(peak, Math.abs(waveform[j]));
console.log('peak amplitude:', peak.toFixed(4));

const wav = float32ToWav(waveform, 24000);
writeFileSync(new URL('./out.wav', import.meta.url), Buffer.from(await wav.arrayBuffer()));
console.log('wrote out.wav');
