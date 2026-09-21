import { build } from 'bun';
import { mkdir, rm } from 'node:fs/promises';

await rm('./dist', { recursive: true, force: true });
await mkdir('./dist', { recursive: true });

for (const entry of ['./src/index.ts', './src/worker.ts']) {
  const result = await build({
    entrypoints: [entry],
    outdir: './dist',
    naming: { entry: entry.endsWith('worker.ts') ? 'pdf-lite.worker.js' : 'pdf-lite.js' },
    target: 'browser',
    minify: true,
    sourcemap: 'none',
  });
  if (!result.success) throw new Error(`Failed to build ${entry}`);
}
