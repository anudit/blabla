// scripts/build-pdfjs-lean.ts
//
// Builds trimmed pdf.js artifacts (main API + worker) from a local checkout of
// the pdf.js repo (default: sibling ../pdf.js). Keeps what blabla uses:
// getDocument / getTextContent / getViewport / page.render (canvas). Drops
// the text layer, XFA, and the annotation editor.
//
// Usage: bun scripts/build-pdfjs-lean.ts
// Output: vendor/pdf/pdf.mjs + vendor/pdf/pdf.worker.mjs

import { build } from "bun";
import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const PDFJS_ROOT = process.env.PDFJS_ROOT ?? path.resolve(import.meta.dir, "../../pdf.js");
const OUT_DIR = path.resolve(import.meta.dir, "../vendor/pdf");

const tmp = path.join(import.meta.dir, ".pdfjs-tmp");
await rm(tmp, { recursive: true, force: true });
await mkdir(path.join(tmp, "external"), { recursive: true });

console.log(`📋 Copying pdf.js source from ${PDFJS_ROOT}...`);
await cp(path.join(PDFJS_ROOT, "src"), path.join(tmp, "src"), { recursive: true });
for (const dir of ["brotli", "jbig2", "openjpeg", "qcms"]) {
  await cp(path.join(PDFJS_ROOT, "external", dir), path.join(tmp, "external", dir), { recursive: true });
}

// Build-time aliases that the real gulp build resolves: GENERIC browser builds
// map these to the real implementations (they are guarded by isNodeJS).
const apiPath = path.join(tmp, "src/display/api.js");
let api = await Bun.file(apiPath).text();
api = api
  .replaceAll('from "display-node_utils"', 'from "./node_utils.js"')
  .replaceAll('from "display-binary_data_factory"', 'from "./binary_data_factory.js"')
  .replaceAll('from "display-network_stream"', 'from "./network_stream.js"')
  // Dead branch at runtime (non-embedded builds) but bundlers still resolve it.
  .replaceAll('import("pdfjs/pdf.worker.js")', 'import("./__worker_stub__.js")');
await Bun.write(apiPath, api);
await Bun.write(path.join(tmp, "src/display/__worker_stub__.js"), "export const WorkerMessageHandler = {};\n");

const stubs: Record<string, string> = {
  // Main thread — api.js imports these statically but the app never uses them.
  // TextLayer.cleanup() is called unconditionally in api.js teardown, and
  // XfaText.textContent() only for XFA-form pages, hence the static shims.
  "/display/text_layer\\.js$": "export class TextLayer { static cleanup() {} }",
  "/display/xfa_text\\.js$": "export class XfaText { static textContent() { return []; } }",
  // Worker — XFA forms and the annotation editor are huge and unused.
  "/core/xfa/factory\\.js$": "export class XFAFactory {}\nexport class XFAPage {}",
  "/core/editor/pdf_editor\\.js$": "export class PDFEditor {}",
  "/core/editor/pdf_images\\.js$": "export async function createImage() { return null; }",
};

const stubPlugin = {
  name: "stub",
  setup(build: any) {
    const entries = Object.entries(stubs).map(([f, code]) => [new RegExp(f), code] as const);
    build.onLoad({ filter: /\.js$/ }, (args: { path: string }) => {
      for (const [re, code] of entries) if (re.test(args.path)) return { contents: code, loader: "js" };
      return undefined;
    });
  },
};

async function emit(name: string, entrySource: string) {
  const entry = path.join(tmp, `${name}-entry.mjs`);
  await Bun.write(entry, entrySource);
  const r = await build({
    entrypoints: [entry],
    target: "browser",
    minify: true,
    plugins: [stubPlugin],
    define: { "process.env.NODE_ENV": '"production"' },
  });
  if (!r.success) {
    for (const l of r.logs) console.error(String(l));
    throw new Error(`${name} bundle failed`);
  }
  const bytes = await r.outputs[0].arrayBuffer();
  const outPath = path.join(OUT_DIR, `${name}.mjs`);
  await Bun.write(outPath, bytes);
  console.log(`✅ ${path.basename(outPath)}: ${(bytes.byteLength / 1024).toFixed(0)} KB raw, ${(Bun.gzipSync(new Uint8Array(bytes)).length / 1024).toFixed(0)} KB gz`);
}

await rm(OUT_DIR, { recursive: true, force: true });
await mkdir(OUT_DIR, { recursive: true });

await emit(
  "pdf",
  `import { getDocument } from "${tmp}/src/display/api.js";\n` +
    `import { GlobalWorkerOptions } from "${tmp}/src/display/worker_options.js";\n` +
    `import { version } from "${tmp}/src/display/api.js";\n` +
    `export { getDocument, GlobalWorkerOptions, version };\n`,
);
await emit("pdf.worker", `import "${tmp}/src/pdf.worker.js";\n`);

await rm(tmp, { recursive: true, force: true });
console.log("🎉 Lean pdf.js artifacts written to vendor/pdf/");
