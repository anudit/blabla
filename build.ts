// build.ts
import { build } from "bun";
import { cp, rm } from "node:fs/promises";

console.log("🧹 Cleaning dist folder...");
await rm("./dist", { recursive: true, force: true });

console.log("📦 Building Frontend (bundle.js)...");
const preactAlias = {
  "react": "preact/compat",
  "react-dom": "preact/compat",
  "react/jsx-runtime": "preact/jsx-runtime",
  "react-dom/client": "preact/compat/client",
};

const result = await build({
  entrypoints: ["./frontend.tsx"],
  outdir: "./dist",
  naming: {
      entry: "bundle.js",
      chunk: "[name]-[hash].js"
  },
  splitting: true,
  target: "browser",
  minify: true,
  metafile: true,
  alias: preactAlias,
});
await Bun.write("./dist/meta.json", JSON.stringify(result.metafile));

console.log("⚙️ Building TTS Worker...");
await build({
  entrypoints: ["./tts.worker.ts"],
  outdir: "./dist",
  target: "browser",
  minify: true,
  alias: preactAlias,
});

console.log("📂 Copying Static Assets...");
await cp("./index.html", "./dist/index.html");
await cp("./manifest.json", "./dist/manifest.json");
// await cp("./logo.png", "./dist/logo.png");
await cp("./16.png", "./dist/16.png");
await cp("./32.png", "./dist/32.png");
await cp("./128.png", "./dist/128.png");
await cp("./180.png", "./dist/180.png");
await cp("./og.jpg", "./dist/og.jpg");
await cp("./sw.js", "./dist/sw.js");
await cp("./sitemap.xml", "./dist/sitemap.xml");
await cp("./robots.txt", "./dist/robots.txt");

// Critical: Copy the PDF worker from node_modules to dist
// This ensures pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs' works
await cp(
  "./node_modules/pdfjs-dist/build/pdf.worker.mjs",
  "./dist/pdf.worker.mjs"
);

console.log("✅ Build Complete! Files are in /dist");
