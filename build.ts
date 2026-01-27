// build.ts
import { build } from "bun";
import { cp, rm } from "node:fs/promises";

console.log("🧹 Cleaning dist folder...");
await rm("./dist", { recursive: true, force: true });

console.log("📦 Building Frontend (bundle.js)...");
await build({
  entrypoints: ["./frontend.tsx"],
  outdir: "./dist",
  naming: "bundle.js", // Explicitly naming it bundle.js to match index.html
  target: "browser",
  minify: true,
});

console.log("⚙️ Building TTS Worker...");
await build({
  entrypoints: ["./tts.worker.ts"],
  outdir: "./dist",
  target: "browser",
  minify: true,
});

console.log("📂 Copying Static Assets...");
await cp("./index.html", "./dist/index.html");
await cp("./manifest.json", "./dist/manifest.json");
await cp("./logo.png", "./dist/logo.png");
await cp("./sw.js", "./dist/sw.js");

// Critical: Copy the PDF worker from node_modules to dist
// This ensures pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs' works
await cp(
  "./node_modules/pdfjs-dist/build/pdf.worker.mjs",
  "./dist/pdf.worker.mjs"
);

console.log("✅ Build Complete! Files are in /dist");
