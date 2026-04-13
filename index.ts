// index.ts
import { serve } from "bun";

async function buildAndGzip(entrypoint: string) {
  const build = await Bun.build({
    entrypoints: [entrypoint],
    target: "browser",
    minify: process.env.NODE_ENV === "production",
    define: {
      "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV || "development"),
    },
  });
  const bytes = new Uint8Array(await build.outputs[0].arrayBuffer());
  const compressed = Bun.gzipSync(bytes);
  return new Response(compressed, {
    headers: {
      'Content-Type': 'application/javascript',
      'Content-Encoding': 'gzip',
      'Vary': 'Accept-Encoding',
    },
  });
}

const server = serve({
  routes: {
    // 1. Build the Main App (Bundling React + dependencies)
    "/bundle.js": () => buildAndGzip("./frontend.tsx"),

    // 2. Build the TTS Worker
    "/tts.worker.js": () => buildAndGzip("./tts.worker.ts"),

    // 3. Serve Static Assets
    "/pdf.worker.min.mjs": Bun.file("node_modules/pdfjs-dist/build/pdf.worker.min.mjs"),
    "/manifest.json": Bun.file("./manifest.json"),
    "/sw.js": Bun.file("./sw.js"),
    // "/logo.png": Bun.file("./logo.png"),
    "/16.png": Bun.file("./16.png"),
    "/32.png": Bun.file("./32.png"),
    "/128.png": Bun.file("./128.png"),
    "/180.png": Bun.file("./180.png"),
    "/og.jpg": Bun.file("./og.jpg"),


    // 4. SEO — robots.txt & sitemap (must be explicit routes; /* would swallow them)
    "/robots.txt": Bun.file("./robots.txt"),
    "/sitemap.xml": Bun.file("./sitemap.xml"),

    // 5. Serve Index (Fallback for SPA)
    "/*": Bun.file("./index.html")
  },
  development: process.env.NODE_ENV !== "production" && {
    hmr: true,
  },
  port: 3031,
});

console.log("🚀 Server running at " + server.url);
