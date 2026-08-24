// index.ts
import { serve } from "bun";

async function buildAndGzip(entrypoint: string) {
  const build = await Bun.build({
    entrypoints: [entrypoint],
    target: "browser",
    minify: process.env.NODE_ENV === "production",
    define: {
      "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV || "development"),
      "process.env.TTS_DEBUG": JSON.stringify(process.env.TTS_DEBUG || ""),
      "process.env.TTS_DEBUG_NODES": JSON.stringify(process.env.TTS_DEBUG_NODES || ""),
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

    // 2b. Build the OCR Worker (detection + recognition off the main thread)
    "/ocr.worker.js": () => buildAndGzip("./ocr.worker.ts"),

    // 2c. TEMPORARY: GPU decode benchmark harness (dev only, not in build.ts).
    "/bench.worker.js": () => buildAndGzip("./inflect-tts/tests/bench.worker.ts"),
    "/bench": () => new Response(
      `<!doctype html><meta charset=utf8><title>decode bench</title>
       <body style="font:14px ui-monospace,monospace;padding:16px">
       <pre id=out>starting…</pre>
       <script type=module>
         const out = document.getElementById('out');
         const log = [];
         const w = new Worker('/bench.worker.js', { type: 'module' });
         w.onmessage = (e) => {
           log.push(JSON.stringify(e.data, null, 2));
           out.textContent = log.join(String.fromCharCode(10));
           console.log('[bench]', JSON.stringify(e.data));
           if (e.data.status === 'done' || e.data.status === 'error') window.__benchResult = e.data;
         };
         w.postMessage({ type: 'run', runs: Number(new URLSearchParams(location.search).get('runs') || 5),
                         skipCpu: new URLSearchParams(location.search).has('skipCpu') });
       </script>`,
      { headers: { "Content-Type": "text/html" } },
    ),

    // 3. Serve Static Assets
    "/pdf.worker.mjs": Bun.file("./vendor/pdf/pdf.worker.mjs"),
    "/manifest.json": Bun.file("./manifest.json"),
    // Dev has no precomputed app shell (that's a build.ts step). An EMPTY
    // shell is unsafe here: sw.js treats non-shell assets as cache-first, so
    // every dev request to /bundle.js and /tts.worker.js — both rebuilt
    // fresh on every request — would get pinned to whatever was cached on
    // first load and never update again. List the same paths build.ts does
    // for production so they keep the network-first treatment in dev too.
    "/sw.js": async () => new Response(
      (await Bun.file("./sw.js").text()).replace(
        "__APP_SHELL_PLACEHOLDER__",
        JSON.stringify(["/", "/index.html", "/bundle.js", "/tts.worker.js", "/ocr.worker.js"]),
      ),
      { headers: { "Content-Type": "application/javascript" } },
    ),
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
  port: process.env.PORT || 3031,
});

console.log("🚀 Server running at " + server.url);
