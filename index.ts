// index.ts
import { serve } from "bun";

const server = serve({
  routes: {
    // 1. Build the Main App (Bundling React + dependencies)
    "/bundle.js": async () => {
      const build = await Bun.build({
        entrypoints: ["./frontend.tsx"],
        target: "browser",
        minify: process.env.NODE_ENV === "production",
      });
      return new Response(build.outputs[0], {
        headers: { 'Content-Type': 'application/javascript' }
      });
    },

    // 2. Build the TTS Worker
    "/tts.worker.js": async () => {
      const build = await Bun.build({
        entrypoints: ["./tts.worker.ts"],
        target: "browser",
        minify: process.env.NODE_ENV === "production",
      });
      return new Response(build.outputs[0], {
        headers: { 'Content-Type': 'application/javascript' }
      });
    },

    // 3. Serve Static Assets
    "/pdf.worker.mjs": Bun.file("node_modules/pdfjs-dist/build/pdf.worker.mjs"),
    "/manifest.json": Bun.file("./manifest.json"),
    "/sw.js": Bun.file("./sw.js"),
    "/logo.png": Bun.file("./logo.png"),
    "/16.png": Bun.file("./16.png"),
    "/32.png": Bun.file("./32.png"),
    "/128.png": Bun.file("./128.png"),
    "/180.png": Bun.file("./180.png"),
    "/og.png": Bun.file("./og.png"),


    // 4. Serve Index (Fallback for SPA)
    "/*": Bun.file("./index.html")
  },
  development: process.env.NODE_ENV !== "production" && {
    hmr: true,
  },
  port: 3031,
});

console.log("🚀 Server running at " + server.url);
