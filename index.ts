import { serve } from "bun";
import index from "./index.html"; // Correct: same dir as index.ts

const server = serve({
  routes: {
    "/tts.worker.js": async () => {
      const build = await Bun.build({
        entrypoints: ["./tts.worker.ts"], // Relative to src/
        target: "browser",
        minify: process.env.NODE_ENV === "production",
      });
      return new Response(build.outputs[0], {
        headers: { 'Content-Type': 'application/javascript' }
      });
    },
    "/pdf.worker.mjs": Bun.file("node_modules/pdfjs-dist/build/pdf.worker.mjs"),

    // Standard routes (for normal requests)
    "/manifest.json": Bun.file("./manifest.json"),
    "/sw.js": Bun.file("./sw.js"),
    "/logo.png": Bun.file("./logo.png"),
    "/frontend.tsx": Bun.file("./frontend.tsx"),
    "/*": index
  },
  development: process.env.NODE_ENV !== "production" && {
    hmr: true,
    console: true,
  },
  port: 3031,
});
console.log("🚀 Server running at " + server.url);
