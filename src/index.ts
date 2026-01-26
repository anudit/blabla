import { serve } from "bun";
import index from "./index.html";

const server = serve({
  routes: {
    "/*": index,
    "/tts.worker.js": async () => {
      const build = await Bun.build({
        entrypoints: ["./src/tts.worker.ts"],
        target: "browser",
        minify: process.env.NODE_ENV === "production",
      });

      // Return the compiled JS
      return new Response(build.outputs[0]);
    },
  },

  development: process.env.NODE_ENV !== "production" && {
    // Enable browser hot reloading in development
    hmr: true,

    // Echo console logs from the browser to the server
    console: true,
  },
});

console.log(`🚀 Server running at ${server.url}`);
