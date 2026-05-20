/**
 * This file is the entry point for the React app, it sets up the root
 * element and renders the App component to the DOM.
 *
 * It is included in `src/index.html`.
 */

import { render } from "preact";
import App from "./App";

const elem = document.getElementById("root")!;
const app = <App />;

render(app, elem);

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    // Remember whether a SW was already in control when this page loaded.
    // Used below to distinguish a first install (no reload needed) from an
    // update (new SW took over → reload to serve fresh assets).
    const hadController = !!navigator.serviceWorker.controller;

    navigator.serviceWorker.register('/sw.js')
      .then((reg) => console.log('SW registered:', reg))
      .catch((err) => console.error('SW registration failed:', err));

    // When a new SW takes control (e.g. after a Vercel deployment), reload once
    // so users automatically get the latest version without Cmd+Shift+R.
    let reloading = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (hadController && !reloading) {
        reloading = true;
        window.location.reload();
      }
    });

  });
}
