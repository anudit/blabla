/**
 * DOM shims for running paddleocr-webgpu inside a Web Worker.
 *
 * The package was written for a DOM context: it uses
 * `document.createElement('canvas')` and `instanceof HTMLCanvasElement`.
 * Workers expose neither, so we map both onto `OffscreenCanvas`. This module is
 * imported FIRST in the worker (ESM evaluates imports in order) so the globals
 * exist before the engine module is evaluated.
 */
const g = globalThis as any;

if (typeof g.OffscreenCanvas !== 'undefined') {
  if (typeof g.HTMLCanvasElement === 'undefined') g.HTMLCanvasElement = OffscreenCanvas;
  if (typeof g.HTMLImageElement === 'undefined') g.HTMLImageElement = class {};
  if (typeof g.document === 'undefined') {
    g.document = {
      createElement(tag: string) {
        if (tag === 'canvas') return new OffscreenCanvas(1, 1);
        throw new Error(`document.createElement('${tag}') is not supported in the OCR worker`);
      },
    };
  }
}
