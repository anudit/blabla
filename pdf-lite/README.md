# pdf-lite

An app-shaped PDF engine for BlaBla.

This is a fresh implementation with a deliberately small public surface. It is
not a PDF viewer and it does not attempt to reproduce the PDF.js API.

## Current scope

- local `ArrayBuffer` / `Uint8Array` input
- classic PDF indirect objects and cross-reference tables
- page-tree traversal
- page count and page geometry
- page viewport transforms
- a narrow text-item API
- a narrow Canvas 2D renderer for basic page content operators
- cancellable render tasks and document cleanup
- an optional dedicated-worker client

## Not implemented yet

The first slice intentionally rejects or degrades these features rather than
silently producing incorrect output:

- xref streams and object streams
- Flate-compressed content streams
- embedded fonts and ToUnicode maps
- image XObjects and masks
- transparency, patterns, shadings, and form XObjects
- encryption/passwords
- annotations, forms, XFA, links, metadata, outlines, and PDF writing

Those are the next implementation milestones. The API is shaped around the
operations used by BlaBla so they can be added without importing viewer code.

## API

```ts
import { PdfDocument } from './src/index';

const pdf = await PdfDocument.open(bytes);
const page = await pdf.getPage(1);
const viewport = page.getViewport({ scale: 1.5 });
const text = await page.getTextContent();
const task = page.render({ canvasContext: ctx, viewport });
await task.promise;
page.cleanup();
await pdf.destroy();
```

## Build

```sh
bun run --cwd pdf-lite build
```

The build emits a main module and a worker module into `pdf-lite/dist`.

## Rendering regression test

The browser render harness exercises the actual Canvas 2D path against
`test/pdf-sample_0.pdf`. It asserts that text extraction contains
`Dummy PDF file` and reports a failure when the output is dominated by solid
black pixels:

```sh
bun run --cwd pdf-lite build
bun run --cwd pdf-lite render:serve
```

To test another fixture:

```sh
PDF_LITE_FIXTURE=./test/test2.pdf bun run --cwd pdf-lite render:serve
```

Open `http://127.0.0.1:4179/` in a browser. The page also exposes the result
as `window.__renderTest` for headless automation and writes the rendered canvas
to a screenshot.
