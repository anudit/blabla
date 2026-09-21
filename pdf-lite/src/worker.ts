import { PdfDocument } from './index';

type Request = { id: number; type: 'open' | 'text' | 'destroy'; bytes?: ArrayBuffer; page?: number };
const docs = new Map<number, PdfDocument>();

self.onmessage = async (event: MessageEvent<Request>) => {
  const message = event.data;
  try {
    if (message.type === 'open') {
      if (!message.bytes) throw new Error('Missing PDF bytes');
      const document = await PdfDocument.open(message.bytes);
      docs.set(message.id, document);
      self.postMessage({ id: message.id, ok: true, numPages: document.numPages });
    } else if (message.type === 'text') {
      const document = docs.get(message.id); if (!document || !message.page) throw new Error('Unknown document');
      const page = await document.getPage(message.page);
      self.postMessage({ id: message.id, ok: true, text: await page.getTextContent() });
    } else if (message.type === 'destroy') {
      await docs.get(message.id)?.destroy(); docs.delete(message.id);
      self.postMessage({ id: message.id, ok: true });
    }
  } catch (error) {
    self.postMessage({ id: message.id, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
};

