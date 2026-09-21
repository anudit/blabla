import { parsePdf, type PdfFile } from './parser';
import { pagesFrom, PdfPage } from './page';
export { InvalidPdfError, PdfLiteError, RenderCancelledError, UnsupportedPdfError } from './errors';
export type * from './types';

export class PdfDocument {
  private constructor(private readonly file: PdfFile, private readonly pages: PdfPage[]) {}

  static async open(bytes: ArrayBuffer | Uint8Array): Promise<PdfDocument> {
    const file = parsePdf(bytes);
    return new PdfDocument(file, pagesFrom(file));
  }

  get numPages() { return this.pages.length; }
  async getPage(pageNumber: number) {
    if (!Number.isInteger(pageNumber) || pageNumber < 1 || pageNumber > this.pages.length) throw new RangeError(`Invalid page number: ${pageNumber}`);
    return this.pages[pageNumber - 1];
  }
  async destroy() { /* Reserved for worker/cache teardown. */ }
}

