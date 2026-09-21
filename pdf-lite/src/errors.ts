export class PdfLiteError extends Error {
  constructor(message: string, public readonly code: string) {
    super(message);
    this.name = 'PdfLiteError';
  }
}

export class InvalidPdfError extends PdfLiteError {
  constructor(message: string) {
    super(message, 'INVALID_PDF');
    this.name = 'InvalidPdfError';
  }
}

export class UnsupportedPdfError extends PdfLiteError {
  constructor(message: string) {
    super(message, 'UNSUPPORTED_PDF');
    this.name = 'UnsupportedPdfError';
  }
}

export class RenderCancelledError extends PdfLiteError {
  constructor() {
    super('PDF rendering was cancelled', 'CANCELLED');
    this.name = 'RenderCancelledError';
  }
}

