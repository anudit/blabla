export type Matrix = [number, number, number, number, number, number];

export interface Viewport {
  width: number;
  height: number;
  scale: number;
  rotation: number;
  transform: Matrix;
  convertToViewportPoint(x: number, y: number): [number, number];
}

export interface TextItem {
  str: string;
  transform: Matrix;
  width: number;
  height: number;
}

export interface TextContent {
  items: TextItem[];
}

export interface PdfAnnotation {
  subtype?: string;
  rect?: number[];
  url?: string;
  contents?: string;
}

export interface RenderTask {
  promise: Promise<void>;
  cancel(): void;
}

export interface RenderParams {
  canvasContext: CanvasRenderingContext2D;
  viewport: Viewport;
  signal?: AbortSignal;
}
