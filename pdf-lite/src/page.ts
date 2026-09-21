import { RenderCancelledError } from './errors';
import { PdfFile, type PdfDict, type PdfValue } from './parser';
import type { Matrix, PdfAnnotation, RenderParams, RenderTask, TextContent, TextItem, Viewport } from './types';
import jpeg from 'jpeg-js';

const identity: Matrix = [1, 0, 0, 1, 0, 0];
const numberArray = (file: PdfFile, value: PdfValue | undefined, fallback: number[]) => {
  const resolved = file.resolve(value);
  return Array.isArray(resolved) && resolved.every(x => typeof x === 'number') ? resolved as number[] : fallback;
};

function multiply(a: Matrix, b: Matrix): Matrix {
  return [a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1], a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3], a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5]];
}

function textTransform(viewport: Viewport, matrix: Matrix): Matrix {
  return multiply(viewport.transform, multiply(matrix, [1, 0, 0, -1, 0, 0]));
}

function transform(m: Matrix, x: number, y: number): [number, number] { return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]; }

export function makeViewport(box: number[], scale: number, rotation = 0): Viewport {
  const width = Math.abs(box[2] - box[0]) * scale;
  const height = Math.abs(box[3] - box[1]) * scale;
  const r = ((rotation % 360) + 360) % 360;
  const transformMatrix: Matrix = r === 90 ? [0, scale, -scale, 0, height, 0] : r === 180 ? [-scale, 0, 0, scale, width, 0] : r === 270 ? [0, -scale, scale, 0, 0, width] : [scale, 0, 0, -scale, -box[0] * scale, box[3] * scale];
  return {
    width: r === 90 || r === 270 ? height : width,
    height: r === 90 || r === 270 ? width : height,
    scale, rotation: r, transform: transformMatrix,
    convertToViewportPoint: (x, y) => transform(transformMatrix, x, y),
  };
}

function asDict(file: PdfFile, value: PdfValue | undefined): PdfDict | undefined {
  const resolved = file.resolve(value);
  return resolved && typeof resolved === 'object' && !Array.isArray(resolved) && !('ref' in resolved) ? resolved : undefined;
}

function collectPages(file: PdfFile, node: PdfValue | undefined, out: PdfDict[] = [], inherited: PdfDict = {}): PdfDict[] {
  const dict = asDict(file, node);
  if (!dict) return out;
  const merged = { ...inherited, ...dict };
  if (PdfFile.pdfName(merged.Type) === 'Page') { out.push(merged); return out; }
  const kids = file.resolve(merged.Kids);
  if (Array.isArray(kids)) for (const kid of kids) collectPages(file, kid, out, merged);
  return out;
}

function operators(source: string): Array<{ op: string; args: (string | number)[] }> {
  const tokens: (string | number)[] = [];
  let i = 0;
  while (i < source.length) {
    if (/\s/.test(source[i])) { i++; continue; }
    if (source[i] === '%') { while (i < source.length && source[i] !== '\n') i++; continue; }
    if (source[i] === '/') { let s = i++; while (i < source.length && !/[\s\[\]()<>\/{%]/.test(source[i])) i++; tokens.push(source.slice(s, i)); continue; }
    if (source[i] === '(') { let s = i++, depth = 1; while (i < source.length && depth) { if (source[i] === '\\') i += 2; else { if (source[i] === '(') depth++; if (source[i] === ')') depth--; i++; } } tokens.push(source.slice(s, i)); continue; }
    if (source[i] === '<') { let s = i++; while (i < source.length && source[i] !== '>') i++; if (i < source.length) i++; tokens.push(source.slice(s, i)); continue; }
    if (source[i] === '[' || source[i] === ']') { tokens.push(source[i++]); continue; }
    // PDF lexical delimiters can appear immediately next to numbers. For
    // example, TeX-generated TJ arrays commonly look like
    // `[(word)-417(next)] TJ`. The number must end before `(`, otherwise the
    // scanner consumes the following strings as part of one invalid token.
    const s = i++; while (i < source.length && !/[\s\[\]()<>\/{%]/.test(source[i])) i++;
    const token = source.slice(s, i); const n = Number(token); tokens.push(Number.isFinite(n) ? n : token);
  }
  const result: Array<{ op: string; args: (string | number)[] }> = []; let args: (string | number)[] = [];
  for (const token of tokens) typeof token === 'string' && /^[A-Za-z*']+$/.test(token) ? (result.push({ op: token, args }), args = []) : args.push(token);
  return result;
}

function textFromArgs(args: (string | number)[], fontMap?: Map<string, string>) {
  let text = '';
  for (const value of args) {
    if (typeof value === 'string') {
      if (value !== '[' && value !== ']') text += decodePdfText(value, fontMap);
      continue;
    }
    // TJ numbers adjust the text cursor in thousandths of a text-space unit.
    // TeX/PDFTeX emits large negative adjustments between words and tiny -1
    // adjustments when it splits a word across glyph runs. Preserve the
    // former as spaces while ignoring the latter.
    if (value <= -100 && text && !text.endsWith(' ')) text += ' ';
  }
  return text;
}

function pdfString(value: string | number): string {
  if (typeof value !== 'string') return String(value);
  const escapes: Record<string, string> = { n: '\n', r: '\r', t: '\t', b: '\b', f: '\f' };
  if (value.startsWith('(')) {
    const source = value.slice(1, -1);
    let text = '';
    for (let i = 0; i < source.length; i++) {
      if (source[i] !== '\\') { text += source[i]; continue; }
      const next = source[++i];
      if (next === undefined) break;
      if (escapes[next]) { text += escapes[next]; continue; }
      if (next === '\\' || next === '(' || next === ')') { text += next; continue; }
      if (/[0-7]/.test(next)) {
        let octal = next;
        while (octal.length < 3 && /[0-7]/.test(source[i + 1] || '')) octal += source[++i];
        text += String.fromCharCode(parseInt(octal, 8));
      } else text += next;
    }
    return text;
  }
  if (!value.startsWith('<') || !value.endsWith('>')) return value;

  const hex = value.slice(1, -1).replace(/\s/g, '');
  if (!/^[0-9a-f]*$/i.test(hex)) return '';
  const padded = hex.length % 2 === 1 ? `${hex}0` : hex;
  const bytes = new Uint8Array(padded.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(padded.slice(i * 2, i * 2 + 2), 16);

  // Many browser-generated PDFs use UTF-16BE hexadecimal strings for text.
  // Decode those directly so the small renderer does not paint the PDF syntax.
  const utf16 = bytes.length >= 2 && (bytes[0] === 0xfe && bytes[1] === 0xff || bytes.length % 2 === 0 && bytes.every((byte, i) => i % 2 === 0 ? byte === 0 : true));
  if (utf16) {
    const start = bytes[0] === 0xfe && bytes[1] === 0xff ? 2 : 0;
    let text = '';
    for (let i = start; i + 1 < bytes.length; i += 2) text += String.fromCharCode((bytes[i] << 8) | bytes[i + 1]);
    return text;
  }
  return String.fromCharCode(...bytes);
}

function hexBytes(value: string): Uint8Array {
  const hex = value.slice(1, -1).replace(/\s/g, '');
  const padded = hex.length % 2 === 1 ? `${hex}0` : hex;
  const bytes = new Uint8Array(padded.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(padded.slice(i * 2, i * 2 + 2), 16);
  return bytes;
}

function hexNumber(value: string) {
  return parseInt(value.replace(/[<>]/g, ''), 16);
}

function decodeHexTarget(value: string): string {
  const bytes = hexBytes(value);
  let text = '';
  for (let i = 0; i + 1 < bytes.length; i += 2) text += String.fromCharCode((bytes[i] << 8) | bytes[i + 1]);
  return bytes.length % 2 ? `${text}${String.fromCharCode(bytes[bytes.length - 1])}` : text;
}

function decodeRangeTarget(hex: string, offset: number): string {
  const raw = hex.slice(1, -1).replace(/\s/g, '');
  if (raw.length <= 4) return String.fromCharCode(parseInt(raw, 16) + offset);
  // TeX CMaps encode supplementary Unicode characters as UTF-16 surrogate
  // pairs, e.g. D835DEE4. Increment the low surrogate for a bfrange.
  if (raw.length === 8) {
    const high = parseInt(raw.slice(0, 4), 16);
    const low = parseInt(raw.slice(4), 16) + offset;
    const carry = Math.floor((low - 0xdc00) / 0x400);
    return String.fromCharCode(high + carry, 0xdc00 + ((low - 0xdc00) % 0x400));
  }
  return decodeHexTarget(`<${raw}>`);
}

function parseToUnicode(source: string): Map<string, string> {
  const map = new Map<string, string>();
  const bfchar = source.match(/beginbfchar([\s\S]*?)endbfchar/g) || [];
  for (const block of bfchar) {
    const pairs = block.matchAll(/(<[0-9a-f]+>)\s*(<[0-9a-f]+>)/gi);
    for (const pair of pairs) map.set(pair[1].slice(1, -1).toUpperCase(), decodeHexTarget(pair[2]));
  }
  const bfrange = source.match(/beginbfrange([\s\S]*?)endbfrange/g) || [];
  for (const block of bfrange) {
    const ranges = block.matchAll(/(<[0-9a-f]+>)\s*(<[0-9a-f]+>)\s*(<[0-9a-f]+>)/gi);
    for (const range of ranges) {
      const start = hexNumber(range[1]); const end = hexNumber(range[2]);
      for (let code = start; code <= end; code++) {
        const offset = code - start;
        map.set(code.toString(16).padStart(range[1].length - 2, '0').toUpperCase(), decodeRangeTarget(range[3], offset));
      }
    }
  }
  return map;
}

function decodePdfText(value: string | number, fontMap?: Map<string, string>): string {
  if (typeof value !== 'string' || !fontMap?.size) return pdfString(value);
  if (value.startsWith('(')) {
    const raw = pdfString(value);
    const width = [...fontMap.keys()].some(key => key.length === 4) ? 2 : 1;
    let text = '';
    for (let i = 0; i < raw.length; i += width) {
      let code = '';
      for (let j = 0; j < width && i + j < raw.length; j++) code += raw.charCodeAt(i + j).toString(16).padStart(2, '0');
      text += fontMap.get(code.toUpperCase()) ?? raw.slice(i, i + width);
    }
    return text;
  }
  if (!value.startsWith('<')) return pdfString(value);
  const hex = value.slice(1, -1).replace(/\s/g, '').toUpperCase();
  const width = [...fontMap.keys()].some(key => key.length === 4) ? 4 : 2;
  let text = '';
  for (let i = 0; i < hex.length; i += width) text += fontMap.get(hex.slice(i, i + width)) ?? pdfString(`<${hex.slice(i, i + width)}>`);
  return text;
}

export class PdfPage {
  constructor(private readonly file: PdfFile, private readonly dict: PdfDict) {}

  getViewport({ scale, rotation = 0 }: { scale: number; rotation?: number }): Viewport {
    const box = numberArray(this.file, this.dict.CropBox ?? this.dict.MediaBox, [0, 0, 612, 792]);
    const pageRotation = typeof this.file.resolve(this.dict.Rotate) === 'number' ? Number(this.file.resolve(this.dict.Rotate)) : 0;
    return makeViewport(box, scale, pageRotation + rotation);
  }

  async getTextContent(): Promise<TextContent> {
    const [source, fontMaps] = await Promise.all([this.contentSource(), this.fontMaps()]); const items: TextItem[] = [];
    let textMatrix: Matrix = [...identity]; let lineMatrix: Matrix = [...identity]; let fontSize = 12; let leading = 0;
    let fontName = '';
    for (const { op, args } of operators(source)) {
      if (op === 'BT') { textMatrix = [...identity]; lineMatrix = [...identity]; leading = 0; }
      if (op === 'Tf' && typeof args.at(-1) === 'number') { fontSize = Number(args.at(-1)); fontName = String(args[0]).replace(/^\//, ''); }
      else if (op === 'Tm' && args.length >= 6) { textMatrix = args.slice(-6) as Matrix; lineMatrix = [...textMatrix]; }
      else if (op === 'Td' && args.length >= 2) { lineMatrix = multiply(lineMatrix, [1, 0, 0, 1, Number(args[0]), Number(args[1])]); textMatrix = [...lineMatrix]; }
      else if (op === 'TD' && args.length >= 2) { leading = -Number(args[1]); lineMatrix = multiply(lineMatrix, [1, 0, 0, 1, Number(args[0]), Number(args[1])]); textMatrix = [...lineMatrix]; }
      else if (op === 'T*') { lineMatrix = multiply(lineMatrix, [1, 0, 0, 1, 0, -leading]); textMatrix = [...lineMatrix]; }
      else if (op === 'Tj' && args.length) { const str = decodePdfText(args[0], fontMaps.get(fontName)); items.push({ str, transform: [...textMatrix], width: str.length * fontSize * 0.5, height: fontSize }); textMatrix = multiply(textMatrix, [1, 0, 0, 1, str.length * fontSize * 0.5, 0]); }
      else if (op === 'TJ' && args.length) { const str = textFromArgs(args, fontMaps.get(fontName)); items.push({ str, transform: [...textMatrix], width: str.length * fontSize * 0.5, height: fontSize }); textMatrix = multiply(textMatrix, [1, 0, 0, 1, str.length * fontSize * 0.5, 0]); }
      else if (op === "'" && args.length) { lineMatrix = multiply(lineMatrix, [1, 0, 0, 1, 0, -fontSize * 1.2]); textMatrix = [...lineMatrix]; const str = decodePdfText(args[0], fontMaps.get(fontName)); items.push({ str, transform: [...textMatrix], width: str.length * fontSize * 0.5, height: fontSize }); }
      else if (op === 'ET') { textMatrix = [...identity]; lineMatrix = [...identity]; }
    }
    return { items };
  }

  async getAnnotations(): Promise<PdfAnnotation[]> {
    const annotations = this.file.resolve(this.dict.Annots);
    if (!Array.isArray(annotations)) return [];
    return annotations.flatMap(annotation => {
      const dict = asDict(this.file, annotation);
      if (!dict) return [];
      const action = asDict(this.file, dict.A);
      const uri = action && typeof this.file.resolve(action.URI) === 'string' ? String(this.file.resolve(action.URI)) : undefined;
      return [{
        subtype: PdfFile.pdfName(dict.Subtype),
        rect: numberArray(this.file, dict.Rect, []),
        url: uri,
        contents: typeof this.file.resolve(dict.Contents) === 'string' ? String(this.file.resolve(dict.Contents)) : undefined,
      }];
    });
  }

  render(params: RenderParams): RenderTask {
    let cancelled = false; const cancel = () => { cancelled = true; };
    const promise = (async () => {
      const { canvasContext: ctx, viewport, signal } = params;
      if (signal?.aborted) throw new RenderCancelledError();
      ctx.save(); ctx.setTransform(...viewport.transform); ctx.fillStyle = '#000';
      let path: Path2D | null = null; let textMatrix: Matrix = [...identity]; let lineMatrix: Matrix = [...identity]; let fontSize = 12; let leading = 0;
      let fontName = ''; const fontMaps = await this.fontMaps();
      const fontFamilies = await this.fontFamilies();
      const resources = asDict(this.file, this.dict.Resources);
      const xobjects = resources ? asDict(this.file, resources.XObject) : undefined;
      const matrixFromContext = (): Matrix => {
        const current = ctx.getTransform();
        return [current.a, current.b, current.c, current.d, current.e, current.f];
      };
      const paintTextArgs = (args: (string | number)[]) => {
        for (const value of args) {
          if (typeof value === 'number') {
            // TJ numbers move the text cursor by -adjustment/1000 text-space
            // units. They are especially important in compact code listings,
            // where nearly every glyph is split into a separate fragment.
            textMatrix = multiply(textMatrix, [1, 0, 0, 1, -value * fontSize / 1000, 0]);
            continue;
          }
          if (value === '[' || value === ']') continue;
          const str = decodePdfText(value, fontMaps.get(fontName));
          if (!str) continue;
          ctx.save();
          ctx.setTransform(...textTransform(viewport, textMatrix));
          ctx.font = `${fontSize}px ${fontFamilies.get(fontName) || 'sans-serif'}`;
          ctx.fillText(str, 0, 0);
          const advance = ctx.measureText(str).width;
          ctx.restore();
          textMatrix = multiply(textMatrix, [1, 0, 0, 1, advance, 0]);
        }
      };
      const renderXObject = async (name: string) => {
        if (!xobjects) return;
        const value = xobjects[name];
        const object = asDict(this.file, value);
        if (!object) return;
        const subtype = PdfFile.pdfName(object.Subtype);
        if (subtype === 'Form') {
          const matrix = numberArray(this.file, object.Matrix, [1, 0, 0, 1, 0, 0]) as Matrix;
          const combined = multiply(matrixFromContext(), matrix);
          const nestedViewport = { ...viewport, transform: combined };
          await new PdfPage(this.file, object).render({ canvasContext: ctx, viewport: nestedViewport }).promise;
          return;
        }
        if (subtype === 'Image') {
          const bytes = await this.file.streamBytes(value);
          const filter = PdfFile.pdfName(object.Filter);
          if (filter !== 'DCTDecode') return;
          const makeCanvas = (width: number, height: number): any => {
            if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
            const canvas = document.createElement('canvas');
            canvas.width = width; canvas.height = height;
            return canvas;
          };
          const drawWithSoftMask = async (image: any, width: number, height: number) => {
            const mask = asDict(this.file, object.SMask);
            if (!mask) { ctx.drawImage(image, 0, 0, 1, 1); return; }
            const sourceCanvas = makeCanvas(width, height);
            const sourceContext = sourceCanvas.getContext('2d');
            if (!sourceContext) return;
            sourceContext.drawImage(image, 0, 0, width, height);
            const pixels = sourceContext.getImageData(0, 0, width, height);
            const alpha = await this.file.streamBytes(mask);
            const count = Math.min(width * height, alpha.length);
            for (let pixel = 0; pixel < count; pixel++) pixels.data[pixel * 4 + 3] = alpha[pixel];
            sourceContext.putImageData(pixels, 0, 0);
            ctx.drawImage(sourceCanvas, 0, 0, 1, 1);
          };
          const colorSpace = PdfFile.pdfName(object.ColorSpace);
          if (colorSpace === 'DeviceCMYK') {
            // Native browser JPEG decoding is not dependable for four-channel
            // CMYK JPEGs: Safari/Chrome commonly paint them as black or with
            // the channels interleaved. Decode the samples and convert the
            // PDF DeviceCMYK values to RGB before putting them on the canvas.
            const decoded = jpeg.decode(bytes, { useTArray: true, formatAsRGBA: false, colorTransform: false });
            const image = ctx.createImageData(decoded.width, decoded.height);
            for (let source = 0, target = 0; target < image.data.length; source += 4, target += 4) {
              const c = decoded.data[source] / 255;
              const m = decoded.data[source + 1] / 255;
              const y = decoded.data[source + 2] / 255;
              const k = decoded.data[source + 3] / 255;
              image.data[target] = Math.round(255 * (1 - Math.min(1, c + k)));
              image.data[target + 1] = Math.round(255 * (1 - Math.min(1, m + k)));
              image.data[target + 2] = Math.round(255 * (1 - Math.min(1, y + k)));
              image.data[target + 3] = 255;
            }
            const imageCanvas = makeCanvas(decoded.width, decoded.height);
            const imageContext = imageCanvas.getContext('2d');
            if (!imageContext) return;
            imageContext.putImageData(image, 0, 0);
            await drawWithSoftMask(imageCanvas, decoded.width, decoded.height);
            return;
          }
          // Decode RGB JPEGs ourselves as well. Native browser JPEG decoders
          // can apply Adobe/YCC transforms differently from the PDF color
          // space, which produces visibly scrambled figures in some PDFs.
          const decoded = jpeg.decode(bytes, { useTArray: true, formatAsRGBA: true, colorTransform: true });
          const image = ctx.createImageData(decoded.width, decoded.height);
          image.data.set(decoded.data);
          const imageCanvas = makeCanvas(decoded.width, decoded.height);
          const imageContext = imageCanvas.getContext('2d');
          if (!imageContext) return;
          imageContext.putImageData(image, 0, 0);
          await drawWithSoftMask(imageCanvas, decoded.width, decoded.height);
        }
      };
      for (const { op, args } of operators(await this.contentSource())) {
        if (cancelled || signal?.aborted) { ctx.restore(); throw new RenderCancelledError(); }
        const n = (...xs: (string | number)[]) => xs.map(Number);
        const rgb = (r: number, g: number, b: number) => `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
        const cmyk = (c: number, m: number, y: number, k: number) => rgb(1 - Math.min(1, c + k), 1 - Math.min(1, m + k), 1 - Math.min(1, y + k));
        if (op === 'q') { ctx.save(); path = null; } else if (op === 'Q') { ctx.restore(); path = null; }
        else if (op === 'cm' && args.length >= 6) ctx.transform(...n(...args.slice(-6)) as Matrix);
        else if (op === 'Do' && args.length && typeof args[0] === 'string') await renderXObject(String(args[0]).replace(/^\//, ''));
        else if (op === 'rg' && args.length >= 3) ctx.fillStyle = rgb(Number(args[0]), Number(args[1]), Number(args[2]));
        else if (op === 'g' && args.length >= 1) ctx.fillStyle = rgb(Number(args[0]), Number(args[0]), Number(args[0]));
        else if (op === 'k' && args.length >= 4) ctx.fillStyle = cmyk(Number(args[0]), Number(args[1]), Number(args[2]), Number(args[3]));
        else if (op === 'RG' && args.length >= 3) ctx.strokeStyle = rgb(Number(args[0]), Number(args[1]), Number(args[2]));
        else if (op === 'G' && args.length >= 1) ctx.strokeStyle = rgb(Number(args[0]), Number(args[0]), Number(args[0]));
        else if (op === 'K' && args.length >= 4) ctx.strokeStyle = cmyk(Number(args[0]), Number(args[1]), Number(args[2]), Number(args[3]));
        else if (op === 'm' && args.length >= 2) { path ||= new Path2D(); path.moveTo(Number(args[0]), Number(args[1])); }
        else if (op === 'l' && path && args.length >= 2) path.lineTo(Number(args[0]), Number(args[1]));
        else if (op === 're' && args.length >= 4) { path ||= new Path2D(); path.rect(Number(args[0]), Number(args[1]), Number(args[2]), Number(args[3])); }
        else if (op === 'h' && path) path.closePath();
        else if ((op === 'W' || op === 'W*') && path) ctx.clip(path);
        else if (op === 'n') path = null;
        else if ((op === 'S' || op === 's') && path) { ctx.stroke(path); path = null; }
        else if ((op === 'f' || op === 'F' || op === 'f*') && path) { ctx.fill(path); path = null; }
        else if ((op === 'B' || op === 'B*' || op === 'b' || op === 'b*') && path) { ctx.fill(path); ctx.stroke(path); path = null; }
        else if (op === 'BT') { textMatrix = [...identity]; lineMatrix = [...identity]; leading = 0; }
        else if (op === 'Tf' && typeof args.at(-1) === 'number') { fontSize = Number(args.at(-1)); fontName = String(args[0]).replace(/^\//, ''); }
        else if (op === 'Tm' && args.length >= 6) { textMatrix = args.slice(-6) as Matrix; lineMatrix = [...textMatrix]; }
        else if (op === 'Td' && args.length >= 2) { lineMatrix = multiply(lineMatrix, [1, 0, 0, 1, Number(args[0]), Number(args[1])]); textMatrix = [...lineMatrix]; }
        else if (op === 'TD' && args.length >= 2) { leading = -Number(args[1]); lineMatrix = multiply(lineMatrix, [1, 0, 0, 1, Number(args[0]), Number(args[1])]); textMatrix = [...lineMatrix]; }
        else if (op === 'T*') { lineMatrix = multiply(lineMatrix, [1, 0, 0, 1, 0, -leading]); textMatrix = [...lineMatrix]; }
        else if (op === 'Tj' && args.length) paintTextArgs(args);
        else if (op === 'TJ' && args.length) paintTextArgs(args);
      }
      ctx.restore();
    })();
    return { promise, cancel };
  }

  cleanup() {}

  private async contentSource(): Promise<string> {
    if ('__stream' in this.dict) return this.file.stream(this.dict);
    const contents = this.file.resolve(this.dict.Contents);
    if (Array.isArray(contents)) return (await Promise.all(contents.map(item => this.file.stream(item)))).join('\n');
    return this.file.stream(contents);
  }

  private async fontMaps(): Promise<Map<string, Map<string, string>>> {
    const result = new Map<string, Map<string, string>>();
    const resources = this.file.resolve(this.dict.Resources);
    if (!resources || typeof resources !== 'object' || Array.isArray(resources) || 'ref' in resources || 'name' in resources) return result;
    const fonts = this.file.resolve((resources as PdfDict).Font);
    if (!fonts || typeof fonts !== 'object' || Array.isArray(fonts) || 'ref' in fonts || 'name' in fonts) return result;
    for (const [name, value] of Object.entries(fonts as PdfDict)) {
      const font = this.file.resolve(value);
      if (!font || typeof font !== 'object' || Array.isArray(font) || 'ref' in font || 'name' in font) continue;
      const cmap = (font as PdfDict).ToUnicode;
      if (cmap === undefined) continue;
      try { result.set(name, parseToUnicode(await this.file.stream(cmap))); } catch { /* optional map */ }
    }
    return result;
  }

  private async fontFamilies(): Promise<Map<string, string>> {
    const result = new Map<string, string>();
    const FontFaceCtor = (globalThis as any).FontFace;
    const fontSet = (globalThis as any).document?.fonts;
    if (!FontFaceCtor || !fontSet) return result;
    const resources = this.file.resolve(this.dict.Resources);
    const fonts = resources && typeof resources === 'object' && !Array.isArray(resources) && !('ref' in resources)
      ? this.file.resolve((resources as PdfDict).Font) : undefined;
    if (!fonts || typeof fonts !== 'object' || Array.isArray(fonts) || 'ref' in fonts || 'name' in fonts) return result;
    for (const [name, value] of Object.entries(fonts as PdfDict)) {
      const font = this.file.resolve(value);
      if (!font || typeof font !== 'object' || Array.isArray(font) || 'ref' in font || 'name' in font) continue;
      const baseName = PdfFile.pdfName((font as PdfDict).BaseFont) || '';
      result.set(name, /Times|NimbusRom/i.test(baseName) ? '"Times New Roman"' : /Monospace|NimbusMon/i.test(baseName) ? 'monospace' : 'serif');
      let descriptor: PdfDict | undefined;
      const direct = (font as PdfDict).FontDescriptor;
      if (direct !== undefined) descriptor = asDict(this.file, direct);
      else {
        const descendants = this.file.resolve((font as PdfDict).DescendantFonts);
        if (Array.isArray(descendants)) descriptor = asDict(this.file, asDict(this.file, descendants[0])?.FontDescriptor);
      }
      if (!descriptor) continue;
      const fileRef = descriptor.FontFile2 ?? descriptor.FontFile3 ?? descriptor.FontFile;
      if (fileRef === undefined) continue;
      try {
        const bytes = await this.file.streamBytes(fileRef);
        const family = `pdf-lite-${name}`;
        const face = new FontFaceCtor(family, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        await face.load();
        fontSet.add(face);
        result.set(name, family);
      } catch { /* Browser support for embedded Type1/CFF varies; use fallback. */ }
    }
    return result;
  }
}

export function pagesFrom(file: PdfFile): PdfPage[] {
  const catalog = [...file.values()].map(value => file.resolve(value)).find(value => value && typeof value === 'object' && !Array.isArray(value) && PdfFile.pdfName((value as PdfDict).Type) === 'Catalog') as PdfDict | undefined;
  if (!catalog) throw new Error('PDF catalog not found');
  return collectPages(file, catalog.Pages).map(dict => new PdfPage(file, dict));
}
