import { InvalidPdfError, UnsupportedPdfError } from './errors';

export type PdfName = { name: string };
export type PdfRef = { ref: number };
export type PdfValue = null | boolean | number | string | PdfName | PdfRef | PdfValue[] | PdfDict;
export interface PdfDict { [key: string]: PdfValue; }

const isRef = (value: PdfValue | undefined): value is PdfRef => !!value && typeof value === 'object' && !Array.isArray(value) && 'ref' in value;
const isDict = (value: PdfValue | undefined): value is PdfDict => !!value && typeof value === 'object' && !Array.isArray(value) && !('ref' in value) && !('name' in value);

const latin1 = (bytes: Uint8Array) => {
  const chunks: string[] = [];
  for (let i = 0; i < bytes.length; i += 8192) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
  return chunks.join('');
};
const bytesFromLatin1 = (value: string) => Uint8Array.from(value, character => character.charCodeAt(0) & 0xff);
const latin1FromBytes = (bytes: Uint8Array) => {
  const chunks: string[] = [];
  for (let i = 0; i < bytes.length; i += 8192) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
  return chunks.join('');
};
const ws = /\s/;

class Tokens {
  constructor(public readonly source: string, public pos = 0) {}

  skip() { while (this.pos < this.source.length && ws.test(this.source[this.pos])) this.pos++; }

  next(): string | null {
    this.skip();
    if (this.pos >= this.source.length) return null;
    const start = this.pos;
    const c = this.source[this.pos];
    if (c === '/') {
      this.pos++;
      while (this.pos < this.source.length && !ws.test(this.source[this.pos]) && !'[]<>/()'.includes(this.source[this.pos])) this.pos++;
      return this.source.slice(start, this.pos);
    }
    if (c === '<' && this.source[this.pos + 1] !== '<') {
      this.pos++;
      while (this.pos < this.source.length && this.source[this.pos] !== '>') this.pos++;
      if (this.pos < this.source.length) this.pos++;
      return this.source.slice(start, this.pos);
    }
    if ('[]<>'.includes(c)) {
      this.pos += c === '<' && this.source[this.pos + 1] === '<' ? 2 : c === '>' && this.source[this.pos + 1] === '>' ? 2 : 1;
      return this.source.slice(start, this.pos);
    }
    if (c === '(') {
      this.pos++;
      let depth = 1;
      while (this.pos < this.source.length && depth) {
        if (this.source[this.pos] === '\\') this.pos += 2;
        else { if (this.source[this.pos] === '(') depth++; if (this.source[this.pos] === ')') depth--; this.pos++; }
      }
      return this.source.slice(start, this.pos);
    }
    while (this.pos < this.source.length && !ws.test(this.source[this.pos]) && !'[]<>/'.includes(this.source[this.pos])) this.pos++;
    return this.source.slice(start, this.pos);
  }
}

function parseValue(t: Tokens): PdfValue {
  const token = t.next();
  if (token === null) throw new InvalidPdfError('Unexpected end of PDF object');
  if (token === '<<') {
    const dict: PdfDict = {};
    while (true) {
      const key = t.next();
      if (key === '>>') return dict;
      if (!key?.startsWith('/')) throw new InvalidPdfError('Invalid dictionary key');
      dict[key.slice(1)] = parseValue(t);
    }
  }
  if (token === '[') {
    const values: PdfValue[] = [];
    while (true) { if (t.source.slice(t.pos).trimStart().startsWith(']')) { t.next(); return values; } values.push(parseValue(t)); }
  }
  if (token.startsWith('/')) return { name: token.slice(1) };
  if (token.startsWith('(')) return decodeString(token.slice(1, -1));
  if (token.startsWith('<') && token.endsWith('>')) return token;
  if (token === 'true' || token === 'false') return token === 'true';
  if (token === 'null') return null;
  const number = Number(token);
  if (Number.isFinite(number)) {
    const save = t.pos;
    const generation = t.next();
    const marker = t.next();
    if (generation && marker === 'R' && /^\d+$/.test(generation)) return { ref: number };
    t.pos = save;
    return number;
  }
  throw new InvalidPdfError(`Unsupported PDF token: ${token}`);
}

function decodeString(value: string): string {
  let out = '';
  for (let i = 0; i < value.length; i++) {
    if (value[i] !== '\\') { out += value[i]; continue; }
    const next = value[++i];
    const escapes: Record<string, string> = { n: '\n', r: '\r', t: '\t', b: '\b', f: '\f', '(': '(', ')': ')', '\\': '\\' };
    if (escapes[next]) out += escapes[next];
    else if (/[0-7]/.test(next)) { let oct = next; while (oct.length < 3 && /[0-7]/.test(value[i + 1] || '')) oct += value[++i]; out += String.fromCharCode(parseInt(oct, 8)); }
    else out += next || '';
  }
  return out;
}

export class PdfFile {
  readonly source: string;
  private readonly objects = new Map<number, PdfValue>();

  private constructor(bytes: Uint8Array) {
    this.source = latin1(bytes);
    if (!this.source.startsWith('%PDF-')) throw new InvalidPdfError('Missing PDF header');
    this.readObjects();
  }

  static from(bytes: ArrayBuffer | Uint8Array) { return new PdfFile(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)); }

  get(id: number): PdfValue | undefined { return this.objects.get(id); }
  values() { return this.objects.values(); }

  resolve(value: PdfValue | undefined): PdfValue | undefined {
    let current = value;
    for (let i = 0; i < 16 && isRef(current); i++) current = this.objects.get(current.ref);
    return current;
  }

  private readObjects() {
    const objectRe = /(\d+)\s+(\d+)\s+obj\b/g;
    let match: RegExpExecArray | null;
    while ((match = objectRe.exec(this.source))) {
      const id = Number(match[1]);
      const bodyStart = objectRe.lastIndex;
      const end = this.source.indexOf('endobj', bodyStart);
      if (end < 0) throw new InvalidPdfError(`Object ${id} has no endobj`);
      const body = this.source.slice(bodyStart, end).trim();
      const streamAt = body.indexOf('stream');
      const parsed = streamAt >= 0 ? body.slice(0, streamAt).trim() : body;
      const value = parseValue(new Tokens(parsed));
      if (streamAt >= 0 && isDict(value) && 'Length' in value) {
        const streamKeyword = this.source.indexOf('stream', bodyStart);
        const afterStream = streamKeyword + 6;
        let start = afterStream;
        if (this.source[start] === '\r') start++;
        if (this.source[start] === '\n') start++;
        const lengthValue = this.resolve(value.Length);
        let length: number;
        if (typeof lengthValue === 'number') length = lengthValue;
        else {
          const endstream = this.source.indexOf('endstream', start);
          length = endstream - start;
          (value as PdfDict).__streamStart = start;
          if (value.Length && typeof value.Length === 'object' && 'ref' in value.Length) (value as PdfDict).__streamLengthRef = value.Length.ref;
        }
        (value as PdfDict).__stream = this.source.slice(start, start + length);
      }
      this.objects.set(id, value);
      objectRe.lastIndex = end + 6;
    }
    for (const value of this.objects.values()) {
      if (!isDict(value)) continue;
      const stream = value as PdfDict & { __stream?: string; __streamStart?: number; __streamLengthRef?: number };
      if (typeof stream.__streamStart !== 'number' || typeof stream.__streamLengthRef !== 'number') continue;
      const length = this.resolve({ ref: stream.__streamLengthRef });
      if (typeof length === 'number') stream.__stream = this.source.slice(stream.__streamStart, stream.__streamStart + length);
    }
    if (!this.objects.size) throw new InvalidPdfError('No PDF objects found');
  }

  static pdfName(value: PdfValue | undefined): string | undefined { return value && typeof value === 'object' && !Array.isArray(value) && 'name' in value ? String(value.name) : undefined; }

  async stream(value: PdfValue | undefined): Promise<string> {
    return latin1FromBytes(await this.streamBytes(value));
  }

  async streamBytes(value: PdfValue | undefined): Promise<Uint8Array> {
    const resolved = this.resolve(value);
    if (!isDict(resolved) || !('__stream' in resolved)) throw new InvalidPdfError('Expected PDF stream');
    const filter = PdfFile.pdfName(resolved.Filter);
    const raw = bytesFromLatin1(String(resolved.__stream));
    if (!filter || filter === 'None') return raw;
    if (filter === 'DCTDecode') return raw;
    if (filter !== 'FlateDecode') throw new UnsupportedPdfError(`Stream filter ${filter} is not implemented yet`);
    if (typeof DecompressionStream === 'undefined') throw new UnsupportedPdfError('FlateDecode requires DecompressionStream');
    const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
}

export function parsePdf(bytes: ArrayBuffer | Uint8Array) { return PdfFile.from(bytes); }
