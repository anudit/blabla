import { PdfDocument } from '../src/index';

function makePdf(streamBytes = new TextEncoder().encode('BT\n/F1 12 Tf\n20 50 Td\n(Hello) Tj\nET\n'), flate = false) {
  const bodies = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R /Resources << >> >>',
    `<< /Length ${streamBytes.length}${flate ? ' /Filter /FlateDecode' : ''} >>\nstream\n${String.fromCharCode(...streamBytes)}endstream`,
  ];
  let output = '%PDF-1.4\n'; const offsets = [0];
  bodies.forEach((body, index) => { offsets.push(output.length); output += `${index + 1} 0 obj\n${body}\nendobj\n`; });
  const xref = output.length; output += `xref\n0 ${bodies.length + 1}\n0000000000 65535 f \n`;
  for (let i = 1; i < offsets.length; i++) output += `${String(offsets[i]).padStart(10, '0')} 00000 n \n`;
  output += `trailer\n<< /Size ${bodies.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Uint8Array.from(output, character => character.charCodeAt(0) & 0xff);
}

const document = await PdfDocument.open(makePdf());
if (document.numPages !== 1) throw new Error(`Expected one page, got ${document.numPages}`);
const page = await document.getPage(1);
const viewport = page.getViewport({ scale: 2 });
const text = await page.getTextContent();
if (viewport.width !== 400 || viewport.height !== 200) throw new Error('Viewport geometry failed');
if (text.items[0]?.str !== 'Hello') throw new Error(`Text extraction failed: ${JSON.stringify(text)}`);
await document.destroy();

const compressed = await new Response(
  new Blob([new TextEncoder().encode('BT\n/F1 12 Tf\n20 50 Td\n(Hello) Tj\nET\n')])
    .stream().pipeThrough(new CompressionStream('deflate')),
).arrayBuffer();
const compressedDocument = await PdfDocument.open(makePdf(new Uint8Array(compressed), true));
const compressedText = await (await compressedDocument.getPage(1)).getTextContent();
if (compressedText.items[0]?.str !== 'Hello') throw new Error('Flate text extraction failed');
await compressedDocument.destroy();

const hexDocument = await PdfDocument.open(makePdf(new TextEncoder().encode('BT\n/F1 12 Tf\n20 50 Td\n<00480065006C006C006F> Tj\nET\n')));
const hexText = await (await hexDocument.getPage(1)).getTextContent();
if (hexText.items[0]?.str !== 'Hello') throw new Error(`Hex text extraction failed: ${JSON.stringify(hexText)}`);
await hexDocument.destroy();
console.log('pdf-lite smoke test passed');
