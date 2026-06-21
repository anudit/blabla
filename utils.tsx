import type { ComponentChildren } from 'preact';

export function findTitleInToc(toc: any[], href: string): string | null {
  for (const entry of toc) {
    // Basic match: if either contains the other, it's likely a match for this chapter
    const entryHref = entry.href || '';
    if (entryHref && (href.includes(entryHref) || entryHref.includes(href))) {
      return entry.label;
    }
    if (entry.subitems) {
      const sub = findTitleInToc(entry.subitems, href);
      if (sub) return sub;
    }
  }
  return null;
}

export function extractWords(text: string): string[] {
  return text.trim().split(/\s+/).filter(Boolean);
}

export function calculateWordTimings(text: string, duration: number): { start: number; end: number; index: number }[] {
  const words = extractWords(text);
  if (words.length === 0 || duration <= 0) return [];
  const totalChars = text.length;
  const timings: { start: number; end: number; index: number }[] = [];
  let charPos = 0;
  words.forEach((word, i) => {
    const startChar = charPos;
    charPos += word.length;
    if (i < words.length - 1) charPos += 1; // space
    timings.push({ start: (startChar / totalChars) * duration, end: (charPos / totalChars) * duration, index: i });
  });
  return timings;
}

export function extractSentences(text: string): string[] {
  // Protect markdown links, images, and inline code from being split at
  // dots/! inside URLs or code (e.g. https://dev.to/... would otherwise
  // be treated as a sentence boundary).
  const saved: string[] = [];
  const safe = text.replace(/\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)|`[^`\n]+`/g, (m) => {
    return `\x00${saved.push(m) - 1}\x00`;
  });

  const sentences: string[] = [];
  const sentenceRegex = /[^.!?]+[.!?]+/g;
  let lastIndex = 0;
  let match;
  while ((match = sentenceRegex.exec(safe)) !== null) {
    const sText = match[0].trim().replace(/\s+/g, ' ');
    if (sText.length > 0) sentences.push(sText);
    lastIndex = match.index + match[0].length;
  }
  const remaining = safe.slice(lastIndex).trim().replace(/\s+/g, ' ');
  if (remaining.length > 0) sentences.push(remaining);

  // Restore the protected patterns
  return sentences.map(s => s.replace(/\x00(\d+)\x00/g, (_, i) => saved[+i] ?? ''));
}

/** Strip inline markdown syntax for clean TTS text. */
export function stripMd(s: string): string {
  return s
    .replace(/!\[[^\]]*\]\([^)]+\)/g, '')   // remove images entirely
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]*)\]\([^)]+\)/g, '$1');
}

/** True if the text looks like markdown (has at least one ATX heading). */
export function isMarkdown(text: string): boolean {
  return /^#{1,6} \S/m.test(text);
}

export function isImageUrl(url: string): boolean {
  return /\.(jpeg|jpg|png|gif|webp|svg|bmp|avif|heic)([?#][^\s)]*)?$/i.test(url) || /^data:image\//i.test(url);
}

// Link-with-image first, then image, then link (all start with `[`)
// Matches [![alt](src)](url), ![alt](src), **bold**, __bold__, ~~strike~~, *italic*, _italic_, `code`, [text](url) where text may be empty
export const INLINE_MD_RE = /\[!\[[^\]]*\]\([^)]+\)\]\([^)]+\)|!\[[^\]]*\]\([^)]+\)|\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`|\[[^\]]*\]\([^)]+\)/g;

/** Render inline markdown to React nodes for visual display. */
export function renderMd(text: string, linkColor: string = '#3b82f6'): ComponentChildren {
  const parts: ComponentChildren[] = [];
  let ki = 0, last = 0;
  for (const m of text.matchAll(new RegExp(INLINE_MD_RE.source, 'g'))) {
    if (m.index! > last) parts.push(text.slice(last, m.index));
    const s = m[0];
    if (s.startsWith('![')) {
      const imgM = s.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
      if (imgM) parts.push(<img key={ki++} src={imgM[2]} alt={imgM[1]} style={{ maxWidth: '100%', borderRadius: '4px', verticalAlign: 'middle', margin: '0 2px' }} />);
    } else if (s.startsWith('**') || s.startsWith('__')) {
      parts.push(<strong key={ki++}>{s.slice(2, -2)}</strong>);
    } else if (s.startsWith('~~')) {
      parts.push(<s key={ki++}>{s.slice(2, -2)}</s>);
    } else if (s.startsWith('*') || s.startsWith('_')) {
      parts.push(<em key={ki++}>{s.slice(1, -1)}</em>);
    } else if (s.startsWith('`')) {
      parts.push(<code key={ki++} style={{ fontFamily: 'monospace', fontSize: '0.88em', padding: '0.1em 0.25em', backgroundColor: 'rgba(127,127,127,0.15)', borderRadius: '3px' }}>{s.slice(1, -1)}</code>);
    } else {
      const imgLinkM = s.match(/^\[!\[([^\]]*)\]\(([^)]+)\)\]\(([^)]+)\)$/);
      if (imgLinkM) {
        // Link wraps an image: [![alt](src)](url) -> render the image
        parts.push(<img key={ki++} src={imgLinkM[2]} alt={imgLinkM[1]} style={{ maxWidth: '100%', borderRadius: '4px', verticalAlign: 'middle', margin: '0 2px' }} />);
      } else {
        const lm = s.match(/^\[([^\]]*)\]\(([^)]+)\)$/);
        if (lm) {
          const linkText = lm[1];
          const linkUrl = lm[2];
          if (!linkText.trim() && isImageUrl(linkUrl)) {
            // Empty link text pointing to an image URL -> render as image
            parts.push(<img key={ki++} src={linkUrl} alt={linkText} style={{ maxWidth: '100%', borderRadius: '4px', verticalAlign: 'middle', margin: '0 2px' }} />);
          } else {
            const label = linkText.trim() || linkUrl;
            parts.push(<a key={ki++} href={linkUrl} target="_blank" rel="noopener noreferrer" style={{ color: linkColor, textDecoration: 'none', textUnderlineOffset: '2px', textDecorationColor: linkColor + '80' }}>{label}</a>);
          }
        } else {
          parts.push(s);
        }
      }
    }
    last = m.index! + s.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length > 0 ? <>{parts}</> : text;
}

// ── EPUB inline-formatting helpers ────────────────────────────────────
export type TextRun = { text: string; em: boolean; strong: boolean; br?: boolean };

/** Walk a DOM element and collect text runs with italic/bold context. */
export function extractRuns(el: Element): TextRun[] {
  const runs: TextRun[] = [];
  function walk(node: Node, em: boolean, strong: boolean) {
    if (node.nodeType === Node.TEXT_NODE) {
      const t = node.textContent ?? '';
      if (t) runs.push({ text: t, em, strong });
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      const tag = (node as Element).tagName.toLowerCase();
      if (tag === 'script' || tag === 'style') return;
      if (tag === 'br') { runs.push({ text: '', em, strong, br: true }); return; }
      const nextEm     = em     || tag === 'em' || tag === 'i';
      const nextStrong = strong || tag === 'strong' || tag === 'b';
      Array.from(node.childNodes).forEach(c => walk(c, nextEm, nextStrong));
    }
  }
  Array.from(el.childNodes).forEach(c => walk(c, false, false));
  return runs;
}

/** Convert runs to a React node, preserving em/strong/br formatting. */
export function runsToReactNode(runs: TextRun[]): ComponentChildren {
  const parts: ComponentChildren[] = [];
  let ki = 0;
  for (const run of runs) {
    if (run.br) { parts.push(<br key={ki++} />); continue; }
    const text = run.text;
    if (!text) continue;
    if (run.em && run.strong) parts.push(<strong key={ki++}><em>{text}</em></strong>);
    else if (run.em)           parts.push(<em key={ki++}>{text}</em>);
    else if (run.strong)       parts.push(<strong key={ki++}>{text}</strong>);
    else                       parts.push(text);
  }
  return parts.length ? <>{parts}</> : null;
}

export function segmentTextLines(canvas: HTMLCanvasElement): { y0: number; y1: number; x0: number; x1: number }[] {
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const { width: W, height: H } = canvas;
  const imgData = ctx.getImageData(0, 0, W, H);
  const data = imgData.data;

  // Grayscale + Otsu binarization (text = dark = 1, background = 0)
  const gray = new Uint8Array(W * H);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    gray[j] = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114) | 0;
  }

  // Otsu threshold
  const hist = new Array(256).fill(0);
  for (let j = 0; j < gray.length; j++) hist[gray[j]]++;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, maxVar = 0, threshold = 127;
  for (let i = 0; i < 256; i++) {
    wB += hist[i];
    if (wB === 0) continue;
    const wF = gray.length - wB;
    if (wF === 0) break;
    sumB += i * hist[i];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) { maxVar = between; threshold = i; }
  }

  // Binary: text pixels (darker than threshold) = 1
  const bw = new Uint8Array(W * H);
  for (let j = 0; j < gray.length; j++) bw[j] = gray[j] < threshold ? 1 : 0;

  // Erase full-width horizontal ruled lines (>55% of row dark = horizontal rule, not text).
  // Also erase ±0 adjacent rows so the gap they create exceeds mergeGap, preventing
  // adjacent text lines from merging across a ruled line.
  const hRuleThresh = Math.floor(W * 0.55);
  const ruleRows = new Set<number>();
  for (let y = 0; y < H; y++) {
    const row = y * W; let cnt = 0;
    for (let x = 0; x < W; x++) cnt += bw[row + x];
    if (cnt > hRuleThresh) ruleRows.add(y);
  }
  const erasePad = 0;
  for (const ry of ruleRows) {
    for (let dy = -erasePad; dy <= erasePad; dy++) {
      const yt = ry + dy;
      if (yt >= 0 && yt < H) bw.fill(0, yt * W, (yt + 1) * W);
    }
  }

  // Erase full-height vertical rules (margin lines, left/right borders).
  // A column dark in >70% of rows is a vertical rule. Text characters
  // occupy at most ~52% of page height even if they appear on every line.
  const vRuleThresh = Math.floor(H * 0.70);
  const colCounts = new Int32Array(W);
  for (let y = 0; y < H; y++) { const row = y * W; for (let x = 0; x < W; x++) colCounts[x] += bw[row + x]; }
  for (let x = 0; x < W; x++) {
    if (colCounts[x] > vRuleThresh) {
      for (let y = 0; y < H; y++) bw[y * W + x] = 0;
    }
  }

  // Horizontal dilation: for each row, a pixel is 1 if any pixel within
  // a horizontal window of size W/6 is 1.
  const dilWin = Math.max(3, Math.floor(W / 6));
  const dilated = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let count = 0;
    for (let x = 0; x < W + dilWin; x++) {
      if (x < W && bw[row + x]) count++;
      if (x >= dilWin && bw[row + x - dilWin]) count--;
      if (x >= dilWin - 1) {
        dilated[row + (x - dilWin + 1)] = count > 0 ? 1 : 0;
      }
    }
  }

  // Horizontal projection for vertical bands
  const proj = new Int32Array(H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let s = 0;
    for (let x = 0; x < W; x++) s += dilated[row + x];
    proj[y] = s;
  }

  const maxProj = Math.max(...proj);
  const minProj = Math.min(...proj);
  const projRange = maxProj - minProj;
  const bandThreshold = minProj + Math.max(2, projRange * 0.08);
  const scale = H / 842;
  const minLineHeight = Math.max(3, Math.round(2.5 * scale));

  const rawLines: { y0: number; y1: number }[] = [];
  let i = 0;
  while (i < H) {
    if (proj[i] > bandThreshold) {
      const start = i;
      while (i < H && proj[i] > bandThreshold) i++;
      const end = i;
      if (end - start > minLineHeight) rawLines.push({ y0: start, y1: end });
    }
    i++;
  }

  // Merge close lines
  const merged: { y0: number; y1: number }[] = [];
  const mergeGap = Math.max(0, Math.round(1.5 * scale));
  for (const ln of rawLines) {
    if (merged.length > 0 && ln.y0 - merged[merged.length - 1].y1 < mergeGap) {
      merged[merged.length - 1].y1 = ln.y1;
    } else {
      merged.push({ y0: ln.y0, y1: ln.y1 });
    }
  }

  // Horizontal crop: for each merged band, find leftmost/rightmost text columns
  // using the original binary image (undilated) for tight character bounds.
  // Also filter out bands with too few text pixels (page footer / blank lines).
  const padH = 8;
  const minTextPx = Math.max(10, Math.round(W * 0.25)); // ~500 px — even a single short word
  const bands = merged
    .map(({ y0, y1 }) => {
      let x0 = W, x1 = 0, count = 0;
      const y0p = Math.max(0, y0 - padH);
      const y1p = Math.min(H, y1 + padH);
      for (let y = y0p; y < y1p; y++) {
        const row = y * W;
        for (let x = 0; x < W; x++) {
          if (bw[row + x]) {
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            count++;
          }
        }
      }
      // Always use full page width so the resize keeps enough CTC time-steps
      // for long text lines (tight bw-derived x-crop reduces T below the
      // minimum needed for reliable CTC decoding with ~75+ chars/line).
      return { y0, y1, x0: 0, x1: W - 1, count };
    });
  const filtered = bands.filter(b => b.count >= minTextPx);
  return filtered.map(({ y0, y1, x0, x1 }) => ({ y0, y1, x0, x1 }));
}

export function parseOcrPageText(text: string) {
  const newSentences: any[] = [];
  const contentData: any[] = [];
  let globalLineIdCounter = 0;

  let paragraphs = text.split(/\n{2,}/).map(p => p.trim()).filter(Boolean);
  if (paragraphs.length <= 1) {
    paragraphs = text.split('\n').map(p => p.trim()).filter(Boolean);
  }

  for (let i = 0; i < paragraphs.length; i++) {
    const paraText = paragraphs[i];
    const startLineId = globalLineIdCounter;
    const paraId = `ocr-para-${i}`;
    const paraSentences: any[] = [];

    const sents = extractSentences(paraText);
    for (const sText of sents) {
      const lineId = globalLineIdCounter++;
      newSentences.push({ text: sText, lines: [lineId] });
      paraSentences.push({ id: lineId, text: sText });
    }

    if (paraSentences.length > 0) {
      contentData.push({
        type: 'paragraph',
        id: paraId,
        sentences: paraSentences,
        startLineId,
        endLineId: globalLineIdCounter - 1
      });
    }
  }

  return { sentences: newSentences, content: contentData };
}

const MODEL_CACHE = 'paddleocr-cache-v1';

export async function loadWithCache(url: string, onProgress: (pct: number) => void): Promise<string> {
  const cache = await caches.open(MODEL_CACHE);
  const cached = await cache.match(url);
  if (cached) {
    onProgress(1);
    return URL.createObjectURL(await cached.blob());
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  
  const totalBytes = parseInt(response.headers.get('Content-Length') ?? '0', 10);
  const reader = response.body?.getReader();
  if (!reader) {
    const clone = response.clone();
    await cache.put(url, response);
    onProgress(1);
    return URL.createObjectURL(await clone.blob());
  }
  
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.byteLength;
      if (totalBytes > 0) {
        onProgress(received / totalBytes);
      }
    }
  }
  
  const blob = new Blob(chunks);
  const cachedResponse = new Response(blob, {
    headers: {
      'Content-Type': response.headers.get('Content-Type') || 'application/octet-stream',
      'Content-Length': String(blob.size),
    }
  });
  await cache.put(url, cachedResponse.clone());
  return URL.createObjectURL(blob);
}

