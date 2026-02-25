import React from 'react';

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
  const safe = text.replace(/!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)|`[^`\n]+`/g, (m) => {
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
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
}

/** True if the text looks like markdown (has at least one ATX heading). */
export function isMarkdown(text: string): boolean {
  return /^#{1,6} \S/m.test(text);
}

// Image must come before link pattern (both start with `[`)
// Matches ![alt](src), **bold**, __bold__, ~~strike~~, *italic*, _italic_, `code`, [text](url)
export const INLINE_MD_RE = /!\[[^\]]*\]\([^)]+\)|\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`|\[[^\]]+\]\([^)]+\)/g;

/** Render inline markdown to React nodes for visual display. */
export function renderMd(text: string, linkColor: string = '#3b82f6'): React.ReactNode {
  const parts: React.ReactNode[] = [];
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
      const lm = s.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (lm) {
        parts.push(<a key={ki++} href={lm[2]} target="_blank" rel="noopener noreferrer" style={{ color: linkColor, textDecoration: 'none', textUnderlineOffset: '2px', textDecorationColor: linkColor + '80' }}>{lm[1]}</a>);
      } else {
        parts.push(s);
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
export function runsToReactNode(runs: TextRun[]): React.ReactNode {
  const parts: React.ReactNode[] = [];
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
