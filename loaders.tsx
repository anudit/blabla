import { unzipSync } from 'fflate';
import {
  sentencesSignal, fileTypeSignal, outlineSignal,
} from './signals';
import {
  findTitleInToc, extractSentences, stripMd, isMarkdown, extractRuns,
} from './utils';

export const loadMarkdown = (
  raw: string,
  setEpubContent: (data: any[]) => void,
  setShowTextInput: (show: boolean) => void,
  setTextInputValue: (val: string) => void
) => {
  const newSentences: any[] = [];
  const contentData: any[] = [];
  const outline: any[] = [];
  let idCounter = 0;

  let content = raw;
  const jinaMarker = '\nMarkdown Content:\n';
  const jinaIdx = raw.indexOf(jinaMarker);
  if (jinaIdx !== -1) content = raw.slice(jinaIdx + jinaMarker.length).trimStart();

  let frontmatter: { title?: string; description?: string; image?: string } | null = null;
  const fmMatch = content.match(/^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*\r?\n/);
  if (fmMatch) {
    frontmatter = {};
    for (const ln of fmMatch[1].split('\n')) {
      const m = ln.match(/^(\w+):\s*(.+)$/);
      if (m) (frontmatter as any)[m[1].toLowerCase()] = m[2].trim();
    }
    content = content.slice(fmMatch[0].length).trimStart();
  }
  if (frontmatter && (frontmatter.title || frontmatter.description || frontmatter.image)) {
    contentData.push({ type: 'frontmatter', id: `fm-${idCounter++}`, ...frontmatter });
  }

  let inFence = false, fenceLines: string[] = [], paraLines: string[] = [], tableLines: string[] = [], currentHeaderId: string | null = null;
  const parseTableRow = (line: string): string[] => {
    const cells = line.split('|').map(c => c.trim());
    if (cells[0] === '') cells.shift();
    if (cells[cells.length - 1] === '') cells.pop();
    return cells;
  };
  const flushTable = () => {
    if (!tableLines.length) return;
    const parsed = tableLines.map(parseTableRow);
    const dataRows = parsed.filter(row => !row.every(c => /^[-:\s]+$/.test(c)));
    if (dataRows.length) {
      const headers = dataRows[0], rows = dataRows.slice(1);
      contentData.push({ type: 'table', id: `table-${idCounter++}`, headers, rows, headerId: currentHeaderId });
    }
    tableLines = [];
  };
  const flushPara = () => {
    if (!paraLines.length) return;
    const block = paraLines.join(' ').replace(/\s+/g, ' ').trim();
    paraLines = [];
    if (!block) return;
    const paraId = `para-${idCounter}`;
    const startLineId = idCounter;
    const paraSentences: any[] = [];
    for (const s of extractSentences(block)) {
      const clean = stripMd(s);
      if (!clean.trim()) continue;
      const lineId = idCounter++;
      newSentences.push({ text: clean, lines: [lineId], headerId: currentHeaderId });
      paraSentences.push({ id: lineId, text: clean });
    }
    if (paraSentences.length > 0) {
      contentData.push({ 
        type: 'paragraph', id: paraId, sentences: paraSentences, headerId: currentHeaderId,
        startLineId, endLineId: idCounter - 1 
      });
    }
  };

  for (const line of content.split('\n')) {
    if (/^(`{3,}|~{3,})/.test(line)) {
      if (!inFence) { flushPara(); inFence = true; fenceLines = []; }
      else { inFence = false; contentData.push({ type: 'code', id: `code-${idCounter++}`, text: fenceLines.join('\n'), headerId: currentHeaderId }); }
      continue;
    }
    if (inFence) { fenceLines.push(line); continue; }
    const hm = line.match(/^(#{1,6})\s+(.+)$/);
    if (hm) {
      flushPara();
      currentHeaderId = `header-${idCounter++}`;
      const headText = stripMd(hm[2].replace(/\s+#+\s*$/, '').trim());
      contentData.push({ type: 'header', id: currentHeaderId, text: headText, level: hm[1].length });
      outline.push({ id: currentHeaderId, text: headText, level: hm[1].length });
      continue;
    }
    if (/^\s{0,3}([-*_]\s*){3,}$/.test(line) && !/\w/.test(line)) { flushPara(); contentData.push({ type: 'hr', id: `hr-${idCounter++}` }); continue; }
    if (!line.trim()) { flushPara(); flushTable(); continue; }
    if (line.includes('|')) { flushPara(); tableLines.push(line); continue; }
    if (tableLines.length) flushTable();
    const imgM = line.match(/^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$/);
    if (imgM) { flushPara(); contentData.push({ type: 'image', id: `img-${idCounter++}`, src: imgM[2], alt: imgM[1], headerId: currentHeaderId }); continue; }
    const li = line.match(/^\s*(?:[-*+]|\d+[.)]) (.*)/);
    if (li) {
      const t = li[1].trim().replace(/!\[[^\]]*\]\([^)]+\)/g, '').trim();
      if (t) paraLines.push(/[.!?:;]$/.test(t) ? t : t + '.');
      continue;
    }
    const bq = line.match(/^>\s*(.*)/);
    if (bq) { paraLines.push(bq[1]); continue; }
    paraLines.push(line);
  }
  if (inFence) contentData.push({ type: 'code', id: `code-${idCounter++}`, text: fenceLines.join('\n') });
  flushTable(); flushPara();
  if (!newSentences.length) return;
  fileTypeSignal.value = 'text';
  outlineSignal.value = outline;
  sentencesSignal.value = newSentences;
  setEpubContent(contentData);
  setShowTextInput(false);
  setTextInputValue('');
};

export const loadText = (
  text: string,
  setEpubContent: (data: any[]) => void,
  setShowTextInput: (show: boolean) => void,
  setTextInputValue: (val: string) => void
) => {
  if (isMarkdown(text)) { loadMarkdown(text, setEpubContent, setShowTextInput, setTextInputValue); return; }
  const newSentences: any[] = [], contentData: any[] = [];
  let globalLineIdCounter = 0;
  const paragraphs = text.split(/\n{2,}/).map(p => p.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim()).filter(p => p.length > 0);
  for (const paraText of paragraphs) {
    const startLineId = globalLineIdCounter;
    const paraId = `para-${globalLineIdCounter}`, paraSentences: any[] = [];
    for (const sText of extractSentences(paraText)) {
      const lineId = globalLineIdCounter++;
      newSentences.push({ text: sText, lines: [lineId] });
      paraSentences.push({ id: lineId, text: sText });
    }
    if (paraSentences.length > 0) {
      contentData.push({ 
        type: 'paragraph', id: paraId, sentences: paraSentences,
        startLineId, endLineId: globalLineIdCounter - 1
      });
    }
  }
  if (newSentences.length === 0) return;
  fileTypeSignal.value = 'text';
  outlineSignal.value = [];
  sentencesSignal.value = newSentences;
  setEpubContent(contentData);
  setShowTextInput(false);
  setTextInputValue('');
};

export const loadEPUB = async (
  data: ArrayBuffer,
  onChunk: (sentences: any[], content: any[], outline: any[], isFinal: boolean) => void,
  setIsDocLoading: (loading: boolean) => void
) => {
  setIsDocLoading(true);
  try {
    const files = unzipSync(new Uint8Array(data)), decoder = new TextDecoder(), parser = new DOMParser();
    const containerXml = decoder.decode(files['META-INF/container.xml']), containerDoc = parser.parseFromString(containerXml, 'text/xml');
    const opfPath = containerDoc.querySelector('rootfile')?.getAttribute('full-path');
    if (!opfPath) throw new Error("No OPF file found");
    const opfDir = opfPath.includes('/') ? opfPath.substring(0, opfPath.lastIndexOf('/') + 1) : '', opfXml = decoder.decode(files[opfPath]), opfDoc = parser.parseFromString(opfXml, 'text/xml');
    const manifest: Record<string, string> = {};
    opfDoc.querySelectorAll('manifest > item').forEach(el => { const id = el.getAttribute('id'), href = el.getAttribute('href'); if (id && href) manifest[id] = href; });
    const spineIds = Array.from(opfDoc.querySelectorAll('spine > itemref')).map(el => el.getAttribute('idref')!);
    let toc: { href: string; label: string }[] = [];
    const navItem = Array.from(opfDoc.querySelectorAll('manifest > item')).find(el => el.getAttribute('properties')?.includes('nav'));
    if (navItem) {
      const navPath = opfDir + navItem.getAttribute('href');
      if (files[navPath]) {
        const navDoc = parser.parseFromString(decoder.decode(files[navPath]), 'text/html');
        toc = Array.from(navDoc.querySelectorAll('nav li a')).map(a => ({ href: a.getAttribute('href') || '', label: a.textContent?.trim() || '' }));
      }
    }
    if (toc.length === 0) {
      const ncxItem = Array.from(opfDoc.querySelectorAll('manifest > item')).find(el => el.getAttribute('media-type') === 'application/x-dtbncx+xml' || el.getAttribute('id') === 'ncx');
      if (ncxItem) {
        const ncxPath = opfDir + ncxItem.getAttribute('href');
        if (files[ncxPath]) {
          const ncxDoc = parser.parseFromString(decoder.decode(files[ncxPath]), 'text/xml');
          toc = Array.from(ncxDoc.querySelectorAll('navPoint')).map(np => ({ href: np.querySelector('content')?.getAttribute('src') || '', label: np.querySelector('navLabel text')?.textContent?.trim() || '' }));
        }
      }
    }

    const allSentences: any[] = [], allContent: any[] = [], allOutline: any[] = [];
    let globalLineIdCounter = 0, lastAddedHeaderText: string | null = null;
    const normalizePath = (path: string) => { const parts = path.split('/'), res: string[] = []; for (const p of parts) { if (p === '..') res.pop(); else if (p !== '.') res.push(p); } return res.join('/'); };
    const sentRe = /[^.!?]+[.!?]+/g;
    let _lastYield = performance.now();

    for (let i = 0; i < spineIds.length; i++) {
      const id = spineIds[i];
      const href = manifest[id]; if (!href) continue;
      const fullPath = normalizePath(opfDir + href), cleanPath = decodeURIComponent(fullPath.split('#')[0]), fileBytes = files[cleanPath]; if (!fileBytes) continue;
      const doc = parser.parseFromString(decoder.decode(fileBytes), 'application/xhtml+xml');
      let chapterTitle = toc.length > 0 ? findTitleInToc(toc, href) : null;
      if (!chapterTitle) { const h1 = doc.querySelector('h1'); if (h1) chapterTitle = h1.textContent?.trim() || null; }
      if (chapterTitle) {
        const norm = chapterTitle.replace(/[\u00a0\s]+/g, ' ').trim();
        if (norm && norm !== lastAddedHeaderText) {
          lastAddedHeaderText = norm;
          const headId = `header-${globalLineIdCounter++}`;
          const headItem = { type: 'header', id: headId, text: norm, level: 1 };
          allContent.push(headItem);
          allOutline.push({ id: headId, text: norm, level: 1 });
        }
      }
      const blockEls = Array.from(doc.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, blockquote')), elementsToProcess = blockEls.length > 0 ? blockEls : (doc.body ? [doc.body] : []);
      for (const el of elementsToProcess) {
        const tag = el.tagName.toLowerCase();
        if (/^h[1-6]$/.test(tag)) {
          const headText = el.textContent?.replace(/[\u00a0\s]+/g, ' ').trim() || '';
          if (!headText || headText === lastAddedHeaderText) continue;
          lastAddedHeaderText = headText;
          const headId = `header-${globalLineIdCounter++}`;
          const headItem = { type: 'header', id: headId, text: headText, level: parseInt(tag[1]) };
          allContent.push(headItem);
          allOutline.push({ id: headId, text: headText, level: parseInt(tag[1]) });
          continue;
        }
        const runs = extractRuns(el), rawText = runs.filter(r => !r.br).map(r => r.text).join(''), cleanText = rawText.replace(/\s+/g, ' ').trim(); if (!cleanText) continue;
        const paraId = `para-${globalLineIdCounter}`, paraSentences: any[] = [];
        const startLineId = globalLineIdCounter;
        let sm: RegExpExecArray | null, lastIdx = 0; sentRe.lastIndex = 0;
        while ((sm = sentRe.exec(cleanText)) !== null) {
          const sText = sm[0].trim();
          if (sText && !(chapterTitle && sText === chapterTitle)) { const lineId = globalLineIdCounter++; allSentences.push({ text: sText, lines: [lineId], headerId: allOutline[allOutline.length-1]?.id }); paraSentences.push({ id: lineId, text: sText }); }
          lastIdx = sm.index + sm[0].length;
        }
        const sRem = cleanText.slice(lastIdx).trim();
        if (sRem && !(chapterTitle && sRem === chapterTitle)) { const lineId = globalLineIdCounter++; allSentences.push({ text: sRem, lines: [lineId], headerId: allOutline[allOutline.length-1]?.id }); paraSentences.push({ id: lineId, text: sRem }); }
        if (paraSentences.length > 0) {
          allContent.push({ 
            type: 'paragraph', id: paraId, sentences: paraSentences, elementType: tag, runs, headerId: allOutline[allOutline.length-1]?.id,
            startLineId, endLineId: globalLineIdCounter - 1
          });
        }
      }
      onChunk([...allSentences], [...allContent], [...allOutline], i === spineIds.length - 1);
      if (performance.now() - _lastYield > 30) {
        await new Promise<void>(r => setTimeout(r, 0));
        _lastYield = performance.now();
      }
    }
  } catch (e) { 
    console.error(e);
    alert("Could not load EPUB file"); 
  } finally { 
    setIsDocLoading(false); 
  }
};

export const loadPDF = async (
  data: ArrayBuffer,
  setPdfDoc: (doc: any) => void,
  setPages: (pages: any[]) => void,
  setIsDocLoading: (loading: boolean) => void
) => {
  setIsDocLoading(true);
  try {
    const pdfjsLib = await import('pdfjs-dist');
    pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';
    const doc = await pdfjsLib.getDocument(data).promise;
    const pageResults = await Promise.all(Array.from({ length: doc.numPages }, (_, i) => doc.getPage(i + 1)));
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    const newPages = pageResults.map((page, i) => ({ viewport: page.getViewport({ scale }), lines: [], pageNumber: i + 1 }));
    const globalLineList: any[] = [];
    for (let i = 0; i < doc.numPages; i++) {
      const page = pageResults[i], textContent = await page.getTextContent();
      const lines = processTextContent(textContent, scale, globalLineList.length, pdfjsLib, newPages[i].viewport);
      globalLineList.push(...lines); newPages[i].lines = lines; page.cleanup();
      if (i % 10 === 0) await new Promise(r => setTimeout(r, 0));
    }
    const fullText = globalLineList.map(l => l.text).join(' '), sentenceRegex = /[^.!?]+[.!?]/g;
    let match, sentenceTexts: { text: string; start: number; end: number }[] = [];
    while ((match = sentenceRegex.exec(fullText)) !== null) sentenceTexts.push({ text: match[0].trim(), start: match.index, end: match.index + match[0].length });
    let cumPos = 0; globalLineList.forEach(line => { line.startPos = cumPos; cumPos += line.text.length + 1; });
    const newSentences = sentenceTexts.map(sent => {
      const sentLines = globalLineList.filter(line => line.startPos < sent.end && (line.startPos + line.text.length + 1) > sent.start).map(line => line.id);
      return { text: sent.text, lines: sentLines };
    }).filter(sent => sent.text && sent.lines.length > 0);
    fileTypeSignal.value = 'pdf';
    outlineSignal.value = [];
    sentencesSignal.value = newSentences;
    setPdfDoc(doc); 
    setPages(newPages); 
  } catch (e) { console.error(e); } finally { setIsDocLoading(false); }
};

const processTextContent = (textContent: any, scale: number, startIndex: number, pdfjsLib: any, viewport: any) => {
  // Map items to viewport coordinates
  const rawItems = textContent.items.map((item: any) => {
    // Convert PDF point to Viewport point
    const [x, y] = viewport.convertToViewportPoint(item.transform[4], item.transform[5]);
    // item.width/height are in PDF units, must be multiplied by viewport.scale
    const w = item.width * viewport.scale;
    const h = (item.height || Math.sqrt(item.transform[0]**2 + item.transform[1]**2)) * viewport.scale;
    return { str: item.str, x, y, w, h };
  }).sort((a: any, b: any) => Math.abs(a.y - b.y) > a.h * 0.4 ? a.y - b.y : a.x - b.x);

  const lines: any[] = []; 
  let currentLine: any = null;
  rawItems.forEach((item: any) => {
    if (!currentLine) {
      currentLine = { items: [item], y: item.y, height: item.h };
    } else if (Math.abs(item.y - currentLine.y) < currentLine.height * 0.8) {
      currentLine.items.push(item);
      currentLine.height = Math.max(currentLine.height, item.h);
    } else {
      lines.push(currentLine);
      currentLine = { items: [item], y: item.y, height: item.h };
    }
  });
  if (currentLine) lines.push(currentLine);

  return lines.map((line: any, idx: number) => {
    const minX = Math.min(...line.items.map((i: any) => i.x));
    const last = line.items[line.items.length - 1];
    const maxX = last.x + last.w;
    const width = maxX - minX;
    
    return {
      id: startIndex + idx, 
      text: line.items.map((i: any) => i.str).join(' '),
      words: line.items.map((i: any) => ({
        text: i.str,
        left: `${((i.x - minX) / width) * 100}%`,
        top: `${((i.y - i.h - (line.y - line.height)) / line.height) * 100}%`,
        width: `${(i.w / width) * 100}%`,
        height: `${(i.h / line.height) * 100}%`
      })),
      left: `${(minX / viewport.width) * 100}%`, 
      top: `${((line.y - line.height) / viewport.height) * 100}%`,
      width: `${(width / viewport.width) * 100}%`, 
      height: `${(line.height / viewport.height) * 100}%`,
      startPos: 0
    };
  });
};
