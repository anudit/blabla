import { useState, useEffect, useRef, useMemo, useCallback, lazy, Suspense } from 'preact/compat';
import { useSignalEffect, useComputed } from '@preact/signals';
import type { JSX } from 'preact';
import { Target, Loader2 } from 'lucide-preact';
import {
  isPlayingSignal, playbackStateSignal, ttsStatusSignal,
  isModelReadySignal, currentSentenceIndexSignal, restartSignal,
  sentencesSignal, fileTypeSignal, playbackSpeedSignal,
  selectedVoiceSignal, currentFileIdSignal, currentFileNameSignal,
  outlineSignal,
} from './signals';
import { THEMES, TT } from './theme';
import { calculateWordTimings, extractWords } from './utils';
import {
  loadMarkdown, loadText, loadEPUB, loadPDF, loadMOBI
} from './loaders';
import { getBookmarks, saveBookmark, removeBookmark } from './components/BookmarkHistory';
import type { BookmarkEntry } from './components/BookmarkHistory';

const LandingCard = lazy(() => import('./components/LandingCard'));
const ContentViewer = lazy(() => import('./components/ContentViewer'));
const BottomBar = lazy(() => import('./components/BottomBar'));

export default function App() {
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [pages, setPages] = useState<any[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');
  const [epubContent, setEpubContent] = useState<any[]>([]);
  const [showTextInput, setShowTextInput] = useState(false);
  const [textInputValue, setTextInputValue] = useState('');
  const [urlInputValue, setUrlInputValue] = useState('');
  const [isUrlLoading, setIsUrlLoading] = useState(false);
  const [urlError, setUrlError] = useState('');
  const [fontSize, setFontSize] = useState(() => {
    const s = parseFloat(localStorage.getItem('fontSize') || '');
    return isNaN(s) ? 1.05 : s;
  });
  const [eggPhase, setEggPhase] = useState<'in' | 'out' | null>(null);
  const [bookmarks, setBookmarks] = useState<BookmarkEntry[]>(() => getBookmarks());
  const [isDocLoading, setIsDocLoading] = useState(false);

  const workerRef = useRef<Worker | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const audioCache = useRef(new Map());
  const pendingFetches = useRef(new Set());
  const currentSource = useRef<AudioBufferSourceNode | null>(null);
  const playbackSessionId = useRef(0);
  const usingFallback = useRef(false);
  const nativeTimeout = useRef<any>(null);
  const wordRafRef = useRef<number | null>(null);
  const wordTimingsRef = useRef<{ start: number; end: number; index: number }[]>([]);
  const audioStartRef = useRef(0);
  const audioResolvers = useRef(new Map<number, (buffer: AudioBuffer) => void>());
  const isWaitingForAudio = useRef(false);
  const eggTimerRef = useRef<any>(null);
  const pendingResumeRef = useRef<number>(-1);
  const isResuming = useRef(false);
  const nextStartTimeRef = useRef(0);
  const lastActiveSentenceRef = useRef<{ id: number, text: string } | null>(null);
  const prevIndexRef = useRef<number>(-1);

  const t = isDarkMode ? THEMES.dark : THEMES.light;
  
  const activeHeaderId = useComputed(() => {
    const idx = currentSentenceIndexSignal.value;
    const sents = sentencesSignal.value;
    if (idx < 0 || !sents[idx]) return null;
    return sents[idx].headerId || null;
  });

  const hasSentences = useComputed(() => sentencesSignal.value.length > 0);

  useEffect(() => {
    const ctx = getAudioContext();
    ctx.resume().catch(console.warn);
    workerRef.current = new Worker("/tts.worker.js", { type: 'module' });
    workerRef.current.onmessage = (e) => {
      const { status, audio, error, text, lineIndex, device, dtype } = e.data;
      if (status === 'ready') {
        ttsStatusSignal.value = `Ready (${device}/${dtype})`;
        isModelReadySignal.value = true;
        workerRef.current?.postMessage({ type: 'generate', text: 'Warm up.', lineIndex: -1, voice: 'af_bella', speed: 1.0 });
      } else if (status === 'complete') {
        try {
          const ctx = getAudioContext();
          const buffer = ctx.createBuffer(1, audio.length, 24000);
          buffer.getChannelData(0).set(audio);
          if (text.startsWith("Hello! I am")) {
            playBufferDirectly(buffer);
          } else if (lineIndex !== undefined) {
            const resolver = audioResolvers.current.get(lineIndex);
            if (resolver) { resolver(buffer); audioResolvers.current.delete(lineIndex); }
          }
        } catch (err) { console.error("Audio conversion failed", err); }
      } else if (status === 'error') {
        if (error.includes("import_promises") || error.includes("tokenizer")) triggerFallback();
      }
    };
    ttsStatusSignal.value = "Downloading Model...";
    workerRef.current.postMessage({ type: 'init' });
    return () => {
      stopAllAudio();
      workerRef.current?.terminate();
      if (audioContext.current) audioContext.current.close();
    };
  }, []);

  useSignalEffect(() => {
    if (isPlayingSignal.value && currentSentenceIndexSignal.value >= 0 && !isWaitingForAudio.current) {
      restartSignal.value; playCurrentSentence();
    }
  });

  // MASTER OBSERVER: Centralized highlight + DOM swap + scroll
  useSignalEffect(() => {
    const idx = currentSentenceIndexSignal.value, sents = sentencesSignal.value, fType = fileTypeSignal.value;
    const prevIdx = prevIndexRef.current;
    
    // 1. Cleanup previous highlight (Targeted instead of querySelectorAll)
    if (prevIdx >= 0 && sents[prevIdx]) {
      const prevUnit = sents[prevIdx];
      prevUnit.lines.forEach((id: number) => {
        const el = document.getElementById(`line-${id}`);
        if (el) {
          el.classList.remove('epub-highlight-active', 'pdf-highlight-active');
          // Note: Text restoration is now handled automatically by SentenceItem re-rendering
        }
      });
      // Also cleanup word highlights if any
      document.querySelectorAll('.word-highlight-active').forEach(el => el.classList.remove('word-highlight-active'));
    }

    const unit = sents[idx];
    prevIndexRef.current = idx;
    if (idx < 0 || !unit) return;

    // 3. Apply highlight
    if (fType === 'pdf') {
      unit.lines.forEach((id: number) => {
        document.getElementById(`line-${id}`)?.classList.add('pdf-highlight-active');
      });
    } else {
      unit.lines.forEach((id: number) => {
        const el = document.getElementById(`line-${id}`);
        if (!el) return;
        (el.textContent === '' && el.closest('p') ? el.closest('p')! : el).classList.add('epub-highlight-active');
        
        // 4. EPUB Special: Inject word spans for granular highlighting
        // We only do this for the first line of the active sentence
        if (id === unit.lines[0]) {
          lastActiveSentenceRef.current = { id, text: unit.text };
          const words = extractWords(unit.text);
          el.innerHTML = '';
          words.forEach((w, wi) => {
            const span = document.createElement('span');
            span.id = `word-${id}-${wi}`;
            span.style.transition = 'background-color 0.08s ease';
            span.textContent = w + (wi < words.length - 1 ? ' ' : '');
            el.appendChild(span);
          });
          el.appendChild(document.createTextNode(' '));
        }
      });
    }

    // 5. Smart Scroll
    const target = document.getElementById(`line-${unit.lines[0]}`);
    if (target) {
      const rect = target.getBoundingClientRect();
      const vh = window.innerHeight || document.documentElement.clientHeight;
      const isOffScreen = rect.top < 80 || rect.bottom > vh - 140;
      
      if (isOffScreen) {
        const instant = isResuming.current; if (instant) isResuming.current = false;
        setTimeout(() => target.scrollIntoView({ 
          behavior: instant ? 'instant' : 'smooth', 
          block: fType === 'pdf' ? 'nearest' : 'center'
        }), 0);
      }
    }
  });

  useSignalEffect(() => {
    const idx = currentSentenceIndexSignal.value, sents = sentencesSignal.value, fType = fileTypeSignal.value, cFileId = currentFileIdSignal.value, cFileName = currentFileNameSignal.value;
    if (idx > 0 && sents.length > 0 && cFileId && cFileName && fType) {
      const isUrl = fType === 'text' && cFileId.startsWith('http');
      saveBookmark({ id: cFileId, fileName: cFileName, sentenceIndex: idx, totalSentences: sents.length, timestamp: Date.now(), fileType: isUrl ? 'url' : fType as 'pdf' | 'epub', preview: (sents[idx]?.text || '').slice(0, 80), ...(isUrl ? { url: cFileId } : {}) });
      setBookmarks(getBookmarks());
    }
  });

  useEffect(() => {
    const h = (e: ClipboardEvent) => {
      if (sentencesSignal.peek().length > 0) return;
      const tag = (document.activeElement as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      const t = (e.clipboardData?.getData('text') ?? '').trim();
      if (!t) return; e.preventDefault();
      if (/^https?:\/\/\S+$/.test(t)) handleUrlLoad(t); else loadText(t, (sents, content, outline, isFinal) => {
        sentencesSignal.value = sents;
        setEpubContent(content);
        outlineSignal.value = outline;
        if (!fileTypeSignal.value && (sents.length > 5 || isFinal)) {
          fileTypeSignal.value = 'text';
          setIsDocLoading(false);
        }
      }, setShowTextInput, setTextInputValue);
    };
    window.addEventListener('paste', h); return () => window.removeEventListener('paste', h);
  }, []);

  useSignalEffect(() => {
    const len = sentencesSignal.value.length;
    const pending = pendingResumeRef.current;
    if (pending <= 0 || len === 0) return;
    // For EPUB: wait until the target sentence is loaded or loading is finalised
    if (len <= pending && fileTypeSignal.value === null) return;
    const idx = Math.min(pending, len - 1);
    pendingResumeRef.current = -1;
    isResuming.current = true;
    // Defer so Preact commits the content DOM before the master observer tries getElementById
    setTimeout(() => {
      currentSentenceIndexSignal.value = idx;
      isPlayingSignal.value = true;
    }, 0);
  });

  const getAudioContext = () => {
    if (!audioContext.current || audioContext.current.state === 'closed') audioContext.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    if (audioContext.current.state !== 'running') audioContext.current.resume().catch(console.warn);
    return audioContext.current;
  };

  const playBufferDirectly = (buffer: AudioBuffer) => {
    const ctx = getAudioContext(), source = ctx.createBufferSource();
    source.buffer = buffer; source.connect(ctx.destination); source.start(0);
    playbackStateSignal.value = "Playing"; source.onended = () => { playbackStateSignal.value = "Ready"; };
  };

  const stopAllAudio = () => {
    if (wordRafRef.current) cancelAnimationFrame(wordRafRef.current);
    document.querySelectorAll('.word-highlight-active').forEach(el => el.classList.remove('word-highlight-active'));
    playbackSessionId.current += 1;
    if (currentSource.current) { try { currentSource.current.stop(); } catch(e){} currentSource.current.disconnect(); currentSource.current = null; }
    window.speechSynthesis.cancel(); if (nativeTimeout.current) clearTimeout(nativeTimeout.current);
    if (audioContext.current && audioContext.current.state === 'running') audioContext.current.suspend();
    audioCache.current.clear(); pendingFetches.current.clear();
    audioResolvers.current.forEach(r => r(null as any)); audioResolvers.current.clear();
    playbackStateSignal.value = "Stopped"; isWaitingForAudio.current = false;
  };

  const playCurrentSentence = async () => {
    const isPlaying = isPlayingSignal.peek(), idx = currentSentenceIndexSignal.peek(), sents = sentencesSignal.peek(), sVoice = selectedVoiceSignal.peek(), pSpeed = playbackSpeedSignal.peek();
    if (!isPlaying || idx === -1) return;
    if (idx >= sents.length) { isPlayingSignal.value = false; playbackStateSignal.value = "Completed"; return; }
    const currentSession = playbackSessionId.current, unit = sents[idx], text = unit.text;
    if (text.trim().length <= 1) { advanceSentence(); return; }

    if (usingFallback.current) {
      window.speechSynthesis.cancel(); if (nativeTimeout.current) clearTimeout(nativeTimeout.current);
      nativeTimeout.current = setTimeout(() => {
        const u = new SpeechSynthesisUtterance(text); u.rate = pSpeed;
        u.onend = () => { if (isPlayingSignal.peek() && currentSession === playbackSessionId.current) advanceSentence(); };
        u.onerror = () => { if (isPlayingSignal.peek()) advanceSentence(); };
        window.speechSynthesis.speak(u); playbackStateSignal.value = "Playing";
      }, 50);
      return;
    }

    const ctx = getAudioContext();
    if (currentSource.current) { try { currentSource.current.stop(); } catch(e){} currentSource.current.disconnect(); currentSource.current = null; }
    for (let i = 1; i <= 3; i++) {
      const nIdx = idx + i;
      if (nIdx < sents.length && !audioCache.current.has(nIdx) && !pendingFetches.current.has(nIdx)) {
        pendingFetches.current.add(nIdx);
        new Promise<AudioBuffer | null>(r => {
          audioResolvers.current.set(nIdx, r);
          workerRef.current?.postMessage({ type: 'generate', text: sents[nIdx].text, lineIndex: nIdx, voice: sVoice, speed: pSpeed });
        }).then(b => { if (b && currentSession === playbackSessionId.current) audioCache.current.set(nIdx, b); pendingFetches.current.delete(nIdx); });
      }
    }

    let buffer = audioCache.current.get(idx);
    if (!buffer) {
      playbackStateSignal.value = "Buffering"; isWaitingForAudio.current = true;
      buffer = await new Promise<AudioBuffer | null>(r => {
        audioResolvers.current.set(idx, r);
        workerRef.current?.postMessage({ type: 'generate', text, lineIndex: idx, voice: sVoice, speed: pSpeed });
      });
      isWaitingForAudio.current = false;
    }
    if (currentSession !== playbackSessionId.current || !isPlayingSignal.peek() || !buffer) return;

    const source = ctx.createBufferSource(); source.buffer = buffer; source.connect(ctx.destination);
    source.onended = () => { if (wordRafRef.current) cancelAnimationFrame(wordRafRef.current); document.querySelectorAll('.word-highlight-active').forEach(el => el.classList.remove('word-highlight-active')); currentSource.current = null; if (isPlayingSignal.peek() && currentSession === playbackSessionId.current) advanceSentence(); };
    currentSource.current = source; playbackStateSignal.value = "Playing";
    wordTimingsRef.current = calculateWordTimings(text, buffer.duration);
    let st = ctx.currentTime; if (nextStartTimeRef.current > st && nextStartTimeRef.current < st + 0.5) st = nextStartTimeRef.current;
    audioStartRef.current = st; nextStartTimeRef.current = st + buffer.duration;
    source.start(st);
    if (fileTypeSignal.peek() !== 'pdf') {
      const animate = () => {
        if (currentSource.current !== source) return;
        const elap = ctx.currentTime - audioStartRef.current;
        const active = wordTimingsRef.current.find(t => elap >= t.start && elap < t.end);
        document.querySelectorAll('.word-highlight-active').forEach(el => el.classList.remove('word-highlight-active'));
        if (active) { const el = document.getElementById(`word-${unit.lines[0]}-${active.index}`); if (el) el.classList.add('word-highlight-active'); }
        if (elap < buffer!.duration) wordRafRef.current = requestAnimationFrame(animate);
      };
      wordRafRef.current = requestAnimationFrame(animate);
    }
  };

  const advanceSentence = () => {
    const next = currentSentenceIndexSignal.peek() + 1;
    for (const key of Array.from(audioCache.current.keys())) { if (key < next - 1 || key > next + 3) audioCache.current.delete(key); }
    currentSentenceIndexSignal.value = next;
  };

  const handleLineClick = useCallback((lineId: number) => {
    isPlayingSignal.value = false; stopAllAudio();
    const idx = sentencesSignal.peek().findIndex(s => s.lines.includes(lineId));
    if (idx !== -1) {
      currentSentenceIndexSignal.value = idx; isPlayingSignal.value = true;
      document.getElementById(`line-${lineId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, []);

  const togglePlay = useCallback(() => {
    if (!isModelReadySignal.peek()) return;
    if (currentSentenceIndexSignal.peek() === -1 && sentencesSignal.peek().length > 0) { currentSentenceIndexSignal.value = 0; isPlayingSignal.value = true; }
    else if (isPlayingSignal.peek()) { isPlayingSignal.value = false; stopAllAudio(); }
    else { isPlayingSignal.value = true; playbackStateSignal.value = "Starting"; }
  }, []);

  const handleTestAudio = useCallback(() => {
    if (!isModelReadySignal.peek()) return; stopAllAudio();
    const text = "Hello! I am ready to read.", pSpeed = playbackSpeedSignal.peek(), sVoice = selectedVoiceSignal.peek();
    if (usingFallback.current) { const u = new SpeechSynthesisUtterance(text); u.rate = pSpeed; window.speechSynthesis.speak(u); playbackStateSignal.value = "Testing"; }
    else { playbackStateSignal.value = "Generating"; workerRef.current?.postMessage({ type: 'generate', text, voice: sVoice, speed: pSpeed }); }
  }, []);

  const resetReader = useCallback(() => {
    isPlayingSignal.value = false; stopAllAudio(); if (pdfDoc) try { pdfDoc.destroy(); } catch(e) {}
    setPdfDoc(null); setPages([]); sentencesSignal.value = []; fileTypeSignal.value = null; setEpubContent([]); currentSentenceIndexSignal.value = -1;
    playbackStateSignal.value = "Idle"; setShowTextInput(false); setTextInputValue(''); currentFileIdSignal.value = null; currentFileNameSignal.value = null; pendingResumeRef.current = -1;
    outlineSignal.value = [];
  }, [pdfDoc]);

  const triggerFallback = useCallback(() => { if (!usingFallback.current) { usingFallback.current = true; ttsStatusSignal.value = "Fallback (Native)"; isWaitingForAudio.current = false; audioResolvers.current.forEach(r => r(null as any)); audioResolvers.current.clear(); } }, []);

  const handleFileDrop = async (e: any) => {
    if (e.preventDefault) e.preventDefault(); setIsDragOver(false);
    const file = e.dataTransfer?.files[0] || e.target?.files?.[0];
    if (file) {
      const fid = `${file.name}:${file.size}`; currentFileIdSignal.value = fid; currentFileNameSignal.value = file.name;
      const exist = getBookmarks().find(b => b.id === fid); if (exist) pendingResumeRef.current = exist.sentenceIndex;
      const r = new FileReader();
      if (file.type === 'application/pdf') { 
        r.onload = (ev) => loadPDF(ev.target?.result as ArrayBuffer, (sents, pdfPages, pdfDocObj, isFinal) => {
          sentencesSignal.value = sents;
          setPages(pdfPages);
          setPdfDoc(pdfDocObj);
          const resumeIdx = pendingResumeRef.current;
          const hasReachedResume = resumeIdx === -1 || sents.length > resumeIdx;
          if (!fileTypeSignal.value && (hasReachedResume || isFinal)) {
            fileTypeSignal.value = 'pdf';
            setIsDocLoading(false);
          }
        }, setIsDocLoading); 
        r.readAsArrayBuffer(file); 
      }
      else if (file.type === 'application/epub+zip' || file.name.endsWith('.epub')) { 
        r.onload = (ev) => loadEPUB(ev.target?.result as ArrayBuffer, (sents, content, outline, isFinal) => {
          sentencesSignal.value = sents;
          setEpubContent(content);
          outlineSignal.value = outline;
          
          // CRITICAL: First chunk received, or we've reached the pending resume index.
          // Activate UI immediately and keep parsing in background.
          const resumeIdx = pendingResumeRef.current;
          const hasReachedResume = resumeIdx === -1 || sents.length > resumeIdx;
          
          if (!fileTypeSignal.value && (hasReachedResume || isFinal)) {
            fileTypeSignal.value = 'epub';
            setIsDocLoading(false);
          }
        }, setIsDocLoading); 
        r.readAsArrayBuffer(file); 
      }
      else if (file.name.endsWith('.mobi') || file.name.endsWith('.azw') || file.name.endsWith('.azw3')) {
        r.onload = (ev) => loadMOBI(ev.target?.result as ArrayBuffer, (sents, content, outline, isFinal) => {
          sentencesSignal.value = sents;
          setEpubContent(content);
          outlineSignal.value = outline;
          
          const resumeIdx = pendingResumeRef.current;
          const hasReachedResume = resumeIdx === -1 || sents.length > resumeIdx;
          
          if (!fileTypeSignal.value && (hasReachedResume || isFinal)) {
            fileTypeSignal.value = 'epub';
            setIsDocLoading(false);
          }
        }, setIsDocLoading);
        r.readAsArrayBuffer(file);
      }
      else if (file.name.endsWith('.md') || file.name.endsWith('.markdown')) { 
        r.onload = (ev) => loadMarkdown(ev.target?.result as string, (sents, content, outline, isFinal) => {
          sentencesSignal.value = sents;
          setEpubContent(content);
          outlineSignal.value = outline;
          const resumeIdx = pendingResumeRef.current;
          const hasReachedResume = resumeIdx === -1 || sents.length > resumeIdx;
          if (!fileTypeSignal.value && (hasReachedResume || isFinal)) {
            fileTypeSignal.value = 'text';
            setIsDocLoading(false);
          }
        }, setShowTextInput, setTextInputValue); 
        r.readAsText(file); 
      }
      else { 
        r.onload = (ev) => loadText(ev.target?.result as string, (sents, content, outline, isFinal) => {
          sentencesSignal.value = sents;
          setEpubContent(content);
          outlineSignal.value = outline;
          const resumeIdx = pendingResumeRef.current;
          const hasReachedResume = resumeIdx === -1 || sents.length > resumeIdx;
          if (!fileTypeSignal.value && (hasReachedResume || isFinal)) {
            fileTypeSignal.value = 'text';
            setIsDocLoading(false);
          }
        }, setShowTextInput, setTextInputValue); 
        r.readAsText(file); 
      }
    }
  };

  const handleUrlLoad = async (u?: string) => {
    const inputUrl = (u || urlInputValue).trim(); if (!inputUrl) return;
    try { new URL(inputUrl); } catch { setUrlError('Invalid URL'); return; }
    
    let targetUrl = inputUrl;
    const gdocMatch = inputUrl.match(/docs\.google\.com\/document\/d\/([a-zA-Z0-9_-]+)/);
    if (gdocMatch) {
      targetUrl = `https://docs.google.com/document/d/${gdocMatch[1]}/export?format=md`;
    }

    setIsUrlLoading(true); setUrlError('');
    try {
      const fetchUrl = gdocMatch 
        ? `https://test.cors.workers.dev/?${targetUrl}`
        : `https://test.cors.workers.dev/?https://markdown.new/${targetUrl}?method=auto&retain_images`;
      
      const res = await fetch(fetchUrl);
      if (!res.ok) {
        if (res.status === 404 || res.status === 403 || res.status === 401) {
          throw new Error('Document inaccessible');
        }
        throw new Error('Failed to load document');
      }

      const md = await res.text(); 
      // Google Docs export might return a HTML login page if private, 
      // even if the status is 200 (depending on the proxy/redirects).
      if (!md.trim() || (gdocMatch && md.includes('<!DOCTYPE html>'))) {
        throw new Error('Document inaccessible');
      }
      
      const titleMatch = md.match(/^Title:\s*(.+)/m);
      const title = titleMatch 
        ? titleMatch[1].trim() 
        : (gdocMatch ? md.match(/^#\s+(.+)/m)?.[1].trim() || "Google Doc" : new URL(inputUrl).hostname);
      
      currentFileIdSignal.value = inputUrl; 
      currentFileNameSignal.value = title;
      const exist = getBookmarks().find(b => b.id === inputUrl); if (exist) pendingResumeRef.current = exist.sentenceIndex;
      setUrlInputValue(''); 
      loadMarkdown(md, (sents, content, outline, isFinal) => {
        sentencesSignal.value = sents;
        setEpubContent(content);
        outlineSignal.value = outline;
        const resumeIdx = pendingResumeRef.current;
        const hasReachedResume = resumeIdx === -1 || sents.length > resumeIdx;
        if (!fileTypeSignal.value && (hasReachedResume || isFinal)) {
          fileTypeSignal.value = 'text';
          setIsDocLoading(false);
        }
      }, setShowTextInput, setTextInputValue);
    } catch (err: any) { 
      if (err.message === 'Document inaccessible') {
        alert('Document inaccessible. Please ensure the document is shared with "Anyone with the link".');
      }
      setUrlError(err.message || 'Load failed'); 
    } finally { setIsUrlLoading(false); }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', backgroundColor: t.bg, fontFamily: 'system-ui, sans-serif', color: t.text, paddingTop: '1rem', transition: TT }}>
      {isDocLoading ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', gap: '0.85rem', color: t.textMuted }}>
          <Loader2 size={28} color={t.textMuted} style={{ animation: 'spin 0.8s linear infinite' }} />
          <span>{currentFileNameSignal.value ? `Loading ${currentFileNameSignal.value}…` : 'Loading…'}</span>
        </div>
      ) : !hasSentences.value ? (
        <Suspense fallback={null}><LandingCard isDarkMode={isDarkMode} t={t} isDragOver={isDragOver} setIsDragOver={setIsDragOver} onFileDrop={handleFileDrop} urlInputValue={urlInputValue} setUrlInputValue={setUrlInputValue} urlError={urlError} setUrlError={setUrlError} isUrlLoading={isUrlLoading} onUrlLoad={handleUrlLoad} onClipboardPaste={async (e: any) => { e.stopPropagation(); try { const t = await navigator.clipboard.readText(); if (/^https?:\/\/\S+$/.test(t)) handleUrlLoad(t); else if (t) loadText(t, (sents, content, outline, isFinal) => { sentencesSignal.value = sents; setEpubContent(content); outlineSignal.value = outline; if (!fileTypeSignal.value && (sents.length > 5 || isFinal)) { fileTypeSignal.value = 'text'; setIsDocLoading(false); } }, setShowTextInput, setTextInputValue); else setShowTextInput(true); } catch { setShowTextInput(true); } }} showTextInput={showTextInput} setShowTextInput={setShowTextInput} textInputValue={textInputValue} setTextInputValue={setTextInputValue} onLoadText={(t: string) => loadText(t, (sents, content, outline, isFinal) => { sentencesSignal.value = sents; setEpubContent(content); outlineSignal.value = outline; if (!fileTypeSignal.value && (sents.length > 5 || isFinal)) { fileTypeSignal.value = 'text'; setIsDocLoading(false); } }, setShowTextInput, setTextInputValue)} bookmarks={bookmarks} onSelectBookmark={(e: any) => e.fileType === 'url' && e.url && handleUrlLoad(e.url)} onDeleteBookmark={(id: string) => { removeBookmark(id); setBookmarks(getBookmarks()); }} /></Suspense>
      ) : (
        <Suspense fallback={null}><ContentViewer fileType={fileTypeSignal.peek()!} pages={pages} pdfDoc={pdfDoc} epubContent={epubContent} activeHeaderId={activeHeaderId} isDarkMode={isDarkMode} t={t} fontSize={fontSize} onLineClick={handleLineClick} /></Suspense>
      )}
      <style>{`
        .epub-highlight-active { background: rgba(250,204,21,0.45); border-radius:3px; } 
        .word-highlight-active { background: #b47a32 !important; color: white !important; border-radius:3px; z-index: 30; } 
        .pdf-highlight-active { background-color: rgba(250, 204, 21, 0.45) !important; box-shadow: 0 0 0 1px rgba(210, 170, 0, 0.4) !important; z-index:20 !important; } 
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
      {eggPhase && <div style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}><div style={{ width: '300px', height: '300px', borderRadius: '50%', backgroundColor: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', animation: eggPhase === 'in' ? 'eggBounceIn 0.55s forwards' : 'eggFadeOut 0.6s forwards' }}><img src="./180.png" style={{ width: '250px', height: '250px' }} /></div></div>}
      <ScrollToCurrentButton />
      <Suspense fallback={null}><BottomBar t={t} isDarkMode={isDarkMode} onToggleTheme={() => { const n = !isDarkMode; setIsDarkMode(n); localStorage.setItem('theme', n ? 'dark' : 'light'); }} playbackSpeed={playbackSpeedSignal.value} hasSentences={hasSentences.value} onTogglePlay={togglePlay} onSpeedChange={(s: number) => { playbackSpeedSignal.value = s; stopAllAudio(); if (isPlayingSignal.peek()) restartSignal.value++; }} usingFallback={usingFallback.current} selectedVoice={selectedVoiceSignal.value} onVoiceChange={(e: any) => { selectedVoiceSignal.value = e.target.value; stopAllAudio(); if (isPlayingSignal.peek()) restartSignal.value++; }} sentencesLength={sentencesSignal.value.length} fontSize={fontSize} onFontSizeChange={(d: number) => { const n = parseFloat(Math.max(0.8, Math.min(1.6, fontSize + d)).toFixed(2)); setFontSize(n); localStorage.setItem('fontSize', String(n)); }} onTestAudio={handleTestAudio} onReset={resetReader} onLogoClick={() => { if (!eggPhase) { setEggPhase('in'); setTimeout(() => { setEggPhase('out'); setTimeout(() => setEggPhase(null), 600); }, 2400); } }} /></Suspense>
    </div>
  );
}

function ScrollToCurrentButton() {
  const isPlaying = isPlayingSignal.value, idx = currentSentenceIndexSignal.value, sents = sentencesSignal.value;
  if (sents.length === 0 || isPlaying || idx < 0) return null;
  return <button onClick={() => { const u = sents[idx]; if (u) document.getElementById(`line-${u.lines[0]}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); }} style={{ position: 'fixed', bottom: '1rem', right: '1rem', zIndex: 200, display: 'flex', alignItems: 'center', padding: '0.4rem 0.7rem', backgroundColor: '#2a2015', color: '#c0b4a4', border: '1px solid #1a1510', borderRadius: '999px', cursor: 'pointer', fontSize: '0.72rem', boxShadow: '0 2px 10px rgba(0,0,0,0.25)' }}><Target size={13} /></button>;
}
