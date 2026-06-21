import { useState, useEffect, useRef, useMemo, useCallback } from 'preact/hooks';
import { lazy, Suspense } from 'preact/compat';
import { useSignalEffect, useComputed } from '@preact/signals';
import type { JSX } from 'preact';
import Icon from './components/icons';
import {
  isPlayingSignal, playbackStateSignal, ttsStatusSignal,
  isModelReadySignal, currentSentenceIndexSignal, restartSignal,
  sentencesSignal, fileTypeSignal, playbackSpeedSignal,
  selectedVoiceSignal, currentFileIdSignal, currentFileNameSignal,
  outlineSignal, offlineReadySignal,
} from './signals';
import { THEMES, TT } from './theme';
import type { ThemeName } from './theme';
import { calculateWordTimings, extractWords, segmentTextLines, parseOcrPageText, loadWithCache } from './utils';
import {
  loadMarkdown, loadText, loadEPUB, loadPDF, loadMOBI, loadDOCX
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
  const [themeName, setThemeName] = useState<ThemeName>(() => {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') return 'quiet';
    if (saved === 'light') return 'original';
    if (saved && saved in THEMES) return saved as ThemeName;
    return 'original';
  });
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

  const [isOcrMode, setIsOcrMode] = useState(false);
  const [ocrCurrentPage, setOcrCurrentPage] = useState(1);
  const [ocrLoadingStage, setOcrLoadingStage] = useState<string | null>(null);
  const [ocrProgress, setOcrProgress] = useState<number | null>(null);
  const [ocrPageLoading, setOcrPageLoading] = useState(false);
  const ocrPagesCacheRef = useRef<Map<number, { sentences: any[], content: any[] }>>(new Map());
  const ocrEngineRef = useRef<any>(null);
  const ocrDetEngineRef = useRef<any>(null);
  const ocrRequestIdRef = useRef(0);
  const ocrInitializingRef = useRef<Promise<any> | null>(null);

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
  const eggTimersRef = useRef<number[]>([]);
  const pendingResumeRef = useRef<number>(-1);
  const isResuming = useRef(false);
  const nextStartTimeRef = useRef(0);
  const lastActiveSentenceRef = useRef<{ id: number, text: string } | null>(null);
  const lastActiveWordRef = useRef<HTMLElement | null>(null);
  const prevIndexRef = useRef<number>(-1);

  const t = THEMES[themeName];
  const isDarkMode = t.isDark;

  const activeHeaderId = useComputed(() => {
    const idx = currentSentenceIndexSignal.value;
    const sents = sentencesSignal.value;
    if (idx < 0 || !sents[idx]) return null;
    return sents[idx].headerId || null;
  });

  const hasSentences = useComputed(() => sentencesSignal.value.length > 0);

  const modelFetchFailedRef = useRef(false);

  useEffect(() => {
    const ctx = getAudioContext();
    ctx.resume().catch(console.warn);
    workerRef.current = new Worker("/tts.worker.js", { type: 'module' });
    workerRef.current.onmessage = (e) => {
      const { status, audio, error, fetchFailed, text, lineIndex, device, dtype } = e.data;
      if (status === 'ready') {
        ttsStatusSignal.value = `Ready (${device}/${dtype})`;
        isModelReadySignal.value = true;
        modelFetchFailedRef.current = false;
        workerRef.current?.postMessage({ type: 'generate', text: 'Warm up.', lineIndex: -1, voice: 'Bella', speed: 1.0 });
      } else if (status === 'models_cached') {
        offlineReadySignal.value = true;
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
        if (error.includes("import_promises") || error.includes("tokenizer")) {
          triggerFallback();
        } else if (fetchFailed) {
          modelFetchFailedRef.current = true;
          ttsStatusSignal.value = 'Offline - connect to download model';
        }
      }
    };
    ttsStatusSignal.value = "Downloading Model...";
    workerRef.current.postMessage({ type: 'init' });

    // Retry model download automatically when the connection is restored
    const onOnline = () => {
      if (modelFetchFailedRef.current && workerRef.current) {
        modelFetchFailedRef.current = false;
        ttsStatusSignal.value = 'Downloading Model...';
        workerRef.current.postMessage({ type: 'init' });
      }
    };
    window.addEventListener('online', onOnline);

    return () => {
      stopAllAudio();
      workerRef.current?.terminate();
      if (audioContext.current) audioContext.current.close();
      window.removeEventListener('online', onOnline);
      eggTimersRef.current.forEach(id => clearTimeout(id));
      eggTimersRef.current = [];
      if (bookmarkDebounceRef.current) { clearTimeout(bookmarkDebounceRef.current); bookmarkDebounceRef.current = null; }
    };
  }, []);

  useSignalEffect(() => {
    if (isPlayingSignal.value && currentSentenceIndexSignal.value >= 0 && !isWaitingForAudio.current) {
      restartSignal.value; playCurrentSentence();
    }
  });

  // Effect A (highlight + scroll): subscribes ONLY to currentSentenceIndexSignal.
  // Peek() on the other signals so loader chunk-emissions don't re-fire this.
  useSignalEffect(() => {
    const idx = currentSentenceIndexSignal.value;
    const sents = sentencesSignal.peek();
    const fType = fileTypeSignal.peek();
    const prevIdx = prevIndexRef.current;

    if (prevIdx >= 0 && sents[prevIdx]) {
      const prevUnit = sents[prevIdx];
      prevUnit.lines.forEach((id: number) => {
        const el = document.getElementById(`line-${id}`);
        if (el) el.classList.remove('epub-highlight-active', 'pdf-highlight-active');
      });
      clearWordHighlight();
      restoreWordSpans();
    }

    const unit = sents[idx];
    prevIndexRef.current = idx;
    if (idx < 0 || !unit) return;

    if (fType === 'pdf') {
      unit.lines.forEach((id: number) => {
        document.getElementById(`line-${id}`)?.classList.add('pdf-highlight-active');
      });
    } else {
      unit.lines.forEach((id: number) => {
        const el = document.getElementById(`line-${id}`);
        if (!el) return;
        (el.textContent === '' && el.closest('p') ? el.closest('p')! : el).classList.add('epub-highlight-active');
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

    let target = document.getElementById(`line-${unit.lines[0]}`);
    if (target) {
      const rect = target.getBoundingClientRect();
      const vh = window.innerHeight || document.documentElement.clientHeight;
      const isOffScreen = rect.top < 80 || rect.bottom > vh - 140;
      if (isOffScreen) {
        const instant = isResuming.current; if (instant) isResuming.current = false;
        if (instant && rect.height === 0) {
          let parent = target.parentElement;
          while (parent) {
            const parentRect = parent.getBoundingClientRect();
            if (parentRect.height > 0) { target = parent; break; }
            parent = parent.parentElement;
          }
        }
        setTimeout(() => target!.scrollIntoView({ behavior: instant ? 'instant' : 'smooth', block: fType === 'pdf' ? 'nearest' : 'center' }), 0);
      }
    }
  });

  // Effect B (cleanup-on-swap): runs only when sentencesSignal replaces (file load).
  useSignalEffect(() => {
    sentencesSignal.value;
    const prevIdx = prevIndexRef.current;
    if (prevIdx < 0) return;
    clearWordHighlight();
    restoreWordSpans();
  });

  const bookmarkDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useSignalEffect(() => {
    const idx = currentSentenceIndexSignal.value, sents = sentencesSignal.value, fType = fileTypeSignal.value, cFileId = currentFileIdSignal.value, cFileName = currentFileNameSignal.value;
    if (idx > 0 && sents.length > 0 && cFileId && cFileName && fType) {
      if (bookmarkDebounceRef.current) clearTimeout(bookmarkDebounceRef.current);
      bookmarkDebounceRef.current = setTimeout(() => {
        const isUrl = fType === 'text' && cFileId.startsWith('http');
        const bFileType = fType === 'ocr' ? 'ocr' : (isUrl ? 'url' : fType);
        saveBookmark({
          id: cFileId,
          fileName: cFileName,
          sentenceIndex: idx,
          totalSentences: sents.length,
          timestamp: Date.now(),
          fileType: bFileType as any,
          preview: fType === 'ocr' ? `[Page ${ocrCurrentPage}] ${(sents[idx]?.text || '').slice(0, 70)}` : (sents[idx]?.text || '').slice(0, 80),
          ...(isUrl ? { url: cFileId } : {}),
          ...(fType === 'ocr' ? { ocrPage: ocrCurrentPage } : {})
        });
        setBookmarks(getBookmarks());
      }, 2000);
    }
    return () => { if (bookmarkDebounceRef.current) { clearTimeout(bookmarkDebounceRef.current); bookmarkDebounceRef.current = null; } };
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

  const clearWordHighlight = () => {
    if (lastActiveWordRef.current) { lastActiveWordRef.current.classList.remove('word-highlight-active'); lastActiveWordRef.current = null; }
  };

  const restoreWordSpans = () => {
    const last = lastActiveSentenceRef.current;
    if (!last) return;
    const el = document.getElementById(`line-${last.id}`);
    if (el && el.querySelector(`[id^="word-${last.id}-"]`)) { el.textContent = last.text + ' '; }
  };

  const stopAllAudio = () => {
    if (wordRafRef.current) cancelAnimationFrame(wordRafRef.current);
    clearWordHighlight();
    restoreWordSpans();
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
    if (idx >= sents.length) {
      if (fileTypeSignal.peek() === 'ocr' && ocrCurrentPage < (pdfDoc?.numPages || 1)) {
        stopAllAudio();
        const nextPage = ocrCurrentPage + 1;
        setOcrCurrentPage(nextPage);
        return;
      }
      isPlayingSignal.value = false; playbackStateSignal.value = "Completed"; return;
    }
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
    source.onended = () => { if (wordRafRef.current) cancelAnimationFrame(wordRafRef.current); clearWordHighlight(); restoreWordSpans(); currentSource.current = null; if (isPlayingSignal.peek() && currentSession === playbackSessionId.current) advanceSentence(); };
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
        clearWordHighlight();
        if (active) { const el = document.getElementById(`word-${unit.lines[0]}-${active.index}`); if (el) { el.classList.add('word-highlight-active'); lastActiveWordRef.current = el; } }
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
    setIsOcrMode(false);
    setOcrCurrentPage(1);
    ocrPagesCacheRef.current.clear();
    setOcrLoadingStage(null);
    setOcrProgress(null);
    setOcrPageLoading(false);
    ocrRequestIdRef.current++;
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
        r.onload = (ev) => loadPDF(ev.target?.result as ArrayBuffer, async (sents, pdfPages, pdfDocObj, isFinal) => {
          if (isFinal && sents.length === 0) {
            setPdfDoc(pdfDocObj);
            setPages(pdfPages);
            setIsOcrMode(true);
            setIsDocLoading(false);
            const targetPage = exist && exist.fileType === 'ocr' && exist.ocrPage ? exist.ocrPage : 1;
            setOcrCurrentPage(targetPage);
            return;
          }

          sentencesSignal.value = sents;
          setPages(pdfPages);
          setPdfDoc(pdfDocObj);
          const resumeIdx = pendingResumeRef.current;
          const hasReachedResume = resumeIdx === -1 || sents.length > resumeIdx;
          if (!fileTypeSignal.value && sents.length > 0 && (hasReachedResume || isFinal)) {
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
      else if (file.name.endsWith('.docx') || file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        r.onload = (ev) => loadDOCX(ev.target?.result as ArrayBuffer, (sents, content, outline, isFinal) => {
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

  const changeOcrPage = useCallback((p: number) => {
    stopAllAudio();
    setOcrCurrentPage(p);
  }, []);

  const startOcrFlow = async (docObj: any, pdfPages: any[], targetPage: number) => {
    if (!docObj) return;

    const reqId = ++ocrRequestIdRef.current;

    if (ocrPagesCacheRef.current.has(targetPage)) {
      const cached = ocrPagesCacheRef.current.get(targetPage)!;
      sentencesSignal.value = cached.sentences;
      setEpubContent(cached.content);
      fileTypeSignal.value = 'ocr';
      const resumeIdx = pendingResumeRef.current;
      currentSentenceIndexSignal.value = resumeIdx !== -1 ? resumeIdx : 0;
      pendingResumeRef.current = -1;
      return;
    }

    // Clear previous page text immediately when navigating to a non-cached page
    sentencesSignal.value = [];
    setEpubContent([]);

    const isEngineReady = !!ocrEngineRef.current && !!ocrDetEngineRef.current;
    if (!isEngineReady) {
      setOcrLoadingStage('Initializing OCR engine...');
    } else {
      setOcrPageLoading(true);
    }
    setOcrProgress(null);

    try {
      if (!ocrEngineRef.current || !ocrDetEngineRef.current) {
        if (!ocrInitializingRef.current) {
          ocrInitializingRef.current = (async () => {
            setOcrLoadingStage('Downloading OCR recognition model (20MB)...');

            const REC_BASE = 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main';
            const DET_BASE = 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main';
            const modelUrl = `${REC_BASE}/inference.onnx`;
            const vocabUrl = `${REC_BASE}/inference.yml`;
            const detModelUrl = `${DET_BASE}/inference.onnx`;

            const cachedModelBlobUrl = await loadWithCache(modelUrl, (pct) => {
              if (reqId === ocrRequestIdRef.current) {
                setOcrProgress(pct * 0.25);
              }
            });

            if (reqId === ocrRequestIdRef.current) {
              setOcrLoadingStage('Downloading OCR detection model...');
            }
            const cachedDetModelBlobUrl = await loadWithCache(detModelUrl, (pct) => {
              if (reqId === ocrRequestIdRef.current) {
                setOcrProgress(0.25 + pct * 0.70);
              }
            });

            if (reqId === ocrRequestIdRef.current) {
              setOcrLoadingStage('Downloading OCR character dictionary...');
            }
            const cachedVocabBlobUrl = await loadWithCache(vocabUrl, () => {});

            if (reqId === ocrRequestIdRef.current) {
              setOcrLoadingStage('Initializing WebGPU pipeline and compiling WGSL shaders...');
              setOcrProgress(0.95);
            }

            const { PaddleOcrEngine, TextDetectionEngine } = await import('paddleocr-webgpu');
            const engine = new PaddleOcrEngine();
            await engine.init({
              modelUrl: cachedModelBlobUrl,
              vocabUrl: cachedVocabBlobUrl,
              channelOrder: 'bgr',
              maxWidth: 2048,
            });
            ocrEngineRef.current = engine;

            const detEngine = new TextDetectionEngine();
            await detEngine.init({
              modelUrl: cachedDetModelBlobUrl,
            });
            ocrDetEngineRef.current = detEngine;
          })().catch(err => {
            ocrInitializingRef.current = null;
            throw err;
          });
        }
        await ocrInitializingRef.current;
      }

      if (reqId !== ocrRequestIdRef.current) return;

      // Ensure that we clear global overlay (since engine is ready now), and show local loading
      setOcrLoadingStage(null);
      setOcrProgress(null);
      setOcrPageLoading(true);

      const page = await docObj.getPage(targetPage);
      const canvas = document.createElement('canvas');
      const viewport = page.getViewport({ scale: 2.7778 });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext('2d')!;
      await page.render({ canvasContext: ctx, viewport }).promise;
      page.cleanup();

      let lines = [];
      try {
        console.log(`[OCR] Running WebGPU detection model on page ${targetPage}...`);
        const detResult = await ocrDetEngineRef.current.detect(canvas);
        // v1.3 TextDetectionEngine returns { boxes: [{ points, score }], shape, inferenceTimeMs }.
        // Convert each polygon to an axis-aligned line box for cropping/recognition.
        lines = detResult.boxes.map((box) => {
          const xs = box.points.map((p) => p[0]);
          const ys = box.points.map((p) => p[1]);
          return {
            x0: Math.max(0, Math.floor(Math.min(...xs))),
            x1: Math.min(canvas.width, Math.ceil(Math.max(...xs))),
            y0: Math.max(0, Math.floor(Math.min(...ys))),
            y1: Math.min(canvas.height, Math.ceil(Math.max(...ys))),
          };
        });
        console.log(`[OCR] Detection found ${lines.length} boxes in ${detResult.inferenceTimeMs.toFixed(1)}ms`);
      } catch (detErr) {
        console.warn('WebGPU text detection failed, falling back to manual segmentTextLines:', detErr);
        lines = segmentTextLines(canvas);
      }

      let pageText = '';

      if (lines.length === 0) {
        const res = await ocrEngineRef.current.recognize(canvas);
        pageText = res.text || '';
      } else {
        // Crop each detected region exactly as the demo does: no padding, no
        // ruled-line erasing (which destroys printed glyphs), source clamped to
        // canvas bounds so drawImage never samples outside the page.
        const lineCanvases = lines.map(({ y0, y1, x0, x1 }) => {
          const lc = document.createElement('canvas');
          const clW = Math.max(1, x1 - x0);
          const chH = Math.max(1, y1 - y0);
          lc.width = clW;
          lc.height = chH;
          const lctx = lc.getContext('2d')!;
          lctx.fillStyle = '#FFFFFF';
          lctx.fillRect(0, 0, clW, chH);
          const srcW = Math.min(clW, canvas.width - x0);
          const srcH = Math.min(chH, canvas.height - y0);
          if (srcW > 0 && srcH > 0) {
            lctx.drawImage(canvas, x0, y0, srcW, srcH, 0, 0, srcW, srcH);
          }
          return lc;
        });

        if (reqId !== ocrRequestIdRef.current) return;

        const results = await ocrEngineRef.current.recognizePipelined(lineCanvases);
        const mappedResults = results.map((r, idx) => {
          const line = lines[idx];
          return {
            text: r.text.trim(),
            y0: line ? line.y0 : 0
          };
        }).filter(item => item.text);
        mappedResults.sort((a, b) => a.y0 - b.y0);
        pageText = mappedResults.map(item => item.text).join('\n');
      }

      console.log(`[OCR] Target Page: ${targetPage}, lines recognized: ${lines.length}, raw text length: ${pageText.length}`);
      console.log(`[OCR] Text Preview:`, pageText.slice(0, 300));

      if (reqId !== ocrRequestIdRef.current) return;

      const { sentences, content } = parseOcrPageText(pageText);

      ocrPagesCacheRef.current.set(targetPage, { sentences, content });

      sentencesSignal.value = sentences;
      setEpubContent(content);
      fileTypeSignal.value = 'ocr';

      const resumeIdx = pendingResumeRef.current;
      currentSentenceIndexSignal.value = resumeIdx !== -1 ? resumeIdx : 0;
      pendingResumeRef.current = -1;

    } catch (err: any) {
      if (reqId === ocrRequestIdRef.current) {
        console.error('OCR Flow failed:', err);
        alert(`OCR failed: ${err.message || err}`);
        resetReader();
      }
    } finally {
      if (reqId === ocrRequestIdRef.current) {
        setOcrLoadingStage(null);
        setOcrProgress(null);
        setOcrPageLoading(false);
      }
    }
  };

  useEffect(() => {
    if (isOcrMode && pdfDoc) {
      startOcrFlow(pdfDoc, pages, ocrCurrentPage);
    }
  }, [ocrCurrentPage, pdfDoc, isOcrMode, pages]);

  const handleUrlLoad = async (u?: string) => {
    const inputUrl = (u || urlInputValue).trim(); if (!inputUrl) return;
    try { new URL(inputUrl); } catch { setUrlError('Invalid URL'); return; }

    let targetUrl = inputUrl;
    const gdocMatch = inputUrl.match(/docs\.google\.com\/document\/d\/([a-zA-Z0-9_-]+)/);
    if (gdocMatch) {
      targetUrl = `https://docs.google.com/document/d/${gdocMatch[1]}/export?format=md`;
    }

    setIsUrlLoading(true); setIsDocLoading(true); setUrlError('');
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s timeout

    try {
      const fetchUrl = gdocMatch
        ? targetUrl
        : `https://urltomd.anudit.workers.dev/${targetUrl}`;

      const res = await fetch(fetchUrl, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!res.ok) {
        if (res.status === 404 || res.status === 403 || res.status === 401) {
          throw new Error('Document inaccessible');
        }
        throw new Error('Failed to load document');
      }

      const md = await res.text();
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

      // Use requestIdleCallback or setTimeout to ensure parsing doesn't block the next frame
      setTimeout(() => {
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
      }, 10);

    } catch (err: any) {
      clearTimeout(timeoutId);
      setIsDocLoading(false);
      if (err.name === 'AbortError') {
        setUrlError('Request timed out. The document might be too large or the proxy is slow.');
      } else {
        if (err.message === 'Document inaccessible') {
          alert('Document inaccessible. Please ensure the document is shared with "Anyone with the link".');
        }
        setUrlError(err.message || 'Load failed');
      }
    } finally { setIsUrlLoading(false); }
  };

  // Auto-load from ?url= query param or /url=<target> path (used by the bookmarklet)
  useEffect(() => {
    let u = new URLSearchParams(window.location.search).get('url');
    const pathMatch = window.location.pathname.match(/^\/url=(.+)$/);
    if (!u && pathMatch) {
      try { u = decodeURIComponent(pathMatch[1]); } catch { u = pathMatch[1]; }
    }
    if (u) {
      history.replaceState(null, '', '/');
      handleUrlLoad(u);
    }
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', backgroundColor: t.bg, fontFamily: 'system-ui, sans-serif', color: t.text, paddingTop: '1rem', transition: TT }}>
      {isDocLoading ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', gap: '0.85rem', color: t.textMuted }}>
          <Icon name="loader-circle" size={28} color={t.textMuted} style={{ animation: 'spin 0.8s linear infinite' }} />
          <span>{currentFileNameSignal.value ? `Loading ${currentFileNameSignal.value}…` : 'Loading…'}</span>
        </div>
      ) : (!hasSentences.value && !isOcrMode) ? (
        <Suspense fallback={null}><LandingCard isDarkMode={isDarkMode} t={t} isDragOver={isDragOver} setIsDragOver={setIsDragOver} onFileDrop={handleFileDrop} urlInputValue={urlInputValue} setUrlInputValue={setUrlInputValue} urlError={urlError} setUrlError={setUrlError} isUrlLoading={isUrlLoading} onUrlLoad={handleUrlLoad} onClipboardPaste={async (e: any) => { e.stopPropagation(); try { const t = await navigator.clipboard.readText(); if (/^https?:\/\/\S+$/.test(t)) handleUrlLoad(t); else if (t) loadText(t, (sents, content, outline, isFinal) => { sentencesSignal.value = sents; setEpubContent(content); outlineSignal.value = outline; if (!fileTypeSignal.value && (sents.length > 5 || isFinal)) { fileTypeSignal.value = 'text'; setIsDocLoading(false); } }, setShowTextInput, setTextInputValue); else setShowTextInput(true); } catch { setShowTextInput(true); } }} showTextInput={showTextInput} setShowTextInput={setShowTextInput} textInputValue={textInputValue} setTextInputValue={setTextInputValue} onLoadText={(t: string) => loadText(t, (sents, content, outline, isFinal) => { sentencesSignal.value = sents; setEpubContent(content); outlineSignal.value = outline; if (!fileTypeSignal.value && (sents.length > 5 || isFinal)) { fileTypeSignal.value = 'text'; setIsDocLoading(false); } }, setShowTextInput, setTextInputValue)} bookmarks={bookmarks} onSelectBookmark={(e: any) => e.fileType === 'url' && e.url && handleUrlLoad(e.url)} onDeleteBookmark={(id: string) => { removeBookmark(id); setBookmarks(getBookmarks()); }} /></Suspense>
      ) : (
        <Suspense fallback={null}><ContentViewer fileType={fileTypeSignal.peek()!} pages={pages} pdfDoc={pdfDoc} epubContent={epubContent} activeHeaderId={activeHeaderId} isDarkMode={isDarkMode} t={t} fontSize={fontSize} onLineClick={handleLineClick} ocrCurrentPage={ocrCurrentPage} setOcrCurrentPage={changeOcrPage} ocrPageLoading={ocrPageLoading} /></Suspense>
      )}
      <style>{`
        html, body { background-color: ${t.bg}; }
        .epub-highlight-active { background: rgba(250,204,21,0.45); border-radius:3px; }
        .word-highlight-active { background: #b47a32 !important; color: white !important; border-radius:3px; z-index: 30; }
        .pdf-highlight-active { background-color: rgba(250, 204, 21, 0.45) !important; box-shadow: 0 0 0 1px rgba(210, 170, 0, 0.4) !important; z-index:20 !important; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes toastIn { from { opacity: 0; transform: translateX(-50%) translateY(8px); } to { opacity: 1; transform: translateX(-50%) translateY(0); } }
        @keyframes toastOut { from { opacity: 1; transform: translateX(-50%) translateY(0); } to { opacity: 0; transform: translateX(-50%) translateY(8px); } }
      `}</style>
      {ocrLoadingStage && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 10000,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          backgroundColor: isDarkMode ? 'rgba(15,15,15,0.85)' : 'rgba(245,239,227,0.85)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          color: t.text,
        }}>
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem',
            padding: '2.5rem', borderRadius: '1.25rem',
            backgroundColor: t.menuBg, border: `1px solid ${t.menuBorder}`,
            boxShadow: '0 8px 32px rgba(0,0,0,0.15)', maxWidth: '400px', width: '90%', boxSizing: 'border-box',
          }}>
            <Icon name="loader-circle" size={32} color={isDarkMode ? '#60a5fa' : '#2563eb'} style={{ animation: 'spin 0.8s linear infinite' }} />
            <div style={{ fontSize: '1.05rem', fontWeight: 600, color: t.headerColor, textAlign: 'center' }}>Running OCR Engine</div>
            <div style={{ fontSize: '0.875rem', color: t.textMuted, textAlign: 'center', lineHeight: 1.4 }}>
              {ocrLoadingStage}
            </div>
            {ocrProgress !== null && (
              <div style={{ width: '100%', height: '6px', backgroundColor: t.inputBorder, borderRadius: '999px', overflow: 'hidden', marginTop: '0.5rem' }}>
                <div style={{ width: `${ocrProgress * 100}%`, height: '100%', backgroundColor: isDarkMode ? '#60a5fa' : '#2563eb', transition: 'width 0.1s ease' }} />
              </div>
            )}
          </div>
        </div>
      )}
      <OfflineToast />
      {eggPhase && <div style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}><div style={{ width: '300px', height: '300px', borderRadius: '50%', backgroundColor: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', animation: eggPhase === 'in' ? 'eggBounceIn 0.55s forwards' : 'eggFadeOut 0.6s forwards' }}><img src="./180.png" style={{ width: '250px', height: '250px' }} /></div></div>}
      <ScrollToCurrentButton />
      <Suspense fallback={null}><BottomBar t={t} isDarkMode={isDarkMode} themeName={themeName} onThemeChange={(name: ThemeName) => { setThemeName(name); localStorage.setItem('theme', name); }} playbackSpeed={playbackSpeedSignal.value} hasSentences={hasSentences.value} onTogglePlay={togglePlay} onSpeedChange={(s: number) => { playbackSpeedSignal.value = s; stopAllAudio(); if (isPlayingSignal.peek()) restartSignal.value++; }} usingFallback={usingFallback.current} selectedVoice={selectedVoiceSignal.value} onVoiceChange={(e: any) => { selectedVoiceSignal.value = e.target.value; stopAllAudio(); if (isPlayingSignal.peek()) restartSignal.value++; }} sentencesLength={sentencesSignal.value.length} fontSize={fontSize} onFontSizeChange={(d: number) => { const n = parseFloat(Math.max(0.8, Math.min(1.6, fontSize + d)).toFixed(2)); setFontSize(n); localStorage.setItem('fontSize', String(n)); }} onTestAudio={handleTestAudio} onReset={resetReader}           onLogoClick={() => { if (!eggPhase) { setEggPhase('in'); const t1 = window.setTimeout(() => { setEggPhase('out'); const t2 = window.setTimeout(() => setEggPhase(null), 600); eggTimersRef.current.push(t2); }, 2400); eggTimersRef.current.push(t1); } }} /></Suspense>
    </div>
  );
}

function OfflineToast() {
  const show = offlineReadySignal.value;
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    if (!show) return;
    const fadeTimer = setTimeout(() => setLeaving(true), 2600);
    const hideTimer = setTimeout(() => { offlineReadySignal.value = false; setLeaving(false); }, 3200);
    return () => { clearTimeout(fadeTimer); clearTimeout(hideTimer); };
  }, [show]);

  if (!show) return null;
  return (
    <div style={{
      position: 'fixed', bottom: '5.5rem', left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 400,
      display: 'flex', alignItems: 'center', gap: '0.4rem',
      padding: '0.45rem 0.9rem',
      background: 'rgba(15,15,15,0.88)',
      backdropFilter: 'blur(10px)',
      WebkitBackdropFilter: 'blur(10px)',
      color: '#e2fce8',
      borderRadius: '999px',
      fontSize: '0.78rem',
      fontWeight: 500,
      letterSpacing: '0.01em',
      boxShadow: '0 2px 12px rgba(0,0,0,0.25)',
      animation: `${leaving ? 'toastOut' : 'toastIn'} 0.3s ease forwards`,
      pointerEvents: 'none',
      whiteSpace: 'nowrap',
    }}>
      <Icon name="check" size={13} color="#4ade80" strokeWidth={2.5} />
      Offline ready
    </div>
  );
}

function ScrollToCurrentButton() {
  const isPlaying = isPlayingSignal.value, idx = currentSentenceIndexSignal.value, sents = sentencesSignal.value;
  if (sents.length === 0 || isPlaying || idx < 0) return null;
  return <button onClick={() => { const u = sents[idx]; if (u) document.getElementById(`line-${u.lines[0]}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); }} style={{ position: 'fixed', bottom: '1rem', right: '1rem', zIndex: 200, display: 'flex', alignItems: 'center', padding: '0.4rem 0.7rem', backgroundColor: '#2a2015', color: '#c0b4a4', border: '1px solid #1a1510', borderRadius: '999px', cursor: 'pointer', fontSize: '0.72rem', boxShadow: '0 2px 10px rgba(0,0,0,0.25)' }}><Icon name="target" size={13} /></button>;
}

function eraseRuledRowsFromCanvas(lctx: CanvasRenderingContext2D, W: number, H: number) {
  const idata = lctx.getImageData(0, 0, W, H);
  const d = idata.data;
  const thresh = Math.floor(W * 0.55);
  for (let y = 0; y < H; y++) {
    let cnt = 0;
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      if (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2] < 180) cnt++;
    }
    if (cnt > thresh) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        d[i] = d[i + 1] = d[i + 2] = d[i + 3] = 255;
      }
    }
  }
  lctx.putImageData(idata, 0, 0);
}
