import React, { useState, useEffect, useRef, useMemo } from 'react';
// import * as pdfjsLib from 'pdfjs-dist';
import { unzipSync } from 'fflate';
import { Target } from 'lucide-react';
import { THEMES, TT } from './theme';
import {
  findTitleInToc, extractWords, calculateWordTimings,
  extractSentences, stripMd, isMarkdown, extractRuns,
} from './utils';
import { getBookmarks, saveBookmark, removeBookmark } from './components/BookmarkHistory';
import type { BookmarkEntry } from './components/BookmarkHistory';
import type { OutlineEntry } from './components/BookOutline';
import LandingCard from './components/LandingCard';
import ContentViewer from './components/ContentViewer';
import BottomBar from './components/BottomBar';

// pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs';

export default function App() {
  // ── State ──────────────────────────────────────────────────────────────
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [pages, setPages] = useState<any[]>([]);
  const [allLines, setAllLines] = useState<any[]>([]);
  const [sentences, setSentences] = useState<any[]>([]);
  const [currentSentenceIndex, setCurrentSentenceIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');
  const [fileType, setFileType] = useState<'pdf' | 'epub' | 'text' | null>(null);
  const [epubContent, setEpubContent] = useState<any[]>([]);
  const [showTextInput, setShowTextInput] = useState(false);
  const [textInputValue, setTextInputValue] = useState('');
  const [urlInputValue, setUrlInputValue] = useState('');
  const [isUrlLoading, setIsUrlLoading] = useState(false);
  const [urlError, setUrlError] = useState('');
  const [ttsStatus, setTtsStatus] = useState("Init");
  const [isModelReady, setIsModelReady] = useState(false);
  const [playbackState, setPlaybackState] = useState("Idle");
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [selectedVoice, setSelectedVoice] = useState('af_bella');
  const [restartTrigger, setRestartTrigger] = useState(0);
  const [fontSize, setFontSize] = useState(() => {
    const s = parseFloat(localStorage.getItem('fontSize') || '');
    return isNaN(s) ? 1.05 : s;
  });
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [eggPhase, setEggPhase] = useState<'in' | 'out' | null>(null);
  const [bookmarks, setBookmarks] = useState<BookmarkEntry[]>(() => getBookmarks());
  const [currentFileId, setCurrentFileId] = useState<string | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string | null>(null);

  // ── Refs ───────────────────────────────────────────────────────────────
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

  // ── Derived ────────────────────────────────────────────────────────────
  const t = isDarkMode ? THEMES.dark : THEMES.light;

  // ── Effects ────────────────────────────────────────────────────────────
  useEffect(() => {
    console.log("[App] Mounting...");
    const ctx = getAudioContext();
    ctx.resume().catch(console.warn);
    workerRef.current = new Worker("/tts.worker.js", { type: 'module' });
    workerRef.current.onmessage = (e) => {
      const { status, audio, error, text, lineIndex, device, dtype } = e.data;
      if (status === 'ready') {
        setTtsStatus(`Ready (${device}/${dtype})`);
        setIsModelReady(true);
      }
      else if (status === 'complete') {
        try {
          const ctx = getAudioContext();
          const buffer = ctx.createBuffer(1, audio.length, 24000);
          buffer.getChannelData(0).set(audio);

          if (text.startsWith("Hello! I am")) {
            playBufferDirectly(buffer);
          } else if (lineIndex !== undefined) {
            const resolver = audioResolvers.current.get(lineIndex);
            if (resolver) {
              resolver(buffer);
              audioResolvers.current.delete(lineIndex);
            }
          }
        } catch (err) {
          console.error("Audio conversion failed", err);
        }
      }
      else if (status === 'error') {
        console.error("[Worker Error]", error);
        if (error.includes("import_promises") || error.includes("tokenizer")) {
          triggerFallback();
        }
      }
    };
    setTtsStatus("Downloading Model...");
    workerRef.current.postMessage({ type: 'init' });

    return () => {
      stopAllAudio();
      workerRef.current?.terminate();
      if (audioContext.current) audioContext.current.close();
    };
  }, []);

  useEffect(() => {
    if (isPlaying && currentSentenceIndex >= 0 && !isWaitingForAudio.current) {
      playCurrentSentence();
    }
  }, [isPlaying, currentSentenceIndex, restartTrigger]);

  // Global Cmd+V paste → URL or text load on landing page
  useEffect(() => {
    if (sentences.length > 0) return;
    const handler = (e: ClipboardEvent) => {
      const tag = (document.activeElement as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      const text = e.clipboardData?.getData('text') ?? '';
      const trimmed = text.trim();
      if (!trimmed) return;
      e.preventDefault();
      if (/^https?:\/\/\S+$/.test(trimmed)) {
        handleUrlLoad(trimmed);
      } else {
        loadText(trimmed);
      }
    };
    window.addEventListener('paste', handler);
    return () => window.removeEventListener('paste', handler);
  }, [sentences.length]);

  // DOM-based highlight + smart scroll (bypasses React diffing entirely)
  useEffect(() => {
    document.querySelectorAll('.epub-highlight-active').forEach(el => el.classList.remove('epub-highlight-active'));
    document.querySelectorAll('.pdf-highlight-active').forEach(el => el.classList.remove('pdf-highlight-active'));

    const unit = sentences[currentSentenceIndex];
    if (currentSentenceIndex < 0 || !unit) return;

    unit.lines.forEach((id: number) => {
      const el = document.getElementById(`line-${id}`);
      if (!el) return;
      if (fileType === 'pdf') {
        el.classList.add('pdf-highlight-active');
      } else {
        const target = (el.textContent === '' && el.closest('p')) ? el.closest('p')! : el;
        target.classList.add('epub-highlight-active');
      }
    });

    const firstId = unit.lines[0];
    const target = document.getElementById(`line-${firstId}`);
    if (!target) return;
    const rect = target.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const isVisible = rect.top >= 80 && rect.bottom <= vh - 120;
    if (!isVisible) {
      setTimeout(() => target.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50);
    }
  }, [currentSentenceIndex, sentences, fileType]);

  useEffect(() => {
    const handleKeyDown = async (e: KeyboardEvent) => {
      const isPaste = (e.metaKey || e.ctrlKey) && e.key === 'v';
      if (!isPaste || sentences.length > 0) return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'TEXTAREA' || tag === 'INPUT') return;
      e.preventDefault();
      try {
        const text = await navigator.clipboard.readText();
        if (text.trim()) loadText(text);
        else setShowTextInput(true);
      } catch {
        setShowTextInput(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sentences.length]);

  useEffect(() => {
    const handleMediaKey = (e: KeyboardEvent) => {
      if (e.key === 'MediaPlayPause' || e.key === 'F8') {
        e.preventDefault();
        togglePlay();
        return;
      }
      if (e.code === 'Space' && sentences.length > 0) {
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener('keydown', handleMediaKey);
    if ('mediaSession' in navigator) {
      navigator.mediaSession.setActionHandler('play', () => togglePlay());
      navigator.mediaSession.setActionHandler('pause', () => togglePlay());
    }
    return () => {
      window.removeEventListener('keydown', handleMediaKey);
      if ('mediaSession' in navigator) {
        try {
          navigator.mediaSession.setActionHandler('play', null);
          navigator.mediaSession.setActionHandler('pause', null);
        } catch (e) {}
      }
    };
  }, [isPlaying, sentences.length, isModelReady, currentSentenceIndex]);

  useEffect(() => {
    if (!('mediaSession' in navigator)) return;
    navigator.mediaSession.playbackState = isPlaying ? 'playing' : 'paused';
  }, [isPlaying]);

  useEffect(() => {
    if (!('mediaSession' in navigator) || !currentFileName) return;
    navigator.mediaSession.metadata = new MediaMetadata({ title: currentFileName, artist: 'blabla reader' });
  }, [currentFileName]);

  // Auto-resume: fires once sentences are loaded if a bookmark was found for this file
  useEffect(() => {
    if (sentences.length > 0 && pendingResumeRef.current > 0) {
      const idx = Math.min(pendingResumeRef.current, sentences.length - 1);
      pendingResumeRef.current = -1;
      setCurrentSentenceIndex(idx);
      setIsPlaying(true);
    }
  }, [sentences]);

  // Save progress to localStorage whenever the sentence index advances
  useEffect(() => {
    if (currentSentenceIndex > 0 && sentences.length > 0 && currentFileId && currentFileName && fileType) {
      const isUrl = currentFileId.startsWith('http://') || currentFileId.startsWith('https://');
      if (fileType === 'text' && !isUrl) return;
      const entry: BookmarkEntry = {
        id: currentFileId,
        fileName: currentFileName,
        sentenceIndex: currentSentenceIndex,
        totalSentences: sentences.length,
        timestamp: Date.now(),
        fileType: isUrl ? 'url' : fileType as 'pdf' | 'epub',
        preview: (sentences[currentSentenceIndex]?.text || '').slice(0, 80),
        ...(isUrl ? { url: currentFileId } : {}),
      };
      saveBookmark(entry);
      setBookmarks(getBookmarks());
    }
  }, [currentSentenceIndex]);

  useEffect(() => {
    return () => { if (eggTimerRef.current) clearTimeout(eggTimerRef.current); };
  }, []);

  // ── Audio helpers ──────────────────────────────────────────────────────
  const getAudioContext = () => {
    if (!audioContext.current || audioContext.current.state === 'closed') {
      audioContext.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 24000
      });
    }
    if (audioContext.current.state !== 'running') {
      audioContext.current.resume().catch(e => console.warn("[AudioCtx] resume() failed", e));
    }
    return audioContext.current;
  };

  const playBufferDirectly = (buffer: AudioBuffer) => {
    const ctx = getAudioContext();
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = 1.0;
    source.connect(ctx.destination);
    source.start(0);
    setPlaybackState("Playing");
    source.onended = () => setPlaybackState("Ready");
  };

  const generateAudioInWorker = (text: string, sentenceIndex: number, voice: string, speed: number): Promise<AudioBuffer | null> => {
    return new Promise((resolve) => {
      if (usingFallback.current || !workerRef.current) {
        resolve(null);
        return;
      }
      audioResolvers.current.set(sentenceIndex, resolve);
      workerRef.current.postMessage({ type: 'generate', text, lineIndex: sentenceIndex, voice, speed });
      setTimeout(() => {
        if (audioResolvers.current.has(sentenceIndex)) {
          audioResolvers.current.delete(sentenceIndex);
          resolve(null);
        }
      }, 30000);
    });
  };

  const drainResolvers = () => {
    audioResolvers.current.forEach(resolve => resolve(null));
    audioResolvers.current.clear();
  };

  const stopAllAudio = () => {
    if (wordRafRef.current) { cancelAnimationFrame(wordRafRef.current); wordRafRef.current = null; }
    document.querySelectorAll('.word-highlight-active').forEach(el => el.classList.remove('word-highlight-active'));
    playbackSessionId.current += 1;
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e){}
      currentSource.current.disconnect();
      currentSource.current = null;
    }
    window.speechSynthesis.cancel();
    if (nativeTimeout.current) clearTimeout(nativeTimeout.current);
    if (audioContext.current) {
      audioContext.current.close();
      audioContext.current = null;
    }
    audioCache.current.clear();
    pendingFetches.current.clear();
    drainResolvers();
    setPlaybackState("Stopped");
    isWaitingForAudio.current = false;
  };

  const processSentenceAudio = async (index: number, sessionId: number, voice: string, speed: number): Promise<AudioBuffer | null> => {
    if (usingFallback.current) return null;
    if (index >= sentences.length) return null;
    if (audioCache.current.has(index)) return audioCache.current.get(index);
    if (pendingFetches.current.has(index)) return null;

    const text = sentences[index].text;
    if (!text.trim()) return null;

    pendingFetches.current.add(index);
    try {
      const buffer = await generateAudioInWorker(text, index, voice, speed);
      if (!buffer) return null;
      if (sessionId === playbackSessionId.current) {
        audioCache.current.set(index, buffer);
      }
      return buffer;
    } catch (err) {
      console.error(`[Buffer] Failed to process sentence ${index}`, err);
      return null;
    } finally {
      pendingFetches.current.delete(index);
    }
  };

  const playCurrentSentence = async () => {
    if (!isPlaying || currentSentenceIndex === -1) return;
    if (currentSentenceIndex >= sentences.length) {
      setIsPlaying(false);
      setPlaybackState("Completed");
      return;
    }

    const currentSession = playbackSessionId.current;
    const unit = sentences[currentSentenceIndex];
    const text = unit.text;

    if (text.trim().length <= 1) {
      advanceSentence();
      return;
    }

    if (usingFallback.current) {
      window.speechSynthesis.cancel();
      if (nativeTimeout.current) clearTimeout(nativeTimeout.current);
      nativeTimeout.current = setTimeout(() => {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = playbackSpeed;
        u.onend = () => {
          if (isPlaying && currentSession === playbackSessionId.current) {
            advanceSentence();
          }
        };
        u.onerror = () => { if (isPlaying) advanceSentence(); };
        window.speechSynthesis.speak(u);
        setPlaybackState("Playing");
      }, 50);
      return;
    }

    getAudioContext();
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e){}
      currentSource.current.disconnect();
      currentSource.current = null;
    }

    for (let i = 1; i <= 3; i++) {
      if (currentSentenceIndex + i < sentences.length && !audioCache.current.has(currentSentenceIndex + i)) {
        processSentenceAudio(currentSentenceIndex + i, currentSession, selectedVoice, playbackSpeed);
      }
    }

    let buffer = audioCache.current.get(currentSentenceIndex);
    if (!buffer) {
      setPlaybackState("Buffering");
      isWaitingForAudio.current = true;
      buffer = await processSentenceAudio(currentSentenceIndex, currentSession, selectedVoice, playbackSpeed);
      isWaitingForAudio.current = false;
    }

    if (usingFallback.current) {
      playCurrentSentence();
      return;
    }

    if (currentSession !== playbackSessionId.current || !isPlaying) return;

    if (buffer) {
      const source = audioContext.current!.createBufferSource();
      source.buffer = buffer;
      source.playbackRate.value = 1.0;
      source.connect(audioContext.current!.destination);

      if (wordRafRef.current) cancelAnimationFrame(wordRafRef.current);

      source.onended = () => {
        if (wordRafRef.current) { cancelAnimationFrame(wordRafRef.current); wordRafRef.current = null; }
        document.querySelectorAll('.word-highlight-active').forEach(el => el.classList.remove('word-highlight-active'));
        currentSource.current = null;
        if (isPlaying && currentSession === playbackSessionId.current) {
          advanceSentence();
        }
      };
      currentSource.current = source;
      setPlaybackState("Playing");

      wordTimingsRef.current = calculateWordTimings(text, buffer.duration);
      audioStartRef.current = audioContext.current!.currentTime;
      source.start(0);

      // rAF loop: direct DOM updates at 60 fps, no React re-renders
      const lineId = unit.lines[0];
      const animateWords = () => {
        if (currentSource.current !== source) return;
        const elapsed = audioContext.current!.currentTime - audioStartRef.current;
        const active = wordTimingsRef.current.find(t => elapsed >= t.start && elapsed < t.end);
        document.querySelectorAll('.word-highlight-active').forEach(el => el.classList.remove('word-highlight-active'));
        if (active) {
          document.getElementById(`word-${lineId}-${active.index}`)?.classList.add('word-highlight-active');
        }
        if (elapsed < buffer.duration) {
          wordRafRef.current = requestAnimationFrame(animateWords);
        }
      };
      wordRafRef.current = requestAnimationFrame(animateWords);
    } else {
      triggerFallback();
      playCurrentSentence();
    }
  };

  const advanceSentence = () => {
    setCurrentSentenceIndex(prev => {
      const next = prev + 1;
      // Sliding-window cache: keep only prev-1..next+3
      for (const key of Array.from(audioCache.current.keys())) {
        if (key < next - 1 || key > next + 3) audioCache.current.delete(key);
      }
      return next;
    });
  };

  // ── Handlers ───────────────────────────────────────────────────────────
  const triggerEasterEgg = () => {
    if (eggPhase !== null) return;
    setEggPhase('in');
    eggTimerRef.current = setTimeout(() => {
      setEggPhase('out');
      setTimeout(() => setEggPhase(null), 600);
    }, 2400);
  };

  const triggerFallback = () => {
    if (usingFallback.current) return;
    usingFallback.current = true;
    setTtsStatus("System Voice");
    setIsModelReady(true);
  };

  const handleLineClick = (lineId: number) => {
    if (wordRafRef.current) { cancelAnimationFrame(wordRafRef.current); wordRafRef.current = null; }
    document.querySelectorAll('.word-highlight-active').forEach(el => el.classList.remove('word-highlight-active'));
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e){}
      currentSource.current.disconnect();
      currentSource.current = null;
    }
    window.speechSynthesis.cancel();
    playbackSessionId.current += 1;
    audioCache.current.clear();
    pendingFetches.current.clear();
    drainResolvers();
    isWaitingForAudio.current = false;

    let targetSentenceIndex = -1;
    for (let i = 0; i < sentences.length; i++) {
      if (sentences[i].lines.includes(lineId)) {
        targetSentenceIndex = i;
        break;
      }
    }
    if (targetSentenceIndex !== -1) {
      setCurrentSentenceIndex(targetSentenceIndex);
      setIsPlaying(true);
      const el = document.getElementById(`line-${lineId}`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const togglePlay = () => {
    if (!isModelReady) return;
    if (currentSentenceIndex === -1 && sentences.length > 0) {
      setCurrentSentenceIndex(0);
      setIsPlaying(true);
      return;
    }
    if (isPlaying) {
      setIsPlaying(false);
      stopAllAudio();
    } else {
      setIsPlaying(true);
      setPlaybackState("Starting");
    }
  };

  const handleTestAudio = () => {
    if (!isModelReady) return;
    stopAllAudio();
    const text = "Hello! I am ready to read.";
    if (usingFallback.current) {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = playbackSpeed;
      window.speechSynthesis.speak(u);
      setPlaybackState("Testing");
    } else {
      setPlaybackState("Generating");
      workerRef.current?.postMessage({ type: 'generate', text, voice: selectedVoice, speed: playbackSpeed });
    }
  };

  const resetReader = () => {
    setIsPlaying(false);
    stopAllAudio();
    if (pdfDoc) { try { pdfDoc.destroy(); } catch(e) {} }
    setPdfDoc(null);
    setPages([]);
    setAllLines([]);
    setSentences([]);
    setFileType(null);
    setEpubContent([]);
    setCurrentSentenceIndex(-1);
    audioCache.current.clear();
    pendingFetches.current.clear();
    drainResolvers();
    setPlaybackState("Idle");
    isWaitingForAudio.current = false;
    setIsMenuOpen(false);
    setShowTextInput(false);
    setTextInputValue('');
    setCurrentFileId(null);
    setCurrentFileName(null);
    pendingResumeRef.current = -1;
  };

  const handleDeleteBookmark = (id: string) => {
    removeBookmark(id);
    setBookmarks(getBookmarks());
  };

  const handleSelectBookmark = (entry: BookmarkEntry) => {
    if (entry.fileType === 'url' && entry.url) {
      handleUrlLoad(entry.url);
    }
  };

  const handleVoiceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newVoice = e.target.value;
    setSelectedVoice(newVoice);
    drainResolvers();
    audioCache.current.clear();
    pendingFetches.current.clear();
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e) {}
      currentSource.current.disconnect();
      currentSource.current = null;
    }
    window.speechSynthesis.cancel();
    playbackSessionId.current += 1;
    isWaitingForAudio.current = false;
    if (isPlaying) setRestartTrigger(p => p + 1);
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    drainResolvers();
    audioCache.current.clear();
    pendingFetches.current.clear();
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e) {}
      currentSource.current.disconnect();
      currentSource.current = null;
    }
    window.speechSynthesis.cancel();
    playbackSessionId.current += 1;
    isWaitingForAudio.current = false;
    if (isPlaying) setRestartTrigger(p => p + 1);
  };

  const handleToggleTheme = () => {
    const next = !isDarkMode;
    setIsDarkMode(next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
  };

  const handleFontSizeChange = (delta: number) => {
    const next = parseFloat(Math.max(0.8, Math.min(1.6, fontSize + delta)).toFixed(2));
    setFontSize(next);
    localStorage.setItem('fontSize', String(next));
  };

  const handleFileDrop = async (e: React.DragEvent<HTMLDivElement> | React.ChangeEvent<HTMLInputElement>) => {
    if ('preventDefault' in e) e.preventDefault();
    setIsDragOver(false);
    let file: File | undefined;
    if ('dataTransfer' in e) {
      file = e.dataTransfer.files[0];
    } else if ('target' in e) {
      file = (e.target as HTMLInputElement).files?.[0];
    }

    if (file) {
      const fileId = `${file.name}:${file.size}`;
      setCurrentFileId(fileId);
      setCurrentFileName(file.name);
      const existing = getBookmarks().find(b => b.id === fileId);
      if (existing) pendingResumeRef.current = existing.sentenceIndex;

      if (file.type === 'application/pdf') {
        const reader = new FileReader();
        reader.onload = (ev) => loadPDF(ev.target?.result as ArrayBuffer);
        reader.readAsArrayBuffer(file);
      } else if (file.type === 'application/epub+zip' || file.name.endsWith('.epub')) {
        const reader = new FileReader();
        reader.onload = (ev) => loadEPUB(ev.target?.result as ArrayBuffer);
        reader.readAsArrayBuffer(file);
      } else if (file.name.endsWith('.md') || file.name.endsWith('.markdown') || file.type === 'text/markdown' || file.type === 'text/x-markdown') {
        const reader = new FileReader();
        reader.onload = (ev) => loadMarkdown(ev.target?.result as string);
        reader.readAsText(file);
      } else if (file.name.endsWith('.txt') || file.type === 'text/plain') {
        const reader = new FileReader();
        reader.onload = (ev) => loadText(ev.target?.result as string);
        reader.readAsText(file);
      } else {
        alert("Please drop a valid PDF, EPUB, Markdown, or TXT file");
      }
    }
  };

  const handleUrlLoad = async (overrideUrl?: string) => {
    const url = (overrideUrl || urlInputValue).trim();
    if (!url) return;
    try { new URL(url); } catch { setUrlError('Please enter a valid URL (include https://)'); return; }
    setIsUrlLoading(true);
    setUrlError('');
    try {
      const res = await fetch(`https://test.cors.workers.dev/?https://markdown.new/${url}?method=auto&retain_images`);
      if (!res.ok) throw new Error(`Could not fetch URL (${res.status})`);
      const markdown = await res.text();
      if (!markdown.trim()) throw new Error('No content returned for that URL');
      const titleLine = markdown.match(/^Title:\s*(.+)/m);
      const pageTitle = titleLine ? titleLine[1].trim() : new URL(url).hostname;
      setCurrentFileId(url);
      setCurrentFileName(pageTitle);
      const existing = getBookmarks().find(b => b.id === url);
      if (existing) pendingResumeRef.current = existing.sentenceIndex;
      setUrlInputValue('');
      loadMarkdown(markdown);
    } catch (err: any) {
      setUrlError(err.message || 'Failed to load URL');
    } finally {
      setIsUrlLoading(false);
    }
  };

  const handleClipboardPaste = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    try {
      const text = await navigator.clipboard.readText();
      const trimmed = text.trim();
      if (/^https?:\/\/\S+$/.test(trimmed)) {
        handleUrlLoad(trimmed);
      } else if (trimmed) {
        loadText(trimmed);
      } else {
        setShowTextInput(true);
      }
    } catch {
      setShowTextInput(true);
    }
  };

  // ── Loaders ────────────────────────────────────────────────────────────
  const loadMarkdown = (raw: string) => {
    const newSentences: any[] = [];
    const contentData: any[] = [];
    let idCounter = 0;

    // Strip AI reader header
    let content = raw;
    const jinaMarker = '\nMarkdown Content:\n';
    const jinaIdx = raw.indexOf(jinaMarker);
    if (jinaIdx !== -1) content = raw.slice(jinaIdx + jinaMarker.length).trimStart();

    // Parse YAML frontmatter (--- ... ---)
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

    let inFence = false;
    let fenceLines: string[] = [];
    let paraLines: string[] = [];
    let tableLines: string[] = [];

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
        const headers = dataRows[0];
        const rows = dataRows.slice(1);
        contentData.push({ type: 'table', id: `table-${idCounter++}`, headers, rows });
      }
      tableLines = [];
    };

    const flushPara = () => {
      if (!paraLines.length) return;
      const block = paraLines.join(' ').replace(/\s+/g, ' ').trim();
      paraLines = [];
      if (!block) return;
      const paraId = `para-${idCounter}`;
      const paraSentences: any[] = [];
      for (const s of extractSentences(block)) {
        const clean = stripMd(s);
        if (!clean.trim()) continue;
        const lineId = idCounter++;
        newSentences.push({ text: clean, lines: [lineId] });
        paraSentences.push({ id: lineId, text: clean, words: extractWords(clean), ...(clean !== s ? { md: s } : {}) });
      }
      if (paraSentences.length > 0) {
        contentData.push({ type: 'paragraph', id: paraId, sentences: paraSentences });
      }
    };

    for (const line of content.split('\n')) {
      if (/^(`{3,}|~{3,})/.test(line)) {
        if (!inFence) { flushPara(); inFence = true; fenceLines = []; }
        else { inFence = false; contentData.push({ type: 'code', id: `code-${idCounter++}`, text: fenceLines.join('\n') }); }
        continue;
      }
      if (inFence) { fenceLines.push(line); continue; }

      const hm = line.match(/^(#{1,6})\s+(.+)$/);
      if (hm) {
        flushPara();
        contentData.push({ type: 'header', id: `header-${idCounter++}`, text: stripMd(hm[2].replace(/\s+#+\s*$/, '').trim()), level: hm[1].length });
        continue;
      }

      if (/^\s{0,3}([-*_]\s*){3,}$/.test(line) && !/\w/.test(line)) {
        flushPara();
        contentData.push({ type: 'hr', id: `hr-${idCounter++}` });
        continue;
      }

      if (!line.trim()) { flushPara(); flushTable(); continue; }

      if (line.includes('|')) {
        flushPara();
        tableLines.push(line);
        continue;
      }
      if (tableLines.length) flushTable();

      const imgM = line.match(/^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$/);
      if (imgM) {
        flushPara();
        contentData.push({ type: 'image', id: `img-${idCounter++}`, src: imgM[2], alt: imgM[1] });
        continue;
      }

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
    flushTable();
    flushPara();

    if (!newSentences.length) return;
    setFileType('text');
    setSentences(newSentences);
    setEpubContent(contentData);
    setShowTextInput(false);
    setTextInputValue('');
  };

  const loadText = (text: string) => {
    if (isMarkdown(text)) { loadMarkdown(text); return; }
    const newSentences: any[] = [];
    const contentData: any[] = [];
    let globalLineIdCounter = 0;

    const paragraphs = text
      .split(/\n{2,}/)
      .map(p => p.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim())
      .filter(p => p.length > 0);

    for (const paraText of paragraphs) {
      const paraId = `para-${globalLineIdCounter}`;
      const paraSentences: any[] = [];
      for (const sText of extractSentences(paraText)) {
        const lineId = globalLineIdCounter++;
        newSentences.push({ text: sText, lines: [lineId] });
        paraSentences.push({ id: lineId, text: sText, words: extractWords(sText) });
      }
      if (paraSentences.length > 0) {
        contentData.push({ type: 'paragraph', id: paraId, sentences: paraSentences });
      }
    }

    if (newSentences.length === 0) return;
    setFileType('text');
    setSentences(newSentences);
    setEpubContent(contentData);
    setShowTextInput(false);
    setTextInputValue('');
  };

  const loadEPUB = async (data: ArrayBuffer) => {
    try {
      setFileType('epub');
      const files = unzipSync(new Uint8Array(data));
      const decoder = new TextDecoder();
      const parser = new DOMParser();

      // 1. Find the .opf file via container.xml
      const containerXml = decoder.decode(files['META-INF/container.xml']);
      const containerDoc = parser.parseFromString(containerXml, 'text/xml');
      const opfPath = containerDoc.querySelector('rootfile')?.getAttribute('full-path');
      if (!opfPath) throw new Error("No OPF file found");

      const opfDir = opfPath.includes('/') ? opfPath.substring(0, opfPath.lastIndexOf('/') + 1) : '';
      const opfXml = decoder.decode(files[opfPath]);
      const opfDoc = parser.parseFromString(opfXml, 'text/xml');

      // 2. Parse Manifest (id -> href)
      const manifest: Record<string, string> = {};
      opfDoc.querySelectorAll('manifest > item').forEach(el => {
        const id = el.getAttribute('id');
        const href = el.getAttribute('href');
        if (id && href) manifest[id] = href;
      });

      // 3. Parse Spine (reading order)
      const spineIds = Array.from(opfDoc.querySelectorAll('spine > itemref')).map(el => el.getAttribute('idref')!);

      // 4. Try to parse TOC for chapter titles
      let toc: { href: string; label: string }[] = [];
      // EPUB 3 Nav
      const navItem = Array.from(opfDoc.querySelectorAll('manifest > item')).find(el => el.getAttribute('properties')?.includes('nav'));
      if (navItem) {
        const navPath = opfDir + navItem.getAttribute('href');
        if (files[navPath]) {
          const navDoc = parser.parseFromString(decoder.decode(files[navPath]), 'text/html');
          toc = Array.from(navDoc.querySelectorAll('nav li a')).map(a => ({
            href: a.getAttribute('href') || '',
            label: a.textContent?.trim() || ''
          }));
        }
      }
      // EPUB 2 NCX fallback
      if (toc.length === 0) {
        const ncxItem = Array.from(opfDoc.querySelectorAll('manifest > item')).find(el => el.getAttribute('media-type') === 'application/x-dtbncx+xml' || el.getAttribute('id') === 'ncx');
        if (ncxItem) {
          const ncxPath = opfDir + ncxItem.getAttribute('href');
          if (files[ncxPath]) {
            const ncxDoc = parser.parseFromString(decoder.decode(files[ncxPath]), 'text/xml');
            toc = Array.from(ncxDoc.querySelectorAll('navPoint')).map(np => ({
              href: np.querySelector('content')?.getAttribute('src') || '',
              label: np.querySelector('navLabel text')?.textContent?.trim() || ''
            }));
          }
        }
      }

      const newSentences: any[] = [];
      const contentData: any[] = [];
      let globalLineIdCounter = 0;
      const normHead = (s: string) => s.replace(/[\u00a0\s]+/g, ' ').trim();
      let lastAddedHeaderText: string | null = null;

      const normalizePath = (path: string) => {
        const parts = path.split('/');
        const result: string[] = [];
        for (const part of parts) {
          if (part === '..') result.pop();
          else if (part !== '.') result.push(part);
        }
        return result.join('/');
      };

      for (const id of spineIds) {
        const href = manifest[id];
        if (!href) continue;
        const fullPath = normalizePath(opfDir + href);
        const cleanPath = decodeURIComponent(fullPath.split('#')[0]);
        const fileBytes = files[cleanPath];
        if (!fileBytes) continue;

        const htmlString = decoder.decode(fileBytes);
        const doc = parser.parseFromString(htmlString, 'application/xhtml+xml');

        let chapterTitle: string | null = null;
        if (toc.length > 0) {
          chapterTitle = findTitleInToc(toc, href);
        }
        if (!chapterTitle) {
          const h1 = doc.querySelector('h1');
          const raw = (h1?.textContent || '').trim();
          if (raw) chapterTitle = raw;
        }

        if (chapterTitle) {
          const norm = normHead(chapterTitle);
          if (norm && norm !== lastAddedHeaderText) {
            lastAddedHeaderText = norm;
            contentData.push({ type: 'header', id: `header-${globalLineIdCounter++}`, text: norm, level: 1 });
          }
        }

        const blockEls = Array.from(doc.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, blockquote'));
        const elementsToProcess: Element[] = blockEls.length > 0 ? blockEls : (doc.body ? [doc.body] : []);

        for (const el of elementsToProcess) {
          const tag = el.tagName.toLowerCase();

          if (/^h[1-6]$/.test(tag)) {
            const headText = normHead(el.textContent || '');
            if (!headText || headText === lastAddedHeaderText) continue;
            lastAddedHeaderText = headText;
            contentData.push({ type: 'header', id: `header-${globalLineIdCounter++}`, text: headText, level: parseInt(tag[1]) });
            continue;
          }

          const runs = extractRuns(el);
          const rawText = runs.filter(r => !r.br).map(r => r.text).join('');
          const cleanText = rawText.replace(/\s+/g, ' ').trim();
          if (!cleanText) continue;

          const paraId = `para-${globalLineIdCounter}`;
          const paraSentences: any[] = [];
          for (const sText of extractSentences(cleanText)) {
            if (chapterTitle && sText.trim() === chapterTitle) continue;
            const lineId = globalLineIdCounter++;
            newSentences.push({ text: sText, lines: [lineId] });
            paraSentences.push({ id: lineId, text: sText, words: extractWords(sText) });
          }
          if (paraSentences.length > 0) {
            contentData.push({ type: 'paragraph', id: paraId, sentences: paraSentences, elementType: tag, runs });
          }
        }
      }
      setSentences(newSentences);
      setEpubContent(contentData);
    } catch (e) {
      console.error("Failed to load EPUB", e);
      alert("Could not load EPUB file");
    }
  };

  const loadPDF = async (data: ArrayBuffer) => {
    setFileType('pdf');
    try {
      // 1. Dynamically import PDF.js
      const pdfjsLib = await import('pdfjs-dist');
     // 2. Set the worker source dynamically
      pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs';

      const doc = await pdfjsLib.getDocument(data).promise;
      setPdfDoc(doc);
      const newPages = [];
      let globalLineList: any[] = [];
      const scale = Math.min(window.devicePixelRatio || 1, 2);

      for (let i = 1; i <= doc.numPages; i++) {
        const page = await doc.getPage(i);
        const viewport = page.getViewport({ scale });
        const textContent = await page.getTextContent();
        const lines = processTextContent(textContent, viewport, scale, globalLineList.length, pdfjsLib);
        globalLineList.push(...lines);
        newPages.push({ viewport, lines, pageNumber: i });
        page.cleanup();
        if (i % 10 === 0) await new Promise(r => setTimeout(r, 0));
      }

      const fullText = globalLineList.map(l => l.text).join(' ');
      const sentenceRegex = /[^.!?]+[.!?]/g;
      let match;
      const sentenceTexts = [];
      while ((match = sentenceRegex.exec(fullText)) !== null) {
        sentenceTexts.push({ text: match[0].trim(), start: match.index, end: match.index + match[0].length });
      }

      let cumPos = 0;
      globalLineList.forEach(line => {
        line.startPos = cumPos;
        cumPos += line.text.length + 1;
      });

      const newSentences = sentenceTexts.map(sent => {
        const sentLines = globalLineList.filter(line =>
          line.startPos < sent.end && (line.startPos + line.text.length + 1) > sent.start
        ).map(line => line.id);
        return { text: sent.text, lines: sentLines };
      }).filter(sent => sent.text && sent.lines.length > 0);

      setSentences(newSentences);
      setAllLines(globalLineList);
      setPages(newPages);
    } catch (e) { console.error(e); }
  };

  const processTextContent = async (textContent: any, viewport: any, scale: number, startIndex: number, pdfjsLib: any) => {
    const items = textContent.items.map((item: any) => {
      const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
      const fontHeight = Math.sqrt((tx[2] * tx[2]) + (tx[3] * tx[3]));
      return {
        str: item.str,
        x: tx[4],
        y: tx[5],
        width: item.width * scale,
        height: item.height > 0 ? item.height * scale : fontHeight,
      };
    });

    items.sort((a: any, b: any) => Math.abs(a.y - b.y) > a.height * 0.2 ? a.y - b.y : a.x - b.x);

    const lines: any[] = [];
    let currentLine: any = null;

    items.forEach((item: any) => {
      if (!currentLine) {
        currentLine = { items: [item], y: item.y, height: item.height };
      } else if (Math.abs(item.y - currentLine.y) < currentLine.height * 0.5) {
        currentLine.items.push(item);
      } else {
        lines.push(currentLine);
        currentLine = { items: [item], y: item.y, height: item.height };
      }
    });
    if (currentLine) lines.push(currentLine);

    return lines.map((line: any, idx: number) => {
      const minX = Math.min(...line.items.map((i: any) => i.x));
      const last = line.items[line.items.length - 1];
      const maxX = last.x + (last.width || last.str.length * 5);
      const width = maxX - minX;

      const leftPct  = (minX / viewport.width) * 100;
      const topPct   = ((line.y - line.height) / viewport.height) * 100;
      const widthPct = (width / viewport.width) * 100;
      const heightPct= (line.height / viewport.height) * 100;

      return {
        id: startIndex + idx,
        text: line.items.map((i: any) => i.str).join(' '),
        left: `${leftPct}%`,
        top: `${topPct}%`,
        width: `${widthPct}%`,
        height: `${heightPct}%`,
        startPos: 0
      };
    });
  };

  // ── Memos ──────────────────────────────────────────────────────────────
  const outline: OutlineEntry[] = useMemo(() =>
    epubContent.filter(i => i.type === 'header').map(i => ({ id: i.id, text: i.text, level: i.level })),
    [epubContent]
  );

  const activeHeaderId = useMemo(() => {
    if (currentSentenceIndex < 0 || !sentences[currentSentenceIndex]) return null;
    const currentLines = new Set<number>(sentences[currentSentenceIndex].lines);
    let lastId: string | null = null;
    for (const item of epubContent) {
      if (item.type === 'header') { lastId = item.id; }
      else if (item.type === 'paragraph') {
        for (const s of item.sentences) {
          if (currentLines.has(s.id)) return lastId;
        }
      }
    }
    return null;
  }, [currentSentenceIndex, epubContent, sentences]);

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      backgroundColor: t.bg,
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
      color: t.text,
      paddingTop: '1rem',
      transition: TT,
    }}>

      {sentences.length === 0 ? (
        <LandingCard
          isDarkMode={isDarkMode}
          t={t}
          isDragOver={isDragOver}
          setIsDragOver={setIsDragOver}
          onFileDrop={handleFileDrop}
          urlInputValue={urlInputValue}
          setUrlInputValue={setUrlInputValue}
          urlError={urlError}
          setUrlError={setUrlError}
          isUrlLoading={isUrlLoading}
          onUrlLoad={handleUrlLoad}
          onClipboardPaste={handleClipboardPaste}
          showTextInput={showTextInput}
          setShowTextInput={setShowTextInput}
          textInputValue={textInputValue}
          setTextInputValue={setTextInputValue}
          onLoadText={loadText}
          bookmarks={bookmarks}
          onSelectBookmark={handleSelectBookmark}
          onDeleteBookmark={handleDeleteBookmark}
        />
      ) : (
        <ContentViewer
          fileType={fileType!}
          pages={pages}
          pdfDoc={pdfDoc}
          epubContent={epubContent}
          outline={outline}
          activeHeaderId={activeHeaderId}
          isDarkMode={isDarkMode}
          t={t}
          fontSize={fontSize}
          onLineClick={handleLineClick}
        />
      )}

      {/* Global CSS — highlight classes and keyframes */}
      <style>{`
        .epub-highlight-active {
          background-color: rgba(250, 204, 21, 0.45) !important;
          border-radius: 3px;
        }
        .word-highlight-active {
          background-color: #b47a32 !important;
          color: inherit !important;
          border-radius: 3px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.25);
        }
        .pdf-highlight-active {
          background-color: rgba(250, 204, 21, 0.45) !important;
          box-shadow: 0 0 0 1px rgba(210, 170, 0, 0.4) !important;
          z-index: 20 !important;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes eggBounceIn {
          0%   { transform: scale(0.15); opacity: 0; }
          55%  { transform: scale(1.18); opacity: 1; }
          75%  { transform: scale(0.93); }
          90%  { transform: scale(1.05); }
          100% { transform: scale(1);    opacity: 1; }
        }
        @keyframes eggFadeOut {
          0%   { transform: scale(1);    opacity: 1; }
          25%  { transform: scale(1.06); opacity: 1; }
          100% { transform: scale(0.5);  opacity: 0; }
        }
      `}</style>

      {/* Easter egg overlay */}
      {eggPhase && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          pointerEvents: 'none',
        }}>
          <div style={{
            width: '300px', height: '300px', borderRadius: '50%',
            backgroundColor: 'white',
            boxShadow: '0 24px 80px rgba(0,0,0,0.28), 0 0 0 1px rgba(0,0,0,0.04)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            animation: eggPhase === 'in'
              ? 'eggBounceIn 0.55s cubic-bezier(0.34,1.56,0.64,1) forwards'
              : 'eggFadeOut 0.6s ease-in forwards',
          }}>
            <img src="./180.png" style={{ width: '250px', height: '250px' }} alt="" />
          </div>
        </div>
      )}

      {/* Scroll-to-current button — visible only when paused */}
      {sentences.length > 0 && !isPlaying && currentSentenceIndex >= 0 && (
        <button
          onClick={() => {
            const unit = sentences[currentSentenceIndex];
            if (!unit) return;
            const el = document.getElementById(`line-${unit.lines[0]}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }}
          title="Scroll to current position"
          style={{
            position: 'fixed', bottom: '1rem', right: '1rem', zIndex: 200,
            display: 'flex', alignItems: 'center', gap: '0.35rem',
            padding: '0.38rem 0.7rem',
            backgroundColor: '#2a2015',
            color: '#c0b4a4',
            border: '1px solid #1a1510',
            borderRadius: '999px',
            cursor: 'pointer',
            fontSize: '0.72rem',
            fontWeight: 700,
            letterSpacing: '0.03em',
            boxShadow: '0 2px 10px rgba(0,0,0,0.25)',
            transition: 'opacity 0.15s',
          }}
        >
          <Target size={13} />
        </button>
      )}

      <BottomBar
        t={t}
        isDarkMode={isDarkMode}
        onToggleTheme={handleToggleTheme}
        isPlaying={isPlaying}
        isModelReady={isModelReady}
        playbackState={playbackState}
        playbackSpeed={playbackSpeed}
        hasSentences={sentences.length > 0}
        onTogglePlay={togglePlay}
        onSpeedChange={handleSpeedChange}
        isMenuOpen={isMenuOpen}
        onToggleMenu={() => setIsMenuOpen(v => !v)}
        ttsStatus={ttsStatus}
        usingFallback={usingFallback.current}
        selectedVoice={selectedVoice}
        onVoiceChange={handleVoiceChange}
        currentSentenceIndex={currentSentenceIndex}
        sentencesLength={sentences.length}
        fontSize={fontSize}
        onFontSizeChange={handleFontSizeChange}
        onTestAudio={handleTestAudio}
        onReset={resetReader}
        onLogoClick={triggerEasterEgg}
      />
    </div>
  );
}
