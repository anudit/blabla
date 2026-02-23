import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import ePub from 'epubjs';
import { Play, Pause, Upload, Loader2, FileText, Beaker, AlertCircle, Activity, Menu, BookOpen, ChevronDown, Clipboard, Sun, Moon } from 'lucide-react';
import BookmarkHistory, { BookmarkEntry, getBookmarks, saveBookmark, removeBookmark } from './components/BookmarkHistory';
import BookOutline, { OutlineEntry } from './components/BookOutline';

// Set worker source
pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs';

const VOICES = [
  { value: 'af_bella', label: 'Bella (Eng F)' },
  { value: 'af_heart', label: 'Heart (Eng F)' },
  { value: 'am_fenrir', label: 'Fenrir (Eng M)' },
  { value: 'am_puck', label: 'Puck (Eng M)' },
];

// ── Fixed yellow highlight (same on both themes) ───────────────────────
const LINE_ACTIVE_STYLE = {
  backgroundColor: 'rgba(250, 204, 21, 0.45)',
  boxShadow: '0 0 0 1px rgba(210, 170, 0, 0.4)',
  zIndex: 20,
};

const EPUB_HIGHLIGHT_STYLE = {
  backgroundColor: 'rgba(250, 204, 21, 0.45)',
};

// ── Static styles (layout / structure, no theming) ─────────────────────
const staticStyles = {
  canvas: { display: 'block', width: '100%', height: 'auto' } as React.CSSProperties,
  overlay: {
    position: 'absolute' as const, top: 0, left: 0, right: 0, bottom: 0,
    zIndex: 10, pointerEvents: 'none' as const,
  },
  lineBase: {
    position: 'absolute' as const, cursor: 'pointer', borderRadius: '2px',
    backgroundColor: 'transparent', transition: 'background-color 0.2s ease',
    pointerEvents: 'auto' as const,
  },
  buttonDisabled: { opacity: 0.5, cursor: 'not-allowed' as const, filter: 'grayscale(1)' },
  playButton: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: '42px', height: '42px', borderRadius: '50%',
    backgroundColor: '#2563eb', color: 'white', border: 'none', cursor: 'pointer' as const,
    boxShadow: '0 2px 4px rgba(37,99,235,0.3)', transition: 'transform 0.1s',
    flexShrink: 0,
  },
  statusLoading: { color: '#2563eb', backgroundColor: '#eff6ff', border: '1px solid #bfdbfe' },
  statusReady:   { color: '#059669', backgroundColor: '#d1fae5', border: '1px solid #6ee7b7' },
  statusFallback:{ color: '#d97706', backgroundColor: '#fef3c7', border: '1px solid #fde68a' },
};

// ── Theme tokens ───────────────────────────────────────────────────────
//   light → muted warm-paper tones
//   dark  → muted dark tones
//   bar always uses the opposite side
const THEMES = {
  light: {
    // Content area
    bg:            '#f5efe3',
    text:          '#3a3028',
    textMuted:     '#7a6e60',
    dropBg:        '#faf6ef',
    dropBorder:    '#c4b8a0',
    epubBg:        '#f5efe3',
    headerColor:   '#2c2218',
    menuBg:        '#f0ebe0',
    menuBorder:    '#d0c4b0',
    selectBg:      '#faf6ef',
    selectBorder:  '#c4b8a0',
    inputBg:       '#faf6ef',
    inputBorder:   '#c4b8a0',
    statBorder:    '#e0d8c8',
    resetBtnBg:    '#e8e0d0',
    testBtnBg:     '#e8f0ff',
    testBtnColor:  '#2563eb',
    testBtnBorder: '#bfdbfe',
    pageShadow:    '0 2px 8px rgba(100,80,60,0.12)',
    speedHighlight:'#d8e8f4',
    // Bottom bar (opposite = dark)
    barBg:         '#2a2015',
    barBorder:     '#1a1510',
    barIconColor:  '#b8ac9c',
    barSpeedBg:    '#3a3020',
    barSpeedColor: '#e8ddd0',
  },
  dark: {
    // Content area
    bg:            '#1a1917',
    text:          '#c8bfb0',
    textMuted:     '#8a8070',
    dropBg:        '#242018',
    dropBorder:    '#4a4235',
    epubBg:        '#1a1917',
    headerColor:   '#e8ddd0',
    menuBg:        '#242018',
    menuBorder:    '#3a3428',
    selectBg:      '#2a2520',
    selectBorder:  '#4a4235',
    inputBg:       '#2a2520',
    inputBorder:   '#4a4235',
    statBorder:    '#2e2a22',
    resetBtnBg:    '#3a3428',
    testBtnBg:     '#1e2d45',
    testBtnColor:  '#93c5fd',
    testBtnBorder: '#1d4ed8',
    pageShadow:    '0 2px 8px rgba(0,0,0,0.5)',
    speedHighlight:'#2a3a50',
    // Bottom bar (opposite = light cream)
    barBg:         '#ede8df',
    barBorder:     '#d0c8b8',
    barIconColor:  '#5a4f44',
    barSpeedBg:    '#ddd8cf',
    barSpeedColor: '#2a2015',
  },
};

// ── PDFPage (receives pageContainerStyle as prop for theming) ──────────
const PDFPage = React.memo(({ data, highlightedLineIds, onLineClick, pageContainerStyle }: any) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderTaskRef = useRef<any>(null);

  useEffect(() => {
    const renderPage = async () => {
      if (canvasRef.current && data) {
        if (renderTaskRef.current) {
          try { await renderTaskRef.current.cancel(); } catch (e) {}
        }
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;
        const renderTask = data.page.render({ canvasContext: ctx, viewport: data.viewport });
        renderTaskRef.current = renderTask;
        try {
          await renderTask.promise;
        } catch (e: any) {
          if (e.name !== 'RenderingCancelledException') console.error("Render error:", e);
        }
      }
    };
    renderPage();
    return () => { if (renderTaskRef.current) renderTaskRef.current.cancel(); };
  }, [data]);

  return (
    <div style={{ ...pageContainerStyle, maxWidth: data.viewport.width }}>
      <canvas ref={canvasRef} width={data.viewport.width} height={data.viewport.height} style={staticStyles.canvas} />
      <div style={staticStyles.overlay}>
        {data.lines.map((line: any) => (
          <div
            key={line.id}
            id={`line-${line.id}`}
            onClick={(e) => { e.stopPropagation(); onLineClick(line.id); }}
            style={{
              ...staticStyles.lineBase,
              ...(highlightedLineIds.includes(line.id) ? LINE_ACTIVE_STYLE : {}),
              left: line.left, top: line.top, width: line.width, height: line.height,
            }}
          />
        ))}
      </div>
    </div>
  );
});

function findTitleInToc(toc: any[], href: string): string | null {
  for (const entry of toc) {
    if (href.includes(entry.href)) return entry.label;
    if (entry.subitems) {
      const sub = findTitleInToc(entry.subitems, href);
      if (sub) return sub;
    }
  }
  return null;
}

function extractSentences(text: string): string[] {
  const sentences: string[] = [];
  const sentenceRegex = /[^.!?]+[.!?]+/g;
  let lastIndex = 0;
  let match;
  while ((match = sentenceRegex.exec(text)) !== null) {
    const sText = match[0].trim().replace(/\s+/g, ' ');
    if (sText.length > 0) sentences.push(sText);
    lastIndex = match.index + match[0].length;
  }
  const remaining = text.slice(lastIndex).trim().replace(/\s+/g, ' ');
  if (remaining.length > 0) sentences.push(remaining);
  return sentences;
}

// ── Markdown helpers (no deps, works offline) ─────────────────────────

/** Strip inline markdown syntax for clean TTS text. */
function stripMd(s: string): string {
  return s
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
}

/** True if the text looks like markdown (has at least one ATX heading). */
function isMarkdown(text: string): boolean {
  return /^#{1,6} \S/m.test(text);
}

// Matches **bold**, __bold__, ~~strike~~, *italic*, _italic_, `code`, [text](url)
const INLINE_MD_RE = /\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`|\[[^\]]+\]\([^)]+\)/g;

/** Render inline markdown to React nodes for visual display. */
function renderMd(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let ki = 0, last = 0;
  for (const m of text.matchAll(new RegExp(INLINE_MD_RE.source, 'g'))) {
    if (m.index! > last) parts.push(text.slice(last, m.index));
    const s = m[0];
    if (s.startsWith('**') || s.startsWith('__')) {
      parts.push(<strong key={ki++}>{s.slice(2, -2)}</strong>);
    } else if (s.startsWith('~~')) {
      parts.push(<s key={ki++}>{s.slice(2, -2)}</s>);
    } else if (s.startsWith('*') || s.startsWith('_')) {
      parts.push(<em key={ki++}>{s.slice(1, -1)}</em>);
    } else if (s.startsWith('`')) {
      parts.push(<code key={ki++} style={{ fontFamily: 'monospace', fontSize: '0.88em', padding: '0.1em 0.25em', backgroundColor: 'rgba(127,127,127,0.15)', borderRadius: '3px' }}>{s.slice(1, -1)}</code>);
    } else {
      // Link: show text, drop href
      parts.push((s.match(/\[([^\]]+)\]/) || ['', s])[1]);
    }
    last = m.index! + s.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length > 0 ? <>{parts}</> : text;
}

export default function App() {
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [pages, setPages] = useState<any[]>([]);
  const [allLines, setAllLines] = useState<any[]>([]);
  const [sentences, setSentences] = useState<any[]>([]);
  const [currentSentenceIndex, setCurrentSentenceIndex] = useState(-1);
  const [highlightedLineIds, setHighlightedLineIds] = useState<number[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  // Theme
  const [isDarkMode, setIsDarkMode] = useState(false);

  // File type state
  const [fileType, setFileType] = useState<'pdf' | 'epub' | 'text' | null>(null);
  const [epubContent, setEpubContent] = useState<any[]>([]);
  const [showTextInput, setShowTextInput] = useState(false);
  const [textInputValue, setTextInputValue] = useState('');

  const [ttsStatus, setTtsStatus] = useState("Init");
  const [isModelReady, setIsModelReady] = useState(false);
  const [playbackState, setPlaybackState] = useState("Idle");
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState('af_bella');

  const workerRef = useRef<Worker | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const audioCache = useRef(new Map());
  const pendingFetches = useRef(new Set());
  const currentSource = useRef<AudioBufferSourceNode | null>(null);
  const playbackSessionId = useRef(0);
  const usingFallback = useRef(false);
  const nativeTimeout = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioResolvers = useRef(new Map<number, (buffer: AudioBuffer) => void>());
  const isWaitingForAudio = useRef(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [eggPhase, setEggPhase] = useState<'in' | 'out' | null>(null);
  const eggTimerRef = useRef<any>(null);

  // Bookmark / history state
  const [bookmarks, setBookmarks] = useState<BookmarkEntry[]>(() => getBookmarks());
  const [currentFileId, setCurrentFileId] = useState<string | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string | null>(null);
  const pendingResumeRef = useRef<number>(-1);

  // ── Derive theme tokens ──────────────────────────────────────────────
  const t = isDarkMode ? THEMES.dark : THEMES.light;

  // Shared transition applied to every element that changes on theme switch
  const TT = 'background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease';

  // ── Themed style objects (computed from t) ───────────────────────────
  const styles = {
    container: {
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      backgroundColor: t.bg,
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
      color: t.text,
      paddingTop: '1rem',
      transition: TT,
    },
    bottomBar: {
      position: 'fixed' as const,
      bottom: '1.5rem',
      left: '50%',
      transform: 'translateX(-50%)',
      backgroundColor: t.barBg,
      padding: '0.4rem 0.75rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.4rem',
      boxShadow: '0 4px 20px rgba(0,0,0,0.28)',
      zIndex: 50,
      borderRadius: '999px',
      border: `1px solid ${t.barBorder}`,
      whiteSpace: 'nowrap' as const,
      transition: TT,
    },
    statusBadgeMenu: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '0.5rem',
      fontSize: '0.75rem',
      padding: '0.5rem',
      borderRadius: '0.375rem',
      fontWeight: '600' as const,
      marginBottom: '0.5rem',
    },
    statItem: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      fontSize: '0.8rem',
      color: t.textMuted,
      padding: '0.5rem 0.25rem',
      borderBottom: `1px solid ${t.statBorder}`,
      transition: TT,
    },
    statLabel: { fontSize: '0.75rem', color: t.textMuted, transition: TT },
    statValue: { fontSize: '0.8rem', fontWeight: '600' as const, color: t.text, fontFamily: 'monospace', transition: TT },
    iconButton: {
      padding: '0.5rem',
      borderRadius: '0.375rem',
      border: 'none',
      cursor: 'pointer' as const,
      backgroundColor: 'transparent',
      color: t.barIconColor,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      transition: TT,
    },
    speedButton: {
      fontSize: '0.8rem',
      fontWeight: '700' as const,
      color: t.barSpeedColor,
      backgroundColor: t.barSpeedBg,
      padding: '0.25rem 0.5rem',
      borderRadius: '0.25rem',
      border: 'none',
      minWidth: '2.5rem',
      cursor: 'pointer' as const,
      transition: TT,
    },
    menuPopover: {
      position: 'absolute' as const,
      bottom: 'calc(100% + 10px)',
      right: '1rem',
      width: '260px',
      backgroundColor: t.menuBg,
      padding: '1rem',
      borderRadius: '0.75rem',
      boxShadow: '0 10px 25px -5px rgba(0,0,0,0.18), 0 8px 10px -6px rgba(0,0,0,0.12)',
      display: 'flex',
      flexDirection: 'column' as const,
      gap: '0.5rem',
      border: `1px solid ${t.menuBorder}`,
      zIndex: 100,
      transition: TT,
    },
    selectLabel: {
      fontSize: '0.75rem',
      fontWeight: '600' as const,
      color: t.textMuted,
      marginTop: '0.5rem',
      marginBottom: '0.25rem',
      transition: TT,
    },
    select: {
      width: '100%',
      padding: '0.5rem',
      borderRadius: '0.375rem',
      border: `1px solid ${t.selectBorder}`,
      backgroundColor: t.selectBg,
      color: t.text,
      fontSize: '0.875rem',
      cursor: 'pointer' as const,
      transition: TT,
    },
    testButton: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '0.5rem',
      padding: '0.5rem',
      fontSize: '0.875rem',
      fontWeight: '600' as const,
      backgroundColor: t.testBtnBg,
      color: t.testBtnColor,
      border: `1px solid ${t.testBtnBorder}`,
      borderRadius: '0.375rem',
      cursor: 'pointer' as const,
      marginTop: '0.5rem',
      transition: TT,
    },
    resetButton: {
      padding: '0.5rem',
      fontSize: '0.875rem',
      backgroundColor: t.resetBtnBg,
      color: '#ef4444',
      border: 'none',
      borderRadius: '0.375rem',
      cursor: 'pointer' as const,
      width: '100%',
      marginTop: '0.5rem',
      transition: TT,
    },
    dropZone: {
      marginTop: '2rem',
      width: '90%',
      maxWidth: '42rem',
      minHeight: '200px',
      border: `3px dashed ${t.dropBorder}`,
      borderRadius: '1rem',
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      cursor: 'pointer' as const,
      backgroundColor: t.dropBg,
      transition: 'all 0.3s ease',
      padding: '1rem',
      textAlign: 'center' as const,
    },
    dropZoneHover: {
      borderColor: '#6b9fd4',
      backgroundColor: isDarkMode ? '#1e2d3d' : '#eef4fb',
      transform: 'scale(1.01)',
    },
    viewer: {
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      width: '100%',
      maxWidth: '48rem',
      padding: '1rem',
      paddingBottom: '6rem',
      boxSizing: 'border-box' as const,
    },
    pageContainer: {
      position: 'relative' as const,
      marginBottom: '1rem',
      boxShadow: t.pageShadow,
      backgroundColor: '#fff', // PDF canvas always renders on white
      width: '100%',
      height: 'auto',
    },
    epubContainer: {
      width: '100%',
      padding: '0.25rem 0',
      lineHeight: '1.8',
      fontSize: '1.05rem',
      textAlign: 'left' as const,
      backgroundColor: t.epubBg,
      color: t.text,
      transition: TT,
    },
    epubSentence: {
      cursor: 'pointer' as const,
      padding: '2px 0',
      transition: `background-color 0.2s, ${TT}`,
      borderRadius: '4px',
    },
  };

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
  }, [isPlaying, currentSentenceIndex]);

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

  // Auto-resume: fires once sentences are loaded if a bookmark was found for this file
  useEffect(() => {
    if (sentences.length > 0 && pendingResumeRef.current > 0) {
      const idx = Math.min(pendingResumeRef.current, sentences.length - 1);
      pendingResumeRef.current = -1;
      setCurrentSentenceIndex(idx);
      setHighlightedLineIds(sentences[idx]?.lines || []);
      setIsPlaying(true);
      setTimeout(() => {
        const lineId = sentences[idx]?.lines?.[0];
        if (lineId !== undefined) {
          const el = document.getElementById(`line-${lineId}`);
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 350);
    }
  }, [sentences]);

  // Save progress to localStorage whenever the sentence index advances
  useEffect(() => {
    if (currentSentenceIndex > 0 && sentences.length > 0 && currentFileId && currentFileName && fileType && fileType !== 'text') {
      const entry: BookmarkEntry = {
        id: currentFileId,
        fileName: currentFileName,
        sentenceIndex: currentSentenceIndex,
        totalSentences: sentences.length,
        timestamp: Date.now(),
        fileType: fileType as 'pdf' | 'epub',
        preview: (sentences[currentSentenceIndex]?.text || '').slice(0, 80),
      };
      saveBookmark(entry);
      setBookmarks(getBookmarks());
    }
  }, [currentSentenceIndex]);

  const triggerEasterEgg = () => {
    if (eggPhase !== null) return;
    setEggPhase('in');
    eggTimerRef.current = setTimeout(() => {
      setEggPhase('out');
      setTimeout(() => setEggPhase(null), 600);
    }, 2400);
  };

  useEffect(() => {
    return () => { if (eggTimerRef.current) clearTimeout(eggTimerRef.current); };
  }, []);

  const triggerFallback = () => {
    if (usingFallback.current) return;
    usingFallback.current = true;
    setTtsStatus("System Voice");
    setIsModelReady(true);
  };

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
    // Speed is already baked into the audio by the TTS model; play at 1.0
    // so pitch is not artificially shifted.
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

  const stopAllAudio = () => {
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

    setHighlightedLineIds(unit.lines);

    setTimeout(() => {
      const firstLineId = unit.lines[0];
      const el = document.getElementById(`line-${firstLineId}`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 50);

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
      // Speed is baked into the audio by the TTS model; play at 1.0 so the
      // model's native tempo is preserved without raising pitch.
      source.playbackRate.value = 1.0;
      source.connect(audioContext.current!.destination);
      source.onended = () => {
        currentSource.current = null;
        if (isPlaying && currentSession === playbackSessionId.current) {
          advanceSentence();
        }
      };
      currentSource.current = source;
      setPlaybackState("Playing");
      source.start(0);
    } else {
      triggerFallback();
      playCurrentSentence();
    }
  };

  const advanceSentence = () => {
    setCurrentSentenceIndex(prev => prev + 1);
  };

  const handleLineClick = (lineId: number) => {
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e){}
      currentSource.current.disconnect();
      currentSource.current = null;
    }
    window.speechSynthesis.cancel();
    playbackSessionId.current += 1;
    audioCache.current.clear();
    pendingFetches.current.clear();
    audioResolvers.current.clear();
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
    setPdfDoc(null);
    setPages([]);
    setAllLines([]);
    setSentences([]);
    setFileType(null);
    setEpubContent([]);
    setCurrentSentenceIndex(-1);
    setHighlightedLineIds([]);
    audioCache.current.clear();
    pendingFetches.current.clear();
    audioResolvers.current.clear();
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

  const handleVoiceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newVoice = e.target.value;
    setSelectedVoice(newVoice);
    audioCache.current.clear();
    pendingFetches.current.clear();
    audioResolvers.current.clear();
    if (isPlaying) {
      stopAllAudio();
      setPlaybackState("Starting");
      playCurrentSentence();
    }
  };

  const handleFileDrop = async (e: React.DragEvent | React.ChangeEvent) => {
    e.preventDefault();
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
      // Check for existing bookmark — loader will set pendingResumeRef
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
      } else {
        alert("Please drop a valid PDF, EPUB or Markdown file");
      }
    }
  };

  const loadMarkdown = (raw: string) => {
    const newSentences: any[] = [];
    const contentData: any[] = [];
    let idCounter = 0;
    let inFence = false;
    let fenceLines: string[] = [];
    let paraLines: string[] = [];

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
        // Store original markdown text only when it differs (has inline syntax)
        paraSentences.push({ id: lineId, text: clean, ...(clean !== s ? { md: s } : {}) });
      }
      if (paraSentences.length > 0) {
        contentData.push({ type: 'paragraph', id: paraId, sentences: paraSentences });
      }
    };

    for (const line of raw.split('\n')) {
      // Fenced code block open/close
      if (/^(`{3,}|~{3,})/.test(line)) {
        if (!inFence) { flushPara(); inFence = true; fenceLines = []; }
        else { inFence = false; contentData.push({ type: 'code', id: `code-${idCounter++}`, text: fenceLines.join('\n') }); }
        continue;
      }
      if (inFence) { fenceLines.push(line); continue; }

      // ATX heading: # … ######
      const hm = line.match(/^(#{1,6})\s+(.+)$/);
      if (hm) {
        flushPara();
        contentData.push({ type: 'header', id: `header-${idCounter++}`, text: stripMd(hm[2].replace(/\s+#+\s*$/, '').trim()), level: hm[1].length });
        continue;
      }

      // Thematic break (---, ***, ___)
      if (/^\s{0,3}([-*_]\s*){3,}$/.test(line) && !/\w/.test(line)) {
        flushPara();
        contentData.push({ type: 'hr', id: `hr-${idCounter++}` });
        continue;
      }

      // Blank line → end of paragraph block
      if (!line.trim()) { flushPara(); continue; }

      // List item — strip marker, ensure ends with sentence-ending punct
      const li = line.match(/^\s*(?:[-*+]|\d+[.)]) (.*)/);
      if (li) {
        const t = li[1].trim();
        paraLines.push(/[.!?:;]$/.test(t) ? t : t + '.');
        continue;
      }

      // Blockquote
      const bq = line.match(/^>\s*(.*)/);
      if (bq) { paraLines.push(bq[1]); continue; }

      paraLines.push(line);
    }

    // Flush any unclosed fence or trailing paragraph
    if (inFence) contentData.push({ type: 'code', id: `code-${idCounter++}`, text: fenceLines.join('\n') });
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
        paraSentences.push({ id: lineId, text: sText });
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

  const handleClipboardPaste = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const text = await navigator.clipboard.readText();
      if (text.trim()) {
        loadText(text);
      } else {
        setShowTextInput(true);
      }
    } catch {
      setShowTextInput(true);
    }
  };

  const loadEPUB = async (data: ArrayBuffer) => {
    try {
      setFileType('epub');
      const book = ePub(data);
      await book.ready;
      const spine = book.spine;
      const newSentences: any[] = [];
      const contentData: any[] = [];
      let globalLineIdCounter = 0;
      const items = (spine as any).items || [];
      const navigation = await book.loaded.navigation;

      for (const item of items) {
        try {
          const doc = await book.load(item.href);
          if (!doc) continue;
          let chapterTitle = null;
          if (navigation && (navigation as any).toc) {
            chapterTitle = findTitleInToc((navigation as any).toc, item.href);
          }
          if (!chapterTitle) {
            const h1 = doc.querySelector('h1');
            if (h1 && h1.innerText.trim().length > 0) chapterTitle = h1.innerText.trim();
          }

          if (chapterTitle) {
            contentData.push({ type: 'header', id: `header-${globalLineIdCounter}`, text: chapterTitle });
          }

          const blockEls = Array.from(doc.querySelectorAll('p, li, blockquote'));
          const elementsToProcess: Element[] = blockEls.length > 0 ? blockEls : [doc.body];

          for (const el of elementsToProcess) {
            const rawText = (el as HTMLElement).innerText || el.textContent || '';
            const cleanText = rawText.trim().replace(/\s+/g, ' ');
            if (!cleanText) continue;

            const paraId = `para-${globalLineIdCounter}`;
            const paraSentences: any[] = [];
            for (const sText of extractSentences(cleanText)) {
              if (chapterTitle && sText.trim() === chapterTitle) continue;
              const lineId = globalLineIdCounter++;
              newSentences.push({ text: sText, lines: [lineId] });
              paraSentences.push({ id: lineId, text: sText });
            }
            if (paraSentences.length > 0) {
              contentData.push({ type: 'paragraph', id: paraId, sentences: paraSentences });
            }
          }
        } catch (err) { console.error("Error loading chapter", err); }
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
      const doc = await pdfjsLib.getDocument(data).promise;
      setPdfDoc(doc);
      const newPages = [];
      let globalLineList: any[] = [];
      for (let i = 1; i <= doc.numPages; i++) {
        const page = await doc.getPage(i);
        const scale = 1.5;
        const viewport = page.getViewport({ scale });
        const textContent = await page.getTextContent();
        const lines = processTextContent(textContent, viewport, scale, globalLineList.length);
        globalLineList.push(...lines);
        newPages.push({ page, viewport, lines, pageNumber: i });
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

  const processTextContent = (textContent: any, viewport: any, scale: number, startIndex: number) => {
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
      }
      else if (Math.abs(item.y - currentLine.y) < currentLine.height * 0.5) {
        currentLine.items.push(item);
      }
      else {
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

  // ── Outline (TOC) derived from epub/text headers ──────────────────────
  const outline: OutlineEntry[] = useMemo(() =>
    epubContent.filter(i => i.type === 'header').map(i => ({ id: i.id, text: i.text, level: i.level })),
    [epubContent]
  );

  // Which header section is currently being read
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

  const getStatusBadgeStyle = () => {
    if (ttsStatus === "Downloading...") return { ...styles.statusBadgeMenu, ...staticStyles.statusLoading };
    if (ttsStatus === "Model Ready")    return { ...styles.statusBadgeMenu, ...staticStyles.statusReady };
    if (ttsStatus === "System Voice")   return { ...styles.statusBadgeMenu, ...staticStyles.statusFallback };
    return { ...styles.statusBadgeMenu, ...staticStyles.statusLoading };
  };

  return (
    <div style={styles.container}>

      {sentences.length === 0 ? (
        <>
        <div
          style={{ ...styles.dropZone, ...(isDragOver ? styles.dropZoneHover : {}) }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleFileDrop}
        >
          <Upload size={32} color={t.textMuted} style={{ marginBottom: '1rem' }} />
          <p style={{ fontSize: '1.1rem', fontWeight: 600, color: t.text, marginBottom: '0.5rem' }}>
            Drop PDF, EPUB or Markdown here
          </p>
          <p style={{ fontSize: '0.85rem', color: t.textMuted }}>
            or tap to browse files
          </p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileDrop}
            accept="application/pdf,.epub,.md,.markdown"
            style={{ display: 'none' }}
          />
          <div style={{ width: '100%', borderTop: `1px solid ${t.dropBorder}`, margin: '1rem 0' }} />
          <button
            onClick={handleClipboardPaste}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
              padding: '0.5rem 1rem', fontSize: '0.875rem', fontWeight: 600,
              backgroundColor: t.testBtnBg, color: t.testBtnColor, border: `1px solid ${t.testBtnBorder}`,
              borderRadius: '0.375rem', cursor: 'pointer',
            }}
          >
            <Clipboard size={16} /> Paste from Clipboard
          </button>
          {showTextInput && (
            <div onClick={(e) => e.stopPropagation()} style={{ width: '100%', marginTop: '1rem' }}>
              <textarea
                value={textInputValue}
                onChange={(e) => setTextInputValue(e.target.value)}
                placeholder="Paste your text here..."
                style={{
                  width: '100%', minHeight: '100px', padding: '0.5rem',
                  borderRadius: '0.375rem', border: `1px solid ${t.inputBorder}`,
                  fontSize: '0.875rem', color: t.text, backgroundColor: t.inputBg,
                  boxSizing: 'border-box' as const, resize: 'vertical',
                }}
              />
              <button
                onClick={(e) => { e.stopPropagation(); if (textInputValue.trim()) loadText(textInputValue); }}
                style={{
                  marginTop: '0.5rem', width: '100%', padding: '0.5rem',
                  fontSize: '0.875rem', fontWeight: 600, backgroundColor: '#2563eb',
                  color: 'white', border: 'none', borderRadius: '0.375rem', cursor: 'pointer',
                }}
              >
                Start Reading
              </button>
            </div>
          )}
        </div>
        <BookmarkHistory
          bookmarks={bookmarks}
          onDelete={handleDeleteBookmark}
          isDarkMode={isDarkMode}
        />
        </>
      ) : (
        <div style={styles.viewer}>
          {fileType === 'pdf' && pages.map(pageData => (
            <PDFPage
              key={pageData.pageNumber}
              data={pageData}
              highlightedLineIds={highlightedLineIds}
              onLineClick={handleLineClick}
              pageContainerStyle={styles.pageContainer}
            />
          ))}

          {(fileType === 'epub' || fileType === 'text') && (
            <>
              <BookOutline entries={outline} activeId={activeHeaderId} isDarkMode={isDarkMode} />
              <div style={styles.epubContainer}>
                {(() => {
                  let headersSeen = 0;
                  const H_SIZE: Record<number, string> = { 1: '1.45rem', 2: '1.25rem', 3: '1.1rem', 4: '1rem', 5: '0.95rem', 6: '0.9rem' };
                  return epubContent.map((item) => {
                    if (item.type === 'header') {
                      const isFirst = headersSeen === 0;
                      headersSeen++;
                      const lvl: number = item.level ?? 1;
                      // Show a section divider only before top-level headings (or EPUB chapters which have no level)
                      const showDivider = !isFirst && lvl <= 1;
                      return (
                        <React.Fragment key={item.id}>
                          {showDivider && (
                            <hr style={{
                              border: 'none',
                              borderTop: `1px solid ${t.statBorder}`,
                              margin: '3rem 0 2.5rem',
                              opacity: 0.45,
                            }} />
                          )}
                          <div
                            id={item.id}
                            style={{
                              fontSize: H_SIZE[lvl] ?? '1.45rem',
                              fontWeight: lvl <= 2 ? 700 : 600,
                              margin: isFirst ? `0 0 ${lvl === 1 ? '1.4rem' : '1rem'}` : `${lvl === 1 ? '2rem' : '1.4rem'} 0 ${lvl === 1 ? '1.4rem' : '0.6rem'}`,
                              color: t.headerColor,
                              lineHeight: 1.25,
                              letterSpacing: lvl === 1 ? '-0.015em' : 'normal',
                              scrollMarginTop: '1.5rem',
                            }}
                          >
                            {item.text}
                          </div>
                        </React.Fragment>
                      );
                    }
                    if (item.type === 'paragraph') {
                      return (
                        <p key={item.id} style={{ margin: '0 0 1.1em', padding: 0, lineHeight: 'inherit' }}>
                          {item.sentences.map((sentence: any) => {
                            const isActive = highlightedLineIds.includes(sentence.id);
                            return (
                              <span
                                key={sentence.id}
                                id={`line-${sentence.id}`}
                                onClick={() => handleLineClick(sentence.id)}
                                style={{ ...styles.epubSentence, ...(isActive ? EPUB_HIGHLIGHT_STYLE : {}) }}
                              >
                                {sentence.md ? renderMd(sentence.md) : sentence.text}{' '}
                              </span>
                            );
                          })}
                        </p>
                      );
                    }
                    if (item.type === 'code') {
                      return (
                        <pre key={item.id} style={{
                          backgroundColor: isDarkMode ? 'rgba(0,0,0,0.35)' : 'rgba(0,0,0,0.04)',
                          border: `1px solid ${t.statBorder}`,
                          borderRadius: '0.5rem',
                          padding: '0.75rem 1rem',
                          overflowX: 'auto',
                          fontSize: '0.82rem',
                          fontFamily: 'monospace',
                          lineHeight: 1.6,
                          margin: '0 0 1.1em',
                          color: t.text,
                          whiteSpace: 'pre',
                        }}>
                          <code>{item.text}</code>
                        </pre>
                      );
                    }
                    if (item.type === 'hr') {
                      return <hr key={item.id} style={{ border: 'none', borderTop: `1px solid ${t.statBorder}`, margin: '1.5rem 0', opacity: 0.5 }} />;
                    }
                    return null;
                  });
                })()}
              </div>
            </>
          )}
        </div>
      )}

      {/* Keyframes */}
      <style>{`
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
            <img src="./logo.png" style={{ width: '250px', height: '250px' }} alt="" />
          </div>
        </div>
      )}

      {/* Bottom Bar */}
      <div style={styles.bottomBar}>
        <button
          onClick={triggerEasterEgg}
          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center' }}
        >
          <img src="./logo.png" style={{ width: '32px', height: '32px' }} alt="logo" />
        </button>

        <button
          onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)}
          style={styles.speedButton}
        >
          {playbackSpeed}x
        </button>

        <button
          onClick={togglePlay}
          disabled={sentences.length === 0 || !isModelReady}
          style={{
            ...staticStyles.playButton,
            ...(sentences.length === 0 || !isModelReady ? staticStyles.buttonDisabled : {})
          }}
        >
          {playbackState === 'Buffering'
            ? <Loader2 size={20} color="white" style={{ animation: 'spin 0.8s linear infinite' }} />
            : isPlaying
              ? <Pause size={20} fill="white" />
              : <Play size={20} fill="white" style={{ marginLeft: '2px' }} />
          }
        </button>

        <button onClick={() => setIsMenuOpen(!isMenuOpen)} style={styles.iconButton}>
          <Menu size={24} />
        </button>

        {/* Theme toggle — at the end of the bar */}
        <button
          onClick={() => setIsDarkMode(!isDarkMode)}
          style={styles.iconButton}
          title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
        </button>

        {isSpeedMenuOpen && (
          <div style={{ ...styles.menuPopover, width: '100px', right: '0', bottom: 'calc(100% + 10px)' }}>
            {[1.0, 1.25, 1.5, 2.0].map(speed => (
              <button
                key={speed}
                onClick={() => {
                  setPlaybackSpeed(speed);
                  // Clear cache so upcoming sentences are re-generated at the
                  // new speed (model bakes speed into audio, so stale buffers
                  // at the old speed must be discarded).
                  audioCache.current.clear();
                  pendingFetches.current.clear();
                  audioResolvers.current.clear();
                  setIsSpeedMenuOpen(false);
                }}
                style={{
                  ...styles.statItem,
                  width: '100%', cursor: 'pointer',
                  backgroundColor: playbackSpeed === speed ? t.speedHighlight : 'transparent',
                  border: 'none',
                  color: t.text,
                }}
              >
                {speed}x
              </button>
            ))}
          </div>
        )}

        {isMenuOpen && (
          <div style={styles.menuPopover}>
            <div style={getStatusBadgeStyle()}>
              {ttsStatus === "Loading..."   && <Loader2 size={14} className="animate-spin" />}
              {ttsStatus === "Model Ready"  && <Activity size={14} />}
              {ttsStatus === "System Voice" && <AlertCircle size={14} />}
              {ttsStatus}
            </div>

            <label style={styles.selectLabel}>Select Voice</label>
            <div style={{ position: 'relative' }}>
              <select
                value={selectedVoice}
                onChange={handleVoiceChange}
                disabled={usingFallback.current || !isModelReady}
                style={{ ...styles.select, ...(usingFallback.current || !isModelReady ? staticStyles.buttonDisabled : {}) }}
              >
                {VOICES.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
              </select>
            </div>

            <div style={{ marginTop: '0.5rem' }}>
              <div style={styles.statItem}>
                <span style={styles.statLabel}>State</span>
                <span style={styles.statValue}>{playbackState}</span>
              </div>
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Progress</span>
                <span style={styles.statValue}>
                  {currentSentenceIndex >= 0 ? `${Math.round((currentSentenceIndex / sentences.length) * 100)}%` : '0%'}
                </span>
              </div>
            </div>

            <button
              onClick={handleTestAudio}
              style={{ ...styles.testButton, ...(!isModelReady ? staticStyles.buttonDisabled : {}) }}
              disabled={!isModelReady}
            >
              <Beaker size={14} /> Test Voice
            </button>
            <button onClick={resetReader} style={styles.resetButton}>Reset Document</button>
          </div>
        )}
      </div>
    </div>
  );
}
