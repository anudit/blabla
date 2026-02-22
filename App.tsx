import React, { useState, useEffect, useRef } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import ePub from 'epubjs';
import { Play, Pause, Upload, Loader2, FileText, Beaker, AlertCircle, Activity, Menu, BookOpen, ChevronDown, Clipboard } from 'lucide-react';

// Set worker source
pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs';

const VOICES = [
  { value: 'af_bella', label: 'Bella (Eng F)' },
  { value: 'af_heart', label: 'Heart (Eng F)' },
  { value: 'am_fenrir', label: 'Fenrir (Eng M)' },
  { value: 'am_puck', label: 'Puck (Eng M)' },
];

const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
    color: '#333',
    paddingTop: '1rem',
  },
  bottomBar: {
    position: 'fixed' as const,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#fff',
    padding: '0.75rem 1rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    boxShadow: '0 -4px 6px -1px rgba(0,0,0,0.05)',
    zIndex: 50,
  },
  logoGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  title: {
    fontSize: '1.1rem',
    fontWeight: '800' as const,
    color: '#2563eb',
    margin: 0,
    letterSpacing: '-0.025em',
  },
  controlsGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
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
    width: '100%',
  },
  statusLoading: { color: '#2563eb', backgroundColor: '#eff6ff', border: '1px solid #bfdbfe' },
  statusReady: { color: '#059669', backgroundColor: '#d1fae5', border: '1px solid #6ee7b7' },
  statusFallback: { color: '#d97706', backgroundColor: '#fef3c7', border: '1px solid #fde68a' },
  statItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: '0.8rem',
    color: '#6b7280',
    padding: '0.5rem 0.25rem',
    borderBottom: '1px solid #f3f4f6',
  },
  statLabel: { fontSize: '0.75rem', color: '#9ca3af' },
  statValue: { fontSize: '0.8rem', fontWeight: '600' as const, color: '#374151', fontFamily: 'monospace' },
  playButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '42px',
    height: '42px',
    borderRadius: '50%',
    backgroundColor: '#2563eb',
    color: 'white',
    border: 'none',
    cursor: 'pointer',
    boxShadow: '0 2px 4px rgba(37, 99, 235, 0.3)',
    transition: 'transform 0.1s',
  },
  iconButton: {
    padding: '0.5rem',
    borderRadius: '0.375rem',
    border: 'none',
    cursor: 'pointer',
    backgroundColor: 'transparent',
    color: '#4b5563',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  speedButton: {
    fontSize: '0.8rem',
    fontWeight: '700',
    color: '#4b5563',
    backgroundColor: '#f3f4f6',
    padding: '0.25rem 0.5rem',
    borderRadius: '0.25rem',
    border: 'none',
    minWidth: '2.5rem',
  },
  menuPopover: {
    position: 'absolute' as const,
    bottom: 'calc(100% + 10px)',
    right: '1rem',
    width: '260px',
    backgroundColor: '#fff',
    padding: '1rem',
    borderRadius: '0.75rem',
    boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.5rem',
    border: '1px solid #e5e7eb',
    zIndex: 100,
  },
  selectLabel: {
    fontSize: '0.75rem',
    fontWeight: '600',
    color: '#6b7280',
    marginTop: '0.5rem',
    marginBottom: '0.25rem',
  },
  select: {
    width: '100%',
    padding: '0.5rem',
    borderRadius: '0.375rem',
    border: '1px solid #d1d5db',
    backgroundColor: '#f9fafb',
    color: '#374151',
    fontSize: '0.875rem',
    cursor: 'pointer',
  },
  testButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
    padding: '0.5rem',
    fontSize: '0.875rem',
    fontWeight: '600' as const,
    backgroundColor: '#eff6ff',
    color: '#2563eb',
    border: '1px solid #bfdbfe',
    borderRadius: '0.375rem',
    cursor: 'pointer',
    marginTop: '0.5rem',
  },
  resetButton: {
    padding: '0.5rem',
    fontSize: '0.875rem',
    backgroundColor: '#f3f4f6',
    color: '#ef4444',
    border: 'none',
    borderRadius: '0.375rem',
    cursor: 'pointer',
    width: '100%',
    marginTop: '0.5rem',
  },
  buttonDisabled: { opacity: 0.5, cursor: 'not-allowed', filter: 'grayscale(1)' },
  dropZone: {
    marginTop: '2rem',
    width: '90%',
    maxWidth: '42rem',
    minHeight: '200px',
    border: '3px dashed #d1d5db',
    borderRadius: '1rem',
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    backgroundColor: '#fff',
    transition: 'all 0.3s ease',
    padding: '1rem',
    textAlign: 'center' as const,
  },
  dropZoneHover: { borderColor: '#3b82f6', backgroundColor: '#eff6ff', transform: 'scale(1.01)' },
  viewer: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    width: '100%',
    maxWidth: '48rem', // Constrain max width for desktop reading comfort
    padding: '1rem',
    paddingBottom: '6rem',
    boxSizing: 'border-box' as const,
  },
  pageContainer: {
    position: 'relative' as const,
    marginBottom: '1rem',
    boxShadow: '0 2px 4px -1px rgba(0,0,0,0.1)',
    backgroundColor: '#fff',
    width: '100%', // Responsive width
    height: 'auto',
  },
  canvas: {
    display: 'block',
    width: '100%', // Scale canvas to container
    height: 'auto',
  },
  overlay: {
    position: 'absolute' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 10,
    pointerEvents: 'none' as const, // Let clicks pass through to container if needed, but we handle line clicks on divs
  },
  lineBase: {
    position: 'absolute' as const,
    cursor: 'pointer',
    borderRadius: '2px',
    backgroundColor: 'transparent',
    transition: 'background-color 0.2s ease',
    pointerEvents: 'auto' as const, // Re-enable pointer events for lines
  },
  lineActive: {
    backgroundColor: 'rgba(255, 255, 0, 0.25)', // Softer yellow
    boxShadow: '0 0 0 1px rgba(220, 220, 0, 0.5)',
    mixBlendMode: 'multiply' as const,
    zIndex: 20,
  },
  epubContainer: {
    width: '100%',
    padding: '0',
    lineHeight: '1.6',
    fontSize: '1.1rem',
    textAlign: 'left' as const,
    backgroundColor: 'white',
    borderRadius: '8px',
  },
  epubSentence: {
    cursor: 'pointer',
    padding: '2px 0',
    transition: 'background-color 0.2s',
    borderRadius: '4px',
  },
  epubHighlight: {
    backgroundColor: '#fef08a',
  }
};

const PDFPage = React.memo(({ data, highlightedLineIds, onLineClick }: any) => {
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
    <div style={{ ...styles.pageContainer, maxWidth: data.viewport.width }}>
      <canvas ref={canvasRef} width={data.viewport.width} height={data.viewport.height} style={styles.canvas} />
      <div style={styles.overlay}>
        {data.lines.map((line: any) => (
          <div
            key={line.id}
            id={`line-${line.id}`}
            onClick={(e) => { e.stopPropagation(); onLineClick(line.id); }}
            style={{
              ...styles.lineBase,
              ...(highlightedLineIds.includes(line.id) ? styles.lineActive : {}),
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

export default function App() {
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [pages, setPages] = useState<any[]>([]);
  const [allLines, setAllLines] = useState<any[]>([]);
  const [sentences, setSentences] = useState<any[]>([]);
  const [currentSentenceIndex, setCurrentSentenceIndex] = useState(-1);
  const [highlightedLineIds, setHighlightedLineIds] = useState<number[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

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
      // Don't intercept if focus is inside a textarea/input
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
    source.playbackRate.value = playbackSpeed;
    source.connect(ctx.destination);
    source.start(0);
    setPlaybackState("Playing");
    source.onended = () => setPlaybackState("Ready");
  };

  const generateAudioInWorker = (text: string, sentenceIndex: number, voice: string): Promise<AudioBuffer | null> => {
    return new Promise((resolve) => {
      if (usingFallback.current || !workerRef.current) {
        resolve(null);
        return;
      }
      audioResolvers.current.set(sentenceIndex, resolve);
      workerRef.current.postMessage({ type: 'generate', text, lineIndex: sentenceIndex, voice });
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

  const processSentenceAudio = async (index: number, sessionId: number, voice: string): Promise<AudioBuffer | null> => {
    if (usingFallback.current) return null;
    if (index >= sentences.length) return null;
    if (audioCache.current.has(index)) return audioCache.current.get(index);
    if (pendingFetches.current.has(index)) return null;

    const text = sentences[index].text;
    if (!text.trim()) return null;

    pendingFetches.current.add(index);
    try {
      const buffer = await generateAudioInWorker(text, index, voice);
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
        processSentenceAudio(currentSentenceIndex + i, currentSession, selectedVoice);
      }
    }

    let buffer = audioCache.current.get(currentSentenceIndex);
    if (!buffer) {
      setPlaybackState("Buffering");
      isWaitingForAudio.current = true;
      buffer = await processSentenceAudio(currentSentenceIndex, currentSession, selectedVoice);
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
      source.playbackRate.value = playbackSpeed;
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
      workerRef.current?.postMessage({ type: 'generate', text, voice: selectedVoice });
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
      if (file.type === 'application/pdf') {
        const reader = new FileReader();
        reader.onload = (ev) => loadPDF(ev.target?.result as ArrayBuffer);
        reader.readAsArrayBuffer(file);
      } else if (file.type === 'application/epub+zip' || file.name.endsWith('.epub')) {
        const reader = new FileReader();
        reader.onload = (ev) => loadEPUB(ev.target?.result as ArrayBuffer);
        reader.readAsArrayBuffer(file);
      } else {
        alert("Please drop a valid PDF or EPUB file");
      }
    }
  };

  const loadText = (text: string) => {
    const newSentences: any[] = [];
    const contentData: any[] = [];
    let globalLineIdCounter = 0;
    const sentenceRegex = /[^.!?]+[.!?]/g;
    let match;
    while ((match = sentenceRegex.exec(text)) !== null) {
      const sText = match[0].trim().replace(/\s+/g, ' ');
      if (sText.length > 0) {
        const lineId = globalLineIdCounter++;
        newSentences.push({ text: sText, lines: [lineId] });
        contentData.push({ type: 'text', id: lineId, text: sText });
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
      // Permission denied or API unavailable — fall back to textarea
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

          const text = doc.body.innerText;
          if (!text || !text.trim()) continue;

          const sentenceRegex = /[^.!?]+[.!?]/g;
          let match;
          while ((match = sentenceRegex.exec(text)) !== null) {
            const sText = match[0].trim().replace(/\s+/g, ' ');
            if (chapterTitle && sText === chapterTitle) continue;
            if (sText.length > 0) {
              const lineId = globalLineIdCounter++;
              newSentences.push({ text: sText, lines: [lineId] });
              contentData.push({ type: 'text', id: lineId, text: sText });
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
        // Using a good scale for clarity
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
    // Process items and transform to Viewport coordinates (Pixel space)
    const items = textContent.items.map((item: any) => {
      const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
      // tx[5] is usually the baseline in top-left origin system
      const fontHeight = Math.sqrt((tx[2] * tx[2]) + (tx[3] * tx[3]));
      return {
        str: item.str,
        x: tx[4],
        y: tx[5],
        width: item.width * scale,
        height: item.height > 0 ? item.height * scale : fontHeight,
      };
    });

    // Sort items to rebuild lines (Top to Bottom, Left to Right)
    items.sort((a: any, b: any) => Math.abs(a.y - b.y) > a.height * 0.2 ? a.y - b.y : a.x - b.x);

    const lines: any[] = [];
    let currentLine: any = null;

    // Group items into lines
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

    // Convert line data to Percentage (%) coordinates relative to the viewport
    return lines.map((line: any, idx: number) => {
      const minX = Math.min(...line.items.map((i: any) => i.x));
      const last = line.items[line.items.length - 1];
      const maxX = last.x + (last.width || last.str.length * 5);
      const width = maxX - minX;

      // Calculate positions as percentages
      const leftPct = (minX / viewport.width) * 100;
      // y is baseline. To box it, we move up by height.
      const topPct = ((line.y - line.height) / viewport.height) * 100;
      const widthPct = (width / viewport.width) * 100;
      const heightPct = (line.height / viewport.height) * 100;

      return {
        id: startIndex + idx,
        text: line.items.map((i: any) => i.str).join(' '),
        // Store percentage strings for CSS
        left: `${leftPct}%`,
        top: `${topPct}%`,
        width: `${widthPct}%`,
        height: `${heightPct}%`,
        // Store raw startPos for sentence mapping logic later (not used for display)
        startPos: 0
      };
    });
  };

  const getStatusBadgeStyle = () => {
    if (ttsStatus === "Downloading...") return { ...styles.statusBadgeMenu, ...styles.statusLoading };
    if (ttsStatus === "Model Ready") return { ...styles.statusBadgeMenu, ...styles.statusReady };
    if (ttsStatus === "System Voice") return { ...styles.statusBadgeMenu, ...styles.statusFallback };
    return { ...styles.statusBadgeMenu, ...styles.statusLoading };
  };

  return (
    <div style={styles.container}>

      {sentences.length === 0 ? (
        <div style={{ ...styles.dropZone, ...(isDragOver ? styles.dropZoneHover : {}) }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleFileDrop}
        >
          <Upload size={32} color="#9ca3af" style={{ marginBottom: '1rem' }} />
          <p style={{ fontSize: '1.1rem', fontWeight: 600, color: '#4b5563', marginBottom: '0.5rem' }}>
            Drop PDF or EPUB here
          </p>
          <p style={{ fontSize: '0.85rem', color: '#6b7280' }}>
            or tap to browse files
          </p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileDrop}
            accept="application/pdf,.epub"
            style={{ display: 'none' }}
          />
          <div style={{ width: '100%', borderTop: '1px solid #e5e7eb', margin: '1rem 0' }} />
          <button
            onClick={handleClipboardPaste}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
              padding: '0.5rem 1rem', fontSize: '0.875rem', fontWeight: 600,
              backgroundColor: '#eff6ff', color: '#2563eb', border: '1px solid #bfdbfe',
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
                  borderRadius: '0.375rem', border: '1px solid #d1d5db',
                  fontSize: '0.875rem', color: '#374151', boxSizing: 'border-box' as const,
                  resize: 'vertical',
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
      ) : (
        <div style={styles.viewer}>
          {fileType === 'pdf' && pages.map(pageData => (
            <PDFPage key={pageData.pageNumber} data={pageData} highlightedLineIds={highlightedLineIds} onLineClick={handleLineClick} />
          ))}

          {(fileType === 'epub' || fileType === 'text') && (
            <div style={styles.epubContainer}>
              {epubContent.map((item) => {
                if (item.type === 'header') return <div key={item.id} style={{fontSize: '1.5rem', fontWeight: 'bold', margin: '2rem 0 1rem 0', color: '#1e3a8a'}}>{item.text}</div>;
                const isActive = highlightedLineIds.includes(item.id);
                return (
                  <span key={item.id} id={`line-${item.id}`} onClick={() => handleLineClick(item.id)} style={{...styles.epubSentence, ...(isActive ? styles.epubHighlight : {})}}>
                      {item.text}{' '}
                  </span>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Bottom Bar: Logo Left, Controls Right */}
      <div style={styles.bottomBar}>
        <div style={styles.logoGroup}>
          <img src="./logo.png" style={{width: "36px", height: "36px"}} alt="logo" />
          <h1 style={styles.title}>BlaBla</h1>
        </div>

        <div style={styles.controlsGroup}>
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
              ...styles.playButton,
              ...(sentences.length === 0 || !isModelReady ? styles.buttonDisabled : {})
            }}
          >
            {isPlaying ? <Pause size={20} fill="white" /> : <Play size={20} fill="white" style={{marginLeft:'2px'}} />}
          </button>

          <button onClick={() => setIsMenuOpen(!isMenuOpen)} style={styles.iconButton}>
            <Menu size={24} />
          </button>
        </div>

        {isSpeedMenuOpen && (
          <div style={{...styles.menuPopover, width: '100px', right: '4rem', bottom: '3.5rem'}}>
            {[1.0, 1.25, 1.5, 2.0].map(speed => (
              <button
                key={speed}
                onClick={() => { setPlaybackSpeed(speed); setIsSpeedMenuOpen(false); }}
                style={{
                  ...styles.statItem,
                  width: '100%', cursor: 'pointer', backgroundColor: playbackSpeed === speed ? '#eff6ff' : 'transparent', border: 'none'
                }}
              >
                {speed}x
              </button>
            ))}
          </div>
        )}

        {isMenuOpen && (
          <div style={styles.menuPopover}>
            {/* Status Badge moved here */}
            <div style={getStatusBadgeStyle()}>
              {ttsStatus === "Loading..." && <Loader2 size={14} className="animate-spin" />}
              {ttsStatus === "Model Ready" && <Activity size={14} />}
              {ttsStatus === "System Voice" && <AlertCircle size={14} />}
              {ttsStatus}
            </div>

            {/* Voice Selector moved here */}
            <label style={styles.selectLabel}>Select Voice</label>
            <div style={{position:'relative'}}>
              <select
                value={selectedVoice}
                onChange={handleVoiceChange}
                disabled={usingFallback.current || !isModelReady}
                style={{...styles.select, ...(usingFallback.current || !isModelReady ? styles.buttonDisabled : {})}}
              >
                {VOICES.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
              </select>
              <ChevronDown size={14} style={{position:'absolute', right:'10px', top:'12px', pointerEvents:'none', color:'#9ca3af'}}/>
            </div>

            {/* Stats */}
            <div style={{marginTop: '0.5rem'}}>
              <div style={styles.statItem}><span style={styles.statLabel}>State</span><span style={styles.statValue}>{playbackState}</span></div>
              <div style={styles.statItem}><span style={styles.statLabel}>Progress</span><span style={styles.statValue}>{currentSentenceIndex >= 0 ? `${Math.round((currentSentenceIndex/sentences.length)*100)}%` : '0%'}</span></div>
            </div>

            {/* Actions */}
            <button onClick={handleTestAudio} style={{...styles.testButton, ...((!isModelReady) ? styles.buttonDisabled : {})}} disabled={!isModelReady}>
              <Beaker size={14} /> Test Voice
            </button>
            <button onClick={resetReader} style={styles.resetButton}>Reset Document</button>
          </div>
        )}
      </div>
    </div>
  );
}
