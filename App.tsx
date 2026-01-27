import React, { useState, useEffect, useRef } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import ePub from 'epubjs';
import { Play, Pause, Upload, Loader2, FileText, Beaker, AlertCircle, Activity, Menu, BookOpen } from 'lucide-react';
import logo from './logo.png';

// Set worker source
pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs';

const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    backgroundColor: 'white',
    fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
    color: '#333',
  },
  header: {
    width: '100%',
    backgroundColor: '#fff',
    padding: '1rem',
    position: 'sticky' as const,
    top: 0,
    zIndex: 50,
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  },
  logoGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  },
  title: {
    fontSize: '1.25rem',
    fontWeight: 'bold' as const,
    color: 'blue',
    margin: 0,
    fontFamily: 'Arial',
  },
  statusGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  statusBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.75rem',
    padding: '0.35rem 0.65rem',
    borderRadius: '0.375rem',
    fontWeight: '600' as const,
  },
  statusLoading: {
    color: '#2563eb',
    backgroundColor: '#eff6ff',
    border: '1px solid #bfdbfe',
  },
  statusReady: {
    color: '#059669',
    backgroundColor: '#d1fae5',
    border: '1px solid #6ee7b7',
  },
  statusFallback: {
    color: '#d97706',
    backgroundColor: '#fef3c7',
    border: '1px solid #fde68a',
  },
  statItem: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    fontSize: '0.7rem',
    color: '#6b7280',
    padding: '0.25rem 0.5rem',
    borderLeft: '1px solid #e5e7eb',
  },
  statLabel: {
    fontSize: '0.65rem',
    color: '#9ca3af',
    marginBottom: '0.1rem',
  },
  statValue: {
    fontSize: '0.75rem',
    fontWeight: '600' as const,
    color: '#374151',
    fontFamily: 'monospace',
  },
  controls: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  playerGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    backgroundColor: '#f3f4f6',
    padding: '0.25rem',
    borderRadius: '0.5rem',
  },
  iconButton: {
    padding: '0.5rem',
    borderRadius: '0.25rem',
    border: 'none',
    cursor: 'pointer',
    backgroundColor: 'transparent',
    color: '#374151',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.2s',
  },
  testButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.5rem 1rem',
    fontSize: '0.875rem',
    fontWeight: '600' as const,
    backgroundColor: '#dbeafe',
    color: '#1e40af',
    border: 'none',
    borderRadius: '0.375rem',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  resetButton: {
    padding: '0.5rem 1rem',
    fontSize: '0.875rem',
    backgroundColor: '#e5e7eb',
    color: '#374151',
    border: 'none',
    borderRadius: '0.375rem',
    cursor: 'pointer',
  },
  dropZone: {
    marginTop: '2.5rem',
    width: '100%',
    maxWidth: '42rem',
    height: '16rem',
    border: '4px dashed #d1d5db',
    borderRadius: '0.75rem',
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    backgroundColor: '#fff',
    transition: 'all 0.3s ease',
  },
  dropZoneHover: {
    borderColor: '#3b82f6',
    backgroundColor: '#eff6ff',
    transform: 'scale(1.01)',
  },
  viewer: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    width: '100%',
    maxWidth: '64rem',
    padding: '2rem',
    paddingBottom: '10rem',
  },
  pageContainer: {
    position: 'relative' as const,
    marginBottom: '20px',
    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)',
    backgroundColor: '#fff',
    display: 'inline-block',
  },
  canvas: {
    display: 'block',
  },
  overlay: {
    position: 'absolute' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 10,
  },
  lineBase: {
    position: 'absolute' as const,
    cursor: 'pointer',
    borderRadius: '2px',
    backgroundColor: 'transparent',
    transition: 'background-color 0.3s ease, box-shadow 0.3s ease',
  },
  lineActive: {
    backgroundColor: 'rgb(70 181 220 / 50%)',
    boxShadow: '0 0 0 2px rgb(70 181 220 / 60%)',
    mixBlendMode: 'multiply' as const,
    zIndex: 20,
  },
  bottomBar: {
    position: 'fixed' as const,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#fff',
    padding: '0.5rem',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '1rem',
    boxShadow: '0 -1px 3px rgba(0,0,0,0.1)',
    zIndex: 50,
    flexWrap: 'wrap' as const,
  },
  menuPopover: {
    position: 'absolute' as const,
    bottom: '100%',
    backgroundColor: '#fff',
    padding: '1rem',
    borderRadius: '0.375rem',
    boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.5rem',
  },
  select: {
    padding: '0.5rem',
    borderRadius: '0.375rem',
    border: '1px solid #d1d5db',
    backgroundColor: '#f3f4f6',
    color: '#374151',
    cursor: 'pointer',
  },
  // EPUB SPECIFIC STYLES
  epubContainer: {
    width: '100%',
    maxWidth: '48rem',
    padding: '2rem',
    paddingBottom: '10rem',
    lineHeight: '1.8',
    fontSize: '1.125rem',
    textAlign: 'left' as const,
  },
  epubChapterTitle: {
    display: 'block',
    fontSize: '1.5rem',
    fontWeight: '700',
    marginTop: '2.5rem',
    marginBottom: '1rem',
    color: '#1e3a8a',
    borderBottom: '2px solid #e5e7eb',
    paddingBottom: '0.5rem',
  },
  epubSentence: {
    cursor: 'pointer',
    padding: '2px 0',
    transition: 'background-color 0.2s',
    borderRadius: '4px',
  },
  epubHighlight: {
    backgroundColor: '#bae6fd',
    boxShadow: '0 0 0 2px #bae6fd',
  }
};

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
  const [fileType, setFileType] = useState<'pdf' | 'epub' | null>(null);
  const [epubContent, setEpubContent] = useState<any[]>([]);

  const [ttsStatus, setTtsStatus] = useState("Init");
  const [isModelReady, setIsModelReady] = useState(false);
  const [cachedCount, setCachedCount] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);
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

  const voices = [
    { value: 'af_bella', label: 'af_bella (F)' },
    { value: 'af_heart', label: 'af_heart (F)' },
    { value: 'am_fenrir', label: 'am_fenrir (M)' },
    { value: 'am_puck', label: 'am_puck (M)' },
  ];

  useEffect(() => {
    console.log("[App] Mounting...");
    const ctx = getAudioContext();
    ctx.resume().catch(console.warn);
    workerRef.current = new Worker("/tts.worker.js", { type: 'module' });
    workerRef.current.onmessage = (e) => {
      const { status, audio, error, text, lineIndex } = e.data;
      if (status === 'ready') {
        setTtsStatus("Model Ready");
        setIsModelReady(true);
        console.log("[Worker] Model Loaded");
      }
      else if (status === 'complete') {
        console.log(`[Worker] Audio generated. Size: ${audio.length} for line: ${lineIndex}`);
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
      console.log("[App] Unmounting...");
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

  const triggerFallback = () => {
    if (usingFallback.current) return;
    console.warn("[TTS] Switching to System Voice Fallback");
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
      if (audioContext.current.state !== 'running') {
        const oscillator = audioContext.current.createOscillator();
        oscillator.connect(audioContext.current.destination);
        oscillator.start();
        oscillator.stop(audioContext.current.currentTime + 0.001);
      }
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

  const updateBufferUI = () => {
    setCachedCount(audioCache.current.size);
    setPendingCount(pendingFetches.current.size);
  };

  const processSentenceAudio = async (index: number, sessionId: number, voice: string): Promise<AudioBuffer | null> => {
    if (usingFallback.current) return null;
    if (index >= sentences.length) return null;
    if (audioCache.current.has(index)) return audioCache.current.get(index);
    if (pendingFetches.current.has(index)) return null;

    const text = sentences[index].text;
    if (!text.trim()) return null;

    pendingFetches.current.add(index);
    updateBufferUI();
    try {
      const buffer = await generateAudioInWorker(text, index, voice);
      if (!buffer) return null;

      if (sessionId === playbackSessionId.current) {
        audioCache.current.set(index, buffer);
        updateBufferUI();
      }
      return buffer;
    } catch (err) {
      console.error(`[Buffer] Failed to process sentence ${index}`, err);
      return null;
    } finally {
      pendingFetches.current.delete(index);
      updateBufferUI();
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

    // Auto-scroll logic for both PDF (div overlay) and EPUB (span)
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

    // Prefetch next 3
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
    updateBufferUI();
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
    const text = "Hello! I am Kokoro.";
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
    // Reset file types
    setFileType(null);
    setEpubContent([]);

    setCurrentSentenceIndex(-1);
    setHighlightedLineIds([]);
    audioCache.current.clear();
    pendingFetches.current.clear();
    audioResolvers.current.clear();
    updateBufferUI();
    setPlaybackState("Idle");
    isWaitingForAudio.current = false;
  };

  const handleVoiceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newVoice = e.target.value;
    setSelectedVoice(newVoice);
    audioCache.current.clear();
    pendingFetches.current.clear();
    audioResolvers.current.clear();
    updateBufferUI();
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

  const loadEPUB = async (data: ArrayBuffer) => {
    try {
      setFileType('epub');
      const book = ePub(data);
      await book.ready;

      // Ensure locations are generated (helps with nav)
      // await book.locations.generate(1000);

      const spine = book.spine;
      const newSentences: any[] = [];
      const contentData: any[] = [];
      let globalLineIdCounter = 0;

      // Access spine items directly
      const items = (spine as any).items || [];

      // Wait for navigation to load to map hrefs to titles
      const navigation = await book.loaded.navigation;

      for (const item of items) {
        try {
          // Load document to extract raw text
          const doc = await book.load(item.href);
          if (!doc) continue;

          // Attempt to find a Chapter Title
          let chapterTitle = null;

          // Strategy 1: Check if this spine item matches a TOC entry
          // book.navigation is an Object where keys are sometimes IDs or HREFs.
          // It's safer to traverse the TOC array if available.
          const findTitleInToc = (toc: any[], href: string): string | null => {
            for (const entry of toc) {
              if (href.includes(entry.href)) return entry.label;
              if (entry.subitems) {
                const sub = findTitleInToc(entry.subitems, href);
                if (sub) return sub;
              }
            }
            return null;
          };

          if (navigation && (navigation as any).toc) {
            chapterTitle = findTitleInToc((navigation as any).toc, item.href);
          }

          // Strategy 2: Fallback to the first <h1> or <h2> in the document itself
          if (!chapterTitle) {
            const h1 = doc.querySelector('h1');
            if (h1 && h1.innerText.trim().length > 0) {
               chapterTitle = h1.innerText.trim();
            } else {
               const h2 = doc.querySelector('h2');
               if (h2 && h2.innerText.trim().length > 0) {
                 chapterTitle = h2.innerText.trim();
               }
            }
          }

          // If we found a title, push a HEADER block
          if (chapterTitle) {
            contentData.push({
              type: 'header',
              id: `header-${globalLineIdCounter}`, // Just a unique key
              text: chapterTitle
            });
          }

          // Get text content - simplified extraction
          const text = doc.body.innerText;
          if (!text || !text.trim()) continue;

          const sentenceRegex = /[^.!?]+[.!?]/g;
          let match;

          while ((match = sentenceRegex.exec(text)) !== null) {
            const sText = match[0].trim().replace(/\s+/g, ' ');

            // Avoid adding the chapter title again if it was just read as text
            if (chapterTitle && sText === chapterTitle) continue;

            if (sText.length > 0) {
              const lineId = globalLineIdCounter++;
              newSentences.push({
                text: sText,
                lines: [lineId]
              });
              contentData.push({
                type: 'text',
                id: lineId,
                text: sText
              });
            }
          }
        } catch (err) {
          console.error("Error loading chapter", err);
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
        globalLineList = [...globalLineList, ...lines];
        newPages.push({ page, viewport, lines, pageNumber: i });
      }

      const fullText = globalLineList.map(l => l.text).join(' ');
      const sentenceRegex = /[^.!?]+[.!?]/g;
      let match;
      const sentenceTexts = [];
      while ((match = sentenceRegex.exec(fullText)) !== null) {
        sentenceTexts.push({
          text: match[0].trim(),
          start: match.index,
          end: match.index + match[0].length
        });
      }
      if (fullText.slice(sentenceTexts[sentenceTexts.length - 1]?.end || 0).trim()) {
        sentenceTexts.push({
          text: fullText.slice(sentenceTexts[sentenceTexts.length - 1]?.end || 0).trim(),
          start: sentenceTexts[sentenceTexts.length - 1]?.end || 0,
          end: fullText.length
        });
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

      // Split long sentences
      const maxTextLength = 1000;
      const finalSentences = [];
      for (let sent of newSentences) {
        if (sent.text.length <= maxTextLength) {
          finalSentences.push(sent);
        } else {
          const subTexts = [];
          let remaining = sent.text;
          while (remaining.length > maxTextLength) {
            let splitPos = remaining.lastIndexOf('.', maxTextLength);
            if (splitPos === -1) splitPos = remaining.lastIndexOf(',', maxTextLength);
            if (splitPos === -1) splitPos = remaining.lastIndexOf(' ', maxTextLength);
            if (splitPos === -1) splitPos = maxTextLength;
            subTexts.push(remaining.slice(0, splitPos + (remaining[splitPos] === '.' ? 1 : 0)).trim());
            remaining = remaining.slice(splitPos + (remaining[splitPos] === '.' ? 1 : 0)).trim();
          }
          if (remaining) subTexts.push(remaining);
          subTexts.forEach(subText => {
            finalSentences.push({ text: subText, lines: sent.lines });
          });
        }
      }
      setSentences(finalSentences);
      setAllLines(globalLineList);
      setPages(newPages);
    } catch (e) {
      console.error(e);
    }
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
      if (!currentLine) currentLine = { items: [item], y: item.y, height: item.height };
      else if (Math.abs(item.y - currentLine.y) < currentLine.height * 0.5) currentLine.items.push(item);
      else { lines.push(currentLine); currentLine = { items: [item], y: item.y, height: item.height }; }
    });
    if (currentLine) lines.push(currentLine);
    return lines.map((line: any, idx: number) => {
      const minX = Math.min(...line.items.map((i: any) => i.x));
      const last = line.items[line.items.length - 1];
      const width = (last.x + (last.width || last.str.length * 5)) - minX;
      return {
        id: startIndex + idx,
        text: line.items.map((i: any) => i.str).join(' '),
        x: minX,
        y: line.y - line.height * 0.85,
        width,
        height: line.height
      };
    });
  };

  const PDFPage = ({ data, highlightedLineIds, onLineClick }: any) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const renderTaskRef = useRef<any>(null);
    useEffect(() => {
      if (canvasRef.current && data) {
        if (renderTaskRef.current) { try { renderTaskRef.current.cancel(); } catch (e) { } }
        const ctx = canvasRef.current.getContext('2d');
        const renderTask = data.page.render({ canvasContext: ctx, viewport: data.viewport });
        renderTaskRef.current = renderTask;
        renderTask.promise.catch((e: any) => { if (e.name !== 'RenderingCancelledException') console.error(e) });
      }
    }, [data]);
    return (
      <div style={{ ...styles.pageContainer, width: data.viewport.width, height: data.viewport.height }}>
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
                left: line.x,
                top: line.y,
                width: line.width,
                height: line.height,
              }}
            />
          ))}
        </div>
      </div>
    );
  };

  const getStatusBadgeStyle = () => {
    if (ttsStatus === "Downloading...") return { ...styles.statusBadge, ...styles.statusLoading };
    if (ttsStatus === "Model Ready") return { ...styles.statusBadge, ...styles.statusReady };
    if (ttsStatus === "System Voice") return { ...styles.statusBadge, ...styles.statusFallback };
    return { ...styles.statusBadge, ...styles.statusLoading };
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.logoGroup}>
          <img src={logo} style={{width: "36px", height: "36px"}} />
          <h1 style={styles.title}>BlaBla</h1>
        </div>
        <div style={styles.controls}>
        </div>
      </div>

      {sentences.length === 0 ? (
        <div style={{ ...styles.dropZone, ...(isDragOver ? styles.dropZoneHover : {}) }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleFileDrop}
        >
          <Upload size={48} color="#9ca3af" style={{ marginBottom: '1rem' }} />
          <p style={{ fontSize: '1.25rem', fontWeight: 600, color: '#4b5563' }}>Drop PDF or EPUB here</p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileDrop}
            accept="application/pdf,.epub"
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div style={styles.viewer}>
          {fileType === 'pdf' && pages.map(pageData => (
            <PDFPage key={pageData.pageNumber} data={pageData} highlightedLineIds={highlightedLineIds} onLineClick={handleLineClick} />
          ))}

          {fileType === 'epub' && (
            <div style={styles.epubContainer}>
              {epubContent.map((item) => {
                if (item.type === 'header') {
                  return (
                    <div key={item.id} style={styles.epubChapterTitle}>
                      {item.text}
                    </div>
                  );
                }

                const isActive = highlightedLineIds.includes(item.id);
                return (
                  <span
                    key={item.id}
                    id={`line-${item.id}`}
                    onClick={() => handleLineClick(item.id)}
                    style={{
                        ...styles.epubSentence,
                        ...(isActive ? styles.epubHighlight : {})
                    }}
                  >
                      {item.text}{' '}
                  </span>
                )
              })}
            </div>
          )}
        </div>
      )}

      <div style={styles.bottomBar}>
        <div style={getStatusBadgeStyle()}>
          {ttsStatus === "Loading..." && <Loader2 size={12} />}
          {ttsStatus === "Ready" && <Activity size={12} />}
          {ttsStatus === "System Voice" && <AlertCircle size={12} />}
          {ttsStatus}
        </div>
        <div style={styles.playerGroup}>
          <button
            onClick={togglePlay}
            disabled={sentences.length === 0 || !isModelReady}
            style={{
              ...styles.iconButton,
              ...(sentences.length === 0 || !isModelReady ? styles.buttonDisabled : {})
            }}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
        </div>
        <button
          onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)}
          style={{
            ...styles.iconButton,
            fontSize: '0.875rem',
            fontWeight: '600',
          }}
        >
          {playbackSpeed}x
        </button>
        {isSpeedMenuOpen && (
          <div style={styles.menuPopover}>
            {[1.0, 1.2, 1.3, 1.5, 1.7, 2.0].map(speed => (
              <button
                key={speed}
                onClick={() => {
                  setPlaybackSpeed(speed);
                  setIsSpeedMenuOpen(false);
                }}
                style={{
                  ...styles.resetButton,
                  backgroundColor: playbackSpeed === speed ? '#dbeafe' : '#e5e7eb',
                  color: playbackSpeed === speed ? '#1e40af' : '#374151',
                }}
              >
                {speed}x
              </button>
            ))}
          </div>
        )}
        <select
          value={selectedVoice}
          onChange={handleVoiceChange}
          disabled={usingFallback.current || !isModelReady}
          style={{
            ...styles.select,
            ...(usingFallback.current || !isModelReady ? styles.buttonDisabled : {}),
          }}
        >
          {voices.map(v => (
            <option key={v.value} value={v.value}>
              {v.label}
            </option>
          ))}
        </select>
        <button onClick={() => setIsMenuOpen(!isMenuOpen)} style={styles.iconButton}>
          <Menu size={20} />
        </button>
        {isMenuOpen && (
          <div style={styles.menuPopover}>
            <button onClick={resetReader} style={styles.resetButton}>Reset</button>
            <button
              onClick={handleTestAudio}
              style={{
                ...styles.testButton,
                ...((!isModelReady) ? styles.buttonDisabled : {})
              }}
              disabled={!isModelReady}
            >
              <Beaker size={16} /> Test Voice
            </button>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Status</span>
              <span style={styles.statValue}>{playbackState}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Cached</span>
              <span style={styles.statValue}>{cachedCount}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Queue</span>
              <span style={styles.statValue}>{pendingCount}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Sentence</span>
              <span style={styles.statValue}>{currentSentenceIndex >= 0 ? `${currentSentenceIndex + 1}/${sentences.length}` : '-'}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
