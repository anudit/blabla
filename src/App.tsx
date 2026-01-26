import React, { useState, useEffect, useRef } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { Play, Pause, Upload, Loader2, FileText, Beaker, AlertCircle, Activity } from 'lucide-react';
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.mjs`;
const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
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
    color: '#1f2937',
    margin: 0,
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
    backgroundColor: 'rgb(0 183 255 / 50%)',
    boxShadow: '0 0 0 2px rgba(0, 183, 255, 0.6)',
    mixBlendMode: 'multiply' as const,
    zIndex: 20,
  },
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
  const [ttsStatus, setTtsStatus] = useState("Init");
  const [isModelReady, setIsModelReady] = useState(false);
  const [cachedCount, setCachedCount] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);
  const [playbackState, setPlaybackState] = useState("Idle");
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
  useEffect(() => {
    console.log("[App] Mounting...");
    const ctx = getAudioContext();
      ctx.resume().catch(console.warn);
    workerRef.current = new Worker("/tts.worker.js", { type: 'module' });
    workerRef.current.onmessage = (e) => {
      const { status, audio, error, text, lineIndex } = e.data;
      if (status === 'ready') {
        setTtsStatus("AI Ready");
        setIsModelReady(true);
        console.log("[Worker] Model Loaded");
      }
      else if (status === 'complete') {
        console.log(`[Worker] Audio generated. Size: ${audio.length} for line: ${lineIndex}`);
        try {
          const ctx = getAudioContext();
          const buffer = ctx.createBuffer(1, audio.length, 24000);
          buffer.getChannelData(0).set(audio);
          console.log(`Created buffer — sampleRate: ${buffer.sampleRate}, duration: ${(audio.length / buffer.sampleRate).toFixed(2)}s`);
          if (text.startsWith("Hello")) {
            playBufferDirectly(buffer);
          } else if (lineIndex !== undefined) {
            const resolver = audioResolvers.current.get(lineIndex);
            if (resolver) {
              console.log(`[Audio] Resolving buffer for line ${lineIndex}`);
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
    setTtsStatus("Loading...");
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
      console.log('[AudioCtx] Recreating closed context');
      audioContext.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 24000
      });
    }
    // Aggressive resume pattern – browsers are super strict in 2024/2025
    if (audioContext.current.state !== 'running') {
      console.log(`[AudioCtx] Current state = ${audioContext.current.state} → forcing resume`);
      // 1. Classic resume
      audioContext.current.resume().catch(e => console.warn("[AudioCtx] resume() failed", e));
      // 2. Some browsers need this dummy unlock
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
    source.connect(ctx.destination);
    source.start(0);
    setPlaybackState("Playing");
    source.onended = () => setPlaybackState("Ready");
  };
  const generateAudioInWorker = (text: string, sentenceIndex: number): Promise<AudioBuffer | null> => {
    return new Promise((resolve) => {
      if (usingFallback.current || !workerRef.current) {
        resolve(null);
        return;
      }
      console.log(`[Generate] Requesting audio for sentence ${sentenceIndex}: "${text.substring(0, 30)}..."`);
      audioResolvers.current.set(sentenceIndex, resolve);
      workerRef.current.postMessage({ type: 'generate', text, lineIndex: sentenceIndex });
      setTimeout(() => {
        if (audioResolvers.current.has(sentenceIndex)) {
          console.log(`[Generate] Timeout for sentence ${sentenceIndex}`);
          audioResolvers.current.delete(sentenceIndex);
          resolve(null);
        }
      }, 30000);
    });
  };
  const stopAllAudio = () => {
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e){}
      currentSource.current = null;
    }
    window.speechSynthesis.cancel();
    if (nativeTimeout.current) clearTimeout(nativeTimeout.current);
    if (audioContext.current) audioContext.current.suspend();
    setPlaybackState("Stopped");
    isWaitingForAudio.current = false;
  };
  const updateBufferUI = () => {
    setCachedCount(audioCache.current.size);
    setPendingCount(pendingFetches.current.size);
  };
  const processSentenceAudio = async (index: number, sessionId: number): Promise<AudioBuffer | null> => {
    if (usingFallback.current) return null;
    if (index >= sentences.length) return null;
    if (audioCache.current.has(index)) {
      console.log(`[Buffer] Cache hit for sentence ${index}`);
      return audioCache.current.get(index);
    }
    if (pendingFetches.current.has(index)) {
      console.log(`[Buffer] Already fetching sentence ${index}`);
      return null;
    }
    const text = sentences[index].text;
    if (!text.trim()) return null;
    console.log(`[Buffer] Starting fetch for sentence ${index}`);
    pendingFetches.current.add(index);
    updateBufferUI();
    try {
      const buffer = await generateAudioInWorker(text, index);
      if (!buffer) {
        console.log(`[Buffer] No buffer returned for sentence ${index}`);
        return null;
      }
      if (sessionId === playbackSessionId.current) {
        console.log(`[Buffer] Caching buffer for sentence ${index}`);
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
    console.log(`[Play] Playing sentence ${currentSentenceIndex}: "${text.substring(0, 30)}..."`);
    setHighlightedLineIds(unit.lines);
    if (usingFallback.current) {
      window.speechSynthesis.cancel();
      if (nativeTimeout.current) clearTimeout(nativeTimeout.current);
      nativeTimeout.current = setTimeout(() => {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.1;
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
    for (let i = 1; i <= 3; i++) {
      if (currentSentenceIndex + i < sentences.length && !audioCache.current.has(currentSentenceIndex + i)) {
        console.log(`[Lookahead] Prefetching sentence ${currentSentenceIndex + i}`);
        processSentenceAudio(currentSentenceIndex + i, currentSession);
      }
    }
    let buffer = audioCache.current.get(currentSentenceIndex);
    if (!buffer) {
      console.log(`[Play] Buffer not cached, fetching sentence ${currentSentenceIndex}`);
      setPlaybackState("Buffering");
      isWaitingForAudio.current = true;
      buffer = await processSentenceAudio(currentSentenceIndex, currentSession);
      isWaitingForAudio.current = false;
    }
    if (usingFallback.current) {
      playCurrentSentence();
      return;
    }
    if (currentSession !== playbackSessionId.current || !isPlaying) {
      console.log(`[Play] Session mismatch or stopped playing`);
      return;
    }
    if (buffer) {
      console.log(`[Play] Playing buffer for sentence ${currentSentenceIndex}`);
      const source = audioContext.current!.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.current!.destination);
      source.onended = () => {
        console.log(`[Play] Audio ended for sentence ${currentSentenceIndex}`);
        if (isPlaying && currentSession === playbackSessionId.current) {
          advanceSentence();
        }
      };
      currentSource.current = source;
      setPlaybackState("Playing");
      source.start(0);
    } else {
      console.log(`[Play] No buffer available, triggering fallback`);
      triggerFallback();
      playCurrentSentence();
    }
  };
  const advanceSentence = () => {
    console.log(`[Advance] Moving from sentence ${currentSentenceIndex} to ${currentSentenceIndex + 1}`);
    setCurrentSentenceIndex(prev => {
      const nextIndex = prev + 1;
      setTimeout(() => {
        if (nextIndex < sentences.length) {
          const firstLineId = sentences[nextIndex].lines[0];
          const el = document.getElementById(`line-${firstLineId}`);
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 50);
      return nextIndex;
    });
  };
  const handleLineClick = (lineId: number) => {
    console.log(`[Click] Line ${lineId} clicked`);
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e){}
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
      window.speechSynthesis.speak(u);
      setPlaybackState("Testing");
    } else {
      setPlaybackState("Generating");
      workerRef.current?.postMessage({ type: 'generate', text });
    }
  };
  const resetReader = () => {
    setIsPlaying(false);
    stopAllAudio();
    setPdfDoc(null);
    setPages([]);
    setAllLines([]);
    setSentences([]);
    setCurrentSentenceIndex(-1);
    setHighlightedLineIds([]);
    audioCache.current.clear();
    pendingFetches.current.clear();
    audioResolvers.current.clear();
    updateBufferUI();
    setPlaybackState("Idle");
    isWaitingForAudio.current = false;
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
    if (file && file.type === 'application/pdf') {
      const reader = new FileReader();
      reader.onload = (ev) => loadPDF(ev.target?.result as ArrayBuffer);
      reader.readAsArrayBuffer(file);
    } else {
      alert("Please drop a valid PDF");
    }
  };
  const loadPDF = async (data: ArrayBuffer) => {
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
      // Build sentences properly
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
      // Map to lines
      let cumPos = 0;
      globalLineList.forEach(line => {
        line.startPos = cumPos;
        cumPos += line.text.length + 1; // + space
      });
      const newSentences = sentenceTexts.map(sent => {
        const sentLines = globalLineList.filter(line =>
          line.startPos < sent.end && (line.startPos + line.text.length + 1) > sent.start
        ).map(line => line.id);
        return { text: sent.text, lines: sentLines };
      }).filter(sent => sent.text && sent.lines.length > 0);
      // Split long sentences
      const maxTextLength = 1000; // Adjust based on model limits ~510 tokens ~2000 chars, but safe
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
            finalSentences.push({ text: subText, lines: sent.lines }); // Approximate same lines
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
    if (ttsStatus === "Loading...") return { ...styles.statusBadge, ...styles.statusLoading };
    if (ttsStatus === "AI Ready") return { ...styles.statusBadge, ...styles.statusReady };
    if (ttsStatus === "System Voice") return { ...styles.statusBadge, ...styles.statusFallback };
    return { ...styles.statusBadge, ...styles.statusLoading };
  };
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.logoGroup}>
          <FileText size={24} color="#2563eb" />
          <h1 style={styles.title}>PDF Reader</h1>
          <div style={styles.statusGroup}>
            <div style={getStatusBadgeStyle()}>
              {ttsStatus === "Loading..." && <Loader2 size={12} />}
              {ttsStatus === "AI Ready" && <Activity size={12} />}
              {ttsStatus === "System Voice" && <AlertCircle size={12} />}
              {ttsStatus}
            </div>
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
        </div>
        <div style={styles.controls}>
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
          <button onClick={resetReader} style={styles.resetButton}>Reset</button>
        </div>
      </div>
      {allLines.length === 0 ? (
        <div style={{ ...styles.dropZone, ...(isDragOver ? styles.dropZoneHover : {}) }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleFileDrop}
        >
          <Upload size={48} color="#9ca3af" style={{ marginBottom: '1rem' }} />
          <p style={{ fontSize: '1.25rem', fontWeight: 600, color: '#4b5563' }}>Drop PDF here</p>
          <input type="file" ref={fileInputRef} onChange={handleFileDrop} accept="application/pdf" style={{ display: 'none' }} />
        </div>
      ) : (
        <div style={styles.viewer}>
          {pages.map(pageData => (
            <PDFPage key={pageData.pageNumber} data={pageData} highlightedLineIds={highlightedLineIds} onLineClick={handleLineClick} />
          ))}
        </div>
      )}
    </div>
  );
}
