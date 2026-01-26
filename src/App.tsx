import React, { useState, useEffect, useRef } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { Play, Pause, Upload, Loader2, FileText, Beaker, AlertCircle } from 'lucide-react';

// --- Configuration ---
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.mjs`;

// --- Style Objects ---
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
    position: 'sticky',
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
    fontWeight: 'bold',
    color: '#1f2937',
    margin: 0,
  },
  loadingTag: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.75rem',
    color: '#2563eb',
    backgroundColor: '#eff6ff',
    padding: '0.25rem 0.5rem',
    borderRadius: '0.25rem',
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
    fontWeight: '600',
    backgroundColor: '#dbeafe',
    color: '#1e40af',
    border: 'none',
    borderRadius: '0.375rem',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  statusText: {
    fontSize: '0.75rem',
    color: '#6b7280',
    fontFamily: 'monospace',
    width: '120px',
    textAlign: 'center',
    borderLeft: '1px solid #d1d5db',
    paddingLeft: '0.5rem',
    marginLeft: '0.25rem',
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
    flexDirection: 'column',
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
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
    maxWidth: '64rem',
    padding: '2rem',
    paddingBottom: '10rem',
  },
  pageContainer: {
    position: 'relative',
    marginBottom: '20px',
    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)',
    backgroundColor: '#fff',
    display: 'inline-block',
  },
  canvas: {
    display: 'block',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 10,
  },
  lineBase: {
    position: 'absolute',
    cursor: 'pointer',
    borderRadius: '2px',
    backgroundColor: 'transparent',
    transition: 'background-color 0.2s ease',
  },
  lineActive: {
    backgroundColor: 'rgba(255, 230, 0, 0.5)',
    boxShadow: '0 0 0 2px rgba(220, 200, 0, 0.6)',
    mixBlendMode: 'multiply',
    zIndex: 20,
  },
};

export default function App() {
  // --- State ---
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [pages, setPages] = useState<any[]>([]);
  const [allLines, setAllLines] = useState<any[]>([]);
  const [currentLineIndex, setCurrentLineIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  // TTS State
  const [ttsStatus, setTtsStatus] = useState("Init");
  const [bufferStatus, setBufferStatus] = useState("Ready");

  // --- Refs ---
  const workerRef = useRef<Worker | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const audioCache = useRef(new Map());
  const pendingFetches = useRef(new Set());
  const currentSource = useRef<AudioBufferSourceNode | null>(null);
  const playbackSessionId = useRef(0);
  const usingFallback = useRef(false);
  const nativeTimeout = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- Initialization ---
  useEffect(() => {
    console.log("[App] Mounting...");

    // Initialize Worker
    workerRef.current = new Worker("/tts.worker.js", {
      type: 'module'
    });

    // Worker Message Listener
    workerRef.current.onmessage = (e) => {
      const { status, audio, error, text } = e.data;

      if (status === 'ready') {
        setTtsStatus("AI Ready");
        console.log("[Worker] Model Loaded");
      }
      else if (status === 'complete') {
        // Find which line this audio belongs to (simple matching or index tracking needed in real app)
        // For now, we process it immediately if it matches pending requests
        console.log(`[Worker] Audio generated. Size: ${audio.length}`);

        // Convert Float32Array to AudioBuffer
        // We need to do this on the main thread
        try {
          const ctx = getAudioContext();
          const buffer = ctx.createBuffer(1, audio.length, 24000);
          buffer.getChannelData(0).set(audio);

          // Store in cache (simplified logic - normally we'd map this back to an index)
          // For the "Test" button, we play immediately
          if (text.startsWith("Hello")) {
            playBufferDirectly(buffer);
          } else {
             // For PDF reading, we'd store in audioCache based on a request ID
             // Since we don't have request IDs in this simple snippet, we'll implement
             // a "callback" style in processLineAudio logic below
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

    // Start Model Load
    setTtsStatus("Loading...");
    workerRef.current.postMessage({ type: 'init' });

    return () => {
      console.log("[App] Unmounting...");
      stopAllAudio();
      workerRef.current?.terminate();
      if (audioContext.current) audioContext.current.close();
    };
  }, []);

  // --- Audio Loop ---
  useEffect(() => {
    if (isPlaying && currentLineIndex >= 0) {
      playCurrentLine();
    }
  }, [isPlaying, currentLineIndex]);


  const triggerFallback = () => {
    if (usingFallback.current) return;
    console.warn("[TTS] Switching to System Voice Fallback");
    usingFallback.current = true;
    setTtsStatus("System Voice");
  };

  const getAudioContext = () => {
    if (!audioContext.current) {
      audioContext.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.current.state === 'suspended') {
      audioContext.current.resume();
    }
    return audioContext.current;
  };

  const playBufferDirectly = (buffer: AudioBuffer) => {
     const ctx = getAudioContext();
     const source = ctx.createBufferSource();
     source.buffer = buffer;
     source.connect(ctx.destination);
     source.start(0);
     setBufferStatus("Playing");
     source.onended = () => setBufferStatus("Ready");
  }

  // --- Core TTS Logic (Worker Based) ---
  const generateAudioInWorker = (text: string): Promise<AudioBuffer | null> => {
    return new Promise((resolve) => {
      if (usingFallback.current || !workerRef.current) {
        resolve(null);
        return;
      }

      // We need a one-time listener for this specific request
      // This is a simple implementation. A proper one would use IDs.
      const tempListener = (e: MessageEvent) => {
        const { status, audio, text: returnedText } = e.data;
        if (status === 'complete' && returnedText === text) {
           workerRef.current?.removeEventListener('message', tempListener);

           const ctx = getAudioContext();
           const buffer = ctx.createBuffer(1, audio.length, 24000);
           buffer.getChannelData(0).set(audio);
           resolve(buffer);
        } else if (status === 'error') {
            workerRef.current?.removeEventListener('message', tempListener);
            resolve(null);
        }
      };

      workerRef.current.addEventListener('message', tempListener);
      workerRef.current.postMessage({ type: 'generate', text });
    });
  };

  // --- Audio Processing ---
  const stopAllAudio = () => {
    if (currentSource.current) {
      try { currentSource.current.stop(); } catch(e){}
      currentSource.current = null;
    }
    window.speechSynthesis.cancel();
    if (nativeTimeout.current) clearTimeout(nativeTimeout.current);
    if (audioContext.current) audioContext.current.suspend();
  };

  const updateBufferUI = () => {
    if (usingFallback.current) {
      setBufferStatus("System Voice");
      return;
    }
    const cached = audioCache.current.size;
    const pending = pendingFetches.current.size;
    setBufferStatus(`Buf:${cached} | Fetch:${pending}`);
  };

  const processLineAudio = async (index: number, sessionId: number) => {
    if (usingFallback.current) return null;
    if (index >= allLines.length) return null;

    if (audioCache.current.has(index)) {
      return audioCache.current.get(index);
    }

    if (pendingFetches.current.has(index)) return null;

    const text = allLines[index].text;
    if (!text.trim()) return null;

    pendingFetches.current.add(index);
    updateBufferUI();

    try {
      // Use Worker instead of direct generation
      const buffer = await generateAudioInWorker(text);

      if (!buffer) {
        // If worker failed silently or returned null, likely fallback needed
        return null;
      }

      if (sessionId === playbackSessionId.current) {
        audioCache.current.set(index, buffer);
      }
      return buffer;
    } catch (err) {
      console.error(`[Buffer] Failed to process line ${index}`, err);
      return null;
    } finally {
      pendingFetches.current.delete(index);
      updateBufferUI();
    }
  };

  const playCurrentLine = async () => {
    if (!isPlaying || currentLineIndex === -1) return;
    if (currentLineIndex >= allLines.length) {
      setIsPlaying(false);
      return;
    }

    const currentSession = playbackSessionId.current;
    const text = allLines[currentLineIndex].text;

    // --- Native Fallback Path ---
    if (usingFallback.current) {
      window.speechSynthesis.cancel();
      if (nativeTimeout.current) clearTimeout(nativeTimeout.current);

      nativeTimeout.current = setTimeout(() => {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.1;
        u.onend = () => {
          if (isPlaying && currentSession === playbackSessionId.current) {
            advanceLine();
          }
        };
        u.onerror = (e) => { if (isPlaying) advanceLine(); };
        window.speechSynthesis.speak(u);
      }, 50);
      return;
    }

    getAudioContext();

    // Lookahead
    for (let i = 1; i <= 3; i++) {
      if (currentLineIndex + i < allLines.length) {
        processLineAudio(currentLineIndex + i, currentSession);
      }
    }

    let buffer = audioCache.current.get(currentLineIndex);
    if (!buffer) {
      setBufferStatus("Buffering...");
      buffer = await processLineAudio(currentLineIndex, currentSession);
    }

    if (usingFallback.current) {
      playCurrentLine();
      return;
    }

    if (currentSession !== playbackSessionId.current || !isPlaying) return;

    if (buffer) {
      const source = audioContext.current!.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.current!.destination);
      source.onended = () => {
        if (isPlaying && currentSession === playbackSessionId.current) {
          advanceLine();
        }
      };
      currentSource.current = source;
      source.start(0);
    } else {
      triggerFallback();
      playCurrentLine();
    }
  };

  const advanceLine = () => {
    setCurrentLineIndex(prev => prev + 1);
  };

  // --- Interaction ---
  const handleLineClick = (index: number) => {
    if (currentSource.current) {
       try { currentSource.current.stop(); } catch(e){}
    }
    window.speechSynthesis.cancel();

    playbackSessionId.current += 1;
    audioCache.current.clear();
    pendingFetches.current.clear();

    setCurrentLineIndex(index);
    setIsPlaying(true);

    const el = document.getElementById(`line-${index}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const togglePlay = () => {
    if (currentLineIndex === -1 && allLines.length > 0) {
      handleLineClick(0);
      return;
    }

    if (isPlaying) {
      setIsPlaying(false);
      stopAllAudio();
    } else {
      setIsPlaying(true);
    }
  };

  const handleTestAudio = () => {
    stopAllAudio();
    const text = "Hello! I am Kokoro.";
    if (usingFallback.current) {
      const u = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(u);
    } else {
      setBufferStatus("Generating...");
      workerRef.current?.postMessage({ type: 'generate', text });
    }
  };

  const resetReader = () => {
    setIsPlaying(false);
    stopAllAudio();
    setPdfDoc(null);
    setPages([]);
    setAllLines([]);
    setCurrentLineIndex(-1);
    audioCache.current.clear();
  };

  // --- PDF Processing (Same as before) ---
  const handleFileDrop = async (e: React.DragEvent | React.ChangeEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    // Handle both DragEvent and ChangeEvent
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
      setAllLines(globalLineList);
      setPages(newPages);
    } catch (e) {
      console.error(e);
    }
  };

  // Helper for text processing (condensed for brevity, same logic as before)
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
    // Sort Y then X
    items.sort((a: any, b: any) => Math.abs(a.y - b.y) > a.height * 0.2 ? b.y - a.y : a.x - b.x);
    // Group lines
    const lines: any[] = [];
    let currentLine: any = null;
    items.forEach((item: any) => {
        if (!currentLine) currentLine = { items: [item], y: item.y, height: item.height };
        else if (Math.abs(item.y - currentLine.y) < currentLine.height * 0.5) currentLine.items.push(item);
        else { lines.push(currentLine); currentLine = { items: [item], y: item.y, height: item.height }; }
    });
    if (currentLine) lines.push(currentLine);

    return lines.map((line: any, idx: number) => {
        const minX = Math.min(...line.items.map((i:any) => i.x));
        const last = line.items[line.items.length - 1];
        const width = (last.x + (last.width || last.str.length * 5)) - minX;
        return {
            id: startIndex + idx,
            text: line.items.map((i:any) => i.str).join(' '),
            x: minX,
            y: line.y - line.height * 0.85,
            width,
            height: line.height
        };
    });
  };

  // --- Sub-Component: PDF Page ---
  const PDFPage = ({ data, currentLineIndex, onLineClick }: any) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const renderTaskRef = useRef<any>(null);

    useEffect(() => {
      if (canvasRef.current && data) {
        if (renderTaskRef.current) { try { renderTaskRef.current.cancel(); } catch(e){} }
        const ctx = canvasRef.current.getContext('2d');
        const renderTask = data.page.render({ canvasContext: ctx, viewport: data.viewport });
        renderTaskRef.current = renderTask;
        renderTask.promise.catch((e:any) => { if(e.name !== 'RenderingCancelledException') console.error(e) });
      }
    }, [data]);

    return (
      <div style={{...styles.pageContainer, width: data.viewport.width, height: data.viewport.height}}>
        <canvas ref={canvasRef} width={data.viewport.width} height={data.viewport.height} style={styles.canvas} />
        <div style={styles.overlay}>
          {data.lines.map((line: any) => (
            <div
              key={line.id}
              id={`line-${line.id}`}
              onClick={(e) => { e.stopPropagation(); onLineClick(line.id); }}
              style={{
                ...styles.lineBase,
                ...(currentLineIndex === line.id ? styles.lineActive : {}),
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

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.logoGroup}>
          <FileText size={24} color="#2563eb" />
          <h1 style={styles.title}>PDF Reader</h1>
          <div style={styles.loadingTag}>
            {ttsStatus === "Loading..." && <Loader2 size={12} className="animate-spin" />}
            {ttsStatus}
          </div>
        </div>

        <div style={styles.controls}>
          <button onClick={handleTestAudio} style={styles.testButton} disabled={ttsStatus === "Loading..."}>
             <Beaker size={16} /> Test Voice
          </button>
          <div style={styles.playerGroup}>
            <button onClick={togglePlay} disabled={allLines.length === 0} style={{...styles.iconButton, opacity: allLines.length === 0 ? 0.5 : 1}}>
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </button>
            <div style={styles.statusText}>{bufferStatus}</div>
          </div>
          <button onClick={resetReader} style={styles.resetButton}>Reset</button>
        </div>
      </div>

      {allLines.length === 0 ? (
        <div style={{...styles.dropZone, ...(isDragOver ? styles.dropZoneHover : {})}}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleFileDrop}
        >
          <Upload size={48} color="#9ca3af" style={{marginBottom: '1rem'}} />
          <p style={{fontSize: '1.25rem', fontWeight: 600, color: '#4b5563'}}>Drop PDF here</p>
          <input type="file" ref={fileInputRef} onChange={handleFileDrop} className="hidden" accept="application/pdf" style={{display: 'none'}} />
        </div>
      ) : (
        <div style={styles.viewer}>
          {pages.map(pageData => (
            <PDFPage key={pageData.pageNumber} data={pageData} currentLineIndex={currentLineIndex} onLineClick={handleLineClick} />
          ))}
        </div>
      )}
    </div>
  );
}
