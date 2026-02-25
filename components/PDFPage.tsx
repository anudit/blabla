import React, { useEffect, useRef } from 'react';
import { staticStyles } from '../theme';

interface PDFPageProps {
  data: any;
  pdfDoc: any;
  onLineClick: (lineId: number) => void;
  pageContainerStyle: React.CSSProperties;
}

// Lazy canvas rendering via IntersectionObserver — pages outside the viewport
// are cleaned up to free GPU/bitmap memory, and re-rendered when scrolled into view.
const PDFPage = React.memo(({ data, pdfDoc, onLineClick, pageContainerStyle }: PDFPageProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const renderTaskRef = useRef<any>(null);
  const pageRef = useRef<any>(null);
  const isVisibleRef = useRef(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !pdfDoc) return;

    const renderPage = async () => {
      if (!canvasRef.current || !isVisibleRef.current) return;
      if (renderTaskRef.current) {
        try { renderTaskRef.current.cancel(); } catch (e) {}
        renderTaskRef.current = null;
      }
      try {
        const page = await pdfDoc.getPage(data.pageNumber);
        pageRef.current = page;
        if (!isVisibleRef.current) { page.cleanup(); pageRef.current = null; return; }
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = data.viewport.width;
        canvas.height = data.viewport.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const renderTask = page.render({ canvasContext: ctx, viewport: data.viewport });
        renderTaskRef.current = renderTask;
        await renderTask.promise;
      } catch (e: any) {
        if (e.name !== 'RenderingCancelledException') console.error("Render error:", e);
      }
    };

    const cleanupPage = () => {
      if (renderTaskRef.current) {
        try { renderTaskRef.current.cancel(); } catch (e) {}
        renderTaskRef.current = null;
      }
      if (pageRef.current) {
        pageRef.current.cleanup();
        pageRef.current = null;
      }
      if (canvasRef.current) {
        canvasRef.current.width = 0;
        canvasRef.current.height = 0;
      }
    };

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (!entry) return;
        if (entry.isIntersecting && !isVisibleRef.current) {
          isVisibleRef.current = true;
          renderPage();
        } else if (!entry.isIntersecting && isVisibleRef.current) {
          isVisibleRef.current = false;
          cleanupPage();
        }
      },
      { rootMargin: '1200px 0px' }
    );
    observer.observe(container);

    return () => {
      observer.disconnect();
      isVisibleRef.current = false;
      cleanupPage();
    };
  }, [data, pdfDoc]);

  // Aspect-ratio placeholder keeps scroll position stable before canvas renders
  const aspectRatio = data.viewport.height / data.viewport.width;

  return (
    <div ref={containerRef} style={{ ...pageContainerStyle, maxWidth: data.viewport.width }}>
      <div style={{ position: 'relative', width: '100%', paddingBottom: `${aspectRatio * 100}%` }}>
        <canvas
          ref={canvasRef}
          style={{ ...staticStyles.canvas, position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
        />
        <div style={{ ...staticStyles.overlay, position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
          {data.lines.map((line: any) => (
            <div
              key={line.id}
              id={`line-${line.id}`}
              onClick={(e) => { e.stopPropagation(); onLineClick(line.id); }}
              style={{
                ...staticStyles.lineBase,
                left: line.left, top: line.top, width: line.width, height: line.height,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}, (prev, next) => prev.data.pageNumber === next.data.pageNumber && prev.pdfDoc === next.pdfDoc);

export default PDFPage;
