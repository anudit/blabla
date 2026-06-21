import { Fragment } from 'preact';
import { memo } from 'preact/compat';
import { useEffect, useRef } from 'preact/hooks';
import type { JSX } from 'preact';
import { useComputed } from '@preact/signals';
import type { Signal } from '@preact/signals';
import { outlineSignal, currentSentenceIndexSignal } from '../signals';
import type { ThemeTokens } from '../theme';
import { TT } from '../theme';
import { renderMd } from '../utils';
import BookOutline from './BookOutline';
import PDFPage from './PDFPage';
import LazyBlock from './LazyBlock';
import Icon from './icons';

interface ContentViewerProps {
  fileType: 'pdf' | 'epub' | 'text' | 'ocr';
  pages: any[];
  pdfDoc: any;
  epubContent: any[];
  activeHeaderId: Signal<string | null>;
  t: ThemeTokens;
  isDarkMode: boolean;
  fontSize: number;
  onLineClick: (lineId: number) => void;
  ocrCurrentPage?: number;
  setOcrCurrentPage?: (page: number) => void;
  ocrPageLoading?: boolean;
}

const H_SIZE: Record<number, string> = { 1: '1.45rem', 2: '1.25rem', 3: '1.1rem', 4: '1rem', 5: '0.95rem', 6: '0.9rem' };

export default function ContentViewer({
  fileType, pages, pdfDoc, epubContent, activeHeaderId,
  t, isDarkMode, fontSize, onLineClick,
  ocrCurrentPage, setOcrCurrentPage, ocrPageLoading,
}: ContentViewerProps) {
  const pageContainerStyle: JSX.CSSProperties = { position: 'relative', marginBottom: '1rem', boxShadow: t.pageShadow, backgroundColor: '#fff', width: '100%', height: 'auto' };
  const lc = isDarkMode ? '#60a5fa' : '#2563eb';

  const handleContainerClick = (e: MouseEvent) => {
    const target = (e.target as HTMLElement).closest('[data-line-id]');
    if (target) {
      const lineId = parseInt(target.getAttribute('data-line-id') || '-1');
      if (lineId !== -1) onLineClick(lineId);
    }
  };

  const wrapperMaxWidth = fileType === 'ocr' ? '110rem' : '48rem';

  return (
    <div 
      onClick={handleContainerClick}
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', maxWidth: wrapperMaxWidth, padding: '1rem', paddingBottom: '6rem', boxSizing: 'border-box' }}
    >
      {fileType === 'pdf' && pages.map(pageData => (
        <PDFPage key={pageData.pageNumber} data={pageData} pdfDoc={pdfDoc} onLineClick={onLineClick} pageContainerStyle={pageContainerStyle} />
      ))}

      {fileType === 'ocr' && (
        <OCRSplitView
          pdfDoc={pdfDoc}
          ocrCurrentPage={ocrCurrentPage || 1}
          setOcrCurrentPage={setOcrCurrentPage}
          epubContent={epubContent}
          t={t}
          isDarkMode={isDarkMode}
          fontSize={fontSize}
          lc={lc}
          ocrPageLoading={ocrPageLoading}
        />
      )}

      {(fileType === 'epub' || fileType === 'text') && (
        <>
          <BookOutline entries={outlineSignal.value} activeId={activeHeaderId} isDarkMode={isDarkMode} />
          <div style={{ width: '100%', padding: '0.25rem 0', lineHeight: '1.8', fontSize: `${fontSize}rem`, textAlign: 'left', backgroundColor: t.epubBg, color: t.text, transition: TT }}>
            {epubContent.map((item) => <RenderItem key={item.id} item={item} t={t} isDarkMode={isDarkMode} lc={lc} />)}
          </div>
        </>
      )}
    </div>
  );
}

const RenderItem = memo(({ item, t, isDarkMode, lc, isOcr }: any) => {
  if (item.type === 'frontmatter') {
    return (
      <div style={{ margin: '0 0 2rem', borderRadius: '0.75rem', overflow: 'hidden', border: `1px solid ${t.statBorder}`, backgroundColor: isDarkMode ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)' }}>
        {item.image && <img src={item.image} alt={item.title || ''} loading="lazy" style={{ width: '100%', maxHeight: '220px', objectFit: 'cover', display: 'block' }} />}
        <div style={{ padding: '1rem 1.1rem' }}>
          {item.title && <p style={{ margin: '0 0 0.3rem', fontSize: '1.3rem', fontWeight: 700, color: t.headerColor, lineHeight: 1.25 }}>{item.title}</p>}
          {item.description && <p style={{ margin: 0, fontSize: '0.875rem', color: t.textMuted, lineHeight: 1.5 }}>{item.description}</p>}
        </div>
      </div>
    );
  }

  if (item.type === 'header') {
    const lvl: number = item.level ?? 1;
    return (
      <Fragment>
        {lvl <= 1 && <hr style={{ border: 'none', borderTop: `1px solid ${t.statBorder}`, margin: '3rem 0 2.5rem', opacity: 0.45 }} />}
        <div id={item.id} style={{ fontSize: H_SIZE[lvl] ?? '1.45rem', fontWeight: lvl <= 2 ? 700 : 600, margin: `${lvl === 1 ? '2rem' : '1.4rem'} 0 ${lvl === 1 ? '1.4rem' : '0.6rem'}`, color: t.headerColor, lineHeight: 1.25, letterSpacing: lvl === 1 ? '-0.015em' : 'normal', scrollMarginTop: '1.5rem' }}>
          {item.text}
        </div>
      </Fragment>
    );
  }

  if (item.type === 'paragraph') {
    const isBlockquote = item.elementType === 'blockquote';
    const isList = item.elementType === 'li';

    // OPTIMIZATION: Only subscribe to currentSentenceIndexSignal if it's within this paragraph's range
    // This prevents all 53k sentences from re-rendering on every index change.
    const isActive = useComputed(() => {
      const idx = currentSentenceIndexSignal.value;
      return idx >= (item.startLineId ?? -1) && idx <= (item.endLineId ?? -1);
    });

    return (
      <LazyBlock id={item.id} startLineId={item.startLineId} endLineId={item.endLineId} compact={isOcr}>
        <p style={{
          margin: isOcr ? '0 0 0.05em' : (isBlockquote ? '0 0 0.9em' : '0 0 1.1em'), padding: 0, paddingLeft: isList ? '1.3em' : isBlockquote ? '1em' : 0,
          borderLeft: isBlockquote ? `3px solid ${t.dropBorder}` : 'none', fontStyle: isBlockquote ? 'italic' : 'normal',
          color: isBlockquote ? t.textMuted : 'inherit', lineHeight: isOcr ? '1.1' : 'inherit', position: 'relative',
        }}>
          {isList && <span style={{ position: 'absolute', left: 0, color: t.textMuted, userSelect: 'none' }}>•</span>}
          {item.sentences.map((s: any) => (
            <SentenceItem key={s.id} s={s} isActive={isActive} />
          ))}
        </p>
      </LazyBlock>
    );
  }

  if (item.type === 'code') {
    return (
      <LazyBlock id={item.id}>
        <pre style={{ backgroundColor: isDarkMode ? 'rgba(0,0,0,0.35)' : 'rgba(0,0,0,0.04)', border: `1px solid ${t.statBorder}`, borderRadius: '0.5rem', padding: '0.75rem 1rem', overflowX: 'auto', fontSize: '0.82rem', fontFamily: 'monospace', lineHeight: 1.6, margin: '0 0 1.1em', color: t.text, whiteSpace: 'pre' }}>
          <code>{item.text}</code>
        </pre>
      </LazyBlock>
    );
  }

  if (item.type === 'table') {
    return (
      <LazyBlock id={item.id}>
        <div style={{ overflowX: 'auto', margin: '0 0 1.5em', borderRadius: '0.5rem', border: `1px solid ${t.statBorder}` }}>
          <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '0.82rem', minWidth: '320px' }}>
            {item.headers.length > 0 && (
              <thead>
                <tr style={{ backgroundColor: isDarkMode ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)' }}>
                  {item.headers.map((h: string, i: number) => <th key={i} style={{ padding: '0.5rem 0.75rem', textAlign: 'left', fontWeight: 600, color: t.text, borderBottom: `1px solid ${t.statBorder}`, whiteSpace: 'nowrap' }}>{renderMd(h, lc)}</th>)}
                </tr>
              </thead>
            )}
            <tbody>
              {item.rows.map((row: string[], ri: number) => (
                <tr key={ri} style={{ backgroundColor: ri % 2 !== 0 ? (isDarkMode ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.015)') : 'transparent' }}>
                  {row.map((cell: string, ci: number) => <td key={ci} style={{ padding: '0.4rem 0.75rem', color: t.text, borderBottom: ri < item.rows.length - 1 ? `1px solid ${t.statBorder}` : 'none', verticalAlign: 'top' }}>{renderMd(cell.replace(/\\(.)/g, '$1'), lc)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </LazyBlock>
    );
  }

  if (item.type === 'image') {
    return (
      <LazyBlock id={item.id}>
        <div style={{ textAlign: 'center', margin: '1.2em 0' }}>
          <img src={item.src} alt={item.alt} loading="lazy" style={{ maxWidth: '100%', borderRadius: '0.5rem', display: 'inline-block' }} />
          {item.alt && <p style={{ fontSize: '0.78rem', color: t.textMuted, margin: '0.4em 0 0', fontStyle: 'italic' }}>{item.alt}</p>}
        </div>
      </LazyBlock>
    );
  }

  if (item.type === 'hr') return <hr style={{ border: 'none', borderTop: `1px solid ${t.statBorder}`, margin: '1.5rem 0', opacity: 0.5 }} />;
  return null;
});

const SentenceItem = ({ s, isActive }: { s: any, isActive: Signal<boolean> }) => {
  return (
    <span 
      id={`line-${s.id}`} 
      data-line-id={s.id} 
      style={{ 
        cursor: 'pointer', 
        padding: '2px 0', 
        transition: `background-color 0.2s, ${TT}`, 
        borderRadius: '4px',
      }}
    >
      {s.text}{' '}
    </span>
  );
};

const OCRSplitView = ({
  pdfDoc, ocrCurrentPage, setOcrCurrentPage, epubContent, t, isDarkMode, fontSize, lc, ocrPageLoading
}: any) => {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: '1.5rem',
      width: '100%',
      boxSizing: 'border-box',
      alignItems: 'flex-start',
    }}>
      {/* Left Pane - Render PDF Page */}
      <div style={{
        flex: '1 1 350px',
        minWidth: '320px',
        display: 'flex',
        flexDirection: 'column',
        boxSizing: 'border-box',
      }}>
        <OCRPDFView pdfDoc={pdfDoc} pageNumber={ocrCurrentPage} t={t} />
      </div>

      {/* Right Pane - OCR Text content */}
      <div style={{
        flex: '1.6 1 480px',
        minWidth: '320px',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: t.epubBg,
        border: `1px solid ${t.statBorder}`,
        borderRadius: '0.75rem',
        padding: '1.25rem',
        boxShadow: t.pageShadow,
        boxSizing: 'border-box',
        minHeight: '400px',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '1rem',
          borderBottom: `1px solid ${t.statBorder}`,
          paddingBottom: '0.5rem',
        }}>
          <span style={{ fontSize: '0.9rem', fontWeight: 600, color: t.headerColor }}>
            Recognized Page Text
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <button
              disabled={ocrCurrentPage === 1}
              onClick={() => setOcrCurrentPage?.(Math.max(1, ocrCurrentPage - 1))}
              style={{
                background: 'none', border: 'none', color: ocrCurrentPage === 1 ? t.textMuted : t.text,
                cursor: ocrCurrentPage === 1 ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', padding: '4px'
              }}
            >
              <Icon name="chevron-left" size={20} />
            </button>
            <span style={{ fontSize: '0.82rem', color: t.textMuted, fontWeight: 500 }}>
              Page {ocrCurrentPage} of {pdfDoc?.numPages || 1}
            </span>
            <button
              disabled={ocrCurrentPage === (pdfDoc?.numPages || 1)}
              onClick={() => setOcrCurrentPage?.(Math.min(pdfDoc?.numPages || 1, ocrCurrentPage + 1))}
              style={{
                background: 'none', border: 'none', color: ocrCurrentPage === (pdfDoc?.numPages || 1) ? t.textMuted : t.text,
                cursor: ocrCurrentPage === (pdfDoc?.numPages || 1) ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', padding: '4px'
              }}
            >
              <Icon name="chevron-right" size={20} />
            </button>
          </div>
        </div>

        {ocrPageLoading ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            flexGrow: 1,
            gap: '0.75rem',
            color: t.textMuted,
            padding: '2rem 0',
          }}>
            <Icon name="loader-circle" size={24} color={isDarkMode ? '#60a5fa' : '#2563eb'} style={{ animation: 'spin 0.8s linear infinite' }} />
            <span style={{ fontSize: '0.85rem' }}>Recognizing text via WebGPU...</span>
          </div>
        ) : (
          <div style={{ width: '100%', lineHeight: '1.1', fontSize: `${fontSize}rem`, textAlign: 'left', transition: TT }}>
            {epubContent.length === 0 ? (
              <div style={{ padding: '2rem 0', textAlign: 'center', color: t.textMuted, fontSize: '0.85rem' }}>
                No text found on this page.
              </div>
            ) : (
              epubContent.map((item: any) => (
                <RenderItem key={item.id} item={item} t={t} isDarkMode={isDarkMode} lc={lc} isOcr={true} />
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const OCRPDFView = ({ pdfDoc, pageNumber, t }: { pdfDoc: any; pageNumber: number; t: any }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderTaskRef = useRef<any>(null);

  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return;

    const render = async () => {
      try {
        const page = await pdfDoc.getPage(pageNumber);
        const canvas = canvasRef.current;
        if (!canvas) return;

        const viewport = page.getViewport({ scale: 1.5 });
        canvas.width = viewport.width;
        canvas.height = viewport.height;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        if (renderTaskRef.current) {
          try { renderTaskRef.current.cancel(); } catch (e) {}
        }
        const renderTask = page.render({ canvasContext: ctx, viewport });
        renderTaskRef.current = renderTask;
        await renderTask.promise;
        page.cleanup();
      } catch (e: any) {
        if (e.name !== 'RenderingCancelledException') {
          console.error("OCR PDF render error:", e);
        }
      }
    };

    render();
    return () => {
      if (renderTaskRef.current) {
        try { renderTaskRef.current.cancel(); } catch (e) {}
      }
    };
  }, [pdfDoc, pageNumber]);

  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'center',
      minHeight: '400px',
    }}>
      <canvas ref={canvasRef} style={{ maxWidth: '100%', maxHeight: '88vh', height: 'auto', width: 'auto', display: 'block', borderRadius: '4px' }} />
    </div>
  );
};



