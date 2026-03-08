import { Fragment } from 'preact';
import { memo } from 'preact/compat';
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

interface ContentViewerProps {
  fileType: 'pdf' | 'epub' | 'text';
  pages: any[];
  pdfDoc: any;
  epubContent: any[];
  activeHeaderId: Signal<string | null>;
  t: ThemeTokens;
  isDarkMode: boolean;
  fontSize: number;
  onLineClick: (lineId: number) => void;
}

const H_SIZE: Record<number, string> = { 1: '1.45rem', 2: '1.25rem', 3: '1.1rem', 4: '1rem', 5: '0.95rem', 6: '0.9rem' };

export default function ContentViewer({
  fileType, pages, pdfDoc, epubContent, activeHeaderId,
  t, isDarkMode, fontSize, onLineClick,
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

  return (
    <div 
      onClick={handleContainerClick}
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', maxWidth: '48rem', padding: '1rem', paddingBottom: '6rem', boxSizing: 'border-box' }}
    >
      {fileType === 'pdf' && pages.map(pageData => (
        <PDFPage key={pageData.pageNumber} data={pageData} pdfDoc={pdfDoc} onLineClick={onLineClick} pageContainerStyle={pageContainerStyle} />
      ))}

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

const RenderItem = memo(({ item, t, isDarkMode, lc }: any) => {
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
      <LazyBlock id={item.id} startLineId={item.startLineId} endLineId={item.endLineId}>
        <p style={{
          margin: isBlockquote ? '0 0 0.9em' : '0 0 1.1em', padding: 0, paddingLeft: isList ? '1.3em' : isBlockquote ? '1em' : 0,
          borderLeft: isBlockquote ? `3px solid ${t.dropBorder}` : 'none', fontStyle: isBlockquote ? 'italic' : 'normal',
          color: isBlockquote ? t.textMuted : 'inherit', lineHeight: 'inherit', position: 'relative',
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
  // Only the active paragraph re-renders its sentences, and even then, 
  // we check if this specific sentence is the active one.
  const isSelected = useComputed(() => isActive.value && currentSentenceIndexSignal.value === s.id);
  
  return (
    <span 
      id={`line-${s.id}`} 
      data-line-id={s.id} 
      style={{ 
        cursor: 'pointer', 
        padding: '2px 0', 
        transition: `background-color 0.2s, ${TT}`, 
        borderRadius: '4px',
        // Optional: you could add a local "active" style here too, 
        // though the global observer handles the DOM injection for word-level spans.
      }}
    >
      {s.text}{' '}
    </span>
  );
};
