import { Fragment } from 'preact';
import type { JSX } from 'preact';
import type { Signal } from '@preact/signals';
import type { ThemeTokens } from '../theme';
import { TT } from '../theme';
import { renderMd } from '../utils';
import type { OutlineEntry } from './BookOutline';
import BookOutline from './BookOutline';
import PDFPage from './PDFPage';
import LazyBlock from './LazyBlock';

interface ContentViewerProps {
  fileType: 'pdf' | 'epub' | 'text';
  // PDF
  pages: any[];
  pdfDoc: any;
  // Epub/Text
  epubContent: any[];
  outline: OutlineEntry[];
  activeHeaderId: Signal<string | null>;
  // Theme
  t: ThemeTokens;
  isDarkMode: boolean;
  fontSize: number;
  // Handlers
  onLineClick: (lineId: number) => void;
}

const H_SIZE: Record<number, string> = {
  1: '1.45rem', 2: '1.25rem', 3: '1.1rem',
  4: '1rem', 5: '0.95rem', 6: '0.9rem',
};

export default function ContentViewer({
  fileType, pages, pdfDoc, epubContent, outline, activeHeaderId,
  t, isDarkMode, fontSize, onLineClick,
}: ContentViewerProps) {
  const pageContainerStyle: JSX.CSSProperties = {
    position: 'relative',
    marginBottom: '1rem',
    boxShadow: t.pageShadow,
    backgroundColor: '#fff',
    width: '100%',
    height: 'auto',
  };

  const epubSentenceStyle: JSX.CSSProperties = {
    cursor: 'pointer',
    padding: '2px 0',
    transition: `background-color 0.2s, ${TT}`,
    borderRadius: '4px',
  };

  const lc = isDarkMode ? '#60a5fa' : '#2563eb';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      width: '100%',
      maxWidth: '48rem',
      padding: '1rem',
      paddingBottom: '6rem',
      boxSizing: 'border-box',
    }}>
      {fileType === 'pdf' && pages.map(pageData => (
        <PDFPage
          key={pageData.pageNumber}
          data={pageData}
          pdfDoc={pdfDoc}
          onLineClick={onLineClick}
          pageContainerStyle={pageContainerStyle}
        />
      ))}

      {(fileType === 'epub' || fileType === 'text') && (
        <>
          <BookOutline entries={outline} activeId={activeHeaderId} isDarkMode={isDarkMode} />
          <div style={{
            width: '100%',
            padding: '0.25rem 0',
            lineHeight: '1.8',
            fontSize: `${fontSize}rem`,
            textAlign: 'left',
            backgroundColor: t.epubBg,
            color: t.text,
            transition: TT,
          }}>
            {renderContent({ epubContent, t, isDarkMode, lc, epubSentenceStyle, onLineClick })}
          </div>
        </>
      )}
    </div>
  );
}

interface RenderContentProps {
  epubContent: any[];
  t: ThemeTokens;
  isDarkMode: boolean;
  lc: string;
  epubSentenceStyle: JSX.CSSProperties;
  onLineClick: (lineId: number) => void;
}

function renderContent({ epubContent, t, isDarkMode, lc, epubSentenceStyle, onLineClick }: RenderContentProps) {
  let headersSeen = 0;

  return epubContent.map((item) => {
    if (item.type === 'frontmatter') {
      return (
        <div key={item.id} style={{
          margin: '0 0 2rem',
          borderRadius: '0.75rem',
          overflow: 'hidden',
          border: `1px solid ${t.statBorder}`,
          backgroundColor: isDarkMode ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
        }}>
          {item.image && (
            <img src={item.image} alt={item.title || ''} loading="lazy" style={{
              width: '100%', maxHeight: '220px', objectFit: 'cover', display: 'block',
            }} />
          )}
          <div style={{ padding: '1rem 1.1rem' }}>
            {item.title && <p style={{ margin: '0 0 0.3rem', fontSize: '1.3rem', fontWeight: 700, color: t.headerColor, lineHeight: 1.25 }}>{item.title}</p>}
            {item.description && <p style={{ margin: 0, fontSize: '0.875rem', color: t.textMuted, lineHeight: 1.5 }}>{item.description}</p>}
          </div>
        </div>
      );
    }

    if (item.type === 'header') {
      const isFirst = headersSeen === 0;
      headersSeen++;
      const lvl: number = item.level ?? 1;
      const showDivider = !isFirst && lvl <= 1;
      return (
        <Fragment key={item.id}>
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
              margin: isFirst
                ? `0 0 ${lvl === 1 ? '1.4rem' : '1rem'}`
                : `${lvl === 1 ? '2rem' : '1.4rem'} 0 ${lvl === 1 ? '1.4rem' : '0.6rem'}`,
              color: t.headerColor,
              lineHeight: 1.25,
              letterSpacing: lvl === 1 ? '-0.015em' : 'normal',
              scrollMarginTop: '1.5rem',
            }}
          >
            {item.text}
          </div>
        </Fragment>
      );
    }

    if (item.type === 'paragraph') {
      const isBlockquote = item.elementType === 'blockquote';
      const isList = item.elementType === 'li';

      // EPUB paragraph — per-sentence word spans, preserves blockquote/list styling
      if (item.runs) {
        return (
          <LazyBlock key={item.id}>
            <p style={{
              margin: isBlockquote ? '0 0 0.9em' : '0 0 1.1em',
              padding: 0,
              paddingLeft: isList ? '1.3em' : isBlockquote ? '1em' : 0,
              borderLeft: isBlockquote ? `3px solid ${t.dropBorder}` : 'none',
              fontStyle: isBlockquote ? 'italic' : 'normal',
              color: isBlockquote ? t.textMuted : 'inherit',
              lineHeight: 'inherit',
              position: 'relative',
            }}>
              {isList && <span style={{ position: 'absolute', left: 0, color: t.textMuted, userSelect: 'none' }}>•</span>}
              {item.sentences.map((sentence: any) => {
                // Words computed here so only visible paragraphs (inside LazyBlock) pay the cost
                const words: string[] = sentence.words ?? sentence.text.trim().split(/\s+/);
                return (
                  <span
                    key={sentence.id}
                    id={`line-${sentence.id}`}
                    onClick={() => onLineClick(sentence.id)}
                    style={epubSentenceStyle}
                  >
                    {words.map((word: string, wi: number) => (
                      <span key={wi} id={`word-${sentence.id}-${wi}`} style={{ transition: 'background-color 0.08s ease' }}>
                        {word}{wi < words.length - 1 ? ' ' : ''}
                      </span>
                    ))}{' '}
                  </span>
                );
              })}
            </p>
          </LazyBlock>
        );
      }

      // Text / Markdown paragraph — word-level spans inside sentence spans
      return (
        <LazyBlock key={item.id}>
          <p style={{ margin: '0 0 1.1em', padding: 0, lineHeight: 'inherit' }}>
            {item.sentences.map((sentence: any) => (
              <span
                key={sentence.id}
                id={`line-${sentence.id}`}
                onClick={() => onLineClick(sentence.id)}
                style={epubSentenceStyle}
              >
                {sentence.words
                  ? sentence.words.map((word: string, wi: number) => (
                      <span key={wi} id={`word-${sentence.id}-${wi}`} style={{ transition: 'background-color 0.08s ease' }}>
                        {word}{wi < sentence.words.length - 1 ? ' ' : ''}
                      </span>
                    ))
                  : (sentence.md ? renderMd(sentence.md, lc) : sentence.text)
                }{' '}
              </span>
            ))}
          </p>
        </LazyBlock>
      );
    }

    if (item.type === 'code') {
      return (
        <LazyBlock key={item.id}>
          <pre style={{
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
        </LazyBlock>
      );
    }

    if (item.type === 'table') {
      return (
        <LazyBlock key={item.id}>
          <div style={{ overflowX: 'auto', margin: '0 0 1.5em', borderRadius: '0.5rem', border: `1px solid ${t.statBorder}` }}>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '0.82rem', minWidth: '320px' }}>
              {item.headers.length > 0 && (
                <thead>
                  <tr style={{ backgroundColor: isDarkMode ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)' }}>
                    {item.headers.map((h: string, i: number) => (
                      <th key={i} style={{ padding: '0.5rem 0.75rem', textAlign: 'left', fontWeight: 600, color: t.text, borderBottom: `1px solid ${t.statBorder}`, whiteSpace: 'nowrap' }}>
                        {renderMd(h, lc)}
                      </th>
                    ))}
                  </tr>
                </thead>
              )}
              <tbody>
                {item.rows.map((row: string[], ri: number) => (
                  <tr key={ri} style={{ backgroundColor: ri % 2 !== 0 ? (isDarkMode ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.015)') : 'transparent' }}>
                    {row.map((cell: string, ci: number) => (
                      <td key={ci} style={{ padding: '0.4rem 0.75rem', color: t.text, borderBottom: ri < item.rows.length - 1 ? `1px solid ${t.statBorder}` : 'none', verticalAlign: 'top' }}>
                        {renderMd(cell.replace(/\\(.)/g, '$1'), lc)}
                      </td>
                    ))}
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
        <LazyBlock key={item.id}>
          <div style={{ textAlign: 'center', margin: '1.2em 0' }}>
            <img
              src={item.src}
              alt={item.alt}
              loading="lazy"
              style={{ maxWidth: '100%', borderRadius: '0.5rem', display: 'inline-block' }}
            />
            {item.alt && <p style={{ fontSize: '0.78rem', color: t.textMuted, margin: '0.4em 0 0', fontStyle: 'italic' }}>{item.alt}</p>}
          </div>
        </LazyBlock>
      );
    }

    if (item.type === 'hr') {
      return <hr key={item.id} style={{ border: 'none', borderTop: `1px solid ${t.statBorder}`, margin: '1.5rem 0', opacity: 0.5 }} />;
    }

    return null;
  });
}
