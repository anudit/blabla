import { useRef } from 'preact/hooks';
import type { JSX } from 'preact';
import { Upload, Globe, Loader2, Clipboard, Bookmark } from 'lucide-preact';
import type { ThemeTokens } from '../theme';
import BookmarkHistory from './BookmarkHistory';
import type { BookmarkEntry } from './BookmarkHistory';

interface LandingCardProps {
  isDarkMode: boolean;
  t: ThemeTokens;
  isDragOver: boolean;
  setIsDragOver: (v: boolean) => void;
  onFileDrop: (e: JSX.TargetedDragEvent<HTMLDivElement> | JSX.TargetedEvent<HTMLInputElement, Event>) => void;
  urlInputValue: string;
  setUrlInputValue: (v: string) => void;
  urlError: string;
  setUrlError: (v: string) => void;
  isUrlLoading: boolean;
  onUrlLoad: (url?: string) => void;
  onClipboardPaste: (e: JSX.TargetedMouseEvent<HTMLButtonElement>) => void;
  showTextInput: boolean;
  setShowTextInput: (v: boolean) => void;
  textInputValue: string;
  setTextInputValue: (v: string) => void;
  onLoadText: (text: string) => void;
  bookmarks: BookmarkEntry[];
  onSelectBookmark: (entry: BookmarkEntry) => void;
  onDeleteBookmark: (id: string) => void;
}

export default function LandingCard({
  isDarkMode, t, isDragOver, setIsDragOver,
  onFileDrop, urlInputValue, setUrlInputValue, urlError, setUrlError,
  isUrlLoading, onUrlLoad, onClipboardPaste,
  showTextInput, setShowTextInput, textInputValue, setTextInputValue, onLoadText,
  bookmarks, onSelectBookmark, onDeleteBookmark,
}: LandingCardProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <>
      {/* ── Landing card ───────────────────────────────────────────────── */}
      <div style={{
        width: '90%', maxWidth: '38rem', marginTop: '3.5rem',
        borderRadius: '1.25rem',
        border: `1px solid ${isDragOver ? '#6b9fd4' : t.dropBorder}`,
        backgroundColor: isDragOver ? (isDarkMode ? '#1a2a3a' : '#eef4fb') : t.dropBg,
        boxShadow: isDragOver
          ? `0 0 0 3px ${isDarkMode ? 'rgba(107,159,212,0.12)' : 'rgba(107,159,212,0.1)'}, 0 2px 8px rgba(0,0,0,0.08)`
          : isDarkMode ? '0 2px 8px rgba(0,0,0,0.28)' : '0 1px 4px rgba(100,80,60,0.08)',
        transition: 'border-color 0.2s, box-shadow 0.2s, background-color 0.2s',
        overflow: 'hidden',
      }}>

        {/* File drop zone — clicking opens file picker */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={onFileDrop as any}
          style={{ padding: '2.25rem 2rem 2rem', textAlign: 'center', cursor: 'pointer' }}
        >
          <div style={{
            width: '42px', height: '42px', borderRadius: '0.75rem', margin: '0 auto 1.1rem',
            backgroundColor: isDarkMode ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Upload size={18} color={t.textMuted} />
          </div>
          <p style={{ fontSize: '0.95rem', fontWeight: 500, color: t.text, margin: '0 0 0.3rem', letterSpacing: '-0.01em' }}>
            Drop a file to start reading
          </p>
          <p style={{ fontSize: '0.75rem', color: t.textMuted, margin: 0, letterSpacing: '0.02em' }}>
            PDF · EPUB · MOBI · Markdown · TXT
          </p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={onFileDrop as any}
            accept="application/pdf,.epub,.mobi,.azw,.azw3,.md,.markdown,.txt,text/plain"
            style={{ display: 'none' }}
          />
        </div>

        {/* Hairline divider */}
        <div style={{ height: '1px', backgroundColor: t.dropBorder, opacity: 0.6 }} />

        {/* URL + clipboard actions */}
        <div onClick={(e) => e.stopPropagation()} style={{ padding: '1.25rem 1.5rem 1.5rem' }}>

          {/* URL row */}
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <Globe size={13} color={t.textMuted} style={{ position: 'absolute', left: '0.65rem', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
              <input
                type="url"
                value={urlInputValue}
                onChange={(e) => { setUrlInputValue((e.target as HTMLInputElement).value); if (urlError) setUrlError(''); }}
                onKeyDown={(e) => { if (e.key === 'Enter') onUrlLoad(); }}
                placeholder="Paste a URL to read as an article…"
                style={{
                  width: '100%', padding: '0.5rem 0.65rem 0.5rem 2rem',
                  fontSize: '0.8rem', color: t.text,
                  backgroundColor: isDarkMode ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.025)',
                  border: `1px solid ${urlError ? '#ef4444' : t.inputBorder}`,
                  borderRadius: '0.625rem', boxSizing: 'border-box', outline: 'none',
                  transition: 'border-color 0.15s',
                }}
              />
            </div>
            <button
              onClick={() => onUrlLoad()}
              disabled={isUrlLoading || !urlInputValue.trim()}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.3rem',
                padding: '0.5rem 0.9rem', fontSize: '0.8rem', fontWeight: 600,
                backgroundColor: isUrlLoading || !urlInputValue.trim() ? 'transparent' : '#2563eb',
                color: isUrlLoading || !urlInputValue.trim() ? t.textMuted : 'white',
                border: `1px solid ${isUrlLoading || !urlInputValue.trim() ? t.inputBorder : '#2563eb'}`,
                borderRadius: '0.625rem',
                cursor: isUrlLoading || !urlInputValue.trim() ? 'default' : 'pointer',
                whiteSpace: 'nowrap', flexShrink: 0, transition: 'all 0.15s',
              }}
            >
              {isUrlLoading
                ? <><Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} /> Fetching…</>
                : 'Load URL'}
            </button>
          </div>
          {urlError && <p style={{ margin: '-0.35rem 0 0.65rem', fontSize: '0.72rem', color: '#ef4444' }}>{urlError}</p>}

          {/* Clipboard button */}
          <button
            onClick={onClipboardPaste}
            style={{
              width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
              padding: '0.5rem', fontSize: '0.8rem', fontWeight: 500,
              backgroundColor: 'transparent', color: t.textMuted,
              border: `1px solid ${t.inputBorder}`,
              borderRadius: '0.625rem', cursor: 'pointer', transition: 'color 0.15s, border-color 0.15s',
            }}
          >
            <Clipboard size={13} /> Paste from Clipboard
          </button>

          {/* Bookmarklet — drag to bookmarks bar */}
          <a
            href={`javascript:(function(){window.location='https://blabla.anudit.dev?url='+encodeURIComponent(window.location.href)})()`}
            onClick={(e) => e.preventDefault()}
            draggable
            title="Drag this to your bookmarks bar. Click it on any page to open it in blabla."
            style={{
              marginTop: '0.5rem',
              width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
              padding: '0.5rem', fontSize: '0.8rem', fontWeight: 500,
              backgroundColor: 'transparent', color: t.textMuted,
              border: `1px dashed ${t.inputBorder}`,
              borderRadius: '0.625rem', cursor: 'grab', transition: 'color 0.15s, border-color 0.15s',
              textDecoration: 'none', boxSizing: 'border-box',
              userSelect: 'none',
            }}
          >
            <Bookmark size={13} />
            <span>Open in BlaBla</span>
            <span style={{ fontSize: '0.68rem', opacity: 0.6, marginLeft: '0.15rem' }}>— Drag to bookmarks bar</span>
          </a>

          {showTextInput && (
            <div style={{ marginTop: '0.85rem' }}>
              <textarea
                value={textInputValue}
                onChange={(e) => setTextInputValue((e.target as HTMLTextAreaElement).value)}
                placeholder="Paste your text here..."
                style={{
                  width: '100%', minHeight: '90px', padding: '0.6rem 0.75rem',
                  borderRadius: '0.625rem', border: `1px solid ${t.inputBorder}`,
                  fontSize: '0.85rem', color: t.text, backgroundColor: 'transparent',
                  boxSizing: 'border-box', resize: 'vertical', outline: 'none',
                }}
              />
              <button
                onClick={(e) => { e.stopPropagation(); if (textInputValue.trim()) onLoadText(textInputValue); }}
                style={{
                  marginTop: '0.5rem', width: '100%', padding: '0.6rem',
                  fontSize: '0.85rem', fontWeight: 600, backgroundColor: '#2563eb',
                  color: 'white', border: 'none', borderRadius: '0.625rem', cursor: 'pointer',
                }}
              >
                Start Reading
              </button>
            </div>
          )}
        </div>
      </div>

      <BookmarkHistory
        bookmarks={bookmarks}
        onSelect={onSelectBookmark}
        onDelete={onDeleteBookmark}
        isDarkMode={isDarkMode}
      />
    </>
  );
}
