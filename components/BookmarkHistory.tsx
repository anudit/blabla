import React from 'react';
import { Clock, Trash2, BookOpen, FileText } from 'lucide-react';

export interface BookmarkEntry {
  id: string;           // fileName + ':' + fileSize
  fileName: string;
  sentenceIndex: number;
  totalSentences: number;
  timestamp: number;
  fileType: 'pdf' | 'epub';
  preview: string;      // text snippet at saved position (max 80 chars)
}

// ── LocalStorage helpers ───────────────────────────────────────────────
const STORAGE_KEY = 'blabla_bookmarks';

export const getBookmarks = (): BookmarkEntry[] => {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
  catch { return []; }
};

export const saveBookmark = (entry: BookmarkEntry) => {
  const rest = getBookmarks().filter(b => b.id !== entry.id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify([entry, ...rest].slice(0, 20)));
};

export const removeBookmark = (id: string) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(getBookmarks().filter(b => b.id !== id)));
};

// ── Relative time helper ───────────────────────────────────────────────
function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const m = Math.floor(diff / 60_000);
  if (m < 1)  return 'Just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7)  return `${d}d ago`;
  return new Date(ts).toLocaleDateString();
}

// ── Component ──────────────────────────────────────────────────────────
interface Props {
  bookmarks: BookmarkEntry[];
  onDelete: (id: string) => void;
  isDarkMode: boolean;
}

export default function BookmarkHistory({ bookmarks, onDelete, isDarkMode }: Props) {
  if (bookmarks.length === 0) return null;

  const t = isDarkMode ? {
    sectionText:    '#8a8070',
    itemBg:         '#242018',
    itemBgHover:    '#2e2a22',
    itemBorder:     '#3a3428',
    text:           '#c8bfb0',
    textMuted:      '#8a8070',
    iconBg:         '#3a3428',
    progressBg:     '#3a3428',
    progressFill:   '#e8b800',
    tagBg:          '#3a3020',
    tagColor:       '#8a8070',
    deleteColor:    '#6a6058',
  } : {
    sectionText:    '#7a6e60',
    itemBg:         '#fff9f2',
    itemBgHover:    '#f5efe3',
    itemBorder:     '#e0d8c8',
    text:           '#3a3028',
    textMuted:      '#7a6e60',
    iconBg:         '#ede8df',
    progressBg:     '#e0d8c8',
    progressFill:   '#d4a000',
    tagBg:          '#ede8df',
    tagColor:       '#7a6e60',
    deleteColor:    '#c0b4a4',
  };

  return (
    <div style={{ width: '90%', maxWidth: '42rem', marginTop: '1.5rem', marginBottom: '2rem' }}>

      {/* Section label */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.6rem' }}>
        <Clock size={13} color={t.sectionText} />
        <span style={{
          fontSize: '0.7rem', fontWeight: 700, color: t.sectionText,
          letterSpacing: '0.07em', textTransform: 'uppercase',
        }}>
          Continue Reading
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
        {bookmarks.map(entry => {
          const pct = Math.min(100, Math.round((entry.sentenceIndex / Math.max(entry.totalSentences, 1)) * 100));
          const Icon = entry.fileType === 'epub' ? BookOpen : FileText;

          return (
            <div
              key={entry.id}
              style={{
                backgroundColor: t.itemBg,
                border: `1px solid ${t.itemBorder}`,
                borderRadius: '0.75rem',
                padding: '0.7rem 0.85rem',
                display: 'flex',
                gap: '0.7rem',
                alignItems: 'center',
                transition: 'background-color 0.15s',
                userSelect: 'none',
              }}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = t.itemBgHover)}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = t.itemBg)}
            >
              {/* File type icon */}
              <div style={{
                flexShrink: 0,
                width: '34px', height: '34px', borderRadius: '0.5rem',
                backgroundColor: t.iconBg,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Icon size={16} color={t.textMuted} />
              </div>

              {/* Main content */}
              <div style={{ flex: 1, minWidth: 0 }}>

                {/* Top row: filename + timestamp */}
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginBottom: '0.15rem' }}>
                  <span style={{
                    fontSize: '0.82rem', fontWeight: 600, color: t.text,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
                  }}>
                    {entry.fileName}
                  </span>
                  <span style={{ fontSize: '0.68rem', color: t.textMuted, flexShrink: 0 }}>
                    {timeAgo(entry.timestamp)}
                  </span>
                </div>

                {/* Preview snippet */}
                {entry.preview && (
                  <p style={{
                    fontSize: '0.75rem', color: t.textMuted, fontStyle: 'italic',
                    margin: '0 0 0.35rem',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    "{entry.preview}{entry.preview.length >= 80 ? '…' : ''}"
                  </p>
                )}

                {/* Progress bar row */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{
                    flex: 1, height: '3px', borderRadius: '999px',
                    backgroundColor: t.progressBg, overflow: 'hidden',
                  }}>
                    <div style={{
                      height: '100%', width: `${pct}%`,
                      backgroundColor: t.progressFill,
                      borderRadius: '999px',
                    }} />
                  </div>
                  <span style={{ fontSize: '0.68rem', color: t.textMuted, flexShrink: 0 }}>
                    {pct}%
                  </span>
                  <span style={{
                    fontSize: '0.6rem', fontWeight: 700,
                    color: t.tagColor, backgroundColor: t.tagBg,
                    padding: '0.1rem 0.35rem', borderRadius: '999px',
                    textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0,
                  }}>
                    {entry.fileType}
                  </span>
                </div>
              </div>

              {/* Delete button */}
              <button
                onClick={e => { e.stopPropagation(); onDelete(entry.id); }}
                title="Remove from history"
                style={{
                  flexShrink: 0, background: 'none', border: 'none', cursor: 'pointer',
                  color: t.deleteColor, padding: '0.3rem', borderRadius: '0.375rem',
                  display: 'flex', alignItems: 'center', transition: 'color 0.15s',
                }}
                onMouseEnter={e => (e.currentTarget.style.color = '#ef4444')}
                onMouseLeave={e => (e.currentTarget.style.color = t.deleteColor)}
              >
                <Trash2 size={14} />
              </button>
            </div>
          );
        })}
      </div>

      <p style={{
        fontSize: '0.68rem', color: t.textMuted, marginTop: '0.6rem',
        textAlign: 'center',
      }}>
        Drop the same file again to resume from where you left off
      </p>
    </div>
  );
}
