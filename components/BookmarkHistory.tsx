import React from 'react';
import { Clock, Trash2, BookOpen, FileText, Globe } from 'lucide-react';

export interface BookmarkEntry {
  id: string;
  fileName: string;
  sentenceIndex: number;
  totalSentences: number;
  timestamp: number;
  fileType: 'pdf' | 'epub' | 'url';
  preview: string;
  url?: string;
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
  onSelect: (entry: BookmarkEntry) => void;
  onDelete: (id: string) => void;
  isDarkMode: boolean;
}

const ICON = { epub: BookOpen, pdf: FileText, url: Globe } as const;

export default function BookmarkHistory({ bookmarks, onSelect, onDelete, isDarkMode }: Props) {
  if (bookmarks.length === 0) return null;

  const c = isDarkMode ? {
    label:        '#6a6058',
    containerBg:  '#242018',
    containerBorder: '#3a3428',
    divider:      '#302c24',
    text:         '#c8bfb0',
    textMuted:    '#7a6e60',
    textFaint:    '#5a5248',
    iconColor:    '#6a6058',
    progressBg:   '#3a3428',
    progressFill: '#d4a000',
    tagBg:        '#302c24',
    tagColor:     '#6a6058',
    deleteFaint:  '#4a4238',
    hoverBg:      '#2c2820',
  } : {
    label:        '#9a8e80',
    containerBg:  '#faf6ef',
    containerBorder: '#e0d8c8',
    divider:      '#ede8df',
    text:         '#3a3028',
    textMuted:    '#7a6e60',
    textFaint:    '#b0a898',
    iconColor:    '#b0a898',
    progressBg:   '#e8e0d0',
    progressFill: '#c89800',
    tagBg:        '#ede8df',
    tagColor:     '#9a8e80',
    deleteFaint:  '#c8bfb0',
    hoverBg:      '#f5efe3',
  };

  const hasLocalFiles = bookmarks.some(b => b.fileType !== 'url');

  return (
    <div style={{ width: '90%', maxWidth: '38rem', marginTop: '1.25rem', marginBottom: '3rem' }}>

      {/* Section label */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginBottom: '0.5rem', paddingLeft: '0.1rem' }}>
        <Clock size={11} color={c.label} />
        <span style={{ fontSize: '0.68rem', fontWeight: 600, color: c.label, letterSpacing: '0.06em', textTransform: 'uppercase' as const }}>
          Continue Reading
        </span>
      </div>

      {/* Grouped list container */}
      <div style={{
        borderRadius: '1rem',
        border: `1px solid ${c.containerBorder}`,
        backgroundColor: c.containerBg,
        overflow: 'hidden',
      }}>
        {bookmarks.map((entry, idx) => {
          const pct = Math.min(100, Math.round((entry.sentenceIndex / Math.max(entry.totalSentences, 1)) * 100));
          const Icon = ICON[entry.fileType] ?? FileText;
          const isUrl = entry.fileType === 'url';
          const isLast = idx === bookmarks.length - 1;

          return (
            <div key={entry.id}>
              <div
                onClick={() => isUrl && onSelect(entry)}
                style={{
                  padding: '0.85rem 1rem',
                  display: 'flex', gap: '0.75rem', alignItems: 'center',
                  cursor: isUrl ? 'pointer' : 'default',
                  backgroundColor: 'transparent',
                  transition: 'background-color 0.12s',
                  userSelect: 'none' as const,
                }}
                onMouseEnter={e => { if (isUrl) (e.currentTarget as HTMLDivElement).style.backgroundColor = c.hoverBg; }}
                onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.backgroundColor = 'transparent'; }}
              >
                {/* Icon */}
                <Icon size={15} color={c.iconColor} style={{ flexShrink: 0, marginTop: '1px' }} />

                {/* Content */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  {/* Title row */}
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginBottom: '0.2rem' }}>
                    <span style={{
                      fontSize: '0.82rem', fontWeight: 500, color: c.text,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
                    }}>
                      {entry.fileName}
                    </span>
                    <span style={{ fontSize: '0.66rem', color: c.textFaint, flexShrink: 0 }}>
                      {timeAgo(entry.timestamp)}
                    </span>
                  </div>

                  {/* Domain for URLs */}
                  {isUrl && entry.url && (
                    <p style={{
                      fontSize: '0.69rem', color: c.textMuted, margin: '0 0 0.18rem',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {(() => { try { return new URL(entry.url).hostname; } catch { return entry.url; } })()}
                    </p>
                  )}

                  {/* Preview */}
                  {entry.preview && (
                    <p style={{
                      fontSize: '0.72rem', color: c.textMuted, fontStyle: 'italic',
                      margin: '0 0 0.3rem',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      "{entry.preview}{entry.preview.length >= 80 ? '…' : ''}"
                    </p>
                  )}

                  {/* Progress row */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <div style={{ flex: 1, height: '2px', borderRadius: '999px', backgroundColor: c.progressBg, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${pct}%`, backgroundColor: c.progressFill, borderRadius: '999px' }} />
                    </div>
                    <span style={{ fontSize: '0.65rem', color: c.textFaint, flexShrink: 0 }}>{pct}%</span>
                    <span style={{
                      fontSize: '0.58rem', fontWeight: 600, color: c.tagColor, backgroundColor: c.tagBg,
                      padding: '0.08rem 0.3rem', borderRadius: '999px',
                      textTransform: 'uppercase' as const, letterSpacing: '0.05em', flexShrink: 0,
                    }}>
                      {entry.fileType}
                    </span>
                  </div>
                </div>

                {/* Delete */}
                <button
                  onClick={e => { e.stopPropagation(); onDelete(entry.id); }}
                  title="Remove"
                  style={{
                    flexShrink: 0, background: 'none', border: 'none', cursor: 'pointer',
                    color: c.deleteFaint, padding: '0.25rem', borderRadius: '0.375rem',
                    display: 'flex', alignItems: 'center', transition: 'color 0.12s',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.color = '#ef4444')}
                  onMouseLeave={e => (e.currentTarget.style.color = c.deleteFaint)}
                >
                  <Trash2 size={13} />
                </button>
              </div>

              {/* Hairline divider between items */}
              {!isLast && <div style={{ height: '1px', backgroundColor: c.divider, marginLeft: '2.5rem' }} />}
            </div>
          );
        })}
      </div>

      {hasLocalFiles && (
        <p style={{ fontSize: '0.66rem', color: c.label, marginTop: '0.5rem', textAlign: 'center' as const }}>
          Drop the same file again to resume from where you left off
        </p>
      )}
    </div>
  );
}
