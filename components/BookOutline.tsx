import { useEffect, useRef, useState } from 'preact/hooks';
import { AlignJustify, X } from 'lucide-preact';
import type { Signal } from '@preact/signals';

export interface OutlineEntry {
  id: string;
  text: string;
  level?: number; // 1-6; present for markdown headings
}

interface Props {
  entries: OutlineEntry[];
  activeId: Signal<string | null>;
  isDarkMode: boolean;
}

export default function BookOutline({ entries, activeId, isDarkMode }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 640);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const currentActiveId = activeId.value;

  // Track mobile breakpoint
  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 640);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Close on outside click / touch
  useEffect(() => {
    if (!isOpen) return;
    const onPointer = (e: MouseEvent | TouchEvent) => {
      const target = e.target as Node;
      if (wrapperRef.current && !wrapperRef.current.contains(target)) setIsOpen(false);
    };
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('touchstart', onPointer);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('touchstart', onPointer);
    };
  }, [isOpen]);

  // Scroll active item into view inside panel
  useEffect(() => {
    if (!isOpen || !currentActiveId || !panelRef.current) return;
    const el = panelRef.current.querySelector(`[data-id="${currentActiveId}"]`) as HTMLElement | null;
    if (el) el.scrollIntoView({ block: 'nearest' });
  }, [currentActiveId, isOpen]);

  if (entries.length === 0) return null;

  const t = isDarkMode ? {
    toggleBg:      '#2a2015',
    toggleColor:   '#c0b4a4',
    toggleBorder:  '#1a1510',
    panelBg:       '#1e1c17',
    panelBorder:   '#3a3428',
    label:         '#6a6058',
    text:          '#c8bfb0',
    textMuted:     '#8a8070',
    activeColor:   '#7eb8f0',
    activeBg:      'rgba(126,184,240,0.10)',
    hoverBg:       'rgba(255,255,255,0.05)',
    divider:       '#3a3428',
  } : {
    toggleBg:      '#2a2015',
    toggleColor:   '#c0b4a4',
    toggleBorder:  '#1a1510',
    panelBg:       '#faf6ef',
    panelBorder:   '#d4c8b4',
    label:         '#a09080',
    text:          '#3a3028',
    textMuted:     '#7a6e60',
    activeColor:   '#2563eb',
    activeBg:      'rgba(37,99,235,0.07)',
    hoverBg:       'rgba(0,0,0,0.04)',
    divider:       '#e0d8c8',
  };

  // Panel width: on mobile cap to viewport, on desktop fixed 252px
  const panelW = isMobile ? `min(85vw, 320px)` : '252px';

  return (
    <div
      ref={wrapperRef}
      style={{ position: 'fixed', top: '1rem', left: '1rem', zIndex: 200 }}
    >
      {/* ── Toggle pill ── */}
      <button
        onClick={() => setIsOpen(o => !o)}
        title={isOpen ? 'Close contents' : 'Table of contents'}
        style={{
          display: 'flex', alignItems: 'center', gap: '0.35rem',
          padding: '0.38rem 0.7rem',
          backgroundColor: t.toggleBg,
          color: t.toggleColor,
          border: `1px solid ${t.toggleBorder}`,
          borderRadius: '999px',
          cursor: 'pointer',
          fontSize: '0.72rem',
          fontWeight: 700,
          letterSpacing: '0.03em',
          boxShadow: '0 2px 10px rgba(0,0,0,0.25)',
          transition: 'opacity 0.15s',
        }}
      >
        {isOpen ? <X size={13} /> : <AlignJustify size={13} />}
      </button>

      {/* ── Collapsible panel ── */}
      {isOpen && (
        <div
          ref={panelRef}
          style={{
            position: 'absolute',
            top: 'calc(100% + 0.5rem)',
            left: 0,
            width: panelW,
            maxHeight: isMobile ? '70vh' : '78vh',
            overflowY: 'auto',
            backgroundColor: t.panelBg,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: '0.85rem',
            boxShadow: '0 12px 40px rgba(0,0,0,0.22)',
            // scrollbar styling (webkit)
          }}
        >
          {/* Label row */}
          <div style={{
            padding: '0.65rem 1rem 0.5rem',
            fontSize: '0.62rem',
            fontWeight: 800,
            color: t.label,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            borderBottom: `1px solid ${t.divider}`,
            position: 'sticky',
            top: 0,
            backgroundColor: t.panelBg,
            zIndex: 1,
          }}>
            Contents
          </div>

          {/* Entries */}
          <div style={{ padding: '0.35rem 0' }}>
            {entries.map(entry => {
              const active = entry.id === currentActiveId;
              return (
                <button
                  key={entry.id}
                  data-id={entry.id}
                  onClick={() => {
                    document.getElementById(entry.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    if (isMobile) setIsOpen(false);
                  }}
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    paddingTop: '0.42rem',
                    paddingBottom: '0.42rem',
                    paddingRight: '1rem',
                    // Indent sub-headings: each level beyond 1 adds 0.75rem
                    paddingLeft: `calc(1rem + ${((entry.level ?? 1) - 1) * 0.75}rem)`,
                    background: active ? t.activeBg : 'none',
                    border: 'none',
                    borderLeft: `2px solid ${active ? t.activeColor : 'transparent'}`,
                    cursor: 'pointer',
                    color: active ? t.activeColor : (entry.level && entry.level > 1 ? t.textMuted : t.text),
                    fontSize: entry.level && entry.level > 2 ? '0.76rem' : '0.82rem',
                    fontWeight: active ? 600 : (entry.level === 1 || !entry.level ? 500 : 400),
                    lineHeight: 1.45,
                    transition: 'background 0.12s, color 0.12s',
                  }}
                  onMouseEnter={e => { if (!active) e.currentTarget.style.background = t.hoverBg; }}
                  onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'none'; }}
                >
                  {entry.text}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
