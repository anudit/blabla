import { useState, useEffect } from 'preact/hooks';
import type { JSX } from 'preact';
import { Play, Pause, Loader2, Menu, Palette, Beaker } from 'lucide-preact';
import type { ThemeTokens, ThemeName } from '../theme';
import { TT, staticStyles, VOICES, THEMES, THEME_META } from '../theme';
import {
  isPlayingSignal, playbackStateSignal, ttsStatusSignal,
  isModelReadySignal, currentSentenceIndexSignal,
} from '../signals';

interface BottomBarProps {
  t: ThemeTokens;
  isDarkMode: boolean;
  themeName: ThemeName;
  onThemeChange: (name: ThemeName) => void;
  playbackSpeed: number;
  hasSentences: boolean;
  onTogglePlay: () => void;
  onSpeedChange: (speed: number) => void;
  usingFallback: boolean;
  selectedVoice: string;
  onVoiceChange: (e: JSX.TargetedEvent<HTMLSelectElement, Event>) => void;
  sentencesLength: number;
  fontSize: number;
  onFontSizeChange: (delta: number) => void;
  onTestAudio: () => void;
  onReset: () => void;
  onLogoClick: () => void;
}

export default function BottomBar({
  t, isDarkMode, themeName, onThemeChange,
  playbackSpeed, hasSentences,
  onTogglePlay, onSpeedChange,
  usingFallback,
  selectedVoice, onVoiceChange,
  sentencesLength,
  fontSize, onFontSizeChange,
  onTestAudio, onReset, onLogoClick,
}: BottomBarProps) {
  const isPlaying     = isPlayingSignal.value;
  const isModelReady  = isModelReadySignal.value;
  const playbackState = playbackStateSignal.value;

  const [isSpeedMenuOpen, setIsSpeedMenuOpen]     = useState(false);
  const [isMenuOpen, setIsMenuOpen]               = useState(false);
  const [isThemePickerOpen, setIsThemePickerOpen] = useState(false);
  const [memGb, setMemGb]   = useState<number | null>(null);
  const [cpuPct, setCpuPct] = useState<number | null>(null);

  useEffect(() => {
    if (!isMenuOpen) return;
    let lastTick = performance.now();
    const id = setInterval(() => {
      const mem = (performance as any).memory;
      if (mem) setMemGb(+(mem.usedJSHeapSize / 1e9).toFixed(2));
      const now = performance.now();
      const drift = Math.max(0, now - lastTick - 1000);
      setCpuPct(Math.min(100, Math.round(drift / 10)));
      lastTick = now;
    }, 1000);
    return () => clearInterval(id);
  }, [isMenuOpen]);

  const closeAll = () => { setIsSpeedMenuOpen(false); setIsMenuOpen(false); setIsThemePickerOpen(false); };

  const iconButtonStyle: JSX.CSSProperties = {
    padding: '0.5rem', borderRadius: '0.375rem', border: 'none', cursor: 'pointer',
    backgroundColor: 'transparent', color: t.barIconColor, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: TT,
  };

  const speedButtonStyle: JSX.CSSProperties = {
    fontSize: '0.8rem', fontWeight: 700, color: t.barSpeedColor, backgroundColor: t.barSpeedBg,
    padding: '0.25rem 0.5rem', borderRadius: '0.25rem', border: 'none', minWidth: '2.5rem', cursor: 'pointer', transition: TT,
  };

  const popoverBase: JSX.CSSProperties = {
    position: 'absolute', bottom: 'calc(100% + 10px)', overflow: 'hidden', backgroundColor: t.menuBg,
    borderRadius: '0.875rem', border: `1px solid ${t.menuBorder}`,
    boxShadow: isDarkMode ? '0 8px 24px rgba(0,0,0,0.55), 0 2px 6px rgba(0,0,0,0.3)' : '0 8px 24px rgba(0,0,0,0.1), 0 2px 6px rgba(0,0,0,0.06)',
  };

  return (
    <div style={{
      position: 'fixed', bottom: '1.5rem', left: '50%', transform: 'translateX(-50%)',
      backgroundColor: t.barBg, padding: '0.4rem 0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem',
      boxShadow: '0 4px 20px rgba(0,0,0,0.28)', zIndex: 50, borderRadius: '999px', border: `1px solid ${t.barBorder}`, whiteSpace: 'nowrap', transition: TT,
    }}>
      <button onClick={onLogoClick} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center' }}>
        <img src="./180.png" style={{ width: '32px', height: '32px' }} alt="logo" />
      </button>

      <button onClick={() => { const next = !isSpeedMenuOpen; closeAll(); setIsSpeedMenuOpen(next); }} style={speedButtonStyle}>
        {playbackSpeed}x
      </button>

      <button
        onClick={onTogglePlay}
        disabled={!hasSentences || !isModelReady}
        style={{ ...staticStyles.playButton, ...(!hasSentences || !isModelReady ? staticStyles.buttonDisabled : {}) }}
      >
        {playbackState === 'Buffering' ? <Loader2 size={20} color="white" style={{ animation: 'spin 0.8s linear infinite' }} /> : isPlaying ? <Pause size={20} fill="white" /> : <Play size={20} fill="white" style={{ marginLeft: '2px' }} />}
      </button>

      <button onClick={() => { const next = !isMenuOpen; closeAll(); setIsMenuOpen(next); }} style={iconButtonStyle}>
        <Menu size={24} />
      </button>

      <button onClick={() => { const next = !isThemePickerOpen; closeAll(); setIsThemePickerOpen(next); }} style={iconButtonStyle} title="Choose theme">
        <Palette size={20} />
      </button>

      {isSpeedMenuOpen && (
        <div style={{ ...popoverBase, left: '2rem', width: '90px' }}>
          {[1.0, 1.25, 1.5, 1.75, 2.0].map((speed, idx, arr) => {
            const active = playbackSpeed === speed;
            const isLast = idx === arr.length - 1;
            return (
              <div key={speed}>
                <button
                  onClick={() => { onSpeedChange(speed); setIsSpeedMenuOpen(false); }}
                  style={{ width: '100%', padding: '0.6rem 0.9rem', background: 'none', border: 'none', cursor: 'pointer', fontSize: '0.85rem', fontWeight: active ? 600 : 400, color: active ? '#2563eb' : t.text, textAlign: 'left', display: 'flex', alignItems: 'center', justifyContent: 'space-between', transition: 'color 0.1s' }}
                >
                  {speed}×
                  {active && <span style={{ fontSize: '0.7rem', color: '#2563eb' }}>✓</span>}
                </button>
                {!isLast && <div style={{ height: '1px', backgroundColor: t.menuBorder, marginLeft: '0.9rem', opacity: 0.6 }} />}
              </div>
            );
          })}
        </div>
      )}

      {isMenuOpen && (
        <SettingsMenu
          t={t}
          isDarkMode={isDarkMode}
          sentencesLength={sentencesLength}
          fontSize={fontSize}
          onFontSizeChange={onFontSizeChange}
          onVoiceChange={onVoiceChange}
          selectedVoice={selectedVoice}
          usingFallback={usingFallback}
          onTestAudio={onTestAudio}
          onReset={onReset}
          popoverBase={popoverBase}
          cpuPct={cpuPct}
          memGb={memGb}
          onClose={() => setIsMenuOpen(false)}
        />
      )}

      {isThemePickerOpen && (
        <ThemePicker
          t={t}
          isDarkMode={isDarkMode}
          themeName={themeName}
          onThemeChange={(name) => { onThemeChange(name); setIsThemePickerOpen(false); }}
          popoverBase={popoverBase}
        />
      )}
    </div>
  );
}

function ThemePicker({ t, isDarkMode, themeName, onThemeChange, popoverBase }: {
  t: ThemeTokens;
  isDarkMode: boolean;
  themeName: ThemeName;
  onThemeChange: (name: ThemeName) => void;
  popoverBase: JSX.CSSProperties;
}) {
  const themeNames = Object.keys(THEME_META) as ThemeName[];

  return (
    <div style={{ ...popoverBase, right: '0', width: '236px', padding: '0.75rem' }}>
      <div style={{ fontSize: '0.7rem', fontWeight: 600, color: t.textMuted, letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: '0.6rem', paddingLeft: '0.15rem' }}>
        Theme
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
        {themeNames.map((name) => {
          const meta = THEME_META[name];
          const theme = THEMES[name];
          const isActive = themeName === name;
          return (
            <button
              key={name}
              onClick={() => onThemeChange(name)}
              title={meta.label}
              style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: '0.3rem', padding: '0.6rem 0.4rem',
                backgroundColor: meta.previewBg,
                borderRadius: '0.625rem',
                border: isActive ? `2.5px solid ${isDarkMode ? '#e0d8c8' : '#2a2015'}` : `1.5px solid ${isDarkMode ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)'}`,
                cursor: 'pointer',
                transition: 'border-color 0.2s ease, transform 0.15s ease, box-shadow 0.15s ease',
                transform: isActive ? 'scale(1.03)' : 'scale(1)',
                boxShadow: isActive
                  ? (isDarkMode ? '0 0 0 1px rgba(224,216,200,0.3)' : '0 0 0 1px rgba(42,32,21,0.15)')
                  : 'none',
                outline: 'none',
              }}
            >
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: meta.previewText, lineHeight: 1, fontFamily: 'Georgia, serif', letterSpacing: '-0.01em' }}>
                Aa
              </span>
              <span style={{ fontSize: '0.62rem', fontWeight: 500, color: meta.previewText, opacity: 0.7, lineHeight: 1 }}>
                {meta.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function SettingsMenu({
  t, isDarkMode, sentencesLength, fontSize, onFontSizeChange,
  onVoiceChange, selectedVoice, usingFallback,
  onTestAudio, onReset, popoverBase, cpuPct, memGb, onClose,
}: any) {
  const currentSentenceIndex = currentSentenceIndexSignal.value;
  const ttsStatus          = ttsStatusSignal.value;
  const isModelReady       = isModelReadySignal.value;
  const playbackState      = playbackStateSignal.value;

  const row: JSX.CSSProperties = { padding: '0.55rem 1rem', display: 'flex', alignItems: 'center' };
  const lbl: JSX.CSSProperties = { fontSize: '0.75rem', color: t.textMuted, minWidth: '62px' };
  const val: JSX.CSSProperties = { fontSize: '0.75rem', fontWeight: 600, color: t.text, fontFamily: 'monospace' };
  const statusDot = isModelReady && !usingFallback ? '#22c55e' : usingFallback ? '#f59e0b' : '#3b82f6';

  return (
    <div style={{ ...popoverBase, right: '2.5rem', width: '248px' }}>
      <div style={{ ...row, gap: '0.5rem' }}>
        <div style={{ width: '6px', height: '6px', borderRadius: '50%', flexShrink: 0, backgroundColor: statusDot }} />
        <span style={{ fontSize: '0.75rem', color: t.textMuted, flex: 1 }}>{ttsStatus}</span>
      </div>
      <div style={{ height: '1px', backgroundColor: t.menuBorder, opacity: 0.5 }} />
      <div style={{ ...row, gap: '0.6rem' }}>
        <span style={lbl}>Voice</span>
        <select value={selectedVoice} onChange={onVoiceChange} disabled={usingFallback || !isModelReady} style={{ flex: 1, padding: '0.3rem 0.4rem', borderRadius: '0.4rem', border: `1px solid ${t.selectBorder}`, backgroundColor: t.selectBg, color: t.text, fontSize: '0.75rem', cursor: 'pointer', outline: 'none', ...(usingFallback || !isModelReady ? staticStyles.buttonDisabled : {}) }}>
          {VOICES.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
        </select>
      </div>
      <div style={{ height: '1px', backgroundColor: t.menuBorder, opacity: 0.5 }} />
      <div style={{ ...row, justifyContent: 'space-between' }}>
        <span style={lbl}>State</span>
        <span style={val}>{playbackState}</span>
      </div>
      <div style={{ ...row, justifyContent: 'space-between', paddingTop: '0.3rem' }}>
        <span style={lbl}>Progress</span>
        <span style={val}>{currentSentenceIndex >= 0 ? `${Math.round((currentSentenceIndex / sentencesLength) * 100)}%` : '0%'}</span>
      </div>
      <div style={{ ...row, justifyContent: 'space-between' }}>
        <span style={lbl}>CPU</span>
        <span style={val}>{cpuPct !== null ? `${cpuPct}%` : '—'}</span>
      </div>
      <div style={{ ...row, justifyContent: 'space-between', paddingBottom: '0.55rem' }}>
        <span style={lbl}>Memory</span>
        <span style={val}>{memGb !== null ? `${memGb} GB` : '—'}</span>
      </div>
      <div style={{ height: '1px', backgroundColor: t.menuBorder, opacity: 0.5 }} />
      <div style={{ ...row, justifyContent: 'space-between', gap: '0.5rem' }}>
        <span style={lbl}>Font size</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          {([['−', -0.05], ['+', 0.05]] as [string, number][]).map(([label, delta]) => (
            <button key={label} onClick={() => onFontSizeChange(delta)} style={{ width: '24px', height: '24px', borderRadius: '0.375rem', border: `1px solid ${t.menuBorder}`, background: 'none', cursor: 'pointer', color: t.text, fontSize: '1rem', lineHeight: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>{label}</button>
          ))}
          <span style={{ ...val, minWidth: '34px', textAlign: 'center' }}>{fontSize.toFixed(2)}x</span>
        </div>
      </div>
      <div style={{ height: '1px', backgroundColor: t.menuBorder, opacity: 0.5 }} />
      <div style={{ padding: '0.6rem 1rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        <button onClick={onTestAudio} disabled={!isModelReady} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', padding: '0.45rem', fontSize: '0.78rem', fontWeight: 500, backgroundColor: 'transparent', color: !isModelReady ? t.textMuted : t.text, border: `1px solid ${t.menuBorder}`, borderRadius: '0.5rem', cursor: !isModelReady ? 'not-allowed' : 'pointer', opacity: !isModelReady ? 0.5 : 1 }}>
          <Beaker size={13} /> Test Voice
        </button>
        <button onClick={() => { onClose(); onReset(); }} style={{ padding: '0.45rem', fontSize: '0.78rem', fontWeight: 500, backgroundColor: 'transparent', color: '#ef4444', border: '1px solid rgba(239,68,68,0.25)', borderRadius: '0.5rem', cursor: 'pointer' }}>Reset Document</button>
      </div>
    </div>
  );
}
