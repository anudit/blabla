import type { JSX } from 'preact';

export const VOICES = [
  { value: 'af_bella', label: 'Bella (Eng F)' },
  { value: 'af_heart', label: 'Heart (Eng F)' },
  { value: 'am_fenrir', label: 'Fenrir (Eng M)' },
  { value: 'am_puck', label: 'Puck (Eng M)' },
];

export const staticStyles = {
  canvas: { display: 'block', width: '100%', height: 'auto' } as JSX.CSSProperties,
  overlay: {
    position: 'absolute' as const, top: 0, left: 0, right: 0, bottom: 0,
    zIndex: 10, pointerEvents: 'none' as const,
  } as JSX.CSSProperties,
  lineBase: {
    position: 'absolute' as const, cursor: 'pointer', borderRadius: '2px',
    backgroundColor: 'transparent', transition: 'background-color 0.2s ease',
    pointerEvents: 'auto' as const,
  } as JSX.CSSProperties,
  buttonDisabled: { opacity: 0.5, cursor: 'not-allowed' as const, filter: 'grayscale(1)' } as JSX.CSSProperties,
  playButton: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: '42px', height: '42px', borderRadius: '50%',
    backgroundColor: '#2563eb', color: 'white', border: 'none', cursor: 'pointer' as const,
    boxShadow: '0 2px 4px rgba(37,99,235,0.3)', transition: 'transform 0.1s',
    flexShrink: 0,
  } as JSX.CSSProperties,
  statusLoading: { color: '#2563eb', backgroundColor: '#eff6ff', border: '1px solid #bfdbfe' },
  statusReady:   { color: '#059669', backgroundColor: '#d1fae5', border: '1px solid #6ee7b7' },
  statusFallback:{ color: '#d97706', backgroundColor: '#fef3c7', border: '1px solid #fde68a' },
};

export const THEMES = {
  light: {
    bg:            '#f5efe3',
    text:          '#3a3028',
    textMuted:     '#7a6e60',
    dropBg:        '#faf6ef',
    dropBorder:    '#c4b8a0',
    epubBg:        '#f5efe3',
    headerColor:   '#2c2218',
    menuBg:        '#f0ebe0',
    menuBorder:    '#d0c4b0',
    selectBg:      '#faf6ef',
    selectBorder:  '#c4b8a0',
    inputBg:       '#faf6ef',
    inputBorder:   '#c4b8a0',
    statBorder:    '#e0d8c8',
    resetBtnBg:    '#e8e0d0',
    testBtnBg:     '#e8f0ff',
    testBtnColor:  '#2563eb',
    testBtnBorder: '#bfdbfe',
    pageShadow:    '0 2px 8px rgba(100,80,60,0.12)',
    speedHighlight:'#d8e8f4',
    barBg:         '#2a2015',
    barBorder:     '#1a1510',
    barIconColor:  '#b8ac9c',
    barSpeedBg:    '#3a3020',
    barSpeedColor: '#e8ddd0',
  },
  dark: {
    bg:            '#1a1917',
    text:          '#c8bfb0',
    textMuted:     '#8a8070',
    dropBg:        '#242018',
    dropBorder:    '#4a4235',
    epubBg:        '#1a1917',
    headerColor:   '#e8ddd0',
    menuBg:        '#242018',
    menuBorder:    '#3a3428',
    selectBg:      '#2a2520',
    selectBorder:  '#4a4235',
    inputBg:       '#2a2520',
    inputBorder:   '#4a4235',
    statBorder:    '#2e2a22',
    resetBtnBg:    '#3a3428',
    testBtnBg:     '#1e2d45',
    testBtnColor:  '#93c5fd',
    testBtnBorder: '#1d4ed8',
    pageShadow:    '0 2px 8px rgba(0,0,0,0.5)',
    speedHighlight:'#2a3a50',
    barBg:         '#ede8df',
    barBorder:     '#d0c8b8',
    barIconColor:  '#5a4f44',
    barSpeedBg:    '#ddd8cf',
    barSpeedColor: '#2a2015',
  },
};

export type ThemeTokens = typeof THEMES.light;

/** Shared CSS transition applied to every themed element */
export const TT = 'background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease';
