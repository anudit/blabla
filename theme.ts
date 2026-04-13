import type { JSX } from 'preact';

export const VOICES = [
  { value: 'Bella',  label: 'Bella (Eng F)' },
  { value: 'Jasper', label: 'Jasper (Eng M)' },
  { value: 'Luna',   label: 'Luna (Eng F)' },
  { value: 'Bruno',  label: 'Bruno (Eng M)' },
  { value: 'Rosie',  label: 'Rosie (Eng F)' },
  { value: 'Hugo',   label: 'Hugo (Eng M)' },
  { value: 'Kiki',   label: 'Kiki (Eng F)' },
  { value: 'Leo',    label: 'Leo (Eng M)' },
];

export const staticStyles = {
  canvas: { display: 'block', width: '100%', height: 'auto' } as JSX.CSSProperties,
  overlay: {
    position: 'absolute' as const, top: 0, left: 0, right: 0, bottom: 0,
    zIndex: 10, pointerEvents: 'none' as const,
  } as JSX.CSSProperties,
  lineBase: {
    position: 'absolute' as const, cursor: 'pointer', borderRadius: '2px',
    // REMOVED backgroundColor: 'transparent' to prevent specificity wars with classes
    transition: 'background-color 0.2s ease',
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

export type ThemeName = 'original' | 'quiet' | 'paper' | 'bold' | 'calm' | 'focus';

export interface ThemeTokens {
  isDark: boolean;
  bg: string;
  text: string;
  textMuted: string;
  dropBg: string;
  dropBorder: string;
  epubBg: string;
  headerColor: string;
  menuBg: string;
  menuBorder: string;
  selectBg: string;
  selectBorder: string;
  inputBg: string;
  inputBorder: string;
  statBorder: string;
  resetBtnBg: string;
  testBtnBg: string;
  testBtnColor: string;
  testBtnBorder: string;
  pageShadow: string;
  speedHighlight: string;
  barBg: string;
  barBorder: string;
  barIconColor: string;
  barSpeedBg: string;
  barSpeedColor: string;
}

export const THEME_META: Record<ThemeName, { label: string; previewBg: string; previewText: string }> = {
  original: { label: 'Original', previewBg: '#f5efe3', previewText: '#3a3028' },
  quiet:    { label: 'Quiet',    previewBg: '#1a1917', previewText: '#c8bfb0' },
  paper:    { label: 'Paper',    previewBg: '#ffffff', previewText: '#111111' },
  bold:     { label: 'Bold',     previewBg: '#0d0d0d', previewText: '#f0f0f0' },
  calm:     { label: 'Calm',     previewBg: '#e8d5b5', previewText: '#3a2c1a' },
  focus:    { label: 'Focus',    previewBg: '#f0f0ec', previewText: '#282828' },
};

export const THEMES: Record<ThemeName, ThemeTokens> = {
  original: {
    isDark: false,
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
  quiet: {
    isDark: true,
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
  paper: {
    isDark: false,
    bg:            '#ffffff',
    text:          '#111111',
    textMuted:     '#777777',
    dropBg:        '#f8f8f8',
    dropBorder:    '#d0d0d0',
    epubBg:        '#ffffff',
    headerColor:   '#000000',
    menuBg:        '#f5f5f5',
    menuBorder:    '#e0e0e0',
    selectBg:      '#ffffff',
    selectBorder:  '#cccccc',
    inputBg:       '#ffffff',
    inputBorder:   '#cccccc',
    statBorder:    '#eeeeee',
    resetBtnBg:    '#f0f0f0',
    testBtnBg:     '#e8f0ff',
    testBtnColor:  '#2563eb',
    testBtnBorder: '#bfdbfe',
    pageShadow:    '0 2px 8px rgba(0,0,0,0.1)',
    speedHighlight:'#dbeafe',
    barBg:         '#1a1a1a',
    barBorder:     '#0a0a0a',
    barIconColor:  '#aaaaaa',
    barSpeedBg:    '#2a2a2a',
    barSpeedColor: '#eeeeee',
  },
  bold: {
    isDark: true,
    bg:            '#0d0d0d',
    text:          '#f0f0f0',
    textMuted:     '#888888',
    dropBg:        '#1a1a1a',
    dropBorder:    '#333333',
    epubBg:        '#0d0d0d',
    headerColor:   '#ffffff',
    menuBg:        '#1a1a1a',
    menuBorder:    '#2a2a2a',
    selectBg:      '#1a1a1a',
    selectBorder:  '#333333',
    inputBg:       '#1a1a1a',
    inputBorder:   '#333333',
    statBorder:    '#222222',
    resetBtnBg:    '#2a2a2a',
    testBtnBg:     '#1a2840',
    testBtnColor:  '#60a5fa',
    testBtnBorder: '#1e40af',
    pageShadow:    '0 2px 8px rgba(0,0,0,0.8)',
    speedHighlight:'#1e3a5f',
    barBg:         '#f0f0f0',
    barBorder:     '#e0e0e0',
    barIconColor:  '#444444',
    barSpeedBg:    '#e0e0e0',
    barSpeedColor: '#111111',
  },
  calm: {
    isDark: false,
    bg:            '#e8d5b5',
    text:          '#3a2c1a',
    textMuted:     '#7a6040',
    dropBg:        '#f0e2c5',
    dropBorder:    '#c8a878',
    epubBg:        '#e8d5b5',
    headerColor:   '#2a1c0a',
    menuBg:        '#e0cca8',
    menuBorder:    '#c0a070',
    selectBg:      '#ecddb8',
    selectBorder:  '#c8a878',
    inputBg:       '#ecddb8',
    inputBorder:   '#c8a878',
    statBorder:    '#d8c090',
    resetBtnBg:    '#d8c090',
    testBtnBg:     '#dde8f5',
    testBtnColor:  '#2563eb',
    testBtnBorder: '#bfdbfe',
    pageShadow:    '0 2px 8px rgba(80,50,20,0.15)',
    speedHighlight:'#c8dcf0',
    barBg:         '#3a2c1a',
    barBorder:     '#2a1c0a',
    barIconColor:  '#c8a878',
    barSpeedBg:    '#4a3c2a',
    barSpeedColor: '#e8d5b5',
  },
  focus: {
    isDark: false,
    bg:            '#f0f0ec',
    text:          '#282828',
    textMuted:     '#686860',
    dropBg:        '#f8f8f4',
    dropBorder:    '#c8c8c0',
    epubBg:        '#f0f0ec',
    headerColor:   '#181818',
    menuBg:        '#e8e8e4',
    menuBorder:    '#cccccc',
    selectBg:      '#f4f4f0',
    selectBorder:  '#c8c8c0',
    inputBg:       '#f4f4f0',
    inputBorder:   '#c8c8c0',
    statBorder:    '#dcdcd8',
    resetBtnBg:    '#dcdcd8',
    testBtnBg:     '#e8f0ff',
    testBtnColor:  '#2563eb',
    testBtnBorder: '#bfdbfe',
    pageShadow:    '0 2px 8px rgba(0,0,0,0.08)',
    speedHighlight:'#d4e4f4',
    barBg:         '#282828',
    barBorder:     '#181818',
    barIconColor:  '#a8a8a0',
    barSpeedBg:    '#383838',
    barSpeedColor: '#e8e8e4',
  },
};

/** Shared CSS transition applied to every themed element */
export const TT = 'background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease';
