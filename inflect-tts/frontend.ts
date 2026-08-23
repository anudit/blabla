/**
 * Text frontend for Inflect-Micro-v2.
 * Port of inflect_nano_v2_frontend.py normalize_text + the eSpeak-ng
 * phonemization used by the official package, then symbol->id mapping
 * with blank interspersion (VITS add_blank).
 */
import { phonemize } from 'phonemizer';

// ── Symbol table (runtime/text/symbols.py) ──────────────────────────────────

const PAD = '_';
const PUNCTUATION = ';:,.!?¡¿—…"«»“” ';
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
const LETTERS_IPA =
  "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

export const SYMBOLS: string[] = [PAD, ...PUNCTUATION.split(''), ...LETTERS.split(''), ...LETTERS_IPA.split('')];

const SYMBOL_TO_ID = new Map<string, number>();
SYMBOLS.forEach((s, i) => SYMBOL_TO_ID.set(s, i));

// ── Number-to-words (subset of num2words for English) ───────────────────────

const ONES = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
  'sixteen', 'seventeen', 'eighteen', 'nineteen',
];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];

function words(n: number): string {
  if (n < 0) return `minus ${words(-n)}`;
  if (n < 20) return ONES[n];
  if (n < 100) {
    const t = TENS[Math.floor(n / 10)];
    return n % 10 ? `${t} ${ONES[n % 10]}` : t;
  }
  if (n < 1000) {
    const h = `${ONES[Math.floor(n / 100)]} hundred`;
    return n % 100 ? `${h} and ${words(n % 100)}` : h;
  }
  if (n < 1_000_000) {
    const th = `${words(Math.floor(n / 1000))} thousand`;
    return n % 1000 ? `${th} ${words(n % 1000)}` : th;
  }
  const mi = `${words(Math.floor(n / 1_000_000))} million`;
  return n % 1_000_000 ? `${mi} ${words(n % 1_000_000)}` : mi;
}

function ordinalWords(n: number): string {
  const ordinals: Record<string, string> = {
    one: 'first', two: 'second', three: 'third', five: 'fifth',
    eight: 'eighth', nine: 'ninth', twelve: 'twelfth',
  };
  if (n % 100 >= 11 && n % 100 <= 13) return `${words(n)}th`;
  const w = words(n);
  const last = w.split(' ').pop()!;
  const mod = ordinals[last] ?? (/y$/.test(last) ? `${last.slice(0, -1)}ieth` : `${last}th`);
  const parts = w.split(' ');
  parts[parts.length - 1] = mod;
  return parts.join(' ');
}

/** num2words equivalent with hyphen/comma removal. */
function numWords(n: number, ord = false): string {
  const text = ord ? ordinalWords(n) : words(n);
  return text.replace(/-/g, ' ').replace(/,/g, '');
}

function digitWords(text: string): string {
  return text
    .split('')
    .filter((ch) => ch >= '0' && ch <= '9')
    .map((ch) => ONES[Number(ch)])
    .join(' ');
}

function identifierDigits(text: string): string {
  const out: string[] = [];
  const chars = text.split('');
  chars.forEach((ch, i) => {
    if (!(ch >= '0' && ch <= '9')) return;
    out.push(ch === '0' && i > 0 ? 'oh' : ONES[Number(ch)]);
  });
  return out.join(' ');
}

// ── Text normalization (port of normalize_text) ─────────────────────────────

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

const WORD_OVERRIDES: [string, string][] = Object.entries({
  Qwen3: 'Qwen three', Qwen: 'Qwen', PyTorch: 'pie torch', SQLite: 'ess cue lite',
  'USB-C': 'you ess bee see', 'RTX 3060': 'ar tee ex thirty sixty',
  'RTX 3090': 'ar tee ex thirty ninety', 'RTX 4090': 'ar tee ex forty ninety',
  'RTX 5080': 'ar tee ex fifty eighty', 'RTX 5090': 'ar tee ex fifty ninety',
});

const ABBREVIATIONS: [string, string][] = Object.entries({
  'Dr.': 'doctor', 'Mr.': 'mister', 'Mrs.': 'missus', 'Ms.': 'miss',
  'Prof.': 'professor', 'St.': 'saint', 'vs.': 'versus', 'etc.': 'et cetera',
  'e.g.': 'for example', 'i.e.': 'that is',
});

const LETTER_NAMES: Record<string, string> = {
  A: 'ay', B: 'bee', C: 'see', D: 'dee', E: 'ee', F: 'eff', G: 'gee', H: 'aitch',
  I: 'eye', J: 'jay', K: 'kay', L: 'ell', M: 'em', N: 'en', O: 'oh', P: 'pee',
  Q: 'cue', R: 'ar', S: 'ess', T: 'tee', U: 'you', V: 'vee', W: 'double you',
  X: 'ex', Y: 'why', Z: 'zee',
};

const PUNCT_TRANSLATION: Record<string, string> = {
  '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
  '\u2013': '-', '\u2014': ', ', '\u2026': '...',
  '(': ', ', ')': ', ', '[': ', ', ']': ', ', '{': ', ', '}': ', ',
};

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function expandIdentifierToken(token: string): string {
  const m = token.match(/^([A-Za-z]?)(\d+)([A-Za-z]?)$/);
  if (!m) return token;
  const [, prefix, digits, suffix] = m;
  const pieces: string[] = [];
  if (prefix) pieces.push(LETTER_NAMES[prefix.toUpperCase()] ?? prefix);
  if (digits.length === 3 || digits.startsWith('0')) pieces.push(identifierDigits(digits));
  else pieces.push(numWords(Number(digits)));
  if (suffix) pieces.push(LETTER_NAMES[suffix.toUpperCase()] ?? suffix);
  return pieces.join(' ');
}

function expandMoney(raw: string): string {
  const cleaned = raw.replace(/,/g, '');
  const dotIdx = cleaned.indexOf('.');
  const dollarsPart = dotIdx >= 0 ? cleaned.slice(0, dotIdx) : cleaned;
  const centsPart = dotIdx >= 0 ? cleaned.slice(dotIdx + 1) : '';
  const dollarCount = parseInt(dollarsPart, 10);
  const parts = [numWords(dollarCount), dollarCount === 1 ? 'dollar' : 'dollars'];
  if (centsPart) {
    const cents = centsPart.slice(0, 2).padEnd(2, '0');
    const centCount = parseInt(cents, 10);
    if (centCount) parts.push('and', numWords(centCount), centCount === 1 ? 'cent' : 'cents');
  }
  return parts.join(' ');
}

function expandTime(hour: number, minute: number, suffixRaw: string | undefined): string {
  const pieces = [numWords(hour)];
  if (minute === 0) pieces.push('o clock');
  else if (minute < 10) pieces.push('oh', numWords(minute));
  else pieces.push(numWords(minute));
  if (suffixRaw) {
    const suffix = suffixRaw.toLowerCase().replace(/\./g, '');
    pieces.push(...suffix.split(''));
  }
  return pieces.join(' ');
}

function isValidDate(y: number, m: number, d: number): boolean {
  const dt = new Date(Date.UTC(y, m - 1, d));
  return dt.getUTCFullYear() === y && dt.getUTCMonth() === m - 1 && dt.getUTCDate() === d;
}

/**
 * Strip wiki/markdown junk so a citation dump doesn't reach the model.
 * Footnotes like `[\[18\]](#citenote-18)` and leftover leading `. "` would
 * otherwise inflate duration and poison GPU buffers on long lines.
 */
export function cleanForTts(raw: string): string {
  let s = raw;
  s = s.replace(/!\[[^\]]*\]\([^)]+\)/g, '');
  // [\[18\]](#citenote-18) and [18](#note)
  s = s.replace(/\[(?:\\\[)?\\?\[?\d+\\?\]?(?:\\\])?\]\([^)]*\)/g, '');
  s = s.replace(/\[([^\]]*)\]\([^)]+\)/g, '$1');
  s = s.replace(/\\?\[\s*\\?\d+\s*\\?\]/g, '');
  s = s.replace(/\(#[\w.-]+\)/g, '');
  s = s.replace(/#citenote-\d+/gi, '');
  s = s.replace(/\\\[|\\\]/g, '');
  s = s.replace(/^[\s."'`]+/, '');
  return s.replace(/\s+/g, ' ').trim();
}

export function normalizeText(input: string): string {
  let text = cleanForTts(input);
  text = text.replace(/[\u2018\u2019\u201c\u201d()\[\]{}\u2013\u2014\u2026]/g, (ch) => PUNCT_TRANSLATION[ch] ?? ch);
  text = text.replace(/\s+/g, ' ').trim();

  for (const [src, dst] of WORD_OVERRIDES) {
    text = text.replace(new RegExp(`\\b${escapeRe(src)}\\b`, 'g'), dst);
  }
  for (const [src, dst] of ABBREVIATIONS) {
    text = text.replace(new RegExp(`\\b${escapeRe(src)}`, 'gi'), dst);
  }

  // "F.B.I." -> "F B I"
  text = text.replace(/\b([A-Z])(?:\.([A-Z]))+\./g, (m) => (m.match(/[A-Z]/g) ?? []).join(' '));
  // apartment/suite/unit/... identifiers
  text = text.replace(
    /\b(apartment|apt\.?|suite|unit|room|flight|extension|order|invoice|locker|aisle|gate)\s+([A-Za-z]?\d{1,4}[A-Za-z]?)\b/gi,
    (_m, label: string, ident: string) => `${label} ${expandIdentifierToken(ident)}`,
  );
  // street numbers before N/S/E/W
  text = text.replace(/\b(\d{3})(?=\s+(?:North|South|East|West)\b)/gi, (m) => digitWords(m));
  // money
  text = text.replace(/\$(\d[\d,]*(?:\.\d{1,2})?)/g, (_m, raw: string) => expandMoney(raw));
  // dates M/D/YYYY
  text = text.replace(/\b(0?[1-9]|1[0-2])\/(0?[1-9]|[12]\d|3[01])\/(20\d{2}|19\d{2})\b/g, (m, mm: string, dd: string, yy: string) => {
    const month = parseInt(mm, 10), day = parseInt(dd, 10), year = parseInt(yy, 10);
    if (!isValidDate(year, month, day)) return m;
    return `${MONTHS[month - 1]} ${numWords(day, true)} ${numWords(year)}`;
  });
  // times H:MM [am|pm]
  text = text.replace(/\b(\d{1,2}):(\d{2})\s*([AaPp]\.?\s*[Mm]\.?)?\b/g, (_m, hh: string, mm: string, sf?: string) =>
    expandTime(parseInt(hh, 10), parseInt(mm, 10), sf),
  );
  // bare hour + am/pm
  text = text.replace(/\b(\d{1,2})\s*([AaPp]\.?\s*[Mm]\.)\b/g, (_m, hh: string, sf: string) =>
    `${numWords(parseInt(hh, 10))} ${sf.toLowerCase().replace(/[^a-z]/g, '').split('').join(' ')}`,
  );
  // phone numbers 555-1234
  text = text.replace(/\b(\d{3})-(\d{4})\b/g, (_m, a: string, b: string) => `${digitWords(a)}, ${digitWords(b)}`);
  // versions 1.2.3.4
  text = text.replace(/\b\d+(?:\.\d+){2,}\b/g, (m) => m.split('.').map((p) => numWords(parseInt(p, 10))).join(' point '));
  // decimals 1.5
  text = text.replace(/\b(\d+)\.(\d+)\b/g, (_m, whole: string, frac: string) =>
    `${numWords(parseInt(whole, 10))} point ${digitWords(frac)}`,
  );
  // ordinals 21st
  text = text.replace(/\b(\d+)(st|nd|rd|th)\b/gi, (_m, num: string) => numWords(parseInt(num, 10), true));
  // remaining numbers
  text = text.replace(/\b\d[\d,]*\b/g, (m) => {
    const value = m.replace(/,/g, '');
    if (value.length >= 5 && !value.startsWith('20')) return digitWords(value);
    return numWords(parseInt(value, 10));
  });
  // acronyms NASA -> N A S A
  text = text.replace(/\b[A-Z]{2,}\b/g, (m) =>
    m.length <= 1 ? m : m.split('').map((ch) => LETTER_NAMES[ch] ?? ch).join(' '),
  );

  text = text.replace(/,(?:\s*,)+/g, ',');
  text = text.replace(/,\s*([.!?])/g, '$1');
  text = text.replace(/\s+([,;:.!?])/g, '$1');
  text = text.replace(/([,;:.!?])(?=\S)/g, '$1 ');
  return text.replace(/\s+/g, ' ').trim();
}

// ── Phonemization ───────────────────────────────────────────────────────────

/** Verified exceptions from inflect_vits_frontend.py */
const PHONEME_OVERRIDES: [string, string][] = [
  ['sˈæskɐtʃˌuːən', 'sɐskˈætʃəwən'],
  ['flʊɹɹˈɛsənt', 'flʊˈɹɛsənt'],
];

function applyPhonemeOverrides(phonemeText: string): string {
  let text = phonemeText;
  for (const [src, dst] of PHONEME_OVERRIDES) text = text.replaceAll(src, dst);
  return text.replace(/\s+/g, ' ').trim();
}

/**
 * English text to IPA phonemes via espeak-ng WASM.
 * Mirrors Inflect-Micro-v2 `run_vits_frontend`: phonemize the full
 * normalized string with punctuation preserved (EspeakBackend
 * preserve_punctuation=True, with_stress=True). xenova/phonemizer splits on
 * .?! and drops those marks, so reinsert them between returned fragments.
 */
export async function textToPhonemes(text: string): Promise<string> {
  const normalized = text.trim();
  if (!normalized) return '';

  const result = await phonemize(normalized, 'en-us');
  const fragments = (Array.isArray(result) ? result : [String(result)])
    .map((s) => s.trim())
    .filter(Boolean);
  if (!fragments.length) return '';

  const marks = normalized.match(/[.!?]+/g) ?? [];
  let phonemized = '';
  for (let i = 0; i < fragments.length; i++) {
    if (i) phonemized += ' ';
    phonemized += fragments[i];
    if (marks[i]) phonemized += marks[i];
  }
  return applyPhonemeOverrides(phonemized);
}

/**
 * Text -> VITS input token ids (with interspersed blanks).
 * Mirrors cleaned_text_to_sequence + commons.intersperse(seq, 0).
 */
export async function textToInputIds(text: string): Promise<{ ids: number[]; phonemes: string; normalized: string }> {
  const normalized = normalizeText(text);
  const phonemes = await textToPhonemes(normalized);
  if (!phonemes) throw new Error('The text frontend produced no speakable tokens.');

  const sequence: number[] = [];
  for (const ch of phonemes) {
    const id = SYMBOL_TO_ID.get(ch);
    if (id !== undefined) sequence.push(id);
  }
  if (!sequence.length) throw new Error('The text frontend produced no speakable tokens.');

  // intersperse blanks: _ b l a n k s _
  const ids: number[] = new Array(sequence.length * 2 + 1).fill(0);
  sequence.forEach((id, j) => { ids[j * 2 + 1] = id; });
  return { ids, phonemes, normalized };
}
