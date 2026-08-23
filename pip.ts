// pip.ts — Picture-in-Picture "now reading" surface.
//
// A hidden <video> is the app's audio sink: the TTS AudioContext is tapped as
// a live MediaStream and played through the element, which is what gives the
// page a real media session — so the macOS play/pause key, the Now Playing
// widget and Bluetooth remotes all reach the reader.
//
// Opening PiP adds a second live track: a canvas painted with the sentence
// being spoken, active word lit. Nothing is pre-rendered — frames are pushed
// as words advance, and the audio streams sentence by sentence as it is
// synthesised.

const W = 1280, H = 720;          // 16:9 @2x — crisp text in a small window
const FPS = 30;
const PAD = 72;

export interface PipTheme { bg: string; text: string; textMuted: string; accent: string }

export interface PipState {
  prev: string;
  words: string[];        // current sentence, tokenised like extractWords()
  wordIndex: number;      // -1 when no word is active
  next: string;
}

export const pipSupported = () =>
  typeof document !== 'undefined' &&
  'pictureInPictureEnabled' in document &&
  (document as any).pictureInPictureEnabled &&
  typeof (HTMLCanvasElement.prototype as any).captureStream === 'function';

const ellipsize = (ctx: CanvasRenderingContext2D, text: string, max: number) => {
  if (!text || ctx.measureText(text).width <= max) return text;
  let s = text;
  while (s.length > 1 && ctx.measureText(s + '…').width > max) s = s.slice(0, -1);
  return s + '…';
};

export class PipSession {
  private canvas = document.createElement('canvas');
  private ctx = this.canvas.getContext('2d', { alpha: false })!;
  private video = document.createElement('video');
  private track: any = null;
  private syncing = false;
  private key = '';
  private paint: (() => void) | null = null;
  private heartbeat: any = null;
  private audio: MediaStream | null = null;

  /** onToggle(play) fires when the PiP window's own play/pause button is used. */
  constructor(private onToggle: (play: boolean) => void, private onLeave: () => void) {
    this.canvas.width = W;
    this.canvas.height = H;
    this.video.muted = false;
    (this.video as any).playsInline = true;
    // Must be laid out (not display:none) for Chrome to accept it into PiP.
    this.video.style.cssText = 'position:fixed;left:0;bottom:0;width:2px;height:2px;opacity:0;pointer-events:none;z-index:-1';
    document.body.appendChild(this.video);

    // The PiP overlay's transport maps onto the element's play/pause. Keep the
    // element itself running (it is a live stream) and forward intent instead.
    this.video.addEventListener('pause', () => {
      if (this.syncing) return;
      this.onToggle(false);
      this.video.play().catch(() => {});
    });
    this.video.addEventListener('play', () => { if (!this.syncing) this.onToggle(true); });
    this.video.addEventListener('leavepictureinpicture', () => this.onLeave());
  }

  get active() { return document.pictureInPictureElement === this.video; }

  /**
   * Make the element the app's audio sink. Resolves false when the browser
   * refuses to play it (no user activation yet) — the caller should then keep
   * audio on the AudioContext's own destination.
   */
  async attach(audio: MediaStream): Promise<boolean> {
    this.audio = audio;
    this.syncing = true;
    this.video.srcObject = new MediaStream(audio.getAudioTracks());
    try { await this.video.play(); } catch { this.syncing = false; return false; }
    this.syncing = false;
    return !this.video.paused;
  }

  get attached() { return !!this.video.srcObject; }

  /**
   * Open the PiP window. `paint` must repaint the canvas — a capture track
   * only emits frames when the canvas is drawn to *after* captureStream(), and
   * the element stays at readyState 0 (so play() aborts and PiP never opens)
   * until a frame arrives.
   */
  async start(paint: () => void) {
    if (!this.audio) throw new Error('attach() first');
    this.paint = paint;
    const captured: MediaStream = (this.canvas as any).captureStream(FPS);
    this.track = captured.getVideoTracks()[0];

    this.syncing = true;                             // our own play() is not user intent
    this.video.srcObject = new MediaStream([this.track, ...this.audio.getAudioTracks()]);
    this.repaint();
    const playing = this.video.play().catch(() => {});
    if (this.video.readyState < 1) {
      await new Promise<void>((res) => {
        const pump = setInterval(() => this.repaint(), 50);  // until it latches on
        const done = () => { clearInterval(pump); res(); };
        this.video.addEventListener('loadedmetadata', done, { once: true });
        setTimeout(done, 2000);                      // never hang on a stalled track
      });
    }
    await playing;
    this.syncing = false;

    await (this.video as any).requestPictureInPicture();

    // Idle heartbeat: a paused reader stops repainting, and a track with no
    // frames eventually stalls the window. One frame a second is enough.
    this.heartbeat = setInterval(() => this.repaint(), 1000) as any;
  }

  private repaint() { this.key = ''; this.paint?.(); }

  /** Close the window but keep the element as the audio sink. */
  async stop() {
    // Guard the whole teardown: exiting PiP and swapping srcObject both fire
    // 'pause' on the element, and that is our own doing, not the reader's.
    this.syncing = true;
    clearInterval(this.heartbeat);
    this.heartbeat = null;
    this.paint = null;
    try { if (this.active) await document.exitPictureInPicture(); } catch {}
    this.track?.stop();
    this.track = null;
    this.key = '';
    if (this.audio) {
      this.video.srcObject = new MediaStream(this.audio.getAudioTracks());
      this.video.play().catch(() => {});
    }
    setTimeout(() => { this.syncing = false; }, 0);
  }

  destroy() {
    this.syncing = true;
    clearInterval(this.heartbeat);
    this.track?.stop();
    this.audio = null;
    this.video.srcObject = null;
    this.video.remove();
  }

  /** Paint a frame and push it to the video track. No-op when nothing changed. */
  render(s: PipState, t: PipTheme, force = false) {
    const key = `${s.prev}|${s.words.join(' ')}|${s.wordIndex}|${s.next}|${t.bg}`;
    if (!force && key === this.key) return;
    this.key = key;

    const c = this.ctx;
    c.fillStyle = t.bg;
    c.fillRect(0, 0, W, H);

    const maxW = W - PAD * 2;
    const bodyFont = '600 50px ui-serif, Georgia, Cambria, serif';
    const lineH = 70;
    c.textBaseline = 'alphabetic';
    c.textAlign = 'left';
    c.font = bodyFont;

    // Wrap the sentence, keeping each word's x so the active one can be lit.
    const lines: { words: { w: string; x: number; width: number; i: number }[] }[] = [];
    let cur: typeof lines[0] = { words: [] };
    const space = c.measureText(' ').width;
    let x = 0;
    s.words.forEach((w, i) => {
      const width = c.measureText(w).width;
      if (x > 0 && x + space + width > maxW) { lines.push(cur); cur = { words: [] }; x = 0; }
      if (x > 0) x += space;
      cur.words.push({ w, x, width, i });
      x += width;
    });
    if (cur.words.length) lines.push(cur);

    // Keep the active word on screen for long sentences: a window of 5 lines.
    const MAX_LINES = 5;
    let from = 0;
    if (lines.length > MAX_LINES) {
      const activeLine = Math.max(0, lines.findIndex(l => l.words.some(w => w.i === s.wordIndex)));
      from = Math.min(Math.max(0, activeLine - 2), lines.length - MAX_LINES);
    }
    const shown = lines.slice(from, from + MAX_LINES);

    const blockH = shown.length * lineH;
    let y = Math.round((H - blockH) / 2) + 36;

    // Surrounding lines, dimmed, only where they are actually adjacent.
    c.font = '400 28px ui-sans-serif, -apple-system, system-ui, sans-serif';
    c.fillStyle = t.textMuted;
    if (s.prev && from === 0) c.fillText(ellipsize(c, s.prev, maxW), PAD, y - lineH - 22);
    if (s.next && from + MAX_LINES >= lines.length) c.fillText(ellipsize(c, s.next, maxW), PAD, y + blockH + 28);

    c.font = bodyFont;
    for (const line of shown) {
      for (const w of line.words) {
        if (w.i === s.wordIndex) {
          c.fillStyle = t.accent;
          const bx = PAD + w.x - 5, by = y - 44, bw = w.width + 10, bh = 62;
          c.beginPath();
          c.roundRect ? c.roundRect(bx, by, bw, bh, 8) : c.rect(bx, by, bw, bh);
          c.fill();
          c.fillStyle = '#ffffff';
        } else {
          c.fillStyle = t.text;
        }
        c.fillText(w.w, PAD + w.x, y);
      }
      y += lineH;
    }
  }

  /** Reflect app playback state onto the element without re-firing our handlers. */
  syncPlaying(playing: boolean) {
    this.syncing = true;
    if (playing && this.video.paused) this.video.play().catch(() => {});
    this.syncing = false;
  }
}
