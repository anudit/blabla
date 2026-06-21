/**
 * Pure pixel-based text line segmentation — Otsu binarization, ruled-line
 * erasure, horizontal dilation and projection-profile band detection.
 *
 * Kept in its own module (no preact / no JSX) so it can be imported from the
 * OCR Web Worker without dragging the UI framework into the worker bundle.
 *
 * Accepts an HTMLCanvasElement or an OffscreenCanvas (worker context).
 */
export function segmentTextLines(
  canvas: HTMLCanvasElement | OffscreenCanvas,
): { y0: number; y1: number; x0: number; x1: number }[] {
  const ctx = (canvas as any).getContext('2d', { willReadFrequently: true })!;
  const { width: W, height: H } = canvas;
  const imgData = ctx.getImageData(0, 0, W, H);
  const data = imgData.data;

  // Grayscale + Otsu binarization (text = dark = 1, background = 0)
  const gray = new Uint8Array(W * H);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    gray[j] = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114) | 0;
  }

  // Otsu threshold
  const hist = new Array(256).fill(0);
  for (let j = 0; j < gray.length; j++) hist[gray[j]]++;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, maxVar = 0, threshold = 127;
  for (let i = 0; i < 256; i++) {
    wB += hist[i];
    if (wB === 0) continue;
    const wF = gray.length - wB;
    if (wF === 0) break;
    sumB += i * hist[i];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) { maxVar = between; threshold = i; }
  }

  // Binary: text pixels (darker than threshold) = 1
  const bw = new Uint8Array(W * H);
  for (let j = 0; j < gray.length; j++) bw[j] = gray[j] < threshold ? 1 : 0;

  // Erase full-width horizontal ruled lines (>55% of row dark = horizontal rule, not text).
  // Also erase ±0 adjacent rows so the gap they create exceeds mergeGap, preventing
  // adjacent text lines from merging across a ruled line.
  const hRuleThresh = Math.floor(W * 0.55);
  const ruleRows = new Set<number>();
  for (let y = 0; y < H; y++) {
    const row = y * W; let cnt = 0;
    for (let x = 0; x < W; x++) cnt += bw[row + x];
    if (cnt > hRuleThresh) ruleRows.add(y);
  }
  const erasePad = 0;
  for (const ry of ruleRows) {
    for (let dy = -erasePad; dy <= erasePad; dy++) {
      const yt = ry + dy;
      if (yt >= 0 && yt < H) bw.fill(0, yt * W, (yt + 1) * W);
    }
  }

  // Erase full-height vertical rules (margin lines, left/right borders).
  // A column dark in >70% of rows is a vertical rule. Text characters
  // occupy at most ~52% of page height even if they appear on every line.
  const vRuleThresh = Math.floor(H * 0.70);
  const colCounts = new Int32Array(W);
  for (let y = 0; y < H; y++) { const row = y * W; for (let x = 0; x < W; x++) colCounts[x] += bw[row + x]; }
  for (let x = 0; x < W; x++) {
    if (colCounts[x] > vRuleThresh) {
      for (let y = 0; y < H; y++) bw[y * W + x] = 0;
    }
  }

  // Horizontal dilation: for each row, a pixel is 1 if any pixel within
  // a horizontal window of size W/6 is 1.
  const dilWin = Math.max(3, Math.floor(W / 6));
  const dilated = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let count = 0;
    for (let x = 0; x < W + dilWin; x++) {
      if (x < W && bw[row + x]) count++;
      if (x >= dilWin && bw[row + x - dilWin]) count--;
      if (x >= dilWin - 1) {
        dilated[row + (x - dilWin + 1)] = count > 0 ? 1 : 0;
      }
    }
  }

  // Horizontal projection for vertical bands
  const proj = new Int32Array(H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let s = 0;
    for (let x = 0; x < W; x++) s += dilated[row + x];
    proj[y] = s;
  }

  const maxProj = Math.max(...proj);
  const minProj = Math.min(...proj);
  const projRange = maxProj - minProj;
  const bandThreshold = minProj + Math.max(2, projRange * 0.08);
  const scale = H / 842;
  const minLineHeight = Math.max(3, Math.round(2.5 * scale));

  const rawLines: { y0: number; y1: number }[] = [];
  let i = 0;
  while (i < H) {
    if (proj[i] > bandThreshold) {
      const start = i;
      while (i < H && proj[i] > bandThreshold) i++;
      const end = i;
      if (end - start > minLineHeight) rawLines.push({ y0: start, y1: end });
    }
    i++;
  }

  // Merge close lines
  const merged: { y0: number; y1: number }[] = [];
  const mergeGap = Math.max(0, Math.round(1.5 * scale));
  for (const ln of rawLines) {
    if (merged.length > 0 && ln.y0 - merged[merged.length - 1].y1 < mergeGap) {
      merged[merged.length - 1].y1 = ln.y1;
    } else {
      merged.push({ y0: ln.y0, y1: ln.y1 });
    }
  }

  // Horizontal crop: for each merged band, find leftmost/rightmost text columns
  // using the original binary image (undilated) for tight character bounds.
  // Also filter out bands with too few text pixels (page footer / blank lines).
  const padH = 8;
  const minTextPx = Math.max(10, Math.round(W * 0.25)); // ~500 px — even a single short word
  const bands = merged
    .map(({ y0, y1 }) => {
      let x0 = W, x1 = 0, count = 0;
      const y0p = Math.max(0, y0 - padH);
      const y1p = Math.min(H, y1 + padH);
      for (let y = y0p; y < y1p; y++) {
        const row = y * W;
        for (let x = 0; x < W; x++) {
          if (bw[row + x]) {
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            count++;
          }
        }
      }
      // Always use full page width so the resize keeps enough CTC time-steps
      // for long text lines (tight bw-derived x-crop reduces T below the
      // minimum needed for reliable CTC decoding with ~75+ chars/line).
      return { y0, y1, x0: 0, x1: W - 1, count };
    });
  const filtered = bands.filter(b => b.count >= minTextPx);
  return filtered.map(({ y0, y1, x0, x1 }) => ({ y0, y1, x0, x1 }));
}
