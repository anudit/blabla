/**
 * GPU scratch memory shared by the graph executor and the fast decode engine.
 *
 * Both used to keep their own size-keyed free list, keyed by the exact byte
 * size of the activation. Every utterance has a different latent length, so
 * every buffer from the previous utterance was a different exact size and none
 * of them could ever be reused — which is why generate() had to purge the whole
 * pool on entry and pay the driver for a fresh allocation of every activation.
 *
 * Rounding allocations to size classes (8 per octave, so at most ~12.5% larger
 * than asked for) makes buffers from utterances of similar length land in the
 * same bucket and actually get reused. A byte budget keeps that from growing
 * without bound: once the free list is over budget, released buffers are
 * destroyed instead of kept.
 */

const MIN_CLASS = 256;

/** GPUBufferUsage.STORAGE | COPY_DST | COPY_SRC */
const STORAGE_USAGE = 128 | 8 | 4;
/** GPUBufferUsage.UNIFORM | COPY_DST */
const UNIFORM_USAGE = 64 | 8;

/** Uniform buffers are always one 256-byte block (see Meta in the shaders). */
const UNIFORM_SIZE = 256;

/** Round up to a size class: 8 buckets per power of two. */
export function sizeClass(bytes: number): number {
  if (bytes <= MIN_CLASS) return MIN_CLASS;
  const exp = 31 - Math.clz32(bytes); // floor(log2(bytes))
  const step = Math.max(MIN_CLASS, 2 ** Math.max(0, exp - 3));
  return Math.ceil(bytes / step) * step;
}

export class GpuScratch {
  private free = new Map<number, any[]>();
  private freeBytes = 0;
  private uniforms: any[] = [];

  constructor(private device: any, private budgetBytes = 128 * 1024 * 1024) {}

  /** A storage buffer of at least `bytes`; its real size is the size class. */
  acquire(bytes: number): any {
    const size = sizeClass(bytes);
    const bucket = this.free.get(size);
    if (bucket && bucket.length) {
      this.freeBytes -= size;
      return bucket.pop();
    }
    return this.device.createBuffer({ size, usage: STORAGE_USAGE });
  }

  /**
   * Hand a buffer back. Only safe once the commands referencing it have been
   * submitted — callers stage them through their own dead list until flush.
   */
  release(buffer: any): void {
    const size = buffer.size;
    if (this.freeBytes + size > this.budgetBytes) {
      try { buffer.destroy(); } catch {}
      return;
    }
    let bucket = this.free.get(size);
    if (!bucket) { bucket = []; this.free.set(size, bucket); }
    bucket.push(buffer);
    this.freeBytes += size;
  }

  /**
   * A uniform buffer holding `words`. A fresh buffer is needed for every
   * dispatch in an encoder — queue.writeBuffer is ordered against submits, not
   * against pass encoding, so rewriting one before submit would feed the new
   * contents to every pass that already referenced it. They become reusable
   * again only after the encoder is submitted; see recycleUniforms().
   */
  uniform(words: Uint32Array): any {
    const buf = this.uniforms.pop()
      ?? this.device.createBuffer({ size: UNIFORM_SIZE, usage: UNIFORM_USAGE });
    this.device.queue.writeBuffer(buf, 0, words, 0, Math.min(words.length, UNIFORM_SIZE / 4));
    return buf;
  }

  /** Take back the uniform buffers used by a submitted encoder. */
  recycleUniforms(used: any[]): void {
    for (const b of used) this.uniforms.push(b);
  }

  /** Destroy everything held here. */
  clear(): void {
    for (const [, bucket] of this.free) {
      for (const b of bucket) try { b.destroy(); } catch {}
    }
    this.free.clear();
    this.freeBytes = 0;
    for (const b of this.uniforms) try { b.destroy(); } catch {}
    this.uniforms = [];
  }
}
