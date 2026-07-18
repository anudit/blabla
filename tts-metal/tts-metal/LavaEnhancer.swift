//
//  LavaEnhancer.swift
//  tts-metal
//
//  A Metal implementation of LavaSR v2 — a Vocos-based universal speech
//  bandwidth-extension / super-resolution model, run as a post-processing layer
//  on the Kitten-TTS output: it restores/enhances the synthesized 24 kHz waveform
//  and upsamples it to 48 kHz. The whole forward pass runs in Metal compute
//  shaders (see Shaders.metal `lava_*` and the reused `reuse_matmul` / iSTFT /
//  conv1d kernels). Weights load from the bundled `lavasr_v2.safetensors`.
//
//  Architecture (vocos.models.VocosBackbone + ISTFTHead):
//    resample → mel-spectrogram(log) → Conv1d embed → LayerNorm
//      → 8× ConvNeXt block (depthwise Conv1d → LN → Linear→GELU→Linear → γ, +res)
//      → LayerNorm → Linear head → magnitude/phase → iSTFT → resample to 48 kHz
//
//  Layout: sequence tensors are either [dim, T] (channel-major) for the conv/
//  depthwise ops or [T, dim] (row-major) for the pointwise Linears; transposes
//  bridge the two.
//

import Foundation
import Metal

final class LavaEnhancer: @unchecked Sendable {

    // MARK: - Model constants (enhancer_v2/config.yaml)
    private let opRate = 44100                 // mel filterbank was built for 44.1 kHz
    private let nFFT = 2048
    private let hop = 512
    private let nMels = 80
    private let dim = 512
    private let interDim = 1536
    private let numLayers = 8
    private var F: Int { nFFT / 2 + 1 }        // 1025
    private var stftPad: Int { (nFFT - hop) / 2 }  // 768 ("same" padding)

    // MARK: - Metal handles
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var library: MTLLibrary!
    private var pipelines: [String: MTLComputePipelineState] = [:]

    // MARK: - Weights
    private var weights: [String: MTLBuffer] = [:]
    private var shapes: [String: [Int]] = [:]
    private(set) var isLoaded = false

    // Batched encoder state
    private var cmdBuf: MTLCommandBuffer?
    private var enc: MTLComputeCommandEncoder?
    private var scratch: [MTLBuffer] = []
    private var lastCommitted: MTLCommandBuffer?

    // Optional per-stage GPU timing (set by the self-test).
    var profile = false
    private var stageT0 = Date()
    private func stage(_ name: String) {
        guard profile else { return }
        flushAndWait()
        let dt = Date().timeIntervalSince(stageT0) * 1000
        print(String(format: "[LavaSR stage] %@ %.0f ms", name, dt)); fflush(stdout)
        stageT0 = Date()
    }

    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
    }

    // MARK: - Load

    @discardableResult
    func load() throws -> Bool {
        guard let lib = device.makeDefaultLibrary() else { throw err("default.metallib missing") }
        library = lib
        compilePipelines()

        guard let url = Bundle.main.url(forResource: "lavasr_v2", withExtension: "safetensors") else {
            throw err("lavasr_v2.safetensors not bundled")
        }
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let tensors = try SafetensorsParser.parse(data)

        // Linear/filterbank weights stored [out,in] but the GPU matmul wants [in,out].
        let transposeSuffixes = ["pwconv1.weight", "pwconv2.weight", "head.out.weight", "mel_scale.fb"]
        for (name, t) in tensors {
            var floats = t.data
            var shape = t.shape
            if shape.count == 2, transposeSuffixes.contains(where: { name.hasSuffix($0) }) {
                let rows = shape[0], cols = shape[1]
                var tr = [Float](repeating: 0, count: rows * cols)
                for r in 0..<rows { for c in 0..<cols { tr[c * rows + r] = floats[r * cols + c] } }
                floats = tr
                shape = [cols, rows]
            }
            guard let buf = device.makeBuffer(bytes: floats, length: max(floats.count, 1) * 4,
                                              options: [.storageModeShared]) else {
                throw err("buffer alloc failed for \(name)")
            }
            buf.label = name
            weights[name] = buf
            shapes[name] = shape
        }
        print("[LavaSR] Loaded \(weights.count) weight tensors")
        isLoaded = true
        return true
    }

    private func compilePipelines() {
        let names = [
            "lava_stft_mag_kernel", "lava_safe_log_kernel", "lava_dwconv1d_kernel",
            "lava_gamma_residual_kernel", "lava_head_to_complex_kernel",
            // reused kernels:
            "reuse_matmul_kernel", "reuse_irfft_kernel", "reuse_ola_kernel",
            "conv1d_tiled_kernel", "conv1d_kernel", "transpose_kernel",
            "layer_norm_kernel", "gelu_kernel",
        ]
        for n in names {
            guard let fn = library.makeFunction(name: n) else { print("[LavaSR] MISSING \(n)"); continue }
            do { pipelines[n] = try device.makeComputePipelineState(function: fn) }
            catch { print("[LavaSR] pipeline fail \(n): \(error)") }
        }
    }

    // MARK: - Helpers

    private func err(_ m: String) -> NSError {
        NSError(domain: "LavaEnhancer", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
    private func w(_ name: String) throws -> MTLBuffer {
        guard let b = weights[name] else { throw err("missing weight \(name)") }
        return b
    }
    private func makeBuffer(_ f: [Float], _ label: String) -> MTLBuffer {
        let b = device.makeBuffer(bytes: f, length: max(f.count, 1) * 4, options: [.storageModeShared])!
        b.label = label; return b
    }
    private func empty(_ n: Int, _ label: String) -> MTLBuffer {
        let b = device.makeBuffer(length: max(n, 1) * 4, options: [.storageModeShared])!
        b.label = label; scratch.append(b); return b
    }
    private func ensureEncoder() {
        if cmdBuf == nil {
            cmdBuf = commandQueue.makeCommandBuffer(); cmdBuf?.label = "lavasr"
            enc = cmdBuf?.makeComputeCommandEncoder()
        }
    }
    private func flush() {
        guard let e = enc, let cb = cmdBuf else { return }
        e.endEncoding()
        let held = scratch; scratch = []
        cb.addCompletedHandler { _ in _ = held }
        cb.commit(); lastCommitted = cb
        cmdBuf = nil; enc = nil
    }
    private func flushAndWait() {
        if let e = enc, let cb = cmdBuf { e.endEncoding(); cb.commit(); cb.waitUntilCompleted() }
        else { lastCommitted?.waitUntilCompleted() }
        lastCommitted = nil; cmdBuf = nil; enc = nil
    }
    private func pipe(_ n: String) -> MTLComputePipelineState {
        guard let p = pipelines[n] else { fatalError("[LavaSR] pipeline \(n) missing") }
        return p
    }
    private func run<T>(_ name: String, _ buffers: [MTLBuffer], total: Int, params: T) {
        ensureEncoder()
        let p = pipe(name)
        enc!.setComputePipelineState(p)
        for (i, b) in buffers.enumerated() { enc!.setBuffer(b, offset: 0, index: i) }
        var params = params
        enc!.setBytes(&params, length: MemoryLayout<T>.stride, index: buffers.count)
        let wg = min(256, p.maxTotalThreadsPerThreadgroup)
        let grid = (total + wg - 1) / wg
        enc!.dispatchThreadgroups(MTLSizeMake(max(grid, 1), 1, 1), threadsPerThreadgroup: MTLSizeMake(wg, 1, 1))
    }
    private func runTG<T>(_ name: String, _ buffers: [MTLBuffer], params: T, groups: MTLSize, tpg: MTLSize) {
        ensureEncoder()
        let p = pipe(name)
        enc!.setComputePipelineState(p)
        for (i, b) in buffers.enumerated() { enc!.setBuffer(b, offset: 0, index: i) }
        var params = params
        enc!.setBytes(&params, length: MemoryLayout<T>.stride, index: buffers.count)
        enc!.dispatchThreadgroups(groups, threadsPerThreadgroup: tpg)
    }
    private func readBuffer(_ b: MTLBuffer, _ count: Int) -> [Float] {
        flushAndWait()
        return b.contents().withMemoryRebound(to: Float.self, capacity: count) {
            Array(UnsafeBufferPointer(start: $0, count: count))
        }
    }

    // MARK: - Param structs (mirror Shaders.metal)
    private struct StftP { var n_fft: UInt32; var hop: UInt32; var num_freq: UInt32; var num_frames: UInt32 }
    private struct SizeP { var size: UInt32 }
    private struct DwP { var channels: UInt32; var length: UInt32; var ksize: UInt32; var pad: UInt32 }
    private struct GammaP { var dim: UInt32; var length: UInt32 }
    private struct HeadP { var num_freq: UInt32; var num_frames: UInt32 }
    private struct MatmulP { var M: UInt32; var K: UInt32; var N: UInt32; var use_bias: UInt32 }
    private struct LayerNormP { var batch_size: UInt32; var hidden_size: UInt32; var eps: Float }
    private struct TransposeP { var rows: UInt32; var cols: UInt32 }
    private struct IrfftP { var n_fft: UInt32; var num_freq: UInt32; var num_frames: UInt32 }
    private struct OlaP { var n_fft: UInt32; var hop: UInt32; var num_frames: UInt32; var pad_len: UInt32; var out_len: UInt32; var trim: UInt32 }
    private struct Conv1dP {
        var in_channels: UInt32; var out_channels: UInt32; var kernel_size: UInt32
        var input_length: UInt32; var output_length: UInt32; var padding: UInt32
        var stride: UInt32; var dilation: UInt32; var use_bias: UInt32
    }

    private lazy var dummyBias = makeBuffer([0], "dummy_bias")

    // MARK: - Primitive ops

    private func matmul(_ A: MTLBuffer, _ B: MTLBuffer, bias: MTLBuffer?, out: MTLBuffer, M: Int, K: Int, N: Int) {
        let p = MatmulP(M: UInt32(M), K: UInt32(K), N: UInt32(N), use_bias: bias == nil ? 0 : 1)
        runTG("reuse_matmul_kernel", [A, B, bias ?? dummyBias, out], params: p,
              groups: MTLSize(width: (N + 63) / 64, height: (M + 63) / 64, depth: 1),
              tpg: MTLSize(width: 256, height: 1, depth: 1))
    }
    private func transpose(_ input: MTLBuffer, rows: Int, cols: Int, label: String) -> MTLBuffer {
        let out = empty(rows * cols, label)
        run("transpose_kernel", [input, out], total: rows * cols, params: TransposeP(rows: UInt32(rows), cols: UInt32(cols)))
        return out
    }
    private func layerNorm(_ input: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, batch: Int, hidden: Int, label: String) -> MTLBuffer {
        let out = empty(batch * hidden, label)
        runTG("layer_norm_kernel", [input, gamma, beta, out],
              params: LayerNormP(batch_size: UInt32(batch), hidden_size: UInt32(hidden), eps: 1e-6),
              groups: MTLSize(width: (batch + 255) / 256, height: 1, depth: 1),
              tpg: MTLSize(width: 256, height: 1, depth: 1))
        return out
    }
    private func conv1d(_ input: MTLBuffer, wName: String, bName: String, inCh: Int, outCh: Int,
                        k: Int, length: Int, pad: Int, label: String) throws -> MTLBuffer {
        let out = empty(outCh * length, label)
        var p = Conv1dP(in_channels: UInt32(inCh), out_channels: UInt32(outCh), kernel_size: UInt32(k),
                        input_length: UInt32(length), output_length: UInt32(length), padding: UInt32(pad),
                        stride: 1, dilation: 1, use_bias: 1)
        let pb = device.makeBuffer(bytes: &p, length: MemoryLayout<Conv1dP>.stride, options: [])!
        scratch.append(pb)
        // conv1d_tiled: one threadgroup row per output channel.
        ensureEncoder()
        enc!.setComputePipelineState(pipe("conv1d_tiled_kernel"))
        enc!.setBuffer(input, offset: 0, index: 0)
        enc!.setBuffer(try w(wName), offset: 0, index: 1)
        enc!.setBuffer(try w(bName), offset: 0, index: 2)
        enc!.setBuffer(out, offset: 0, index: 3)
        enc!.setBuffer(pb, offset: 0, index: 4)
        enc!.dispatchThreadgroups(MTLSize(width: (length + 255) / 256, height: outCh, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        return out
    }

    // MARK: - ConvNeXt block

    private func convNeXtBlock(_ x: MTLBuffer, T: Int, prefix: String) throws -> MTLBuffer {
        // x: [dim, T]. residual = x.
        // depthwise conv
        let dw = empty(dim * T, "\(prefix)_dw")
        run("lava_dwconv1d_kernel",
            [x, try w("\(prefix).dwconv.weight"), try w("\(prefix).dwconv.bias"), dw],
            total: dim * T, params: DwP(channels: UInt32(dim), length: UInt32(T), ksize: 7, pad: 3))
        // -> [T, dim], LayerNorm
        let dwT = transpose(dw, rows: dim, cols: T, label: "\(prefix)_dwT")   // [T, dim]
        let ln = layerNorm(dwT, gamma: try w("\(prefix).norm.weight"), beta: try w("\(prefix).norm.bias"),
                           batch: T, hidden: dim, label: "\(prefix)_ln")
        // pwconv1 (Linear dim->inter) + GELU
        let h1 = empty(T * interDim, "\(prefix)_h1")
        matmul(ln, try w("\(prefix).pwconv1.weight"), bias: try w("\(prefix).pwconv1.bias"),
               out: h1, M: T, K: dim, N: interDim)
        let g = empty(T * interDim, "\(prefix)_gelu")
        run("gelu_kernel", [h1, g], total: T * interDim, params: SizeP(size: UInt32(T * interDim)))
        // pwconv2 (Linear inter->dim)
        let h2 = empty(T * dim, "\(prefix)_h2")
        matmul(g, try w("\(prefix).pwconv2.weight"), bias: try w("\(prefix).pwconv2.bias"),
               out: h2, M: T, K: interDim, N: dim)
        // gamma * h2 + residual -> [dim, T]
        let out = empty(dim * T, "\(prefix)_out")
        run("lava_gamma_residual_kernel", [h2, x, try w("\(prefix).gamma"), out],
            total: dim * T, params: GammaP(dim: UInt32(dim), length: UInt32(T)))
        return out
    }

    // MARK: - Public entry point

    /// Enhance + upsample. `input` is mono PCM at `inputSR`; returns mono PCM at `targetSR`.
    func enhance(_ input: [Float], inputSR: Double, targetSR: Double) throws -> [Float] {
        guard isLoaded, input.count > 8 else { return input }
        scratch.removeAll(keepingCapacity: true)
        if profile { stageT0 = Date() }

        // 1. Resample to the model's operating rate (44.1 kHz, matching the mel bank).
        let sig = (abs(inputSR - Double(opRate)) < 1) ? input
                                                       : Resampler.resample(input, from: inputSR, to: Double(opRate))
        let Lw = sig.count
        let T = (Lw + 2 * stftPad - nFFT) / hop + 1     // frames (center=False + 'same' pad)
        if T < 2 { return input }

        // 2. Reflect-pad + STFT magnitude.
        let padLen = Lw + 2 * stftPad
        var padded = [Float](repeating: 0, count: padLen)
        for i in 0..<padLen {
            var s = i - stftPad
            if s < 0 { s = -s }
            if s >= Lw { s = 2 * Lw - 2 - s }
            s = min(max(s, 0), Lw - 1)
            padded[i] = sig[s]
        }
        let sigBuf = makeBuffer(padded, "lava_signal")
        let stftWin = try w("feature_extractor.mel_spec.spectrogram.window")
        let mag = empty(F * T, "lava_mag")
        run("lava_stft_mag_kernel", [sigBuf, stftWin, mag], total: F * T,
            params: StftP(n_fft: UInt32(nFFT), hop: UInt32(hop), num_freq: UInt32(F), num_frames: UInt32(T)))

        // 3. Mel = fbᵀ · mag  → [80, T], then safe_log.
        let mel = empty(nMels * T, "lava_mel")
        matmul(try w("feature_extractor.mel_spec.mel_scale.fb"), mag, bias: nil, out: mel,
               M: nMels, K: F, N: T)
        let melLog = empty(nMels * T, "lava_mel_log")
        run("lava_safe_log_kernel", [mel, melLog], total: nMels * T, params: SizeP(size: UInt32(nMels * T)))
        stage("mel")

        // 4. embed Conv1d(80→512, k7) → [512, T], then initial LayerNorm.
        let emb = try conv1d(melLog, wName: "backbone.embed.weight", bName: "backbone.embed.bias",
                             inCh: nMels, outCh: dim, k: 7, length: T, pad: 3, label: "lava_embed")
        let embT = transpose(emb, rows: dim, cols: T, label: "lava_embT")
        let normed = layerNorm(embT, gamma: try w("backbone.norm.weight"), beta: try w("backbone.norm.bias"),
                               batch: T, hidden: dim, label: "lava_norm")
        var x = transpose(normed, rows: T, cols: dim, label: "lava_norm_ct")   // [dim, T]
        stage("embed")

        // 5. 8× ConvNeXt blocks.
        for i in 0..<numLayers {
            x = try convNeXtBlock(x, T: T, prefix: "backbone.convnext.\(i)")
            if i % 2 == 1 { flush() }
        }
        stage("8x convnext")

        // 6. final LayerNorm → [T, dim], head Linear → [T, 2F].
        let xT = transpose(x, rows: dim, cols: T, label: "lava_finalT")
        let xf = layerNorm(xT, gamma: try w("backbone.final_layer_norm.weight"),
                           beta: try w("backbone.final_layer_norm.bias"), batch: T, hidden: dim, label: "lava_final_ln")
        let head = empty(T * 2 * F, "lava_head")
        matmul(xf, try w("head.out.weight"), bias: try w("head.out.bias"), out: head, M: T, K: dim, N: 2 * F)

        // 7. magnitude/phase → complex spectrum.
        let re = empty(F * T, "lava_re")
        let im = empty(F * T, "lava_im")
        run("lava_head_to_complex_kernel", [head, re, im], total: F * T,
            params: HeadP(num_freq: UInt32(F), num_frames: UInt32(T)))
        stage("head")

        // 8. iSTFT (n_fft=2048, hop=512, 'same' trim = 768).
        let istftWin = try w("head.istft.window")
        let frames = empty(T * nFFT, "lava_frames")
        run("reuse_irfft_kernel", [re, im, istftWin, frames], total: T * nFFT,
            params: IrfftP(n_fft: UInt32(nFFT), num_freq: UInt32(F), num_frames: UInt32(T)))
        let olaPad = (T - 1) * hop + nFFT
        let outLen = olaPad - 2 * stftPad
        let waveBuf = empty(outLen, "lava_wave")
        run("reuse_ola_kernel", [frames, istftWin, waveBuf], total: outLen,
            params: OlaP(n_fft: UInt32(nFFT), hop: UInt32(hop), num_frames: UInt32(T),
                         pad_len: UInt32(olaPad), out_len: UInt32(outLen), trim: UInt32(stftPad)))
        var wave = readBuffer(waveBuf, outLen)
        scratch.removeAll(keepingCapacity: true)
        stage("istft")

        // Align to the resampled input length, then resample op-rate → target (48 kHz).
        if wave.count < Lw { wave.append(contentsOf: [Float](repeating: 0, count: Lw - wave.count)) }
        else if wave.count > Lw { wave = Array(wave[0..<Lw]) }
        if abs(Double(opRate) - targetSR) > 1 {
            wave = Resampler.resample(wave, from: Double(opRate), to: targetSR)
        }
        return wave
    }

    /// On-device smoke test / profiler. Triggered by `REUSE_SELFTEST=1`.
    func selfTest() {
        let sr = 24000.0
        let secs = Double(ProcessInfo.processInfo.environment["REUSE_SECS"] ?? "3.5") ?? 3.5
        let n = Int(secs * sr)
        var x = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Double(i) / sr
            x[i] = Float(0.3 * sin(2 * .pi * 220 * t) + 0.15 * sin(2 * .pi * 660 * t))
        }
        // Warm up pipelines/allocations so the reported numbers reflect steady state.
        _ = try? enhance(x, inputSR: sr, targetSR: 48000)
        profile = true
        let t0 = Date()
        do {
            let y = try enhance(x, inputSR: sr, targetSR: 48000)
            let dt = Date().timeIntervalSince(t0)
            var rms: Double = 0, peak: Float = 0, nan = 0
            for v in y { rms += Double(v * v); peak = max(peak, abs(v)); if !v.isFinite { nan += 1 } }
            rms = (y.isEmpty ? 0 : (rms / Double(y.count)).squareRoot())
            let audioSecs = Double(n) / sr
            print(String(format: "[LavaSR selftest] in=%d out=%d rms=%.4f peak=%.4f nonfinite=%d time=%.3fs realtime=%.1fx",
                         n, y.count, rms, peak, nan, dt, audioSecs / dt))
        } catch {
            print("[LavaSR selftest] FAILED: \(error)")
        }
        fflush(stdout)
    }
}

// MARK: - Lanczos resampler (CPU)

enum Resampler {
    /// High-quality arbitrary-ratio resample via a Lanczos-windowed sinc kernel.
    static func resample(_ x: [Float], from srcSR: Double, to dstSR: Double, a: Int = 16) -> [Float] {
        if x.isEmpty { return x }
        let ratio = dstSR / srcSR
        let outN = Int((Double(x.count) * ratio).rounded())
        if outN <= 0 { return [] }
        var out = [Float](repeating: 0, count: outN)
        let n = x.count
        let scale = min(1.0, ratio)          // kernel cutoff in source samples
        let af = Double(a)
        x.withUnsafeBufferPointer { xp in
            for o in 0..<outN {
                let pos = Double(o) / ratio
                let center = Int(pos.rounded(.down))
                let half = Int((af / scale).rounded(.up))
                var acc = 0.0, norm = 0.0
                var k = center - half
                while k <= center + half + 1 {
                    if k >= 0 && k < n {
                        let t = (pos - Double(k)) * scale
                        let wgt = lanczos(t, af)
                        acc += Double(xp[k]) * wgt
                        norm += wgt
                    }
                    k += 1
                }
                out[o] = Float(norm != 0 ? acc / norm : 0)
            }
        }
        return out
    }
    private static func lanczos(_ t: Double, _ a: Double) -> Double {
        if t == 0 { return 1 }
        if t <= -a || t >= a { return 0 }
        let pt = Double.pi * t
        return (sin(pt) / pt) * (sin(pt / a) / (pt / a))
    }
}
