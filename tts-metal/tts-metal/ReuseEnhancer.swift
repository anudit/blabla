//
//  ReuseEnhancer.swift
//  tts-metal
//
//  A Metal implementation of NVIDIA's RE-USE (SEMamba) universal speech
//  enhancement model, run as a post-processing layer on the Kitten-TTS output:
//  it denoises/restores the synthesized 24 kHz waveform and bandwidth-extends
//  (upsamples) it to 48 kHz. The full forward pass — STFT, a dense conv encoder,
//  30 bidirectional Time–Frequency Mamba blocks, magnitude/phase conv decoders,
//  and the iSTFT — runs entirely in Metal compute shaders (see Shaders.metal,
//  the `reuse_*` kernels). Weights load from the bundled `model.safetensors`.
//
//  Layout conventions (matching the kernels):
//    * 2D feature maps: channel-major, row-major  x[c,h,w] = buf[(c*H+h)*W+w]
//    * Mamba sequences:  [batch, L, D] row-major   x[b,l,d] = buf[(b*L+l)*D+d]
//

import Foundation
import Metal

final class ReuseEnhancer: @unchecked Sendable {

    // MARK: - Model constants (from config.json / model.safetensors)
    private let baseNFFT = 320, baseHop = 40, baseSR = 8000
    private let hidFeature = 64          // encoder/decoder channels & Mamba d_model
    private let dInner = 256             // expand(4) * d_model(64)
    private let dState = 16
    private let dConv = 4
    private let dtRank = 4               // ceil(d_model / d_state)
    private let numLayers = 30

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

    // Optional per-stage GPU timing (set by the self-test).
    var profile = false
    private var stageT0 = Date()
    private func stage(_ name: String) {
        guard profile else { return }
        flushAndWait()
        let dt = Date().timeIntervalSince(stageT0) * 1000
        print(String(format: "[RE-USE stage] %@ %.0f ms", name, dt)); fflush(stdout)
        stageT0 = Date()
    }

    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
    }

    // MARK: - Load

    @discardableResult
    func load() throws -> Bool {
        guard let lib = device.makeDefaultLibrary() else {
            throw err("default.metallib missing")
        }
        library = lib
        compilePipelines()

        guard let url = Bundle.main.url(forResource: "model", withExtension: "safetensors") else {
            throw err("model.safetensors not bundled")
        }
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let tensors = try SafetensorsParser.parse(data)

        // Linear weights are stored [out, in] but the GPU matmul wants [in, out].
        let transposeSuffixes = ["in_proj.weight", "x_proj.weight", "dt_proj.weight",
                                 "out_proj.weight", "output_proj.weight"]
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
            guard let buf = device.makeBuffer(bytes: floats, length: floats.count * 4,
                                              options: [.storageModeShared]) else {
                throw err("buffer alloc failed for \(name)")
            }
            buf.label = name
            weights[name] = buf
            shapes[name] = shape
        }
        print("[RE-USE] Loaded \(weights.count) weight tensors")
        isLoaded = true
        return true
    }

    private func compilePipelines() {
        let names = [
            "reuse_stft_kernel", "reuse_conv2d_kernel", "reuse_conv2d_tiled_kernel", "reuse_instance_norm2d_kernel",
            "reuse_prelu_kernel", "reuse_pad2d_kernel", "reuse_pixelshuffle_w_kernel",
            "reuse_transpose_hw_kernel", "reuse_flip_kernel", "reuse_mamba_conv_kernel",
            "reuse_selective_scan_kernel", "reuse_zero_frac_kernel", "reuse_spec_to_complex_kernel",
            "reuse_irfft_kernel", "reuse_ola_kernel", "reuse_permute3d_kernel",
            "reuse_slice_cols_kernel", "reuse_concat_cols_kernel", "reuse_crop2d_kernel",
            "reuse_atan2_kernel",
            "reuse_matmul_kernel",
            // reused from the Kitten pipeline:
            "matmul_kernel", "layer_norm_kernel", "add_kernel", "concat_channels_kernel",
            "transpose_kernel",
        ]
        for n in names {
            guard let fn = library.makeFunction(name: n) else { print("[RE-USE] MISSING \(n)"); continue }
            do { pipelines[n] = try device.makeComputePipelineState(function: fn) }
            catch { print("[RE-USE] pipeline fail \(n): \(error)") }
        }
    }

    // MARK: - Buffer / dispatch helpers

    private func err(_ m: String) -> NSError {
        NSError(domain: "ReuseEnhancer", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
    private func w(_ name: String) throws -> MTLBuffer {
        guard let b = weights[name] else { throw err("missing weight \(name)") }
        return b
    }
    private func shape(_ name: String) -> [Int] { shapes[name] ?? [] }

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
            cmdBuf = commandQueue.makeCommandBuffer(); cmdBuf?.label = "reuse"
            enc = cmdBuf?.makeComputeCommandEncoder()
        }
    }

    /// Commit the current command buffer WITHOUT blocking. Default (tracked) Metal
    /// resources get automatic cross-command-buffer dependency tracking, so the next
    /// command buffer on the same queue correctly reads results produced here. The
    /// scratch buffers are retained by a completion handler until the GPU is done,
    /// then released — the CPU never stalls waiting for the GPU mid-pass.
    private var lastCommitted: MTLCommandBuffer?

    private func flush() {
        guard let e = enc, let cb = cmdBuf else { return }
        e.endEncoding()
        let held = scratch
        scratch = []
        cb.addCompletedHandler { _ in _ = held }   // keep buffers alive until completion
        cb.commit()
        lastCommitted = cb
        cmdBuf = nil; enc = nil
    }

    private func flushAndWait() {
        if let e = enc, let cb = cmdBuf {
            e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        } else {
            lastCommitted?.waitUntilCompleted()   // drain async-committed work too
        }
        lastCommitted = nil
        cmdBuf = nil; enc = nil
    }

    private func pipe(_ n: String) -> MTLComputePipelineState {
        guard let p = pipelines[n] else { fatalError("[RE-USE] pipeline \(n) missing") }
        return p
    }

    /// Dispatch a 1D-gridded kernel with an inline params struct.
    private func run<T>(_ name: String, _ buffers: [MTLBuffer], total: Int, params: T) {
        ensureEncoder()
        let p = pipe(name)
        enc!.setComputePipelineState(p)
        for (i, b) in buffers.enumerated() { enc!.setBuffer(b, offset: 0, index: i) }
        var params = params
        enc!.setBytes(&params, length: MemoryLayout<T>.stride, index: buffers.count)
        let wg = min(256, p.maxTotalThreadsPerThreadgroup)
        let grid = (total + wg - 1) / wg
        enc!.dispatchThreadgroups(MTLSizeMake(max(grid, 1), 1, 1),
                                  threadsPerThreadgroup: MTLSizeMake(wg, 1, 1))
    }

    /// Dispatch with explicit threadgroup grid (for reductions / matmul / instance-norm).
    private func runTG<T>(_ name: String, _ buffers: [MTLBuffer], params: T,
                          groups: MTLSize, tpg: MTLSize) {
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
    private struct Conv2dP { var in_ch: UInt32; var out_ch: UInt32; var H: UInt32; var W: UInt32
        var kh: UInt32; var kw: UInt32; var pad_h: UInt32; var pad_w: UInt32
        var str_h: UInt32; var str_w: UInt32; var dil_h: UInt32; var dil_w: UInt32
        var out_h: UInt32; var out_w: UInt32; var use_bias: UInt32; var _pad: UInt32 }
    private struct InstNormP { var channels: UInt32; var length: UInt32; var eps: Float }
    private struct ChanLenP { var channels: UInt32; var length: UInt32 }
    private struct Pad2dP { var channels: UInt32; var H: UInt32; var W: UInt32
        var pt: UInt32; var pb: UInt32; var pl: UInt32; var pr: UInt32 }
    private struct PixShuffleP { var out_ch: UInt32; var r: UInt32; var H: UInt32; var W: UInt32 }
    private struct TransHWP { var channels: UInt32; var H: UInt32; var W: UInt32 }
    private struct FlipP { var batch: UInt32; var L: UInt32; var dim: UInt32 }
    private struct MambaConvP { var batch: UInt32; var L: UInt32; var d_inner: UInt32; var in_stride: UInt32; var k: UInt32 }
    private struct ScanP { var batch: UInt32; var L: UInt32; var d_inner: UInt32; var d_state: UInt32; var dbl_stride: UInt32; var xz_stride: UInt32 }
    private struct SpecP { var num_freq: UInt32; var num_frames: UInt32 }
    private struct IrfftP { var n_fft: UInt32; var num_freq: UInt32; var num_frames: UInt32 }
    private struct OlaP { var n_fft: UInt32; var hop: UInt32; var num_frames: UInt32; var pad_len: UInt32; var out_len: UInt32; var trim: UInt32 }
    private struct Permute3dP { var d0: UInt32; var d1: UInt32; var d2: UInt32; var p0: UInt32; var p1: UInt32; var p2: UInt32 }
    private struct SliceColsP { var rows: UInt32; var in_stride: UInt32; var col_off: UInt32; var col_count: UInt32 }
    private struct ConcatColsP { var rows: UInt32; var cols_a: UInt32; var cols_b: UInt32 }
    private struct CropP { var in_h: UInt32; var in_w: UInt32; var out_h: UInt32; var out_w: UInt32 }
    private struct MatmulP { var M: UInt32; var K: UInt32; var N: UInt32; var use_bias: UInt32 }
    private struct LayerNormP { var batch_size: UInt32; var hidden_size: UInt32; var eps: Float }
    private struct SizeP { var size: UInt32 }
    private struct ConcatChP { var channels_a: UInt32; var channels_b: UInt32; var length: UInt32 }
    private struct TransposeP { var rows: UInt32; var cols: UInt32 }

    private lazy var dummyBias = makeBuffer([0], "dummy_bias")

    // MARK: - Primitive ops

    private func matmul(_ A: MTLBuffer, _ B: MTLBuffer, bias: MTLBuffer?, out: MTLBuffer,
                        M: Int, K: Int, N: Int) {
        let p = MatmulP(M: UInt32(M), K: UInt32(K), N: UInt32(N), use_bias: bias == nil ? 0 : 1)
        // Register-blocked GEMM: 64×64 tile per threadgroup, 256 threads (4×4 each).
        runTG("reuse_matmul_kernel", [A, B, bias ?? dummyBias, out], params: p,
              groups: MTLSize(width: (N + 63) / 64, height: (M + 63) / 64, depth: 1),
              tpg: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func add(_ a: MTLBuffer, _ b: MTLBuffer, _ out: MTLBuffer, _ n: Int) {
        run("add_kernel", [a, b, out], total: n, params: SizeP(size: UInt32(n)))
    }

    private func conv2d(_ input: MTLBuffer, wName: String, bName: String?,
                        inCh: Int, H: Int, W: Int, outCh: Int, kh: Int, kw: Int,
                        padH: Int, padW: Int, strH: Int, strW: Int, dilH: Int, dilW: Int,
                        label: String) throws -> (MTLBuffer, Int, Int) {
        let outH = (H + 2 * padH - dilH * (kh - 1) - 1) / strH + 1
        let outW = (W + 2 * padW - dilW * (kw - 1) - 1) / strW + 1
        let out = empty(outCh * outH * outW, label)
        let bias = bName != nil ? try w(bName!) : dummyBias
        let p = Conv2dP(in_ch: UInt32(inCh), out_ch: UInt32(outCh), H: UInt32(H), W: UInt32(W),
                        kh: UInt32(kh), kw: UInt32(kw), pad_h: UInt32(padH), pad_w: UInt32(padW),
                        str_h: UInt32(strH), str_w: UInt32(strW), dil_h: UInt32(dilH), dil_w: UInt32(dilW),
                        out_h: UInt32(outH), out_w: UInt32(outW), use_bias: bName != nil ? 1 : 0, _pad: 0)
        let rowSize = inCh * kh * kw
        let outHW = outH * outW
        if rowSize <= 4096 {
            // Tiled path: cache this channel's weights in threadgroup memory, reuse
            // across all spatial positions (the dominant cost for the dense blocks).
            runTG("reuse_conv2d_tiled_kernel", [input, try w(wName), bias, out], params: p,
                  groups: MTLSize(width: (outHW + 255) / 256, height: outCh, depth: 1),
                  tpg: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            run("reuse_conv2d_kernel", [input, try w(wName), bias, out], total: outCh * outHW, params: p)
        }
        return (out, outH, outW)
    }

    private func instanceNorm2d(_ input: MTLBuffer, gName: String, bName: String,
                                channels: Int, length: Int, label: String) throws -> MTLBuffer {
        let out = empty(channels * length, label)
        let p = InstNormP(channels: UInt32(channels), length: UInt32(length), eps: 1e-5)
        runTG("reuse_instance_norm2d_kernel", [input, try w(gName), try w(bName), out], params: p,
              groups: MTLSize(width: channels, height: 1, depth: 1),
              tpg: MTLSize(width: 256, height: 1, depth: 1))
        return out
    }

    private func prelu(_ input: MTLBuffer, wName: String, channels: Int, length: Int,
                       label: String) throws -> MTLBuffer {
        let out = empty(channels * length, label)
        run("reuse_prelu_kernel", [input, try w(wName), out], total: channels * length,
            params: ChanLenP(channels: UInt32(channels), length: UInt32(length)))
        return out
    }

    /// conv2d → InstanceNorm2d(affine) → PReLU, the repeated `Sequential(conv, IN, PReLU)`.
    private func convINPReLU(_ prefix: String, input: MTLBuffer, inCh: Int, H: Int, W: Int,
                             outCh: Int, kh: Int, kw: Int, padH: Int, padW: Int,
                             strH: Int, strW: Int, dilH: Int, dilW: Int,
                             label: String) throws -> (MTLBuffer, Int, Int) {
        let (c, oh, ow) = try conv2d(input, wName: "\(prefix).0.weight", bName: "\(prefix).0.bias",
                                     inCh: inCh, H: H, W: W, outCh: outCh, kh: kh, kw: kw,
                                     padH: padH, padW: padW, strH: strH, strW: strW,
                                     dilH: dilH, dilW: dilW, label: "\(label)_c")
        let n = try instanceNorm2d(c, gName: "\(prefix).1.weight", bName: "\(prefix).1.bias",
                                   channels: outCh, length: oh * ow, label: "\(label)_n")
        let a = try prelu(n, wName: "\(prefix).2.weight", channels: outCh, length: oh * ow, label: "\(label)_p")
        return (a, oh, ow)
    }

    // MARK: - Dense encoder / dense block

    private func denseBlock(_ prefix: String, input: MTLBuffer, H: Int, W: Int,
                            label: String) throws -> MTLBuffer {
        var skip = input
        var skipCh = hidFeature
        var last = input
        for i in 0..<4 {
            let dil = 1 << i
            let (x, _, _) = try convINPReLU("\(prefix).dense_block.\(i)", input: skip,
                                            inCh: skipCh, H: H, W: W, outCh: hidFeature,
                                            kh: 3, kw: 3, padH: dil, padW: 1,
                                            strH: 1, strW: 1, dilH: dil, dilW: 1, label: "\(label)_db\(i)")
            // skip = cat([x, skip], dim=channel)
            let newCh = skipCh + hidFeature
            let cat = empty(newCh * H * W, "\(label)_cat\(i)")
            run("concat_channels_kernel", [x, skip, cat], total: newCh * H * W,
                params: ConcatChP(channels_a: UInt32(hidFeature), channels_b: UInt32(skipCh),
                                  length: UInt32(H * W)))
            skip = cat; skipCh = newCh; last = x
        }
        return last
    }

    private func denseEncoder(_ input2ch: MTLBuffer, Tp: Int, Fp: Int) throws -> (MTLBuffer, Int, Int) {
        // dense_conv_1: 2 -> 64, 1x1
        let (c1, h1, w1) = try convINPReLU("dense_encoder.dense_conv_1", input: input2ch,
                                           inCh: 2, H: Tp, W: Fp, outCh: hidFeature,
                                           kh: 1, kw: 1, padH: 0, padW: 0, strH: 1, strW: 1,
                                           dilH: 1, dilW: 1, label: "enc_c1")
        // dense block (depth 4)
        let db = try denseBlock("dense_encoder.dense_block", input: c1, H: h1, W: w1, label: "enc_db")
        // dense_conv_2: 64 -> 64, (1,3) stride (4,2)
        let (c2, h2, w2) = try convINPReLU("dense_encoder.dense_conv_2", input: db,
                                           inCh: hidFeature, H: h1, W: w1, outCh: hidFeature,
                                           kh: 1, kw: 3, padH: 0, padW: 0, strH: 4, strW: 2,
                                           dilH: 1, dilW: 1, label: "enc_c2")
        return (c2, h2, w2)  // [64, T2, F2]
    }

    // MARK: - Mamba

    private func permute3d(_ input: MTLBuffer, d0: Int, d1: Int, d2: Int,
                           p0: Int, p1: Int, p2: Int, label: String) -> MTLBuffer {
        let total = d0 * d1 * d2
        let out = empty(total, label)
        run("reuse_permute3d_kernel", [input, out], total: total,
            params: Permute3dP(d0: UInt32(d0), d1: UInt32(d1), d2: UInt32(d2),
                               p0: UInt32(p0), p1: UInt32(p1), p2: UInt32(p2)))
        return out
    }

    /// A single `Mamba` (mamba_ssm) forward pass over X [batch, L, D=64].
    private func mamba(_ prefix: String, _ X: MTLBuffer, batch: Int, L: Int) throws -> MTLBuffer {
        let D = hidFeature, di = dInner, ns = dState
        let rows = batch * L
        let dbl = dtRank + 2 * ns              // 36

        // in_proj (no bias): [rows, D] @ [D, 2*di] -> xz [rows, 2*di]
        let xz = empty(rows * 2 * di, "\(prefix)_xz")
        matmul(X, try w("\(prefix).in_proj.weight"), bias: nil, out: xz, M: rows, K: D, N: 2 * di)

        // depthwise causal conv over L on the first di channels, + bias, + SiLU
        let u = empty(rows * di, "\(prefix)_u")
        run("reuse_mamba_conv_kernel",
            [xz, try w("\(prefix).conv1d.weight"), try w("\(prefix).conv1d.bias"), u],
            total: rows * di,
            params: MambaConvP(batch: UInt32(batch), L: UInt32(L), d_inner: UInt32(di),
                               in_stride: UInt32(2 * di), k: UInt32(dConv)))

        // x_proj: u [rows, di] @ [di, dbl] -> xdbl [rows, dbl]
        let xdbl = empty(rows * dbl, "\(prefix)_xdbl")
        matmul(u, try w("\(prefix).x_proj.weight"), bias: nil, out: xdbl, M: rows, K: di, N: dbl)

        // dt: slice first dtRank cols, project to di (bias handled inside scan)
        let dtSlice = empty(rows * dtRank, "\(prefix)_dtslice")
        run("reuse_slice_cols_kernel", [xdbl, dtSlice], total: rows * dtRank,
            params: SliceColsP(rows: UInt32(rows), in_stride: UInt32(dbl), col_off: 0, col_count: UInt32(dtRank)))
        let dtRaw = empty(rows * di, "\(prefix)_dtraw")
        matmul(dtSlice, try w("\(prefix).dt_proj.weight"), bias: nil, out: dtRaw, M: rows, K: dtRank, N: di)

        // selective scan
        let y = empty(rows * di, "\(prefix)_y")
        run("reuse_selective_scan_kernel",
            [u, dtRaw, xdbl, xz, try w("\(prefix).A_log"), try w("\(prefix).D"),
             try w("\(prefix).dt_proj.bias"), y],
            total: batch * di,
            params: ScanP(batch: UInt32(batch), L: UInt32(L), d_inner: UInt32(di),
                          d_state: UInt32(ns), dbl_stride: UInt32(dbl), xz_stride: UInt32(2 * di)))

        // out_proj (no bias): y [rows, di] @ [di, D] -> [rows, D]
        let out = empty(rows * D, "\(prefix)_outproj")
        matmul(y, try w("\(prefix).out_proj.weight"), bias: nil, out: out, M: rows, K: di, N: D)
        return out
    }

    /// Bidirectional MambaBlock over X [batch, L, 64].
    private func mambaBlock(_ prefix: String, _ X: MTLBuffer, batch: Int, L: Int) throws -> MTLBuffer {
        let D = hidFeature
        let rows = batch * L

        // forward
        let moF = try mamba("\(prefix).forward_blocks", X, batch: batch, L: L)
        let outFw = empty(rows * D, "\(prefix)_outfw")
        add(moF, X, outFw, rows * D)

        // backward: flip along L, run, add flipped X, flip result back
        let Xr = empty(rows * D, "\(prefix)_Xr")
        run("reuse_flip_kernel", [X, Xr], total: rows * D,
            params: FlipP(batch: UInt32(batch), L: UInt32(L), dim: UInt32(D)))
        let moB = try mamba("\(prefix).backward_blocks", Xr, batch: batch, L: L)
        let sumB = empty(rows * D, "\(prefix)_sumb")
        add(moB, Xr, sumB, rows * D)
        let outBw = empty(rows * D, "\(prefix)_outbw")
        run("reuse_flip_kernel", [sumB, outBw], total: rows * D,
            params: FlipP(batch: UInt32(batch), L: UInt32(L), dim: UInt32(D)))

        // cat([out_fw, out_bw], -1) -> [rows, 128]
        let cat = empty(rows * 2 * D, "\(prefix)_cat")
        run("reuse_concat_cols_kernel", [outFw, outBw, cat], total: rows * 2 * D,
            params: ConcatColsP(rows: UInt32(rows), cols_a: UInt32(D), cols_b: UInt32(D)))

        // output_proj: [rows, 128] @ [128, 64] + bias
        let proj = empty(rows * D, "\(prefix)_proj")
        matmul(cat, try w("\(prefix).output_proj.weight"), bias: try w("\(prefix).output_proj.bias"),
               out: proj, M: rows, K: 2 * D, N: D)

        // LayerNorm(64)
        let normed = empty(rows * D, "\(prefix)_norm")
        runTG("layer_norm_kernel", [proj, try w("\(prefix).norm.weight"), try w("\(prefix).norm.bias"), normed],
              params: LayerNormP(batch_size: UInt32(rows), hidden_size: UInt32(D), eps: 1e-5),
              groups: MTLSize(width: (rows + 255) / 256, height: 1, depth: 1),
              tpg: MTLSize(width: 256, height: 1, depth: 1))
        return normed
    }

    /// One TFMambaBlock over feature map x [c=64, t, f].
    private func tfMambaBlock(_ layer: Int, _ x: MTLBuffer, T2: Int, F2: Int) throws -> MTLBuffer {
        let c = hidFeature
        // time path: [c,t,f] -> [f,t,c]  (batch=f, L=t, D=c)
        let xTime = permute3d(x, d0: c, d1: T2, d2: F2, p0: 2, p1: 1, p2: 0, label: "L\(layer)_ftc")
        let moT = try mambaBlock("TSMamba.\(layer).time_mamba", xTime, batch: F2, L: T2)
        let xT = empty(F2 * T2 * c, "L\(layer)_timeres")
        add(moT, xTime, xT, F2 * T2 * c)

        // freq path: [f,t,c] -> [t,f,c]  (batch=t, L=f, D=c)
        let xFreq = permute3d(xT, d0: F2, d1: T2, d2: c, p0: 1, p1: 0, p2: 2, label: "L\(layer)_tfc")
        let moF = try mambaBlock("TSMamba.\(layer).freq_mamba", xFreq, batch: T2, L: F2)
        let xF = empty(T2 * F2 * c, "L\(layer)_freqres")
        add(moF, xFreq, xF, T2 * F2 * c)

        // back to [c,t,f]: from [t,f,c] take input axes (2,0,1)
        return permute3d(xF, d0: T2, d1: F2, d2: c, p0: 2, p1: 0, p2: 1, label: "L\(layer)_ctf")
    }

    // MARK: - Decoders

    private func transposeHW(_ input: MTLBuffer, channels: Int, H: Int, W: Int, label: String) -> MTLBuffer {
        let out = empty(channels * H * W, label)
        run("reuse_transpose_hw_kernel", [input, out], total: channels * H * W,
            params: TransHWP(channels: UInt32(channels), H: UInt32(H), W: UInt32(W)))
        return out  // now [channels, W, H]
    }

    /// SPConvTranspose2d upsampling along the width dim by factor r, then IN + PReLU.
    private func upConv(_ prefix: String, input: MTLBuffer, inCh: Int, H: Int, W: Int,
                        r: Int, label: String) throws -> (MTLBuffer, Int, Int) {
        // pad width by 1 on each side
        let padded = empty(inCh * H * (W + 2), "\(label)_pad")
        run("reuse_pad2d_kernel", [input, padded], total: inCh * H * (W + 2),
            params: Pad2dP(channels: UInt32(inCh), H: UInt32(H), W: UInt32(W),
                           pt: 0, pb: 0, pl: 1, pr: 1))
        // conv (1,3) -> out_ch*r channels, out width = W
        let outCh = inCh   // decoders keep 64 channels
        let (conv, ch, cw) = try conv2d(padded, wName: "\(prefix).0.conv.weight", bName: "\(prefix).0.conv.bias",
                                        inCh: inCh, H: H, W: W + 2, outCh: outCh * r, kh: 1, kw: 3,
                                        padH: 0, padW: 0, strH: 1, strW: 1, dilH: 1, dilW: 1, label: "\(label)_conv")
        // pixel shuffle along width: [outCh*r, H, cw] -> [outCh, H, cw*r]
        let shuffled = empty(outCh * ch * cw * r, "\(label)_shuf")
        run("reuse_pixelshuffle_w_kernel", [conv, shuffled], total: outCh * ch * cw * r,
            params: PixShuffleP(out_ch: UInt32(outCh), r: UInt32(r), H: UInt32(ch), W: UInt32(cw)))
        let outW = cw * r
        let n = try instanceNorm2d(shuffled, gName: "\(prefix).1.weight", bName: "\(prefix).1.bias",
                                   channels: outCh, length: ch * outW, label: "\(label)_in")
        let a = try prelu(n, wName: "\(prefix).2.weight", channels: outCh, length: ch * outW, label: "\(label)_pr")
        return (a, ch, outW)
    }

    /// Shared decoder trunk: dense block, up_conv1 (freq x2), up_conv2 (time x4).
    private func decoderTrunk(_ base: String, x: MTLBuffer, T2: Int, F2: Int) throws -> (MTLBuffer, Int, Int) {
        let db = try denseBlock("\(base).dense_block", input: x, H: T2, W: F2, label: "\(base)_db")
        stage("  \(base).dense_block")
        // up_conv1: upsample freq (W) by 2
        let (u1, h1, w1) = try upConv("\(base).up_conv1", input: db, inCh: hidFeature, H: T2, W: F2,
                                      r: 2, label: "\(base)_up1")   // [64, T2, 2F2]
        stage("  \(base).up_conv1")
        // up_conv2 operates on the transposed map to upsample time by 4
        let tp = transposeHW(u1, channels: hidFeature, H: h1, W: w1, label: "\(base)_up2tp")  // [64, 2F2, T2]
        let (u2, h2, w2) = try upConv("\(base).up_conv2", input: tp, inCh: hidFeature, H: w1, W: h1,
                                      r: 4, label: "\(base)_up2")   // [64, 2F2, 4T2]
        stage("  \(base).up_conv2")
        let back = transposeHW(u2, channels: hidFeature, H: h2, W: w2, label: "\(base)_up2back") // [64, 4T2, 2F2]
        return (back, w2, h2)   // (buf, outH=4T2, outW=2F2)
    }

    /// Transpose [1, H, W] time/freq map to [W, H] then crop to [F, T].
    private func toFreqTimeCropped(_ input: MTLBuffer, H: Int, W: Int, F: Int, T: Int,
                                   label: String) -> MTLBuffer {
        let tr = transposeHW(input, channels: 1, H: H, W: W, label: "\(label)_tr")  // [W, H]
        let out = empty(F * T, "\(label)_crop")
        run("reuse_crop2d_kernel", [tr, out], total: F * T,
            params: CropP(in_h: UInt32(W), in_w: UInt32(H), out_h: UInt32(F), out_w: UInt32(T)))
        return out
    }

    private func magDecoder(_ x: MTLBuffer, T2: Int, F2: Int, F: Int, T: Int) throws -> MTLBuffer {
        let (trunk, oh, ow) = try decoderTrunk("mask_decoder", x: x, T2: T2, F2: F2)
        let (fin, _, _) = try conv2d(trunk, wName: "mask_decoder.final_conv.weight",
                                     bName: "mask_decoder.final_conv.bias",
                                     inCh: hidFeature, H: oh, W: ow, outCh: 1, kh: 1, kw: 1,
                                     padH: 0, padW: 0, strH: 1, strW: 1, dilH: 1, dilW: 1, label: "mag_final")
        return toFreqTimeCropped(fin, H: oh, W: ow, F: F, T: T, label: "mag_out")
    }

    private func phaseDecoder(_ x: MTLBuffer, T2: Int, F2: Int, F: Int, T: Int) throws -> MTLBuffer {
        let (trunk, oh, ow) = try decoderTrunk("phase_decoder", x: x, T2: T2, F2: F2)
        let (xr, _, _) = try conv2d(trunk, wName: "phase_decoder.phase_conv_r.weight",
                                    bName: "phase_decoder.phase_conv_r.bias",
                                    inCh: hidFeature, H: oh, W: ow, outCh: 1, kh: 1, kw: 1,
                                    padH: 0, padW: 0, strH: 1, strW: 1, dilH: 1, dilW: 1, label: "pha_r")
        let (xi, _, _) = try conv2d(trunk, wName: "phase_decoder.phase_conv_i.weight",
                                    bName: "phase_decoder.phase_conv_i.bias",
                                    inCh: hidFeature, H: oh, W: ow, outCh: 1, kh: 1, kw: 1,
                                    padH: 0, padW: 0, strH: 1, strW: 1, dilH: 1, dilW: 1, label: "pha_i")
        let ph = empty(oh * ow, "pha_atan2")
        run("reuse_atan2_kernel", [xi, xr, ph], total: oh * ow, params: SizeP(size: UInt32(oh * ow)))
        return toFreqTimeCropped(ph, H: oh, W: ow, F: F, T: T, label: "pha_out")
    }

    // MARK: - STFT / ISTFT

    private func hannWindow(_ n: Int) -> [Float] {
        var w = [Float](repeating: 0, count: n)
        for k in 0..<n { w[k] = 0.5 - 0.5 * cosf(2 * Float.pi * Float(k) / Float(n)) }  // periodic
        return w
    }

    // MARK: - Public entry point

    /// Enhance + upsample. `input` is mono PCM at `inputSR`; returns mono PCM at `targetSR`.
    func enhance(_ input: [Float], inputSR: Double, targetSR: Double) throws -> [Float] {
        guard isLoaded, input.count > 8 else { return input }
        scratch.removeAll(keepingCapacity: true)

        // 1. Resample to target rate (bandwidth extension input).
        let signal = (abs(inputSR - targetSR) < 1) ? input
                                                    : Resampler.resample(input, from: inputSR, to: targetSR)
        let Lw = signal.count

        // STFT params scaled to the target rate (mirrors inference.py make_even scaling).
        let nfft = makeEven(baseNFFT * Int(targetSR) / baseSR)     // 1920 @ 48 kHz
        let hop = makeEven(baseHop * Int(targetSR) / baseSR)       // 240
        let F = nfft / 2 + 1                                       // 961
        let T = Lw / hop + 1                                       // frames (center=True)
        if T < 4 { return input }

        // 2. Reflect-pad and STFT on GPU.
        let pad = nfft / 2
        let padLen = Lw + 2 * pad
        var padded = [Float](repeating: 0, count: padLen)
        for i in 0..<padLen {
            var src = i - pad
            if src < 0 { src = -src }                       // reflect (no edge repeat)
            if src >= Lw { src = 2 * Lw - 2 - src }
            src = min(max(src, 0), Lw - 1)
            padded[i] = signal[src]
        }
        let sigBuf = makeBuffer(padded, "reuse_signal")
        let winBuf = makeBuffer(hannWindow(nfft), "reuse_window")

        let mag = empty(F * T, "stft_mag")
        let pha = empty(F * T, "stft_pha")
        run("reuse_stft_kernel", [sigBuf, winBuf, mag, pha], total: F * T,
            params: StftP(n_fft: UInt32(nfft), hop: UInt32(hop), num_freq: UInt32(F), num_frames: UInt32(T)))

        // 3. Build the padded [2, T+2, F+2] encoder input.
        //    rearrange 'f t -> t f' per channel, cat(mag, pha), then zero-pad (T+2, F+2).
        let magT = empty(T * F, "magT")
        run("transpose_kernel", [mag, magT], total: F * T, params: TransposeP(rows: UInt32(F), cols: UInt32(T)))
        let phaT = empty(T * F, "phaT")
        run("transpose_kernel", [pha, phaT], total: F * T, params: TransposeP(rows: UInt32(F), cols: UInt32(T)))
        let twoCh = empty(2 * T * F, "twoCh")
        run("concat_channels_kernel", [magT, phaT, twoCh], total: 2 * T * F,
            params: ConcatChP(channels_a: 1, channels_b: 1, length: UInt32(T * F)))
        let Tp = T + 2, Fp = F + 2
        let encIn = empty(2 * Tp * Fp, "encIn")
        run("reuse_pad2d_kernel", [twoCh, encIn], total: 2 * Tp * Fp,
            params: Pad2dP(channels: 2, H: UInt32(T), W: UInt32(F), pt: 0, pb: 2, pl: 0, pr: 2))

        stage("stft+prep")
        // 4. Dense encoder → [64, T2, F2]
        var (feat, T2, F2) = try denseEncoder(encIn, Tp: Tp, Fp: Fp)
        stage("dense_encoder")

        // 5. 30 TFMamba blocks. Commit (without blocking) every few layers to bound
        //    the command-buffer size and let the GPU start early. `feat` stays alive
        //    via its local reference; the rest of each batch's scratch is retained by
        //    the completion handler until the GPU finishes with it.
        for layer in 0..<numLayers {
            feat = try tfMambaBlock(layer, feat, T2: T2, F2: F2)
            if layer % 3 == 2 { flush() }
        }
        stage("30x mamba")

        // 6. Decoders → denoised magnitude & phase [F, T]
        let ampFT = try magDecoder(feat, T2: T2, F2: F2, F: F, T: T)
        let phaFT = try phaseDecoder(feat, T2: T2, F2: F2, F: F, T: T)
        stage("decoders")

        // 7. amp/phase → complex (with artifact-frame zeroing)
        let zeroFrac = empty(T, "zeroFrac")
        run("reuse_zero_frac_kernel", [ampFT, zeroFrac], total: T,
            params: SpecP(num_freq: UInt32(F), num_frames: UInt32(T)))
        let re = empty(F * T, "spec_re")
        let im = empty(F * T, "spec_im")
        run("reuse_spec_to_complex_kernel", [ampFT, phaFT, zeroFrac, re, im], total: F * T,
            params: SpecP(num_freq: UInt32(F), num_frames: UInt32(T)))

        // 8. iSTFT: per-frame windowed irfft, then overlap-add + normalize.
        let frames = empty(T * nfft, "istft_frames")
        run("reuse_irfft_kernel", [re, im, winBuf, frames], total: T * nfft,
            params: IrfftP(n_fft: UInt32(nfft), num_freq: UInt32(F), num_frames: UInt32(T)))
        let outLen = (T - 1) * hop
        let waveBuf = empty(outLen, "istft_wave")
        run("reuse_ola_kernel", [frames, winBuf, waveBuf], total: outLen,
            params: OlaP(n_fft: UInt32(nfft), hop: UInt32(hop), num_frames: UInt32(T),
                         pad_len: UInt32(padLen), out_len: UInt32(outLen), trim: UInt32(pad)))

        var wave = readBuffer(waveBuf, outLen)
        scratch.removeAll(keepingCapacity: true)

        // Match the resampled input length (epsilon pad / trim), like pad_or_trim_to_match.
        if wave.count < Lw { wave.append(contentsOf: [Float](repeating: 1e-8, count: Lw - wave.count)) }
        else if wave.count > Lw { wave = Array(wave[0..<Lw]) }
        return wave
    }

    private func makeEven(_ v: Int) -> Int { v % 2 == 0 ? v : v + 1 }

    /// On-device smoke test: enhance a synthetic 24 kHz tone and log shape/energy.
    /// Triggered by `REUSE_SELFTEST=1`. Exercises every kernel end-to-end.
    func selfTest() {
        let sr = 24000.0
        let secs = Double(ProcessInfo.processInfo.environment["REUSE_SECS"] ?? "3.5") ?? 3.5
        let n = Int(secs * sr)
        var x = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Double(i) / sr
            x[i] = Float(0.3 * sin(2 * .pi * 220 * t) + 0.15 * sin(2 * .pi * 660 * t))
        }
        profile = true
        let t0 = Date()
        do {
            let y = try enhance(x, inputSR: sr, targetSR: 48000)
            let dt = Date().timeIntervalSince(t0)
            var rms: Double = 0, peak: Float = 0, nan = 0
            for v in y { rms += Double(v * v); peak = max(peak, abs(v)); if !v.isFinite { nan += 1 } }
            rms = (y.isEmpty ? 0 : (rms / Double(y.count)).squareRoot())
            print(String(format: "[RE-USE selftest] in=%d out=%d (%.2fx) rms=%.4f peak=%.4f nonfinite=%d time=%.2fs",
                         n, y.count, Double(y.count) / Double(n), rms, peak, nan, dt))
        } catch {
            print("[RE-USE selftest] FAILED: \(error)")
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
        // When downsampling, widen the kernel to avoid aliasing; here we upsample (ratio>1).
        let scale = min(1.0, ratio)          // kernel cutoff in source samples
        let af = Double(a)
        x.withUnsafeBufferPointer { xp in
            for o in 0..<outN {
                let pos = Double(o) / ratio    // position in source-sample coordinates
                let center = Int(pos.rounded(.down))
                let half = Int((af / scale).rounded(.up))
                var acc = 0.0
                var norm = 0.0
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
