//
//  MetalTtsEngine.swift
//  tts-metal
//
//  Swift/Metal port of src/engine.ts — Kitten TTS inference runtime.
//

import Foundation
import Metal
import Accelerate

// Not @MainActor: inference is heavy (GPU encode + CPU DSP) and must run off the
// main thread. Access is serialized by TtsController's `canGenerate` guard, so the
// mutable state is only ever touched by one task at a time.
final class MetalTtsEngine: @unchecked Sendable {
    // MARK: - Public state

    enum Status: Equatable {
        case uninitialized
        case loading(String)
        case ready
        case generating(String)
        case error(String)
    }
    private(set) var status: Status = .uninitialized

    // MARK: - Metal handles

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var library: MTLLibrary!
    private var pipelines: [String: MTLComputePipelineState] = [:]

    // MARK: - Weights & voices

    private var weights: [String: WeightTensor] = [:]
    private var weightAliases: [String: String] = [:]
    private var voices: [String: [Float]] = [:]

    struct WeightTensor {
        let buffer: MTLBuffer
        let shape: [Int]
        var size: Int { shape.reduce(1, *) }
    }

    // Param structs (mirrors of Shaders.metal — used for setBytes payloads)
    struct EmbeddingParams { var seq_len: UInt32; var embed_dim: UInt32; var vocab_size: UInt32 }
    struct LayerNormParams { var batch_size: UInt32; var hidden_size: UInt32; var eps: Float }
    struct MatmulParams { var M: UInt32; var K: UInt32; var N: UInt32; var use_bias: UInt32 }
    struct MhaParams { var seq_len: UInt32; var num_heads: UInt32; var head_dim: UInt32; var scale: Float }
    struct Conv1dParams { var in_channels: UInt32; var out_channels: UInt32; var kernel_size: UInt32; var input_length: UInt32; var output_length: UInt32; var padding: UInt32; var stride: UInt32; var dilation: UInt32; var use_bias: UInt32 }
    struct LstmParams { var seq_len: UInt32; var input_size: UInt32; var hidden_size: UInt32; var num_directions: UInt32 }
    struct TransposeParams { var rows: UInt32; var cols: UInt32 }
    struct SizeParams { var size: UInt32 }
    struct SizeAlphaParams { var size: UInt32; var alpha: Float }
    struct ScaleParams { var size: UInt32; var _pad: UInt32; var scale: Float }
    struct InstanceNormParams { var channels: UInt32; var length: UInt32; var eps: Float }
    struct AdainParams { var channels: UInt32; var length: UInt32 }
    struct AdainRowMajorParams { var channels: UInt32; var total: UInt32 }
    struct ConvTranspose1dParams { var in_channels: UInt32; var out_channels: UInt32; var kernel_size: UInt32; var input_length: UInt32; var output_length: UInt32; var stride: UInt32; var padding: UInt32; var use_bias: UInt32 }
    struct DepthwiseConvTParams { var channels: UInt32; var kernel_size: UInt32; var input_length: UInt32; var output_length: UInt32; var stride: UInt32; var padding: UInt32 }
    struct Resize1dParams { var channels: UInt32; var input_length: UInt32; var output_length: UInt32 }
    struct ConcatChannelsParams { var channels_a: UInt32; var channels_b: UInt32; var length: UInt32 }
    struct ConcatBroadcastParams { var rows: UInt32; var cols_a: UInt32; var cols_b: UInt32 }
    struct ReflectionPadParams { var channels: UInt32; var input_length: UInt32; var pad_left: UInt32; var pad_right: UInt32 }
    struct ExpandParams { var seq_len: UInt32; var dim: UInt32; var total_frames: UInt32 }
    struct IstftParams { var gen_length: UInt32; var waveform_length: UInt32; var bins: UInt32; var kernel_size: UInt32; var stride: UInt32 }

    // MARK: - Detected dimensions (mini defaults)

    var lstmHidden = 256
    var lstmBidir = 512
    var textEncChannels = 512
    var styleDim = 256
    var styleHalf = 128
    var lstmInputSize = 640
    var bertEmbedDim = 128
    var bertHiddenSize = 768
    var bertNumHeads = 12
    var bertHeadDim = 64
    var bertFfnDim = 2048
    var bertNumLayers = 12
    var numPredLstmPairs = 3
    var bertProjDim = 512
    var numTextEncCnnBlocks = 3
    var decEncodeOutCh = 1024
    var decDecodeOutCh = 1024
    var decDecode3OutCh = 512
    var hifiUps0OutCh = 256
    var hifiUps1OutCh = 128
    var predBlock0OutCh = 512
    var predBlock1OutCh = 256

    // MARK: - Encoder batching

    private var cmdBuf: MTLCommandBuffer?
    private var enc: MTLComputeCommandEncoder?
    private var deferredDestroys: [MTLBuffer] = []

    // Cached CPU copies of sin-generator weights (avoid read-backs every inference)
    private var sinGenWeights: SinGenWeights?
    struct SinGenWeights {
        let linearWeight: [Float]   // [9]
        let linearBias: [Float]     // [1]
        let fwdReal: [Float]        // [11*20]
        let fwdImag: [Float]        // [11*20]
    }

    // Timing log
    struct StageTime { let name: String; let ms: Double }
    private(set) var lastTimings: [StageTime] = []
    private var stageStart: Double = 0
    private var timingsAccum: [(String, Double)] = []
    var profile = false

    // MARK: - Init

init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
    }

    @discardableResult
    func load() throws -> Bool {
        status = .loading("Compiling Metal shaders")
        // Xcode compiles Shaders.metal at build time into default.metallib inside the app bundle.
        guard let lib = device.makeDefaultLibrary() else {
            status = .error("default.metallib not found in bundle")
            throw NSError(domain: "TtsEngine", code: 11, userInfo: [NSLocalizedDescriptionKey: "default.metallib not found"])
        }
        library = lib
        compilePipelines()

        status = .loading("Parsing ONNX model")
        guard let onnxURL = Bundle.main.url(forResource: "kitten_tts_mini_v0_8", withExtension: "onnx") else {
            status = .error("ONNX bundle missing")
            return false
        }
        let onnxData = try Data(contentsOf: onnxURL, options: .alwaysMapped)
        let parser = OnnxParser(onnxData)
        let tensors = try parser.parseInitializers()
        print("[KittenTTS] Parsed \(tensors.count) weight tensors")

        status = .loading("Dequantizing & uploading weights")
        try uploadWeights(tensors)
        print("[KittenTTS] Uploaded \(weights.count) weight buffers")

        buildOnnxAliases()
        detectDimensions()

        status = .loading("Loading voices")
        guard let voicesURL = Bundle.main.url(forResource: "voices", withExtension: "npz") else {
            status = .error("voices.npz bundle missing")
            return false
        }
        let voicesData = try Data(contentsOf: voicesURL, options: .alwaysMapped)
        let npz = NpzParser.parse(voicesData)
        for (name, arr) in npz {
            voices[name] = arr.data
            print("[KittenTTS] Loaded voice: \(name) (\(arr.shape))")
        }
        for (_, data) in voices {
            styleDim = data.count / 400
            styleHalf = styleDim / 2
            lstmInputSize = lstmBidir + styleHalf
            break
        }

        status = .ready
        return true
    }

    private func compilePipelines() {
        let names = [
            "embedding_kernel", "layer_norm_kernel", "matmul_kernel", "matmul_gelu_kernel",
            "conv1d_kernel", "conv1d_tiled_kernel", "instance_norm_kernel", "adain_kernel",
            "adain_row_major_kernel", "snake_kernel", "leaky_relu_kernel", "gelu_kernel",
            "tanh_kernel", "sigmoid_kernel", "conv_transpose1d_kernel",
            "depthwise_conv_transpose1d_kernel", "resize1d_kernel", "softmax_kernel",
            "mha_kernel", "add_kernel", "scale_kernel", "add_scale_kernel", "concat_channels_kernel",
            "concat_broadcast_kernel", "reflection_pad1d_kernel", "alpha_residual_kernel",
            "transpose_kernel", "lstm_kernel", "expand_row_major_kernel",
            "expand_channel_first_kernel", "istft_kernel",
        ]
        for name in names {
            guard let fn = library.makeFunction(name: name) else {
                print("[KittenTTS] MISSING kernel \(name)")
                continue
            }
            do {
                let p = try device.makeComputePipelineState(function: fn)
                pipelines[name] = p
            } catch {
                print("[KittenTTS] Pipeline fail \(name): \(error)")
            }
        }
        print("[KittenTTS] Compiled \(pipelines.count) pipelines")
    }

    // MARK: - Weight upload & dequantization

    private func uploadWeights(_ tensors: [String: OnnxTensor]) throws {
        for (name, tensor) in tensors {
            if name.hasSuffix("_scale") || name.hasSuffix("_zero_point") { continue }
            let total = tensor.dims.reduce(1, *)
            if total == 0 { continue }
            if tensor.rawData.isEmpty {
                print("[KittenTTS] Skipping empty \(name)")
                continue
            }
            let floats: [Float]
            switch tensor.dataType {
            case OnnxDtype.float32:
                floats = OnnxDequant.float32Data(tensor.rawData)
            case OnnxDtype.float16:
                floats = OnnxDequant.float16Array(tensor.rawData)
            case OnnxDtype.int8:
                let (scales, zps) = scaleZp(for: name, in: tensors, tensor: tensor)
                floats = dequantInt8(tensor.rawData, scales: scales, zeroPoints: zps, total: total)
            case OnnxDtype.uint8:
                let (scales, zps) = scaleZp(for: name, in: tensors, tensor: tensor)
                floats = dequantUint8(tensor.rawData, scales: scales, zeroPoints: zps, total: total)
            case OnnxDtype.int64:
                continue
            default:
                print("[KittenTTS] Skipping \(name): dtype \(tensor.dataType)")
                continue
            }
            guard let buf = device.makeBuffer(bytes: floats,
                                             length: floats.count * MemoryLayout<Float>.size,
                                             options: [.storageModeShared]) else {
                throw NSError(domain: "TtsEngine", code: 10, userInfo: [NSLocalizedDescriptionKey: "Buffer alloc failed: \(name)"])
            }
            buf.label = name
            weights[name] = WeightTensor(buffer: buf, shape: tensor.dims)
        }
    }

    private func scaleZp(for qName: String, in tensors: [String: OnnxTensor], tensor: OnnxTensor)
        -> ([Float], [Int32]) {
        let base = qName.hasSuffix("_quantized") ? String(qName.dropLast("_quantized".count)) : qName
        var scales: [Float] = [1.0]
        var zeroPoints: [Int32] = [0]
        if let st = tensors["\(base)_scale"], st.rawData.count >= 4 {
            scales = OnnxDequant.float32Data(st.rawData)
        }
        if let zt = tensors["\(base)_zero_point"], zt.rawData.count >= 1 {
            let num = zt.dims.reduce(1, *)
            let raw = zt.rawData
            let isInt32 = (raw.count == num * 4) && raw.count != num
            zeroPoints = []
            if isInt32 {
                raw.withUnsafeBytes { ptr in
                    let i32 = ptr.bindMemory(to: Int32.self).baseAddress!
                    for i in 0..<num { zeroPoints.append(i32[i]) }
                }
            } else if zt.dataType == OnnxDtype.int8 {
                raw.withUnsafeBytes { ptr in
                    let i8 = ptr.bindMemory(to: Int8.self).baseAddress!
                    for i in 0..<raw.count { zeroPoints.append(Int32(i8[i])) }
                }
            } else {
                raw.withUnsafeBytes { ptr in
                    let u8 = ptr.bindMemory(to: UInt8.self).baseAddress!
                    for i in 0..<raw.count { zeroPoints.append(Int32(u8[i])) }
                }
            }
        }
        return (scales, zeroPoints)
    }

    private func dequantInt8(_ data: Data, scales: [Float], zeroPoints: [Int32], total: Int) -> [Float] {
        var out = [Float](repeating: 0, count: total)
        let i8: [Int8] = data.withUnsafeBytes { Array($0.bindMemory(to: Int8.self)) }
        if scales.count == 1 {
            let s = scales[0], zp = Float(Int(zeroPoints[0]))
            for j in 0..<total { out[j] = (Float(i8[j]) - zp) * s }
        } else {
            let axis = total / scales.count
            for a in 0..<scales.count {
                let s = scales[a], zp = Float(Int(zeroPoints[a]))
                let off = a * axis
                for j in 0..<axis { out[off + j] = (Float(i8[off + j]) - zp) * s }
            }
        }
        return out
    }

    private func dequantUint8(_ data: Data, scales: [Float], zeroPoints: [Int32], total: Int) -> [Float] {
        var out = [Float](repeating: 0, count: total)
        let u8: [UInt8] = data.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }
        if scales.count == 1 {
            let s = scales[0], zp = Float(Int(zeroPoints[0]))
            for j in 0..<total { out[j] = (Float(u8[j]) - zp) * s }
        } else {
            let axis = total / scales.count
            for a in 0..<scales.count {
                let s = scales[a], zp = Float(Int(zeroPoints[a]))
                let off = a * axis
                for j in 0..<axis { out[off + j] = (Float(u8[off + j]) - zp) * s }
            }
        }
        return out
    }

    // MARK: - Alias building (canonical mini → actual ids)

    private func buildOnnxAliases() {
        let canonMatMul = [5883, 5884, 5887, 5890, 5894, 5895, 5896, 6040, 6245, 6388]
        let canonLstm = [5873, 5874, 5875, 6093, 6094, 6095, 6143, 6144, 6145,
                        6193, 6194, 6195, 6242, 6243, 6244, 6291, 6292, 6293]
        var mm: [(Int, String)] = []
        var ls: [(Int, String)] = []
        for name in weights.keys {
            let base = name.hasSuffix("_quantized") ? String(name.dropLast("_quantized".count)) : name
            if let m = matchLinnum(base, prefix: "onnx::MatMul_") { mm.append((m.0, m.1)); continue }
            if let m = matchLinnum(base, prefix: "onnx::LSTM_")   { ls.append((m.0, m.1)); continue }
        }
        mm.sort { $0.0 < $1.0 }
        ls.sort { $0.0 < $1.0 }
        var uniqMM: [(Int, String)] = []
        var seen = Set<String>()
        for e in mm where !seen.contains(e.1) { seen.insert(e.1); uniqMM.append(e) }
        var uniqLS: [(Int, String)] = []
        seen.removeAll()
        for e in ls where !seen.contains(e.1) { seen.insert(e.1); uniqLS.append(e) }

        func createAlias(_ canon: String, _ actual: String) {
            if canon == actual { return }
            for suffix in ["", "_quantized"] {
                let canonName = canon + suffix
                let actualWithSuffix = actual + suffix
                if weights[actualWithSuffix] != nil {
                    weightAliases[canonName] = actualWithSuffix
                } else if weights[actual] != nil {
                    weightAliases[canonName] = actual
                }
            }
        }
        let mmCount = min(canonMatMul.count, uniqMM.count)
        for i in 0..<mmCount {
            createAlias("onnx::MatMul_\(canonMatMul[i])", uniqMM[i].1)
        }
        let numActualLstms = uniqLS.count / 3
        let numCanonLstms = canonLstm.count / 3
        let numActualPred = numActualLstms - 3
        let numCanonPred = numCanonLstms - 3
        for cg in 0..<numCanonLstms {
            let ag: Int
            if cg == 0 { ag = 0 }
            else if cg <= numCanonPred {
                if cg <= numActualPred { ag = cg } else { continue }
            } else if cg == numCanonPred + 1 {
                ag = numActualPred + 1
            } else {
                ag = numActualPred + 2
            }
            for w in 0..<3 {
                let cIdx = cg * 3 + w
                let aIdx = ag * 3 + w
                if cIdx < canonLstm.count && aIdx < uniqLS.count {
                    createAlias("onnx::LSTM_\(canonLstm[cIdx])", uniqLS[aIdx].1)
                }
            }
        }
    }

    private func matchLinnum(_ s: String, prefix: String) -> (Int, String)? {
        guard s.hasPrefix(prefix) else { return nil }
        let rem = String(s.dropFirst(prefix.count))
        if let n = Int(rem) { return (n, s) }
        return nil
    }

    // MARK: - Detect dimensions

    private func detectDimensions() {
        if let t = weights["kmodel.text_encoder.embedding.weight"] { textEncChannels = t.shape[1] }
        if let r = tryGetWeight("onnx::LSTM_5875_quantized") ?? tryGetWeight("onnx::LSTM_5875"),
           r.shape.count == 3 {
            lstmHidden = r.shape[1]
            lstmBidir = 2 * lstmHidden
        }
        if let b = weights["kmodel.bert.embeddings.word_embeddings.weight"] {
            bertEmbedDim = b.shape[1]
        }
        if let b = weights["kmodel.bert.encoder.embedding_hidden_mapping_in.bias"] {
            bertHiddenSize = b.size
        }
        if let b = weights["kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight"] {
            bertHiddenSize = b.size
        }
        bertHeadDim = 64
        bertNumHeads = max(1, bertHiddenSize / bertHeadDim)
        if let b = weights["kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias"] {
            bertFfnDim = b.size
        }
        if let b = weights["kmodel.bert_encoder.bias"] { bertProjDim = b.size }
        for (_, d) in voices {
            styleDim = d.count / 400
            styleHalf = styleDim / 2
            break
        }
        lstmInputSize = lstmBidir + styleHalf

        numPredLstmPairs = 0
        for i in 0..<10 {
            if tryGetWeight("kmodel.predictor.text_encoder.lstms.\(2*i+1).fc.weight_quantized") != nil {
                numPredLstmPairs = i + 1
            } else { break }
        }
        numTextEncCnnBlocks = 0
        for i in 0..<10 {
            if tryGetWeight("kmodel.text_encoder.cnn.\(i).0.weight_quantized") != nil ||
               tryGetWeight("kmodel.text_encoder.cnn.\(i).0.bias") != nil {
                numTextEncCnnBlocks = i + 1
            } else { break }
        }
        if let b = weights["kmodel.decoder.encode.conv2.bias"] { decEncodeOutCh = b.size }
        if let b = weights["kmodel.decoder.decode.0.conv2.bias"] { decDecodeOutCh = b.size }
        if let b = weights["kmodel.decoder.decode.3.conv2.bias"] { decDecode3OutCh = b.size }
        if let b = weights["kmodel.decoder.generator.ups.0.bias"] { hifiUps0OutCh = b.size }
        if let b = weights["kmodel.decoder.generator.ups.1.bias"] { hifiUps1OutCh = b.size }
        if let b = weights["kmodel.predictor.N.0.conv1.bias"] { predBlock0OutCh = b.size }
        if let b = weights["kmodel.predictor.N.1.conv1.bias"] { predBlock1OutCh = b.size }

        print("[KittenTTS] dims: lstmH=\(lstmHidden) bidir=\(lstmBidir) bertH=\(bertHiddenSize) bertHeads=\(bertNumHeads) bertFfn=\(bertFfnDim) style=\(styleDim) styleHalf=\(styleHalf) predPairs=\(numPredLstmPairs) cnn=\(numTextEncCnnBlocks) decEnc=\(decEncodeOutCh) decDec=\(decDecodeOutCh) decDec3=\(decDecode3OutCh) ups0=\(hifiUps0OutCh) ups1=\(hifiUps1OutCh) predB0=\(predBlock0OutCh) predB1=\(predBlock1OutCh)")
    }

    // MARK: - Weight lookup

    private func tryGetWeight(_ name: String) -> WeightTensor? {
        if let w = weights[name] { return w }
        if let alias = weightAliases[name], let w = weights[alias] { return w }
        if name.hasSuffix("_quantized") {
            let base = String(name.dropLast("_quantized".count))
            if let w = weights[base] { return w }
            if let alias = weightAliases[base], let w = weights[alias] { return w }
        } else {
            let q = name + "_quantized"
            if let w = weights[q] { return w }
            if let alias = weightAliases[q], let w = weights[alias] { return w }
        }
        return nil
    }
    private func requireWeight(_ name: String) throws -> WeightTensor {
        guard let w = tryGetWeight(name) else {
            throw NSError(domain: "TtsEngine", code: 20, userInfo: [NSLocalizedDescriptionKey: "Missing weight: \(name)"])
        }
        return w
    }

    // MARK: - Buffer helpers

    private func makeBuffer(_ floats: [Float], label: String) -> MTLBuffer {
        let buf = device.makeBuffer(bytes: floats, length: floats.count * 4, options: [.storageModeShared])!
        buf.label = label
        return buf
    }
    private func makeBuffer(_ ints: [Int32], label: String) -> MTLBuffer {
        let buf = device.makeBuffer(bytes: ints, length: ints.count * 4, options: [.storageModeShared])!
        buf.label = label
        return buf
    }
    private func empty(_ elements: Int, label: String) -> MTLBuffer {
        let buf = device.makeBuffer(length: elements * 4, options: [.storageModeShared])!
        buf.label = label
        return buf
    }
    private func deferDestroy(_ b: MTLBuffer?) { if let b = b { deferredDestroys.append(b) } }

    // MARK: - Encoder batching

    private func ensureEncoder() {
        if cmdBuf == nil {
            cmdBuf = commandQueue.makeCommandBuffer()
            cmdBuf?.label = "tts-metal"
            enc = cmdBuf!.makeComputeCommandEncoder()
            enc?.label = "tts"
        }
    }
    private func flush() {
        if let e = enc {
            e.endEncoding()
            cmdBuf?.commit()
        }
        for b in deferredDestroys { /* Metal releases on next cycle; no-op */ _ = b }
        deferredDestroys.removeAll()
        cmdBuf = nil
        enc = nil
    }

    // Full sync: flush + wait until GPU completes (only at read-back boundaries)
    private func flushAndWait() {
        if let e = enc {
            e.endEncoding()
            cmdBuf?.commit()
            cmdBuf?.waitUntilCompleted()
        }
        for b in deferredDestroys { _ = b }
        deferredDestroys.removeAll()
        cmdBuf = nil
        enc = nil
    }

    private func startStage(_ name: String) {
        status = .generating(name)
        if profile { stageStart = currentTimeMs() }
    }
    private func endStage(_ name: String) {
        flush()
        if profile {
            let elapsed = currentTimeMs() - stageStart
            timingsAccum.append((name, elapsed))
        }
    }
    private func currentTimeMs() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000.0
    }

    // Synchronous read-back: flush, then copy buffer to a temp staging buffer
    private func readBuffer(_ buf: MTLBuffer, size: Int) -> [Float] {
        flushAndWait()
        guard let blitCmd = commandQueue.makeCommandBuffer(),
              let blit = blitCmd.makeBlitCommandEncoder() else { return [] }
        let staging = device.makeBuffer(length: size * 4, options: [.storageModeShared])!
        blit.copy(from: buf, sourceOffset: 0, to: staging, destinationOffset: 0, size: size * 4)
        blit.endEncoding()
        blitCmd.commit()
        blitCmd.waitUntilCompleted()
        return staging.contents().withMemoryRebound(to: Float.self, capacity: size) { ptr in
            Array(UnsafeBufferPointer(start: ptr, count: size))
        }
    }

    // MARK: - Pipeline dispatch helpers (one per kernel)

    private func ensurePipeline(_ name: String) -> MTLComputePipelineState {
        guard let p = pipelines[name] else { fatalError("pipeline \(name) missing") }
        return p
    }

    private func dispatch(_ pipeline: MTLComputePipelineState, buffers: [MTLBuffer?], params: UnsafeRawPointer? = nil, paramLength: Int = 0, gridX: Int, gridY: Int = 1, gridZ: Int = 1, wgX: Int = 256, wgY: Int = 1, wgZ: Int = 1, label: String? = nil) {
        ensureEncoder()
        if let lbl = label {
            enc!.pushDebugGroup(lbl)
        }
        enc!.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() where b != nil { enc!.setBuffer(b!, offset: 0, index: i) }
        if let p = params {
            enc!.setBytes(p, length: paramLength, index: buffers.count)
        }
        enc!.dispatchThreadgroups(MTLSizeMake(gridX, gridY, gridZ),
                                  threadsPerThreadgroup: MTLSizeMake(wgX, wgY, wgZ))
        if label != nil { enc!.popDebugGroup() }
    }

    private func dispatch1d(_ pipeline: MTLComputePipelineState, buffers: [MTLBuffer?], params: UnsafeRawPointer? = nil, paramLength: Int = 0, total: Int, label: String? = nil) {
        let wg = 256
        let grid = (total + wg - 1) / wg
        dispatch(pipeline, buffers: buffers, params: params, paramLength: paramLength, gridX: max(grid, 1), wgX: wg, label: label)
    }

    // ── Embedding ─────────────────────────────
    private func dispatchEmbedding(emb: MTLBuffer, ids: MTLBuffer, out: MTLBuffer,
                                   seqLen: Int, embedDim: Int, vocab: Int) {
        let p = ensurePipeline("embedding_kernel")
        var params = EmbeddingParams(seq_len: UInt32(seqLen), embed_dim: UInt32(embedDim), vocab_size: UInt32(vocab))
        withUnsafePointer(to: &params) { ptr in
            dispatch1d(p, buffers: [emb, ids, out], params: ptr, paramLength: 12, total: seqLen * embedDim)
        }
    }

    private func dispatchLayerNorm(input: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, out: MTLBuffer,
                                   batchSize: Int, hiddenSize: Int, eps: Float) {
        let p = ensurePipeline("layer_norm_kernel")
        var params = LayerNormParams(batch_size: UInt32(batchSize), hidden_size: UInt32(hiddenSize), eps: eps)
        withUnsafePointer(to: &params) { ptr in
            dispatch(p, buffers: [input, gamma, beta, out], params: ptr, paramLength: 12,
                     gridX: (batchSize + 255) / 256, wgX: 256)
        }
    }

    private func dispatchMatmul(A: MTLBuffer, B: MTLBuffer, bias: MTLBuffer, out: MTLBuffer,
                                M: Int, K: Int, N: Int, useBias: Bool) {
        let p = ensurePipeline("matmul_kernel")
        var params = MatmulParams(M: UInt32(M), K: UInt32(K), N: UInt32(N), use_bias: useBias ? 1 : 0)
        let tgX = 16, tgY = 16
        withUnsafePointer(to: &params) { ptr in
            dispatch(p, buffers: [A, B, bias, out], params: ptr, paramLength: 16,
                     gridX: (M + tgX - 1) / tgX, gridY: (N + tgY - 1) / tgY, wgX: tgX, wgY: tgY)
        }
    }

    private func dispatchMatmulGelu(A: MTLBuffer, B: MTLBuffer, bias: MTLBuffer, out: MTLBuffer,
                                    M: Int, K: Int, N: Int) {
        let p = ensurePipeline("matmul_gelu_kernel")
        var params = MatmulParams(M: UInt32(M), K: UInt32(K), N: UInt32(N), use_bias: 1)
        let tgX = 16, tgY = 16
        withUnsafePointer(to: &params) { ptr in
            dispatch(p, buffers: [A, B, bias, out], params: ptr, paramLength: 16,
                     gridX: (M + tgX - 1) / tgX, gridY: (N + tgY - 1) / tgY, wgX: tgX, wgY: tgY)
        }
    }

    private func dispatchMHA(Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, out: MTLBuffer,
                             seqLen: Int, heads: Int, headDim: Int, scale: Float) {
        let p = ensurePipeline("mha_kernel")
        var params = MhaParams(seq_len: UInt32(seqLen), num_heads: UInt32(heads),
                               head_dim: UInt32(headDim), scale: scale)
        let wg = 64
        withUnsafePointer(to: &params) { ptr in
            dispatch(p, buffers: [Q, K, V, out], params: ptr, paramLength: 16,
                     gridX: (headDim + wg - 1) / wg, gridY: heads * seqLen, wgX: wg)
        }
    }

    private func dispatchConv1d(input: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, out: MTLBuffer,
                                inCh: Int, outCh: Int, kernel: Int, inLen: Int, outLen: Int,
                                padding: Int, stride: Int, dilation: Int, useBias: Bool) {
        let useTiled = inCh * kernel <= 4096
        if useTiled {
            let p = ensurePipeline("conv1d_tiled_kernel")
            var params = Conv1dParams(in_channels: UInt32(inCh), out_channels: UInt32(outCh),
                                      kernel_size: UInt32(kernel), input_length: UInt32(inLen),
                                      output_length: UInt32(outLen), padding: UInt32(padding),
                                      stride: UInt32(stride), dilation: UInt32(dilation),
                                      use_bias: useBias ? 1 : 0)
            let pb = device.makeBuffer(bytes: &params, length: 36, options: [])!
            dispatch(p, buffers: [input, weight, bias, out, pb],
                     gridX: (outLen + 255) / 256, gridY: outCh, wgX: 256)
            deferDestroy(pb)
        } else {
            let p = ensurePipeline("conv1d_kernel")
            var params = Conv1dParams(in_channels: UInt32(inCh), out_channels: UInt32(outCh),
                                      kernel_size: UInt32(kernel), input_length: UInt32(inLen),
                                      output_length: UInt32(outLen), padding: UInt32(padding),
                                      stride: UInt32(stride), dilation: UInt32(dilation),
                                      use_bias: useBias ? 1 : 0)
            let pb = device.makeBuffer(bytes: &params, length: 36, options: [])!
            dispatch1d(p, buffers: [input, weight, bias, out, pb], total: outCh * outLen)
            deferDestroy(pb)
        }
    }

    private func dispatchLSTM(input: MTLBuffer, W: MTLBuffer, R: MTLBuffer, bias: MTLBuffer,
                              out: MTLBuffer, seqLen: Int, inputSize: Int, hidden: Int, dirs: Int) {
        let p = ensurePipeline("lstm_kernel")
        var params = LstmParams(seq_len: UInt32(seqLen), input_size: UInt32(inputSize),
                                hidden_size: UInt32(hidden), num_directions: UInt32(dirs))
        let pb = device.makeBuffer(bytes: &params, length: 16, options: [])!
        // WGSL: workgroup x=256 covers hidden, workgroup y covers dirs
        dispatch(p, buffers: [input, W, R, bias, out, pb],
                 gridX: (hidden + 255) / 256, gridY: dirs, wgX: 256)
        deferDestroy(pb)
    }

    private func dispatchTranspose(input: MTLBuffer, out: MTLBuffer, rows: Int, cols: Int) {
        let p = ensurePipeline("transpose_kernel")
        var params = TransposeParams(rows: UInt32(rows), cols: UInt32(cols))
        let pb = device.makeBuffer(bytes: &params, length: 8, options: [])!
        dispatch1d(p, buffers: [input, out, pb], total: rows * cols)
        deferDestroy(pb)
    }

    private func dispatchLeakyRelu(input: MTLBuffer, out: MTLBuffer, size: Int, alpha: Float) {
        let p = ensurePipeline("leaky_relu_kernel")
        var params = SizeAlphaParams(size: UInt32(size), alpha: alpha)
        let pb = device.makeBuffer(bytes: &params, length: 8, options: [])!
        dispatch1d(p, buffers: [input, out, pb], total: size)
        deferDestroy(pb)
    }

    private func dispatchSigmoid(input: MTLBuffer, out: MTLBuffer, size: Int) {
        let p = ensurePipeline("sigmoid_kernel")
        var params = SizeParams(size: UInt32(size))
        let pb = device.makeBuffer(bytes: &params, length: 4, options: [])!
        dispatch1d(p, buffers: [input, out, pb], total: size)
        deferDestroy(pb)
    }

    private func dispatchAdd(a: MTLBuffer, b: MTLBuffer, out: MTLBuffer, size: Int) {
        let p = ensurePipeline("add_kernel")
        var params = SizeParams(size: UInt32(size))
        let pb = device.makeBuffer(bytes: &params, length: 4, options: [])!
        dispatch1d(p, buffers: [a, b, out, pb], total: size)
        deferDestroy(pb)
    }

    private func dispatchScale(input: MTLBuffer, out: MTLBuffer, size: Int, scale: Float) {
        let p = ensurePipeline("scale_kernel")
        var params = ScaleParams(size: UInt32(size), _pad: 0, scale: scale)
        let pb = device.makeBuffer(bytes: &params, length: 16, options: [])!
        dispatch1d(p, buffers: [input, out, pb], total: size)
        deferDestroy(pb)
    }

    private func dispatchAddScale(a: MTLBuffer, b: MTLBuffer, out: MTLBuffer, size: Int, scale: Float) {
        let p = ensurePipeline("add_scale_kernel")
        var params = ScaleParams(size: UInt32(size), _pad: 0, scale: scale)
        let pb = device.makeBuffer(bytes: &params, length: 16, options: [])!
        dispatch1d(p, buffers: [a, b, out, pb], total: size)
        deferDestroy(pb)
    }

    private func dispatchInstanceNorm(input: MTLBuffer, out: MTLBuffer, channels: Int, length: Int, eps: Float) {
        let p = ensurePipeline("instance_norm_kernel")
        var params = InstanceNormParams(channels: UInt32(channels), length: UInt32(length), eps: eps)
        let pb = device.makeBuffer(bytes: &params, length: 12, options: [])!
        dispatch(p, buffers: [input, out, pb], gridX: channels, wgX: 256)
        deferDestroy(pb)
    }

    private func dispatchAdaIN(normed: MTLBuffer, styleFc: MTLBuffer, out: MTLBuffer,
                               channels: Int, length: Int) {
        let p = ensurePipeline("adain_kernel")
        var params = AdainParams(channels: UInt32(channels), length: UInt32(length))
        let pb = device.makeBuffer(bytes: &params, length: 8, options: [])!
        dispatch1d(p, buffers: [normed, styleFc, out, pb], total: channels * length)
        deferDestroy(pb)
    }

    private func dispatchAdaINRowMajor(normed: MTLBuffer, styleFc: MTLBuffer, out: MTLBuffer,
                                       channels: Int, rows: Int) {
        let p = ensurePipeline("adain_row_major_kernel")
        let total = channels * rows
        var params = AdainRowMajorParams(channels: UInt32(channels), total: UInt32(total))
        let pb = device.makeBuffer(bytes: &params, length: 8, options: [])!
        dispatch1d(p, buffers: [normed, styleFc, out, pb], total: total)
        deferDestroy(pb)
    }

    private func dispatchSnake(input: MTLBuffer, alpha: MTLBuffer, out: MTLBuffer,
                               channels: Int, length: Int) {
        let p = ensurePipeline("snake_kernel")
        var params = AdainParams(channels: UInt32(channels), length: UInt32(length))
        let pb = device.makeBuffer(bytes: &params, length: 8, options: [])!
        dispatch1d(p, buffers: [input, alpha, out, pb], total: channels * length)
        deferDestroy(pb)
    }

    private func dispatchConvTranspose1d(input: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, out: MTLBuffer,
                                         inCh: Int, outCh: Int, kernel: Int, inLen: Int, outLen: Int,
                                         stride: Int, padding: Int, useBias: Bool) {
        let p = ensurePipeline("conv_transpose1d_kernel")
        var params = ConvTranspose1dParams(in_channels: UInt32(inCh), out_channels: UInt32(outCh),
                                            kernel_size: UInt32(kernel), input_length: UInt32(inLen),
                                            output_length: UInt32(outLen), stride: UInt32(stride),
                                            padding: UInt32(padding), use_bias: useBias ? 1 : 0)
        let pb = device.makeBuffer(bytes: &params, length: 32, options: [])!
        dispatch1d(p, buffers: [input, weight, bias, out, pb], total: outCh * outLen)
        deferDestroy(pb)
    }

    private func dispatchDepthwiseConvTranspose1d(input: MTLBuffer, weight: MTLBuffer, out: MTLBuffer,
                                                   channels: Int, kernel: Int, inLen: Int, outLen: Int,
                                                   stride: Int, padding: Int) {
        let p = ensurePipeline("depthwise_conv_transpose1d_kernel")
        var params = DepthwiseConvTParams(channels: UInt32(channels), kernel_size: UInt32(kernel),
                                          input_length: UInt32(inLen), output_length: UInt32(outLen),
                                          stride: UInt32(stride), padding: UInt32(padding))
        let pb = device.makeBuffer(bytes: &params, length: 24, options: [])!
        dispatch1d(p, buffers: [input, weight, out, pb], total: channels * outLen)
        deferDestroy(pb)
    }

    private func dispatchResize1d(input: MTLBuffer, out: MTLBuffer, channels: Int, inLen: Int, outLen: Int) {
        let p = ensurePipeline("resize1d_kernel")
        var params = Resize1dParams(channels: UInt32(channels), input_length: UInt32(inLen),
                                    output_length: UInt32(outLen))
        let pb = device.makeBuffer(bytes: &params, length: 12, options: [])!
        dispatch1d(p, buffers: [input, out, pb], total: channels * outLen)
        deferDestroy(pb)
    }

    private func dispatchConcatChannels(a: MTLBuffer, b: MTLBuffer, out: MTLBuffer,
                                        chA: Int, chB: Int, length: Int) {
        let p = ensurePipeline("concat_channels_kernel")
        var params = ConcatChannelsParams(channels_a: UInt32(chA), channels_b: UInt32(chB), length: UInt32(length))
        let pb = device.makeBuffer(bytes: &params, length: 12, options: [])!
        dispatch1d(p, buffers: [a, b, out, pb], total: (chA + chB) * length)
        deferDestroy(pb)
    }

    private func dispatchConcatBroadcast(a: MTLBuffer, b: MTLBuffer, out: MTLBuffer,
                                         rows: Int, colsA: Int, colsB: Int) {
        let p = ensurePipeline("concat_broadcast_kernel")
        var params = ConcatBroadcastParams(rows: UInt32(rows), cols_a: UInt32(colsA), cols_b: UInt32(colsB))
        let pb = device.makeBuffer(bytes: &params, length: 12, options: [])!
        dispatch1d(p, buffers: [a, b, out, pb], total: rows * (colsA + colsB))
        deferDestroy(pb)
    }

    private func dispatchReflectionPad1d(input: MTLBuffer, out: MTLBuffer,
                                         channels: Int, inLen: Int, padL: Int, padR: Int) {
        let p = ensurePipeline("reflection_pad1d_kernel")
        var params = ReflectionPadParams(channels: UInt32(channels), input_length: UInt32(inLen),
                                         pad_left: UInt32(padL), pad_right: UInt32(padR))
        let pb = device.makeBuffer(bytes: &params, length: 16, options: [])!
        let outLen = inLen + padL + padR
        dispatch1d(p, buffers: [input, out, pb], total: channels * outLen)
        deferDestroy(pb)
    }

    private func dispatchExpandRowMajor(input: MTLBuffer, cumsum: MTLBuffer, out: MTLBuffer,
                                        seqLen: Int, dim: Int, total: Int) {
        let p = ensurePipeline("expand_row_major_kernel")
        var params = ExpandParams(seq_len: UInt32(seqLen), dim: UInt32(dim), total_frames: UInt32(total))
        let pb = device.makeBuffer(bytes: &params, length: 12, options: [])!
        dispatch1d(p, buffers: [input, cumsum, out, pb], total: total * dim)
        deferDestroy(pb)
    }

    private func dispatchExpandChannelFirst(input: MTLBuffer, cumsum: MTLBuffer, out: MTLBuffer,
                                            seqLen: Int, dim: Int, total: Int) {
        let p = ensurePipeline("expand_channel_first_kernel")
        var params = ExpandParams(seq_len: UInt32(seqLen), dim: UInt32(dim), total_frames: UInt32(total))
        let pb = device.makeBuffer(bytes: &params, length: 12, options: [])!
        dispatch1d(p, buffers: [input, cumsum, out, pb], total: total * dim)
        deferDestroy(pb)
    }

    private func dispatchISTFT(convPost: MTLBuffer, wReal: MTLBuffer, wImag: MTLBuffer, out: MTLBuffer,
                               genLen: Int, waveLen: Int, bins: Int, kernel: Int, stride: Int) {
        let p = ensurePipeline("istft_kernel")
        var params = IstftParams(gen_length: UInt32(genLen), waveform_length: UInt32(waveLen),
                                 bins: UInt32(bins), kernel_size: UInt32(kernel), stride: UInt32(stride))
        let pb = device.makeBuffer(bytes: &params, length: 20, options: [])!
        dispatch1d(p, buffers: [convPost, wReal, wImag, out, pb], total: waveLen)
        deferDestroy(pb)
    }

    // No-op-eliminated unused kernels reserved for parity
    private func dispatchAlphaResidual(_: MTLBuffer, _: MTLBuffer, _: MTLBuffer, _: MTLBuffer,
                                       _: Int, _: Int) { }

    // MARK: - Block helpers (runAdaINResNetBlock / runDecoderBlock / runHiFiGANResBlock / buildDecodeInput / generateSourceExcitation)

    private struct AdaINResBlockResult { let output: MTLBuffer; let outChannels: Int; let outLength: Int }

    private func runAdaINResNetBlock(input: MTLBuffer, style: MTLBuffer,
                                     inCh: Int, length: Int, prefix: String,
                                     hasConv1x1: Bool, outCh: Int,
                                     pool: (weightName: String, channels: Int)?) throws -> AdaINResBlockResult {
        var curLen = length

        // norm1
        let norm1 = empty(inCh * curLen, label: "adain_norm1")
        dispatchInstanceNorm(input: input, out: norm1, channels: inCh, length: curLen, eps: 1e-5)
        let fc1W = try requireWeight("\(prefix).norm1.fc.weight_quantized")
        let fc1B = try requireWeight("\(prefix).norm1.fc.bias")
        let fc1Size = fc1B.size
        let style1 = empty(fc1Size, label: "norm1_style")
        dispatchMatmul(A: style, B: fc1W.buffer, bias: fc1B.buffer, out: style1,
                       M: 1, K: styleHalf, N: fc1Size, useBias: true)
        let adain1 = empty(inCh * curLen, label: "adain1_out")
        dispatchAdaIN(normed: norm1, styleFc: style1, out: adain1, channels: inCh, length: curLen)
        deferDestroy(norm1); deferDestroy(style1)

        let act1 = empty(inCh * curLen, label: "adain_act1")
        dispatchLeakyRelu(input: adain1, out: act1, size: inCh * curLen, alpha: 0.2)
        deferDestroy(adain1)

        var conv1Input = act1
        if let pool = pool {
            let pw = try requireWeight(pool.weightName)
            let pOutLen = curLen * 2
            let pOut = empty(pool.channels * pOutLen, label: "pool_out")
            dispatchDepthwiseConvTranspose1d(input: act1, weight: pw.buffer, out: pOut,
                                             channels: pool.channels, kernel: 3, inLen: curLen, outLen: pOutLen,
                                             stride: 2, padding: 1)
            deferDestroy(act1)
            conv1Input = pOut
            curLen = pOutLen
        }

        // conv1
        let c1w = try requireWeight("\(prefix).conv1.weight_quantized")
        let c1b = try requireWeight("\(prefix).conv1.bias")
        let conv1 = empty(outCh * curLen, label: "adain_conv1")
        dispatchConv1d(input: conv1Input, weight: c1w.buffer, bias: c1b.buffer, out: conv1,
                       inCh: inCh, outCh: outCh, kernel: 3, inLen: curLen, outLen: curLen,
                       padding: 1, stride: 1, dilation: 1, useBias: true)
        deferDestroy(conv1Input)

        // norm2
        let norm2 = empty(outCh * curLen, label: "adain_norm2")
        dispatchInstanceNorm(input: conv1, out: norm2, channels: outCh, length: curLen, eps: 1e-5)
        let fc2W = try requireWeight("\(prefix).norm2.fc.weight_quantized")
        let fc2B = try requireWeight("\(prefix).norm2.fc.bias")
        let fc2Size = fc2B.size
        let style2 = empty(fc2Size, label: "norm2_style")
        dispatchMatmul(A: style, B: fc2W.buffer, bias: fc2B.buffer, out: style2,
                       M: 1, K: styleHalf, N: fc2Size, useBias: true)
        let adain2 = empty(outCh * curLen, label: "adain2_out")
        dispatchAdaIN(normed: norm2, styleFc: style2, out: adain2, channels: outCh, length: curLen)
        deferDestroy(conv1); deferDestroy(norm2); deferDestroy(style2)

        let act2 = empty(outCh * curLen, label: "adain_act2")
        dispatchLeakyRelu(input: adain2, out: act2, size: outCh * curLen, alpha: 0.2)
        deferDestroy(adain2)

        // conv2
        let c2w = try requireWeight("\(prefix).conv2.weight_quantized")
        let c2b = try requireWeight("\(prefix).conv2.bias")
        let conv2 = empty(outCh * curLen, label: "adain_conv2")
        dispatchConv1d(input: act2, weight: c2w.buffer, bias: c2b.buffer, out: conv2,
                       inCh: outCh, outCh: outCh, kernel: 3, inLen: curLen, outLen: curLen,
                       padding: 1, stride: 1, dilation: 1, useBias: true)
        deferDestroy(act2)

        // residual
        let residual: MTLBuffer
        if let pool = pool {
            let resized = empty(inCh * curLen, label: "adain_resized")
            dispatchResize1d(input: input, out: resized, channels: inCh, inLen: length, outLen: curLen)
            let c1x1w = try requireWeight("\(prefix).conv1x1.weight_quantized")
            residual = empty(outCh * curLen, label: "adain_res_proj")
            dispatchConv1d(input: resized, weight: c1x1w.buffer, bias: c1x1w.buffer, out: residual,
                           inCh: inCh, outCh: outCh, kernel: 1, inLen: curLen, outLen: curLen,
                           padding: 0, stride: 1, dilation: 1, useBias: false)
            deferDestroy(resized)
        } else if hasConv1x1 && inCh != outCh {
            let c1x1w = try requireWeight("\(prefix).conv1x1.weight_quantized")
            residual = empty(outCh * curLen, label: "adain_res_proj")
            dispatchConv1d(input: input, weight: c1x1w.buffer, bias: c1x1w.buffer, out: residual,
                           inCh: inCh, outCh: outCh, kernel: 1, inLen: curLen, outLen: curLen,
                           padding: 0, stride: 1, dilation: 1, useBias: false)
        } else {
            residual = input
        }

        let rawSum = empty(outCh * curLen, label: "adain_raw_sum")
        dispatchAdd(a: conv2, b: residual, out: rawSum, size: outCh * curLen)
        deferDestroy(conv2)
        let output = empty(outCh * curLen, label: "adain_block_out")
        dispatchScale(input: rawSum, out: output, size: outCh * curLen, scale: 1.0 / Float(Double.pi / 4.0).squareRoot())
        deferDestroy(rawSum)
        if residual !== input { deferDestroy(residual) }
        return AdaINResBlockResult(output: output, outChannels: outCh, outLength: curLen)
    }

    private func runDecoderBlock(input: MTLBuffer, style: MTLBuffer,
                                 inCh: Int, outCh: Int, length: Int, prefix: String,
                                 hasConv1x1: Bool, pool: (weightName: String, channels: Int)?) throws -> MTLBuffer {
        var curLen = length
        let norm1 = empty(inCh * curLen, label: "dec_norm1")
        dispatchInstanceNorm(input: input, out: norm1, channels: inCh, length: curLen, eps: 1e-5)
        let fc1W = try requireWeight("\(prefix).norm1.fc.weight_quantized")
        let fc1B = try requireWeight("\(prefix).norm1.fc.bias")
        let fc1Size = fc1B.size
        let style1 = empty(fc1Size, label: "dec_norm1_style")
        dispatchMatmul(A: style, B: fc1W.buffer, bias: fc1B.buffer, out: style1,
                       M: 1, K: styleHalf, N: fc1Size, useBias: true)
        let adain1 = empty(inCh * curLen, label: "dec_adain1")
        dispatchAdaIN(normed: norm1, styleFc: style1, out: adain1, channels: inCh, length: curLen)
        deferDestroy(norm1); deferDestroy(style1)
        let act1 = empty(inCh * curLen, label: "dec_act1")
        dispatchLeakyRelu(input: adain1, out: act1, size: inCh * curLen, alpha: 0.2)
        deferDestroy(adain1)

        var conv1Input = act1
        if let pool = pool {
            let pw = try requireWeight(pool.weightName)
            let pOutLen = curLen * 2
            let pOut = empty(pool.channels * pOutLen, label: "dec_pool_out")
            dispatchDepthwiseConvTranspose1d(input: act1, weight: pw.buffer, out: pOut,
                                             channels: pool.channels, kernel: 3, inLen: curLen, outLen: pOutLen,
                                             stride: 2, padding: 1)
            deferDestroy(act1)
            conv1Input = pOut
            curLen = pOutLen
        }
        let c1w = try requireWeight("\(prefix).conv1.weight_quantized")
        let c1b = try requireWeight("\(prefix).conv1.bias")
        let conv1 = empty(outCh * curLen, label: "dec_conv1")
        dispatchConv1d(input: conv1Input, weight: c1w.buffer, bias: c1b.buffer, out: conv1,
                       inCh: inCh, outCh: outCh, kernel: 3, inLen: curLen, outLen: curLen,
                       padding: 1, stride: 1, dilation: 1, useBias: true)
        deferDestroy(conv1Input)

        let norm2 = empty(outCh * curLen, label: "dec_norm2")
        dispatchInstanceNorm(input: conv1, out: norm2, channels: outCh, length: curLen, eps: 1e-5)
        let fc2W = try requireWeight("\(prefix).norm2.fc.weight_quantized")
        let fc2B = try requireWeight("\(prefix).norm2.fc.bias")
        let fc2Size = fc2B.size
        let style2 = empty(fc2Size, label: "dec_norm2_style")
        dispatchMatmul(A: style, B: fc2W.buffer, bias: fc2B.buffer, out: style2,
                       M: 1, K: styleHalf, N: fc2Size, useBias: true)
        let adain2 = empty(outCh * curLen, label: "dec_adain2")
        dispatchAdaIN(normed: norm2, styleFc: style2, out: adain2, channels: outCh, length: curLen)
        deferDestroy(conv1); deferDestroy(norm2); deferDestroy(style2)
        let act2 = empty(outCh * curLen, label: "dec_act2")
        dispatchLeakyRelu(input: adain2, out: act2, size: outCh * curLen, alpha: 0.2)
        deferDestroy(adain2)

        let c2w = try requireWeight("\(prefix).conv2.weight_quantized")
        let c2b = try requireWeight("\(prefix).conv2.bias")
        let conv2 = empty(outCh * curLen, label: "dec_conv2")
        dispatchConv1d(input: act2, weight: c2w.buffer, bias: c2b.buffer, out: conv2,
                       inCh: outCh, outCh: outCh, kernel: 3, inLen: curLen, outLen: curLen,
                       padding: 1, stride: 1, dilation: 1, useBias: true)
        deferDestroy(act2)

        let sqrt2Inv: Float = 1.0 / Float(Double.pi / 4.0).squareRoot()
        let rawSum: MTLBuffer
        if let _ = pool {
            let resized = empty(inCh * curLen, label: "dec_resized")
            dispatchResize1d(input: input, out: resized, channels: inCh, inLen: length, outLen: curLen)
            let c1x1w = try requireWeight("\(prefix).conv1x1.weight_quantized")
            let residual = empty(outCh * curLen, label: "dec_res_proj")
            dispatchConv1d(input: resized, weight: c1x1w.buffer, bias: c1x1w.buffer, out: residual,
                           inCh: inCh, outCh: outCh, kernel: 1, inLen: curLen, outLen: curLen,
                           padding: 0, stride: 1, dilation: 1, useBias: false)
            rawSum = empty(outCh * curLen, label: "dec_raw_sum")
            dispatchAdd(a: conv2, b: residual, out: rawSum, size: outCh * curLen)
            deferDestroy(conv2); deferDestroy(resized); deferDestroy(residual)
        } else if hasConv1x1 {
            let c1x1w = try requireWeight("\(prefix).conv1x1.weight_quantized")
            let residual = empty(outCh * curLen, label: "dec_res_proj")
            dispatchConv1d(input: input, weight: c1x1w.buffer, bias: c1x1w.buffer, out: residual,
                           inCh: inCh, outCh: outCh, kernel: 1, inLen: curLen, outLen: curLen,
                           padding: 0, stride: 1, dilation: 1, useBias: false)
            rawSum = empty(outCh * curLen, label: "dec_raw_sum")
            dispatchAdd(a: conv2, b: residual, out: rawSum, size: outCh * curLen)
            deferDestroy(conv2); deferDestroy(residual)
        } else {
            rawSum = conv2
        }
        let output = empty(outCh * curLen, label: "dec_block_out")
        dispatchScale(input: rawSum, out: output, size: outCh * curLen, scale: sqrt2Inv)
        deferDestroy(rawSum)
        return output
    }

    private func buildDecodeInput(features: MTLBuffer, f0Conv: MTLBuffer, nConv: MTLBuffer,
                                  asrRes: MTLBuffer?, featureCh: Int, length: Int, asrCh: Int) throws -> MTLBuffer {
        let asr = asrRes ?? empty(asrCh * length, label: "asr_zero")
        let c1 = empty((featureCh + asrCh) * length, label: "dec_concat1")
        dispatchConcatChannels(a: features, b: asr, out: c1, chA: featureCh, chB: asrCh, length: length)
        let c2 = empty((featureCh + asrCh + 1) * length, label: "dec_concat2")
        dispatchConcatChannels(a: c1, b: f0Conv, out: c2, chA: featureCh + asrCh, chB: 1, length: length)
        let out = empty((featureCh + asrCh + 2) * length, label: "dec_concat3")
        dispatchConcatChannels(a: c2, b: nConv, out: out, chA: featureCh + asrCh + 1, chB: 1, length: length)
        deferDestroy(c1); deferDestroy(c2)
        if asrRes == nil { deferDestroy(asr) }
        return out
    }

    private func runHiFiGANResBlock(input: MTLBuffer, style: MTLBuffer,
                                    channels: Int, length: Int, prefix: String) throws -> MTLBuffer {
        var current = input
        let dilations = [1, 3, 5]
        for i in 0..<3 {
            let norm1 = empty(channels * length, label: "hifi_norm1_\(i)")
            dispatchInstanceNorm(input: current, out: norm1, channels: channels, length: length, eps: 1e-5)
            let a1fcW = try requireWeight("\(prefix).adain1.\(i).fc.weight_quantized")
            let a1fcB = try requireWeight("\(prefix).adain1.\(i).fc.bias")
            let a1Size = a1fcB.size
            let a1Style = empty(a1Size, label: "hifi_a1_style_\(i)")
            dispatchMatmul(A: style, B: a1fcW.buffer, bias: a1fcB.buffer, out: a1Style,
                           M: 1, K: styleHalf, N: a1Size, useBias: true)
            let a1Out = empty(channels * length, label: "hifi_a1_\(i)")
            dispatchAdaIN(normed: norm1, styleFc: a1Style, out: a1Out, channels: channels, length: length)
            deferDestroy(norm1); deferDestroy(a1Style)

            let alpha1 = try requireWeight("\(prefix).alpha1.\(i)")
            let sn1 = empty(channels * length, label: "hifi_sn1_\(i)")
            dispatchSnake(input: a1Out, alpha: alpha1.buffer, out: sn1, channels: channels, length: length)
            deferDestroy(a1Out)

            let c1W = try requireWeight("\(prefix).convs1.\(i).weight_quantized")
            let c1B = try requireWeight("\(prefix).convs1.\(i).bias")
            let k1 = c1W.shape[2]
            let d = dilations[i]
            let pad1 = Int((Float(k1) * Float(d) - Float(d)) / 2.0)
            let c1Out = empty(channels * length, label: "hifi_c1_\(i)")
            dispatchConv1d(input: sn1, weight: c1W.buffer, bias: c1B.buffer, out: c1Out,
                           inCh: channels, outCh: channels, kernel: k1, inLen: length, outLen: length,
                           padding: pad1, stride: 1, dilation: d, useBias: true)
            deferDestroy(sn1)

            let norm2 = empty(channels * length, label: "hifi_norm2_\(i)")
            dispatchInstanceNorm(input: c1Out, out: norm2, channels: channels, length: length, eps: 1e-5)
            let a2fcW = try requireWeight("\(prefix).adain2.\(i).fc.weight_quantized")
            let a2fcB = try requireWeight("\(prefix).adain2.\(i).fc.bias")
            let a2Size = a2fcB.size
            let a2Style = empty(a2Size, label: "hifi_a2_style_\(i)")
            dispatchMatmul(A: style, B: a2fcW.buffer, bias: a2fcB.buffer, out: a2Style,
                           M: 1, K: styleHalf, N: a2Size, useBias: true)
            let a2Out = empty(channels * length, label: "hifi_a2_\(i)")
            dispatchAdaIN(normed: norm2, styleFc: a2Style, out: a2Out, channels: channels, length: length)
            deferDestroy(c1Out); deferDestroy(norm2); deferDestroy(a2Style)

            let alpha2 = try requireWeight("\(prefix).alpha2.\(i)")
            let sn2 = empty(channels * length, label: "hifi_sn2_\(i)")
            dispatchSnake(input: a2Out, alpha: alpha2.buffer, out: sn2, channels: channels, length: length)
            deferDestroy(a2Out)

            let c2W = try requireWeight("\(prefix).convs2.\(i).weight_quantized")
            let c2B = try requireWeight("\(prefix).convs2.\(i).bias")
            let k2 = c2W.shape[2]
            let pad2 = Int((Float(k2) - 1) / 2.0)
            let c2Out = empty(channels * length, label: "hifi_c2_\(i)")
            dispatchConv1d(input: sn2, weight: c2W.buffer, bias: c2B.buffer, out: c2Out,
                           inCh: channels, outCh: channels, kernel: k2, inLen: length, outLen: length,
                           padding: pad2, stride: 1, dilation: 1, useBias: true)
            deferDestroy(sn2)

            let resOut = empty(channels * length, label: "hifi_res_\(i)")
            dispatchAdd(a: c2Out, b: current, out: resOut, size: channels * length)
            deferDestroy(c2Out)
            if current !== input { deferDestroy(current) }
            current = resOut
        }
        return current
    }

    private func generateSourceExcitation(f0ProjBuf: MTLBuffer, f0Length: Int, stftLen: Int) throws -> MTLBuffer {
        let sampleRate = 24000.0
        let nHarm = 9
        let voicedScale: Float = 0.1
        let unvoicedScale: Float = 0.003

        let f0Data = readBuffer(f0ProjBuf, size: f0Length)
        let waveLen = (stftLen - 1) * 5

        // f0 upsample (nearest-neighbour).
        var f0Up = [Float](repeating: 0, count: waveLen)
        let ratio = Double(waveLen) / Double(f0Length)
        f0Up.withUnsafeMutableBufferPointer { up in
            f0Data.withUnsafeBufferPointer { src in
                for i in 0..<waveLen {
                    up[i] = src[min(Int(Double(i) / ratio), f0Length - 1)]
                }
            }
        }

        // Harmonic source: [waveLen x nHarm] row-major. The voiced branch is a phase
        // recurrence (inherently sequential), so this loop stays scalar — but with
        // unsafe pointers (no bounds checks / ARC) and a cheap xorshift PRNG in place of
        // the two per-sample `Double.random` calls that dominated the old profile.
        var harm = [Float](repeating: 0, count: waveLen * nHarm)
        var rng: UInt64 = 0x2545F4914F6CDD1D
        harm.withUnsafeMutableBufferPointer { h in
            f0Up.withUnsafeBufferPointer { f in
                for k in 0..<nHarm {
                    let hIdx = Double(k + 1)
                    var phase: Double = 0
                    for t in 0..<waveLen {
                        let f0 = Double(f[t])
                        if f0 > 10 {
                            phase += f0 * hIdx / sampleRate
                            phase -= phase.rounded(.down)
                            h[t * nHarm + k] = Float(sin(2 * .pi * phase)) * voicedScale
                        } else {
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
                            let u1 = Double(rng >> 11) * (1.0 / 9_007_199_254_740_992.0)
                            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
                            let u2 = Double(rng >> 11) * (1.0 / 9_007_199_254_740_992.0)
                            let z = sqrt(-2.0 * log(u1 + 1e-10)) * cos(2 * .pi * u2)
                            h[t * nHarm + k] = Float(z) * unvoicedScale
                            phase = 0
                        }
                    }
                }
            }
        }
        if sinGenWeights == nil {
            let lw = try requireWeight("onnx::MatMul_6388")
            let lb = try requireWeight("kmodel.decoder.generator.m_source.l_linear.bias")
            let fr = try requireWeight("kmodel.decoder.generator.stft.weight_forward_real")
            let fi = try requireWeight("kmodel.decoder.generator.stft.weight_forward_imag")
            sinGenWeights = SinGenWeights(
                linearWeight: readBuffer(lw.buffer, size: lw.size),
                linearBias:   readBuffer(lb.buffer, size: lb.size),
                fwdReal:      readBuffer(fr.buffer, size: fr.size),
                fwdImag:      readBuffer(fi.buffer, size: fi.size))
        }
        let sgw = sinGenWeights!

        // wave[t] = tanh( bias + Σ_k harm[t,k] * linearWeight[k] )
        // = a matrix-vector product harm(waveLen × nHarm) · linearWeight(nHarm), then tanh.
        var wave = [Float](repeating: 0, count: waveLen)
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    Int32(waveLen), Int32(nHarm), 1.0,
                    harm, Int32(nHarm), sgw.linearWeight, 1,
                    0.0, &wave, 1)
        var bias = sgw.linearBias.first ?? 0
        var wn = Int32(waveLen)
        wave.withUnsafeMutableBufferPointer { wp in
            let b = wp.baseAddress!
            vDSP_vsadd(b, 1, &bias, b, 1, vDSP_Length(waveLen))
            vvtanhf(b, b, &wn)
        }

        // Edge-pad by 10 samples on each side (replicate).
        var padded = [Float](repeating: 0, count: waveLen + 20)
        wave.withUnsafeBufferPointer { w in
            padded.withUnsafeMutableBufferPointer { p in
                let first = w[0], last = w[waveLen - 1]
                for i in 0..<10 { p[i] = first }
                memcpy(p.baseAddress! + 10, w.baseAddress!, waveLen * 4)
                for i in 0..<10 { p[waveLen + 10 + i] = last }
            }
        }

        // Forward STFT as two GEMMs. realOut(11 × stftLen) = fwdReal(11 × kernel) · W(kernel × stftLen),
        // where W[k, t] = padded[t*stride + k] is the sliding-window matrix.
        let stride = 5, kernel = 20
        var win = [Float](repeating: 0, count: kernel * stftLen)
        padded.withUnsafeBufferPointer { p in
            win.withUnsafeMutableBufferPointer { W in
                for k in 0..<kernel {
                    let row = k * stftLen
                    for t in 0..<stftLen { W[row + t] = p[t * stride + k] }
                }
            }
        }
        var realOut = [Float](repeating: 0, count: 11 * stftLen)
        var imagOut = [Float](repeating: 0, count: 11 * stftLen)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    11, Int32(stftLen), Int32(kernel), 1.0,
                    sgw.fwdReal, Int32(kernel), win, Int32(stftLen),
                    0.0, &realOut, Int32(stftLen))
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    11, Int32(stftLen), Int32(kernel), 1.0,
                    sgw.fwdImag, Int32(kernel), win, Int32(stftLen),
                    0.0, &imagOut, Int32(stftLen))

        // Magnitude / phase over all 11×stftLen bins at once.
        let half = 11 * stftLen
        var noise = [Float](repeating: 0, count: 22 * stftLen)
        var mag = [Float](repeating: 0, count: half)
        var sq = [Float](repeating: 0, count: half)
        var eps: Float = 1e-14
        var hn = Int32(half)
        vDSP_vsq(realOut, 1, &mag, 1, vDSP_Length(half))   // mag = real²
        vDSP_vsq(imagOut, 1, &sq, 1, vDSP_Length(half))    // sq  = imag²
        mag.withUnsafeMutableBufferPointer { mp in
            let m = mp.baseAddress!
            sq.withUnsafeBufferPointer { s in
                vDSP_vadd(m, 1, s.baseAddress!, 1, m, 1, vDSP_Length(half)) // mag = real²+imag²
            }
            vDSP_vsadd(m, 1, &eps, m, 1, vDSP_Length(half))                 // + eps
            vvsqrtf(m, m, &hn)                                             // sqrt(...)
        }
        // Magnitude -> bins [0, 11), phase = atan2(imag, real) -> bins [11, 22).
        noise.withUnsafeMutableBufferPointer { n in
            let base = n.baseAddress!
            mag.withUnsafeBufferPointer { m in
                memcpy(base, m.baseAddress!, half * 4)
            }
            vvatan2f(base + half, imagOut, realOut, &hn)
        }
        return makeBuffer(noise, label: "noise_source")
    }

    // MARK: - runBertEncoder / runTextEncoder

    private func runBertEncoder(inputIdsBuf: MTLBuffer, embeddingSum: MTLBuffer, seqLen: Int) throws -> MTLBuffer {
        let embedDim = bertEmbedDim
        let hidden = bertHiddenSize
        let scale: Float = 1.0 / Float(Double(bertHeadDim).squareRoot())

        let embLnW = try requireWeight("kmodel.bert.embeddings.LayerNorm.weight")
        let embLnB = try requireWeight("kmodel.bert.embeddings.LayerNorm.bias")
        let normedEmb = empty(seqLen * embedDim, label: "bert_emb_ln")
        dispatchLayerNorm(input: embeddingSum, gamma: embLnW.buffer, beta: embLnB.buffer, out: normedEmb,
                          batchSize: seqLen, hiddenSize: embedDim, eps: 1e-12)

        let projW = try requireWeight("onnx::MatMul_5883_quantized")
        let projB = try requireWeight("kmodel.bert.encoder.embedding_hidden_mapping_in.bias")
        var hiddenBuf = empty(seqLen * hidden, label: "bert_proj")
        dispatchMatmul(A: normedEmb, B: projW.buffer, bias: projB.buffer, out: hiddenBuf,
                       M: seqLen, K: embedDim, N: hidden, useBias: true)
        deferDestroy(normedEmb)

        let prefix = "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0"
        let qW = try requireWeight("onnx::MatMul_5884_quantized")
        let kW = try requireWeight("onnx::MatMul_5887_quantized")
        let vW = try requireWeight("onnx::MatMul_5890_quantized")
        let qB = try requireWeight("\(prefix).attention.query.bias")
        let kB = try requireWeight("\(prefix).attention.key.bias")
        let vB = try requireWeight("\(prefix).attention.value.bias")
        let aOutW = try requireWeight("onnx::MatMul_5894_quantized")
        let aOutB = try requireWeight("\(prefix).attention.dense.bias")
        let attnLnW = try requireWeight("\(prefix).attention.LayerNorm.weight")
        let attnLnB = try requireWeight("\(prefix).attention.LayerNorm.bias")
        let ffnUpW = try requireWeight("onnx::MatMul_5895_quantized")
        let ffnUpB = try requireWeight("\(prefix).ffn.bias")
        let ffnDownW = try requireWeight("onnx::MatMul_5896_quantized")
        let ffnDownB = try requireWeight("\(prefix).ffn_output.bias")
        let fullLnW = try requireWeight("\(prefix).full_layer_layer_norm.weight")
        let fullLnB = try requireWeight("\(prefix).full_layer_layer_norm.bias")

        for _ in 0..<bertNumLayers {
            let Q = empty(seqLen * hidden, label: "Q"); deferDestroy(Q)
            let K = empty(seqLen * hidden, label: "K"); deferDestroy(K)
            let V = empty(seqLen * hidden, label: "V"); deferDestroy(V)
            dispatchMatmul(A: hiddenBuf, B: qW.buffer, bias: qB.buffer, out: Q,
                           M: seqLen, K: hidden, N: hidden, useBias: true)
            dispatchMatmul(A: hiddenBuf, B: kW.buffer, bias: kB.buffer, out: K,
                           M: seqLen, K: hidden, N: hidden, useBias: true)
            dispatchMatmul(A: hiddenBuf, B: vW.buffer, bias: vB.buffer, out: V,
                           M: seqLen, K: hidden, N: hidden, useBias: true)
            let attn = empty(seqLen * hidden, label: "attn"); deferDestroy(attn)
            dispatchMHA(Q: Q, K: K, V: V, out: attn, seqLen: seqLen, heads: bertNumHeads,
                        headDim: bertHeadDim, scale: scale)
            let attnProj = empty(seqLen * hidden, label: "attnProj"); deferDestroy(attnProj)
            dispatchMatmul(A: attn, B: aOutW.buffer, bias: aOutB.buffer, out: attnProj,
                           M: seqLen, K: hidden, N: hidden, useBias: true)
            let attnRes = empty(seqLen * hidden, label: "attnRes"); deferDestroy(attnRes)
            dispatchAdd(a: hiddenBuf, b: attnProj, out: attnRes, size: seqLen * hidden)
            let attnNormed = empty(seqLen * hidden, label: "attnNormed"); deferDestroy(attnNormed)
            dispatchLayerNorm(input: attnRes, gamma: attnLnW.buffer, beta: attnLnB.buffer, out: attnNormed,
                              batchSize: seqLen, hiddenSize: hidden, eps: 1e-12)
            let ffnUp = empty(seqLen * bertFfnDim, label: "ffnUp"); deferDestroy(ffnUp)
            dispatchMatmulGelu(A: attnNormed, B: ffnUpW.buffer, bias: ffnUpB.buffer, out: ffnUp,
                               M: seqLen, K: hidden, N: bertFfnDim)
            let ffnDown = empty(seqLen * hidden, label: "ffnDown"); deferDestroy(ffnDown)
            dispatchMatmul(A: ffnUp, B: ffnDownW.buffer, bias: ffnDownB.buffer, out: ffnDown,
                           M: seqLen, K: bertFfnDim, N: hidden, useBias: true)
            let ffnRes = empty(seqLen * hidden, label: "ffnRes"); deferDestroy(ffnRes)
            dispatchAdd(a: attnNormed, b: ffnDown, out: ffnRes, size: seqLen * hidden)
            let newHidden = empty(seqLen * hidden, label: "hiddenN")
            dispatchLayerNorm(input: ffnRes, gamma: fullLnW.buffer, beta: fullLnB.buffer, out: newHidden,
                              batchSize: seqLen, hiddenSize: hidden, eps: 1e-12)
            deferDestroy(hiddenBuf)
            hiddenBuf = newHidden
        }
        return hiddenBuf
    }

    private func runTextEncoder(inputIdsBuf: MTLBuffer, seqLen: Int) throws -> MTLBuffer {
        let ch = textEncChannels
        let k = 5, pad = 2
        let embW = try requireWeight("kmodel.text_encoder.embedding.weight")
        let embOut = empty(seqLen * ch, label: "te_emb")
        dispatchEmbedding(emb: embW.buffer, ids: inputIdsBuf, out: embOut, seqLen: seqLen, embedDim: ch, vocab: 178)
        let transposed = empty(seqLen * ch, label: "te_transposed")
        dispatchTranspose(input: embOut, out: transposed, rows: seqLen, cols: ch)
        deferDestroy(embOut)

        var cur = transposed
        for i in 0..<numTextEncCnnBlocks {
            let cw = try requireWeight("kmodel.text_encoder.cnn.\(i).0.weight_quantized")
            let cb = try requireWeight("kmodel.text_encoder.cnn.\(i).0.bias")
            let lg = try requireWeight("kmodel.text_encoder.cnn.\(i).1.gamma")
            let lb = try requireWeight("kmodel.text_encoder.cnn.\(i).1.beta")
            let convOut = empty(ch * seqLen, label: "te_conv\(i)")
            dispatchConv1d(input: cur, weight: cw.buffer, bias: cb.buffer, out: convOut,
                           inCh: ch, outCh: ch, kernel: k, inLen: seqLen, outLen: seqLen,
                           padding: pad, stride: 1, dilation: 1, useBias: true)
            deferDestroy(cur)
            let preNorm = empty(seqLen * ch, label: "te_pre\(i)")
            dispatchTranspose(input: convOut, out: preNorm, rows: ch, cols: seqLen)
            deferDestroy(convOut)
            let normed = empty(seqLen * ch, label: "te_normed\(i)")
            dispatchLayerNorm(input: preNorm, gamma: lg.buffer, beta: lb.buffer, out: normed,
                              batchSize: seqLen, hiddenSize: ch, eps: 1e-5)
            deferDestroy(preNorm)
            let postNorm = empty(ch * seqLen, label: "te_post\(i)")
            dispatchTranspose(input: normed, out: postNorm, rows: seqLen, cols: ch)
            deferDestroy(normed)
            let activated = empty(ch * seqLen, label: "te_act\(i)")
            dispatchLeakyRelu(input: postNorm, out: activated, size: ch * seqLen, alpha: 0.2)
            deferDestroy(postNorm)
            cur = activated
        }
        let lstmIn = empty(seqLen * ch, label: "te_lstm_in")
        dispatchTranspose(input: cur, out: lstmIn, rows: ch, cols: seqLen)
        deferDestroy(cur)
        let lstmW = try requireWeight("onnx::LSTM_5874_quantized")
        let lstmR = try requireWeight("onnx::LSTM_5875_quantized")
        let lstmB = try requireWeight("onnx::LSTM_5873")
        let out = empty(seqLen * 2 * lstmHidden, label: "te_lstm_out")
        dispatchLSTM(input: lstmIn, W: lstmW.buffer, R: lstmR.buffer, bias: lstmB.buffer,
                     out: out, seqLen: seqLen, inputSize: ch, hidden: lstmHidden, dirs: 2)
        deferDestroy(lstmIn)
        return out
    }

    // MARK: - High-level generate()

    func generate(inputIds: [Int], voice: String, speed: Float, textLength: Int) throws -> [Float] {
        timingsAccum.removeAll()
        let voiceKey = TtsConfig.voiceAliases[voice] ?? voice
        guard let v = voices[voiceKey] else {
            status = .error("Voice not found: \(voice)"); throw NSError(domain: "TtsEngine", code: 30, userInfo: [NSLocalizedDescriptionKey: "voice"])
        }
        let refId = min(textLength > 0 ? textLength : inputIds.count, 399)
        let styleVec = Array(v[(refId * styleDim)..<((refId + 1) * styleDim)])
        let styleBuf = makeBuffer(styleVec, label: "style")
        let stylePred = makeBuffer(Array(styleVec[styleHalf..<styleDim]), label: "stylePred")
        let styleDec = makeBuffer(Array(styleVec[0..<styleHalf]), label: "styleDec")
        let inputIdsBuf = makeBuffer(inputIds.map(Int32.init), label: "ids")
        let seqLen = inputIds.count

        // 1. BERT embedding
        startStage("1/8 BERT embedding")
        let wE = try requireWeight("kmodel.bert.embeddings.word_embeddings.weight")
        let pE = try requireWeight("kmodel.bert.embeddings.position_embeddings.weight")
        let tE = try requireWeight("kmodel.bert.embeddings.token_type_embeddings.weight")
        let eDim = bertEmbedDim
        let wordEmb = empty(seqLen * eDim, label: "wordEmb")
        dispatchEmbedding(emb: wE.buffer, ids: inputIdsBuf, out: wordEmb, seqLen: seqLen, embedDim: eDim, vocab: 178)
        let tokenTypeIds = makeBuffer([Int32](repeating: 0, count: seqLen), label: "ttype_ids")
        let tokEmb = empty(seqLen * eDim, label: "ttypeEmb")
        dispatchEmbedding(emb: tE.buffer, ids: tokenTypeIds, out: tokEmb, seqLen: seqLen, embedDim: eDim, vocab: 2)
        let wpt = empty(seqLen * eDim, label: "wtt")
        dispatchAdd(a: wordEmb, b: tokEmb, out: wpt, size: seqLen * eDim)
        deferDestroy(wordEmb); deferDestroy(tokEmb)
        var posIds = [Int32](repeating: 0, count: seqLen)
        for i in 0..<seqLen { posIds[i] = Int32(min(i, 511)) }
        let posIdsBuf = makeBuffer(posIds, label: "pos_ids")
        let posEmb = empty(seqLen * eDim, label: "posEmb")
        dispatchEmbedding(emb: pE.buffer, ids: posIdsBuf, out: posEmb, seqLen: seqLen, embedDim: eDim, vocab: 512)
        let bertEmb = empty(seqLen * eDim, label: "bert_emb_sum")
        dispatchAdd(a: wpt, b: posEmb, out: bertEmb, size: seqLen * eDim)
        deferDestroy(wpt); deferDestroy(posEmb); deferDestroy(tokenTypeIds)
        endStage("BERT embedding")

        // 2. ALBERT encoder
        startStage("2/8 ALBERT encoder")
        let bertOut = try runBertEncoder(inputIdsBuf: inputIdsBuf, embeddingSum: bertEmb, seqLen: seqLen)
        deferDestroy(bertEmb)
        endStage("ALBERT encoder")

        // 3. Text encoder
        startStage("3/8 Text encoder")
        let textEncOut = try runTextEncoder(inputIdsBuf: inputIdsBuf, seqLen: seqLen)
        endStage("Text encoder")

        // 4. Predictor encoder
        startStage("4/8 Predictor encoder")
        let bertProjW = try requireWeight("onnx::MatMul_6040_quantized")
        let bertProjB = try requireWeight("kmodel.bert_encoder.bias")
        let bertProjOut = empty(seqLen * bertProjDim, label: "bert_proj")
        dispatchMatmul(A: bertOut, B: bertProjW.buffer, bias: bertProjB.buffer, out: bertProjOut,
                       M: seqLen, K: bertHiddenSize, N: bertProjDim, useBias: true)
        deferDestroy(bertOut)

        let predLstmConfigs: [(W: String, R: String, B: String)] = [
            ("onnx::LSTM_6094_quantized", "onnx::LSTM_6095_quantized", "onnx::LSTM_6093"),
            ("onnx::LSTM_6144_quantized", "onnx::LSTM_6145_quantized", "onnx::LSTM_6143"),
            ("onnx::LSTM_6194_quantized", "onnx::LSTM_6195_quantized", "onnx::LSTM_6193"),
        ]
        let predFcConfigs: [(w: String, b: String)] = [
            ("kmodel.predictor.text_encoder.lstms.1.fc.weight_quantized", "kmodel.predictor.text_encoder.lstms.1.fc.bias"),
            ("kmodel.predictor.text_encoder.lstms.3.fc.weight_quantized", "kmodel.predictor.text_encoder.lstms.3.fc.bias"),
            ("kmodel.predictor.text_encoder.lstms.5.fc.weight_quantized", "kmodel.predictor.text_encoder.lstms.5.fc.bias"),
        ]
        var predTextFeatures = bertProjOut
        for li in 0..<numPredLstmPairs {
            let cfg = predLstmConfigs[li]
            let fcCfg = predFcConfigs[li]
            let lstmIn = empty(seqLen * lstmInputSize, label: "pred_lstm\(li)_in")
            dispatchConcatBroadcast(a: predTextFeatures, b: stylePred, out: lstmIn,
                                    rows: seqLen, colsA: lstmBidir, colsB: styleHalf)
            let lstmW = try requireWeight(cfg.W)
            let lstmR = try requireWeight(cfg.R)
            let lstmB = try requireWeight(cfg.B)
            let lstmOut = empty(seqLen * 2 * lstmHidden, label: "pred_lstm\(li)_out")
            dispatchLSTM(input: lstmIn, W: lstmW.buffer, R: lstmR.buffer, bias: lstmB.buffer,
                         out: lstmOut, seqLen: seqLen, inputSize: lstmInputSize, hidden: lstmHidden, dirs: 2)
            deferDestroy(lstmIn)

            let fcW = try requireWeight(fcCfg.w)
            let fcB = try requireWeight(fcCfg.b)
            let fcOutDim = fcB.size
            let fcOut = empty(fcOutDim, label: "pred_fc\(li)")
            dispatchMatmul(A: stylePred, B: fcW.buffer, bias: fcB.buffer, out: fcOut,
                           M: 1, K: styleHalf, N: fcOutDim, useBias: true)

            let lnIdx = 2 * li + 1
            var lnW = "kmodel.predictor.text_encoder.lstms.\(lnIdx).norm.weight"
            var lnB = "kmodel.predictor.text_encoder.lstms.\(lnIdx).norm.bias"
            if weights[lnW] == nil { lnW = "/text_encoder/lstms.\(lnIdx)/Constant_7_output_0"; lnB = "/text_encoder/lstms.\(lnIdx)/Constant_8_output_0" }
            let lnGamma = try requireWeight(lnW)
            let lnBeta = try requireWeight(lnB)
            let normed = empty(seqLen * lstmBidir, label: "pred_ln\(li)")
            dispatchLayerNorm(input: lstmOut, gamma: lnGamma.buffer, beta: lnBeta.buffer, out: normed,
                              batchSize: seqLen, hiddenSize: lstmBidir, eps: 1e-5)
            deferDestroy(lstmOut)
            let adainOut = empty(seqLen * lstmBidir, label: "pred_adain\(li)")
            dispatchAdaINRowMajor(normed: normed, styleFc: fcOut, out: adainOut, channels: lstmBidir, rows: seqLen)
            deferDestroy(normed)
            if predTextFeatures !== bertProjOut { deferDestroy(predTextFeatures) }
            predTextFeatures = adainOut
        }
        endStage("Predictor encoder")

        // 5. Duration
        startStage("5/8 Duration")
        let durIn = empty(seqLen * lstmInputSize, label: "dur_in")
        dispatchConcatBroadcast(a: predTextFeatures, b: stylePred, out: durIn,
                                rows: seqLen, colsA: lstmBidir, colsB: styleHalf)
        let dW = try requireWeight("onnx::LSTM_6243_quantized")
        let dR = try requireWeight("onnx::LSTM_6244_quantized")
        let dB = try requireWeight("onnx::LSTM_6242")
        let dLstmOut = empty(seqLen * 2 * lstmHidden, label: "dur_lstm")
        dispatchLSTM(input: durIn, W: dW.buffer, R: dR.buffer, bias: dB.buffer,
                     out: dLstmOut, seqLen: seqLen, inputSize: lstmInputSize, hidden: lstmHidden, dirs: 2)
        deferDestroy(durIn)
        let dpW = try requireWeight("onnx::MatMul_6245")
        let dpB = try requireWeight("kmodel.predictor.duration_proj.linear_layer.bias")
        let durProj = empty(seqLen * 50, label: "dur_proj")
        dispatchMatmul(A: dLstmOut, B: dpW.buffer, bias: dpB.buffer, out: durProj,
                       M: seqLen, K: lstmBidir, N: 50, useBias: true)
        deferDestroy(dLstmOut)
        let durSig = empty(seqLen * 50, label: "dur_sig")
        dispatchSigmoid(input: durProj, out: durSig, size: seqLen * 50)
        deferDestroy(durProj)
        let durSigmoidData = readBuffer(durSig, size: seqLen * 50)
        deferDestroy(durSig)

        var durations = [Int32](repeating: 0, count: seqLen)
        var cumsum = [UInt32](repeating: 0, count: seqLen)
        var totalFrames = 0
        for i in 0..<seqLen {
            var sum: Float = 0
            for j in 0..<50 { sum += durSigmoidData[i * 50 + j] }
            let d = max(0, Int((sum / speed).rounded()))
            durations[i] = Int32(d)
            totalFrames += d
            cumsum[i] = UInt32(totalFrames)
        }
        let cumsumBuf = makeBuffer(cumsum.map { Int32(bitPattern: $0) }, label: "cumsum")

        let sharedLstmIn = empty(totalFrames * lstmInputSize, label: "shared_in")
        dispatchExpandRowMajor(input: durIn, cumsum: cumsumBuf, out: sharedLstmIn,
                                seqLen: seqLen, dim: lstmInputSize, total: totalFrames)
        let expandedText = empty(lstmBidir * totalFrames, label: "expanded_text")
        dispatchExpandChannelFirst(input: textEncOut, cumsum: cumsumBuf, out: expandedText,
                                    seqLen: seqLen, dim: lstmBidir, total: totalFrames)
        deferDestroy(textEncOut)

        let sW = try requireWeight("onnx::LSTM_6292_quantized")
        let sR = try requireWeight("onnx::LSTM_6293_quantized")
        let sB = try requireWeight("onnx::LSTM_6291")
        let sharedLstmOut = empty(totalFrames * 2 * lstmHidden, label: "shared_lstm_out")
        dispatchLSTM(input: sharedLstmIn, W: sW.buffer, R: sR.buffer, bias: sB.buffer,
                     out: sharedLstmOut, seqLen: totalFrames, inputSize: lstmInputSize, hidden: lstmHidden, dirs: 2)
        deferDestroy(sharedLstmIn)
        let sharedT = empty(lstmBidir * totalFrames, label: "shared_T")
        dispatchTranspose(input: sharedLstmOut, out: sharedT, rows: totalFrames, cols: lstmBidir)
        deferDestroy(sharedLstmOut)

        // N predictor
        var nFeat = sharedT
        var nCh = lstmBidir
        var nLen = totalFrames
        for bi in 0..<3 {
            let prefix = "kmodel.predictor.N.\(bi)"
            let pool: (weightName: String, channels: Int)? = (bi == 1) ? (weightName: "\(prefix).pool.weight", channels: predBlock0OutCh) : nil
            let r = try runAdaINResNetBlock(input: nFeat, style: stylePred,
                                             inCh: nCh, length: nLen, prefix: prefix,
                                             hasConv1x1: bi == 1,
                                             outCh: bi == 0 ? predBlock0OutCh : predBlock1OutCh,
                                             pool: pool)
            deferDestroy(nFeat)
            nFeat = r.output; nCh = r.outChannels; nLen = r.outLength
        }
        let nPW = try requireWeight("kmodel.predictor.N_proj.weight_quantized")
        let nPB = try requireWeight("kmodel.predictor.N_proj.bias")
        let nProjOut = empty(nLen, label: "n_proj")
        dispatchConv1d(input: nFeat, weight: nPW.buffer, bias: nPB.buffer, out: nProjOut,
                       inCh: nCh, outCh: 1, kernel: 1, inLen: nLen, outLen: nLen,
                       padding: 0, stride: 1, dilation: 1, useBias: true)
        deferDestroy(nFeat)
        let baseFrames = totalFrames

        // F0 predictor
        var f0Feat = sharedT
        var f0Ch = lstmBidir
        var f0Len = baseFrames
        for bi in 0..<3 {
            let prefix = "kmodel.predictor.F0.\(bi)"
            let pool: (weightName: String, channels: Int)? = (bi == 1) ? (weightName: "\(prefix).pool.weight", channels: predBlock0OutCh) : nil
            let r = try runAdaINResNetBlock(input: f0Feat, style: stylePred,
                                             inCh: f0Ch, length: f0Len, prefix: prefix,
                                             hasConv1x1: bi == 1,
                                             outCh: bi == 0 ? predBlock0OutCh : predBlock1OutCh,
                                             pool: pool)
            if bi > 0 { deferDestroy(f0Feat) }
            f0Feat = r.output; f0Ch = r.outChannels; f0Len = r.outLength
        }
        let f0PW = try requireWeight("kmodel.predictor.F0_proj.weight_quantized")
        let f0PB = try requireWeight("kmodel.predictor.F0_proj.bias")
        let f0ProjOut = empty(f0Len, label: "f0_proj")
        dispatchConv1d(input: f0Feat, weight: f0PW.buffer, bias: f0PB.buffer, out: f0ProjOut,
                       inCh: f0Ch, outCh: 1, kernel: 1, inLen: f0Len, outLen: f0Len,
                       padding: 0, stride: 1, dilation: 1, useBias: true)
        deferDestroy(f0Feat)
        endStage("Duration + N + F0")

        // 6. Decoder
        startStage("6/8 Decoder")
        let f0CW = try requireWeight("kmodel.decoder.F0_conv.weight")
        let f0CB = try requireWeight("kmodel.decoder.F0_conv.bias")
        let f0Conv = empty(baseFrames, label: "f0_conv")
        dispatchConv1d(input: f0ProjOut, weight: f0CW.buffer, bias: f0CB.buffer, out: f0Conv,
                       inCh: 1, outCh: 1, kernel: 3, inLen: f0Len, outLen: baseFrames,
                       padding: 1, stride: 2, dilation: 1, useBias: true)
        let nCW = try requireWeight("kmodel.decoder.N_conv.weight")
        let nCB = try requireWeight("kmodel.decoder.N_conv.bias")
        let nConv = empty(baseFrames, label: "n_conv")
        dispatchConv1d(input: nProjOut, weight: nCW.buffer, bias: nCB.buffer, out: nConv,
                       inCh: 1, outCh: 1, kernel: 3, inLen: nLen, outLen: baseFrames,
                       padding: 1, stride: 2, dilation: 1, useBias: true)
        deferDestroy(nProjOut)

        let decInCh = lstmBidir + 2
        let mid = empty((lstmBidir + 1) * baseFrames, label: "dec_mid")
        dispatchConcatChannels(a: expandedText, b: f0Conv, out: mid, chA: lstmBidir, chB: 1, length: baseFrames)
        let decoderIn = empty(decInCh * baseFrames, label: "decoder_in")
        dispatchConcatChannels(a: mid, b: nConv, out: decoderIn, chA: lstmBidir + 1, chB: 1, length: baseFrames)
        deferDestroy(mid); deferDestroy(predTextFeatures); deferDestroy(sharedT)

        let encodeOut = try runDecoderBlock(input: decoderIn, style: styleDec,
                                            inCh: decInCh, outCh: decEncodeOutCh, length: baseFrames,
                                            prefix: "kmodel.decoder.encode",
                                            hasConv1x1: true, pool: nil)
        deferDestroy(decoderIn)

        let asrW = try requireWeight("kmodel.decoder.asr_res.0.weight_quantized")
        let asrB = try requireWeight("kmodel.decoder.asr_res.0.bias")
        let asrCh = asrB.size
        let asrRes = empty(asrCh * baseFrames, label: "asr_res")
        dispatchConv1d(input: expandedText, weight: asrW.buffer, bias: asrB.buffer, out: asrRes,
                       inCh: lstmBidir, outCh: asrCh, kernel: 1, inLen: baseFrames, outLen: baseFrames,
                       padding: 0, stride: 1, dilation: 1, useBias: true)
        deferDestroy(expandedText)

        var decodeIn = try buildDecodeInput(features: encodeOut, f0Conv: f0Conv, nConv: nConv,
                                            asrRes: asrRes, featureCh: decEncodeOutCh, length: baseFrames, asrCh: asrCh)
        deferDestroy(encodeOut)
        let decodeInCh = decEncodeOutCh + asrCh + 2
        var decFrames = baseFrames
        var decodeOut = decodeIn
        for di in 0..<4 {
            let prefix = "kmodel.decoder.decode.\(di)"
            let outCh = di < 3 ? decDecodeOutCh : decDecode3OutCh
            if di == 3 {
                decodeOut = try runDecoderBlock(input: decodeIn, style: styleDec,
                                                inCh: decodeInCh, outCh: outCh, length: decFrames,
                                                prefix: prefix, hasConv1x1: true,
                                                pool: (weightName: "\(prefix).pool.weight", channels: decodeInCh))
                decFrames = decFrames * 2
            } else {
                decodeOut = try runDecoderBlock(input: decodeIn, style: styleDec,
                                                inCh: decodeInCh, outCh: outCh, length: decFrames,
                                                prefix: prefix, hasConv1x1: true, pool: nil)
            }
            deferDestroy(decodeIn)
            if di < 3 {
                decodeIn = try buildDecodeInput(features: decodeOut, f0Conv: f0Conv, nConv: nConv,
                                                asrRes: asrRes, featureCh: outCh, length: decFrames, asrCh: asrCh)
                deferDestroy(decodeOut)
            }
        }
        deferDestroy(f0Conv); deferDestroy(nConv); deferDestroy(asrRes)
        endStage("Decoder")

        // 7. HiFi-GAN
        startStage("7/8 HiFi-GAN")
        var genFeat = decodeOut
        var genCh = decDecode3OutCh
        var genLen = decFrames
        let preUps0 = empty(genCh * genLen, label: "pre_ups0")
        dispatchLeakyRelu(input: genFeat, out: preUps0, size: genCh * genLen, alpha: 0.1)
        deferDestroy(genFeat)
        genFeat = preUps0
        let ups0W = try requireWeight("kmodel.decoder.generator.ups.0.weight")
        let ups0B = try requireWeight("kmodel.decoder.generator.ups.0.bias")
        let ups0Len = genLen * 10
        let ups0Out = empty(hifiUps0OutCh * ups0Len, label: "ups0")
        dispatchConvTranspose1d(input: genFeat, weight: ups0W.buffer, bias: ups0B.buffer, out: ups0Out,
                                 inCh: genCh, outCh: hifiUps0OutCh, kernel: 20, inLen: genLen, outLen: ups0Len,
                                 stride: 10, padding: 5, useBias: true)
        deferDestroy(genFeat)
        genFeat = ups0Out; genCh = hifiUps0OutCh; genLen = ups0Len

        let stftLen = genLen * 6 + 1
        let noiseInput = try generateSourceExcitation(f0ProjBuf: f0ProjOut, f0Length: f0Len, stftLen: stftLen)
        deferDestroy(f0ProjOut)

        let nc0W = try requireWeight("kmodel.decoder.generator.noise_convs.0.weight_quantized")
        let nc0B = try requireWeight("kmodel.decoder.generator.noise_convs.0.bias")
        let nc0Len = Int(floor(Double(stftLen + 6 - 12) / 6.0)) + 1
        let nc0Out = empty(hifiUps0OutCh * nc0Len, label: "nc0")
        dispatchConv1d(input: noiseInput, weight: nc0W.buffer, bias: nc0B.buffer, out: nc0Out,
                       inCh: 22, outCh: hifiUps0OutCh, kernel: 12, inLen: stftLen, outLen: nc0Len,
                       padding: 3, stride: 6, dilation: 1, useBias: true)
        let nr0 = try runHiFiGANResBlock(input: nc0Out, style: styleDec,
                                         channels: hifiUps0OutCh, length: nc0Len,
                                         prefix: "kmodel.decoder.generator.noise_res.0")
        deferDestroy(nc0Out)
        let noisyUps0 = empty(genCh * genLen, label: "noisy_ups0")
        dispatchAdd(a: genFeat, b: nr0, out: noisyUps0, size: genCh * genLen)
        deferDestroy(genFeat); deferDestroy(nr0)
        genFeat = noisyUps0

        let r0 = try runHiFiGANResBlock(input: genFeat, style: styleDec,
                                        channels: genCh, length: genLen,
                                        prefix: "kmodel.decoder.generator.resblocks.0")
        let r1 = try runHiFiGANResBlock(input: genFeat, style: styleDec,
                                        channels: genCh, length: genLen,
                                        prefix: "kmodel.decoder.generator.resblocks.1")
        let rAvg0 = empty(genCh * genLen, label: "rAvg0")
        dispatchAddScale(a: r0, b: r1, out: rAvg0, size: genCh * genLen, scale: 0.5)
        deferDestroy(r0); deferDestroy(r1); deferDestroy(genFeat)
        genFeat = rAvg0

        let preUps1 = empty(genCh * genLen, label: "pre_ups1")
        dispatchLeakyRelu(input: genFeat, out: preUps1, size: genCh * genLen, alpha: 0.1)
        deferDestroy(genFeat)
        genFeat = preUps1
        let ups1W = try requireWeight("kmodel.decoder.generator.ups.1.weight")
        let ups1B = try requireWeight("kmodel.decoder.generator.ups.1.bias")
        let ups1Len = genLen * 6
        let ups1Out = empty(hifiUps1OutCh * ups1Len, label: "ups1")
        dispatchConvTranspose1d(input: genFeat, weight: ups1W.buffer, bias: ups1B.buffer, out: ups1Out,
                                 inCh: genCh, outCh: hifiUps1OutCh, kernel: 12, inLen: genLen, outLen: ups1Len,
                                 stride: 6, padding: 3, useBias: true)
        deferDestroy(genFeat)
        genFeat = ups1Out; genCh = hifiUps1OutCh; genLen = ups1Len

        let padded = empty(genCh * (genLen + 1), label: "padded")
        dispatchReflectionPad1d(input: genFeat, out: padded, channels: genCh, inLen: genLen, padL: 1, padR: 0)
        deferDestroy(genFeat)
        genFeat = padded; genLen = genLen + 1

        let nc1W = try requireWeight("kmodel.decoder.generator.noise_convs.1.weight_quantized")
        let nc1B = try requireWeight("kmodel.decoder.generator.noise_convs.1.bias")
        let nc1Out = empty(hifiUps1OutCh * stftLen, label: "nc1")
        dispatchConv1d(input: noiseInput, weight: nc1W.buffer, bias: nc1B.buffer, out: nc1Out,
                       inCh: 22, outCh: hifiUps1OutCh, kernel: 1, inLen: stftLen, outLen: stftLen,
                       padding: 0, stride: 1, dilation: 1, useBias: true)
        deferDestroy(noiseInput)
        let nr1 = try runHiFiGANResBlock(input: nc1Out, style: styleDec,
                                         channels: hifiUps1OutCh, length: stftLen,
                                         prefix: "kmodel.decoder.generator.noise_res.1")
        deferDestroy(nc1Out)
        let noisyPad = empty(genCh * genLen, label: "noisyPad")
        dispatchAdd(a: genFeat, b: nr1, out: noisyPad, size: genCh * genLen)
        deferDestroy(genFeat); deferDestroy(nr1)
        genFeat = noisyPad

        let r2 = try runHiFiGANResBlock(input: genFeat, style: styleDec,
                                        channels: genCh, length: genLen,
                                        prefix: "kmodel.decoder.generator.resblocks.2")
        let r3 = try runHiFiGANResBlock(input: genFeat, style: styleDec,
                                        channels: genCh, length: genLen,
                                        prefix: "kmodel.decoder.generator.resblocks.3")
        let rAvg1 = empty(genCh * genLen, label: "rAvg1")
        dispatchAddScale(a: r2, b: r3, out: rAvg1, size: genCh * genLen, scale: 0.5)
        deferDestroy(r2); deferDestroy(r3); deferDestroy(genFeat)
        genFeat = rAvg1

        let postLeaky = empty(genCh * genLen, label: "postLeaky")
        dispatchLeakyRelu(input: genFeat, out: postLeaky, size: genCh * genLen, alpha: 0.01)
        deferDestroy(genFeat)
        let cpW = try requireWeight("kmodel.decoder.generator.conv_post.weight_quantized")
        let cpB = try requireWeight("kmodel.decoder.generator.conv_post.bias")
        let convPost = empty(22 * genLen, label: "conv_post")
        dispatchConv1d(input: postLeaky, weight: cpW.buffer, bias: cpB.buffer, out: convPost,
                       inCh: genCh, outCh: 22, kernel: 7, inLen: genLen, outLen: genLen,
                       padding: 3, stride: 1, dilation: 1, useBias: true)
        deferDestroy(postLeaky)
        endStage("HiFi-GAN")

        // 8. iSTFT
        startStage("8/8 iSTFT")
        let waveLen = (genLen - 1) * 5 + 20
        let stftRealW = try requireWeight("kmodel.decoder.generator.stft.weight_backward_real")
        let stftImagW = try requireWeight("kmodel.decoder.generator.stft.weight_backward_imag")
        let waveGpu = empty(waveLen, label: "waveform_gpu")
        dispatchISTFT(convPost: convPost, wReal: stftRealW.buffer, wImag: stftImagW.buffer, out: waveGpu,
                      genLen: genLen, waveLen: waveLen, bins: 11, kernel: 20, stride: 5)
        deferDestroy(convPost)
        let waveData = readBuffer(waveGpu, size: waveLen)
        deferDestroy(waveGpu)
        endStage("iSTFT")

        // Trim 10 from each side
        let trimmed = Array(waveData[10..<(waveLen - 10)])
        status = .ready
        lastTimings = timingsAccum.map { StageTime(name: $0.0, ms: $0.1) }
        return trimmed
    }
}