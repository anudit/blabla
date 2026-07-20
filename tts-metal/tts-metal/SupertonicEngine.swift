//
//  SupertonicEngine.swift
//  tts-metal
//
//  Metal port of Supertonic 3 — a flow-matching latent TTS. Runs the four ONNX
//  sub-models (text_encoder, duration_predictor, vector_estimator, vocoder) as
//  hand-written Metal compute, loading weights by name via OnnxParser.
//
//  Pipeline (see docs/ARCHITECTURE.md):
//    text ─▶ tokenize ─▶ text_encoder → text_emb[256,T]
//                        duration_predictor → per-token durations → latent length L
//    z ~ N(0,1)[144,L] ─▶ ODE loop over vector_estimator (N steps) → latent[144,L]
//                      ─▶ vocoder → wav @ 44.1 kHz
//
//  NOTE ON VALIDATION: there is no local onnxruntime to diff against, so the
//  numerically-sensitive details (attention masking, the vocoder 144→24 unfold,
//  the exact flow-matching parametrisation) are implemented to the most likely
//  convention and marked `// VALIDATE`. They are verified on-device by comparing
//  output to supertonic-3/audio_samples/*_supertonic3.wav. Everything compiles and
//  the structure mirrors the ONNX graph in docs/onnx_maps/.
//

import Foundation
import Metal
import Accelerate

final class SupertonicEngine: @unchecked Sendable {

    // MARK: - Config (from supertonic-3/onnx/tts.json)

    struct Config {
        // latent geometry
        let latentDim = 24
        let chunkCompress = 6
        var latentCh: Int { latentDim * chunkCompress }   // 144
        let sampleRate = 44100

        // text encoder
        let charEmb = 256
        let teConvNextLayers = 6
        let teConvNextKsz = 5
        let teConvNextInter = 1024
        let teConvNextDil = [1, 1, 2, 2, 4, 4]
        let teAttnHeads = 4
        let teAttnLayers = 4
        let teAttnHidden = 256
        let teAttnFilter = 1024
        let teAttnWindow = 4          // VITS relative-position window (typical); VALIDATE

        // vector field
        let vfDim = 512
        let vfTimeDim = 64
        let vfTimeHidden = 256
        let vfSuperBlocks = 4         // 24 flattened main_blocks = 4 × 6 sub-layers
        let vfConvNextInter = 2048
        let vfConvNextKsz = 5
        let vfCross0Heads = 8
        let vfRotaryBase: Float = 10000
        let vfLastConvNextLayers = 4

        // vocoder / AE decoder
        let aeDecLayers = 10
        let aeDecDil = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
        let aeDecKsz = 7
        let aeDecInter = 2048
        let aeHdim = 512

        // flow-matching ODE
        var odeSteps = 8              // quality/speed knob; VALIDATE default
    }
    var config = Config()
    let sampleRate = 44100

    // MARK: - Metal

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private var library: MTLLibrary!
    private var pipelines: [String: MTLComputePipelineState] = [:]

    private var cmd: MTLCommandBuffer?
    private var enc: MTLComputeCommandEncoder?

    // MARK: - Weights

    struct Weight { let buf: MTLBuffer; let shape: [Int]; var count: Int { shape.reduce(1, *) } }
    // Namespaced by model: weights["te"]["char_embedder..."] etc.
    private var W: [String: [String: Weight]] = ["te": [:], "dp": [:], "vf": [:], "vo": [:]]
    private var filmWeights: [Weight] = []
    private var filmBiases: [MTLBuffer] = []

    private let weightAliasMap: [String: String] = [
        "vector_field.main_blocks.5.attention.W_value.linear.MatMul.weight": "onnx::MatMul_3407",
        "vector_field.main_blocks.11.attention.W_value.linear.MatMul.weight": "onnx::MatMul_3452",
        "vector_field.main_blocks.17.attention.W_value.linear.MatMul.weight": "onnx::MatMul_3497",
        "vector_field.main_blocks.23.attention.W_value.linear.MatMul.weight": "onnx::MatMul_3542",
        "vector_field.main_blocks.3.attn.W_key.linear.MatMul.weight": "onnx::MatMul_3391",
        "vector_field.main_blocks.3.attn.W_value.linear.MatMul.weight": "onnx::MatMul_3392",
        "vector_field.main_blocks.9.attn.W_key.linear.MatMul.weight": "onnx::MatMul_3436",
        "vector_field.main_blocks.9.attn.W_value.linear.MatMul.weight": "onnx::MatMul_3437",
        "vector_field.main_blocks.15.attn.W_key.linear.MatMul.weight": "onnx::MatMul_3481",
        "vector_field.main_blocks.15.attn.W_value.linear.MatMul.weight": "onnx::MatMul_3482",
        "vector_field.main_blocks.21.attn.W_key.linear.MatMul.weight": "onnx::MatMul_3526",
        "vector_field.main_blocks.21.attn.W_value.linear.MatMul.weight": "onnx::MatMul_3527",
        "vector_field.main_blocks.5.attention.W_key.linear.MatMul.weight": "onnx::MatMul_3406",
        "vector_field.main_blocks.11.attention.W_key.linear.MatMul.weight": "onnx::MatMul_3451",
        "vector_field.main_blocks.17.attention.W_key.linear.MatMul.weight": "onnx::MatMul_3496",
        "vector_field.main_blocks.23.attention.W_key.linear.MatMul.weight": "onnx::MatMul_3541",
        "vector_field.main_blocks.1.linear.linear.MatMul.weight": "onnx::MatMul_3384",
        "vector_field.main_blocks.7.linear.linear.MatMul.weight": "onnx::MatMul_3429",
        "vector_field.main_blocks.13.linear.linear.MatMul.weight": "onnx::MatMul_3474",
        "vector_field.main_blocks.19.linear.linear.MatMul.weight": "onnx::MatMul_3519",
        "vector_field.main_blocks.3.attn.W_query.linear.MatMul.weight": "onnx::MatMul_3390",
        "vector_field.main_blocks.3.attn.out_fc.linear.MatMul.weight": "onnx::MatMul_3399",
        "vector_field.main_blocks.5.attention.W_query.linear.MatMul.weight": "onnx::MatMul_3405",
        "vector_field.main_blocks.5.attention.out_fc.linear.MatMul.weight": "onnx::MatMul_3408",
        "vector_field.main_blocks.9.attn.W_query.linear.MatMul.weight": "onnx::MatMul_3435",
        "vector_field.main_blocks.9.attn.out_fc.linear.MatMul.weight": "onnx::MatMul_3444",
        "vector_field.main_blocks.11.attention.W_query.linear.MatMul.weight": "onnx::MatMul_3450",
        "vector_field.main_blocks.11.attention.out_fc.linear.MatMul.weight": "onnx::MatMul_3453",
        "vector_field.main_blocks.15.attn.W_query.linear.MatMul.weight": "onnx::MatMul_3480",
        "vector_field.main_blocks.15.attn.out_fc.linear.MatMul.weight": "onnx::MatMul_3489",
        "vector_field.main_blocks.17.attention.W_query.linear.MatMul.weight": "onnx::MatMul_3495",
        "vector_field.main_blocks.17.attention.out_fc.linear.MatMul.weight": "onnx::MatMul_3498",
        "vector_field.main_blocks.21.attn.W_query.linear.MatMul.weight": "onnx::MatMul_3525",
        "vector_field.main_blocks.21.attn.out_fc.linear.MatMul.weight": "onnx::MatMul_3534",
        "vector_field.main_blocks.23.attention.W_query.linear.MatMul.weight": "onnx::MatMul_3540",
        "vector_field.main_blocks.23.attention.out_fc.linear.MatMul.weight": "onnx::MatMul_3543",
        "speech_prompted_text_encoder.attention1.W_query.linear.MatMul.weight": "onnx::MatMul_3680",
        "speech_prompted_text_encoder.attention1.W_key.linear.MatMul.weight": "onnx::MatMul_3681",
        "speech_prompted_text_encoder.attention1.W_value.linear.MatMul.weight": "onnx::MatMul_3682",
        "speech_prompted_text_encoder.attention1.out_fc.linear.MatMul.weight": "onnx::MatMul_3683",
        "speech_prompted_text_encoder.attention2.W_query.linear.MatMul.weight": "onnx::MatMul_3684",
        "speech_prompted_text_encoder.attention2.W_key.linear.MatMul.weight": "onnx::MatMul_3685",
        "speech_prompted_text_encoder.attention2.W_value.linear.MatMul.weight": "onnx::MatMul_3686",
        "speech_prompted_text_encoder.attention2.out_fc.linear.MatMul.weight": "onnx::MatMul_3687",
        "decoder.embed.net.Conv.weight": "onnx::Conv_1441",
        "decoder.embed.net.Conv.bias": "onnx::Conv_1442",
        "decoder.head.act.PRelu.slope": "onnx::PRelu_1506",
    ]

    // MARK: - Tokenizer & voices

    private var unicodeIndexer: [Int32] = []     // codepoint -> token id (or -1)
    struct VoiceStyle { let styleTtl: [Float]; let styleDp: [Float] }  // [50*256], [8*16]
    private(set) var voices: [String: VoiceStyle] = [:]

    enum Status: Equatable { case uninitialized, loading(String), ready, error(String) }
    private(set) var status: Status = .uninitialized

    // Timing
    struct StageTime { let name: String; let ms: Double }
    private(set) var lastTimings: [StageTime] = []
    var profile = false

    init(device: MTLDevice, queue: MTLCommandQueue) {
        self.device = device
        self.queue = queue
    }

    // MARK: - Load

    @discardableResult
    func load() throws -> Bool {
        status = .loading("Compiling Metal shaders")
        guard let lib = device.makeDefaultLibrary() else {
            status = .error("default.metallib not found"); return false
        }
        library = lib
        compilePipelines()

        let models: [(String, String)] = [
            ("te", "text_encoder"), ("dp", "duration_predictor"),
            ("vf", "vector_estimator"), ("vo", "vocoder"),
        ]
        for (ns, file) in models {
            status = .loading("Loading \(file)")
            guard let url = Bundle.main.url(forResource: file, withExtension: "onnx",
                                            subdirectory: "supertonic") ??
                            Bundle.main.url(forResource: file, withExtension: "onnx") else {
                status = .error("\(file).onnx missing from bundle"); return false
            }
            let data = try Data(contentsOf: url, options: .alwaysMapped)
            let tensors = try OnnxParser(data).parseInitializers()
            try uploadWeights(tensors, into: ns)
            print("[Supertonic] \(file): \(W[ns]!.count) weight buffers")
        }

        status = .loading("Loading tokenizer")
        try loadTokenizer()

        status = .loading("Loading voices")
        try loadVoices()

        // Cache film weights and biases
        let allVfInits = W["vf"]!

        func blockIndex(of name: String) -> Int {
            if let range = name.range(of: "main_blocks.") {
                let suffix = name[range.upperBound...]
                if let dotIndex = suffix.firstIndex(of: ".") {
                    let numStr = suffix[..<dotIndex]
                    return Int(numStr) ?? 0
                }
            }
            return 0
        }

        var tempWeights: [(String, Weight)] = []
        for (name, wt) in allVfInits {
            if wt.shape == [64, 512] {
                tempWeights.append((name, wt))
            }
        }
        tempWeights.sort(by: { blockIndex(of: $0.0) < blockIndex(of: $1.0) })
        self.filmWeights = tempWeights.map { $0.1 }

        var tempBiases: [(String, MTLBuffer)] = []
        for (name, wt) in allVfInits {
            if name.contains("linear.linear.bias") {
                tempBiases.append((name, wt.buf))
            }
        }
        tempBiases.sort(by: { blockIndex(of: $0.0) < blockIndex(of: $1.0) })
        self.filmBiases = tempBiases.map { $0.1 }
        print("[Supertonic] Cached \(filmWeights.count) film weights and \(filmBiases.count) film biases")

        status = .ready
        return true
    }

    private func compilePipelines() {
        let names = [
            // reused from Kitten / LavaSR / RE-USE
            "matmul_kernel", "matmul_gelu_kernel", "matmul_relu_kernel", "layer_norm_kernel", "transpose_kernel",
            "conv1d_kernel", "conv1d_tiled_kernel", "gelu_kernel", "tanh_kernel", "sigmoid_kernel",
            "add_kernel", "scale_kernel", "add_scale_kernel", "softmax_kernel",
            "lava_dwconv1d_kernel", "lava_gamma_residual_kernel", "reuse_prelu_kernel",
            "reuse_matmul_kernel", "reuse_matmul_gelu_kernel", "reuse_matmul_relu_kernel",
            // new Supertonic kernels
            "st_gelu_erf_kernel", "st_softplus_kernel", "st_film_kernel", "st_add_col_kernel",
            "st_rope_kernel", "st_mha_kernel", "st_rel_attn_kernel", "st_sinusoid_kernel",
            "st_dwconv1d_kernel", "st_causal_conv1d_kernel", "st_causal_dwconv1d_kernel",
            "st_dwconv1d_edge_kernel", "st_rope_norm_kernel", "st_cfg_euler_kernel",
            "copy_kernel", "transpose_batched_kernel", "st_dwconv1d_edge_batched_kernel",
            "lava_gamma_residual_batched_kernel", "st_add_col_batched_kernel",
        ]
        for n in names {
            guard let fn = library.makeFunction(name: n) else { print("[Supertonic] MISSING \(n)"); continue }
            do { pipelines[n] = try device.makeComputePipelineState(function: fn) }
            catch { print("[Supertonic] pipeline fail \(n): \(error)") }
        }
        print("[Supertonic] compiled \(pipelines.count) pipelines")
    }

    private func uploadWeights(_ tensors: [String: OnnxTensor], into ns: String) throws {
        for (name, t) in tensors {
            if name.hasSuffix("_scale") || name.hasSuffix("_zero_point") { continue }
            let total = t.dims.reduce(1, *)
            if total == 0 || t.rawData.isEmpty { continue }
            let floats: [Float]
            switch t.dataType {
            case OnnxDtype.float32: floats = OnnxDequant.float32Data(t.rawData)
            case OnnxDtype.float16: floats = OnnxDequant.float16Array(t.rawData)
            case OnnxDtype.int64: continue
            default: continue    // int8/uint8 quant handled in a later pass if present
            }
            guard let buf = device.makeBuffer(bytes: floats, length: floats.count * 4,
                                              options: [.storageModeShared]) else { continue }
            buf.label = "\(ns).\(name)"
            W[ns]![name] = Weight(buf: buf, shape: t.dims)
        }
    }

    private func loadTokenizer() throws {
        guard let url = Bundle.main.url(forResource: "unicode_indexer", withExtension: "json",
                                        subdirectory: "supertonic") ??
                        Bundle.main.url(forResource: "unicode_indexer", withExtension: "json") else {
            throw err("unicode_indexer.json missing")
        }
        let arr = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [Int] ?? []
        unicodeIndexer = arr.map { Int32($0) }
        print("[Supertonic] tokenizer: \(unicodeIndexer.count) codepoints")
    }

    private func loadVoices() throws {
        // voice_styles/*.json each has style_ttl [1,50,256] and style_dp [1,8,16]
        let names = ["david-deep", "F1","F2","F3","F4","F5","M1","M2","M3","M4","M5"]
        for name in names {
            guard let url = Bundle.main.url(forResource: name, withExtension: "json",
                                            subdirectory: "voice_styles") ??
                            Bundle.main.url(forResource: name, withExtension: "json") else { continue }
            guard let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
            else { continue }
            func flat(_ key: String) -> [Float] {
                guard let d = obj[key] as? [String: Any],
                      let data = d["data"] else { return [] }
                var out: [Float] = []
                func rec(_ x: Any) {
                    if let a = x as? [Any] { for e in a { rec(e) } }
                    else if let n = x as? NSNumber { out.append(n.floatValue) }
                }
                rec(data)
                return out
            }
            let ttl = flat("style_ttl"), dp = flat("style_dp")
            if ttl.count == 50 * 256 && dp.count == 8 * 16 {
                voices[name] = VoiceStyle(styleTtl: ttl, styleDp: dp)
            }
        }
        print("[Supertonic] loaded \(voices.count) voices")
    }

    // MARK: - Tokenize

    func preprocess(_ text: String, lang: String = "en") -> String {
        var t = text.decomposedStringWithCompatibilityMapping   // NFKD
        t = t.replacingOccurrences(of: "\u{2013}", with: "-")
             .replacingOccurrences(of: "\u{2014}", with: "-")
             .replacingOccurrences(of: "\u{2019}", with: "'")
             .replacingOccurrences(of: "\u{201C}", with: "\"")
             .replacingOccurrences(of: "\u{201D}", with: "\"")
        t = t.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
        let enders: Set<Character> = [".", "!", "?", ";", ":", ",", "'", "\"", ")", "]", "}", "…"]
        if let last = t.last, !enders.contains(last) { t += "." }
        return "<\(lang)>\(t)</\(lang)>"
    }

    func tokenize(_ text: String) -> [Int32] {
        var ids: [Int32] = []
        for scalar in preprocess(text).unicodeScalars {
            let cp = Int(scalar.value)
            if cp < unicodeIndexer.count {
                let id = unicodeIndexer[cp]
                if id >= 0 { ids.append(id) }
            }
        }
        return ids
    }

    // MARK: - Public generate

    /// Synthesize `text` with voice `voiceName`, returning mono 44.1 kHz PCM.
    func generate(_ text: String, voiceName: String, totalSteps: Int = 8, speed: Float = 1.05) throws -> [Float] {
        guard status == .ready else { throw err("engine not ready") }
        guard let voice = voices[voiceName] ?? voices["M1"] else { throw err("no voice") }
        var timings: [StageTime] = []
        func stage<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
            let t0 = now(); let r = try body(); flushAndWait()
            timings.append(StageTime(name: name, ms: now() - t0)); return r
        }

        let ids = tokenize(text)
        guard !ids.isEmpty else { return [] }
        let T = ids.count
        let idsBuf = makeIntBuf(ids)
        let styleTtl = makeBuf(voice.styleTtl)     // [50,256]
        let styleDp = makeBuf(voice.styleDp)       // [8,16]

        // 1) text encoder → text_emb [256, T]
        let textEmb = try stage("text_encoder") {
            try runTextEncoder(ids: idsBuf, T: T, styleTtl: styleTtl)
        }
        printBufStats("textEmb", textEmb, count: 256 * T)

        // 2) duration predictor → latent length L
        let L = try stage("duration_predictor") {
            try runDurationPredictor(ids: idsBuf, T: T, styleDp: styleDp, speed: speed)
        }
        guard L > 0 else { return [] }

        // 3) flow-matching ODE over vector_estimator → latent [144, L]
        let latent = try stage("flow_matching") {
            try runFlowMatching(textEmb: textEmb, T: T, styleTtl: styleTtl, L: L, totalSteps: totalSteps)
        }
        printBufStats("latent", latent, count: 144 * L)

        // 4) vocoder → waveform
        let wav = try stage("vocoder") {
            try runVocoder(latent: latent, L: L)
        }

        lastTimings = timings
        if profile { for t in timings { print(String(format: "[Supertonic] %-20s %.1f ms", (t.name as NSString).utf8String!, t.ms)) } }

        let wavMean = wav.reduce(0, +) / Float(wav.count)
        let wavVar = wav.map { ($0 - wavMean) * ($0 - wavMean) }.reduce(0, +) / Float(wav.count)
        print(String(format: "[Stats] wav: count=%d mean=%.6f std=%.6f min=%.6f max=%.6f", wav.count, wavMean, sqrt(wavVar), wav.min() ?? 0, wav.max() ?? 0))

        return wav
    }

    private func printBufStats(_ label: String, _ buf: MTLBuffer, count: Int) {
        let arr = readBuf(buf, count: count)
        let mean = arr.reduce(0, +) / Float(count)
        let variance = arr.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(count)
        let std = sqrt(variance)
        let minVal = arr.min() ?? 0
        let maxVal = arr.max() ?? 0
        print(String(format: "[Stats] %@: count=%d mean=%.6f std=%.6f min=%.6f max=%.6f", label, count, mean, std, minVal, maxVal))
    }

    // MARK: - Stage: text encoder

    // char embed → 6× ConvNeXt (dilated) → 4× rel-pos attn block → proj_out
    private func runTextEncoder(ids: MTLBuffer, T: Int, styleTtl: MTLBuffer) throws -> MTLBuffer {
        let C = config.charEmb
        // char embedding: gather rows of char_embedder weight [V, C] → x[C, T] (channel-major)
        var x = try embedChannelMajor(ns: "te", embName: findWeight("te", contains: "char_embedder"), ids: ids, T: T, C: C)
        printBufStats("x after embed", x, count: C * T)

        // ConvNeXt stack
        for i in 0..<config.teConvNextLayers {
            x = try convNextLayer(ns: "te", prefix: convnextPrefix("te", "convnext.convnext.\(i)"),
                                  x: x, C: C, T: T, ksz: config.teConvNextKsz,
                                  dil: config.teConvNextDil[i], inter: config.teConvNextInter)
            printBufStats("x after convnext.\(i)", x, count: C * T)
        }
        let convnextOut = x   // save for residual
        // rel-pos attention encoder (operates channel-major → the ONNX conv_q/k/v are Conv1d [outCh,inCh,1])
        // The ONNX attn_encoder works in channel-major throughout. We convert to row-major for our matmul helpers.
        var xt = try toRowMajor(x, rows: C, cols: T)   // [T, C]
        for i in 0..<config.teAttnLayers {
            xt = try relAttnBlock(ns: "te", layer: i, x: xt, T: T, C: C,
                                  heads: config.teAttnHeads, filter: config.teAttnFilter)
            printBufStats("xt after attn.\(i)", xt, count: C * T)
        }
        // ONNX: attn_encoder output (channel-major) + convnext.5 output → *text_mask → proj_out
        // proj_out in ONNX is NOT a linear layer — it's just Mul by text_mask (node [1827]).
        let attn_out = try toChannelMajor(xt, rows: T, cols: C)   // [C, T]
        let projOut = empty(C * T)
        dispatchAdd(a: attn_out, b: convnextOut, out: projOut, size: C * T)  // residual
        // We skip text_mask multiplication since we don't pad (batch=1, all positions valid).
        printBufStats("projOut (pre-speech_prompted)", projOut, count: C * T)

        // speech_prompted_text_encoder: two tanh-scored cross-attention layers fusing
        // the voice style into the text. Tiny tensors (T×256, 50×256) → done on CPU for
        // exactness. Mirrors the ONNX graph precisely (see speechPromptedEncoder).
        let projCM = readBuf(projOut, count: C * T)               // channel-major [C,T]
        let stl = readBuf(styleTtl, count: 50 * C)                // style_ttl [50,256] row-major
        guard let styleKey = rawWeight("te", "tts.ttl.style_encoder.style_token_layer.style_key")
        else { throw err("style_key weight missing") }
        let resultArr = try speechPromptedEncoder(projCM: projCM, T: T, C: C,
                                                  styleTtl: stl, styleKey: styleKey)
        let result = makeBuf(resultArr)                           // channel-major [C,T]
        printBufStats("text_emb final", result, count: C * T)
        return result
    }

    /// Raw weight as a Float array. Tries the exact ONNX initializer name first, then
    /// the alias resolver (so descriptive `...MatMul.weight` names resolve to `onnx::MatMul_*`).
    private func rawWeight(_ ns: String, _ name: String) -> [Float]? {
        if let w = W[ns]?[name] { return readBuf(w.buf, count: w.count) }
        guard let buf = weightBuf(ns, name) else { return nil }
        return readBuf(buf, count: buf.length / 4)
    }

    /// CPU matmul: out[M,N] = A[M,K] · W[K,N] (+ bias[N]).  W is row-major [in,out]
    /// exactly as stored in ONNX (used as the B operand of an ONNX MatMul → no transpose).
    /// Uses Accelerate BLAS (multi-threaded, vectorized) — the flow-matching cross
    /// attentions used to run this as a scalar triple loop, which was the CPU hang.
    private func cpuMatmul(_ A: [Float], _ W: [Float], _ bias: [Float]?, M: Int, K: Int, N: Int) -> [Float] {
        var out = [Float](repeating: 0, count: M * N)
        A.withUnsafeBufferPointer { ap in
            W.withUnsafeBufferPointer { wp in
                out.withUnsafeMutableBufferPointer { op in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(M), Int32(N), Int32(K), 1.0,
                                ap.baseAddress, Int32(K),
                                wp.baseAddress, Int32(N), 0.0,
                                op.baseAddress, Int32(N))
                }
            }
        }
        if let bias = bias {
            for m in 0..<M {
                let off = m * N
                for n in 0..<N { out[off + n] += bias[n] }
            }
        }
        return out
    }

    /// speech_prompted_text_encoder — exact CPU port of the ONNX subgraph.
    ///   h        = transpose(projOut) → [T,C]              (query source, the residual)
    ///   attn(q)  = ( softmax( (qWq · tanh(styleKey·Wk)^T) / 16 ) · (styleTtl·Wv) ) · Wo
    ///   add1     = attn1(h)   + h
    ///   add_1    = attn2(add1) + h                          (residual is h, NOT add1)
    ///   out      = LayerNorm_C(add_1) → transpose → [C,T]
    /// 2 heads, head_dim 128, fixed score divisor 16 (= sqrt(256)). text_mask is all-ones
    /// for our unpadded single sequence, so its Where/Mul terms are identities and dropped.
    private func speechPromptedEncoder(projCM: [Float], T: Int, C: Int,
                                       styleTtl: [Float], styleKey: [Float]) throws -> [Float] {
        let heads = 2, hd = C / heads, S = 50
        let scale: Float = 1.0 / 16.0

        // h [T,C] = transpose of channel-major projOut [C,T]
        var h = [Float](repeating: 0, count: T * C)
        for c in 0..<C { for t in 0..<T { h[t * C + c] = projCM[c * T + t] } }

        func attn(_ idx: Int, _ q_src: [Float]) throws -> [Float] {
            let p = "onnx::MatMul_"
            let (wqN, wkN, wvN, woN) = idx == 1
                ? (p+"3680", p+"3681", p+"3682", p+"3683")
                : (p+"3684", p+"3685", p+"3686", p+"3687")
            let bp = "tts.ttl.speech_prompted_text_encoder.attention\(idx)."
            guard let Wq = rawWeight("te", wqN), let Wk = rawWeight("te", wkN),
                  let Wv = rawWeight("te", wvN), let Wo = rawWeight("te", woN)
            else { throw err("speech_prompted attention\(idx) weights missing") }
            let bq = rawWeight("te", bp+"W_query.linear.bias")
            let bk = rawWeight("te", bp+"W_key.linear.bias")
            let bv = rawWeight("te", bp+"W_value.linear.bias")
            let bo = rawWeight("te", bp+"out_fc.linear.bias")

            let Q = cpuMatmul(q_src, Wq, bq, M: T, K: C, N: C)      // [T,C]
            let K = cpuMatmul(styleKey, Wk, bk, M: S, K: C, N: C)   // [S,C]
            let V = cpuMatmul(styleTtl, Wv, bv, M: S, K: C, N: C)   // [S,C]

            var ctx = [Float](repeating: 0, count: T * C)
            var scores = [Float](repeating: 0, count: S)
            for head in 0..<heads {
                let hoff = head * hd
                for t in 0..<T {
                    var mx = -Float.greatestFiniteMagnitude
                    for s in 0..<S {
                        var dot: Float = 0
                        for d in 0..<hd { dot += Q[t * C + hoff + d] * tanh(K[s * C + hoff + d]) }
                        let sc = dot * scale
                        scores[s] = sc
                        if sc > mx { mx = sc }
                    }
                    var sum: Float = 0
                    for s in 0..<S { let e = exp(scores[s] - mx); scores[s] = e; sum += e }
                    let inv = 1.0 / sum
                    for d in 0..<hd {
                        var acc: Float = 0
                        for s in 0..<S { acc += scores[s] * V[s * C + hoff + d] }
                        ctx[t * C + hoff + d] = acc * inv
                    }
                }
            }
            return cpuMatmul(ctx, Wo, bo, M: T, K: C, N: C)         // [T,C]
        }

        let o1 = try attn(1, h)
        var add1 = [Float](repeating: 0, count: T * C)
        for i in 0..<(T * C) { add1[i] = o1[i] + h[i] }
        let o2 = try attn(2, add1)
        var add_1 = [Float](repeating: 0, count: T * C)
        for i in 0..<(T * C) { add_1[i] = o2[i] + h[i] }

        // LayerNorm over channel dim C
        guard let lnW = rawWeight("te", "tts.ttl.speech_prompted_text_encoder.norm.norm.weight"),
              let lnB = rawWeight("te", "tts.ttl.speech_prompted_text_encoder.norm.norm.bias")
        else { throw err("speech_prompted norm weights missing") }
        var out = [Float](repeating: 0, count: C * T)              // channel-major [C,T]
        for t in 0..<T {
            var mean: Float = 0
            for c in 0..<C { mean += add_1[t * C + c] }
            mean /= Float(C)
            var varc: Float = 0
            for c in 0..<C { let d = add_1[t * C + c] - mean; varc += d * d }
            varc /= Float(C)
            let inv = 1.0 / (varc + 1e-6).squareRoot()
            for c in 0..<C {
                let n = (add_1[t * C + c] - mean) * inv * lnW[c] + lnB[c]
                out[c * T + t] = n                                  // transpose → channel-major
            }
        }
        return out
    }

    // MARK: - Stage: duration predictor

    // Latent length from predicted duration (SECONDS), matching the reference:
    //   chunk_size = base_chunk_size * chunk_compress_factor (= 512*6 = 3072)
    //   L = ceil(duration_seconds * sample_rate / chunk_size)
    // VALIDATE: `durSeconds` is currently estimated from character count; the real
    // duration_predictor graph (dp) should replace the estimate.
    private func dpEmbedAndConcat(ids: MTLBuffer, T: Int) throws -> MTLBuffer {
        guard let emb = weightBuf("dp", "char_embedder.weight"),
              let sent = weightBuf("dp", "sentence_token")
        else { throw err("dp embed weights missing") }
        let table = readBuf(emb, count: 8322 * 64) // V=8322, C=64
        let sentToken = readBuf(sent, count: 64) // [1, 64, 1]
        let idArr = readIntBuf(ids, count: T)

        var out = [Float](repeating: 0, count: 64 * (T + 1))
        for c in 0..<64 {
            out[c * (T + 1)] = sentToken[c]
            for t in 0..<T {
                let id = Int(idArr[t])
                out[c * (T + 1) + 1 + t] = (id >= 0 && id < 8322) ? table[id * 64 + c] : 0
            }
        }
        return makeBuf(out)
    }

    private func runDurationPredictor(ids: MTLBuffer, T: Int, styleDp: MTLBuffer, speed: Float) throws -> Int {
        return try autoreleasepool {
            let C = 64
            var x = try dpEmbedAndConcat(ids: ids, T: T)

            // 6 ConvNeXt layers
            for i in 0..<6 {
                x = try convNextLayer(ns: "dp", prefix: "sentence_encoder.convnext.convnext.\(i)",
                                      x: x, C: C, T: T + 1, ksz: 5, dil: [1, 1, 2, 2, 4, 4][i], inter: 256)
            }
            let convnextOut = x

            // 2 rel-pos attention encoder layers
            var xt = try toRowMajor(x, rows: C, cols: T + 1)
            for i in 0..<2 {
                xt = try relAttnBlock(ns: "dp", layer: i, x: xt, T: T + 1, C: C, heads: 2, filter: 256)
            }

            let attn_out = try toChannelMajor(xt, rows: T + 1, cols: C)
            let addOut = empty(C * (T + 1))
            dispatchAdd(a: attn_out, b: convnextOut, out: addOut, size: C * (T + 1))

            let addArr = readBuf(addOut, count: C * (T + 1))
            var token0 = [Float](repeating: 0, count: C)
            for c in 0..<C { token0[c] = addArr[c * (T + 1)] }

            guard let projW = rawWeight("dp", "sentence_encoder.proj_out.net.weight") else {
                throw err("dp proj_out weight missing")
            }
            var token0Proj = [Float](repeating: 0, count: C)
            for c in 0..<C {
                var sum: Float = 0
                for i in 0..<C { sum += token0[i] * projW[c * C + i] }
                token0Proj[c] = sum
            }

            guard let w0 = rawWeight("dp", "predictor.layers.0.weight"),
                  let b0 = rawWeight("dp", "predictor.layers.0.bias"),
                  let w1 = rawWeight("dp", "predictor.layers.1.weight"),
                  let b1 = rawWeight("dp", "predictor.layers.1.bias"),
                  let actW = rawWeight("dp", "predictor.activation.weight")
            else { throw err("dp MLP weights missing") }

            let styleDpArr = readBuf(styleDp, count: 128)
            var combined = token0Proj + styleDpArr

            var h = [Float](repeating: 0, count: 128)
            for j in 0..<128 {
                var sum = b0[j]
                for i in 0..<192 { sum += combined[i] * w0[j * 192 + i] }
                h[j] = sum
            }

            let slope = actW[0]
            var hAct = [Float](repeating: 0, count: 128)
            for j in 0..<128 { hAct[j] = h[j] > 0 ? h[j] : h[j] * slope }

            var outVal = b1[0]
            for j in 0..<128 { outVal += hAct[j] * w1[j] }

            let durSeconds = exp(outVal) / speed
            print("[DurationPredict] predicted duration = \(durSeconds) s")

            let chunkSize = Double(512 * config.chunkCompress)
            let L = Int((Double(durSeconds) * Double(config.sampleRate) / chunkSize).rounded(.up))
            return max(1, L)
        }
    }

    // MARK: - Stage: flow matching

    // Precomputed, generation-constant conditioning buffers (text + style, for both
    // the conditional and unconditional CFG branches) plus the shared RoPE theta table.
    private struct VfCond {
        let textR_cond: MTLBuffer, textR_null: MTLBuffer            // [T,256] row-major
        let sK_cond: MTLBuffer, sV_cond: MTLBuffer                  // [S,256]
        let sK_null: MTLBuffer, sV_null: MTLBuffer                  // [S,256]
        let S_cond: Int, S_null: Int
        let theta: MTLBuffer
    }

    private func prepConditioning(textEmb: MTLBuffer, T: Int, styleTtl: MTLBuffer) throws -> VfCond {
        let C = config.charEmb
        guard let styleK_cond = rawWeight("vf", "/vector_estimator/Expand_output_0"),
              let textNullTok = rawWeight("vf", "vector_estimator.tts.ttl.uncond_masker.text_special_token"),
              let styleK_null = rawWeight("vf", "vector_estimator.tts.ttl.uncond_masker.style_key_special_token"),
              let styleV_null = rawWeight("vf", "vector_estimator.tts.ttl.uncond_masker.style_value_special_token"),
              let theta = rawWeight("vf", "vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.theta")
        else { throw err("vector_field conditioning tokens missing") }
        // text (row-major [T,256]): cond = transpose of textEmb[256,T]; null = broadcast token.
        let textR_cond = try toRowMajor(textEmb, rows: C, cols: T)
        var textNullR = [Float](repeating: 0, count: T * C)
        for t in 0..<T { for c in 0..<C { textNullR[t * C + c] = textNullTok[c] } }
        return VfCond(textR_cond: textR_cond, textR_null: makeBuf(textNullR),
                      sK_cond: makeBuf(styleK_cond), sV_cond: styleTtl,
                      sK_null: makeBuf(styleK_null), sV_null: makeBuf(styleV_null),
                      S_cond: styleK_cond.count / C, S_null: styleK_null.count / C,
                      theta: makeBuf(theta))
    }

    // Flow-matching loop with classifier-free guidance, fully on-GPU. Each step runs
    // the vector estimator for the conditional and unconditional conditioning, then
    // combines the velocities and takes an Euler step — all as Metal kernels, so the
    // ODE state stays resident and there is no per-step CPU round-trip:
    //   v = w1·v_cond − w2·v_uncond   (w1=4, w2=3);  x_{k+1} = x_k + v / total_step
    private func runFlowMatching(textEmb: MTLBuffer, T: Int, styleTtl: MTLBuffer, L: Int, totalSteps: Int) throws -> MTLBuffer {
        let ch = config.latentCh
        let cond = try prepConditioning(textEmb: textEmb, T: T, styleTtl: styleTtl)
        var xBuf = makeBuf(gaussian(ch * L))    // x_0 ~ N(0,1)  [144, L]
        let inv = 1.0 / Float(totalSteps)
        for k in 0..<totalSteps {
            let tEmbBuf = makeBuf(try timeEmbeddingArr(cur: Float(k), total: Float(totalSteps)))  // [64]
            // one batched pass yields both velocities: vB = [2, 144, L] (b0 cond, b1 uncond)
            let vB = try vfVelocityBatched(x: xBuf, L: L, tEmbBuf: tEmbBuf, T: T, cond: cond)
            let vCond = empty(ch * L); dispatchCopy(src: vB, dst: vCond, size: ch * L, srcOff: 0, dstOff: 0)
            let vUncond = empty(ch * L); dispatchCopy(src: vB, dst: vUncond, size: ch * L, srcOff: ch * L, dstOff: 0)
            let nx = empty(ch * L)
            dispatchCfgEuler(x: xBuf, vCond: vCond, vUncond: vUncond, out: nx,
                             size: ch * L, w1: 4.0, w2: 3.0, invTotal: inv)
            xBuf = nx
            if k == 0 { printBufStats("vfield[0] step0", xBuf, count: ch * L) }
        }
        return xBuf
    }

    // One flow-matching step (CPU array in/out) — retained for the ST_VALIDATE harness,
    // which feeds a fixed reference noise/text_emb and diffs the result. Delegates to the
    // batched GPU path so validation exercises the same kernels the production loop uses.
    func vectorEstimatorStep(x: [Float], textEmb: MTLBuffer, T: Int, styleTtl: MTLBuffer,
                             L: Int, curStep: Float, totalStep: Float) throws -> [Float] {
        let ch = config.latentCh
        let cond = try prepConditioning(textEmb: textEmb, T: T, styleTtl: styleTtl)
        let xBuf = makeBuf(x)
        let tEmbBuf = makeBuf(try timeEmbeddingArr(cur: curStep, total: totalStep))
        let vB = try vfVelocityBatched(x: xBuf, L: L, tEmbBuf: tEmbBuf, T: T, cond: cond)
        let vCond = empty(ch * L); dispatchCopy(src: vB, dst: vCond, size: ch * L, srcOff: 0, dstOff: 0)
        let vUncond = empty(ch * L); dispatchCopy(src: vB, dst: vUncond, size: ch * L, srcOff: ch * L, dstOff: 0)
        let nx = empty(ch * L)
        dispatchCfgEuler(x: xBuf, vCond: vCond, vUncond: vUncond, out: nx,
                         size: ch * L, w1: 4.0, w2: 3.0, invTotal: 1.0 / totalStep)
        let out = readBuf(nx, count: ch * L)
        if curStep == 0.0 { printArrStats("vfield[0] step0", out) }
        return out
    }

    // Batched vector estimator: processes the conditional (b=0) and unconditional (b=1)
    // CFG passes together as a batch of 2, so the ConvNeXt matmuls run at M = 2·L and the
    // GPU stays occupied. The ConvNeXt backbone and time conditioning are batch-parallel;
    // each cross-attention is applied per-branch with its own conditioning (the only place
    // the two passes differ). Returns velocities [2, 144, L] channel-major.
    private func vfVelocityBatched(x: MTLBuffer, L: Int, tEmbBuf: MTLBuffer, T: Int, cond: VfCond) throws -> MTLBuffer {
        let D = config.vfDim, ch = config.latentCh, B = 2
        let p = "vector_estimator.tts.ttl.vector_field.main_blocks"
        // batched input: the ODE state is shared, so duplicate x into both batch slots.
        let xB = empty(B * ch * L)
        dispatchCopy(src: x, dst: xB, size: ch * L, srcOff: 0, dstOff: 0)
        dispatchCopy(src: x, dst: xB, size: ch * L, srcOff: 0, dstOff: ch * L)
        var h = try conv1x1Batched(ns: "vf", prefix: "vector_estimator.tts.ttl.vector_field.proj_in.net",
                                   x: xB, inCh: ch, outCh: D, L: L, batch: B)
        for g in 0..<config.vfSuperBlocks {
            let base = 6 * g
            for j in 0..<4 {
                h = try convNextLayerBatched(ns: "vf", prefix: "\(p).\(base+0).convnext.\(j)",
                                             x: h, C: D, T: L, ksz: config.vfConvNextKsz,
                                             dil: [1,2,4,8][j], inter: config.vfConvNextInter, batch: B)
            }
            h = try addTimeCondGPUBatched(h: h, tEmbBuf: tEmbBuf, D: D, L: L,
                                          prefix: "\(p).\(base+1).linear.linear", batch: B)
            h = try convNextLayerBatched(ns: "vf", prefix: "\(p).\(base+2).convnext.0",
                                         x: h, C: D, T: L, ksz: config.vfConvNextKsz, dil: 1,
                                         inter: config.vfConvNextInter, batch: B)
            // RoPE cross-attention to text (+ norm), per branch
            h = try vfAttnBatched(h, L: L, B: B) { hb, b in
                try self.vfRopeCrossAttnTextGPU(h: hb, L: L, textR: b == 0 ? cond.textR_cond : cond.textR_null,
                                                T: T, theta: cond.theta,
                                                attnPrefix: "\(p).\(base+3).attn", normPrefix: "\(p).\(base+3).norm.norm")
            }
            h = try convNextLayerBatched(ns: "vf", prefix: "\(p).\(base+4).convnext.0",
                                         x: h, C: D, T: L, ksz: config.vfConvNextKsz, dil: 1,
                                         inter: config.vfConvNextInter, batch: B)
            // tanh cross-attention to style (+ norm), per branch
            h = try vfAttnBatched(h, L: L, B: B) { hb, b in
                try self.vfTanhCrossAttnStyleGPU(h: hb, L: L,
                                                 styleK: b == 0 ? cond.sK_cond : cond.sK_null,
                                                 styleV: b == 0 ? cond.sV_cond : cond.sV_null,
                                                 S: b == 0 ? cond.S_cond : cond.S_null,
                                                 attnPrefix: "\(p).\(base+5).attention", normPrefix: "\(p).\(base+5).norm.norm")
            }
        }
        for j in 0..<config.vfLastConvNextLayers {
            h = try convNextLayerBatched(ns: "vf", prefix: "vector_estimator.tts.ttl.vector_field.last_convnext.convnext.\(j)",
                                         x: h, C: D, T: L, ksz: config.vfConvNextKsz, dil: 1,
                                         inter: config.vfConvNextInter, batch: B)
        }
        return try conv1x1Batched(ns: "vf", prefix: "vector_estimator.tts.ttl.vector_field.proj_out.net",
                                  x: h, inCh: D, outCh: ch, L: L, batch: B)
    }

    // Apply a per-branch cross-attention to a batched channel-major tensor [B, C, L]:
    // slice each batch's [C, L], run the (single-batch) GPU attention with its own
    // conditioning, and scatter the result back. Attention is the only per-branch step.
    private func vfAttnBatched(_ h: MTLBuffer, L: Int, B: Int,
                               _ attend: (MTLBuffer, Int) throws -> MTLBuffer) rethrows -> MTLBuffer {
        let C = config.vfDim
        let out = empty(B * C * L)
        for b in 0..<B {
            let hb = empty(C * L)
            dispatchCopy(src: h, dst: hb, size: C * L, srcOff: b * C * L, dstOff: 0)
            let ob = try attend(hb, b)
            dispatchCopy(src: ob, dst: out, size: C * L, srcOff: 0, dstOff: b * C * L)
        }
        return out
    }

    // Batched ConvNeXt block (channel-major [B, C, T], symmetric edge padding). Mirrors
    // convNextLayer but flattens to [B·T, C] for the LayerNorm + pointwise matmuls so both
    // CFG branches share one dispatch. The vector field always has a gamma scale.
    private func convNextLayerBatched(ns: String, prefix: String, x: MTLBuffer, C: Int, T: Int,
                                      ksz: Int, dil: Int, inter: Int, batch B: Int) throws -> MTLBuffer {
        let residual = x
        let pad = dil * (ksz - 1) / 2
        guard let dwW = weightBuf(ns, "\(prefix).dwconv.weight") ?? weightBuf(ns, "\(prefix).dwconv.net.weight"),
              let dwB = weightBuf(ns, "\(prefix).dwconv.bias") ?? weightBuf(ns, "\(prefix).dwconv.net.bias")
        else { return x }
        let dw = empty(B * C * T)
        dispatchEdgeDwConvBatched(input: x, weight: dwW, bias: dwB, out: dw, C: C, T: T, ksz: ksz, pad: pad, dil: dil, batch: B)
        let dwT = empty(B * T * C)                                   // [B·T, C] row-major
        dispatchTransposeBatched(input: dw, out: dwT, rows: C, cols: T, batch: B)
        let normed = empty(B * T * C)
        if let g = weightBuf(ns, "\(prefix).norm.norm.weight"), let bb = weightBuf(ns, "\(prefix).norm.norm.bias") {
            dispatchLayerNorm(input: dwT, gamma: g, beta: bb, out: normed, rows: B * T, C: C)
        }
        guard let w1 = weightBuf(ns, "\(prefix).pwconv1.weight"), let b1 = weightBuf(ns, "\(prefix).pwconv1.bias"),
              let w2 = weightBuf(ns, "\(prefix).pwconv2.weight"), let b2 = weightBuf(ns, "\(prefix).pwconv2.bias")
        else { return x }
        let hid = empty(B * T * inter)
        matmulRowMajorWT(A: normed, Wt: w1, bias: b1, out: hid, M: B * T, K: C, N: inter, act: "gelu")
        let proj = empty(B * T * C)
        matmulRowMajorWT(A: hid, Wt: w2, bias: b2, out: proj, M: B * T, K: inter, N: C)
        let out = empty(B * C * T)
        if let gamma = weightBuf(ns, "\(prefix).gamma") {
            dispatchGammaResidualBatched(h: proj, residual: residual, gamma: gamma, out: out, C: C, T: T, batch: B)
        } else {
            let projC = empty(B * C * T)
            dispatchTransposeBatched(input: proj, out: projC, rows: T, cols: C, batch: B)
            dispatchAdd(a: residual, b: projC, out: out, size: B * C * T)
        }
        return out
    }

    // Batched conv1x1 (pointwise): [B, inCh, L] → [B, outCh, L] via one [B·L, outCh] matmul.
    private func conv1x1Batched(ns: String, prefix: String, x: MTLBuffer, inCh: Int, outCh: Int, L: Int, batch B: Int) throws -> MTLBuffer {
        guard let w = weightBuf(ns, "\(prefix).weight") else { return x }
        let bias = weightBuf(ns, "\(prefix).bias")
        let xt = empty(B * L * inCh)                                 // [B·L, inCh]
        dispatchTransposeBatched(input: x, out: xt, rows: inCh, cols: L, batch: B)
        let outT = empty(B * L * outCh)
        matmulRowMajorWT(A: xt, Wt: w, bias: bias, out: outT, M: B * L, K: inCh, N: outCh)
        let out = empty(B * outCh * L)                              // [B, outCh, L]
        dispatchTransposeBatched(input: outT, out: out, rows: L, cols: outCh, batch: B)
        return out
    }

    // Batched GPU time conditioning: h[b,c,l] += (W·tEmb + b)[c] (same proj for both branches).
    private func addTimeCondGPUBatched(h: MTLBuffer, tEmbBuf: MTLBuffer, D: Int, L: Int, prefix: String, batch B: Int) throws -> MTLBuffer {
        guard let w = weightBuf("vf", "\(prefix).weight") else { return h }   // stored [64,512] (in,out)
        let b = weightBuf("vf", "\(prefix).bias")
        let projBuf = empty(D)
        matmulRowMajor(A: tEmbBuf, B: w, bias: b, out: projBuf, M: 1, K: 64, N: D)
        let out = empty(B * D * L)
        dispatchAddColBatched(x: h, vec: projBuf, out: out, C: D, L: L, batch: B)
        return out
    }

    // MARK: - Stage: vocoder

    // latent[144,L] → denorm → unfold 144→(24×6L) → AE decoder (10 ConvNeXt) → head → wav
    private func runVocoder(latent: MTLBuffer, L: Int) throws -> [Float] {
        let ld = config.latentDim, ccf = config.chunkCompress
        let subL = ccf * L
        // ONNX node[0]: latent / normalizer.scale (scalar 0.25). Folded into the
        // per-channel denorm below since it commutes with the reshape.
        var scaleVal: Float = 1
        if let s = weightBuf("vo", "tts.ttl.normalizer.scale") { scaleVal = readBuf(s, count: 1).first ?? 1 }
        // unfold [144,L] → [24,6L]
        var x = try unfoldLatent(latent: latent, latentDim: ld, ccf: ccf, L: L)
        // denorm: (x / scale) * latent_std + latent_mean  =  x * (std/scale) + mean
        if let mean = weightBuf("vo", "latent_mean"), let std = weightBuf("vo", "latent_std") {
            let stdScaled = readBuf(std, count: ld).map { $0 / scaleVal }
            x = try denorm(x: x, mean: mean, std: makeBuf(stdScaled), C: ld, T: subL)
        }
        // embed conv 24→512, k7 (weight name is opaque onnx::Conv_NNNN → find by shape)
        let hdim = config.aeHdim
        guard let ewName = weightNameByShape("vo", [hdim, ld, 7]), let ew = weightBuf("vo", ewName) else {
            throw err("vocoder embed conv [512,24,7] not found")
        }
        var eb: MTLBuffer? = nil
        if let num = Int(ewName.split(separator: "_").last ?? "") { eb = weightBuf("vo", "onnx::Conv_\(num + 1)") }
        var h = empty(hdim * subL)
        dispatchCausalConv1d(input: x, weight: ew, bias: eb, out: h, inCh: ld, outCh: hdim, ksz: 7, L: subL, dil: 1)
        // 10 ConvNeXt (causal + edge-padded)
        for i in 0..<config.aeDecLayers {
            h = try convNextLayer(ns: "vo", prefix: "tts.ae.decoder.convnext.\(i)",
                                  x: h, C: hdim, T: subL, ksz: config.aeDecKsz,
                                  dil: config.aeDecDil[i], inter: config.aeDecInter, causal: true)
        }
        h = try applyBatchNorm(ns: "vo", x: h, C: hdim, T: subL)
        let wav = try vocoderHead(ns: "vo", h: h, hdim: hdim, T: subL)
        return wav
    }

    private func weightNameByShape(_ ns: String, _ shape: [Int]) -> String? {
        W[ns]?.first(where: { $0.value.shape == shape })?.key
    }

    // MARK: - ConvNeXt building block (shared by all models)

    // x is channel-major [C, T]. dwconv(k) → LayerNorm(channel) → pwconv1 C→I → GELU
    // → pwconv2 I→C → gamma·h + residual.
    private func convNextLayer(ns: String, prefix: String, x: MTLBuffer, C: Int, T: Int,
                               ksz: Int, dil: Int, inter: Int, causal: Bool = false,
                               symEdge: Bool = false) throws -> MTLBuffer {
        let residual = x
        // depthwise conv padding modes:
        //  - AE decoder (vocoder): causal + edge (pad_left=dil*(k-1), replicate)
        //  - vector_field:         symmetric + edge (replicate) padding
        //  - text/TTL:             symmetric zero padding
        let pad = dil * (ksz - 1) / 2
        let dw = empty(C * T)
        guard let dwW = weightBuf(ns, "\(prefix).dwconv.weight") ?? weightBuf(ns, "\(prefix).dwconv.net.weight"),
              let dwB = weightBuf(ns, "\(prefix).dwconv.bias") ?? weightBuf(ns, "\(prefix).dwconv.net.bias")
        else { return x }
        if causal {
            dispatchCausalDwConv(input: x, weight: dwW, bias: dwB, out: dw, C: C, L: T, ksz: ksz, dil: dil)
        } else if symEdge {
            dispatchEdgeDwConv(input: x, weight: dwW, bias: dwB, out: dw, C: C, T: T, ksz: ksz, pad: pad, dil: dil)
        } else {
            dispatchDwConv(input: x, weight: dwW, bias: dwB, out: dw, C: C, T: T, ksz: ksz, pad: pad, dil: dil)
        }
        // LayerNorm over channels: work row-major [T, C]
        let dwT = try toRowMajor(dw, rows: C, cols: T)   // [T, C]
        let normed = empty(T * C)
        if let g = weightBuf(ns, "\(prefix).norm.norm.weight"), let b = weightBuf(ns, "\(prefix).norm.norm.bias") {
            dispatchLayerNorm(input: dwT, gamma: g, beta: b, out: normed, rows: T, C: C)
        }
        // pwconv1 C→I (+GELU), pwconv2 I→C   (pointwise = row-major matmul over C)
        guard let w1 = weightBuf(ns, "\(prefix).pwconv1.weight"), let b1 = weightBuf(ns, "\(prefix).pwconv1.bias"),
              let w2 = weightBuf(ns, "\(prefix).pwconv2.weight"), let b2 = weightBuf(ns, "\(prefix).pwconv2.bias")
        else { return x }
        // weights are [I, C, 1] and [C, I, 1]; interpret as [I,C] and [C,I], transpose for row-major matmul
        let hid = empty(T * inter)
        matmulRowMajorWT(A: normed, Wt: w1, bias: b1, out: hid, M: T, K: C, N: inter, act: "gelu")
        let proj = empty(T * C)
        matmulRowMajorWT(A: hid, Wt: w2, bias: b2, out: proj, M: T, K: inter, N: C)
        // gamma · h[T,C] + residual[C,T]  → [C, T]
        let out = empty(C * T)
        if let gamma = weightBuf(ns, "\(prefix).gamma") {
            dispatchGammaResidual(h: proj, residual: residual, gamma: gamma, out: out, C: C, T: T)
        } else {
            // no gamma: transpose proj back and add
            let projC = try toChannelMajor(proj, rows: T, cols: C)
            dispatchAdd(a: residual, b: projC, out: out, size: C * T)
        }
        return out
    }

    // MARK: - Attention blocks

    // VITS relative-position self-attention + FFN block (row-major [T, C])
    private func relAttnBlock(ns: String, layer: Int, x: MTLBuffer, T: Int, C: Int,
                             heads: Int, filter: Int) throws -> MTLBuffer {
        // q/k/v via conv_q/k/v (1x1 conv == linear). Falls back to identity if weights absent.
        let hd = C / heads
        func proj(_ which: String) -> MTLBuffer {
            guard let w = weightBuf(ns, "attn_encoder.attn_layers.\(layer).conv_\(which).weight") else { return x }
            let b = weightBuf(ns, "attn_encoder.attn_layers.\(layer).conv_\(which).bias")
            let out = empty(T * C)
            matmulRowMajorWT(A: x, Wt: w, bias: b, out: out, M: T, K: C, N: C)
            return out
        }
        let q = proj("q"), k = proj("k"), v = proj("v")
        if layer == 3 {
            printBufStats("layer 3 q", q, count: T * C)
            printBufStats("layer 3 k", k, count: T * C)
            printBufStats("layer 3 v", v, count: T * C)
        }
        let attn = empty(T * C)
        let relK = weightBuf(ns, "attn_encoder.attn_layers.\(layer).emb_rel_k")
        let relV = weightBuf(ns, "attn_encoder.attn_layers.\(layer).emb_rel_v")
        if let relK = relK, let relV = relV {
            dispatchRelAttn(Q: q, K: k, V: v, relK: relK, relV: relV, out: attn,
                            T: T, heads: heads, hd: hd, window: config.teAttnWindow,
                            scale: 1.0 / Float(hd).squareRoot())
        } else {
            dispatchMHA(Q: q, K: k, V: v, mask: nil, out: attn, Lq: T, Lkv: T, heads: heads, hd: hd,
                        scale: 1.0 / Float(hd).squareRoot())
        }
        if layer == 3 {
            printBufStats("layer 3 attn", attn, count: T * C)
        }
        // out proj conv_o
        var o = attn
        if let wo = weightBuf(ns, "attn_encoder.attn_layers.\(layer).conv_o.weight") {
            let bo = weightBuf(ns, "attn_encoder.attn_layers.\(layer).conv_o.bias")
            let out = empty(T * C)
            matmulRowMajorWT(A: attn, Wt: wo, bias: bo, out: out, M: T, K: C, N: C)
            o = out
        }
        if layer == 3 {
            printBufStats("layer 3 o", o, count: T * C)
        }
        // residual + norm_1
        var h = empty(T * C)
        dispatchAdd(a: x, b: o, out: h, size: T * C)
        h = try applyNorm(ns: ns, name: "attn_encoder.norm_layers_1.\(layer).norm", x: h, rows: T, C: C)
        if layer == 3 {
            printBufStats("layer 3 h (post norm_1)", h, count: T * C)
        }
        // FFN: conv_1 C→filter (relu) → conv_2 filter→C
        var ff = h
        if let w1 = weightBuf(ns, "attn_encoder.ffn_layers.\(layer).conv_1.weight"),
           let w2 = weightBuf(ns, "attn_encoder.ffn_layers.\(layer).conv_2.weight") {
            let b1 = weightBuf(ns, "attn_encoder.ffn_layers.\(layer).conv_1.bias")
            let b2 = weightBuf(ns, "attn_encoder.ffn_layers.\(layer).conv_2.bias")
            if layer == 3 {
                printBufStats("layer 3 w1 weight", w1, count: filter * C)
                if let b1 = b1 { printBufStats("layer 3 b1 bias", b1, count: filter) }
                printBufStats("layer 3 w2 weight", w2, count: filter * C)
                if let b2 = b2 { printBufStats("layer 3 b2 bias", b2, count: C) }
            }
            let hid = empty(T * filter)
            matmulRowMajorWT(A: h, Wt: w1, bias: b1, out: hid, M: T, K: C, N: filter, act: "relu")
            if layer == 3 {
                saveBin("/tmp/layer3_h.bin", h, count: T * C)
                printBufStats("layer 3 FFN hid", hid, count: T * filter)
                saveBin("/tmp/layer3_hid.bin", hid, count: T * filter)
            }
            let out = empty(T * C)
            matmulRowMajorWT(A: hid, Wt: w2, bias: b2, out: out, M: T, K: filter, N: C)
            ff = out
        }
        if layer == 3 {
            printBufStats("layer 3 ff", ff, count: T * C)
            saveBin("/tmp/layer3_ff.bin", ff, count: T * C)
        }
        var h2 = empty(T * C)
        dispatchAdd(a: h, b: ff, out: h2, size: T * C)
        if layer == 3 {
            printBufStats("layer 3 h2 before norm_2", h2, count: T * C)
            if let g = weightBuf(ns, "attn_encoder.norm_layers_2.3.norm.weight"),
               let b = weightBuf(ns, "attn_encoder.norm_layers_2.3.norm.bias") {
                printBufStats("layer 3 norm_layers_2.3.norm gamma", g, count: C)
                printBufStats("layer 3 norm_layers_2.3.norm beta", b, count: C)
            }
        }
        h2 = try applyNorm(ns: ns, name: "attn_encoder.norm_layers_2.\(layer).norm", x: h2, rows: T, C: C)
        return h2
    }

    // MARK: - Vector-field time embedding + attentions (CPU, exact ONNX port)

    // time_encoder: sinusoid(t) [64] → Gemm→256 → Mish → Gemm→64.
    //   sinusoid: arg = (cur/total)·1000·theta[f]; emb = concat(sin, cos) over 32 bands.
    //   Mish(x) = x·tanh(softplus(x)).
    private func timeEmbeddingArr(cur: Float, total: Float) throws -> [Float] {
        let t = cur / max(total, 1)
        // frequency bands (Constant_3), fixed [32]
        let freqs: [Float] = [1.0, 0.74296391, 0.55199546, 0.41011271, 0.30469894, 0.22638035,
            0.16819243, 0.12496091, 0.092841454, 0.068977855, 0.051248062, 0.038075458,
            0.028288694, 0.021017481, 0.015615228, 0.011601552, 0.0086195357, 0.0064040045,
            0.0047579445, 0.0035349815, 0.0026263641, 0.0019512930, 0.0014497405, 0.0010771049,
            0.00080025021, 0.00059455709, 0.00044173450, 0.00032819266, 0.00024383534,
            0.00018116087, 0.00013459600, 0.000099999990]
        var sinu = [Float](repeating: 0, count: 64)
        for f in 0..<32 {
            let ang = t * 1000.0 * freqs[f]
            sinu[f] = sin(ang); sinu[32 + f] = cos(ang)
        }
        guard let w0 = rawWeight("vf", "vector_estimator.tts.ttl.vector_field.time_encoder.mlp.0.linear.weight"),
              let b0 = rawWeight("vf", "vector_estimator.tts.ttl.vector_field.time_encoder.mlp.0.linear.bias"),
              let w2 = rawWeight("vf", "vector_estimator.tts.ttl.vector_field.time_encoder.mlp.2.linear.weight"),
              let b2 = rawWeight("vf", "vector_estimator.tts.ttl.vector_field.time_encoder.mlp.2.linear.bias")
        else { throw err("time_encoder weights missing") }
        // Gemm uses Linear weights [out,in] → out = W·x + b (row of W dotted with x)
        func linear(_ x: [Float], _ w: [Float], _ b: [Float], inN: Int, outN: Int) -> [Float] {
            var o = [Float](repeating: 0, count: outN)
            for j in 0..<outN { var a = b[j]; for i in 0..<inN { a += w[j * inN + i] * x[i] }; o[j] = a }
            return o
        }
        let g0 = linear(sinu, w0, b0, inN: 64, outN: 256)             // [256]
        var mish = [Float](repeating: 0, count: 256)
        for i in 0..<256 { let sp = log(1 + exp(g0[i])); mish[i] = g0[i] * tanh(sp) }
        return linear(mish, w2, b2, inN: 256, outN: 64)               // [64]
    }

    // ── GPU cross-attentions (projections + RoPE/tanh + softmax + out-proj + norm) ──
    // Fully on-device: the Q/K/V/out projections are GPU matmuls, RoPE/tanh are GPU
    // kernels, the attention core reuses st_mha_kernel, and (O+residual)→LayerNorm→
    // transpose stays on GPU. Removes the per-block CPU round-trips (readBuf/makeBuf)
    // that stalled the flow-matching pipeline. Weights are onnx::MatMul [in,out], used
    // directly as the B operand (no transpose) — mirrors the CPU cpuMatmul convention.

    // RoPE cross-attention to text. h[512,L] CM, textR[T,256] row-major → [512,L] CM.
    private func vfRopeCrossAttnTextGPU(h: MTLBuffer, L: Int, textR: MTLBuffer, T: Int,
                                        theta: MTLBuffer, attnPrefix: String, normPrefix: String) throws -> MTLBuffer {
        let C = config.vfDim, Ctext = config.charEmb, heads = 8, hd = 64
        let hR = try toRowMajor(h, rows: C, cols: L)            // [L,512]
        func wb(_ s: String) -> MTLBuffer? { weightBuf("vf", "\(attnPrefix).\(s)") }
        guard let Wq = wb("W_query.linear.weight"), let Wk = wb("W_key.linear.weight"),
              let Wv = wb("W_value.linear.weight"), let Wo = wb("out_fc.linear.weight")
        else { throw err("vf rope attn weights missing (\(attnPrefix))") }
        let bq = wb("W_query.linear.bias"), bk = wb("W_key.linear.bias")
        let bv = wb("W_value.linear.bias"), bo = wb("out_fc.linear.bias")
        let Q = empty(L * C); matmulRowMajor(A: hR, B: Wq, bias: bq, out: Q, M: L, K: C, N: C)
        let K = empty(T * C); matmulRowMajor(A: textR, B: Wk, bias: bk, out: K, M: T, K: Ctext, N: C)
        let V = empty(T * C); matmulRowMajor(A: textR, B: Wv, bias: bv, out: V, M: T, K: Ctext, N: C)
        let Qr = empty(L * C); dispatchRopeNorm(x: Q, out: Qr, theta: theta, n: L, len: L, heads: heads, hd: hd)
        let Kr = empty(T * C); dispatchRopeNorm(x: K, out: Kr, theta: theta, n: T, len: T, heads: heads, hd: hd)
        let ctx = empty(L * C)
        dispatchMHA(Q: Qr, K: Kr, V: V, mask: nil, out: ctx, Lq: L, Lkv: T, heads: heads, hd: hd, scale: 1.0 / 16.0)
        let O = empty(L * C); matmulRowMajor(A: ctx, B: Wo, bias: bo, out: O, M: L, K: C, N: C)
        return try residualNormGPU(O: O, residualR: hR, L: L, C: C, normPrefix: normPrefix)
    }

    // tanh cross-attention to style. h[512,L] CM, styleK/styleV[50,256] → [512,L] CM.
    private func vfTanhCrossAttnStyleGPU(h: MTLBuffer, L: Int, styleK: MTLBuffer, styleV: MTLBuffer,
                                         S: Int, attnPrefix: String, normPrefix: String) throws -> MTLBuffer {
        let C = config.vfDim, Dqk = 256, heads = 2, hd = 128
        let hR = try toRowMajor(h, rows: C, cols: L)            // [L,512]
        func wb(_ s: String) -> MTLBuffer? { weightBuf("vf", "\(attnPrefix).\(s)") }
        guard let Wq = wb("W_query.linear.weight"), let Wk = wb("W_key.linear.weight"),
              let Wv = wb("W_value.linear.weight"), let Wo = wb("out_fc.linear.weight")
        else { throw err("vf tanh attn weights missing (\(attnPrefix))") }
        let bq = wb("W_query.linear.bias"), bk = wb("W_key.linear.bias")
        let bv = wb("W_value.linear.bias"), bo = wb("out_fc.linear.bias")
        let Q = empty(L * Dqk); matmulRowMajor(A: hR, B: Wq, bias: bq, out: Q, M: L, K: C, N: Dqk)
        let K = empty(S * Dqk); matmulRowMajor(A: styleK, B: Wk, bias: bk, out: K, M: S, K: Dqk, N: Dqk)
        let V = empty(S * Dqk); matmulRowMajor(A: styleV, B: Wv, bias: bv, out: V, M: S, K: Dqk, N: Dqk)
        let Kt = empty(S * Dqk); dispatchTanh(input: K, out: Kt, size: S * Dqk)   // keys through tanh
        let ctx = empty(L * Dqk)
        dispatchMHA(Q: Q, K: Kt, V: V, mask: nil, out: ctx, Lq: L, Lkv: S, heads: heads, hd: hd, scale: 1.0 / 16.0)
        let O = empty(L * C); matmulRowMajor(A: ctx, B: Wo, bias: bo, out: O, M: L, K: Dqk, N: C)
        return try residualNormGPU(O: O, residualR: hR, L: L, C: C, normPrefix: normPrefix)
    }

    // (O + residualR) row-major [L,C] → LayerNorm over C (eps 1e-6) → channel-major [C,L], all GPU.
    private func residualNormGPU(O: MTLBuffer, residualR: MTLBuffer, L: Int, C: Int, normPrefix: String) throws -> MTLBuffer {
        let add = empty(L * C)
        dispatchAdd(a: O, b: residualR, out: add, size: L * C)
        let normed = try applyNorm(ns: "vf", name: normPrefix, x: add, rows: L, C: C)   // [L,C]
        return try toChannelMajor(normed, rows: L, cols: C)                              // [C,L]
    }


    private func printArrStats(_ label: String, _ a: [Float]) {
        guard !a.isEmpty else { return }
        var mn = a[0], mx = a[0], sum: Float = 0, sq: Float = 0
        for v in a { mn = min(mn, v); mx = max(mx, v); sum += v; sq += v * v }
        let mean = sum / Float(a.count)
        let std = (sq / Float(a.count) - mean * mean).squareRoot()
        print(String(format: "[Supertonic] %@ std=%.4f min=%.3f max=%.3f", label, std, mn, mx))
    }

    private func unfoldLatent(latent: MTLBuffer, latentDim: Int, ccf: Int, L: Int) throws -> MTLBuffer {
        // [144, L] → [24, 6L]: channel c in 0..24, subframe s in 0..6 → time s + ccf*l.  VALIDATE
        // Reuse transpose/permute conventions; here do a CPU gather for clarity (small).
        let src = readBuf(latent, count: latentDim * ccf * L)
        var out = [Float](repeating: 0, count: latentDim * ccf * L)
        for c in 0..<latentDim {
            for s in 0..<ccf {
                let inCh = c * ccf + s
                for l in 0..<L {
                    out[c * (ccf * L) + (l * ccf + s)] = src[inCh * L + l]
                }
            }
        }
        return makeBuf(out)
    }

    private func denorm(x: MTLBuffer, mean: MTLBuffer, std: MTLBuffer, C: Int, T: Int) throws -> MTLBuffer {
        // x * std + mean, per channel. Reuse FiLM (scale=std, shift=mean, add_one=0).
        let out = empty(C * T)
        dispatchFilm(x: x, scale: std, shift: mean, out: out, C: C, T: T, addOne: false)
        return out
    }

    private func convInit(ns: String, x: MTLBuffer, inCh: Int, outCh: Int, L: Int, prefix: String?) throws -> MTLBuffer {
        // Initial conv (kernel 7) mapping latentDim→hdim.
        guard let p = prefix else { return x }
        return try conv1D(ns: ns, prefix: p, x: x, inCh: inCh, outCh: outCh, T: L, ksz: 7, pad: 3, dil: 1)
    }

    private func applyBatchNorm(ns: String, x: MTLBuffer, C: Int, T: Int) throws -> MTLBuffer {
        guard let w = W[ns]?["tts.ae.decoder.final_norm.norm.weight"],
              let b = W[ns]?["tts.ae.decoder.final_norm.norm.bias"],
              let mean = W[ns]?["tts.ae.decoder.final_norm.norm.running_mean"],
              let var_ = W[ns]?["tts.ae.decoder.final_norm.norm.running_var"]
        else { return x }

        let w_arr = readBuf(w.buf, count: C)
        let b_arr = readBuf(b.buf, count: C)
        let mean_arr = readBuf(mean.buf, count: C)
        let var_arr = readBuf(var_.buf, count: C)

        var scale = [Float](repeating: 0, count: C)
        var shift = [Float](repeating: 0, count: C)
        let eps: Float = 1e-5

        for c in 0..<C {
            let s = w_arr[c] / sqrt(var_arr[c] + eps)
            scale[c] = s
            shift[c] = b_arr[c] - mean_arr[c] * s
        }

        let scaleBuf = makeBuf(scale)
        let shiftBuf = makeBuf(shift)
        let out = empty(C * T)
        dispatchFilm(x: x, scale: scaleBuf, shift: shiftBuf, out: out, C: C, T: T, addOne: false)
        return out
    }

    private func dispatchConv1d(input: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer?, out: MTLBuffer,
                                inCh: Int, outCh: Int, T: Int, ksz: Int, pad: Int, dil: Int) {
        struct Params {
            var in_channels: UInt32
            var out_channels: UInt32
            var kernel_size: UInt32
            var input_length: UInt32
            var output_length: UInt32
            var padding: UInt32
            var stride: UInt32
            var dilation: UInt32
            var use_bias: UInt32
        }
        var p = Params(in_channels: UInt32(inCh), out_channels: UInt32(outCh), kernel_size: UInt32(ksz),
                       input_length: UInt32(T), output_length: UInt32(T), padding: UInt32(pad),
                       stride: 1, dilation: UInt32(dil), use_bias: bias != nil ? 1 : 0)
        dispatch1d("conv1d_kernel", [input, weight, bias ?? emptyMask, out], &p, MemoryLayout<Params>.size, outCh * T)
    }

    private func dispatchPrelu(input: MTLBuffer, slope: MTLBuffer, out: MTLBuffer, C: Int, T: Int) {
        struct Params { var channels: UInt32; var length: UInt32; var slopeShared: UInt32 }
        var p = Params(channels: UInt32(C), length: UInt32(T), slopeShared: slope.length == 4 ? 1 : 0)
        dispatch1d("reuse_prelu_kernel", [input, slope, out], &p, MemoryLayout<Params>.size, C * T)
    }

    private func vocoderHead(ns: String, h: MTLBuffer, hdim: Int, T: Int) throws -> [Float] {
        guard let w1 = weightBuf(ns, "tts.ae.decoder.head.layer1.net.weight"),
              let b1 = weightBuf(ns, "tts.ae.decoder.head.layer1.net.bias"),
              let slope = weightBuf(ns, "onnx::PRelu_1506"),
              let w2 = weightBuf(ns, "tts.ae.decoder.head.layer2.weight")
        else { throw err("missing vocoder head weights") }

        let h1 = empty(2048 * T)
        dispatchCausalConv1d(input: h, weight: w1, bias: b1, out: h1, inCh: hdim, outCh: 2048, ksz: 3, L: T, dil: 1)

        let h1_act = empty(2048 * T)
        dispatchPrelu(input: h1, slope: slope, out: h1_act, C: 2048, T: T)

        let h2 = empty(512 * T)
        dispatchCausalConv1d(input: h1_act, weight: w2, bias: nil, out: h2, inCh: 2048, outCh: 512, ksz: 1, L: T, dil: 1)

        let wavBuf = empty(512 * T)
        dispatchTranspose(input: h2, out: wavBuf, rows: 512, cols: T)

        return readBuf(wavBuf, count: 512 * T)
    }

    // MARK: - conv1x1 / embed / layout helpers

    private func conv1x1(ns: String, prefix: String, x: MTLBuffer, inCh: Int, outCh: Int, L: Int) throws -> MTLBuffer {
        // weight [outCh, inCh, 1] → treat as [outCh, inCh]. Compute row-major over L:
        // out[L, outCh] = x^T[L, inCh] · W^T ; then to channel-major.
        guard let w = weightBuf(ns, "\(prefix).weight") else { return x }
        let b = weightBuf(ns, "\(prefix).bias")
        let xt = try toRowMajor(x, rows: inCh, cols: L)     // [L, inCh]
        let outT = empty(L * outCh)
        matmulRowMajorWT(A: xt, Wt: w, bias: b, out: outT, M: L, K: inCh, N: outCh)
        return try toChannelMajor(outT, rows: L, cols: outCh)
    }

    private func conv1D(ns: String, prefix: String, x: MTLBuffer, inCh: Int, outCh: Int, T: Int,
                        ksz: Int, pad: Int, dil: Int = 1) throws -> MTLBuffer {
        guard let w = weightBuf(ns, "\(prefix).weight") else { return x }
        let b = weightBuf(ns, "\(prefix).bias")
        let out = empty(outCh * T)
        dispatchConv1d(input: x, weight: w, bias: b, out: out, inCh: inCh, outCh: outCh, T: T, ksz: ksz, pad: pad, dil: dil)
        return out
    }

    private func embedChannelMajor(ns: String, embName: String?, ids: MTLBuffer, T: Int, C: Int) throws -> MTLBuffer {
        guard let name = embName, let emb = W[ns]![name] else { return empty(C * T) }
        let table = readBuf(emb.buf, count: emb.count)      // [V, C]
        let idArr = readIntBuf(ids, count: T)
        var out = [Float](repeating: 0, count: C * T)       // channel-major [C, T]
        for t in 0..<T {
            let id = Int(idArr[t])
            for c in 0..<C { out[c * T + t] = table[id * C + c] }
        }
        return makeBuf(out)
    }

    private func applyNorm(ns: String, name: String, x: MTLBuffer, rows: Int, C: Int) throws -> MTLBuffer {
        guard let g = weightBuf(ns, "\(name).weight"), let b = weightBuf(ns, "\(name).bias") else { return x }
        let out = empty(rows * C)
        dispatchLayerNorm(input: x, gamma: g, beta: b, out: out, rows: rows, C: C)
        return out
    }

    private func applyNormChannelMajor(ns: String, name: String, x: MTLBuffer, C: Int, L: Int) throws -> MTLBuffer {
        guard weightBuf(ns, "\(name).weight") != nil else { return x }
        let rowMajor = try toRowMajor(x, rows: C, cols: L)
        let normed = try applyNorm(ns: ns, name: name, x: rowMajor, rows: L, C: C)
        return try toChannelMajor(normed, rows: L, cols: C)
    }

    // MARK: - Weight lookup

    private func weightBuf(_ ns: String, _ contains: String) -> MTLBuffer? {
        if let w = W[ns]?[contains] { return w.buf }

        // Resolve via alias map
        let normalized = contains
            .replacingOccurrences(of: "vector_estimator.tts.ttl.", with: "")
            .replacingOccurrences(of: "tts.ttl.", with: "")
            .replacingOccurrences(of: "speech_prompted_text_encoder.", with: "")
            .replacingOccurrences(of: "tts.ae.", with: "")

        if let alias = weightAliasMap[normalized] {
            if let w = W[ns]?[alias] { return w.buf }
        }

        if normalized.hasSuffix(".weight") {
            let matmulKey = normalized.replacingOccurrences(of: ".weight", with: ".MatMul.weight")
            if let alias = weightAliasMap[matmulKey] {
                if let w = W[ns]?[alias] { return w.buf }
            }
        }

        // suffix / substring match
        if let hit = W[ns]?.first(where: { $0.key.hasSuffix(contains) || $0.key.contains(contains) }) {
            print("[WeightBuf] matched '\(contains)' to '\(hit.key)'")
            return hit.value.buf
        }
        return nil
    }
    private func findWeight(_ ns: String, contains: String, and: String? = nil) -> String? {
        W[ns]?.keys.first(where: { $0.contains(contains) && (and == nil || $0.contains(and!)) })
    }
    private func convnextPrefix(_ ns: String, _ p: String) -> String {
        // find a key prefix that resolves; fall back to given
        if W[ns]?.keys.contains(where: { $0.hasPrefix(p) }) == true { return p }
        return p
    }

    // MARK: - Dispatch helpers

    private func dispatchDwConv(input: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, out: MTLBuffer,
                               C: Int, T: Int, ksz: Int, pad: Int, dil: Int) {
        struct P { var channels: UInt32; var length: UInt32; var ksize: UInt32; var pad: UInt32; var dilation: UInt32 }
        var p = P(channels: UInt32(C), length: UInt32(T), ksize: UInt32(ksz), pad: UInt32(pad), dilation: UInt32(dil))
        dispatch1d("st_dwconv1d_kernel", [input, weight, bias, out], &p, MemoryLayout<P>.size, C * T)
    }

    private func dispatchEdgeDwConv(input: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, out: MTLBuffer,
                                    C: Int, T: Int, ksz: Int, pad: Int, dil: Int) {
        struct P { var channels: UInt32; var length: UInt32; var ksize: UInt32; var pad: UInt32; var dilation: UInt32 }
        var p = P(channels: UInt32(C), length: UInt32(T), ksize: UInt32(ksz), pad: UInt32(pad), dilation: UInt32(dil))
        dispatch1d("st_dwconv1d_edge_kernel", [input, weight, bias, out], &p, MemoryLayout<P>.size, C * T)
    }

    private func dispatchLayerNorm(input: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, out: MTLBuffer, rows: Int, C: Int) {
        struct P { var batch: UInt32; var hidden: UInt32; var eps: Float }
        var p = P(batch: UInt32(rows), hidden: UInt32(C), eps: 1e-6)
        dispatch1d("layer_norm_kernel", [input, gamma, beta, out], &p, 12, rows)
    }

    private func dispatchGammaResidual(h: MTLBuffer, residual: MTLBuffer, gamma: MTLBuffer, out: MTLBuffer, C: Int, T: Int) {
        struct P { var dim: UInt32; var length: UInt32 }
        var p = P(dim: UInt32(C), length: UInt32(T))
        dispatch1d("lava_gamma_residual_kernel", [h, residual, gamma, out], &p, 8, C * T)
    }

    private func dispatchAdd(a: MTLBuffer, b: MTLBuffer, out: MTLBuffer, size: Int) {
        struct P { var size: UInt32 }
        var p = P(size: UInt32(size))
        dispatch1d("add_kernel", [a, b, out], &p, 4, size)
    }

    private struct CausalConvP { var in_ch: UInt32; var out_ch: UInt32; var ksz: UInt32; var length: UInt32; var dilation: UInt32; var use_bias: UInt32 }
    private func dispatchCausalConv1d(input: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer?, out: MTLBuffer,
                                      inCh: Int, outCh: Int, ksz: Int, L: Int, dil: Int) {
        var p = CausalConvP(in_ch: UInt32(inCh), out_ch: UInt32(outCh), ksz: UInt32(ksz),
                            length: UInt32(L), dilation: UInt32(dil), use_bias: bias != nil ? 1 : 0)
        dispatch1d("st_causal_conv1d_kernel", [input, weight, bias ?? emptyMask, out], &p,
                   MemoryLayout<CausalConvP>.size, outCh * L)
    }
    private func dispatchCausalDwConv(input: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, out: MTLBuffer,
                                      C: Int, L: Int, ksz: Int, dil: Int) {
        var p = CausalConvP(in_ch: UInt32(C), out_ch: UInt32(C), ksz: UInt32(ksz),
                            length: UInt32(L), dilation: UInt32(dil), use_bias: 1)
        dispatch1d("st_causal_dwconv1d_kernel", [input, weight, bias, out], &p,
                   MemoryLayout<CausalConvP>.size, C * L)
    }

    private func dispatchAddScale(a: MTLBuffer, b: MTLBuffer, out: MTLBuffer, size: Int, scaleA: Float, scaleB: Float) {
        // out = a + scaleB*b  (scaleA assumed 1). Reuse scale then add for generality.
        let sb = empty(size)
        struct SP { var size: UInt32; var _pad: UInt32; var scale: Float }
        var sp = SP(size: UInt32(size), _pad: 0, scale: scaleB)
        dispatch1d("scale_kernel", [b, sb], &sp, 12, size)
        dispatchAdd(a: a, b: sb, out: out, size: size)
    }

    private func dispatchFilm(x: MTLBuffer, scale: MTLBuffer, shift: MTLBuffer, out: MTLBuffer, C: Int, T: Int, addOne: Bool) {
        struct P { var channels: UInt32; var length: UInt32; var add_one: UInt32 }
        var p = P(channels: UInt32(C), length: UInt32(T), add_one: addOne ? 1 : 0)
        dispatch1d("st_film_kernel", [x, scale, shift, out], &p, 12, C * T)
    }

    private func dispatchRope(x: MTLBuffer, out: MTLBuffer, L: Int, heads: Int, hd: Int, base: Float) {
        struct P { var seq_len: UInt32; var num_heads: UInt32; var head_dim: UInt32; var base: Float }
        var p = P(seq_len: UInt32(L), num_heads: UInt32(heads), head_dim: UInt32(hd), base: base)
        dispatch1d("st_rope_kernel", [x, out], &p, 16, L * heads * (hd / 2))
    }

    private func dispatchMHA(Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, mask: MTLBuffer?, out: MTLBuffer,
                            Lq: Int, Lkv: Int, heads: Int, hd: Int, scale: Float) {
        struct P { var lq: UInt32; var lkv: UInt32; var num_heads: UInt32; var head_dim: UInt32; var scale: Float }
        var p = P(lq: UInt32(Lq), lkv: UInt32(Lkv), num_heads: UInt32(heads), head_dim: UInt32(hd), scale: scale)
        dispatch1d("st_mha_kernel", [Q, K, V, mask, out], &p, 20, Lq * heads * hd)
    }

    private func dispatchRelAttn(Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, relK: MTLBuffer, relV: MTLBuffer,
                                out: MTLBuffer, T: Int, heads: Int, hd: Int, window: Int, scale: Float) {
        struct P { var seq_len: UInt32; var num_heads: UInt32; var head_dim: UInt32; var window: UInt32; var scale: Float }
        var p = P(seq_len: UInt32(T), num_heads: UInt32(heads), head_dim: UInt32(hd), window: UInt32(window), scale: scale)
        dispatch1d("st_rel_attn_kernel", [Q, K, V, relK, relV, nil, out], &p, 20, T * heads * hd)
    }

    // Rotate-half RoPE with normalised position, applied to [n, heads*hd] row-major.
    private func dispatchRopeNorm(x: MTLBuffer, out: MTLBuffer, theta: MTLBuffer, n: Int, len: Int, heads: Int, hd: Int) {
        struct P { var n: UInt32; var len: UInt32; var num_heads: UInt32; var head_dim: UInt32 }
        var p = P(n: UInt32(n), len: UInt32(len), num_heads: UInt32(heads), head_dim: UInt32(hd))
        dispatch1d("st_rope_norm_kernel", [x, out, theta], &p, 16, n * heads * (hd / 2))
    }

    // Broadcast-add a length-C vector across L (channel-major [C, L]).
    private func dispatchAddCol(x: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, C: Int, L: Int) {
        struct P { var channels: UInt32; var length: UInt32; var add_one: UInt32 }
        var p = P(channels: UInt32(C), length: UInt32(L), add_one: 0)
        dispatch1d("st_add_col_kernel", [x, vec, out], &p, 12, C * L)
    }

    // CFG Euler combine on GPU: out = x + (w1·vCond − w2·vUncond)/total.
    private func dispatchCfgEuler(x: MTLBuffer, vCond: MTLBuffer, vUncond: MTLBuffer, out: MTLBuffer,
                                  size: Int, w1: Float, w2: Float, invTotal: Float) {
        struct P { var size: UInt32; var w1: Float; var w2: Float; var inv_total: Float }
        var p = P(size: UInt32(size), w1: w1, w2: w2, inv_total: invTotal)
        dispatch1d("st_cfg_euler_kernel", [x, vCond, vUncond, out], &p, 16, size)
    }

    // Elementwise tanh (reuses the existing tanh_kernel).
    private func dispatchTanh(input: MTLBuffer, out: MTLBuffer, size: Int) {
        struct P { var size: UInt32 }
        var p = P(size: UInt32(size))
        dispatch1d("tanh_kernel", [input, out], &p, 4, size)
    }

    // ── Batched (B) dispatch helpers (CFG cond+uncond processed together) ──
    private func dispatchCopy(src: MTLBuffer, dst: MTLBuffer, size: Int, srcOff: Int, dstOff: Int) {
        struct P { var size: UInt32; var src_off: UInt32; var dst_off: UInt32 }
        var p = P(size: UInt32(size), src_off: UInt32(srcOff), dst_off: UInt32(dstOff))
        dispatch1d("copy_kernel", [src, dst], &p, 12, size)
    }
    // Per-batch transpose: B blocks of [rows,cols] → [cols,rows].
    private func dispatchTransposeBatched(input: MTLBuffer, out: MTLBuffer, rows: Int, cols: Int, batch: Int) {
        struct P { var rows: UInt32; var cols: UInt32; var batch: UInt32 }
        var p = P(rows: UInt32(rows), cols: UInt32(cols), batch: UInt32(batch))
        dispatch1d("transpose_batched_kernel", [input, out], &p, 12, batch * rows * cols)
    }
    private func dispatchEdgeDwConvBatched(input: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, out: MTLBuffer,
                                           C: Int, T: Int, ksz: Int, pad: Int, dil: Int, batch: Int) {
        struct P { var channels: UInt32; var length: UInt32; var ksize: UInt32; var pad: UInt32; var dilation: UInt32; var batch: UInt32 }
        var p = P(channels: UInt32(C), length: UInt32(T), ksize: UInt32(ksz), pad: UInt32(pad), dilation: UInt32(dil), batch: UInt32(batch))
        dispatch1d("st_dwconv1d_edge_batched_kernel", [input, weight, bias, out], &p, MemoryLayout<P>.size, batch * C * T)
    }
    private func dispatchGammaResidualBatched(h: MTLBuffer, residual: MTLBuffer, gamma: MTLBuffer, out: MTLBuffer, C: Int, T: Int, batch: Int) {
        struct P { var dim: UInt32; var length: UInt32; var batch: UInt32 }
        var p = P(dim: UInt32(C), length: UInt32(T), batch: UInt32(batch))
        dispatch1d("lava_gamma_residual_batched_kernel", [h, residual, gamma, out], &p, 12, batch * C * T)
    }
    private func dispatchAddColBatched(x: MTLBuffer, vec: MTLBuffer, out: MTLBuffer, C: Int, L: Int, batch: Int) {
        struct P { var dim: UInt32; var length: UInt32; var batch: UInt32 }
        var p = P(dim: UInt32(C), length: UInt32(L), batch: UInt32(batch))
        dispatch1d("st_add_col_batched_kernel", [x, vec, out], &p, 12, batch * C * L)
    }

    private func dispatchSinusoid(t: MTLBuffer, out: MTLBuffer, rows: Int, dim: Int, maxPeriod: Float) {
        struct P { var rows: UInt32; var dim: UInt32; var max_period: Float }
        var p = P(rows: UInt32(rows), dim: UInt32(dim), max_period: maxPeriod)
        dispatch1d("st_sinusoid_kernel", [t, out], &p, 12, rows * (dim / 2))
    }

    // Row-major matmul: out[M,N] = A[M,K] · B[K,N] (+bias). B is already [K,N].
    // Uses the register-blocked GEMM (64×64 output tile, 4×4 per thread) with a
    // fused activation epilogue — several× the arithmetic intensity of the naive
    // one-output-per-thread kernel that dominated the flow-matching loop.
    private func matmulRowMajor(A: MTLBuffer, B: MTLBuffer, bias: MTLBuffer?, out: MTLBuffer, M: Int, K: Int, N: Int, act: String = "none") {
        struct P { var M: UInt32; var K: UInt32; var N: UInt32; var use_bias: UInt32 }
        var p = P(M: UInt32(M), K: UInt32(K), N: UInt32(N), use_bias: bias != nil ? 1 : 0)
        let name: String
        switch act {
        case "gelu": name = "reuse_matmul_gelu_kernel"
        case "relu": name = "reuse_matmul_relu_kernel"
        default: name = "reuse_matmul_kernel"
        }
        let BM = 64, BN = 64
        dispatch(name, [A, B, bias ?? emptyMask, out], &p, 16,
                 gridX: (N + BN - 1) / BN, gridY: (M + BM - 1) / BM, wgX: 256, wgY: 1)
    }

    // Cache of transposed constant weights ([N,K] → [K,N]). The weight buffers are
    // immutable, so the transpose only needs to happen once per weight, not once per
    // matmul call (the flow-matching loop reuses each weight 16× per generation).
    private var transposeCache: [ObjectIdentifier: MTLBuffer] = [:]

    // Row-major matmul where the weight is stored transposed [N, K] (PyTorch Linear /
    // conv1x1 [outCh, inCh]). Computes out[M,N] = A[M,K] · W^T. Transpose W to [K,N] first.
    private func matmulRowMajorWT(A: MTLBuffer, Wt: MTLBuffer, bias: MTLBuffer?, out: MTLBuffer, M: Int, K: Int, N: Int, act: String = "none") {
        let key = ObjectIdentifier(Wt)
        let Bkn: MTLBuffer
        if let cached = transposeCache[key] {
            Bkn = cached
        } else {
            let t = empty(K * N)
            dispatchTranspose(input: Wt, out: t, rows: N, cols: K)   // [N,K] -> [K,N]
            flushAndWait()                                          // materialize once
            transposeCache[key] = t
            Bkn = t
        }
        matmulRowMajor(A: A, B: Bkn, bias: bias, out: out, M: M, K: K, N: N, act: act)
    }

    private func dispatchTranspose(input: MTLBuffer, out: MTLBuffer, rows: Int, cols: Int) {
        struct P { var rows: UInt32; var cols: UInt32 }
        var p = P(rows: UInt32(rows), cols: UInt32(cols))
        dispatch1d("transpose_kernel", [input, out], &p, 8, rows * cols)
    }

    private func toRowMajor(_ x: MTLBuffer, rows: Int, cols: Int) throws -> MTLBuffer {
        // [rows, cols] channel-major -> [cols, rows] row-major (a transpose)
        let out = empty(rows * cols)
        dispatchTranspose(input: x, out: out, rows: rows, cols: cols)
        return out
    }
    private func toChannelMajor(_ x: MTLBuffer, rows: Int, cols: Int) throws -> MTLBuffer {
        let out = empty(rows * cols)
        dispatchTranspose(input: x, out: out, rows: rows, cols: cols)
        return out
    }

    // MARK: - Metal plumbing

    private lazy var emptyMask: MTLBuffer = device.makeBuffer(length: 4, options: [.storageModeShared])!

    private func dispatch(_ name: String, _ buffers: [MTLBuffer?], _ params: UnsafeRawPointer, _ plen: Int,
                          gridX: Int, gridY: Int = 1, wgX: Int = 256, wgY: Int = 1) {
        ensureEncoder()
        guard let p = pipelines[name] else { print("[Supertonic] no pipeline \(name)"); return }
        enc!.setComputePipelineState(p)
        for (i, b) in buffers.enumerated() { enc!.setBuffer(b, offset: 0, index: i) }
        enc!.setBytes(params, length: plen, index: buffers.count)
        enc!.dispatchThreadgroups(MTLSizeMake(gridX, gridY, 1), threadsPerThreadgroup: MTLSizeMake(wgX, wgY, 1))
    }
    private func dispatch1d(_ name: String, _ buffers: [MTLBuffer?], _ params: UnsafeRawPointer, _ plen: Int, _ total: Int) {
        let wg = 256, grid = max(1, (total + wg - 1) / wg)
        dispatch(name, buffers, params, plen, gridX: grid, wgX: wg)
    }

    private func ensureEncoder() {
        if cmd == nil { cmd = queue.makeCommandBuffer(); enc = cmd!.makeComputeCommandEncoder() }
    }
    private func flushAndWait() {
        if let e = enc { e.endEncoding(); cmd?.commit(); cmd?.waitUntilCompleted() }
        cmd = nil; enc = nil
    }
    private func readBuf(_ b: MTLBuffer, count: Int) -> [Float] {
        flushAndWait()
        return b.contents().withMemoryRebound(to: Float.self, capacity: count) {
            Array(UnsafeBufferPointer(start: $0, count: count))
        }
    }
    private func readIntBuf(_ b: MTLBuffer, count: Int) -> [Int32] {
        flushAndWait()
        return b.contents().withMemoryRebound(to: Int32.self, capacity: count) {
            Array(UnsafeBufferPointer(start: $0, count: count))
        }
    }
    private func makeBuf(_ f: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: f, length: max(1, f.count) * 4, options: [.storageModeShared])!
    }
    private func makeIntBuf(_ v: [Int32]) -> MTLBuffer {
        device.makeBuffer(bytes: v, length: max(1, v.count) * 4, options: [.storageModeShared])!
    }
    private func empty(_ n: Int) -> MTLBuffer {
        device.makeBuffer(length: max(1, n) * 4, options: [.storageModeShared])!
    }
    private func saveBin(_ path: String, _ buf: MTLBuffer, count: Int) {
        let arr = readBuf(buf, count: count)
        let data = Data(bytes: arr, count: count * 4)
        try? data.write(to: URL(fileURLWithPath: path))
        print("Saved \(path)")
    }
    private func gaussian(_ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            let u1 = Float.random(in: 1e-6...1), u2 = Float.random(in: 0...1)
            let r = (-2 * log(u1)).squareRoot()
            out[i] = r * cos(2 * .pi * u2)
            if i + 1 < n { out[i + 1] = r * sin(2 * .pi * u2) }
            i += 2
        }
        return out
    }
    private func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000 }
    private func err(_ m: String) -> NSError { NSError(domain: "Supertonic", code: 1, userInfo: [NSLocalizedDescriptionKey: m]) }

    // MARK: - Stage validation (ST_VALIDATE=1) against /tmp/st_ref dumps

    private func loadF32(_ path: String) -> [Float] {
        guard let d = FileManager.default.contents(atPath: path) else { return [] }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    private func loadI32(_ path: String) -> [Int32] {
        guard let d = FileManager.default.contents(atPath: path) else { return [] }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }
    private func compareStage(_ name: String, _ got: [Float], _ ref: [Float]) {
        guard got.count == ref.count, !got.isEmpty else {
            print(String(format: "[VALIDATE] %-16@ COUNT MISMATCH got=%d ref=%d", name as NSString, got.count, ref.count))
            return
        }
        let n = got.count
        var mg: Float = 0, mr: Float = 0
        for i in 0..<n { mg += got[i]; mr += ref[i] }
        mg /= Float(n); mr /= Float(n)
        var cov: Float = 0, vg: Float = 0, vr: Float = 0, l2d: Float = 0, l2r: Float = 0, maxd: Float = 0
        for i in 0..<n {
            let a = got[i] - mg, b = ref[i] - mr
            cov += a * b; vg += a * a; vr += b * b
            let d = got[i] - ref[i]
            l2d += d * d; l2r += ref[i] * ref[i]
            maxd = max(maxd, abs(d))
        }
        let corr = cov / (sqrt(vg * vr) + 1e-12)
        let relL2 = sqrt(l2d / (l2r + 1e-12))
        print(String(format: "[VALIDATE] %-16@ corr=%.4f relL2=%.4f maxAbs=%.4f | gotStd=%.4f refStd=%.4f",
                     name as NSString, corr, relL2, maxd, sqrt(vg/Float(n)), sqrt(vr/Float(n))))
    }

    /// Deterministic stage isolation. Loads /tmp/st_ref/* (from tools/dump_stages.py)
    /// and compares each Metal stage element-wise, feeding reference inputs so a
    /// divergence localises to exactly one stage.
    func validate() {
        setvbuf(stdout, nil, _IONBF, 0)
        do { if status != .ready { try load() } } catch { print("[VALIDATE] load failed: \(error)"); exit(1) }
        let dir = "/tmp/st_ref/"
        guard let meta = try? String(contentsOfFile: dir + "meta.txt", encoding: .utf8) else {
            print("[VALIDATE] no /tmp/st_ref/meta.txt — run tools/dump_stages.py first"); exit(1)
        }
        var T = 0, L = 0
        for line in meta.split(separator: "\n") {
            let kv = line.split(separator: "="); guard kv.count == 2, let v = Int(kv[1]) else { continue }
            if kv[0] == "T" { T = v }; if kv[0] == "L" { L = v }
        }
        print("[VALIDATE] T=\(T) L=\(L)")
        let C = config.charEmb
        let ids = loadI32(dir + "text_ids")
        let styleTtl = makeBuf(loadF32(dir + "style_ttl"))
        let refTextEmb = loadF32(dir + "text_emb")
        let noise = loadF32(dir + "noise")
        let refStep0 = loadF32(dir + "step0")
        let refLatent = loadF32(dir + "latent")
        let refWav = loadF32(dir + "wav")

        // A0) text-encoder sub-stages: embedding + ConvNeXt 0..5
        do {
            let embName = findWeight("te", contains: "char_embedder")
            var x = try embedChannelMajor(ns: "te", embName: embName, ids: makeIntBuf(ids), T: T, C: C)
            compareStage("te.embed", readBuf(x, count: C * T),
                         loadF32(dir + "te_text_encoder_convnext_convnext.0_Mul_output_0.f32"))
            for i in 0..<config.teConvNextLayers {
                x = try convNextLayer(ns: "te", prefix: convnextPrefix("te", "convnext.convnext.\(i)"),
                                      x: x, C: C, T: T, ksz: config.teConvNextKsz,
                                      dil: config.teConvNextDil[i], inter: config.teConvNextInter)
                if i == 0 { compareStage("te.convnext0", readBuf(x, count: C * T),
                             loadF32(dir + "te_text_encoder_convnext_convnext.0_Add_output_0.f32")) }
            }
            compareStage("te.convnext5", readBuf(x, count: C * T),
                         loadF32(dir + "te_text_encoder_convnext_convnext.5_Add_output_0.f32"))
        } catch { print("[VALIDATE] te substage err \(error)") }

        // A) text encoder (deterministic, full)
        do {
            let te = try runTextEncoder(ids: makeIntBuf(ids), T: T, styleTtl: styleTtl)
            compareStage("text_encoder", readBuf(te, count: C * T), refTextEmb)
        } catch { print("[VALIDATE] text_encoder err \(error)") }

        // B) vector field, step 0, fed REFERENCE text_emb + REFERENCE noise
        do {
            let vf = try vectorEstimatorStep(x: noise, textEmb: makeBuf(refTextEmb), T: T,
                                             styleTtl: styleTtl, L: L, curStep: 0, totalStep: 8)
            compareStage("vfield[0]", vf, refStep0)
        } catch { print("[VALIDATE] vfield err \(error)") }

        // C0) vocoder sub-stages (fed the reference latent)
        do {
            let ld = config.latentDim, ccf = config.chunkCompress, subL = ccf * L, hdim = config.aeHdim
            var scaleVal: Float = 1
            if let s = weightBuf("vo", "tts.ttl.normalizer.scale") { scaleVal = readBuf(s, count: 1).first ?? 1 }
            var x = try unfoldLatent(latent: makeBuf(refLatent), latentDim: ld, ccf: ccf, L: L)
            compareStage("voc.unfold", readBuf(x, count: ld * subL), loadF32(dir + "voc_Reshape_1_output_0.f32"))
            if let mean = weightBuf("vo", "latent_mean"), let std = weightBuf("vo", "latent_std") {
                let ss = readBuf(std, count: ld).map { $0 / scaleVal }
                x = try denorm(x: x, mean: mean, std: makeBuf(ss), C: ld, T: subL)
            }
            compareStage("voc.denorm", readBuf(x, count: ld * subL), loadF32(dir + "voc_Add_output_0.f32"))
            guard let ewName = weightNameByShape("vo", [hdim, ld, 7]), let ew = weightBuf("vo", ewName) else { throw err("no embed") }
            var eb: MTLBuffer? = nil
            if let n = Int(ewName.split(separator: "_").last ?? "") { eb = weightBuf("vo", "onnx::Conv_\(n + 1)") }
            var h = empty(hdim * subL)
            dispatchCausalConv1d(input: x, weight: ew, bias: eb, out: h, inCh: ld, outCh: hdim, ksz: 7, L: subL, dil: 1)
            compareStage("voc.embed", readBuf(h, count: hdim * subL), loadF32(dir + "voc_decoder_embed_net_Conv_output_0.f32"))
            for i in 0..<config.aeDecLayers {
                h = try convNextLayer(ns: "vo", prefix: "tts.ae.decoder.convnext.\(i)", x: h, C: hdim, T: subL,
                                      ksz: config.aeDecKsz, dil: config.aeDecDil[i], inter: config.aeDecInter, causal: true)
                if i == 0 { compareStage("voc.convnext0", readBuf(h, count: hdim * subL),
                             loadF32(dir + "voc_decoder_convnext_0_Add_output_0.f32")) }
            }
            h = try applyBatchNorm(ns: "vo", x: h, C: hdim, T: subL)
            compareStage("voc.batchnorm", readBuf(h, count: hdim * subL),
                         loadF32(dir + "voc_decoder_final_norm_BatchNormalization_output_0.f32"))
        } catch { print("[VALIDATE] voc substage err \(error)") }

        // C) vocoder, fed the REFERENCE final latent (isolates the vocoder)
        do {
            let wav = try runVocoder(latent: makeBuf(refLatent), L: L)
            compareStage("vocoder", wav, refWav)
            writeWav(wav, to: URL(fileURLWithPath: "/tmp/supertonic_voctest.wav"), sampleRate: config.sampleRate)
            print("[VALIDATE] wrote /tmp/supertonic_voctest.wav")
        } catch { print("[VALIDATE] vocoder err \(error)") }
        exit(0)
    }

    // MARK: - Self-test (SUPERTONIC_SELFTEST=1)

    /// On-device smoke test: loads if needed, runs the full pipeline on a fixed
    /// sentence, prints per-stage timings + a realtime factor, and writes a WAV to
    /// /tmp so the (currently provisional) output can be inspected. Correctness of
    /// the audio is still gated by the `VALIDATE` items in docs/PORT_STATUS.md.
    func selfTest() {
        setvbuf(stdout, nil, _IONBF, 0)   // unbuffered so logs survive a headless run
        print("[Supertonic] ── self-test ──")
        do {
            if status != .ready { try load() }
        } catch {
            print("[Supertonic] load failed: \(error)"); return
        }
        print("[Supertonic] status=\(status) voices=\(voices.count) tokenizerCodepoints=\(unicodeIndexer.count)")
        let text = ProcessInfo.processInfo.environment["SUPERTONIC_TEXT"]
            ?? "A gentle breeze moved through the open window while everyone listened to the story."
        let voice = ProcessInfo.processInfo.environment["SUPERTONIC_VOICE"] ?? "M1"
        if let steps = ProcessInfo.processInfo.environment["SUPERTONIC_STEPS"].flatMap({ Int($0) }) {
            config.odeSteps = steps
        }
        profile = true
        let ids = tokenize(text)
        print("[Supertonic] text=\"\(text)\" → \(ids.count) tokens, voice=\(voice), odeSteps=\(config.odeSteps)")
        let t0 = now()
        do {
            let wav = try generate(text, voiceName: voice)
            let ms = now() - t0
            let audioSec = Double(wav.count) / Double(config.sampleRate)
            print(String(format: "[Supertonic] total %.1f ms → %d samples (%.2fs) RTF %.2fx",
                         ms, wav.count, audioSec, audioSec / (ms / 1000)))
            for t in lastTimings { print(String(format: "[Supertonic]   %@ %.1f ms", t.name, t.ms)) }
            let url = URL(fileURLWithPath: "/tmp/supertonic_selftest.wav")
            writeWav(wav, to: url, sampleRate: config.sampleRate)
            print("[Supertonic] wrote \(url.path)")
        } catch {
            print("[Supertonic] generate failed: \(error)")
        }
        // Headless self-test: exit so the run terminates and flushes cleanly.
        if ProcessInfo.processInfo.environment["SUPERTONIC_SELFTEST_EXIT"] != "0" {
            print("[Supertonic] self-test done, exiting")
            exit(0)
        }
    }

    private func writeWav(_ samples: [Float], to url: URL, sampleRate: Int) {
        let n = samples.count
        var data = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        let byteRate = UInt32(sampleRate * 2)
        data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + n * 2)); data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(byteRate); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(UInt32(n * 2))
        for s in samples {
            let v = Int16(max(-1, min(1, s)) * 32767)
            var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        }
        try? data.write(to: url)
    }
}
