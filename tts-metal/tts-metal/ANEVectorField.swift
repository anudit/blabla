//
//  ANEVectorField.swift
//  tts-metal
//
//  Apple Neural Engine path for the flow-matching stage.
//
//  The hand-written Metal port in SupertonicEngine reimplements vector_estimator.onnx
//  as ~2000 fp32 compute dispatches per generation. That runs at roughly 0.3 TFLOPS on
//  an M2 Max — the ODE state is only L≈80 frames wide, so a 64×64-tiled GEMM launches
//  ~96 threadgroups onto 38 cores and the GPU is idle by construction. The ANE fills its
//  pipeline on graph *depth* rather than width, which is exactly the shape of this model
//  (28 ConvNeXt blocks ≈ 90 layers of 1×1 and depthwise conv at C=512).
//
//  One important property of the ONNX graph, confirmed by reading its tail:
//
//      out = latent_mask * ( noisy_latent + (1/total_step) * (4·v_cond − 3·v_uncond) )
//
//  The model performs the *whole* Euler step internally — both classifier-free-guidance
//  branches, the 4/−3 combination, the 1/total_step update and the mask. It returns
//  x_{k+1}, not a velocity. So this path replaces runFlowMatching in its entirety:
//  there is no external CFG and no external Euler integration, and the batch dimension
//  stays 1 (the graph doubles it internally for the two CFG branches).
//
//  Shapes are static — the ANE compiler rejects dynamic shapes outright rather than
//  falling back — so L is padded up to one of `buckets` and masked. Padding is not
//  bit-exact with running at the true length: the ConvNeXt edge-padding convention at
//  the sequence end differs, and with a receptive field spanning 28 dilated blocks that
//  one boundary difference reaches every position. Measured, it moves the final latent
//  by ~2.6% — against ~102% run-to-run variation from the random x_0 the ODE already
//  starts from, i.e. ~2.6% of the noise the pipeline has anyway. The bucket models live
//  in one multifunction .mlpackage so the 61M weights are stored once (128 MB total)
//  rather than per bucket (615 MB).
//

import Foundation
import CoreML
import Metal

final class ANEVectorField: @unchecked Sendable {

    /// Static shapes compiled into the multifunction model, ascending. Both axes are
    /// baked into each compiled function, so the model is a grid of (T, L) pairs.
    ///
    /// The two axes are not equally costly to pad. Text padding is *bit-identical* to
    /// running at the true length — the text axis is only ever consumed through masked
    /// cross-attention, never convolved along — so `tBuckets` can be coarse. Latent
    /// padding carries the ~2.6% boundary effect described above regardless of how much
    /// is padded, so `lBuckets` is kept fine-grained purely to avoid wasting compute.
    static let tBuckets = [96, 192, 304]
    static let lBuckets = [64, 96, 128, 192, 288]

    static func tBucket(for T: Int) -> Int? { tBuckets.first { $0 >= T } }
    static func lBucket(for L: Int) -> Int? { lBuckets.first { $0 >= L } }

    /// Whether this (T, L) pair has a compiled function; the caller falls back to the
    /// Metal path when it does not.
    static func supports(T: Int, L: Int) -> Bool { tBucket(for: T) != nil && lBucket(for: L) != nil }

    private let modelURL: URL

    // Reached from the synthesis queue and the background warmer. The lock is held only
    // for the O(1) dictionary read/write, never across the MLModel load — that can block
    // for seconds on a cold ANE compile, and holding a lock across it would serialize the
    // warmer against synthesis, which is exactly what this is here to avoid. A cell can
    // therefore be compiled twice in a race; the loser's instance is simply discarded.
    private var models: [String: MLModel] = [:]  // "T96_L64" -> loaded function
    private let cacheLock = NSLock()

    /// Scratch tensors, reused across the 8 ODE steps so each iteration is a bare
    /// predict() with no allocation.
    private var xArr: MLMultiArray
    private var textArr: MLMultiArray
    private var styleArr: MLMultiArray
    private var latMaskArr: MLMultiArray
    private var txtMaskArr: MLMultiArray
    private var curArr: MLMultiArray
    private var totArr: MLMultiArray

    enum ANEError: Error, LocalizedError {
        case modelMissing
        case notWarmedYet(Int, Int)
        case tooLong(Int, Int)
        var errorDescription: String? {
            switch self {
            case .modelMissing: return "ve_grid.mlmodelc / .mlpackage not found in bundle"
            case .notWarmedYet(let t, let l): return "(T=\(t), L=\(l)) not warmed yet"
            case .tooLong(let t, let l):
                return "(T=\(t), L=\(l)) exceeds largest ANE bucket (T=\(ANEVectorField.tBuckets.last!), L=\(ANEVectorField.lBuckets.last!))"
            }
        }
    }

    /// Resolves the compiled model in the bundle. Xcode compiles the bundled
    /// .mlpackage to .mlmodelc; if only the package is present (e.g. a resource copy
    /// that skipped the CoreML build rule) it is compiled once at load.
    init() throws {
        if let u = Bundle.main.url(forResource: "ve_grid", withExtension: "mlmodelc",
                                   subdirectory: "supertonic")
                ?? Bundle.main.url(forResource: "ve_grid", withExtension: "mlmodelc") {
            modelURL = u
        } else if let pkg = Bundle.main.url(forResource: "ve_grid", withExtension: "mlpackage",
                                            subdirectory: "supertonic")
                ?? Bundle.main.url(forResource: "ve_grid", withExtension: "mlpackage") {
            modelURL = try MLModel.compileModel(at: pkg)
        } else {
            throw ANEError.modelMissing
        }
        func arr(_ shape: [Int]) throws -> MLMultiArray {
            try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        }
        let tb = ANEVectorField.tBuckets[0], lb = ANEVectorField.lBuckets[0]
        xArr       = try arr([1, 144, lb])
        textArr    = try arr([1, 256, tb])
        styleArr   = try arr([1, 50, 256])
        latMaskArr = try arr([1, 1, lb])
        txtMaskArr = try arr([1, 1, tb])
        curArr     = try arr([1])
        totArr     = try arr([1])
    }

    /// Loads one function, ANE-compiling it if the system has no cached program for it:
    /// 5-11 s cold, ~0.5 s once cached. The cache is keyed to the compiled model, so it
    /// survives app rebuilds and only regenerating ve_grid makes it cold again. Only ever
    /// call this off the generation path; `run` will not compile a cell itself.
    @discardableResult
    func prewarm(T: Int, L: Int) -> Bool {
        guard let tb = ANEVectorField.tBucket(for: T), let lb = ANEVectorField.lBucket(for: L) else { return false }
        return (try? model(tb, lb)) != nil
    }

    /// Every (T, L) cell, smallest first — short sentences are both the most common and
    /// the quickest to compile, so warming in this order makes the ANE useful soonest.
    static var allCells: [(Int, Int)] {
        tBuckets.flatMap { t in lBuckets.map { (t, $0) } }
    }

    /// True when this cell is already loaded, i.e. usable without blocking.
    func isWarm(T: Int, L: Int) -> Bool {
        guard let tb = ANEVectorField.tBucket(for: T), let lb = ANEVectorField.lBucket(for: L) else { return false }
        cacheLock.lock(); defer { cacheLock.unlock() }
        return models["T\(tb)_L\(lb)"] != nil
    }

    private func model(_ tb: Int, _ lb: Int) throws -> MLModel {
        let key = "T\(tb)_L\(lb)"
        cacheLock.lock(); let cached = models[key]; cacheLock.unlock()
        if let cached { return cached }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        cfg.functionName = key
        let m = try MLModel(contentsOf: modelURL, configuration: cfg)   // slow; not locked
        cacheLock.lock(); let winner = models[key] ?? m; models[key] = winner; cacheLock.unlock()
        return winner
    }

    private static func ptr(_ a: MLMultiArray) -> UnsafeMutablePointer<Float> {
        a.dataPointer.bindMemory(to: Float.self, capacity: a.count)
    }

    /// Runs the full `totalSteps`-step ODE on the ANE.
    ///
    /// - Parameters:
    ///   - x0: initial noise, channel-major [144, L]
    ///   - textEmb: text encoder output, channel-major [256, T]
    ///   - styleTtl: voice style, [50 * 256]
    /// - Returns: the final latent, channel-major [144, L]
    func run(x0: [Float], L: Int, textEmb: [Float], T: Int, styleTtl: [Float],
             totalSteps: Int) throws -> [Float] {
        guard let tb = ANEVectorField.tBucket(for: T),
              let bk = ANEVectorField.lBucket(for: L) else { throw ANEError.tooLong(T, L) }
        // Loading a function costs ~0.5 s warm and 5-11 s when the system has to compile
        // the ANE program. Neither belongs inside a generation, so this path uses only
        // cells the background warmer has already loaded; anything else falls back to
        // Metal for now and picks up the ANE on a later sentence.
        cacheLock.lock(); let m = models["T\(tb)_L\(bk)"]; cacheLock.unlock()
        guard let m else { throw ANEError.notWarmedYet(tb, bk) }

        // Reshape the scratch tensors to this (T, L) pair. MLMultiArray has no reshape,
        // so rebuild only those whose length changed — they are the small tensors.
        if xArr.shape[2].intValue != bk {
            xArr       = try MLMultiArray(shape: [1, 144, bk].map(NSNumber.init), dataType: .float32)
            latMaskArr = try MLMultiArray(shape: [1, 1, bk].map(NSNumber.init), dataType: .float32)
        }
        if textArr.shape[2].intValue != tb {
            textArr    = try MLMultiArray(shape: [1, 256, tb].map(NSNumber.init), dataType: .float32)
            txtMaskArr = try MLMultiArray(shape: [1, 1, tb].map(NSNumber.init), dataType: .float32)
        }
        let xp = Self.ptr(xArr), lmp = Self.ptr(latMaskArr)

        // x0 into a zero-padded [144, bk]; mask 1 over the real frames, 0 over the pad.
        memset(xp, 0, 144 * bk * MemoryLayout<Float>.size)
        x0.withUnsafeBufferPointer { src in
            for c in 0..<144 { (xp + c * bk).update(from: src.baseAddress! + c * L, count: L) }
        }
        for i in 0..<bk { lmp[i] = i < L ? 1 : 0 }

        // text_emb into a zero-padded [256, tb]; text_mask 1 over the real tokens.
        let tp = Self.ptr(textArr)
        memset(tp, 0, 256 * tb * MemoryLayout<Float>.size)
        textEmb.withUnsafeBufferPointer { src in
            for c in 0..<256 { (tp + c * tb).update(from: src.baseAddress! + c * T, count: T) }
        }
        styleTtl.withUnsafeBufferPointer { Self.ptr(styleArr).update(from: $0.baseAddress!, count: 50 * 256) }
        let tmp = Self.ptr(txtMaskArr); for i in 0..<tb { tmp[i] = i < T ? 1 : 0 }
        Self.ptr(totArr)[0] = Float(totalSteps)

        let opts = MLPredictionOptions()
        for k in 0..<totalSteps {
            Self.ptr(curArr)[0] = Float(k)
            let feats = try MLDictionaryFeatureProvider(dictionary: [
                "noisy_latent": MLFeatureValue(multiArray: xArr),
                "text_emb":     MLFeatureValue(multiArray: textArr),
                "style_ttl":    MLFeatureValue(multiArray: styleArr),
                "latent_mask":  MLFeatureValue(multiArray: latMaskArr),
                "text_mask":    MLFeatureValue(multiArray: txtMaskArr),
                "current_step": MLFeatureValue(multiArray: curArr),
                "total_step":   MLFeatureValue(multiArray: totArr),
            ])
            let out = try m.prediction(from: feats, options: opts)
            guard let y = out.featureValue(for: "denoised_latent")?.multiArrayValue else {
                throw ANEError.modelMissing
            }
            // The graph returns x_{k+1}; feed it straight back in.
            Self.ptr(xArr).update(from: Self.ptr(y), count: 144 * bk)
        }

        // Trim the bucket padding back to [144, L].
        var out = [Float](repeating: 0, count: 144 * L)
        out.withUnsafeMutableBufferPointer { dst in
            for c in 0..<144 { (dst.baseAddress! + c * L).update(from: xp + c * bk, count: L) }
        }
        return out
    }
}
