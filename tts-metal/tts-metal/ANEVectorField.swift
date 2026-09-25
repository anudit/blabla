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
    private var yArr: MLMultiArray      // ping-pong partner of xArr (see `run`)
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
        yArr       = try arr([1, 144, lb])
        textArr    = try arr([1, 256, tb])
        styleArr   = try arr([1, 50, 256])
        latMaskArr = try arr([1, 1, lb])
        txtMaskArr = try arr([1, 1, tb])
        curArr     = try arr([1])
        totArr     = try arr([1])
    }

    /// Every (T, L) cell in warm-up priority order. L tracks T (≈0.85 frames per token at
    /// 1× speed), so the cells on that diagonal cover almost every real sentence and go
    /// first; the off-diagonal ones only serve unusually fast or slow speech.
    static var allCells: [(Int, Int)] {
        let diagonal = [(96, 64), (96, 96), (192, 128), (192, 192), (96, 128), (304, 192), (304, 288)]
        let rest = tBuckets.flatMap { t in lBuckets.map { (t, $0) } }
            .filter { c in !diagonal.contains { $0 == c } }
        return diagonal + rest
    }

    // MARK: - Background warm-up

    /// The cells warmed at launch, most common first: short-to-medium sentences at normal
    /// speed. Everything else is loaded on demand via `request`, ahead of the sentence
    /// that needs it. Each cell load costs ~0.5 s and ~1 J of CPU even from a warm cache
    /// (the MIL program is re-parsed and re-specialized every time), so warming all 15 on
    /// every launch was ~16 J and ~7 CPU-seconds — far more than synthesis itself uses —
    /// for cells most sessions never touch.
    static let eagerCells = [(96, 96), (192, 128), (96, 64), (192, 192)]

    private var warmQueue: [String] = []          // pending cells, front = next
    private var attempted: Set<String> = []       // queued, loading, loaded or failed
    private var activeWorkers = 0
    private var maxWorkers = 1                    // raised once the probe shows a warm cache
    private var warmStarted = false               // all guarded by cacheLock
    private var probeDone = false

    /// GPU bridge: the same functions compiled for .cpuAndGPU, used for a cell only until
    /// its ANE program is loaded, then released. CoreML's GPU path loads a cell in ~0.06 s
    /// plus a ~0.2 s first-predict compile (from its own warm cache) and runs the 8 steps
    /// in ~75 ms at L≈80 — against ~470 ms for the hand-written Metal fallback — so the
    /// sentences synthesized before the ANE cells arrive (a play press right after launch,
    /// or the minutes of serialized compiles on a cold ANE cache) are not stuck on Metal.
    private var gpuModels: [String: MLModel] = [:] // guarded by cacheLock
    private let gpuQueue = DispatchQueue(label: "ane.gpu-bridge", qos: .userInitiated)

    /// Which compute unit ran the last `run`: "ane" or "gpu". Synthesis queue only.
    private(set) var lastBackend = ""

    /// True once nothing is queued or loading (the launch set, plus any requests).
    var warmupComplete: Bool {
        cacheLock.lock(); defer { cacheLock.unlock() }
        return warmStarted && warmQueue.isEmpty && activeWorkers == 0
    }

    /// Loads `eagerCells` in the background.
    ///
    /// With a cold ANE program cache (first launch of the app from a new location, or after
    /// ve_grid.mlmodelc is regenerated) each cell compiles in 6-10 s, and the system's ANE
    /// compiler serializes those compiles no matter how many are submitted. Once cached a
    /// cell loads in ~0.5 s, and those loads *do* run in parallel. So the first cell is
    /// loaded alone as a probe: if it came back fast the cache is warm and later loads run
    /// up to 3 wide; if not they stay serial, where going wide would only delay the cells
    /// needed first. (Width 3 keeps the launch peak footprint under ~1 GB; 6 was ~1.2 GB.)
    func startWarmup() {
        cacheLock.lock()
        if warmStarted { cacheLock.unlock(); return }
        warmStarted = true
        let keys = ANEVectorField.eagerCells.map { "T\($0.0)_L\($0.1)" }
        warmQueue = keys
        attempted.formUnion(keys)
        activeWorkers = 1                          // the probe runs alone
        cacheLock.unlock()
        for key in keys { bridge(key) }
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            let t0 = Date()
            let key = popNext()
            let ok = key.map(load) ?? false
            let probe = Date().timeIntervalSince(t0)
            cacheLock.lock()
            activeWorkers -= 1
            maxWorkers = ok && probe < 2.0 ? 3 : 1
            probeDone = true
            cacheLock.unlock()
            pump()
        }
    }

    /// A sentence of this shape is coming up (or just fell back to Metal): make sure its
    /// cell is loaded, ahead of anything else still queued.
    func request(T: Int, L: Int) {
        guard let tb = ANEVectorField.tBucket(for: T), let lb = ANEVectorField.lBucket(for: L) else { return }
        let key = "T\(tb)_L\(lb)"
        cacheLock.lock()
        var isNew = false
        if !attempted.contains(key) {
            attempted.insert(key)
            warmQueue.insert(key, at: 0)
            isNew = true
        } else if let i = warmQueue.firstIndex(of: key), i > 0 {
            warmQueue.remove(at: i); warmQueue.insert(key, at: 0)
        }
        // maxWorkers stays 1 while the probe runs and when it found the cache cold. Warm, a
        // new cell's ANE load (~0.5 s) is about as quick as a GPU bridge, so only bridge
        // when the ANE may be 6-10 s per queued compile away.
        let bridgeIt = isNew && maxWorkers == 1
        cacheLock.unlock()
        if bridgeIt { bridge(key) }
        pump()
    }

    /// Starts workers until the queue is drained or `maxWorkers` are running.
    private func pump() {
        cacheLock.lock()
        var spawn = 0
        while warmStarted && activeWorkers < maxWorkers && spawn < warmQueue.count {
            activeWorkers += 1; spawn += 1
        }
        cacheLock.unlock()
        for _ in 0..<spawn {
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                while let key = popOrRetire() { _ = load(key) }
            }
        }
    }

    private func popNext() -> String? {
        cacheLock.lock(); defer { cacheLock.unlock() }
        return warmQueue.isEmpty ? nil : warmQueue.removeFirst()
    }

    /// Next queued cell, or nil after retiring the calling worker — atomically, so a
    /// `request` can never land between "queue empty" and "worker gone" and be stranded.
    private func popOrRetire() -> String? {
        cacheLock.lock(); defer { cacheLock.unlock() }
        if warmQueue.isEmpty { activeWorkers -= 1; return nil }
        return warmQueue.removeFirst()
    }

    private func load(_ key: String) -> Bool {
        let parts = key.dropFirst().split(separator: "_L")
        guard let tb = Int(parts[0]), let lb = Int(parts[1]) else { return false }
        let t0 = Date()
        guard (try? model(tb, lb)) != nil else { return false }
        cacheLock.lock(); gpuModels[key] = nil; cacheLock.unlock()   // ANE supersedes the bridge
        PerfLog.log(String(format: "ANE cell %@ warm (%.2fs)", key, Date().timeIntervalSince(t0)))
        return true
    }

    /// Loads `key` for the GPU and runs one throwaway predict (the first predict is where
    /// the GPU program is compiled), unless its ANE cell is already loaded. Serial, so
    /// bridge cells arrive in request order without piling up concurrent compiles.
    private func bridge(_ key: String) {
        gpuQueue.async { [self] in
            // Once the probe shows a warm ANE cache every ANE cell is ~0.5 s away, so further
            // bridging would only spend energy (and a 2-4 s GPU compile on a build's first run).
            cacheLock.lock()
            let skip = models[key] != nil || gpuModels[key] != nil || (probeDone && maxWorkers > 1)
            cacheLock.unlock()
            if skip { return }
            let parts = key.dropFirst().split(separator: "_L")
            guard let tb = Int(parts[0]), let lb = Int(parts[1]) else { return }
            let t0 = Date()
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndGPU
            cfg.functionName = key
            guard let m = try? MLModel(contentsOf: modelURL, configuration: cfg) else { return }
            func zeros(_ shape: [Int]) -> MLFeatureValue? {
                (try? MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)).map {
                    memset($0.dataPointer, 0, $0.count * 4); return MLFeatureValue(multiArray: $0)
                }
            }
            let shapes: [String: [Int]] = ["noisy_latent": [1, 144, lb], "text_emb": [1, 256, tb],
                                           "style_ttl": [1, 50, 256], "latent_mask": [1, 1, lb],
                                           "text_mask": [1, 1, tb], "current_step": [1], "total_step": [1]]
            var inputs = shapes.compactMapValues { zeros($0) }
            inputs["total_step"] = MLFeatureValue(multiArray: {   // 1/total_step inside the graph
                let a = try! MLMultiArray(shape: [1], dataType: .float32); a[0] = 8; return a }())
            guard let feats = try? MLDictionaryFeatureProvider(dictionary: inputs),
                  (try? m.prediction(from: feats)) != nil else { return }
            cacheLock.lock()
            if models[key] == nil { gpuModels[key] = m }
            cacheLock.unlock()
            PerfLog.log(String(format: "GPU bridge cell %@ ready (%.2fs)", key, Date().timeIntervalSince(t0)))
        }
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
        cacheLock.lock()
        let ane = models["T\(tb)_L\(bk)"], gpu = gpuModels["T\(tb)_L\(bk)"]
        cacheLock.unlock()
        guard let m = ane ?? gpu else { throw ANEError.notWarmedYet(tb, bk) }
        lastBackend = ane != nil ? "ane" : "gpu"

        // Reshape the scratch tensors to this (T, L) pair. MLMultiArray has no reshape,
        // so rebuild only those whose length changed — they are the small tensors.
        if xArr.shape[2].intValue != bk {
            xArr       = try MLMultiArray(shape: [1, 144, bk].map(NSNumber.init), dataType: .float32)
            yArr       = try MLMultiArray(shape: [1, 144, bk].map(NSNumber.init), dataType: .float32)
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

        // Ping-pong between xArr and yArr: each step's output is written by CoreML
        // straight into the buffer the next step reads (outputBackings), so a step is a
        // bare predict() with no output allocation and no copy back.
        func provider(_ x: MLMultiArray) throws -> MLDictionaryFeatureProvider {
            try MLDictionaryFeatureProvider(dictionary: [
                "noisy_latent": MLFeatureValue(multiArray: x),
                "text_emb":     MLFeatureValue(multiArray: textArr),
                "style_ttl":    MLFeatureValue(multiArray: styleArr),
                "latent_mask":  MLFeatureValue(multiArray: latMaskArr),
                "text_mask":    MLFeatureValue(multiArray: txtMaskArr),
                "current_step": MLFeatureValue(multiArray: curArr),
                "total_step":   MLFeatureValue(multiArray: totArr),
            ])
        }
        let fromX = try provider(xArr), fromY = try provider(yArr)
        let intoY = MLPredictionOptions(); intoY.outputBackings = ["denoised_latent": yArr]
        let intoX = MLPredictionOptions(); intoX.outputBackings = ["denoised_latent": xArr]
        for k in 0..<totalSteps {
            Self.ptr(curArr)[0] = Float(k)
            // The graph returns x_{k+1}; it lands in the other buffer and is read next step.
            _ = try m.prediction(from: k % 2 == 0 ? fromX : fromY, options: k % 2 == 0 ? intoY : intoX)
        }
        let final = Self.ptr(totalSteps % 2 == 0 ? xArr : yArr)

        // Trim the bucket padding back to [144, L].
        var out = [Float](repeating: 0, count: 144 * L)
        out.withUnsafeMutableBufferPointer { dst in
            for c in 0..<144 { (dst.baseAddress! + c * L).update(from: final + c * bk, count: L) }
        }
        return out
    }
}
