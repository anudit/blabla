//
//  SupertonicOrtEngine.swift
//  tts-metal
//
//  Working Supertonic 3 pipeline via ONNX Runtime (the correct-audio path).
//  Mirrors the reference supertonic/core.py exactly. The pure-Metal port in
//  SupertonicEngine.swift remains the optimisation goal; this gets the model
//  actually producing speech end-to-end in the app today.
//

import Foundation
import OnnxRuntimeBindings

final class SupertonicOrtEngine: @unchecked Sendable {

    let sampleRate = 44100
    private let baseChunk = 512
    private let ccf = 6
    private let ldim = 24

    private var env: ORTEnv!
    private var dp: ORTSession!     // duration_predictor
    private var te: ORTSession!     // text_encoder
    private var ve: ORTSession!     // vector_estimator
    private var vo: ORTSession!     // vocoder
    private var indexer: [Int] = []

    struct StageTime { let name: String; let ms: Double }
    private(set) var lastTimings: [StageTime] = []

    enum EngErr: Error { case missing(String) }

    // MARK: - Load

    func load() throws {
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        func session(_ name: String) throws -> ORTSession {
            guard let url = Bundle.main.url(forResource: name, withExtension: "onnx",
                                            subdirectory: "supertonic") ??
                            Bundle.main.url(forResource: name, withExtension: "onnx") else {
                throw EngErr.missing("\(name).onnx")
            }
            return try ORTSession(env: env, modelPath: url.path, sessionOptions: opts)
        }
        dp = try session("duration_predictor")
        te = try session("text_encoder")
        ve = try session("vector_estimator")
        vo = try session("vocoder")

        guard let idxURL = Bundle.main.url(forResource: "unicode_indexer", withExtension: "json",
                                           subdirectory: "supertonic") ??
                           Bundle.main.url(forResource: "unicode_indexer", withExtension: "json") else {
            throw EngErr.missing("unicode_indexer.json")
        }
        indexer = try JSONSerialization.jsonObject(with: Data(contentsOf: idxURL)) as? [Int] ?? []
    }

    // MARK: - Voice styles

    struct Voice { let ttl: [Float]; let dp: [Float] }   // [1,50,256], [1,8,16]

    func loadVoice(_ name: String) -> Voice? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "json",
                                        subdirectory: "voice_styles") ??
                        Bundle.main.url(forResource: name, withExtension: "json"),
              let obj = try? JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        else { return nil }
        func flat(_ key: String) -> [Float] {
            guard let d = obj[key] as? [String: Any], let data = d["data"] else { return [] }
            var out: [Float] = []
            func rec(_ x: Any) {
                if let a = x as? [Any] { for e in a { rec(e) } }
                else if let n = x as? NSNumber { out.append(n.floatValue) }
            }
            rec(data); return out
        }
        let ttl = flat("style_ttl"), dp = flat("style_dp")
        guard ttl.count == 50 * 256, dp.count == 8 * 16 else { return nil }
        return Voice(ttl: ttl, dp: dp)
    }

    // MARK: - Text preprocessing + tokenize (matches core.py)

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

    func tokenize(_ text: String) -> [Int64] {
        preprocess(text).unicodeScalars.compactMap { s in
            let cp = Int(s.value)
            guard cp < indexer.count else { return nil }
            let id = indexer[cp]
            return id >= 0 ? Int64(id) : nil
        }
    }

    // MARK: - Generate

    func generate(_ text: String, voiceName: String, totalSteps: Int = 8, speed: Float = 1.05) throws -> [Float] {
        guard let voice = loadVoice(voiceName) ?? loadVoice("M1") else { throw EngErr.missing("voice") }
        var timings: [StageTime] = []
        func timed<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
            let t0 = now(); let r = try body(); timings.append(StageTime(name: name, ms: now() - t0)); return r
        }

        let ids = tokenize(text)
        let T = ids.count
        guard T > 0 else { return [] }
        let textMask = [Float](repeating: 1, count: T)

        let idsV = try intValue(ids, shape: [1, T])
        let textMaskV = try floatValue(textMask, shape: [1, 1, T])
        let ttlV = try floatValue(voice.ttl, shape: [1, 50, 256])
        let dpV = try floatValue(voice.dp, shape: [1, 8, 16])

        // duration predictor → seconds
        var dur = try timed("duration_predictor") {
            try floats(run(dp, ["text_ids": idsV, "style_dp": dpV, "text_mask": textMaskV], ["duration"])["duration"]!)
        }
        dur = dur.map { $0 / speed }
        let durMax = dur.max() ?? 1

        // text encoder
        let textEmbV = try timed("text_encoder") {
            try run(te, ["text_ids": idsV, "style_ttl": ttlV, "text_mask": textMaskV], ["text_emb"])["text_emb"]!
        }

        // noisy latent
        let chunk = baseChunk * ccf
        let wavLenMax = Int((durMax * Float(sampleRate)).rounded(.down))
        let latentLen = max(1, (wavLenMax + chunk - 1) / chunk)
        let latentDim = ldim * ccf
        var xt = gaussian(latentDim * latentLen)             // [1,144,L], mask all-ones for 1 utt
        let latentMask = [Float](repeating: 1, count: latentLen)
        let latentMaskV = try floatValue(latentMask, shape: [1, 1, latentLen])
        let totalStepV = try floatValue([Float(totalSteps)], shape: [1])

        // flow matching (vector estimator returns next latent directly)
        xt = try timed("flow_matching") {
            var x = xt
            for step in 0..<totalSteps {
                let xV = try floatValue(x, shape: [1, latentDim, latentLen])
                let csV = try floatValue([Float(step)], shape: [1])
                let out = try run(ve, [
                    "noisy_latent": xV, "text_emb": textEmbV, "style_ttl": ttlV,
                    "text_mask": textMaskV, "latent_mask": latentMaskV,
                    "current_step": csV, "total_step": totalStepV,
                ], ["denoised_latent"])["denoised_latent"]!
                x = try floats(out)
            }
            return x
        }

        // vocoder
        let wav = try timed("vocoder") {
            let xV = try floatValue(xt, shape: [1, latentDim, latentLen])
            return try floats(run(vo, ["latent": xV], ["wav_tts"])["wav_tts"]!)
        }
        lastTimings = timings
        return wav
    }

    // MARK: - ORT helpers

    private func run(_ s: ORTSession, _ inputs: [String: ORTValue], _ outputs: [String]) throws -> [String: ORTValue] {
        try s.run(withInputs: inputs, outputNames: Set(outputs), runOptions: nil)
    }

    private func floatValue(_ arr: [Float], shape: [Int]) throws -> ORTValue {
        let data = NSMutableData(bytes: arr, length: arr.count * MemoryLayout<Float>.size)
        return try ORTValue(tensorData: data, elementType: .float, shape: shape.map { NSNumber(value: $0) })
    }
    private func intValue(_ arr: [Int64], shape: [Int]) throws -> ORTValue {
        let data = NSMutableData(bytes: arr, length: arr.count * MemoryLayout<Int64>.size)
        return try ORTValue(tensorData: data, elementType: .int64, shape: shape.map { NSNumber(value: $0) })
    }
    private func floats(_ v: ORTValue) throws -> [Float] {
        let data = try v.tensorData() as Data
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func gaussian(_ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n); var i = 0
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
}
