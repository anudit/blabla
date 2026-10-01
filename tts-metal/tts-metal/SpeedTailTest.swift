//
//  SpeedTailTest.swift
//  tts-metal
//
//  Diagnostic for the "last word is cut off at 1.5× and faster" report
//  (SUPERTONIC_SPEEDTEST=1). Synthesizes a few sentences at every reader speed
//  and dumps each take. The tail loudness it prints only catches a hard cut;
//  the model's real failure mode is skipping words, which needs a transcript —
//  run the dumped .wav files through Whisper and diff against `texts`.
//
//     SUPERTONIC_SPEEDTEST=1 SUPERTONIC_SEED=1 SUPERTONIC_SPEEDTEST_DUMP=/path/dir Blabla
//
//  Prints one "[TAIL] ..." line per take and exits.
//

import Foundation

@MainActor
enum SpeedTailTest {
    static let texts: [(String, String)] = [
        ("son", "He wrote a letter in four paragraphs to his son."),
        ("story", "A gentle breeze moved through the open window while everyone listened to the story."),
        ("dublin", "Hamilton recalled a walk along the Royal Canal in Dublin, Ireland."),
        ("quick", "Stop right there."),
        ("question", "Did you remember to bring the documents?"),
    ]

    static func run() async {
        setvbuf(stdout, nil, _IONBF, 0)
        let env = ProcessInfo.processInfo.environment
        let dump = env["SUPERTONIC_SPEEDTEST_DUMP"]
        let hub = EngineHub.shared
        hub.ensureLoaded()
        while !hub.ready && !hub.failed { try? await Task.sleep(nanoseconds: 2_000_000) }
        guard hub.ready else { print("[TAIL] engine failed to load"); exit(1) }
        let sr = TtsConfig.enhancedSampleRate
        for speed in ReaderController.speedChoices {
            var worst = 0.0
            for (name, text) in texts {
                let w = (try? await hub.generate(text, voice: "M1", speed: Float(speed))) ?? []
                guard !w.isEmpty else { print("[TAIL] \(name) speed=\(speed) empty"); continue }
                let tail = Int(0.040 * sr)
                let overall = rms(w[...])
                let end = rms(w[max(0, w.count - tail)...])
                let ratio = overall > 0 ? end / overall : 0
                worst = max(worst, ratio)
                print(String(format: "[TAIL] speed=%.2f %-8@ audio_s=%.2f tail_rel=%.3f",
                             speed, name as NSString, Double(w.count) / sr, ratio))
                if let dump {
                    let url = URL(fileURLWithPath: dump).appendingPathComponent("\(name)_\(speed).wav")
                    writeWav(w, to: url, sampleRate: Int(sr))
                }
            }
            print(String(format: "[TAIL] speed=%.2f worst_tail_rel=%.3f", speed, worst))
        }
        exit(0)
    }

    private static func rms(_ s: ArraySlice<Float>) -> Double {
        guard !s.isEmpty else { return 0 }
        return (s.reduce(0.0) { $0 + Double($1 * $1) } / Double(s.count)).squareRoot()
    }

    private static func writeWav(_ samples: [Float], to url: URL, sampleRate: Int) {
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let n = samples.count
        data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + n * 2)); data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(UInt32(n * 2))
        for s in samples { u16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
        try? data.write(to: url)
    }
}
