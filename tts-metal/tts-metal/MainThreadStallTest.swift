//
//  MainThreadStallTest.swift
//  tts-metal
//
//  Diagnostic for the "beachball + frozen karaoke" report (SUPERTONIC_STALLTEST=1).
//
//  Karaoke highlighting is driven by a 60 Hz Timer on the main run loop, so a frozen
//  highlight and a spinning beachball are the same fault: the main thread stopped being
//  serviced. This harness reproduces the reader's actual access pattern — a @MainActor
//  scheduling loop awaiting EngineHub.generate over sentences of varying length — while
//  a 60 Hz main-run-loop timer records how late each tick is. Whatever blocks the main
//  thread in the app shows up here as tick lateness, attributable to a specific stage.
//
//  Run both ways to attribute:
//     SUPERTONIC_STALLTEST=1 SUPERTONIC_ANE=0 Blabla     (Metal baseline)
//     SUPERTONIC_STALLTEST=1 SUPERTONIC_ANE=1 Blabla
//

import Foundation

@MainActor
enum MainThreadStallTest {

    /// Sentences chosen to span several (T, L) grid cells, so cold-cell CoreML loads and
    /// ANE program compiles land mid-run exactly as they do while the reader is playing.
    private static let sentences: [String] = [
        "Hello there.",
        "A gentle breeze moved through the open window while everyone listened.",
        "Short one.",
        "The afternoon light fell across the floorboards in long pale stripes, and somewhere outside a dog barked twice and then fell silent again, leaving the room quieter than before.",
        "Another brief line.",
        "She turned the page slowly, aware that the story was nearly finished and that whatever happened next would have to carry the weight of everything that came before it.",
    ]

    static func run() async {
        setvbuf(stdout, nil, _IONBF, 0)
        let mode = ProcessInfo.processInfo.environment["SUPERTONIC_ANE"] == "0" ? "METAL" : "ANE"
        print("[Stall] ── main-thread stall test (\(mode)) ──")

        let hub = EngineHub.shared
        hub.ensureLoaded()
        while !hub.ready && !hub.failed { try? await Task.sleep(nanoseconds: 50_000_000) }
        guard hub.ready else { print("[Stall] engine failed to load"); return }
        print("[Stall] engine ready")

        // 60 Hz main-run-loop timer, exactly as startKaraoke() installs it.
        let period = 1.0 / 60.0
        var last = Date()
        var worst: Double = 0
        var lateTicks = 0
        var ticks = 0
        var stallLog: [String] = []
        var currentStage = "idle"

        let timer = Timer(timeInterval: period, repeats: true) { _ in
            let now = Date()
            let gap = now.timeIntervalSince(last)
            last = now
            ticks += 1
            let late = gap - period
            if late > 0.1 {                       // >100 ms is a visible freeze
                lateTicks += 1
                worst = max(worst, late)
                stallLog.append(String(format: "  stalled %.0f ms during: %@", late * 1000, currentStage))
            }
        }
        RunLoop.main.add(timer, forMode: .common)

        // Mirror ReaderController.speakTask: a @MainActor loop awaiting generation.
        // SUPERTONIC_PASSES>1 keeps the run alive past the background ANE warmer, so the
        // later passes exercise the warm ANE path and the switchover itself.
        let passes = Int(ProcessInfo.processInfo.environment["SUPERTONIC_PASSES"] ?? "") ?? 1
        for pass in 1...passes {
            for (i, s) in sentences.enumerated() {
                currentStage = "pass \(pass) sentence \(i + 1)/\(sentences.count) (\(s.count) chars)"
                let t0 = Date()
                do {
                    let w = try await hub.generate(s, voice: "M1", speed: 1.0)
                    let ms = Date().timeIntervalSince(t0) * 1000
                    print(String(format: "[Stall] %@ -> %d samples in %.0f ms", currentStage, w.count, ms))
                } catch {
                    print("[Stall] \(currentStage) failed: \(error)")
                }
            }
        }
        currentStage = "done"
        timer.invalidate()

        print(String(format: "\n[Stall] %@: %d ticks, %d stalls >100ms, worst %.0f ms",
                     mode, ticks, lateTicks, worst * 1000))
        for l in stallLog.prefix(20) { print(l) }
        let expected = Int(Date().timeIntervalSince(last) / period)
        _ = expected
        print("[Stall] verdict: \(lateTicks == 0 ? "main thread stayed responsive" : "MAIN THREAD BLOCKED")")
        exit(0)
    }
}
