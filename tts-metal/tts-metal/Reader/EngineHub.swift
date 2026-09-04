//
//  EngineHub.swift
//  tts-metal
//
//  Loads the Supertonic Metal engine exactly once and shares it between the
//  menu-bar selection reader and the BlaBla document reader.
//

import Foundation
import Metal
import Combine

@MainActor
final class EngineHub: ObservableObject {
    static let shared = EngineHub()

    @Published private(set) var ready = false
    @Published private(set) var statusText = "Loading model…"
    @Published private(set) var failed = false

    private(set) var engine: SupertonicEngine?
    private var loadStarted = false

    private init() {}

    func ensureLoaded() {
        guard !loadStarted else { return }
        loadStarted = true

        // Overlaps AVAudioEngine's first-start hardware negotiation (100ms+)
        // with the model load below, instead of paying it on the first
        // sentence's time-to-first-audio.
        AudioPlayer.shared.prewarm()

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            statusText = "Metal not available."
            failed = true
            return
        }
        let eng = SupertonicEngine(device: device, queue: queue)
        Task.detached(priority: .utility) { [weak self] () -> Void in
            do { try eng.load() }
            catch { print("[EngineHub] load failed: \(error)") }
            eng.profile = true   // per-stage synthesis timings via PerfLog/print — see SupertonicEngine.generate
            await MainActor.run { [weak self] in
                self?.engine = eng
                if eng.status == .ready {
                    self?.ready = true
                    self?.statusText = "Ready"
                } else {
                    self?.failed = true
                    self?.statusText = "Supertonic load failed."
                }
            }

            // Supertonic 3 on-device smoke test (off unless SUPERTONIC_SELFTEST=1).
            if ProcessInfo.processInfo.environment["SUPERTONIC_SELFTEST"] == "1" {
                eng.selfTest()
            }
            // Stage-by-stage numerical validation against /tmp/st_ref (ST_VALIDATE=1).
            if ProcessInfo.processInfo.environment["ST_VALIDATE"] == "1" {
                eng.validate()
            }
        }
    }

    /// Serial queue that owns the engine. Two things depend on this:
    ///
    ///  1. Synthesis must not run on the main thread. `Task.detached` looks like it
    ///     guarantees that, but this type is `@MainActor`, so the closure inherited main
    ///     actor isolation and the whole pipeline — `waitUntilCompleted`, thousands of
    ///     blocking `IOGPUResourceCreate` calls — ran on the main thread, freezing the UI
    ///     and stopping the 60 Hz karaoke timer for the duration of every sentence.
    ///  2. `SupertonicEngine` is not thread-safe: it carries a single in-flight
    ///     MTLCommandBuffer/encoder pair plus scratch state, so overlapping generations
    ///     would interleave encoder writes. A serial queue serializes them by construction.
    ///
    /// It is `nonisolated` so it can be reached without hopping to the main actor.
    private nonisolated let engineQueue = DispatchQueue(label: "supertonic.engine",
                                                        qos: .userInitiated)

    /// Synthesize one chunk at 48 kHz (resampled from the engine's native rate).
    /// Runs entirely off the main actor — see `engineQueue`.
    nonisolated func generate(_ text: String, voice: String, speed: Float) async throws -> [Float] {
        guard let eng = await self.engine else { return [] }
        PerfLog.log("EngineHub.generate start (\(text.count) chars)")
        let result: [Float] = try await withCheckedThrowingContinuation { cont in
            engineQueue.async {
                do {
                    let wave = try eng.generate(text, voiceName: voice,
                                                speed: max(0.7, min(2.0, speed * 1.05)))
                    if wave.isEmpty { cont.resume(returning: []); return }
                    cont.resume(returning: Resampler.resample(wave,
                                                              from: Double(eng.sampleRate),
                                                              to: TtsConfig.enhancedSampleRate))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
        PerfLog.log("EngineHub.generate done")
        return result
    }
}
