//
//  TtsController.swift
//  tts-metal
//

import Foundation
import Metal
import SwiftUI
import Combine

@MainActor
final class TtsController: ObservableObject {
    @Published var statusText: String = "Initializing…"
    @Published var canGenerate: Bool = false
    @Published var isPlaying: Bool = false
    @Published var timingLog: [(name: String, ms: Double)] = []
    @Published var lastTotalMs: Double = 0

    var statusColor: Color { .green }

    private var engine: MetalTtsEngine?
    private let audio = AudioPlayer.shared
    private var loaded = false

    init() {
        Task { await load() }
    }

    private func load() async {
        statusText = "Loading Metal device…"
        guard let device = MTLCreateSystemDefaultDevice() else {
            statusText = "Metal not available on this device."
            return
        }
        guard let queue = device.makeCommandQueue() else {
            statusText = "Could not create command queue."
            return
        }
        let eng = MetalTtsEngine(device: device, commandQueue: queue)
        do {
            try eng.load()
        } catch {
            statusText = "Load failed: \(error.localizedDescription)"
            return
        }
        // Preload phonemizer (dictionary + en_rules) during warmup so the
        // first generate() call doesn't pay the ~50-100ms load cost.
        statusText = "Warming up phonemizer…"
        await Task.detached(priority: .utility) {
            Phonemizer.warmup()
        }.value
        self.engine = eng
        self.loaded = true
        self.canGenerate = true
        statusText = "Ready (kitten-tts-mini · 80M params)"
    }

    func generate(text: String, voice: String, speed: Float) async {
        guard let engine = engine, loaded else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return }

        canGenerate = false
        isPlaying = true
        let totalStart = DispatchTime.now().uptimeNanoseconds

        statusText = "Phonemizing…"
        let ids = Phonemizer.textToInputIds(trimmed)
        if ids.isEmpty { statusText = "No phonemes produced."; canGenerate = true; isPlaying = false; return }

        statusText = "Running inference on Metal…"
        do {
            // Run the heavy GPU-encode + CPU-DSP pipeline off the main thread so the UI
            // stays responsive during the ~hundreds-of-ms inference.
            let waveform = try await Task.detached(priority: .userInitiated) {
                try engine.generate(inputIds: ids, voice: voice, speed: speed, textLength: trimmed.count)
            }.value
            let totalMs = Double(DispatchTime.now().uptimeNanoseconds - totalStart) / 1_000_000.0
            lastTotalMs = totalMs
            timingLog = engine.lastTimings.map { (name: $0.name, ms: $0.ms) }
            statusText = "Generated \(String(format: "%.2f", Double(waveform.count) / TtsConfig.sampleRate))s of audio in \(String(format: "%.0f", totalMs)) ms"
            audio.play(waveform, sampleRate: TtsConfig.sampleRate)
        } catch {
            statusText = "Inference error: \(error.localizedDescription)"
        }
        canGenerate = true
        isPlaying = false
    }

    func stop() {
        audio.stop()
        isPlaying = false
    }
}