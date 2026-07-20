//
//  TtsController.swift
//  tts-metal
//
//  Drives the menu-bar TTS: loads the model, reads the system selection, chunks long
//  text, and streams synthesized chunks into the gapless audio queue with transport
//  (play / pause / stop) controls.
//

import Foundation
import Metal
import SwiftUI
import Combine
import Carbon.HIToolbox

@MainActor
final class TtsController: ObservableObject {

    enum Phase: Equatable {
        case loading, idle, generating, speaking, paused, error(String)
    }

    @Published private(set) var phase: Phase = .loading {
        didSet { updateWaveAnimation() }
    }
    @Published private(set) var statusText: String = "Loading model…"
    @Published private(set) var accessibilityGranted: Bool = SelectionReader.isTrusted
    /// Animated speaker-wave level (1…3) while speaking; drives the toolbar icon.
    @Published private(set) var waveLevel: Int = 2

    // Persisted user settings.
    @Published var speed: Double = (UserDefaults.standard.object(forKey: "speed") as? Double) ?? 1.0 {
        didSet { UserDefaults.standard.set(speed, forKey: "speed") }
    }

    @Published private(set) var supertonicReady: Bool = false
    /// Supertonic voice (M1..M5, F1..F5).
    @Published var supertonicVoice: String = UserDefaults.standard.string(forKey: "stVoice") ?? "M1" {
        didSet { UserDefaults.standard.set(supertonicVoice, forKey: "stVoice") }
    }

    private var supertonic: SupertonicEngine?
    private let audio = AudioPlayer.shared
    private var loaded = false

    private var hotKey: GlobalHotKey?
    private var speakTask: Task<Void, Never>?
    private var waveAnimTask: Task<Void, Never>?
    private var totalChunks = 0
    private var enqueuedChunks = 0
    private var hasPromptedForAccess = false

    init() {
        audio.onAllFinished = { [weak self] in self?.onPlaybackDrained() }
        registerHotKey()
        startAccessibilityWatch()
        Task { await load() }
    }

    // MARK: - Derived UI state

    var isBusy: Bool { phase == .generating || phase == .speaking || phase == .paused }
    var canControl: Bool { loaded }

    /// SF Symbol for the toolbar item, reflecting the current state.
    var menuBarIcon: String {
        switch phase {
        case .loading:               return "hourglass"
        case .idle:                  return "speaker.slash.fill"
        case .generating, .speaking: return "speaker.wave.\(waveLevel).fill"
        case .paused:                return "pause.fill"
        case .error:                 return "exclamationmark.triangle.fill"
        }
    }

    var statusColor: Color {
        switch phase {
        case .loading:    return .orange
        case .error:      return .red
        case .idle:       return .secondary
        default:          return .green
        }
    }

    // MARK: - Toolbar wave animation

    private func updateWaveAnimation() {
        let animate = (phase == .generating || phase == .speaking)
        if animate {
            guard waveAnimTask == nil else { return }
            waveAnimTask = Task { @MainActor in
                var level = 1
                while !Task.isCancelled {
                    waveLevel = level
                    level = level % 3 + 1
                    try? await Task.sleep(nanoseconds: 350_000_000)
                }
            }
        } else {
            waveAnimTask?.cancel()
            waveAnimTask = nil
        }
    }

    // MARK: - Model load

    private func load() async {
        statusText = "Loading Metal device…"
        guard let device = MTLCreateSystemDefaultDevice() else {
            fail("Metal not available."); return
        }
        guard let queue = device.makeCommandQueue() else {
            fail("Could not create command queue."); return
        }

        statusText = "Loading Supertonic models…"
        // Load Supertonic 3 (Metal)
        let stEng = SupertonicEngine(device: device, queue: queue)
        let stOk = await Task.detached(priority: .utility) { () -> Bool in
            do { try stEng.load(); return true }
            catch { print("[Supertonic] Metal load failed: \(error)"); return false }
        }.value

        if stOk {
            supertonic = stEng
            supertonicReady = true
            loaded = true
            phase = .idle
            statusText = "Ready — select text, then press ⌥⌘R"
        } else {
            fail("Supertonic load failed.")
        }

        // Supertonic 3 on-device smoke test (off unless SUPERTONIC_SELFTEST=1).
        if ProcessInfo.processInfo.environment["SUPERTONIC_SELFTEST"] == "1" {
            statusText = "Supertonic self-test…"
            await Task.detached(priority: .utility) {
                let st = SupertonicEngine(device: device, queue: queue)
                st.selfTest()
            }.value
        }
        // Stage-by-stage numerical validation against /tmp/st_ref (ST_VALIDATE=1).
        if ProcessInfo.processInfo.environment["ST_VALIDATE"] == "1" {
            let st = SupertonicEngine(device: device, queue: queue)
            st.validate()
        }
    }

    private func fail(_ message: String) {
        phase = .error(message)
        statusText = message
    }

    // MARK: - Hotkey

    private func registerHotKey() {
        // ⌥⌘R reads the current selection from anywhere in the system.
        hotKey = GlobalHotKey(keyCode: UInt32(kVK_ANSI_R),
                              modifiers: UInt32(cmdKey | optionKey)) { [weak self] in
            Task { @MainActor in self?.readSelection() }
        }
    }

    // MARK: - Accessibility

    /// Poll trust state so the permission banner clears automatically once granted
    /// (and reappears if access is later revoked) without needing to reopen the popover.
    private func startAccessibilityWatch() {
        Task { @MainActor in
            while !Task.isCancelled {
                refreshAccessibility()
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
        }
    }

    func refreshAccessibility() {
        let trusted = SelectionReader.isTrusted
        if trusted != accessibilityGranted { accessibilityGranted = trusted }
        if trusted { hasPromptedForAccess = false }  // allow a fresh prompt if revoked later
    }

    func requestAccessibility() {
        SelectionReader.requestTrust()
        Task {
            try? await Task.sleep(nanoseconds: 500_000_000)
            refreshAccessibility()
        }
    }

    // MARK: - Read selection → speak

    func readSelection() {
        guard loaded else { return }
        guard SelectionReader.isTrusted else {
            accessibilityGranted = false
            statusText = "Enable Accessibility for tts-metal, then relaunch."
            // Prompt at most once per launch so repeated ⌥⌘R presses don't spam the dialog.
            if !hasPromptedForAccess {
                hasPromptedForAccess = true
                requestAccessibility()
            }
            return
        }
        statusText = "Reading selection…"
        Task {
            let text = await Task.detached(priority: .userInitiated) {
                SelectionReader.currentSelection()
            }.value
            let clean = text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if clean.isEmpty {
                statusText = "No text selected."
                if phase != .generating && phase != .speaking { phase = .idle }
                return
            }
            speak(clean)
        }
    }

    func speak(_ rawText: String) {
        guard loaded, let stEngine = supertonic else { return }
        let chunks = TextChunker.chunk(rawText)
        guard !chunks.isEmpty else { return }

        stop()                     // cancel anything in flight, reset the queue
        totalChunks = chunks.count
        enqueuedChunks = 0
        phase = .generating
        statusText = chunks.count > 1 ? "Synthesizing \(chunks.count) chunks…" : "Synthesizing…"

        let rate = Float(speed)
        // The audio graph runs at a fixed 48 kHz; everything we enqueue is 48 kHz.
        let outRate = TtsConfig.enhancedSampleRate
        let stVoice = supertonicVoice
        speakTask = Task { @MainActor in
            for (index, chunk) in chunks.enumerated() {
                if Task.isCancelled { return }
                // Sliding-window backpressure: stay at most 5 chunks ahead of playback,
                // so generation of the next sentence overlaps playback of the current one
                // without synthesizing the whole document up front.
                await audio.reserveSlot(limit: 5)
                if Task.isCancelled { return }
                let wave: [Float]
                do {
                    wave = try await Task.detached(priority: .userInitiated) { () -> [Float] in
                        // Supertonic 3 path (Metal compute, 44.1 kHz → resample to 48 kHz graph).
                        let w = try stEngine.generate(chunk, voiceName: stVoice, speed: max(0.7, min(2.0, rate * 1.05)))
                        if w.isEmpty { return [] }
                        return Resampler.resample(w, from: Double(stEngine.sampleRate), to: outRate)
                    }.value
                } catch {
                    statusText = "Error: \(error.localizedDescription)"
                    continue
                }
                if Task.isCancelled { return }
                enqueuedChunks += 1
                if !wave.isEmpty { audio.enqueue(wave) }
                if phase == .generating && !audio.isPaused { phase = .speaking }
                if chunks.count > 1 {
                    statusText = "Speaking \(index + 1) of \(chunks.count)…"
                }
            }
            // If everything already drained while we were still generating, settle now.
            if !audio.isActive && phase != .paused { finishSpeaking() }
        }
    }

    private func onPlaybackDrained() {
        // Only truly finished once every chunk has been enqueued (and not paused).
        guard enqueuedChunks >= totalChunks, phase != .paused else { return }
        finishSpeaking()
    }

    private func finishSpeaking() {
        phase = .idle
        statusText = "Ready — select text, then press ⌥⌘R"
    }

    // MARK: - Transport

    /// Play/pause primary action. When idle, reads the current selection.
    func togglePlayPause() {
        switch phase {
        case .speaking, .generating:
            audio.pause()
            phase = .paused
            statusText = "Paused"
        case .paused:
            audio.resume()
            phase = audio.isActive ? .speaking : .generating
            statusText = "Speaking…"
        case .idle, .error:
            readSelection()
        case .loading:
            break
        }
    }

    func stop() {
        speakTask?.cancel()
        speakTask = nil
        audio.stop()
        totalChunks = 0
        enqueuedChunks = 0
        if loaded, phase != .loading {
            phase = .idle
            statusText = "Ready — select text, then press ⌥⌘R"
        }
    }
}
