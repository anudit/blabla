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

    private var supertonicReady: Bool = false
    /// Supertonic voice (M1..M5, F1..F5).
    @Published var supertonicVoice: String = UserDefaults.standard.string(forKey: "stVoice") ?? "M1" {
        didSet { UserDefaults.standard.set(supertonicVoice, forKey: "stVoice") }
    }

    private let hub = EngineHub.shared
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
        statusText = "Loading model…"
        hub.ensureLoaded()

        // 20 ms, matching ReaderController.waitForEngine: the load itself is only a few
        // hundred ms, so a 200 ms poll added up to that much again before ⌥⌘R worked.
        while !hub.ready && !hub.failed {
            statusText = hub.statusText
            try? await Task.sleep(nanoseconds: 20_000_000)
        }

        if hub.ready {
            supertonicReady = true
            loaded = true
            phase = .idle
            statusText = "Ready — select text, then press ⌥⌘R"
        } else {
            fail(hub.statusText)
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
        guard loaded, hub.ready else { return }
        let chunks = TextChunker.chunk(rawText)
        guard !chunks.isEmpty else { return }

        stop()                     // cancel anything in flight, reset the queue
        totalChunks = chunks.count
        enqueuedChunks = 0
        phase = .generating
        statusText = chunks.count > 1 ? "Synthesizing \(chunks.count) chunks…" : "Synthesizing…"

        let rate = Float(speed)
        let stVoice = supertonicVoice
        let gen = audio.currentGeneration
        // Queue the ANE cells these chunks need (only the common ones load at launch).
        let hub = self.hub
        Task { await hub.prefetch(Array(chunks.prefix(8)), speed: rate) }
        speakTask = Task { @MainActor in
            for (index, chunk) in chunks.enumerated() {
                if Task.isCancelled { return }
                // Sliding-window backpressure: stay at most 5 chunks ahead of playback,
                // so generation of the next sentence overlaps playback of the current one
                // without synthesizing the whole document up front.
                await audio.reserveSlot(limit: 5)
                if Task.isCancelled || gen != audio.currentGeneration { return }
                let wave: [Float]
                do {
                    wave = try await hub.generate(chunk, voice: stVoice, speed: rate)
                    if wave.isEmpty { continue }
                } catch {
                    statusText = "Error: \(error.localizedDescription)"
                    continue
                }
                if Task.isCancelled || gen != audio.currentGeneration { return }
                enqueuedChunks += 1
                audio.enqueue(wave)
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
