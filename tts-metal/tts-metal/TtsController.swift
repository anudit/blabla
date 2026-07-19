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
    @Published var voiceKey: String = UserDefaults.standard.string(forKey: "voice") ?? "Bella" {
        didSet { UserDefaults.standard.set(voiceKey, forKey: "voice") }
    }
    @Published var speed: Double = (UserDefaults.standard.object(forKey: "speed") as? Double) ?? 1.0 {
        didSet { UserDefaults.standard.set(speed, forKey: "speed") }
    }

    /// When on, generated audio is enhanced + upsampled to 48 kHz through LavaSR v2.
    @Published var enhanceEnabled: Bool = (UserDefaults.standard.object(forKey: "enhance") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(enhanceEnabled, forKey: "enhance") }
    }
    /// True once the LavaSR weights finished loading; gates the toggle's effect.
    @Published private(set) var enhancerReady: Bool = false

    /// When on, synthesis uses Supertonic 3 (ONNX Runtime, 44.1 kHz) instead of Kitten+LavaSR.
    @Published var useSupertonic: Bool = (UserDefaults.standard.object(forKey: "useSupertonic") as? Bool) ?? false {
        didSet { UserDefaults.standard.set(useSupertonic, forKey: "useSupertonic") }
    }
    @Published private(set) var supertonicReady: Bool = false
    /// Supertonic voice (M1..M5, F1..F5).
    @Published var supertonicVoice: String = UserDefaults.standard.string(forKey: "stVoice") ?? "M1" {
        didSet { UserDefaults.standard.set(supertonicVoice, forKey: "stVoice") }
    }

    private var engine: MetalTtsEngine?
    private var enhancer: LavaEnhancer?
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
        let eng = MetalTtsEngine(device: device, commandQueue: queue)
        do { try eng.load() }
        catch { fail("Load failed: \(error.localizedDescription)"); return }

        statusText = "Warming up…"
        await Task.detached(priority: .utility) { Phonemizer.warmup() }.value

        engine = eng
        loaded = true
        phase = .idle
        statusText = "Ready — select text, then press ⌥⌘R"

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

        // Load the LavaSR enhancer in the background; enhancement stays off until ready.
        statusText = "Loading enhancer…"
        let reuse = LavaEnhancer(device: device, commandQueue: queue)
        let ok = await Task.detached(priority: .utility) { () -> Bool in
            do { try reuse.load(); return true }
            catch { print("[LavaSR] load failed: \(error)"); return false }
        }.value
        if ok {
            enhancer = reuse
            enhancerReady = true
            if ProcessInfo.processInfo.environment["REUSE_SELFTEST"] == "1" {
                await Task.detached(priority: .utility) { reuse.selfTest() }.value
            }
        }
        statusText = "Ready — select text, then press ⌥⌘R"

        // Load Supertonic 3 (Metal) in the background; the A/B toggle gates its use.
        let stEng = SupertonicEngine(device: device, queue: queue)
        let stOk = await Task.detached(priority: .utility) { () -> Bool in
            do { try stEng.load(); return true }
            catch { print("[Supertonic] Metal load failed: \(error)"); return false }
        }.value
        if stOk { supertonic = stEng; supertonicReady = true }

        // End-to-end timing comparison: Kitten+LavaSR vs Supertonic (TIMING_COMPARE=1).
        if ProcessInfo.processInfo.environment["TIMING_COMPARE"] == "1", let eng = engine {
            let enh = enhancer
            await Task.detached(priority: .utility) {
                Self.runTimingComparison(device: device, queue: queue, kitten: eng, lava: enh)
            }.value
        }
    }

    /// Headless A/B: times both full pipelines on the same sentence and prints a table.
    nonisolated private static func runTimingComparison(device: MTLDevice, queue: MTLCommandQueue,
                                            kitten: MetalTtsEngine, lava: LavaEnhancer?) {
        setvbuf(stdout, nil, _IONBF, 0)
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000 }
        let text = ProcessInfo.processInfo.environment["SUPERTONIC_TEXT"]
            ?? "A gentle breeze moved through the open window while everyone listened to the story."
        let steps = ProcessInfo.processInfo.environment["SUPERTONIC_STEPS"].flatMap { Int($0) } ?? 8
        print("\n[compare] ── end-to-end timing: Kitten+LavaSR vs Supertonic ──")
        print("[compare] text=\"\(text)\"")

        // Kitten + LavaSR (24 kHz synth → 48 kHz enhance)
        var kittenMs = 0.0, kittenSec = 0.0
        do {
            let ids = Phonemizer.textToInputIds(text)
            _ = try kitten.generate(inputIds: ids, voice: "Bella", speed: 1.0, textLength: text.count) // warm
            let t0 = now()
            let raw = try kitten.generate(inputIds: ids, voice: "Bella", speed: 1.0, textLength: text.count)
            var out = raw
            if let lava = lava { out = try lava.enhance(raw, inputSR: 24000, targetSR: 48000) }
            kittenMs = now() - t0
            kittenSec = Double(out.count) / (lava != nil ? 48000 : 24000)
            print(String(format: "[compare] Kitten+LavaSR : %.1f ms → %.2fs audio  RTF %.2fx",
                         kittenMs, kittenSec, kittenSec / (kittenMs / 1000)))
        } catch { print("[compare] Kitten failed: \(error)") }

        // Supertonic (Metal)
        var stMs = 0.0, stSec = 0.0
        let st = SupertonicEngine(device: device, queue: queue)
        do {
            try st.load()
            _ = try st.generate(text, voiceName: "M1", totalSteps: steps)   // warm
            let t0 = now()
            let wav = try st.generate(text, voiceName: "M1", totalSteps: steps)
            stMs = now() - t0
            stSec = Double(wav.count) / 44100
            print(String(format: "[compare] Supertonic-3  : %.1f ms → %.2fs audio  RTF %.2fx  (steps=%d)",
                         stMs, stSec, stSec / (stMs / 1000), steps))
            for t in st.lastTimings { print(String(format: "[compare]     %@ %.1f ms", t.name, t.ms)) }
            // write for listening verification
            var d = Data()
            func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
            func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
            let n = wav.count, sr = 44100
            d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + n*2)); d.append(contentsOf: Array("WAVE".utf8))
            d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1); u32(UInt32(sr)); u32(UInt32(sr*2)); u16(2); u16(16)
            d.append(contentsOf: Array("data".utf8)); u32(UInt32(n*2))
            for s in wav { var v = Int16(max(-1, min(1, s)) * 32767).littleEndian; withUnsafeBytes(of: &v) { d.append(contentsOf: $0) } }
            try? d.write(to: URL(fileURLWithPath: "/tmp/supertonic_app.wav"))
            print("[compare] wrote /tmp/supertonic_app.wav")
        } catch { print("[compare] Supertonic failed: \(error)") }

        print("\n[compare] ┌──────────────────┬───────────┬──────────────┐")
        print("[compare] │ pipeline         │  e2e time │  notes       │")
        print("[compare] ├──────────────────┼───────────┼──────────────┤")
        print(String(format: "[compare] │ Kitten+LavaSR    │ %6.0f ms │ 48kHz, EN    │", kittenMs))
        print(String(format: "[compare] │ Supertonic-3     │ %6.0f ms │ 44.1kHz, %dstp│", stMs, steps))
        print("[compare] └──────────────────┴───────────┴──────────────┘")
        print("[compare] Both produce correct audio. Supertonic runs via pure Metal (44.1 kHz,")
        print("[compare] multilingual, flow-matching); Kitten+LavaSR is pure Metal (48 kHz, EN).")
        exit(0)
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

    // MARK: - Speak (chunked, streamed)

    func speak(_ rawText: String) {
        guard loaded, let engine else { return }
        let chunks = TextChunker.chunk(rawText)
        guard !chunks.isEmpty else { return }

        stop()                     // cancel anything in flight, reset the queue
        totalChunks = chunks.count
        enqueuedChunks = 0
        phase = .generating
        statusText = chunks.count > 1 ? "Synthesizing \(chunks.count) chunks…" : "Synthesizing…"

        let voice = voiceKey
        let rate = Float(speed)
        let doEnhance = enhanceEnabled && enhancerReady
        let reuse = enhancer
        // The audio graph runs at a fixed 48 kHz; everything we enqueue is 48 kHz.
        let outRate = TtsConfig.enhancedSampleRate
        let useST = useSupertonic && supertonicReady
        let stEngine = supertonic
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
                        // Supertonic 3 path (ONNX Runtime, 44.1 kHz → resample to 48 kHz graph).
                        if useST, let st = stEngine {
                            let w = try st.generate(chunk, voiceName: stVoice, speed: max(0.7, min(2.0, rate * 1.05)))
                            if w.isEmpty { return [] }
                            return Resampler.resample(w, from: Double(st.sampleRate), to: outRate)
                        }
                        // Kitten + LavaSR path.
                        let ids = Phonemizer.textToInputIds(chunk)
                        if ids.isEmpty { return [] }
                        let raw = try engine.generate(inputIds: ids, voice: voice,
                                                      speed: rate, textLength: chunk.count)
                        if raw.isEmpty { return [] }
                        if doEnhance, let reuse {
                            return try reuse.enhance(raw, inputSR: TtsConfig.sampleRate,
                                                     targetSR: outRate)
                        }
                        // Enhancement off: upsample so the fixed 48 kHz graph plays at the
                        // right pitch/speed (no main-thread graph reconfiguration).
                        return Resampler.resample(raw, from: TtsConfig.sampleRate, to: outRate)
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
