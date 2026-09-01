//
//  AudioPlayer.swift
//  tts-metal
//
//  Plays a queue of Float32 mono PCM chunks at a fixed 48 kHz via AVAudioEngine.
//  Chunks are scheduled back-to-back for gapless playback, with pause/resume/stop.
//
//  The graph is created once at 48 kHz and never reconfigured — reconnecting an
//  AVAudioEngine (engine.stop()/connect) on the main actor was a source of severe
//  main-thread hangs between utterances. All audio therefore arrives at 48 kHz
//  (the producer resamples the raw model output when needed).
//
//  `reserveSlot(limit:)` provides backpressure so the producer can generate ahead
//  of playback with a bounded look-ahead window instead of synthesizing the whole
//  document up front.
//
//  Reader mode adds *tagged* buffers: each scheduled buffer carries a sentence id
//  and fires a completion callback when it finishes playing, which drives the
//  reader's sentence-advance/karaoke/bookmark logic. A master GainNode provides
//  user volume for both modes.
//

import Foundation
import AVFoundation

@MainActor
final class AudioPlayer {
    static let shared = AudioPlayer()

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat

    private var pending = 0          // buffers scheduled but not yet finished playing
    private var generation = 0       // bumped on stop() to invalidate stale completions
    private var userPaused = false

    // Tagged completions (reader mode): sentence id per scheduled buffer, FIFO.
    private var tagQueue: [(tag: Int, onDone: ((Int) -> Void)?)] = []

    // Backpressure: producers awaiting a free look-ahead slot.
    private var slotWaiters: [CheckedContinuation<Void, Never>] = []

    /// Called on the main actor when the last scheduled (untagged) buffer finishes playing.
    var onAllFinished: (() -> Void)?

    private init() {
        format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                               sampleRate: TtsConfig.enhancedSampleRate,
                               channels: 1, interleaved: false)!
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
    }

    var isActive: Bool { pending > 0 }
    var isPaused: Bool { userPaused }
    var queueDepth: Int { pending }
    /// Session token — producers compare against their captured value to detect stops.
    var currentGeneration: Int { generation }

    // MARK: - Volume

    var volume: Float {
        get { player.volume }
        set { player.volume = max(0, min(1, newValue)) }
    }

    // MARK: - Backpressure

    /// Suspend until fewer than `limit` buffers are queued (or the session is stopped).
    func reserveSlot(limit: Int) async {
        let gen = generation
        while pending >= limit && gen == generation {
            await withCheckedContinuation { slotWaiters.append($0) }
        }
    }

    private func releaseWaiters() {
        guard !slotWaiters.isEmpty else { return }
        let waiters = slotWaiters
        slotWaiters.removeAll()
        for w in waiters { w.resume() }
    }

    private func startEngineIfNeeded() {
        if !engine.isRunning {
            engine.prepare()
            do { try engine.start() }
            catch { print("[AudioPlayer] engine start failed: \(error)") }
        }
    }

    /// Starts the audio hardware ahead of the first `enqueue`/`enqueueTagged`
    /// call. `AVAudioEngine.start()` negotiates with the system's audio HAL
    /// and can take well over 100ms the first time it runs — measured via
    /// PerfLog, that cost otherwise landed entirely on the very first
    /// sentence's time-to-first-audio. Call this once, early (e.g. while the
    /// TTS model is still loading), so it overlaps with other startup work
    /// instead of adding to playback latency.
    func prewarm() {
        startEngineIfNeeded()
    }

    private func makeBuffer(_ samples: [Float]) -> AVAudioPCMBuffer? {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(samples.count)) else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channel = buffer.floatChannelData![0]
        samples.withUnsafeBufferPointer { p in
            for i in 0..<samples.count { channel[i] = max(-1.0, min(1.0, p[i])) }
        }
        return buffer
    }

    // MARK: - Untagged playback (menu-bar selection reading)

    /// Append a chunk (already at 48 kHz) to the playback queue. Plays immediately
    /// unless paused.
    func enqueue(_ samples: [Float]) {
        guard let buffer = makeBuffer(samples) else { return }
        startEngineIfNeeded()
        pending += 1
        let gen = generation
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in self?.bufferFinished(gen) }
        }
        if !userPaused { player.play() }
    }

    private func bufferFinished(_ gen: Int) {
        guard gen == generation else { return }   // belongs to a stopped session
        pending -= 1
        if pending <= 0 {
            pending = 0
            onAllFinished?()
        }
        releaseWaiters()                          // a look-ahead slot opened up
    }

    // MARK: - Tagged playback (reader mode)

    /// Schedule a sentence's audio; `onDone(tag)` fires on the main actor once that
    /// sentence has fully played back.
    func enqueueTagged(_ samples: [Float], tag: Int, onDone: ((Int) -> Void)? = nil) {
        guard let buffer = makeBuffer(samples) else { return }
        startEngineIfNeeded()
        pending += 1
        tagQueue.append((tag, onDone))
        let gen = generation
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in self?.taggedBufferFinished(gen) }
        }
        if !userPaused { player.play() }
    }

    private func taggedBufferFinished(_ gen: Int) {
        guard gen == generation else { return }
        let entry = tagQueue.isEmpty ? nil : tagQueue.removeFirst()
        pending -= 1
        if pending <= 0 {
            pending = 0
            if entry?.onDone == nil { onAllFinished?() }
        }
        releaseWaiters()
        if let entry = entry, let cb = entry.onDone {
            cb(entry.tag)
        }
    }

    // MARK: - Transport

    func pause() {
        userPaused = true
        player.pause()
    }

    func resume() {
        guard userPaused else { return }
        userPaused = false
        startEngineIfNeeded()
        player.play()
    }

    /// Drop everything currently queued without invalidating this session.
    func flushQueue() {
        tagQueue.removeAll()
        pending = 0
        player.stop()
        releaseWaiters()
        generation += 1   // invalidate stale callbacks from flushed buffers
    }

    func stop() {
        generation += 1
        pending = 0
        tagQueue.removeAll()
        userPaused = false
        player.stop()
        releaseWaiters()                          // unblock any producer waiting on a slot
    }
}
