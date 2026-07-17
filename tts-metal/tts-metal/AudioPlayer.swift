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
//  (the producer resamples the raw 24 kHz TTS output when enhancement is off).
//
//  `reserveSlot(limit:)` provides backpressure so the producer can generate ahead
//  of playback with a bounded look-ahead window instead of synthesizing the whole
//  document up front.
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

    // Backpressure: producers awaiting a free look-ahead slot.
    private var slotWaiters: [CheckedContinuation<Void, Never>] = []

    /// Called on the main actor when the last scheduled buffer finishes playing.
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

    /// Suspend until fewer than `limit` buffers are queued (or the session is stopped).
    /// Lets the producer stay at most `limit` chunks ahead of playback.
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

    /// Append a chunk (already at 48 kHz) to the playback queue. Plays immediately
    /// unless paused.
    func enqueue(_ samples: [Float]) {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channel = buffer.floatChannelData![0]
        samples.withUnsafeBufferPointer { p in
            for i in 0..<samples.count { channel[i] = max(-1.0, min(1.0, p[i])) }
        }

        startEngineIfNeeded()
        let gen = generation
        pending += 1
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

    func stop() {
        generation += 1
        pending = 0
        userPaused = false
        player.stop()
        releaseWaiters()                          // unblock any producer waiting on a slot
    }
}
