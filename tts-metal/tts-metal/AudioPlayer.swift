//
//  AudioPlayer.swift
//  tts-metal
//
//  Plays a queue of Float32 mono PCM chunks at 24 kHz via AVAudioEngine.
//  Chunks are scheduled back-to-back for gapless playback, with pause/resume/stop.
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

    /// Called on the main actor when the last scheduled buffer finishes playing.
    var onAllFinished: (() -> Void)?

    private init() {
        format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                               sampleRate: TtsConfig.sampleRate,
                               channels: 1, interleaved: false)!
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
    }

    var isActive: Bool { pending > 0 }
    var isPaused: Bool { userPaused }

    private func startEngineIfNeeded() {
        if !engine.isRunning {
            engine.prepare()
            do { try engine.start() }
            catch { print("[AudioPlayer] engine start failed: \(error)") }
        }
    }

    /// Append a chunk to the playback queue. Plays immediately unless paused.
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
    }
}
