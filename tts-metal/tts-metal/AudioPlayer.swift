//
//  AudioPlayer.swift
//  tts-metal
//
//  Plays Float32 mono PCM at 24 kHz via AVAudioEngine.
//

import Foundation
import AVFoundation

final class AudioPlayer {
    static let shared = AudioPlayer()

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var pcmFormat: AVAudioFormat?
    private var lastBuffer: AVAudioPCMBuffer?

    init() {
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: pcmFormat)
    }

    private func ensureEngine() {
        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                print("[AudioPlayer] Failed to start engine: \(error)")
            }
        }
        if !player.isPlaying { player.play() }
    }

    func play(_ samples: [Float], sampleRate: Double) {
        guard !samples.isEmpty else { return }

        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: sampleRate,
                                   channels: 1,
                                   interleaved: false)
        guard let format = format, let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                                                 frameCapacity: AVAudioFrameCount(samples.count)) else {
            return
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channelData = buffer.floatChannelData![0]
        samples.withUnsafeBufferPointer { ptr in
            for i in 0..<samples.count {
                channelData[i] = max(-1.0, min(1.0, ptr[i]))
            }
        }

        if engine.mainMixerNode.outputFormat(forBus: 0).sampleRate != format.sampleRate {
            engine.disconnectNodeInput(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
        } else if pcmFormat == nil {
            engine.disconnectNodeInput(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
        }
        pcmFormat = format

        ensureEngine()
        player.stop()
        lastBuffer = buffer
        player.scheduleBuffer(buffer, at: nil, options: [.interrupts], completionHandler: nil)
        ensureEngine()
    }

    func stop() {
        player.stop()
        lastBuffer = nil
    }
}