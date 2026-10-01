//
//  TimeStretcher.swift
//  tts-metal
//
//  Pitch-preserving speed-up for synthesized speech.
//
//  Supertonic speeds speech up by shrinking the latent length it is asked to
//  fill, and past ~1.25× it stops fitting the text into that length: it drops
//  words, most visibly the last one (at 1.75× "Stop right there." came back as
//  "You"). So the model is asked for at most `maxModelSpeed`, and whatever is
//  left of the requested speed is applied here to the finished waveform.
//
//  Measured with SpeedTailTest + Whisper over 5 sentences × 1.5/1.75/2× × 3
//  seeds: model-only speed-up 2/15 last words right; capped at 1.3, 1.7% WER
//  (short sentences lose their first word); capped at 1.2, 0% WER.
//
//  AVAudioUnitTimePitch in an offline manual-rendering engine does the
//  stretch. It is driven synchronously from the engine queue, after synthesis,
//  so the samples a caller gets back are already at the final speed — sample
//  counts stay true for karaoke timing and the reader's cache, and playback
//  needs no changes.
//

import AVFoundation

final class TimeStretcher {
    /// Fastest speed the model is asked to synthesize at directly.
    static let maxModelSpeed: Float = 1.2

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let timePitch = AVAudioUnitTimePitch()
    private let format: AVAudioFormat
    private let renderBuffer: AVAudioPCMBuffer
    private let blockSize: AVAudioFrameCount = 4096

    /// Not thread-safe: one instance per serial queue.
    init?(sampleRate: Double) {
        guard let f = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                    channels: 1, interleaved: false) else { return nil }
        format = f
        // Spectral quality setting; the default is tuned for music and smears consonants.
        timePitch.overlap = 16
        engine.attach(player)
        engine.attach(timePitch)
        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)
        do {
            try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: blockSize)
            try engine.start()
        } catch {
            print("[TimeStretcher] unavailable: \(error)")
            return nil
        }
        guard let rb = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat,
                                        frameCapacity: blockSize) else { return nil }
        renderBuffer = rb
    }

    /// Plays `samples` back `rate`× faster at the same pitch. Output length is
    /// `samples.count / rate`. Offline rendering adds no processing delay, so
    /// the output lines up with the input from its first sample.
    func stretch(_ samples: [Float], rate: Float) -> [Float] {
        guard abs(rate - 1) > 0.005, !samples.isEmpty else { return samples }
        timePitch.rate = rate
        timePitch.reset()

        // Trailing silence flushes the unit's internal delay so the end of the
        // last word comes out instead of staying buffered.
        let flush = Int(format.sampleRate * 0.25)
        guard let input = AVAudioPCMBuffer(pcmFormat: format,
                                           frameCapacity: AVAudioFrameCount(samples.count + flush))
        else { return samples }
        input.frameLength = input.frameCapacity
        let dst = input.floatChannelData![0]
        samples.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: samples.count)
        }
        (dst + samples.count).update(repeating: 0, count: flush)

        player.scheduleBuffer(input, completionHandler: nil)
        player.play()
        defer { player.stop() }

        let wanted = Int((Double(samples.count) / Double(rate)).rounded())
        var out: [Float] = []
        out.reserveCapacity(wanted)
        while out.count < wanted {
            let n = AVAudioFrameCount(min(Int(blockSize), wanted - out.count))
            guard (try? engine.renderOffline(n, to: renderBuffer)) == .success else { break }
            let p = renderBuffer.floatChannelData![0]
            out.append(contentsOf: UnsafeBufferPointer(start: p, count: Int(renderBuffer.frameLength)))
        }
        return Array(out.prefix(wanted))
    }
}
