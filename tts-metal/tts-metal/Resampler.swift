//
//  Resampler.swift
//  tts-metal
//
//  Created by Antigravity on 2026-07-20.
//

import Foundation

enum Resampler {
    /// High-quality arbitrary-ratio resample via a Lanczos-windowed sinc kernel.
    static func resample(_ x: [Float], from srcSR: Double, to dstSR: Double, a: Int = 16) -> [Float] {
        if x.isEmpty { return x }
        let ratio = dstSR / srcSR
        let outN = Int((Double(x.count) * ratio).rounded())
        if outN <= 0 { return [] }
        var out = [Float](repeating: 0, count: outN)
        let n = x.count
        let scale = min(1.0, ratio)          // kernel cutoff in source samples
        let af = Double(a)
        x.withUnsafeBufferPointer { xp in
            for o in 0..<outN {
                let pos = Double(o) / ratio
                let center = Int(pos.rounded(.down))
                let half = Int((af / scale).rounded(.up))
                var acc = 0.0, norm = 0.0
                var k = center - half
                while k <= center + half + 1 {
                    if k >= 0 && k < n {
                        let t = (pos - Double(k)) * scale
                        let wgt = lanczos(t, af)
                        acc += Double(xp[k]) * wgt
                        norm += wgt
                    }
                    k += 1
                }
                out[o] = Float(norm != 0 ? acc / norm : 0)
            }
        }
        return out
    }
    private static func lanczos(_ t: Double, _ a: Double) -> Double {
        if t == 0 { return 1 }
        if t <= -a || t >= a { return 0 }
        let pt = Double.pi * t
        return (sin(pt) / pt) * (sin(pt / a) / (pt / a))
    }
}
