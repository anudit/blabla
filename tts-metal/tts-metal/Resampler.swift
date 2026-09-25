//
//  Resampler.swift
//  tts-metal
//
//  Created by Antigravity on 2026-07-20.
//

import Foundation
import Accelerate

enum Resampler {
    /// High-quality arbitrary-ratio resample via a Lanczos-windowed sinc kernel.
    ///
    /// When the rate ratio reduces to a small fraction (44.1k → 48k is 160/147) every
    /// output sample falls on one of `up` fixed sub-sample phases, so the kernel weights
    /// are tabulated once per phase and each output becomes a single dot product. The
    /// direct form below evaluated two `sin` calls per tap — ~100 ms for a 5.5 s sentence,
    /// sitting on the time-to-first-audio path. Output matches the direct form.
    static func resample(_ x: [Float], from srcSR: Double, to dstSR: Double, a: Int = 16) -> [Float] {
        if x.isEmpty { return x }
        if let (up, down) = rationalRatio(srcSR, dstSR), up <= 1024 {
            return resamplePolyphase(x, up: up, down: down, a: a)
        }
        return resampleDirect(x, from: srcSR, to: dstSR, a: a)
    }

    /// dstSR/srcSR as up/down in lowest terms, when both rates are whole numbers.
    private static func rationalRatio(_ srcSR: Double, _ dstSR: Double) -> (Int, Int)? {
        guard srcSR == srcSR.rounded(), dstSR == dstSR.rounded(), srcSR > 0, dstSR > 0 else { return nil }
        var p = Int(dstSR), q = Int(srcSR)
        var g = p, h = q
        while h != 0 { (g, h) = (h, g % h) }
        p /= g; q /= g
        return (p, q)
    }

    private struct PhaseTable { let taps: Int; let first: Int; let weights: [Float] }   // [up][taps]
    private static var tableCache: [String: PhaseTable] = [:]
    private static let tableLock = NSLock()

    /// Kernel weights for each of the `up` phases, laid out exactly as the direct form's
    /// tap loop: output o reads source samples center-half ... center+half+1 where
    /// center = floor(o·down/up), with t = (pos − k)·scale.
    private static func phaseTable(up: Int, down: Int, a: Int) -> PhaseTable {
        let key = "\(up)/\(down)/\(a)"
        tableLock.lock(); defer { tableLock.unlock() }
        if let t = tableCache[key] { return t }
        let ratio = Double(up) / Double(down)
        let scale = min(1.0, ratio)
        let half = Int((Double(a) / scale).rounded(.up))
        let taps = 2 * half + 2
        var w = [Float](repeating: 0, count: up * taps)
        for ph in 0..<up {
            let frac = Double(ph) / Double(up)          // pos − center
            for j in 0..<taps {
                let t = (frac + Double(half - j)) * scale  // k = center − half + j
                w[ph * taps + j] = Float(lanczos(t, Double(a)))
            }
        }
        let t = PhaseTable(taps: taps, first: -half, weights: w)
        tableCache[key] = t
        return t
    }

    private static func resamplePolyphase(_ x: [Float], up: Int, down: Int, a: Int) -> [Float] {
        let outN = Int((Double(x.count) * Double(up) / Double(down)).rounded())
        if outN <= 0 { return [] }
        let tab = phaseTable(up: up, down: down, a: a)
        let taps = tab.taps, n = x.count
        // Per-phase normalization (the direct form divides by the sum of weights of the
        // taps that land inside the signal; for interior samples that is all of them).
        var norms = [Float](repeating: 0, count: up)
        tab.weights.withUnsafeBufferPointer { wp in
            for ph in 0..<up { vDSP_sve(wp.baseAddress! + ph * taps, 1, &norms[ph], vDSP_Length(taps)) }
        }
        var out = [Float](repeating: 0, count: outN)
        x.withUnsafeBufferPointer { xp in
            tab.weights.withUnsafeBufferPointer { wp in
                out.withUnsafeMutableBufferPointer { op in
                    for o in 0..<outN {
                        let num = o * down
                        let center = num / up, ph = num - center * up
                        let k0 = center + tab.first
                        let w = wp.baseAddress! + ph * taps
                        if k0 >= 0 && k0 + taps <= n {
                            var acc: Float = 0
                            vDSP_dotpr(xp.baseAddress! + k0, 1, w, 1, &acc, vDSP_Length(taps))
                            op[o] = norms[ph] != 0 ? acc / norms[ph] : 0
                        } else {
                            // Edge: only taps inside the signal count, renormalized over them.
                            var acc: Float = 0, norm: Float = 0
                            for j in 0..<taps {
                                let k = k0 + j
                                if k >= 0 && k < n { acc += xp[k] * w[j]; norm += w[j] }
                            }
                            op[o] = norm != 0 ? acc / norm : 0
                        }
                    }
                }
            }
        }
        return out
    }

    private static func resampleDirect(_ x: [Float], from srcSR: Double, to dstSR: Double, a: Int) -> [Float] {
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
