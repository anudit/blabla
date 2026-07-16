//
//  TextChunker.swift
//  tts-metal
//
//  Splits arbitrary (possibly very long) text into speakable chunks, preferring
//  sentence boundaries so each chunk sounds natural and stays small enough to
//  synthesize quickly.
//

import Foundation

enum TextChunker {
    /// Break `text` into chunks no longer than `maxChars`, splitting on sentence
    /// boundaries where possible and falling back to whitespace for over-long sentences.
    static func chunk(_ text: String, maxChars: Int = 300) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        // 1. Segment into sentences using Foundation's linguistic tokenizer.
        var sentences: [String] = []
        trimmed.enumerateSubstrings(in: trimmed.startIndex..<trimmed.endIndex,
                                    options: [.bySentences, .localized]) { sub, _, _, _ in
            if let s = sub?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty {
                sentences.append(s)
            }
        }
        if sentences.isEmpty { sentences = [trimmed] }

        // 2. Greedily pack sentences into chunks, splitting any that exceed maxChars.
        var chunks: [String] = []
        var current = ""
        func flush() {
            let c = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !c.isEmpty { chunks.append(c) }
            current = ""
        }
        for sentence in sentences {
            for piece in splitLong(sentence, maxChars: maxChars) {
                if current.isEmpty {
                    current = piece
                } else if current.count + 1 + piece.count <= maxChars {
                    current += " " + piece
                } else {
                    flush()
                    current = piece
                }
            }
        }
        flush()
        return chunks
    }

    /// Hard-wrap a single over-long sentence on whitespace so no piece exceeds maxChars.
    private static func splitLong(_ s: String, maxChars: Int) -> [String] {
        if s.count <= maxChars { return [s] }
        var out: [String] = []
        var buf = ""
        for token in s.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }) {
            let t = String(token)
            if buf.isEmpty {
                buf = t
            } else if buf.count + 1 + t.count <= maxChars {
                buf += " " + t
            } else {
                out.append(buf)
                buf = t
            }
        }
        if !buf.isEmpty { out.append(buf) }
        return out
    }
}
