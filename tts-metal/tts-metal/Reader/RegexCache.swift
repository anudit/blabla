//
//  RegexCache.swift
//  tts-metal
//
//  Process-wide cache of compiled NSRegularExpression instances. Loading a big
//  EPUB/DOCX/etc. runs dozens of fixed regex patterns per paragraph (HTML
//  parsing, sentence splitting, TTS text normalization); recompiling each
//  pattern on every call dominated document load time. Compile once, reuse.
//

import Foundation

enum RegexCache {
    private static var cache: [String: NSRegularExpression] = [:]
    private static let lock = NSLock()

    static func regex(_ pattern: String, options: NSRegularExpression.Options = []) -> NSRegularExpression? {
        let key = "\(options.rawValue):\(pattern)"
        lock.lock()
        if let cached = cache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        guard let re = try? NSRegularExpression(pattern: pattern, options: options) else { return nil }
        lock.lock()
        cache[key] = re
        lock.unlock()
        return re
    }

    /// Cached-regex equivalent of `String.replacingOccurrences(of:with:options:.regularExpression)`.
    static func replace(_ input: String, pattern: String, with template: String,
                        options: NSRegularExpression.Options = []) -> String {
        guard let re = regex(pattern, options: options) else { return input }
        let range = NSRange(input.startIndex..., in: input)
        return re.stringByReplacingMatches(in: input, range: range, withTemplate: template)
    }

    /// Cached-regex equivalent of `String.range(of:options:.regularExpression)`.
    static func firstMatchRange(_ input: String, pattern: String,
                                options: NSRegularExpression.Options = []) -> Range<String.Index>? {
        guard let re = regex(pattern, options: options) else { return nil }
        let range = NSRange(input.startIndex..., in: input)
        guard let m = re.firstMatch(in: input, range: range) else { return nil }
        return Range(m.range, in: input)
    }
}
