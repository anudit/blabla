//
//  TTSTextNormalizer.swift
//  tts-metal
//
//  Text cleanup before synthesis. Ported from blabla's inflect-tts/frontend.ts
//  cleanForTts: markdown/wiki stripping, punctuation translation, abbreviation,
//  number, money, date, time and ordinal expansion.
//

import Foundation

enum TTSTextNormalizer {

    private static let wordOverrides: [String: String] = [
        "Qwen3": "Qwen three", "PyTorch": "pie torch", "SQLite": "ess cue lite",
        "USB-C": "USB see", "RTX 3060": "R T X thirty sixty", "RTX 3090": "R T X thirty ninety",
        "RTX 4090": "R T X forty ninety", "RTX 5080": "R T X fifty eighty",
        "RTX 5090": "R T X fifty ninety",
    ]

    private static let monthNames = ["January", "February", "March", "April", "May", "June",
                                     "July", "August", "September", "October", "November", "December"]

    /// Full pipeline used before synthesis: structure-determining cleanup
    /// followed by content-only numeric expansion. See `cleanForTtsFast` /
    /// `expandNumericForms` for why this is split in two.
    static func cleanForTts(_ input: String) -> String {
        expandNumericForms(cleanForTtsFast(input))
    }

    /// Markdown/wiki stripping, punctuation translation, whitespace collapse,
    /// word overrides and abbreviation expansion ("Dr." → "Doctor").
    ///
    /// This is the subset of normalization that can change sentence
    /// *boundaries* — abbreviation expansion removes ambiguous mid-sentence
    /// periods before `SentenceSplitter` runs — so it must complete before a
    /// document's sentence array is built. Everything else (money/date/time/
    /// phone/version/ordinal expansion, in `expandNumericForms`) only rewrites
    /// content *within* an already-bounded sentence and is safe to defer:
    /// `ReaderDocument` builds sentences from this fast pass alone so a book
    /// is readable immediately, then upgrades sentence text to the full
    /// `cleanForTts` result in the background (see
    /// `ReaderDocument.withFullyNormalizedSentences()`), while playback
    /// applies `expandNumericForms` at the point of synthesis so audio is
    /// always fully normalized regardless of background timing.
    static func cleanForTtsFast(_ input: String) -> String {
        var t = input

        // Strip images, wiki citation refs, markdown links (keep link text).
        t = RegexCache.replace(t, pattern: #"!\[[^\]]*\]\([^)]*\)"#, with: "")
        t = RegexCache.replace(t, pattern: #"\[(?:\[?[0-9]{1,3}\]?)\]\(#[^)]*\)"#, with: "")
        t = RegexCache.replace(t, pattern: #"\[([^\]]+)\]\([^)]*\)"#, with: "$1")
        t = RegexCache.replace(t, pattern: #"\[\[[^\]]*\]\]\[?citenote[^]]*\]?"#, with: "")

        // Punctuation translation
        t = t.replacingOccurrences(of: "\u{2018}", with: "'")
        t = t.replacingOccurrences(of: "\u{2019}", with: "'")
        t = t.replacingOccurrences(of: "\u{201C}", with: "\"")
        t = t.replacingOccurrences(of: "\u{201D}", with: "\"")
        t = t.replacingOccurrences(of: "\u{2014}", with: ", ")
        t = t.replacingOccurrences(of: "\u{2026}", with: "...")
        t = t.replacingOccurrences(of: "[", with: ",").replacingOccurrences(of: "]", with: ",")
        t = t.replacingOccurrences(of: "{", with: "").replacingOccurrences(of: "}", with: "")

        // Collapse whitespace
        t = RegexCache.replace(t, pattern: "\\s+", with: " ")
        if t.count > 4000 { /* long docs handled by callers chunk-wise */ }
        t = t.trimmingCharacters(in: .whitespacesAndNewlines)

        t = wordOverrides.reduce(t) { $0.replacingOccurrences(of: $1.key, with: $1.value) }
        t = abbreviations(t)
        t = acronyms(t)
        return t
    }

    /// Money/date/time/phone/version/ordinal/digit-grouping expansion.
    /// Content-only — never introduces or removes a sentence terminator —
    /// and idempotent, so it's safe to apply more than once (e.g. once
    /// speculatively in the background, once for real at synthesis time).
    static func expandNumericForms(_ text: String) -> String {
        var t = money(text)
        t = dates(t)
        t = times(t)
        t = phoneNumbers(t)
        t = versions(t)
        t = numbers(t)
        return t
    }

    private static func abbreviations(_ s: String) -> String {
        var out = s
        let pairs: [(String, String)] = [
            (#"\bDr\."#, "Doctor"), (#"\bMr\."#, "Mister"), (#"\bMrs\."#, "Misses"),
            (#"\bMs\."#, "Miss"), (#"\bProf\."#, "Professor"), (#"\bSt\."#, "Saint"),
            (#"\bvs\."#, "versus"), (#"\betc\."#, "etcetera"),
            (#"\be\.g\."#, "for example"), (#"\bi\.e\."#, "that is"),
        ]
        for (p, r) in pairs {
            out = RegexCache.replace(out, pattern: p, with: r)
        }
        // "F.B.I." → "F B I"
        out = dottedAcronyms(out)
        return out
    }

    /// "F.B.I." style dotted sequences → "F B I".
    private static func dottedAcronyms(_ s: String) -> String {
        guard let re = RegexCache.regex(#"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?"#) else { return s }
        let ns = s as NSString
        var result = ""
        var cursor = 0
        re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m else { return }
            result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            let word = ns.substring(with: m.range).filter { $0 != "." }
            result += word.map(String.init).joined(separator: " ")
            cursor = m.range.location + m.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    /// Spell out ALL-CAPS acronyms of 2–5 letters ("NASA" stays; "API" → spoken naturally by model).
    private static func acronyms(_ s: String) -> String { s }

    private static func money(_ s: String) -> String {
        guard let re = RegexCache.regex(#"(\$|€|£)([0-9][0-9,]*(?:\.[0-9]+)?)([bBmMkK])?"#) else { return s }
        let ns = s as NSString
        var result = ""
        var cursor = 0
        re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m, m.numberOfRanges > 3 else { return }
            result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            let sym = ns.substring(with: m.range(at: 1))
            let amountStr = ns.substring(with: m.range(at: 2)).replacingOccurrences(of: ",", with: "")
            let suffix = m.range(at: 3).location != NSNotFound ? ns.substring(with: m.range(at: 3)) : ""

            var words = ""
            switch sym {
            case "$": words = "dollars"
            case "€": words = "euros"
            case "£": words = "pounds"
            default: break
            }
            let amount = Double(amountStr) ?? 0
            var spoken = spellNumber(amountStr) ?? String(amount)
            if suffix == "B" || suffix == "b" { spoken += " billion" }
            else if suffix == "M" || suffix == "m" { spoken += " million" }
            else if suffix == "K" || suffix == "k" { spoken += " thousand" }
            // Cents: "five dollars and fifty cents" (no trailing currency word).
            let hasCents = amountStr.contains(".") && amountStr[amountStr.index(after: amountStr.firstIndex(of: ".")!)...].count == 2
                && Int(amountStr[amountStr.index(after: amountStr.firstIndex(of: ".")!)...]) ?? 0 > 0
            if hasCents {
                let dot = amountStr.firstIndex(of: ".")!
                let centsPart = Int(amountStr[amountStr.index(after: dot)...]) ?? 0
                let whole = spellNumber(String(amountStr[..<dot])) ?? "zero"
                spoken = "\(whole) \(words) and \(spellNumber(String(centsPart)) ?? "") cents"
                result += spoken
            } else {
                result += "\(spoken) \(words)"
            }
            cursor = m.range.location + m.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    private static func dates(_ s: String) -> String {
        // M/D/YYYY or MM/DD/YYYY
        guard let re = RegexCache.regex(#"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"#) else { return s }
        let ns = s as NSString
        var result = ""
        var cursor = 0
        re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m, m.numberOfRanges > 3,
                  let mo = Int(ns.substring(with: m.range(at: 1))), (1...12).contains(mo),
                  let day = Int(ns.substring(with: m.range(at: 2))), (1...31).contains(day),
                  let year = Int(ns.substring(with: m.range(at: 3))) else { return }
            result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            let ordinalDay = ordinal(day)
            result += "\(monthNames[mo - 1]) \(ordinalDay), \(spellNumber(String(year)) ?? String(year))"
            cursor = m.range.location + m.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    private static func times(_ s: String) -> String {
        // H:MM am/pm
        guard let re = RegexCache.regex(#"\b(\d{1,2}):(\d{2})\s*([apAP])\.?[mM]\.?"#) else { return s }
        let ns = s as NSString
        var result = ""
        var cursor = 0
        re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m, m.numberOfRanges > 3,
                  let h = Int(ns.substring(with: m.range(at: 1))), (1...12).contains(h),
                  let mm = Int(ns.substring(with: m.range(at: 2))), (0...59).contains(mm) else { return }
            let meridiem = ns.substring(with: m.range(at: 3)).lowercased() == "a" ? "A M" : "P M"
            let hourWord = spellNumber(String(h)) ?? String(h)
            result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            if mm == 0 {
                result += "\(hourWord) \(meridiem)"
            } else {
                let minuteSpoken = mm < 10 ? "oh \(spellNumber(String(mm)) ?? String(mm))" : (spellNumber(String(mm)) ?? String(mm))
                result += "\(hourWord) \(minuteSpoken) \(meridiem)"
            }
            cursor = m.range.location + m.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    private static func phoneNumbers(_ s: String) -> String {
        guard let re = RegexCache.regex(#"\b\d{3}[- ]\d{3}[- ]\d{4}\b"#) else { return s }
        let ns = s as NSString
        var result = ""
        var cursor = 0
        re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m else { return }
            result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            let digits = ns.substring(with: m.range).compactMap { $0.isNumber ? String($0) : nil }.joined()
            result += digits.map(String.init).joined(separator: " ")
            cursor = m.range.location + m.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    /// Version strings x.y.z → "x point y point z".
    private static func versions(_ s: String) -> String {
        guard let re = RegexCache.regex(#"\bv(\d+(?:\.\d+)+)\b"#) else { return s }
        let ns = s as NSString
        var result = ""
        var cursor = 0
        re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m, m.numberOfRanges > 1 else { return }
            result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            let v = ns.substring(with: m.range(at: 1)).split(separator: ".").map(String.init)
            result += v.joined(separator: " point ")
            cursor = m.range.location + m.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    /// Ordinal suffixes & big-number digit reading.
    private static func numbers(_ s: String) -> String {
        var t = s

        // Ordinals: 1st, 22nd, 113th …
        if let re = RegexCache.regex(#"\b(\d{1,3})(st|nd|rd|th)\b"#) {
            let ns = t as NSString
            var result = ""
            var cursor = 0
            re.enumerateMatches(in: t, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
                guard let m = m, m.numberOfRanges > 2,
                      let n = Int(ns.substring(with: m.range(at: 1))) else { return }
                result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
                result += ordinal(n)
                cursor = m.range.location + m.range.length
            }
            result += ns.substring(from: cursor)
            t = result
        }

        // ≥5-digit numbers read digit-by-digit (except years starting with 20).
        if let re = RegexCache.regex(#"\d{5,}"#) {
            let ns = t as NSString
            var result = ""
            var cursor = 0
            re.enumerateMatches(in: t, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
                guard let m = m else { return }
                let num = ns.substring(with: m.range)
                result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
                if num.hasPrefix("20") && num.count == 5 {
                    result += spellNumber(num) ?? num
                } else {
                    result += num.map(String.init).joined(separator: " ")
                }
                cursor = m.range.location + m.range.length
            }
            result += ns.substring(from: cursor)
            t = result
        }

        // Apartment/suite/unit/room/flight/gate identifiers → letters spaced.
        if let re = RegexCache.regex(#"\b((?:apt|suite|unit|ste|room|rm|flight|flt|gate)\s+#?)([0-9]*-?[0-9]*[A-Za-z])(?:\s|$)"#, options: .caseInsensitive) {
            let ns = t as NSString
            var result = ""
            var cursor = 0
            re.enumerateMatches(in: t, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
                guard let m = m, m.numberOfRanges > 2 else { return }
                result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
                let label = ns.substring(with: m.range(at: 1))
                let code = ns.substring(with: m.range(at: 2))
                let expanded = code.map { c -> String in
                    c.isLetter ? String(c) : (c == "0" ? " oh" : String(c))
                }.joined(separator: " ")
                result += label + expanded + " "
                cursor = m.range.location + m.range.length
            }
            result += ns.substring(from: cursor)
            t = result
        }
        return t
    }

    private static let irregularOrdinals = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
                                            "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"]

    static func ordinal(_ n: Int) -> String {
        if n < irregularOrdinals.count { return irregularOrdinals[n] }
        let teen = (11...13).contains(n % 100)
        let suffix: String
        switch n % 10 {
        case 1: suffix = teen ? "th" : "st"
        case 2: suffix = teen ? "th" : "nd"
        case 3: suffix = teen ? "th" : "rd"
        default: suffix = "th"
        }
        return "\(n)\(suffix)"
    }

    /// Basic integer-to-words spelling for money/date contexts.
    static func spellNumber(_ raw: String) -> String? {
        let cleaned = raw.filter { $0.isNumber }
        guard let n = Int(cleaned) else { return nil }
        return numberToWords(n)
    }

    private static func numberToWords(_ n: Int) -> String {
        let ones = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
                    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
                    "eighteen","nineteen"]
        let tens = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

        func below1000(_ n: Int) -> String {
            if n < 20 { return ones[n] }
            if n < 100 { return tens[n / 10] + (n % 10 > 0 ? "-" + ones[n % 10] : "") }
            return ones[n / 100] + " hundred" + (n % 100 > 0 ? " " + below1000(n % 100) : "")
        }
        if n < 1000 { return below1000(n) }
        var parts: [String] = []
        var rest = n
        let scales: [(Int, String)] = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]
        for (value, name) in scales where rest >= value {
            parts.append("\(below1000(rest / value)) \(name)")
            rest %= value
        }
        if rest > 0 { parts.append(below1000(rest)) }
        return parts.joined(separator: " ")
    }
}
