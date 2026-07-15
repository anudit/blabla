//
//  Phonemizer.swift
//  tts-metal
//
//  Pure-Swift port of src/phonemizer.ts (dictionary + rules path).
//

import Foundation

enum Phonemizer {

    /// Preload the phonemizer resources (dictionary + en_rules) so that the
    /// first `textToInputIds` call doesn't pay the load cost. Safe to call
    /// multiple times; subsequent calls are no-ops.
    static func warmup() {
        loadDictionaryIfNeeded()
        loadRulesIfNeeded()
    }

    static func phonemesToInputIds(_ phonemes: String) -> [Int] {
        var ids: [Int] = [0]
        let scalarTokens = tokenizePhonemes(phonemes)
        let joined = scalarTokens.joined(separator: " ")

        for ch in joined {
            if let idx = TtsConfig.symbolToIndex[String(ch)] {
                ids.append(idx)
            }
        }
        ids.append(0)
        return ids
    }

    private static func tokenizePhonemes(_ s: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        for ch in s {
            if ch.isLetter || ch.isNumber || ch == "_" {
                current.append(ch)
            } else if !ch.isWhitespace {
                if !current.isEmpty { tokens.append(current); current = "" }
                tokens.append(String(ch))
            } else {
                if !current.isEmpty { tokens.append(current); current = "" }
            }
        }
        if !current.isEmpty { tokens.append(current) }
        return tokens
    }

    private static let punctChars = Set(";:,.!?¡¿—…“«»”„'")

    private static func isPunct(_ ch: Character) -> Bool { punctChars.contains(ch) }

    static let functionWordConnected: [String: String] = [
        "a": "ɐ", "an": "ɐn", "the": "ðə", "to": "tə", "of": "ʌv", "in": "ɪn",
        "on": "ɔn", "at": "æt", "by": "baɪ", "for": "fɔːɹ", "or": "ɔːɹ",
        "and": "ænd", "but": "bʌt", "that": "ðæt", "this": "ðɪs", "these": "ðiːz",
        "those": "ðoʊz", "i": "aɪ", "you": "juː", "he": "hiː", "she": "ʃiː",
        "we": "wiː", "it": "ɪt", "is": "ɪz", "was": "wʌz", "are": "ɑːɹ",
        "were": "wɜː", "be": "biː", "been": "bɪn", "have": "hæv", "has": "hæz",
        "had": "hæd", "do": "duː", "does": "dʌz", "did": "dɪd", "will": "wɪl",
        "would": "wʊd", "could": "kʊd", "should": "ʃʊd", "can": "kæn", "may": "meɪ",
        "might": "maɪt", "must": "mʌst", "shall": "ʃæl", "not": "nˌɑːt",
        "no": "nˈoʊ", "if": "ɪf", "how": "hˌaʊ", "with": "wɪð", "from": "fɹʌm",
        "your": "jʊɹ", "my": "maɪ", "his": "hɪz", "her": "hɜː", "its": "ɪts",
        "our": "aʊɚ", "their": "ðɛɹ", "some": "sˌʌm", "new": "nˈuː", "all": "ˈɔːl"
    ]

    static let letterNames: [Character: String] = [
        "a": "ˈeɪ", "b": "bˈiː", "c": "sˈiː", "d": "dˈiː", "e": "ˈiː",
        "f": "ˈɛf", "g": "dʒˈiː", "h": "ˈeɪtʃ", "i": "ˈaɪ", "j": "dʒˈeɪ",
        "k": "kˈeɪ", "l": "ˈɛl", "m": "ˈɛm", "n": "ˈɛn", "o": "ˈoʊ",
        "p": "pˈiː", "q": "kjˈuː", "r": "ˈɑːɹ", "s": "ˈɛs", "t": "tˈiː",
        "u": "jˈuː", "v": "vˈiː", "w": "dˈʌbəljˌuː", "x": "ˈɛks", "y": "wˈaɪ",
        "z": "zˈiː"
    ]

    static let letterPhonemes: [Character: String] = [
        "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f",
        "g": "ɡ", "h": "h", "i": "ɪ", "j": "dʒ", "k": "k", "l": "l",
        "m": "m", "n": "n", "o": "ɑː", "p": "p", "q": "k", "r": "ɹ",
        "s": "s", "t": "t", "u": "ʌ", "v": "v", "w": "w", "x": "ks",
        "y": "j", "z": "z"
    ]

    private static var dictionary: [String: String]?
    private static let dictQueue = DispatchQueue(label: "tts-metal.dict", qos: .utility)
    private static var dictLoaded = false

    static func loadDictionaryIfNeeded() {
        guard !dictLoaded else { return }
        dictQueue.sync {
            if dictLoaded { return }
            guard let url = Bundle.main.url(forResource: "espeak-en-dict", withExtension: "tsv") else {
                print("[Phonemizer] WARNING: espeak-en-dict.tsv not found in bundle")
                dictLoaded = true; dictionary = [:]; return
            }
            do {
                let text = try String(contentsOf: url, encoding: .utf8)
                var map: [String: String] = [:]
                for line in text.components(separatedBy: "\n") {
                    guard let tab = line.firstIndex(of: "\t") else { continue }
                    let word = String(line[..<tab])
                    let phonemes = String(line[line.index(after: tab)...])
                    map[word] = phonemes
                }
                dictionary = map
                print("[Phonemizer] Loaded dictionary: \(map.size) entries")
            } catch {
                print("[Phonemizer] Failed to load dictionary: \(error)")
                dictionary = [:]
            }
            dictLoaded = true
        }
    }

    static func loadRulesIfNeeded() {
        if EspeakRules.isRulesLoaded { return }
        guard let url = Bundle.main.url(forResource: "en_rules", withExtension: nil) else {
            print("[Phonemizer] WARNING: en_rules not found in bundle")
            return
        }
        if let text = try? String(contentsOf: url, encoding: .utf8) {
            EspeakRules.initRules(text)
            print("[Phonemizer] Loaded en_rules")
        }
    }

    static func textToPhonemes(_ text: String) -> String {
        loadDictionaryIfNeeded()
        loadRulesIfNeeded()
        let dict = dictionary ?? [:]

        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.isEmpty { return "" }

        let rawTokens = normalized.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
        var cleanWords: [String] = []
        var leadingPuncts: [String] = []
        var trailingPuncts: [String] = []

        for token in rawTokens {
            var word = token
            var trailing = ""
            while let last = word.last, isPunct(last) {
                trailing = String(last) + trailing
                word = String(word.dropLast())
            }
            var leading = ""
            while let first = word.first, isPunct(first) {
                leading += String(first)
                word = String(word.dropFirst())
            }
            cleanWords.append(word)
            trailingPuncts.append(trailing)
            leadingPuncts.append(leading)
        }

        let wordCount = cleanWords.filter { !$0.isEmpty }.count
        var parts: [String] = []
        for i in 0..<cleanWords.count {
            var part = ""
            part += leadingPuncts[i]
            if !cleanWords[i].isEmpty {
                let lower = cleanWords[i].lowercased()
                if wordCount > 1, let connected = functionWordConnected[lower] {
                    part += connected
                } else {
                    part += lookupWord(lower, dict: dict, original: cleanWords[i])
                }
            }
            part += trailingPuncts[i]
            if !part.isEmpty { parts.append(part) }
        }
        return parts.joined(separator: " ")
    }

    private static func lookupWord(_ word: String, dict: [String: String], original: String?) -> String {
        if let v = dict[word] { return v }

        if word.hasSuffix("'s") {
            let base = String(word.dropLast(2))
            if let v = dict[base] { return v + "z" }
        }
        if word.hasSuffix("s") && word.count > 2 {
            let base = String(word.dropLast())
            if let v = dict[base] {
                let lastChar = base.last ?? Character(" ")
                let voicedSet = Set("bdgjlmnrvwzaeiou")
                let suffix = voicedSet.contains(lastChar) ? "z" : "s"
                return v + suffix
            }
            if word.hasSuffix("es") {
                let base2 = String(word.dropLast())
                if let v = dict[base2] { return v + "z" }
            }
        }
        if word.hasSuffix("ed") && word.count > 3 {
            let base = String(word.dropLast(2))
            if let v = dict[base] { return v + "d" }
            if let v = dict[base + "e"] { return String(v.dropLast()) + "d" }
        }
        if word.hasSuffix("ing") && word.count > 4 {
            let base = String(word.dropLast(3))
            if let v = dict[base] { return v + "ɪŋ" }
            if let v = dict[base + "e"] { return String(v.dropLast()) + "ɪŋ" }
            if base.count > 1, base.last == base.dropLast().last {
                let dedup = String(base.dropLast())
                if let v = dict[dedup] { return v + "ɪŋ" }
            }
        }
        if word.hasSuffix("ly") && word.count > 3 {
            let base = String(word.dropLast(2))
            if let v = dict[base] { return v + "li" }
        }
        if word.hasSuffix("er") && word.count > 3 {
            let base = String(word.dropLast(2))
            if let v = dict[base] { return v + "ɚ" }
            if let v = dict[base + "e"] { return String(v.dropLast()) + "ɚ" }
        }
        if word.hasSuffix("est") && word.count > 4 {
            let base = String(word.dropLast(3))
            if let v = dict[base] { return v + "ɪst" }
        }
        if word.hasSuffix("ness") && word.count > 5 {
            let base = String(word.dropLast(4))
            if let v = dict[base] { return v + "nəs" }
        }
        if word.contains("-") {
            let pcs = word.split(separator: "-").map { String($0) }
            let phonemeParts = pcs.map { p -> String in
                if let c = functionWordConnected[p] { return c }
                return lookupWord(p, dict: dict, original: nil)
            }
            return phonemeParts.joined()
        }
        let orig = original ?? word
        if orig.count >= 2 && orig == orig.uppercased() && orig.allSatisfy({ $0.isLetter }) {
            let spelled = orig.flatMap { letterNames[$0] ?? String($0) }.joined()
            if spelled != word { return spelled }
        }
        if let compound = trySplitCompound(word, dict: dict) { return compound }
        if EspeakRules.isRulesLoaded {
            return EspeakRules.wordToIPA(word)
        }
        return word.flatMap { letterPhonemes[$0] ?? String($0) }.joined()
    }

    private static func trySplitCompound(_ word: String, dict: [String: String]) -> String? {
        let chars = Array(word)
        for i in stride(from: chars.count - 2, through: 2, by: -1) {
            let prefix = String(chars[..<i])
            let suffix = String(chars[i...])
            if let p = dict[prefix], let s = dict[suffix] {
                return p + s
            }
            if suffix.hasSuffix("s") {
                let trimmed = String(suffix.dropLast())
                if let s = dict[trimmed], let p = dict[prefix] {
                    return p + s + "z"
                }
            }
        }
        return nil
    }

    static func textToInputIds(_ text: String) -> [Int] {
        phonemesToInputIds(textToPhonemes(text))
    }
}

private extension Dictionary where Key == String, Value == String {
    var size: Int { count }
}