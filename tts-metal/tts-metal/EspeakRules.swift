//
//  EspeakRules.swift
//  tts-metal
//
//  Pure-Swift port of src/espeak-rules.ts — letter-to-phoneme rules for unknown words.
//

import Foundation

enum EspeakRules {

    static let phonemeToIpa: [String: String] = [
        "a": "æ", "a#": "ɐ", "A:": "ɑː", "A@": "ɑːɹ", "aa": "ɑː", "a:": "ɑː",
        "ai": "aɪ", "aI": "aɪ", "aU": "aʊ", "au": "aʊ",
        "e": "ɛ", "E": "ɛ", "E2": "ɛ", "e#": "ɛ", "eI": "eɪ", "e@": "ɛɹ",
        "i": "ɪ", "i:": "iː", "I": "ɪ", "I2": "ɪ", "I#": "ɪ",
        "0": "ɒ", "0#": "ɒ",
        "oU": "oʊ", "O:": "ɔː", "O@": "ɔːɹ", "O2": "ɒ", "OI": "ɔɪ",
        "u:": "uː", "U": "ʊ", "V": "ʌ", "VR": "ɜːɹ", "3:": "ɜː", "3": "ɜ",
        "@": "ə", "@2": "ə", "@5": "ə", "@L": "əl", "@-": "ə",
        "b": "b", "d": "d", "f": "f", "g": "ɡ", "h": "h", "j": "j",
        "k": "k", "l": "l", "L": "l", "m": "m", "n": "n", "N": "ŋ",
        "p": "p", "r": "ɹ", "R": "ɹ", "s": "s", "S": "ʃ",
        "t": "t", "t2": "t", "T": "θ", "D": "ð", "v": "v", "w": "w",
        "x": "x", "z": "z", "Z": "ʒ", "dZ": "dʒ", "tS": "tʃ", "?": "ʔ",
        "'": "ˈ", ",": "ˌ", "%": "", ":": "ː", "=": "", "#": "", "-": "", "|": "",
        "_": " ",
        "IR": "ɪɹ", "th": "tθ", "n-": "n", "z#": "z", "z/2": "z"
    ]

    static let phonemeKeysSorted: [String] = {
        phonemeToIpa.keys.sorted { $0.count > $1.count }
    }()

    static func espeakToIPA(_ espeak: String) -> String {
        var result = ""
        let chars = Array(espeak)
        var i = 0
        while i < chars.count {
            var matched = false
            for key in phonemeKeysSorted {
                let k = Array(key)
                if i + k.count <= chars.count, String(chars[i..<i+k.count]) == key {
                    result += phonemeToIpa[key] ?? ""
                    i += k.count
                    matched = true
                    break
                }
            }
            if !matched { i += 1 }
        }
        return result
    }

    // ── Rule parsing ──

    struct Rule {
        let pattern: String
        let pre: String
        let post: String
        let phonemes: String
        let conditionNum: Int
        let conditionNeg: Bool
    }

    private static var letterGroups: [String: Set<String>] = [:]
    private static var ruleGroups: [String: [Rule]] = [:]

    private static let vowels = Set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüý")
    private static let consonants = Set("bcdfghjklmnpqrstvwxz")
    private static let voicedConsonants = Set("bdgjlmnrvwz")
    private static let letterSet = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàáâãäåæèéêëìíîïòóôõöùúûüýÿ")

    private static func isVowel(_ ch: Character) -> Bool { vowels.contains(ch.lowercased().first ?? ch) }
    private static func isConsonant(_ ch: Character) -> Bool { consonants.contains(ch.lowercased().first ?? ch) }
    private static func isLetter(_ ch: Character) -> Bool { letterSet.contains(ch) }
    private static func isDigit(_ ch: Character) -> Bool { ch.isASCII && ch.isNumber }

    private static func matchPost(_ cond: String, _ text: [Character], _ pos: Int) -> Int {
        var ci = 0
        var ti = pos
        while ci < cond.count {
            let cc = cond[cond.index(cond.startIndex, offsetBy: ci)]
            switch cc {
            case "?", "@", "$": return ti - pos
            case "+":
                if ti < text.count && isLetter(text[ti]) {
                    ci += 1; ti += 1; continue
                }
                return -1
            case "_":
                if ti >= text.count || !isLetter(text[ti]) {
                    ci += 1; continue
                }
                return -1
            case "#":
                if ti >= text.count || !isLetter(text[ti]) || isVowel(text[ti]) {
                    ci += 1; continue
                }
                return -1
            case "A":
                if ti < text.count && isVowel(text[ti]) { ci += 1; ti += 1; continue }
                return -1
            case "B":
                if ti < text.count && voicedConsonants.contains(text[ti].lowercased().first ?? text[ti]) { ci += 1; ti += 1; continue }
                return -1
            case "C":
                if ti < text.count && isConsonant(text[ti]) { ci += 1; ti += 1; continue }
                return -1
            case "D":
                if ti < text.count && isDigit(text[ti]) { ci += 1; ti += 1; continue }
                return -1
            case "K":
                if ti >= text.count || !isVowel(text[ti]) {
                    ci += 1
                    if ti < text.count && isLetter(text[ti]) { ti += 1 }
                    continue
                }
                return -1
            case "N":
                if ti < text.count, let lc = text[ti].lowercased().first, "mn".contains(lc) { ci += 1; ti += 1; continue }
                if ti + 1 < text.count, text[ti].lowercased().first == "n", text[ti + 1].lowercased().first == "g" {
                    ci += 1; ti += 2; continue
                }
                return -1
            case "X":
                if ti < text.count && isLetter(text[ti]) { ci += 1; ti += 1; continue }
                return -1
            case "Y":
                if ti < text.count && isConsonant(text[ti]) { ci += 1; ti += 1; continue }
                if ti >= text.count { ci += 1; continue }
                return -1
            case "L":
                let nextIdx = cond.index(cond.startIndex, offsetBy: ci + 1)
                let next2Idx = cond.index(cond.startIndex, offsetBy: ci + 2)
                if ci + 2 < cond.count, isDigit(cond[nextIdx]), isDigit(cond[next2Idx]) {
                    let groupId = "\(cond[nextIdx])\(cond[next2Idx])"
                    let group = letterGroups[groupId] ?? []
                    ci += 3
                    var found = false
                    for entry in group {
                        let arr = Array(entry)
                        if ti + arr.count <= text.count, String(text[ti..<ti+arr.count]).lowercased() == entry {
                            ti += arr.count; found = true; break
                        }
                    }
                    if !found { return -1 }
                    continue
                }
                fallthrough
            default:
                if ti < text.count && text[ti].lowercased().first == cc.lowercased().first {
                    ci += 1; ti += 1; continue
                }
                return -1
            }
        }
        return ti - pos
    }

    private static func matchPre(_ cond: String, _ text: [Character], _ pos: Int) -> Int {
        let condChars = Array(cond)
        var ci = condChars.count - 1
        var ti = pos - 1
        while ci >= 0 {
            let cc = condChars[ci]
            switch cc {
            case "?", "@", "$": return pos - ti - 1
            case "_":
                if ti < 0 || !isLetter(text[ti]) { ci -= 1; continue }
                return -1
            case "#":
                if ti < 0 || !isLetter(text[ti]) || isVowel(text[ti]) { ci -= 1; continue }
                return -1
            case "&":
                if ti >= 0 && isLetter(text[ti]) { ci -= 1; ti -= 1; continue }
                return -1
            case "A":
                if ti >= 0 && isVowel(text[ti]) { ci -= 1; ti -= 1; continue }
                return -1
            case "B":
                if ti >= 0, let lc = text[ti].lowercased().first, voicedConsonants.contains(lc) { ci -= 1; ti -= 1; continue }
                return -1
            case "C":
                if ti >= 0 && isConsonant(text[ti]) { ci -= 1; ti -= 1; continue }
                return -1
            case "D":
                if ti >= 0 && isDigit(text[ti]) { ci -= 1; ti -= 1; continue }
                return -1
            case "K":
                if ti < 0 || !isVowel(text[ti]) {
                    ci -= 1
                    if ti >= 0 && isLetter(text[ti]) { ti -= 1 }
                    continue
                }
                return -1
            case "X":
                if ti >= 0 && isLetter(text[ti]) { ci -= 1; ti -= 1; continue }
                return -1
            default:
                if cc.isASCII && cc.isNumber, ci >= 2, condChars[ci - 2] == "L",
                   condChars[ci - 1].isASCII && condChars[ci - 1].isNumber {
                    let groupId = "\(condChars[ci - 1])\(cc)"
                    let group = letterGroups[groupId] ?? []
                    ci -= 3
                    var found = false
                    for entry in group {
                        let arr = Array(entry)
                        let start = ti - arr.count + 1
                        if start >= 0, String(text[start...ti]).lowercased() == entry {
                            ti = start - 1; found = true; break
                        }
                    }
                    if !found { return -1 }
                    continue
                }
                if ti >= 0 && text[ti].lowercased().first == cc.lowercased().first { ci -= 1; ti -= 1; continue }
                return -1
            }
        }
        return pos - ti - 1
    }

    static func parseRules(_ rulesText: String) {
        ruleGroups.removeAll()
        letterGroups.removeAll()

        var currentGroup = ""
        var inReplace = false
        let lines = rulesText.components(separatedBy: "\n")

        for lineRaw in lines {
            var line = lineRaw
            if let range = line.range(of: "//") { line = String(line[..<range.lowerBound]) }
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }

            let parts = trimmed.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            if let first = parts.first, first.hasPrefix(".L"), first.count >= 4 {
                let groupId = String(first.dropFirst(2))
                let entries = parts.dropFirst().map { $0.lowercased() }
                letterGroups[groupId] = Set(entries)
                continue
            }

            if trimmed == ".replace" { inReplace = true; continue }
            if trimmed.hasPrefix(".group") {
                inReplace = false
                let gName = trimmed.dropFirst(".group".count).trimmingCharacters(in: .whitespaces)
                currentGroup = String(gName)
                if ruleGroups[currentGroup] == nil { ruleGroups[currentGroup] = [] }
                continue
            }
            if inReplace { continue }

            if currentGroup.isEmpty && ruleGroups[""] == nil { ruleGroups[""] = [] }

            if let rule = parseRuleLine(String(trimmed)) {
                ruleGroups[currentGroup, default: []].append(rule)
            }
        }
    }

    private static func parseRuleLine(_ line: String) -> Rule? {
        if line.isEmpty { return nil }
        var conditionNum = 0
        var conditionNeg = false
        let chars = Array(line)
        var idx = 0
        while idx < chars.count && (chars[idx] == " " || chars[idx] == "\t") { idx += 1 }

        if idx < chars.count && (chars[idx] == "?" || chars[idx] == "!") {
            if idx + 1 < chars.count, chars[idx] == "!", chars[idx + 1] == "?" {
                conditionNeg = true; idx += 2
            } else if idx + 1 < chars.count, chars[idx] == "?", chars[idx + 1] == "!" {
                conditionNeg = true; idx += 2
            } else if chars[idx] == "?" {
                idx += 1
            }
            var numStr = ""
            while idx < chars.count && chars[idx].isASCII && chars[idx].isNumber {
                numStr.append(chars[idx]); idx += 1
            }
            conditionNum = Int(numStr) ?? 0
        }

        while idx < chars.count && (chars[idx] == " " || chars[idx] == "\t") { idx += 1 }
        let rest = String(chars[idx...])
        if rest.isEmpty { return nil }

        var pre = ""
        var pattern = ""
        var post = ""
        var phonemes = ""

        let parenClose = rest.firstIndex(of: ")")
        let parenOpen = rest.firstIndex(of: "(")
        var patternStart = rest.startIndex

        if let pc = parenClose {
            pre = String(rest[..<pc]).trimmingCharacters(in: .whitespaces)
            patternStart = rest.index(after: pc)
        }
        while patternStart < rest.endIndex, rest[patternStart] == " " {
            patternStart = rest.index(after: patternStart)
        }

        if let po = parenOpen, parenClose.map({ $0 < po }) ?? true {
            pattern = String(rest[patternStart..<po]).trimmingCharacters(in: .whitespaces)
            var postEnd = rest.index(after: po)
            while postEnd < rest.endIndex && rest[postEnd] != " " && rest[postEnd] != "\t" {
                postEnd = rest.index(after: postEnd)
            }
            post = String(rest[rest.index(after: po)..<postEnd])
            phonemes = String(rest[postEnd...]).trimmingCharacters(in: .whitespaces)
        } else {
            var patternEnd = patternStart
            while patternEnd < rest.endIndex && rest[patternEnd] != " " && rest[patternEnd] != "\t" {
                patternEnd = rest.index(after: patternEnd)
            }
            pattern = String(rest[patternStart..<patternEnd]).trimmingCharacters(in: .whitespaces)
            phonemes = String(rest[patternEnd...]).trimmingCharacters(in: .whitespaces)
        }

        if pattern.isEmpty { return nil }
        return Rule(pattern: pattern, pre: pre, post: post, phonemes: phonemes,
                    conditionNum: conditionNum, conditionNeg: conditionNeg)
    }

    private static func applyRulesToWord(_ word: String) -> String {
        let lower = word.lowercased()
        let chars = Array(lower)
        var result = ""
        var i = 0
        while i < chars.count {
            var bestRule: Rule? = nil
            var bestPatternLen = 0
            var bestScore = -1

            var groupKeys: [String] = []
            let maxLen = min(4, chars.count - i)
            for len in stride(from: maxLen, through: 1, by: -1) {
                let key = String(chars[i..<i+len])
                if ruleGroups[key] != nil && !groupKeys.contains(key) { groupKeys.append(key) }
            }
            let singleKey = String(chars[i])
            if !groupKeys.contains(singleKey) && ruleGroups[singleKey] != nil { groupKeys.append(singleKey) }

            for groupKey in groupKeys {
                guard let rules = ruleGroups[groupKey] else { continue }
                for rule in rules {
                    if rule.conditionNum > 0 { continue }
                    let patternLower = rule.pattern.lowercased()
                    let pChars = Array(patternLower)
                    if i + pChars.count > chars.count { continue }
                    if String(chars[i..<i+pChars.count]) != patternLower { continue }

                    let patternEnd = i + pChars.count
                    if !rule.pre.isEmpty {
                        if matchPre(rule.pre, chars, i) < 0 { continue }
                    }
                    if !rule.post.isEmpty {
                        if matchPost(rule.post, chars, patternEnd) < 0 { continue }
                    }
                    let score = pChars.count * 100 + rule.pre.count * 10 + rule.post.count
                    if score > bestScore {
                        bestScore = score
                        bestRule = rule
                        bestPatternLen = pChars.count
                    }
                }
            }

            if let drules = ruleGroups[""] {
                for rule in drules {
                    if rule.conditionNum > 0 { continue }
                    let patternLower = rule.pattern.lowercased()
                    let pChars = Array(patternLower)
                    if i + pChars.count > chars.count { continue }
                    if String(chars[i..<i+pChars.count]) != patternLower { continue }
                    let patternEnd = i + pChars.count
                    if !rule.pre.isEmpty, matchPre(rule.pre, chars, i) < 0 { continue }
                    if !rule.post.isEmpty, matchPost(rule.post, chars, patternEnd) < 0 { continue }
                    let score = pChars.count * 100 + rule.pre.count * 10 + rule.post.count
                    if score > bestScore {
                        bestScore = score
                        bestRule = rule
                        bestPatternLen = pChars.count
                    }
                }
            }

            if let rule = bestRule {
                result += rule.phonemes
                i += bestPatternLen
            } else {
                i += 1
            }
        }
        return result
    }

    static func wordToIPA(_ word: String) -> String {
        espeakToIPA(applyRulesToWord(word))
    }

    private static var _rulesLoaded = false
    static var isRulesLoaded: Bool { _rulesLoaded }

    static func initRules(_ text: String) {
        parseRules(text)
        _rulesLoaded = true
    }
}