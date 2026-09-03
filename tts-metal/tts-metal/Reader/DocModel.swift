//
//  DocModel.swift
//  tts-metal
//
//  Core document model for the reader: blocks, sentences, outline entries and
//  char-proportional word timing used for karaoke highlighting.
//  Ported from blabla's utils.tsx / ContentView.tsx data structures.
//

import Foundation

// MARK: - Blocks

enum BlockContent: Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case code(String)
    case quote(String)
    case listItem(String)
    case table(rows: [[String]])
    /// `src` is the key into `ReaderDocument.resources` for embedded images
    /// (EPUB/MOBI), or an absolute http(s) URL for web/markdown documents.
    /// Empty when the source had no usable image reference.
    case image(alt: String, src: String)
    /// Figure caption / image credit — displayed, but never spoken.
    case caption(String)
    case rule
    case frontmatter(title: String?, description: String?, image: String?)
}

struct DocBlock {
    var content: BlockContent

    /// Speakable text for this block (empty for non-speakable blocks).
    var speechText: String {
        switch content {
        case .heading(_, let t):        return t
        case .paragraph(let t):         return t
        case .code:                     return ""
        case .quote(let t):             return t
        case .listItem(let t):          return t
        case .table(let rows):          return rows.map { $0.joined(separator: ", ") }.joined(separator: ". ")
        case .image:                    return ""
        case .caption:                  return ""
        case .rule:                     return ""
        case .frontmatter:              return ""
        }
    }

    /// What the reader puts on screen for this block: the publisher's own
    /// wording, untouched by TTS normalization.
    ///
    /// `speechText` is deliberately *not* this string. It feeds the
    /// synthesizer, so it has curly quotes flattened to ASCII, em dashes
    /// turned into ", ", ellipses into "...", numerals spelled out — all
    /// correct for speech and all wrong on a page. Rendering from
    /// `speechText` (which the reader used to do, by displaying the
    /// normalized sentence strings) is what made a typeset book read as
    /// mangled plain text. Blocks that are shown but never spoken (captions,
    /// code) have text here and none in `speechText`.
    var displayText: String {
        switch content {
        case .heading(_, let t):        return t
        case .paragraph(let t):         return t
        case .code(let t):              return t
        case .quote(let t):             return t
        case .listItem(let t):          return t
        case .caption(let t):           return t
        case .table(let rows):          return rows.map { $0.joined(separator: "\t") }.joined(separator: "\n")
        case .image:                    return ""
        case .rule:                     return ""
        case .frontmatter:              return ""
        }
    }
}

struct OutlineEntry: Identifiable {
    let id = UUID()
    let level: Int
    let title: String
    let blockIndex: Int
}

/// One speakable sentence with a stable global id and its owning block.
/// `text` starts as the "fast" (structure-determining) normalization and is
/// upgraded in place to the fully-normalized form once `expandNumericForms`
/// has run for it — see `ReaderDocument.withFullyNormalizedSentences()`.
struct RSentence: Identifiable, Equatable {
    let id: Int
    var text: String
    let blockIndex: Int
    /// Where this sentence sits inside its block's `displayText`, as UTF-16
    /// offsets. The reader lays out the block's original text once and uses
    /// these ranges to highlight, hit-test and scroll to a sentence, so what
    /// is spoken and what is shown stay in step even though the two strings
    /// differ (see `DocBlock.displayText`).
    let range: NSRange
    /// The publisher's own text for this sentence — what is drawn on screen.
    let displayText: String
}

enum FileTypeKind: String {
    case pdf, epub, mobi, docx, url, text, ocr
    var label: String {
        switch self {
        case .pdf: return "PDF"
        case .epub: return "EPUB"
        case .mobi: return "MOBI"
        case .docx: return "DOCX"
        case .url:  return "URL"
        case .text: return "TXT"
        case .ocr:  return "OCR"
        }
    }
}

struct ReaderDocument {
    var title: String
    var fileType: FileTypeKind
    var sourceID: String          // identity used for bookmark resume ("name:size" or URL)
    var fileName: String
    var previewURL: String?       // set for .url docs so history entries are re-openable
    var sourceFilePath: String?   // absolute path for file-based docs (for history auto-reopen)
    var frontmatter: BlockContent?
    var blocks: [DocBlock]
    var outline: [OutlineEntry]
    /// Decoded bytes for embedded images, keyed by the `src` of an
    /// `.image` block (the archive-relative path for EPUB/MOBI).
    var resources: [String: Data] = [:]
    /// Page number each sentence starts on (PDF/OCR only; otherwise empty).
    var sentencePages: [Int] = []
    var pageCount: Int = 0
    /// Speakable sentences — boundaries + fast-normalized text computed once
    /// at init so document open (and bookmark resume, which needs a stable
    /// global sentence index) is instant; text is upgraded to fully
    /// normalized form in the background afterward.
    var sentences: [RSentence]
    /// Sentences grouped by block index for rendering/tap-to-jump.
    var sentencesByBlock: [Int: [RSentence]]

    init(title: String, fileType: FileTypeKind, sourceID: String, fileName: String,
         previewURL: String? = nil, sourceFilePath: String? = nil,
         frontmatter: BlockContent? = nil,
         blocks: [DocBlock], outline: [OutlineEntry],
         resources: [String: Data] = [:],
         sentencePages: [Int] = [], pageCount: Int = 0) {
        self.title = title
        self.fileType = fileType
        self.sourceID = sourceID
        self.fileName = fileName
        self.previewURL = previewURL
        self.sourceFilePath = sourceFilePath
        self.frontmatter = frontmatter
        self.blocks = blocks
        self.outline = outline
        self.resources = resources
        self.sentencePages = sentencePages
        self.pageCount = pageCount

        // Per-block fast-normalize + sentence-split is pure and independent
        // of every other block (only the resulting global IDs need to be
        // assigned in order), so the CPU-bound regex work — the actual cost
        // of opening a book — runs across every core instead of serially.
        // Sentences are split over the block's *display* text and keep the
        // range they occupied there, then each one is normalized for speech
        // individually. Splitting the normalized string instead (as this used
        // to) threw away every offset into the original, which is why the
        // reader could only show the normalized text back to the user.
        var perBlock = [[(range: NSRange, spoken: String, display: String)]](repeating: [], count: blocks.count)
        perBlock.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: blocks.count) { bi in
                guard !blocks[bi].speechText.isEmpty else { return }
                let original = blocks[bi].displayText
                let ns = original as NSString
                buf[bi] = SentenceSplitter.extractRanges(original).compactMap { r in
                    let display = ns.substring(with: r)
                    // Drops junk like bare page numbers ("7", "28") that land
                    // as their own block/sentence in scanned or Calibre-style
                    // EPUB output — too short to be a real sentence.
                    guard display.count > 3 else { return nil }
                    // Fast pass only: determines the spoken form without the
                    // (much costlier) money/date/time/phone/version/ordinal
                    // expansion — see TTSTextNormalizer.cleanForTtsFast.
                    let t = TTSTextNormalizer.cleanForTtsFast(display)
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !t.isEmpty else { return nil }
                    return (r, t.hasPunctuationTerminal ? t : t + ".", display)
                }
            }
        }

        var out: [RSentence] = []
        var byBlock: [Int: [RSentence]] = [:]
        out.reserveCapacity(blocks.count * 2)
        var nextID = 0
        for (bi, items) in perBlock.enumerated() {
            for item in items {
                let s = RSentence(id: nextID, text: item.spoken, blockIndex: bi,
                                  range: item.range, displayText: item.display)
                out.append(s)
                byBlock[bi, default: []].append(s)
                nextID += 1
            }
        }
        self.sentences = out
        self.sentencesByBlock = byBlock
    }

    /// Upgrades every sentence's text from the fast pass to the fully
    /// normalized form (money/date/time/phone/version/ordinal expansion).
    /// Sentence identity, count and block mapping are unchanged — only the
    /// displayed text improves — so this is safe to apply in the background
    /// after the document is already on screen. Runs across all CPU cores
    /// since each sentence's expansion is independent.
    func withFullyNormalizedSentences() -> ReaderDocument {
        var copy = self
        var upgraded = sentences
        let count = upgraded.count
        guard count > 0 else { return copy }
        upgraded.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: count) { i in
                buf[i].text = TTSTextNormalizer.expandNumericForms(buf[i].text)
            }
        }
        var byBlock: [Int: [RSentence]] = [:]
        byBlock.reserveCapacity(sentencesByBlock.count)
        for s in upgraded {
            byBlock[s.blockIndex, default: []].append(s)
        }
        copy.sentences = upgraded
        copy.sentencesByBlock = byBlock
        return copy
    }
}

extension String {
    var hasPunctuationTerminal: Bool {
        guard let last = utf8.last.map({ Character(UnicodeScalar($0)) }) else { return false }
        return ".!?…。！？；;:".contains(last)
    }
}

// MARK: - Sentence splitter (port of utils.tsx extractSentences)

enum SentenceSplitter {

    /// Protects links/images/inline-code/decimals from being split by the
    /// sentence regex via placeholder swapping.
    private static func protect(_ text: String) -> (String, [String]) {
        var stash: [String] = []
        var out = text

        func stashReplace(_ pattern: String) {
            guard let re = RegexCache.regex(pattern) else { return }
            let ns = out as NSString
            var result = ""
            var cursor = 0
            re.enumerateMatches(in: out, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
                guard let m = m else { return }
                result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
                stash.append(ns.substring(with: m.range))
                result += "\u{0}\(stash.count - 1)\u{0}"
                cursor = m.range.location + m.range.length
            }
            result += ns.substring(from: cursor)
            out = result
        }

        stashReplace(#"\[[^\]]*\]\([^)]*\)"#)    // markdown links
        stashReplace(#"!\[[^\]]*\]\([^)]*\)"#)    // markdown images
        stashReplace(#"`[^`]+`"#)                 // inline code
        stashReplace(#"<[^>]+>"#)                  // stray tags
        stashReplace(#"\d+(?:\.\d+)+"#)            // decimals / version numbers
        // Initials & dotted acronyms (U.S.A., J.K.) — require at least two
        // dotted letters so this doesn't swallow every sentence-final word
        // ("cat." would otherwise match as a 1-rep "acronym" via its final
        // "t."), which hid real sentence terminators and made whole blocks
        // collapse into one giant "sentence" (and made the boundary regex
        // scan huge no-match stretches, which is where load time went).
        stashReplace(#"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?"#)
        return (out, stash)
    }

    private static func restore(_ text: String, _ stash: [String]) -> String {
        guard !stash.isEmpty else { return text }
        guard let re = RegexCache.regex("\u{0}(\\d+)\u{0}") else { return text }
        let ns = text as NSString
        var result = ""
        var cursor = 0
        re.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m, m.numberOfRanges > 1 else { return }
            result += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
            let idx = Int(ns.substring(with: m.range(at: 1))) ?? 0
            result += idx < stash.count ? stash[idx] : ""
            cursor = m.range.location + m.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    /// Split text into sentences at `.!?…` boundaries.
    static func extract(_ raw: String) -> [String] {
        let text = RegexCache.replace(raw, pattern: "\\s+\\n", with: "\n")
        guard !text.isEmpty else { return [] }

        let (protected_, stash) = protect(text)
        var sentences: [String] = []
        guard let re = RegexCache.regex(#"[^.!?…]+[.!?…]+"#) else { return [text] }
        let ns = protected_ as NSString
        var consumed = 0
        re.enumerateMatches(in: protected_, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m else { return }
            if m.range.location > consumed {
                push(String(ns.substring(with: NSRange(location: consumed, length: m.range.location - consumed))), into: &sentences)
            }
            push(ns.substring(with: m.range), into: &sentences)
            consumed = m.range.location + m.range.length
        }
        if consumed < ns.length {
            push(ns.substring(from: consumed), into: &sentences)
        }
        return sentences.map { restore($0, stash).trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
    }

    /// Patterns whose sentence-terminal characters are false boundaries.
    /// Same intent as `protect`, but these are used to *blank* the offending
    /// characters in a same-length copy of the text rather than to swap them
    /// for placeholders, so the resulting ranges still address the original.
    private static let maskPatterns = [
        #"!\[[^\]]*\]\([^)]*\)"#,               // markdown images
        #"\[[^\]]*\]\([^)]*\)"#,                // markdown links
        #"`[^`]+`"#,                              // inline code
        #"<[^>]+>"#,                              // stray tags
        #"\d+(?:[.,]\d+)+"#,                      // decimals / version numbers
        // Initials & dotted acronyms — at least two dotted letters, so an
        // ordinary sentence-final word ("cat.") isn't read as an acronym.
        #"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?"#,
    ]

    /// Abbreviations whose period is not a full stop ("Mr.", "approx.").
    /// `cleanForTtsFast` expands these before splitting the spoken copy; the
    /// displayed copy keeps them, so the boundary scan has to know about them
    /// too — see `maskAbbreviationPeriods`.
    private static let abbreviationWords: Set<String> = [
        "mr", "mrs", "ms", "dr", "prof", "rev", "hon", "st", "vs", "etc",
        "inc", "ltd", "co", "corp", "jr", "sr", "no", "nos", "fig", "figs",
        "vol", "vols", "approx", "dept", "est", "eq", "ch", "chap", "ed",
        "eds", "al", "ca", "cf", "pp", "p", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sept", "sep", "oct", "nov", "dec",
    ]

    /// Same match set as the former `\b(?:mr|mrs|...)\.` alternation
    /// (case-insensitive), but as a direct scan instead of a ~50-branch
    /// regex — measured 3x slower than this loop on real book text, and was
    /// by far the costliest single pattern here (it alone was ~90% of this
    /// function's time on a full-length novel). A regex alternation forces
    /// the engine to retry every branch at every word-boundary position in
    /// the text; scanning for maximal `\w`-runs followed by "." and a set
    /// lookup finds the same matches in one pass with no backtracking.
    ///
    /// `\w`-runs (not letter-runs) is what reproduces `\b`'s actual
    /// semantics: `\b` requires a transition between a word character
    /// (letter, digit or underscore) and a non-word character, so scanning by
    /// letters alone would treat "2mr." as having the boundary the original
    /// regex denies it (no boundary between a digit and a letter). A run that
    /// includes a digit or underscore never matches the all-letter
    /// `abbreviationWords` set regardless, so nothing extra is needed to
    /// reject it — the set lookup already does that.
    private static func maskAbbreviationPeriods(in units: inout [UInt16], terminals: Set<UInt16>) {
        var runStart: Int? = nil
        for i in 0..<units.count {
            let u = units[i]
            let isWordChar = (u >= 0x41 && u <= 0x5A) || (u >= 0x61 && u <= 0x7A)   // A-Z a-z
                || (u >= 0x30 && u <= 0x39) || u == 0x5F                             // 0-9 _
                || (u > 0x7F && Character(Unicode.Scalar(u) ?? Unicode.Scalar(0)).isLetter)
            if isWordChar {
                if runStart == nil { runStart = i }
            } else {
                if let start = runStart, u == 0x2E /* . */ {
                    let word = String(decoding: units[start..<i], as: UTF16.self).lowercased()
                    if abbreviationWords.contains(word), terminals.contains(units[i]) {
                        units[i] = 0x20
                    }
                }
                runStart = nil
            }
        }
    }

    /// Sentence ranges *within* `text`, as UTF-16 offsets into that exact
    /// string.
    ///
    /// `extract` rewrites the text as it goes (placeholder swapping, then
    /// trimming), so its output can no longer be located in the input. This
    /// runs the same boundary scan over a mask that is character-for-character
    /// the same length as the original — every non-terminal `.`/`!`/`?`/`…`
    /// replaced by a space — so each match maps straight back onto the
    /// publisher's own characters. That is what lets the reader lay out real
    /// typography and still know where each spoken sentence begins and ends.
    static func extractRanges(_ text: String) -> [NSRange] {
        let ns = text as NSString
        guard ns.length > 0 else { return [] }

        var units = Array(text.utf16)
        let terminals: Set<UInt16> = [0x2E, 0x21, 0x3F, 0x2026]   // . ! ? …
        let full = NSRange(location: 0, length: ns.length)
        for pattern in maskPatterns {
            guard let re = RegexCache.regex(pattern, options: .caseInsensitive) else { continue }
            re.enumerateMatches(in: text, range: full) { m, _, _ in
                guard let m = m else { return }
                for i in m.range.location..<m.range.upperBound where terminals.contains(units[i]) {
                    units[i] = 0x20
                }
            }
        }
        maskAbbreviationPeriods(in: &units, terminals: terminals)
        let masked = String(decoding: units, as: UTF16.self)
        // Only BMP scalars were substituted, so this must hold; bail to a
        // single whole-text sentence rather than emit ranges that don't line up.
        guard (masked as NSString).length == ns.length,
              let re = RegexCache.regex(#"[^.!?…]+[.!?…]+"#) else {
            return trim(full, in: ns).map { [$0] } ?? []
        }

        var out: [NSRange] = []
        var consumed = 0
        func take(_ r: NSRange) {
            if let t = trim(r, in: ns) { out.append(t) }
        }
        re.enumerateMatches(in: masked, range: full) { m, _, _ in
            guard let m = m else { return }
            if m.range.location > consumed {
                take(NSRange(location: consumed, length: m.range.location - consumed))
            }
            take(m.range)
            consumed = m.range.upperBound
        }
        if consumed < ns.length {
            take(NSRange(location: consumed, length: ns.length - consumed))
        }
        return out
    }

    /// Shrinks a range past leading/trailing whitespace, returning nil if
    /// nothing but whitespace is left.
    private static func trim(_ r: NSRange, in ns: NSString) -> NSRange? {
        func isSpace(_ i: Int) -> Bool {
            guard let u = Unicode.Scalar(ns.character(at: i)) else { return false }
            return CharacterSet.whitespacesAndNewlines.contains(u)
        }
        var lo = r.location
        var hi = r.upperBound
        while lo < hi, isSpace(lo) { lo += 1 }
        while hi > lo, isSpace(hi - 1) { hi -= 1 }
        return hi > lo ? NSRange(location: lo, length: hi - lo) : nil
    }

    private static func push(_ s: String, into arr: inout [String]) {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { arr.append(t) }
    }

    /// blabla's splitText: cap a chunk at `limit` chars, preferring `.!?` then `,;:` boundaries.
    static func splitChunk(_ text: String, limit: Int = 280) -> [String] {
        if text.count <= limit { return [text] }
        var out: [String] = []
        var remaining = Substring(text)
        while remaining.count > limit {
            let head = remaining.prefix(limit)
            var cut = head.lastIndex(where: { ".!?".contains($0) }).map { remaining.index(after: $0) }
                ?? head.lastIndex(where: { ",;:".contains($0) }).map { remaining.index(after: $0) }
                ?? head.lastIndex(of: " ").map { remaining.index(after: $0) }
                ?? remaining.index(remaining.startIndex, offsetBy: limit)
            cut = min(cut, remaining.endIndex)
            let piece = String(remaining[..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { out.append(piece) }
            remaining = remaining[cut...]
        }
        let tail = remaining.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out
    }
}

// MARK: - Word timing (port of utils.tsx calculateWordTimings)

struct WordTiming: Equatable {
    let word: String
    let startFrac: Double   // 0...1 of the audio duration
    let endFrac: Double
}

enum WordTimingCalculator {
    // The reader view recomputes this for the active sentence on every
    // 60fps karaoke tick (only `activeWordIndex` changed, not the sentence),
    // so a single-entry memo avoids re-splitting/re-measuring the same
    // string dozens of times a second. Bounded to one entry — only the
    // currently-active sentence is ever queried repeatedly.
    private static var lastSentence: String?
    private static var lastResult: [WordTiming] = []

    /// Distributes duration proportionally to character counts (incl. trailing space).
    static func timings(for sentence: String) -> [WordTiming] {
        if sentence == lastSentence { return lastResult }
        let result = computeTimings(for: sentence)
        lastSentence = sentence
        lastResult = result
        return result
    }

    private static func computeTimings(for sentence: String) -> [WordTiming] {
        let words = sentence.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard !words.isEmpty else { return [] }
        let counts = words.map { Double($0.count + 1) }
        let total = counts.reduce(0, +)
        var out: [WordTiming] = []
        var acc = 0.0
        for (i, w) in words.enumerated() {
            let start = acc / total
            acc += counts[i]
            out.append(WordTiming(word: w, startFrac: start, endFrac: acc / total))
        }
        return out
    }
}

// MARK: - Themes (port of theme.ts)

struct ReaderTheme {
    let name: String
    let isDark: Bool
    let bg: String
    let text: String
    let textMuted: String
    let dropBg: String
    let dropBorder: String
    let headerColor: String
    let menuBg: String
    let menuBorder: String
    let inputBg: String
    let inputBorder: String
    let barBg: String
    let barBorder: String
    let barIconColor: String
    let speedHighlight: String
    let accent: String

    static let all: [ReaderTheme] = [
        ReaderTheme(name: "Original", isDark: false, bg: "#f5efe3", text: "#3a3028", textMuted: "#7a6e60",
                    dropBg: "#faf6ef", dropBorder: "#c4b8a0", headerColor: "#2c2218", menuBg: "#f0ebe0",
                    menuBorder: "#d0c4b0", inputBg: "#faf6ef", inputBorder: "#c4b8a0",
                    barBg: "#2a2015", barBorder: "#1a1510", barIconColor: "#b8ac9c",
                    speedHighlight: "#d8e8f4", accent: "#2563eb"),
        ReaderTheme(name: "Quiet", isDark: true, bg: "#1a1917", text: "#c8bfb0", textMuted: "#8a8070",
                    dropBg: "#242018", dropBorder: "#4a4235", headerColor: "#e8ddd0", menuBg: "#242018",
                    menuBorder: "#3a3428", inputBg: "#2a2520", inputBorder: "#4a4235",
                    barBg: "#ede8df", barBorder: "#d0c8b8", barIconColor: "#5a4f44",
                    speedHighlight: "#2a3a50", accent: "#60a5fa"),
        ReaderTheme(name: "Paper", isDark: false, bg: "#ffffff", text: "#111111", textMuted: "#777777",
                    dropBg: "#f8f8f8", dropBorder: "#d0d0d0", headerColor: "#000000", menuBg: "#f5f5f5",
                    menuBorder: "#e0e0e0", inputBg: "#ffffff", inputBorder: "#cccccc",
                    barBg: "#1a1a1a", barBorder: "#0a0a0a", barIconColor: "#aaaaaa",
                    speedHighlight: "#dbeafe", accent: "#2563eb"),
        ReaderTheme(name: "Bold", isDark: true, bg: "#0d0d0d", text: "#f0f0f0", textMuted: "#888888",
                    dropBg: "#1a1a1a", dropBorder: "#333333", headerColor: "#ffffff", menuBg: "#1a1a1a",
                    menuBorder: "#2a2a2a", inputBg: "#1a1a1a", inputBorder: "#333333",
                    barBg: "#f0f0f0", barBorder: "#e0e0e0", barIconColor: "#444444",
                    speedHighlight: "#1e3a5f", accent: "#60a5fa"),
        ReaderTheme(name: "Calm", isDark: false, bg: "#e8d5b5", text: "#3a2c1a", textMuted: "#7a6040",
                    dropBg: "#f0e2c5", dropBorder: "#c8a878", headerColor: "#2a1c0a", menuBg: "#e0cca8",
                    menuBorder: "#c0a070", inputBg: "#ecddb8", inputBorder: "#c8a878",
                    barBg: "#3a2c1a", barBorder: "#2a1c0a", barIconColor: "#c8a878",
                    speedHighlight: "#c8dcf0", accent: "#2563eb"),
        ReaderTheme(name: "Focus", isDark: false, bg: "#f0f0ec", text: "#282828", textMuted: "#6a6a6a",
                    dropBg: "#f8f8f4", dropBorder: "#c8c8b8", headerColor: "#181818", menuBg: "#e8e8e0",
                    menuBorder: "#c0c0b0", inputBg: "#f8f8f4", inputBorder: "#c8c8b8",
                    barBg: "#22221c", barBorder: "#12120e", barIconColor: "#a8a890",
                    speedHighlight: "#d8e4f0", accent: "#2563eb"),
    ]

    static func named(_ name: String) -> ReaderTheme {
        all.first { $0.name == name } ?? all[0]
    }
}
