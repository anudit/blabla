//
//  HTMLToBlocks.swift
//  tts-metal
//
//  Tolerant HTML → DocBlock converter. Handles headings, paragraphs, lists,
//  blockquotes, pre/code, tables, images and rules; strips scripts/styles.
//  Used by the EPUB loader and the URL loader. Ported from blabla's
//  markdown/epub block extraction behavior.
//

import Foundation

final class HTMLToBlocks {

    private var blocks: [DocBlock] = []
    private var outline: [OutlineEntry] = []
    private var text = ""                 // accumulated inline text
    private var paragraphOpen = false
    private var skipDepth = 0             // inside script/style/head
    private var quoteText: String?
    private var listItemTag: String?      // "ul" | "ol"
    private var tableRows: [[String]] = []
    private var currentRow: [String] = []
    private var currentCell = ""
    private var cellIsOpen = false
    private var codeText: String?
    private var pendingHr = false
    private static let voidTags: Set<String> = ["br", "hr", "img", "meta", "link", "input", "col", "area", "base", "embed", "source", "track", "wbr"]

    struct Result {
        var blocks: [DocBlock]
        var outline: [OutlineEntry]
        var title: String?
    }
    private var firstHeading: String?

    func parse(_ html: String) -> Result {
        reset()
        let src = html.replacingOccurrences(of: "\u{feff}", with: "")
        var i = src.startIndex
        while i < src.endIndex {
            let c = src[i]
            if c == "<", let tagEnd = findTagEnd(src, from: i) {
                flushBreakIfNeeded()
                handleTag(String(src[src.index(after: i)..<tagEnd]))
                i = src.index(after: tagEnd)
            } else {
                if skipDepth == 0 { text.append(c) }
                i = src.index(after: i)
            }
        }
        closeParagraph()
        return Result(blocks: blocks.compactMap(simplify), outline: outline, title: firstHeading)
    }

    private func reset() {
        blocks = []; outline = []; text = ""; paragraphOpen = false
        skipDepth = 0; quoteText = nil; listItemTag = nil
        tableRows = []; currentRow = []; currentCell = ""; cellIsOpen = false
        codeText = nil; firstHeading = nil
    }

    private func simplify(_ b: DocBlock) -> DocBlock? {
        switch b.content {
        case .paragraph(let t) where t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
            return nil
        case .heading(_, let t) where t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
            return nil
        default:
            return b
        }
    }

    // MARK: - Tag handling

    private func handleTag(_ raw: String) {
        if raw.hasPrefix("!") { return }                       // comment / doctype
        let closing = raw.hasPrefix("/")
        let body = closing ? String(raw.dropFirst()) : raw
        let name = body.prefix(while: { $0.isLetter || $0.isNumber }).lowercased()

        switch name {
        case "script", "style", "head", "svg", "noscript":
            if closing { skipDepth = max(0, skipDepth - 1) }
            else { skipDepth += 1; closeParagraph() }

        case "p":
            closeParagraph()
            paragraphOpen = true

        case "h1", "h2", "h3", "h4", "h5", "h6":
            if closing {
                let level = Int(String(name.last!)) ?? 6
                emitParagraphAsHeading(level: level)
            } else {
                closeParagraph()
                paragraphOpen = true
            }

        case "br":
            text.append(" ")

        case "hr":
            closeParagraph()
            blocks.append(DocBlock(content: .rule))

        case "blockquote":
            if closing {
                if let q = quoteText {
                    let t = clean(q)
                    if !t.isEmpty { blocks.append(DocBlock(content: .quote(t))) }
                }
                quoteText = nil
            } else {
                closeParagraph()
                quoteText = ""
            }

        case "pre":
            if closing {
                if let c = codeText {
                    let t = Self.decodeEntities(c).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !t.isEmpty { blocks.append(DocBlock(content: .code(t))) }
                }
                codeText = nil
            } else {
                closeParagraph()
                codeText = ""
            }

        case "li":
            if closing {
                let t = clean(text)
                text = ""
                if !t.isEmpty { blocks.append(DocBlock(content: .listItem(t.hasPunctuationTerminal ? t : t + "."))) }
            } else {
                closeParagraph()
            }

        case "table":
            if closing { finishTable() }
            else { closeParagraph() }

        case "tr":
            if closing {
                if cellIsOpen { currentRow.append(clean(text)); text = ""; cellIsOpen = false }
                if !currentRow.isEmpty { tableRows.append(currentRow); currentRow = [] }
            } else {
                if cellIsOpen { currentRow.append(clean(text)); text = ""; cellIsOpen = false }
                currentRow = []
            }

        case "td", "th":
            if closing {
                if cellIsOpen { currentRow.append(clean(text)); text = ""; cellIsOpen = false }
            } else {
                if cellIsOpen { currentRow.append(clean(text)); text = "" }
                cellIsOpen = true
            }

        case "img":
            // <img alt="..."> — record caption-ish alt text as a non-speakable marker.
            let alt = attribute(named: "alt", in: body)
            blocks.append(DocBlock(content: .image(alt: alt)))

        case "div", "section", "article", "header", "footer", "main", "nav", "aside",
             "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "body", "html", "span":
            if ["ul", "ol"].contains(name) && !closing { closeParagraph() }
            if ["div", "section", "article", "figure"].contains(name) && closing { closeParagraph() }

        default:
            break
        }
    }

    private func flushBreakIfNeeded() {}

    private func closeParagraph() {
        guard paragraphOpen else { return }
        paragraphOpen = false
        let t = clean(text)
        text = ""
        if !t.isEmpty { blocks.append(DocBlock(content: .paragraph(t))) }
    }

    private func emitParagraphAsHeading(level: Int) {
        paragraphOpen = false
        let t = clean(text)
        text = ""
        guard !t.isEmpty else { return }
        blocks.append(DocBlock(content: .heading(level: level, text: t)))
        outline.append(OutlineEntry(level: level, title: t, blockIndex: blocks.count - 1))
        if firstHeading == nil, level <= 2 { firstHeading = t }
    }

    private func finishTable() {
        closeParagraph()
        if cellIsOpen { currentRow.append(clean(text)); text = ""; cellIsOpen = false }
        if !currentRow.isEmpty { tableRows.append(currentRow); currentRow = [] }
        // Drop separator rows like | --- | --- |
        let rows = tableRows.filter { row in !row.allSatisfy { $0.range(of: #"^[-–—:\s|]*$"#, options: .regularExpression) != nil } }
        if !rows.isEmpty { blocks.append(DocBlock(content: .table(rows: rows))) }
        tableRows = []
    }

    // MARK: - Helpers

    private func findTagEnd(_ s: String, from: String.Index) -> String.Index? {
        var i = s.index(after: from)
        var inQuote: Character?
        while i < s.endIndex {
            let c = s[i]
            if let q = inQuote {
                if c == q { inQuote = nil }
            } else if c == "\"" || c == "'" {
                inQuote = c
            } else if c == ">" {
                return i
            }
            i = s.index(after: i)
        }
        return nil
    }

    private func attribute(named attr: String, in tagBody: String) -> String {
        for pattern in [#"\#(attr)\s*=\s*"([^"]*)""#, #"\#(attr)\s*=\s*'([^']*)'"#, #"\#(attr)\s*=\s*([^\s>]+)"#] {
            if let re = RegexCache.regex(pattern, options: .caseInsensitive),
               let m = re.firstMatch(in: tagBody, range: NSRange(tagBody.startIndex..., in: tagBody)),
               m.numberOfRanges > 1, let r = Range(m.range(at: 1), in: tagBody) {
                return Self.decodeEntities(String(tagBody[r]))
            }
        }
        return ""
    }

    private func clean(_ s: String) -> String {
        RegexCache.replace(Self.decodeEntities(s), pattern: "\\s+", with: " ")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static let namedEntities: [String: String] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ",
        "mdash": "—", "ndash": "–", "hellip": "…", "rsquo": "\u{2019}", "lsquo": "\u{2018}",
        "rdquo": "\u{201D}", "ldquo": "\u{201C}", "copy": "©", "reg": "®", "trade": "™",
        "eacute": "é", "egrave": "è", "agrave": "à", "ccedil": "ç", "uuml": "ü", "ouml": "ö",
        "auml": "ä", "szlig": "ß", "ntilde": "ñ", "euro": "€", "pound": "£", "deg": "°"
    ]

    static func decodeEntities(_ s: String) -> String {
        guard s.contains("&") else { return s }
        var out = ""
        var i = s.startIndex
        while i < s.endIndex {
            if s[i] == "&", let semi = s[i...].firstIndex(of: ";"),
               s.distance(from: i, to: semi) <= 10 {
                let ent = String(s[s.index(after: i)..<semi])
                if ent.hasPrefix("#x") || ent.hasPrefix("#X"), let v = UInt32(ent.dropFirst(2), radix: 16),
                   let scalar = Unicode.Scalar(v) {
                    out.append(Character(scalar))
                } else if ent.hasPrefix("#"), let v = UInt32(ent.dropFirst()), let scalar = Unicode.Scalar(v) {
                    out.append(Character(scalar))
                } else if let rep = namedEntities[ent.lowercased()] {
                    out.append(rep)
                } else {
                    out += s[i...semi]
                }
                i = s.index(after: semi)
            } else {
                out.append(s[i])
                i = s.index(after: i)
            }
        }
        return out
    }
}
