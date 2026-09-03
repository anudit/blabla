//
//  HTMLToBlocks.swift
//  tts-metal
//
//  Tolerant HTML → DocBlock converter. Handles headings, paragraphs, lists,
//  blockquotes, pre/code, tables, images and rules; strips scripts/styles.
//  Used by the EPUB loader and the URL loader. Ported from blabla's
//  markdown/epub block extraction behavior.
//
//  Two things beyond plain tag handling matter for real ebooks:
//
//  * Publisher exports (InDesign, Calibre, Sigil) almost never emit <h1>…<h6>.
//    Everything is <p class="Heading-3">, <p class="Quote">, <p class="ImageLine">.
//    Structure therefore has to be recovered from class names — see `ParaKind`.
//  * Footnote/endnote reference markers are inline spans holding a bare number.
//    Left in place they glue onto the preceding word ("…between people.75 Under
//    experimental…"), which both reads wrong on screen and is spoken aloud as a
//    number in the middle of a sentence — so they're dropped, see `capture`.
//

import Foundation

final class HTMLToBlocks {

    /// Archive directory of the document being parsed, used to resolve
    /// relative `<img src>` into a full archive path (EPUB). Empty for
    /// sources with no archive (MOBI, raw HTML).
    private let baseDir: String
    /// Page URL for web documents, used to absolutize relative image srcs.
    private let baseURL: URL?

    init(baseDir: String = "", baseURL: URL? = nil) {
        self.baseDir = baseDir
        self.baseURL = baseURL
    }

    private var blocks: [DocBlock] = []
    private var outline: [OutlineEntry] = []
    private var anchors: [String: Int] = [:]   // element id → index of the block it introduces
    private var text = ""                 // accumulated inline text
    private var paragraphOpen = false
    private var paragraphKind: ParaKind = .paragraph
    private var skipDepth = 0             // inside script/style/head
    private var svgDepth = 0              // inside <svg> — text is noise, <image> is not
    private var quoteText: String?
    private var listItemTag: String?      // "ul" | "ol"
    private var tableRows: [[String]] = []
    private var currentRow: [String] = []
    private var currentCell = ""
    private var cellIsOpen = false
    private var codeText: String?
    private var pendingHr = false
    /// Nesting depth of inline elements (span/a/sup) — see `capture`.
    private var inlineDepth = 0
    /// Text of an inline element being held back: footnote markers are
    /// discarded outright (`drop`), <sup> is kept only if it doesn't look
    /// like a reference marker.
    private var capture: (depth: Int, text: String, drop: Bool)?
    private static let voidTags: Set<String> = ["br", "hr", "img", "meta", "link", "input", "col", "area", "base", "embed", "source", "track", "wbr"]

    struct Result {
        var blocks: [DocBlock]
        var outline: [OutlineEntry]
        var title: String?
        /// Element id → block index, so an EPUB TOC href with a fragment
        /// ("chapter.xhtml#_idParaDest-12") can land on the right block even
        /// when many TOC entries share one spine file.
        var anchors: [String: Int] = [:]
    }
    private var firstHeading: String?

    func parse(_ html: String) -> Result {
        reset()
        let src = html.replacingOccurrences(of: "\u{feff}", with: "")
        var i = src.startIndex
        while i < src.endIndex {
            let c = src[i]
            if c == "<", let tagEnd = findTagEnd(src, from: i) {
                handleTag(String(src[src.index(after: i)..<tagEnd]))
                i = src.index(after: tagEnd)
            } else {
                if skipDepth == 0 && svgDepth == 0 { append(c) }
                i = src.index(after: i)
            }
        }
        closeParagraph()
        return Result(blocks: blocks, outline: outline, title: firstHeading, anchors: anchors)
    }

    private func reset() {
        blocks = []; outline = []; anchors = [:]; text = ""; paragraphOpen = false
        paragraphKind = .paragraph
        skipDepth = 0; svgDepth = 0; quoteText = nil; listItemTag = nil
        tableRows = []; currentRow = []; currentCell = ""; cellIsOpen = false
        codeText = nil; firstHeading = nil
        inlineDepth = 0; capture = nil
    }

    private func append(_ c: Character) {
        if capture != nil { capture!.text.append(c) } else { text.append(c) }
    }

    private func append(_ s: String) {
        if capture != nil { capture!.text += s } else { text += s }
    }

    // MARK: - Paragraph kinds recovered from class names

    private enum ParaKind: Equatable {
        case paragraph
        case heading(Int)
        case quote
        case listItem
        case caption
        case rule
    }

    /// Maps a publisher stylesheet class onto document structure. Matching is
    /// deliberately loose (substring, case-insensitive) because every export
    /// tool names these differently: "Heading-3", "chapter_title", "h2",
    /// "ImageLine", "Image-Atrrib-Text" (sic — InDesign's own typo).
    private static func kind(forClass raw: String) -> ParaKind? {
        guard !raw.isEmpty else { return nil }
        let c = raw.lowercased()

        // Explicitly numbered heading classes: heading-3, head2, title1, h4.
        if let re = RegexCache.regex(#"\b(?:heading|header|head|title|h)[-_ ]?([1-6])\b"#),
           let m = re.firstMatch(in: c, range: NSRange(c.startIndex..., in: c)),
           m.numberOfRanges > 1, let r = Range(m.range(at: 1), in: c),
           let level = Int(c[r]) {
            return .heading(level)
        }

        // "Label" is InDesign's own class for a figure caption ("Image 1 – …"),
        // and it is the one caption name that carries no "caption"/"credit"
        // substring — left unmapped it renders as body text *and* gets spoken
        // aloud in the middle of a chapter.
        if c.contains("caption") || c.contains("credit") || c.contains("atrrib")
            || c.contains("attrib") || c.contains("label") || c.contains("figure-title") {
            return .caption
        }
        if c.contains("rule") || c.contains("ornament") || c.contains("divider") {
            return .rule
        }
        if c.contains("quote") || c.contains("epigraph") || c.contains("pullquote") || c.contains("extract") {
            return .quote
        }
        if c.contains("bullet") || c.contains("listnumbered") || c.contains("list-number")
            || c.contains("list-bullet") || c.contains("listitem") || c.contains("list-item") {
            return .listItem
        }
        // Unnumbered structural headings: "Part-Start", "Chapter-heading",
        // "Contradiction-heading", "Additional-sources-header", "BookTitle".
        if c.contains("part-") || c.contains("part_") || c.contains("parttitle") { return .heading(1) }
        if c.contains("chapter") && (c.contains("head") || c.contains("title") || c.contains("num")) {
            return .heading(2)
        }
        // Unnumbered fallbacks rank *below* any explicitly numbered heading
        // class. A chapter that uses "Heading-4" for its section heads and
        // "Additional-sources-header" for its endmatter head has to see the
        // latter as the deeper of the two; scoring the generic one at 3 made
        // it the shallowest heading in the chapter and pushed every real
        // section head a level further in than it belongs.
        if c.contains("subhead") || c.contains("crosshead") { return .heading(5) }
        if c.contains("heading") || c.contains("header") { return .heading(4) }
        return nil
    }

    /// Inline classes whose text is a reference marker, never prose.
    private static func isFootnoteRefClass(_ raw: String) -> Bool {
        let c = raw.lowercased()
        return c.contains("footnote-number") || c.contains("footnotenumber")
            || c.contains("endnote-reference") || c.contains("endnotereference")
            || c.contains("footnotelink") || c.contains("noteref")
            || c.contains("footnote-ref") || c.contains("endnote-ref")
    }

    /// A held-back <sup> that reads as "1", "12,13", "[4]", "*" is a
    /// reference marker; anything longer (units, "E=mc2" style exponents,
    /// "1st") is real content and gets kept.
    private static func looksLikeRefMarker(_ s: String) -> Bool {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, t.count <= 8 else { return false }
        return t.range(of: #"^[\d\s.,;\[\]()*†‡§¶–-]+$"#, options: .regularExpression) != nil
    }

    // MARK: - Tag handling

    private func handleTag(_ raw: String) {
        if raw.hasPrefix("!") || raw.hasPrefix("?") { return }   // comment / doctype / XML decl
        let closing = raw.hasPrefix("/")
        let body = closing ? String(raw.dropFirst()) : raw
        let selfClosing = body.hasSuffix("/")
        let name = body.prefix(while: { $0.isLetter || $0.isNumber }).lowercased()

        if !closing, skipDepth == 0, body.contains("id=") {
            let id = attribute(named: "id", in: body)
            // Record the block this anchor introduces. Pending paragraph text
            // hasn't been flushed yet, so the next appended block is the one
            // the anchor belongs to.
            if !id.isEmpty, anchors[id] == nil { anchors[id] = blocks.count }
        }

        switch name {
        case "script", "style", "head", "noscript":
            if closing { skipDepth = max(0, skipDepth - 1) }
            else if !selfClosing { skipDepth += 1; closeParagraph() }

        case "svg":
            // Cover pages wrap the artwork in <svg><image xlink:href=…/></svg>.
            // The vector markup itself is noise, but that <image> is the page,
            // so the skip here is text-only and `image` is still handled below.
            if closing { svgDepth = max(0, svgDepth - 1) }
            else if !selfClosing { svgDepth += 1; closeParagraph() }

        case "p":
            if closing {
                closeParagraph()
            } else {
                closeParagraph()
                paragraphOpen = true
                paragraphKind = classKind(in: body) ?? .paragraph
                if paragraphKind == .rule {
                    paragraphOpen = false
                    appendBlock(.rule)
                }
            }

        case "h1", "h2", "h3", "h4", "h5", "h6":
            if closing {
                let level = Int(String(name.last!)) ?? 6
                emitParagraphAsHeading(level: level)
            } else {
                closeParagraph()
                paragraphOpen = true
                paragraphKind = .heading(Int(String(name.last!)) ?? 6)
            }

        case "br":
            append(" ")

        case "hr":
            closeParagraph()
            appendBlock(.rule)

        case "blockquote":
            if closing {
                if let q = quoteText {
                    let t = clean(q)
                    if !t.isEmpty { appendBlock(.quote(t)) }
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
                    if !t.isEmpty { appendBlock(.code(t)) }
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
                paragraphOpen = false
                if !t.isEmpty { appendBlock(.listItem(t.hasPunctuationTerminal ? t : t + ".")) }
            } else {
                closeParagraph()
                paragraphOpen = true
                paragraphKind = .listItem
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

        case "img", "image":
            guard !closing, skipDepth == 0 else { break }
            emitImage(from: body)

        case "span", "a", "sup":
            // Inline elements. Tracked by depth so a footnote marker's text
            // can be withheld until its own closing tag — see `capture`.
            if selfClosing { break }
            if closing {
                if let c = capture, c.depth == inlineDepth {
                    if !c.drop, !Self.looksLikeRefMarker(c.text) { text += c.text }
                    capture = nil
                }
                inlineDepth = max(0, inlineDepth - 1)
            } else {
                inlineDepth += 1
                if capture == nil, skipDepth == 0, svgDepth == 0 {
                    let cls = body.contains("class=") ? attribute(named: "class", in: body) : ""
                    if Self.isFootnoteRefClass(cls) || attribute(named: "epub:type", in: body) == "noteref" {
                        capture = (inlineDepth, "", true)
                    } else if name == "sup" {
                        capture = (inlineDepth, "", false)
                    }
                }
            }

        case "div", "section", "article", "header", "footer", "main", "nav", "aside",
             "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "body", "html":
            if ["ul", "ol"].contains(name) && !closing { closeParagraph() }
            if ["div", "section", "article", "figure", "figcaption"].contains(name) {
                if closing {
                    closeParagraph()
                } else {
                    // Some exporters (e.g. Calibre) wrap every paragraph in a
                    // styled <div> instead of <p>. Without opening a paragraph
                    // here, that div's text is accumulated but never flushed
                    // (closeParagraph() is a no-op while paragraphOpen is
                    // false), so the whole document silently parses to zero
                    // blocks. Treat block-level div/section/article/figure
                    // like <p>: flush whatever's pending, then start capturing
                    // this element's own text as a paragraph.
                    closeParagraph()
                    paragraphOpen = true
                    paragraphKind = name == "figcaption" ? .caption : (classKind(in: body) ?? .paragraph)
                }
            }

        default:
            break
        }
    }

    private func classKind(in tagBody: String) -> ParaKind? {
        guard tagBody.contains("class=") else { return nil }
        return Self.kind(forClass: attribute(named: "class", in: tagBody))
    }

    private func appendBlock(_ content: BlockContent) {
        blocks.append(DocBlock(content: content))
    }

    /// Flushes any text accumulated before the image so the picture keeps its
    /// position in the flow, then appends the image itself.
    private func emitImage(from tagBody: String) {
        let pending = clean(text)
        text = ""
        if !pending.isEmpty { emit(pending, as: paragraphKind) }

        var src = attribute(named: "src", in: tagBody)
        if src.isEmpty { src = attribute(named: "xlink:href", in: tagBody) }
        if src.isEmpty { src = attribute(named: "href", in: tagBody) }
        // Publishers routinely set alt to the asset's own file name
        // ("EIcover.jpg"), which is not a description of anything — shown as a
        // caption it prints the filename under the cover art.
        var alt = attribute(named: "alt", in: tagBody)
        if alt.range(of: #"^[\w .-]+\.(?:jpe?g|png|gif|svg|webp|tiff?)$"#,
                     options: [.regularExpression, .caseInsensitive]) != nil { alt = "" }
        guard !src.isEmpty || !alt.isEmpty else { return }
        appendBlock(.image(alt: alt, src: resolve(src)))
    }

    private func resolve(_ src: String) -> String {
        guard !src.isEmpty else { return "" }
        if src.hasPrefix("http://") || src.hasPrefix("https://") || src.hasPrefix("data:") { return src }
        if src.hasPrefix("//") { return "https:" + src }
        if let base = baseURL, let abs = URL(string: src, relativeTo: base)?.absoluteString { return abs }
        return EPUBLoader.resolve(src, base: baseDir)
    }

    private func closeParagraph() {
        // An unbalanced inline tag inside the paragraph must not leak its
        // capture into the next one.
        if let c = capture {
            if !c.drop, !Self.looksLikeRefMarker(c.text) { text += c.text }
            capture = nil
        }
        inlineDepth = 0
        guard paragraphOpen else { text = ""; return }
        paragraphOpen = false
        let t = clean(text)
        text = ""
        guard !t.isEmpty else { return }
        emit(t, as: paragraphKind)
        paragraphKind = .paragraph
    }

    private func emit(_ t: String, as kind: ParaKind) {
        switch kind {
        case .paragraph:
            appendBlock(.paragraph(t))
        case .heading(let level):
            appendBlock(.heading(level: level, text: t))
            outline.append(OutlineEntry(level: level, title: t, blockIndex: blocks.count - 1))
            if firstHeading == nil, level <= 2 { firstHeading = t }
        case .quote:
            appendBlock(.quote(t))
        case .listItem:
            appendBlock(.listItem(t.hasPunctuationTerminal ? t : t + "."))
        case .caption:
            appendBlock(.caption(t))
        case .rule:
            appendBlock(.rule)
        }
    }

    private func emitParagraphAsHeading(level: Int) {
        paragraphOpen = false
        let t = clean(text)
        text = ""
        paragraphKind = .paragraph
        guard !t.isEmpty else { return }
        emit(t, as: .heading(level))
    }

    private func finishTable() {
        closeParagraph()
        if cellIsOpen { currentRow.append(clean(text)); text = ""; cellIsOpen = false }
        if !currentRow.isEmpty { tableRows.append(currentRow); currentRow = [] }
        // Drop separator rows like | --- | --- |
        let rows = tableRows.filter { row in !row.allSatisfy { $0.range(of: #"^[-–—:\s|]*$"#, options: .regularExpression) != nil } }
        if !rows.isEmpty { appendBlock(.table(rows: rows)) }
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
