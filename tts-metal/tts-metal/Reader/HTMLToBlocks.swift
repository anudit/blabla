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

    /// The book's stylesheets (EPUB). Any `<style>` blocks in the parsed
    /// document are added to it per parse.
    private let baseStyles: CSSStyleSheet
    private var styles = CSSStyleSheet.empty

    init(baseDir: String = "", baseURL: URL? = nil, styles: CSSStyleSheet = .empty) {
        self.baseDir = baseDir
        self.baseURL = baseURL
        self.baseStyles = styles
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
    /// Open inline elements (span/a/sup/i/…), innermost last. `styled` means
    /// an open style marker was written for it and a close one is owed.
    private var inlineStack: [(name: String, styled: Bool, link: Bool)] = []
    /// Text of an inline element being held back: footnote markers are
    /// discarded outright (`drop`), <sup> is kept only if it doesn't look
    /// like a reference marker. `linked` records a hyperlink around or
    /// inside it — the strongest sign a superscript number is a note ref.
    private var capture: (depth: Int, text: String, drop: Bool, linked: Bool)?
    /// Open block containers (div/section/blockquote/…) and the alignment
    /// each one set, so `text-align` inherits the way it does in CSS.
    private var containerStack: [(name: String, align: CSSStyleSheet.Align?)] = []
    /// Alignment of the paragraph being accumulated.
    private var paragraphAlign: CSSStyleSheet.Align?
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
        styles = baseStyles
        if src.range(of: "<style", options: .caseInsensitive) != nil,
           let re = RegexCache.regex(#"<style[^>]*>([\s\S]*?)</style>"#, options: .caseInsensitive) {
            let ns = src as NSString
            for m in re.matches(in: src, range: NSRange(location: 0, length: ns.length)) {
                styles.add(css: ns.substring(with: m.range(at: 1)))
            }
        }
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
        inlineStack = []; capture = nil
        containerStack = []; paragraphAlign = nil
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
        let t = stripMarkers(s).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, t.count <= 8 else { return false }
        return t.range(of: #"^[\d\s.,;\[\]()*†‡§¶–-]+$"#, options: .regularExpression) != nil
    }

    /// Whether a held-back <sup> is a note reference rather than content.
    /// Marker-shaped text alone isn't enough: in a maths book `2<sup>5</sup>`
    /// and `x<sup>2</sup>` are exponents, and dropping them turned "2⁵ (= 32)"
    /// into "2 (= 32)". A reference is linked to its note, or sits after
    /// punctuation/space ("…between people.⁷⁵"); an exponent follows the
    /// letter, digit or bracket it raises.
    private func isNoteReference(_ c: (depth: Int, text: String, drop: Bool, linked: Bool)) -> Bool {
        guard Self.looksLikeRefMarker(c.text) else { return false }
        if c.linked { return true }
        guard let prev = text.unicodeScalars.last(where: { !Self.isMarker($0) }) else { return true }
        return CharacterSet.whitespacesAndNewlines.contains(prev)
            || ".,;:!?\"'\u{201D}\u{2019}\u{2014}".unicodeScalars.contains(prev)
    }

    // MARK: - Inline style markers
    //
    // Styled inline elements write a private-use scalar into the text stream
    // where they open and close. The markers travel through the existing
    // accumulate → capture → clean pipeline untouched, and `cleanStyled`
    // turns them into `StyleRun` ranges while it collapses whitespace — so
    // offsets are computed against the final display text, not the raw HTML.
    // Plane-15 private use, which no book text uses.

    private static let markerBase: UInt32 = 0xF0000
    private static let closeMarker = Character(Unicode.Scalar(0xF00FF as UInt32)!)

    private static func openMarker(_ style: InlineStyle) -> Character {
        Character(Unicode.Scalar(markerBase + UInt32(style.rawValue))!)
    }

    private static func isMarker(_ s: Unicode.Scalar) -> Bool {
        s.value >= markerBase && s.value <= 0xF00FF
    }

    private static func stripMarkers(_ s: String) -> String {
        guard s.unicodeScalars.contains(where: isMarker) else { return s }
        var v = String.UnicodeScalarView()
        v.append(contentsOf: s.unicodeScalars.filter { !isMarker($0) })
        return String(v)
    }

    /// Inline style an element asks for: what the tag means by default,
    /// overridden by the stylesheet, overridden by its own `style=`.
    private func inlineStyle(tag: String, body: String) -> InlineStyle {
        var d = CSSStyleSheet.Declarations()
        switch tag {
        case "i", "em", "cite", "var", "dfn": d.italic = true
        case "b", "strong": d.bold = true
        case "sup": d.vertical = 1
        case "sub": d.vertical = -1
        default: break
        }
        d.merge(cssDeclarations(tag: tag, body: body))
        var s: InlineStyle = []
        if let i = d.italic { s.insert(i ? .italic : .upright) }
        if d.bold == true { s.insert(.bold) }
        if d.vertical == 1 { s.insert(.superscript) }
        if d.vertical == -1 { s.insert(.subscript) }
        return s
    }

    private func cssDeclarations(tag: String, body: String) -> CSSStyleSheet.Declarations {
        let classes = body.contains("class=")
            ? attribute(named: "class", in: body).lowercased().split(separator: " ").map(String.init)
            : []
        var d = styles.isEmpty ? CSSStyleSheet.Declarations()
                               : styles.declarations(tag: tag, classes: classes)
        if body.contains("style=") {
            d.merge(CSSStyleSheet.parseDeclarations(attribute(named: "style", in: body)))
        }
        if body.contains("align=") {
            switch attribute(named: "align", in: body).lowercased() {
            case "center": d.align = .center
            case "right": d.align = .right
            case "left": d.align = .left
            case "justify": d.align = .justify
            default: break
            }
        }
        return d
    }

    /// Nearest alignment set by an enclosing container.
    private var inheritedAlign: CSSStyleSheet.Align? {
        containerStack.last(where: { $0.align != nil })?.align
    }

    private func openInline(_ name: String, body: String) {
        let link = name == "a" && body.contains("href=")
        inlineStack.append((name, false, link))
        if capture == nil, skipDepth == 0, svgDepth == 0 {
            let cls = body.contains("class=") ? attribute(named: "class", in: body) : ""
            if Self.isFootnoteRefClass(cls) || attribute(named: "epub:type", in: body) == "noteref" {
                capture = (inlineStack.count, "", true, link)
            } else if name == "sup" {
                capture = (inlineStack.count, "", false, inlineStack.contains { $0.link })
            }
        } else if link, capture != nil {
            capture!.linked = true
        }
        let style = inlineStyle(tag: name, body: body)
        if !style.isEmpty {
            append(Self.openMarker(style))
            inlineStack[inlineStack.count - 1].styled = true
        }
    }

    private func closeInline(_ name: String) {
        guard let idx = inlineStack.lastIndex(where: { $0.name == name }) else { return }
        // Pops any unclosed children too, so a stray tag can't strand a
        // style or a capture.
        while inlineStack.count > idx {
            let e = inlineStack.removeLast()
            if e.styled { append(Self.closeMarker) }
            if let c = capture, c.depth == inlineStack.count + 1 {
                capture = nil
                if !c.drop, !isNoteReference(c) { text += c.text }
            }
        }
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
                paragraphAlign = cssDeclarations(tag: "p", body: body).align ?? inheritedAlign
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

        case "center":
            if closing { popContainer("center") }
            else { closeParagraph(); containerStack.append(("center", .center)) }

        case "blockquote":
            if closing { popContainer("blockquote") }
            else if !selfClosing {
                containerStack.append(("blockquote", cssDeclarations(tag: name, body: body).align))
            }
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
                let t = cleanStyled(text)
                text = ""
                paragraphOpen = false
                if !t.text.isEmpty { emit(t, as: .listItem) }
                paragraphAlign = nil
            } else {
                closeParagraph()
                paragraphOpen = true
                paragraphKind = .listItem
                paragraphAlign = cssDeclarations(tag: "li", body: body).align ?? inheritedAlign
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

        case "span", "a", "sup", "sub", "i", "em", "b", "strong", "cite", "var", "dfn",
             "small", "abbr", "u", "font":
            // Inline elements. Tracked as a stack so a footnote marker's text
            // can be withheld until its own closing tag (see `capture`), and
            // so styled ones (<i>, <sub>, a CSS-italic span) can mark where
            // their style starts and ends in the text.
            if selfClosing { break }
            if closing { closeInline(name) } else { openInline(name, body: body) }

        case "div", "section", "article", "header", "footer", "main", "nav", "aside",
             "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "body", "html":
            if ["ul", "ol"].contains(name) && !closing { closeParagraph() }
            if ["div", "section", "article", "figure", "figcaption", "aside", "body"].contains(name) {
                if closing { popContainer(name) }
                else if !selfClosing {
                    containerStack.append((name, cssDeclarations(tag: name, body: body).align))
                }
            }
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
                    paragraphAlign = inheritedAlign
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
        let pending = cleanStyled(text)
        text = ""
        if !pending.text.isEmpty { emit(pending, as: paragraphKind) }

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
            capture = nil
            if !c.drop, !isNoteReference(c) { text += c.text }
        }
        inlineStack = []
        guard paragraphOpen else { text = ""; return }
        paragraphOpen = false
        let t = cleanStyled(text)
        text = ""
        guard !t.text.isEmpty else { return }
        emit(t, as: paragraphKind)
        paragraphKind = .paragraph
        paragraphAlign = nil
    }

    private func popContainer(_ name: String) {
        guard let idx = containerStack.lastIndex(where: { $0.name == name }) else { return }
        containerStack.removeSubrange(idx...)
    }

    private func emit(_ styled: (text: String, runs: [StyleRun]), as kind: ParaKind) {
        let t = styled.text
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
            appendBlock(.rule); return
        }
        blocks[blocks.count - 1].runs = styled.runs
        switch paragraphAlign {
        case .center: blocks[blocks.count - 1].alignment = .center
        case .right:  blocks[blocks.count - 1].alignment = .right
        case .left:   blocks[blocks.count - 1].alignment = .left
        case .justify, nil: break
        }
    }

    private func emitParagraphAsHeading(level: Int) {
        paragraphOpen = false
        let t = cleanStyled(text)
        text = ""
        paragraphKind = .paragraph
        guard !t.text.isEmpty else { return }
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
        cleanStyled(s).text
    }

    /// Decodes entities, collapses whitespace runs to one space and trims —
    /// and, in the same pass, turns the inline style markers into ranges
    /// over the cleaned string. A space between two differently styled
    /// stretches belongs to the one before it, so `x<sub>2 </sub>and` keeps
    /// the gap out of the subscript.
    private func cleanStyled(_ s: String) -> (text: String, runs: [StyleRun]) {
        let decoded = Self.decodeEntities(s)
        var out = String.UnicodeScalarView()
        var length = 0                       // UTF-16 length of `out`
        var lastWasSpace = true              // suppresses leading space
        var pendingSpace = false
        var stack: [InlineStyle] = []
        var current: InlineStyle = []
        var runStart = 0
        var runs: [StyleRun] = []

        func effective() -> InlineStyle {
            var e: InlineStyle = []
            for st in stack {
                if st.contains(.italic) { e.insert(.italic); e.remove(.upright) }
                if st.contains(.upright) { e.remove(.italic); e.insert(.upright) }
                if st.contains(.bold) { e.insert(.bold) }
                if st.contains(.superscript) { e.insert(.superscript); e.remove(.subscript) }
                if st.contains(.subscript) { e.insert(.subscript); e.remove(.superscript) }
            }
            return e
        }
        func flushSpace() {
            if pendingSpace, !lastWasSpace { out.append(" "); length += 1; lastWasSpace = true }
            pendingSpace = false
        }
        func styleChanged() {
            let next = effective()
            guard next != current else { return }
            flushSpace()
            // A space at the end of the run ("<sub>2 </sub>") stays unstyled,
            // so the gap after a subscript is a full-size space.
            let end = lastWasSpace ? length - 1 : length
            if !current.isEmpty, end > runStart {
                runs.append(StyleRun(range: NSRange(location: runStart, length: end - runStart), style: current))
            }
            current = next
            runStart = length
        }

        for sc in decoded.unicodeScalars {
            if Self.isMarker(sc) {
                if sc.value == 0xF00FF { if !stack.isEmpty { stack.removeLast() } }
                else { stack.append(InlineStyle(rawValue: UInt8(sc.value - Self.markerBase))) }
                styleChanged()
            } else if CharacterSet.whitespacesAndNewlines.contains(sc) {
                pendingSpace = true
            } else {
                flushSpace()
                out.append(sc)
                length += sc.utf16.count
                lastWasSpace = false
            }
        }
        if !current.isEmpty, length > runStart {
            runs.append(StyleRun(range: NSRange(location: runStart, length: length - runStart), style: current))
        }
        if lastWasSpace, length > 0 {          // a space flushed at a style change, then nothing
            out.removeLast(); length -= 1
        }
        // Clip to the trimmed length and merge touching runs of one style
        // ("<i>x</i><i>2</i>").
        var merged: [StyleRun] = []
        for r in runs {
            let upper = min(r.range.upperBound, length)
            guard upper > r.range.location else { continue }
            let clipped = NSRange(location: r.range.location, length: upper - r.range.location)
            if let last = merged.last, last.style == r.style, last.range.upperBound == clipped.location {
                merged[merged.count - 1] = StyleRun(
                    range: NSRange(location: last.range.location, length: clipped.upperBound - last.range.location),
                    style: r.style)
            } else {
                merged.append(StyleRun(range: clipped, style: r.style))
            }
        }
        return (String(out), merged)
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
