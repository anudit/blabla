//
//  BookRenderer.swift
//  tts-metal
//
//  Turns a `ReaderDocument` into one `NSAttributedString` laid out like a
//  typeset book, plus the index the reader needs to find a sentence inside it.
//
//  Why one string instead of a view per block: the reader used to render each
//  block as its own SwiftUI view inside a `LazyVStack`, with each sentence a
//  separate `Text` in a custom `Layout`. That made three things impossible at
//  once — you couldn't select text across sentences (each span was its own
//  view), ⌘F and bookmark-resume couldn't scroll to anything the lazy stack
//  hadn't mounted yet, and real book typography (justification, hanging
//  indents, drop-in images, hyphenation) has no expression in a stack of
//  independently-measured views. TextKit does all three natively: the whole
//  book is one text storage, so every character has a resolvable position
//  whether or not it has been drawn.
//

import AppKit

/// The laid-out book: the string to display and the maps back to the document.
struct RenderedBook {
    let attributed: NSAttributedString
    /// Sentence id → its range in `attributed`.
    let sentenceRanges: [Int: NSRange]
    /// Block index → its range in `attributed` (headings included, so the
    /// outline sidebar can scroll to a section that has no sentences).
    let blockRanges: [Int: NSRange]
    /// Sentence ids ordered by position, with each one's start location, so a
    /// click anywhere in the text resolves to a sentence by binary search.
    let ordered: [(location: Int, upper: Int, id: Int)]

    /// The sentence containing `location`, or the nearest one before it.
    func sentence(at location: Int) -> Int? {
        guard !ordered.isEmpty else { return nil }
        var lo = 0, hi = ordered.count - 1, best: Int?
        while lo <= hi {
            let mid = (lo + hi) / 2
            if ordered[mid].location <= location {
                best = ordered[mid].id
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
        return best
    }

    static let empty = RenderedBook(attributed: NSAttributedString(),
                                    sentenceRanges: [:], blockRanges: [:], ordered: [])
}

enum BookRenderer {

    /// Layout metrics, in points, at a font scale of 1.
    private enum Metric {
        static let bodySize: CGFloat = 16.5
        static let lineHeightMultiple: CGFloat = 1.42
        static let paragraphSpacing: CGFloat = 11
    }

    /// Builds the book. Pure and self-contained so it can run off the main
    /// thread — a full-length novel takes tens of milliseconds, but a book
    /// with dozens of large images spends real time decoding them.
    static func render(document: ReaderDocument,
                       theme: ReaderTheme,
                       fontScale: CGFloat,
                       columnWidth: CGFloat) -> RenderedBook {
        let out = NSMutableAttributedString()
        var sentenceRanges: [Int: NSRange] = [:]
        var blockRanges: [Int: NSRange] = [:]
        var ordered: [(location: Int, upper: Int, id: Int)] = []

        let body = Metric.bodySize * fontScale
        let text = NSColor(hex: theme.text)
        let muted = NSColor(hex: theme.textMuted)
        let header = NSColor(hex: theme.headerColor)

        for (bi, block) in document.blocks.enumerated() {
            let start = out.length
            switch block.content {
            case .heading(let level, let title):
                out.append(heading(title, level: level, body: body, color: header))
            case .paragraph, .quote, .listItem, .table:
                out.append(paragraph(block, body: body, color: text, muted: muted))
            case .caption(let c):
                out.append(caption(c, body: body, color: text))
            case .code(let c):
                out.append(code(c, body: body, color: text))
            case .image(let alt, let src):
                out.append(image(src: src, alt: alt, document: document,
                                 columnWidth: columnWidth, body: body, color: muted))
            case .rule:
                out.append(rule(body: body, color: muted))
            case .frontmatter:
                continue
            }

            // Sentences are positioned by their offset inside the block's own
            // display text, which is exactly what was just appended (leading
            // decorations like a list bullet shift it, hence `textOffset`).
            let offset = start + textOffset(for: block)
            for s in document.sentencesByBlock[bi] ?? [] {
                let r = NSRange(location: offset + s.range.location, length: s.range.length)
                guard r.upperBound <= out.length else { continue }
                sentenceRanges[s.id] = r
                ordered.append((r.location, r.upperBound, s.id))
            }
            if out.length > start {
                blockRanges[bi] = NSRange(location: start, length: out.length - start)
            }
        }

        // The text runs edge to edge under the floating transport bar, so the
        // book ends with enough empty space for its last lines to be scrolled
        // out from behind it. (`textContainerInset` can't do this: it applies
        // the same inset to the top, which would push the opening down.)
        if out.length > 0 {
            let tail = NSMutableParagraphStyle()
            tail.paragraphSpacingBefore = 120
            out.append(NSAttributedString(string: "\n", attributes: [
                .font: NSFont.systemFont(ofSize: body),
                .paragraphStyle: tail,
            ]))
        }

        ordered.sort { $0.location < $1.location }
        return RenderedBook(attributed: out, sentenceRanges: sentenceRanges,
                            blockRanges: blockRanges, ordered: ordered)
    }

    /// How far into a block's rendered run its `displayText` actually starts.
    /// Only list items prepend anything (the bullet and its tab).
    private static func textOffset(for block: DocBlock) -> Int {
        if case .listItem = block.content { return 2 }
        return 0
    }

    // MARK: - Block styles

    /// Heading treatment by depth, following the way a printed book grades its
    /// structure rather than a flat "each level is a bit smaller" ramp: a part
    /// title is a page-opening slab, a chapter title is large and light, and
    /// the section heads inside a chapter are small, centred and set in caps
    /// or italic so a run of them doesn't read as a wall of bold.
    private static func heading(_ title: String, level: Int,
                                body: CGFloat, color: NSColor) -> NSAttributedString {
        let l = max(1, min(6, level))
        let style = NSMutableParagraphStyle()
        style.lineHeightMultiple = 1.15
        style.paragraphSpacingBefore = [46, 40, 30, 24, 20, 18][l - 1]
        style.paragraphSpacing = [20, 22, 12, 9, 8, 8][l - 1]
        style.alignment = (l == 3 || l == 4) ? .center : .natural

        var display = title
        var font: NSFont
        var attrs: [NSAttributedString.Key: Any] = [:]

        switch l {
        case 1:
            font = .systemFont(ofSize: body * 2.05, weight: .heavy)
            attrs[.underlineStyle] = NSUnderlineStyle.single.rawValue
            attrs[.kern] = -body * 0.03
        case 2:
            font = italic(.systemFont(ofSize: body * 1.72, weight: .regular))
            attrs[.kern] = -body * 0.02
        case 3:
            display = title.uppercased()
            font = .systemFont(ofSize: body * 1.06, weight: .bold)
            attrs[.kern] = body * 0.03
        case 4:
            font = italic(.systemFont(ofSize: body * 1.06, weight: .regular))
        default:
            font = .systemFont(ofSize: body * 1.0, weight: .semibold)
        }

        attrs[.font] = font
        attrs[.foregroundColor] = color
        attrs[.paragraphStyle] = style
        return NSAttributedString(string: display + "\n", attributes: attrs)
    }

    private static func paragraph(_ block: DocBlock, body: CGFloat,
                                  color: NSColor, muted: NSColor) -> NSAttributedString {
        let style = NSMutableParagraphStyle()
        style.lineHeightMultiple = Metric.lineHeightMultiple
        style.paragraphSpacing = Metric.paragraphSpacing
        // Justified with hyphenation, the way the reference typesetting is
        // set. Without a hyphenation factor, justification of a narrow column
        // opens rivers of white space between words.
        style.alignment = .justified
        style.hyphenationFactor = 0.9

        var attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: body, weight: .regular),
            .foregroundColor: color,
            .paragraphStyle: style,
        ]
        var prefix = ""

        switch block.content {
        case .quote:
            style.headIndent = body * 1.6
            style.firstLineHeadIndent = body * 1.6
            style.tailIndent = -body * 1.6
            style.paragraphSpacingBefore = body * 0.7
            style.paragraphSpacing = body * 0.9
            style.alignment = .natural
            attrs[.font] = italic(.systemFont(ofSize: body * 1.02, weight: .regular))
            attrs[.foregroundColor] = muted
        case .listItem:
            style.headIndent = body * 1.6
            style.firstLineHeadIndent = body * 0.4
            style.alignment = .natural
            style.tabStops = [NSTextTab(textAlignment: .left, location: body * 1.6)]
            style.paragraphSpacing = body * 0.35
            prefix = "•\t"
        case .table:
            // Cells are separated by tabs in `displayText`; evenly spaced tab
            // stops keep the columns aligned without leaving the text system.
            style.alignment = .natural
            style.tabStops = (1...8).map {
                NSTextTab(textAlignment: .left, location: CGFloat($0) * body * 6)
            }
            style.paragraphSpacingBefore = body * 0.6
            style.paragraphSpacing = body * 0.6
            attrs[.font] = NSFont.systemFont(ofSize: body * 0.92, weight: .regular)
        default:
            break
        }

        return NSAttributedString(string: prefix + block.displayText + "\n", attributes: attrs)
    }

    /// Figure captions: set tight under the artwork, bold and a size down, so
    /// they read as apparatus rather than as the next paragraph of prose.
    private static func caption(_ text: String, body: CGFloat, color: NSColor) -> NSAttributedString {
        let style = NSMutableParagraphStyle()
        style.lineHeightMultiple = 1.25
        style.alignment = .justified
        style.hyphenationFactor = 0.9
        style.paragraphSpacing = body * 1.4
        return NSAttributedString(string: text + "\n", attributes: [
            .font: NSFont.systemFont(ofSize: body * 0.86, weight: .semibold),
            .foregroundColor: color,
            .paragraphStyle: style,
        ])
    }

    private static func code(_ text: String, body: CGFloat, color: NSColor) -> NSAttributedString {
        let style = NSMutableParagraphStyle()
        style.lineHeightMultiple = 1.2
        style.headIndent = body
        style.firstLineHeadIndent = body
        style.paragraphSpacingBefore = body * 0.6
        style.paragraphSpacing = body * 0.9
        return NSAttributedString(string: text + "\n", attributes: [
            .font: NSFont.monospacedSystemFont(ofSize: body * 0.82, weight: .regular),
            .foregroundColor: color,
            .paragraphStyle: style,
        ])
    }

    /// A scene break. Books set these as a centred ornament, not a full rule —
    /// a hairline across the column reads as a UI divider, not as part of the
    /// text.
    private static func rule(body: CGFloat, color: NSColor) -> NSAttributedString {
        let style = NSMutableParagraphStyle()
        style.alignment = .center
        style.paragraphSpacingBefore = body * 1.1
        style.paragraphSpacing = body * 1.1
        return NSAttributedString(string: "* * *\n", attributes: [
            .font: NSFont.systemFont(ofSize: body * 0.9, weight: .regular),
            .foregroundColor: color.withAlphaComponent(0.6),
            .paragraphStyle: style,
            .kern: body * 0.2,
        ])
    }

    // MARK: - Images

    /// Inline artwork as a text attachment, so a picture sits in the flow and
    /// scrolls, selects and paginates with the prose around it.
    private static func image(src: String, alt: String, document: ReaderDocument,
                              columnWidth: CGFloat, body: CGFloat, color: NSColor) -> NSAttributedString {
        let style = NSMutableParagraphStyle()
        style.alignment = .center
        style.paragraphSpacingBefore = body * 1.1
        style.paragraphSpacing = body * 0.7

        guard let data = document.resources[src], let nsImage = NSImage(data: data) else {
            guard !alt.isEmpty else { return NSAttributedString() }
            return NSAttributedString(string: alt + "\n", attributes: [
                .font: italic(.systemFont(ofSize: body * 0.86, weight: .regular)),
                .foregroundColor: color,
                .paragraphStyle: style,
            ])
        }

        let attachment = NSTextAttachment()
        attachment.image = nsImage
        // Never upscale past the artwork's own pixel size, and cap the height
        // so a full-page plate doesn't push everything after it off-screen.
        let natural = nsImage.size
        let maxWidth = min(columnWidth, natural.width > 0 ? natural.width : columnWidth)
        let maxHeight = columnWidth * 1.25
        var w = maxWidth
        var h = natural.width > 0 ? natural.height * (w / natural.width) : maxWidth
        if h > maxHeight {
            h = maxHeight
            w = natural.height > 0 ? natural.width * (h / natural.height) : maxWidth
        }
        attachment.bounds = CGRect(x: 0, y: 0, width: w.rounded(), height: h.rounded())

        let out = NSMutableAttributedString(attributedString: NSAttributedString(attachment: attachment))
        out.append(NSAttributedString(string: "\n"))
        out.addAttributes([.paragraphStyle: style], range: NSRange(location: 0, length: out.length))
        return out
    }

    // MARK: - Helpers

    private static func italic(_ font: NSFont) -> NSFont {
        let descriptor = font.fontDescriptor.withSymbolicTraits(.italic)
        return NSFont(descriptor: descriptor, size: font.pointSize) ?? font
    }
}

extension NSColor {
    /// Parses the `#rrggbb` strings the reader themes are written in.
    convenience init(hex: String) {
        var s = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("#") { s.removeFirst() }
        var value: UInt64 = 0
        Scanner(string: s).scanHexInt64(&value)
        let r = CGFloat((value >> 16) & 0xff) / 255
        let g = CGFloat((value >> 8) & 0xff) / 255
        let b = CGFloat(value & 0xff) / 255
        self.init(srgbRed: r, green: g, blue: b, alpha: 1)
    }
}
