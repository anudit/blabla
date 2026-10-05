//
//  EPUBLoader.swift
//  tts-metal
//
//  EPUB parsing: container.xml → OPF → spine order, nav-doc or NCX table of
//  contents, chapter conversion via HTMLToBlocks. Ported from blabla's loadEPUB.
//

import Foundation

enum EPUBLoader {

    /// One table-of-contents entry, with the nesting depth it had in the
    /// nav document / NCX — publisher TOCs are trees (Part → Chapter →
    /// Section) and flattening them to one level loses the whole shape of
    /// the book in the sidebar.
    private struct TOCEntry {
        let file: String        // spine-relative file name, no fragment
        let fragment: String    // element id the entry points at ("" if none)
        let title: String
        let level: Int          // 1-based nesting depth
    }

    static func load(zip: ZipArchive, fileName: String, sourceID: String,
                     previewURL: String? = nil) throws -> ReaderDocument {
        // 1. container.xml → OPF path
        guard let containerXML = zip.read("META-INF/container.xml"),
              let container = String(data: containerXML, encoding: .utf8),
              let opfPath = firstAttribute("full-path", inTag: "rootfile", xml: container) else {
            throw LoaderError.failed("Invalid EPUB: missing container.xml.")
        }
        guard let opfData = zip.read(opfPath), let opf = String(data: opfData, encoding: .utf8) else {
            throw LoaderError.failed("Invalid EPUB: missing OPF.")
        }
        let opfDir = (opfPath as NSString).deletingLastPathComponent

        // 2. Manifest id → href, plus the image items and the nav document.
        var manifest: [String: String] = [:]
        var imageHrefs: [String] = []
        var cssHrefs: [String] = []
        var navHref: String?
        for item in tagBodies(named: "item", in: opf) {
            guard let id = attributeValue("id", in: item), let href = attributeValue("href", in: item) else { continue }
            manifest[id] = href
            let media = attributeValue("media-type", in: item)?.lowercased() ?? ""
            if media.hasPrefix("image/") { imageHrefs.append(href) }
            if media == "text/css" { cssHrefs.append(href) }
            if (attributeValue("properties", in: item) ?? "").contains("nav") { navHref = href }
        }

        // 3. Spine order
        guard let spineBody = tagBody(named: "spine", in: opf) else {
            throw LoaderError.failed("Invalid EPUB: missing spine.")
        }
        let spineIds = tagBodies(named: "itemref", in: "<spine>\(spineBody)</spine>")
                        .compactMap { attributeValue("idref", in: $0) }
        let hrefs = spineIds.compactMap { manifest[$0] }

        // 4. TOC: EPUB3 nav document preferred (declared via
        //    properties="nav" — it is usually *not* in the spine, so looking
        //    for it there missed it), then the EPUB2 NCX.
        var toc: [TOCEntry] = []
        if let nav = navHref {
            toc = parseNavDoc(zip: zip, href: resolve(nav, base: opfDir))
        }
        if toc.isEmpty, let ncxId = spineIdsNCX(opf: opf, manifest: manifest)
            ?? zip.entries.keys.first(where: { $0.lowercased().hasSuffix(".ncx") }) {
            let path = ncxId.hasSuffix(".ncx") && zip.entries[ncxId] != nil ? ncxId : resolve(ncxId, base: opfDir)
            toc = parseNCX(zip: zip, href: path)
        }

        // 5. Embedded images, decoded once up front and handed to the
        //    document so `.image` blocks can render inline.
        var resources: [String: Data] = [:]
        for href in imageHrefs {
            let full = resolve(href, base: opfDir)
            if let d = zip.read(full) ?? zip.read((full as NSString).lastPathComponent) {
                resources[full] = d
            }
        }

        // 5b. Stylesheets, read once for the whole book: alignment, italics
        //     and super/subscript set by class live here, not in the markup.
        //     Merged into one sheet in manifest order — books link the same
        //     few sheets from every chapter.
        var styles = CSSStyleSheet()
        for href in cssHrefs {
            let full = resolve(href, base: opfDir)
            if let d = zip.read(full) ?? zip.read((full as NSString).lastPathComponent),
               let css = decode(d) {
                styles.add(css: css)
            }
        }

        // 6. Convert spine documents to blocks. Each chapter's zip-read +
        // HTMLToBlocks parse is independent of every other chapter — only
        // the final concatenation needs original spine order — so this runs
        // across every core instead of one chapter at a time.
        struct ChapterResult {
            var blocks: [DocBlock] = []
            var outline: [OutlineEntry] = []
        }
        var chapterResults = [ChapterResult](repeating: ChapterResult(), count: hrefs.count)
        chapterResults.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: hrefs.count) { i in
                let href = hrefs[i]
                let full = resolve(href, base: opfDir)
                guard let data = zip.read(full) ?? zip.read((full as NSString).lastPathComponent),
                      let html = decode(data) else { return }
                let baseDir = (full as NSString).deletingLastPathComponent
                let result = HTMLToBlocks(baseDir: baseDir, styles: styles).parse(html)

                let chapterName = (full as NSString).lastPathComponent
                let entries = toc.filter { $0.file == chapterName }
                let merged = merge(chapter: result, tocEntries: entries)
                buf[i] = ChapterResult(blocks: merged.blocks, outline: merged.outline)
            }
        }

        var blocks: [DocBlock] = []
        var outline: [OutlineEntry] = []
        for chapter in chapterResults {
            let offset = blocks.count
            blocks.append(contentsOf: chapter.blocks)
            for entry in chapter.outline {
                outline.append(OutlineEntry(level: entry.level, title: entry.title,
                                            blockIndex: offset + entry.blockIndex))
            }
        }

        guard !blocks.isEmpty else { throw LoaderError.failed("EPUB contained no readable text.") }

        // Any image the manifest didn't declare (or declared under a path
        // that doesn't match how the chapter referenced it) is still in the
        // archive — look it up directly rather than dropping the picture.
        for block in blocks {
            if case .image(_, let src) = block.content, !src.isEmpty, resources[src] == nil,
               let d = zip.read(src) ?? zip.read((src as NSString).lastPathComponent) {
                resources[src] = d
            }
        }

        let metaTitle = tagBody(named: "dc:title", in: opf).map(stripTags).flatMap { $0.isEmpty ? nil : $0 }
        let docTitle = metaTitle
            ?? fileName.replacingOccurrences(of: "\\.[^.]+$", with: "", options: .regularExpression)

        return ReaderDocument(title: docTitle, fileType: .epub, sourceID: sourceID, fileName: docTitle,
                              previewURL: previewURL, frontmatter: nil, blocks: blocks, outline: outline,
                              resources: resources)
    }

    /// The book's cover image and its creator, read straight from the OPF
    /// without converting any chapter — cheap enough for a library grid.
    /// Tries, in order: the EPUB3 `properties="cover-image"` item, the EPUB2
    /// `<meta name="cover" content="…">` pointer, then any image item whose
    /// id or file name says "cover".
    static func coverInfo(zip: ZipArchive) -> (image: Data?, author: String?) {
        guard let containerXML = zip.read("META-INF/container.xml"),
              let container = String(data: containerXML, encoding: .utf8),
              let opfPath = firstAttribute("full-path", inTag: "rootfile", xml: container),
              let opfData = zip.read(opfPath), let opf = decode(opfData) else { return (nil, nil) }
        let opfDir = (opfPath as NSString).deletingLastPathComponent
        let author = tagBody(named: "dc:creator", in: opf).map(stripTags).flatMap { $0.isEmpty ? nil : $0 }

        var byId: [String: String] = [:]
        var images: [(id: String, href: String, props: String)] = []
        for item in tagBodies(named: "item", in: opf) {
            guard let id = attributeValue("id", in: item), let href = attributeValue("href", in: item) else { continue }
            byId[id] = href
            if (attributeValue("media-type", in: item) ?? "").lowercased().hasPrefix("image/") {
                images.append((id, href, attributeValue("properties", in: item) ?? ""))
            }
        }
        var candidates: [String] = []
        if let c = images.first(where: { $0.props.contains("cover-image") }) { candidates.append(c.href) }
        for meta in tagBodies(named: "meta", in: opf)
        where attributeValue("name", in: meta)?.lowercased() == "cover" {
            if let id = attributeValue("content", in: meta), let href = byId[id] { candidates.append(href) }
        }
        if let c = images.first(where: { $0.id.lowercased().contains("cover") || $0.href.lowercased().contains("cover") }) {
            candidates.append(c.href)
        }
        for href in candidates {
            let full = resolve(href, base: opfDir)
            if let d = zip.read(full) ?? zip.read((full as NSString).lastPathComponent), !d.isEmpty {
                return (d, author)
            }
        }
        return (nil, author)
    }

    /// Weaves the TOC entries that point into this chapter together with the
    /// headings the chapter's own markup produced.
    ///
    /// A TOC title almost always duplicates a heading that is already the
    /// first line of the chapter, so blindly prepending it (as this used to)
    /// printed every chapter title twice. Instead: if the block the entry
    /// points at already *is* that heading, adopt it — keeping the TOC's
    /// nesting level, which the markup alone can't know — and only synthesise
    /// a heading block when the chapter genuinely has none.
    private static func merge(chapter: HTMLToBlocks.Result, tocEntries: [TOCEntry]) -> (blocks: [DocBlock], outline: [OutlineEntry]) {
        var blocks = chapter.blocks

        struct Pending { let index: Int; let title: String; let level: Int }
        var inserts: [Pending] = []
        var adopted: [Int: (title: String, level: Int)] = [:]

        for entry in tocEntries {
            let target = entry.fragment.isEmpty ? 0 : (chapter.anchors[entry.fragment] ?? 0)
            // The anchor sits on the heading element itself, but a leading
            // <a id> wrapper can push it one block early — accept either.
            let match = (target..<min(target + 2, blocks.count)).first { idx in
                if case .heading(_, let t) = blocks[idx].content { return matches(t, entry.title) }
                return false
            }
            if let idx = match {
                if case .heading(_, let t) = blocks[idx].content {
                    blocks[idx].content = .heading(level: entry.level, text: t)
                }
                adopted[idx] = (entry.title, entry.level)
            } else {
                inserts.append(Pending(index: min(target, blocks.count), title: entry.title, level: entry.level))
            }
        }

        // Rank the chapter's own heading levels below the TOC entry that
        // owns the chapter. Stylesheet classes number headings on the
        // publisher's own scale ("Heading-4" for what is really the first
        // subsection), and distinct levels are compressed to consecutive
        // ranks so the sidebar doesn't show a chapter's sub-sections two or
        // three indents deeper than their parent.
        //
        // The base is taken from the nearest *preceding* TOC heading rather
        // than the file-wide maximum: publishers regularly pack several
        // chapters into one spine document, and a file-wide base pushed every
        // section of the first chapter as deep as the last chapter's nesting,
        // so a chapter's own sub-heads showed up two or three indents below
        // where they belong.
        let ownEntries = chapter.outline.filter { adopted[$0.blockIndex] == nil }
        let ranks = Array(Set(ownEntries.map(\.level))).sorted()
        let tocLevelAt: [Int: Int] = adopted.mapValues(\.level)
            .merging(Dictionary(inserts.map { ($0.index, $0.level) }, uniquingKeysWith: min),
                     uniquingKeysWith: min)
        func base(before blockIndex: Int) -> Int {
            (tocLevelAt.filter { $0.key <= blockIndex }.map(\.value).max() ?? 0) + 1
        }
        func depth(_ level: Int, at blockIndex: Int) -> Int {
            base(before: blockIndex) + (ranks.firstIndex(of: level) ?? 0)
        }

        var outline = ownEntries.map {
            OutlineEntry(level: depth($0.level, at: $0.blockIndex), title: $0.title, blockIndex: $0.blockIndex)
        }
        for (idx, a) in adopted {
            outline.append(OutlineEntry(level: a.level, title: a.title, blockIndex: idx))
        }
        for entry in outline where adopted[entry.blockIndex] == nil {
            if case .heading(_, let t) = blocks[entry.blockIndex].content {
                blocks[entry.blockIndex].content = .heading(level: entry.level, text: t)
            }
        }

        guard !inserts.isEmpty else {
            return (blocks, outline.sorted { $0.blockIndex < $1.blockIndex })
        }

        // Apply the insertions in one pass, remapping every index that moves.
        inserts.sort { $0.index < $1.index }
        var merged: [DocBlock] = []
        merged.reserveCapacity(blocks.count + inserts.count)
        var shiftFor = [Int](repeating: 0, count: blocks.count + 1)
        var pending = 0
        var inserted: [(entry: Pending, index: Int)] = []
        for old in 0...blocks.count {
            for p in inserts where p.index == old {
                inserted.append((p, merged.count))
                merged.append(DocBlock(content: .heading(level: p.level, text: p.title)))
                pending += 1
            }
            shiftFor[old] = pending
            if old < blocks.count { merged.append(blocks[old]) }
        }
        var remapped = outline.map {
            OutlineEntry(level: $0.level, title: $0.title, blockIndex: $0.blockIndex + shiftFor[$0.blockIndex])
        }
        for i in inserted {
            remapped.append(OutlineEntry(level: i.entry.level, title: i.entry.title, blockIndex: i.index))
        }
        return (merged, remapped.sorted { $0.blockIndex < $1.blockIndex })
    }

    /// Loose title comparison — TOC text and heading text differ by trailing
    /// punctuation, curly quotes and collapsed whitespace often enough that
    /// exact equality would miss most matches and re-print every title.
    private static func matches(_ a: String, _ b: String) -> Bool {
        func norm(_ s: String) -> String {
            s.folding(options: [.diacriticInsensitive, .caseInsensitive, .widthInsensitive], locale: nil)
                .replacingOccurrences(of: "[^a-z0-9]+", with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespaces)
        }
        let (x, y) = (norm(a), norm(b))
        guard !x.isEmpty, !y.isEmpty else { return false }
        return x == y || x.hasPrefix(y) || y.hasPrefix(x)
    }

    private static func spineIdsNCX(opf: String, manifest: [String: String]) -> String? {
        guard let spine = RegexCache.regex(#"<spine\b[^>]*>"#, options: .caseInsensitive),
              let m = spine.firstMatch(in: opf, range: NSRange(opf.startIndex..., in: opf)),
              let r = Range(m.range, in: opf),
              let tocId = attributeValue("toc", in: String(opf[r])) else { return nil }
        return manifest[tocId]
    }

    // MARK: - Nav doc (EPUB3)

    /// Walks the nav document's nested `<ol>`s so each entry keeps its depth.
    private static func parseNavDoc(zip: ZipArchive, href: String) -> [TOCEntry] {
        guard let data = zip.read(href), let html = decode(data) else { return [] }
        // Restrict to the toc nav when the document declares several
        // (landmarks / page-list navs would otherwise pollute the outline).
        let scope: String = {
            guard let re = RegexCache.regex(#"<nav\b[^>]*epub:type\s*=\s*["']toc["'][^>]*>"#,
                                            options: .caseInsensitive),
                  let m = re.firstMatch(in: html, range: NSRange(html.startIndex..., in: html)) else { return html }
            let ns = html as NSString
            let start = m.range.upperBound
            let closeRange = RegexCache.regex(#"</nav\s*>"#, options: .caseInsensitive)?
                .firstMatch(in: html, range: NSRange(location: start, length: ns.length - start))?.range
            let end = closeRange?.location ?? ns.length
            return ns.substring(with: NSRange(location: start, length: end - start))
        }()

        guard let re = RegexCache.regex(#"<(/?)(ol|a)\b([^>]*)>"#, options: .caseInsensitive) else { return [] }
        let ns = scope as NSString
        var out: [TOCEntry] = []
        var depth = 0
        var openAnchor: (href: String, textStart: Int)?
        re.enumerateMatches(in: scope, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m else { return }
            let isClose = ns.substring(with: m.range(at: 1)) == "/"
            let tag = ns.substring(with: m.range(at: 2)).lowercased()
            let attrs = ns.substring(with: m.range(at: 3))
            switch (tag, isClose) {
            case ("ol", false): depth += 1
            case ("ol", true):  depth = max(0, depth - 1)
            case ("a", false):
                guard let h = attributeValue("href", in: attrs) else { return }
                openAnchor = (h, m.range.upperBound)
            case ("a", true):
                guard let a = openAnchor else { return }
                openAnchor = nil
                let title = stripTags(ns.substring(with: NSRange(location: a.textStart,
                                                                length: m.range.location - a.textStart)))
                guard !title.isEmpty else { return }
                out.append(entry(href: a.href, title: title, level: max(1, depth)))
            default: break
            }
        }
        return out
    }

    // MARK: - NCX (EPUB2)

    /// navPoints nest to express the TOC tree; the previous flat
    /// `tagBodies(named:"navPoint")` scan both lost that depth and truncated
    /// every parent at its first child's `</navPoint>`, so nested books lost
    /// most of their contents. This walks the tags in order instead.
    private static func parseNCX(zip: ZipArchive, href: String) -> [TOCEntry] {
        guard let data = zip.read(href), let xml = decode(data) else { return [] }
        guard let re = RegexCache.regex(#"<(/?)(navPoint|text|content)\b([^>]*?)(/?)>"#,
                                        options: .caseInsensitive) else { return [] }
        let ns = xml as NSString
        var out: [TOCEntry] = []
        var depth = 0
        var labels: [Int: String] = [:]     // depth → most recent <text> label
        var textStart: Int?
        re.enumerateMatches(in: xml, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m = m else { return }
            let isClose = ns.substring(with: m.range(at: 1)) == "/"
            let tag = ns.substring(with: m.range(at: 2)).lowercased()
            let attrs = ns.substring(with: m.range(at: 3))
            switch (tag, isClose) {
            case ("navpoint", false): depth += 1; labels[depth] = nil
            case ("navpoint", true):  labels[depth] = nil; depth = max(0, depth - 1)
            case ("text", false):     textStart = m.range.upperBound
            case ("text", true):
                if let s = textStart {
                    let t = stripTags(ns.substring(with: NSRange(location: s, length: m.range.location - s)))
                    if labels[depth] == nil, !t.isEmpty { labels[depth] = t }
                }
                textStart = nil
            case ("content", false):
                guard depth > 0, let label = labels[depth], let src = attributeValue("src", in: attrs) else { return }
                out.append(entry(href: src, title: label, level: depth))
            default: break
            }
        }
        return out
    }

    private static func entry(href: String, title: String, level: Int) -> TOCEntry {
        let parts = href.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false)
        let file = (String(parts[0]) as NSString).lastPathComponent
        let fragment = parts.count > 1 ? String(parts[1]) : ""
        return TOCEntry(file: file, fragment: fragment, title: title, level: level)
    }

    // MARK: - Helpers

    static func resolve(_ href: String, base: String) -> String {
        let cleanPath = href.split(separator: "#").first.map(String.init) ?? href
        guard !base.isEmpty else { return cleanPath.deletingLeadingSlash() }
        var components = base.split(separator: "/").map(String.init)
        for part in cleanPath.split(separator: "/").map(String.init) {
            if part == "." { continue }
            if part == ".." { _ = components.popLast() }
            else { components.append(part) }
        }
        return components.joined(separator: "/")
    }

    static func decode(_ data: Data) -> String? {
        String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .isoLatin1)
            ?? String(data: data, encoding: .utf16)
    }
}

extension String {
    fileprivate func deletingLeadingSlash() -> String { hasPrefix("/") ? String(dropFirst()) : self }
}

// MARK: - Tiny XML helpers shared across loaders

func firstAttribute(_ attr: String, inTag tag: String, xml: String) -> String? {
    guard let re = RegexCache.regex(#"<\#(tag)\b[^>]*>"#, options: .caseInsensitive),
          let m = re.firstMatch(in: xml, range: NSRange(xml.startIndex..., in: xml)),
          let r = Range(m.range, in: xml) else { return nil }
    return attributeValue(attr, in: String(xml[r]))
}

func tagBodies(named tag: String, in xml: String) -> [String] {
    guard let open = RegexCache.regex(#"<\#(tag)\b([^>]*?)(/>|>)"#, options: .caseInsensitive),
          let close = RegexCache.regex(#"</\#(tag)\s*>"#, options: .caseInsensitive) else { return [] }
    var out: [String] = []
    let ns = xml as NSString
    var searchRange = NSRange(location: 0, length: ns.length)
    while searchRange.location < ns.length, let m = open.firstMatch(in: xml, options: [], range: searchRange) {
        let attrs = ns.substring(with: m.range(at: 1))
        let closer = ns.substring(with: m.range(at: 2))
        if closer.contains("/") {          // self-closing <tag … />
            out.append(attrs)
        } else {                            // paired <tag …> body </tag>
            let start = m.range.upperBound
            let rest = NSRange(location: start, length: ns.length - start)
            let closeRange = close.firstMatch(in: xml, options: [], range: rest)?.range
                ?? NSRange(location: start, length: 0)
            out.append(ns.substring(with: NSRange(location: start, length: closeRange.location - start)))
            searchRange.location = closeRange.location + closeRange.length
            searchRange.length = ns.length - searchRange.location
            continue
        }
        searchRange.location = m.range.upperBound
        searchRange.length = ns.length - searchRange.location
    }
    return out
}

func tagBody(named tag: String, in xml: String) -> String? {
    tagBodies(named: tag, in: xml).first
}

func stripTags(_ s: String) -> String {
    let noTags = RegexCache.replace(s, pattern: #"<[^>]+>"#, with: "")
    return RegexCache.replace(HTMLToBlocks.decodeEntities(noTags), pattern: "\\s+", with: " ")
                 .trimmingCharacters(in: .whitespacesAndNewlines)
}

func attributeValue(_ attr: String, in tagText: String) -> String? {
    for pattern in [#"\#(attr)\s*=\s*"([^"]*)""#, #"\#(attr)\s*=\s*'([^']*)'"#] {
        if let re = RegexCache.regex(pattern, options: .caseInsensitive),
           let m = re.firstMatch(in: tagText, range: NSRange(tagText.startIndex..., in: tagText)),
           m.numberOfRanges > 1, let r = Range(m.range(at: 1), in: tagText) {
            return HTMLToBlocks.decodeEntities(String(tagText[r]))
        }
    }
    return nil
}
