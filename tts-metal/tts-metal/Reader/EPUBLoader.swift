//
//  EPUBLoader.swift
//  tts-metal
//
//  EPUB parsing: container.xml → OPF → spine order, nav-doc or NCX table of
//  contents, chapter conversion via HTMLToBlocks. Ported from blabla's loadEPUB.
//

import Foundation

enum EPUBLoader {

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

        // 2. Manifest id → href
        var manifest: [String: String] = [:]
        for item in tagBodies(named: "item", in: opf) {
            if let id = attributeValue("id", in: item), let href = attributeValue("href", in: item) {
                manifest[id] = href
            }
        }

        // 3. Spine order
        guard let spineBody = tagBody(named: "spine", in: opf) else {
            throw LoaderError.failed("Invalid EPUB: missing spine.")
        }
        let spineIds = tagBodies(named: "itemref", in: "<spine>\(spineBody)</spine>")
                        .compactMap { attributeValue("idref", in: $0) }
        let hrefs = spineIds.compactMap { manifest[$0] }

        // 4. TOC titles: EPUB3 nav document preferred, then NCX
        var tocTitles: [(href: String, title: String)] = []
        let navEntry = spineIds.first { id in
            guard let href = manifest[id] else { return false }
            if href.lowercased().contains("nav") { return true }
            if let data = zip.read(resolve(href, base: opfDir)),
               let s = decode(data), s.contains("epub:type=\"toc\"") || s.contains("nav epub:type") { return true }
            return false
        }
        if let navId = navEntry, let navHref = manifest[navId] {
            tocTitles = parseNavDoc(zip: zip, href: resolve(navHref, base: opfDir))
        } else if let ncxPath = zip.entries.keys.first(where: { $0.lowercased().hasSuffix(".ncx") }) {
            tocTitles = parseNCX(zip: zip, href: ncxPath)
        }

        // 5. Convert spine documents to blocks. Each chapter's zip-read +
        // HTMLToBlocks parse is independent of every other chapter — only
        // the final concatenation needs original spine order — so this runs
        // across every core instead of one chapter at a time.
        struct ChapterResult {
            var blocks: [DocBlock] = []
            var outline: [OutlineEntry] = []
            var chapterTitle: String?
        }
        var chapterResults = [ChapterResult](repeating: ChapterResult(), count: hrefs.count)
        chapterResults.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: hrefs.count) { i in
                let href = hrefs[i]
                let full = resolve(href, base: opfDir)
                guard let data = zip.read(full) ?? zip.read((full as NSString).lastPathComponent),
                      let html = decode(data) else { return }
                let result = HTMLToBlocks().parse(html)

                // Chapter title from TOC by matching filename
                let chapterName = (full as NSString).lastPathComponent.split(separator: "#").first.map(String.init) ?? full
                let chapterTitle = tocTitles.first { ($0.href as NSString).lastPathComponent == chapterName }?.title

                buf[i] = ChapterResult(blocks: result.blocks, outline: result.outline, chapterTitle: chapterTitle)
            }
        }

        var blocks: [DocBlock] = []
        var outline: [OutlineEntry] = []
        var docTitle = fileName.replacingOccurrences(of: "\\.[^.]+$", with: "", options: .regularExpression)

        for chapter in chapterResults {
            let offset = blocks.count
            blocks.append(contentsOf: chapter.blocks)

            if let ct = chapter.chapterTitle {
                blocks.insert(DocBlock(content: .heading(level: 1, text: ct)), at: offset)
                outline.append(OutlineEntry(level: 1, title: ct, blockIndex: offset))
                if docTitle == fileName.replacingOccurrences(of: "\\.[^.]+$", with: "", options: .regularExpression), outline.count == 1 {
                    docTitle = ct
                }
            }
            for entry in chapter.outline {
                outline.append(OutlineEntry(level: entry.level + 1, title: entry.title,
                                            blockIndex: offset + entry.blockIndex))
            }
        }

        guard !blocks.isEmpty else { throw LoaderError.failed("EPUB contained no readable text.") }

        return ReaderDocument(title: docTitle, fileType: .epub, sourceID: sourceID, fileName: docTitle,
                              previewURL: previewURL, frontmatter: nil, blocks: blocks, outline: outline)
    }

    // MARK: - Nav doc (EPUB3)

    private static func parseNavDoc(zip: ZipArchive, href: String) -> [(String, String)] {
        guard let data = zip.read(href), let html = decode(data) else { return [] }
        var out: [(String, String)] = []
        for anchor in tagBodies(named: "a", in: html) {
            guard let h = attributeValue("href", in: anchor) else { continue }
            let text = stripTags(anchor)
            if !text.isEmpty { out.append((h, text)) }
        }
        return out
    }

    // MARK: - NCX (EPUB2)

    private static func parseNCX(zip: ZipArchive, href: String) -> [(String, String)] {
        guard let data = zip.read(href), let xml = decode(data) else { return [] }
        var out: [(String, String)] = []
        for point in tagBodies(named: "navPoint", in: xml) {
            let label = tagBody(named: "text", in: point).map(stripTags) ?? ""
            let src = tagBody(named: "content", in: xml) != nil
                ? attributeValue("src", in: "<content \(tagBody(named: "content", in: point) ?? "")/>") ?? ""
                : ""
            if !label.isEmpty, !src.isEmpty { out.append((src, label)) }
        }
        return out
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
