//
//  Loaders.swift
//  tts-metal
//
//  Document loading front-end. Dispatches by file type and converts any supported
//  source (TXT/MD, PDF, EPUB, MOBI/AZW, DOCX) into a ReaderDocument.
//  Ported from blabla's loaders.tsx.
//

import Foundation
import UniformTypeIdentifiers

enum LoaderError: LocalizedError {
    case unsupported(String)
    case failed(String)
    var errorDescription: String? {
        switch self {
        case .unsupported(let m): return m
        case .failed(let m): return m
        }
    }
}

enum DocLoader {

    static let supportedExtensions = ["pdf", "epub", "mobi", "azw", "azw3", "docx", "md", "markdown", "txt"]

    static func detectMarkdown(_ text: String) -> Bool {
        guard let re = RegexCache.regex("^\\#{1,6} \\S", options: .anchorsMatchLines) else { return false }
        return re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    /// Main entry: load a local file URL.
    static func load(fileURL url: URL) throws -> ReaderDocument {
        let ext = url.pathExtension.lowercased()
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
        let name = url.lastPathComponent
        let sourceID = "\(name):\(size)"

        var doc: ReaderDocument
        switch ext {
        case "pdf":
            doc = try PDFLoader.load(url: url, sourceID: sourceID)
        case "epub":
            // Read errors (including lost file access on resume) must reach the
            // UI as read errors, not be misreported as a corrupt archive.
            guard let zip = try ZipArchive.open(url) else { throw LoaderError.failed("Not a valid EPUB (bad ZIP).") }
            doc = try EPUBLoader.load(zip: zip, fileName: name, sourceID: sourceID)
        case "mobi", "azw", "azw3":
            doc = try MOBILoader.load(url: url, fileName: name, sourceID: sourceID)
        case "docx":
            doc = try DOCXLoader.load(url: url, fileName: name, sourceID: sourceID)
        case "md", "markdown":
            let text = try String(contentsOf: url, encoding: .utf8)
            doc = MarkdownLoader.load(text: text, title: name, sourceID: sourceID, fileName: name)
        case "txt":
            let utf8Text = try? String(contentsOf: url, encoding: .utf8)
            let latinText = (utf8Text == nil) ? try? String(contentsOf: url, encoding: .isoLatin1) : nil
            let text = utf8Text ?? latinText ?? ""
            doc = TextLoader.load(text: text, title: name, sourceID: sourceID, fileName: name,
                                  markdown: detectMarkdown(text))
        default:
            throw LoaderError.unsupported("Unsupported file type: .\(ext)")
        }
        doc.sourceFilePath = url.path
        return doc
    }

    /// Load raw pasted / typed text.
    static func loadText(_ text: String, title: String = "Pasted text") -> ReaderDocument {
        let md = detectMarkdown(text)
        return md ? MarkdownLoader.load(text: text, title: title, sourceID: "paste:\(title)", fileName: title)
                  : TextLoader.load(text: text, title: title, sourceID: "paste:\(title)", fileName: title, markdown: false)
    }

    /// Fetch a URL and convert the page into blocks. Google Docs is special-cased to its Markdown export endpoint (like blabla).
    static func load(urlString raw: String) async throws -> ReaderDocument {
        var target = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if !target.hasPrefix("http://") && !target.hasPrefix("https://") {
            target = "https://" + target
        }
        guard let url = URL(string: target) else {
            throw LoaderError.failed("Invalid URL.")
        }
        // Google Docs export as markdown
        var fetchURL = url
        if url.host?.hasSuffix("docs.google.com") == true, url.path.contains("/document/") {
            fetchURL = url.appendingPathComponent("export", isDirectory: false)
            fetchURL.append(queryItems: [URLQueryItem(name: "format", value: "md")])
        }

        let (data, resp) = try await URLSession.shared.data(from: fetchURL)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw LoaderError.failed("Document inaccessible — check sharing permissions.")
        }

        let mime = (http.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
        if mime.contains("application/pdf") || target.lowercased().hasSuffix(".pdf") {
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".pdf")
            try data.write(to: tmp)
            return try PDFLoader.load(url: tmp, sourceID: "url:\(target)")
        }
        if mime.contains("application/epub") {
            guard let zip = ZipArchive(data: data) else { throw LoaderError.failed("Bad EPUB.") }
            return try EPUBLoader.load(zip: zip, fileName: url.lastPathComponent, sourceID: "url:\(target)",
                                       previewURL: target)
        }

        let body = String(decoding: data, as: UTF8.self)
        let htmlTitle = extractTitle(body) ?? url.host ?? "Web page"

        if detectMarkdown(body), !mime.contains("html") {
            return MarkdownLoader.load(text: body, title: htmlTitle, sourceID: "url:\(target)",
                                       fileName: htmlTitle, previewURL: target)
        }

        // Extract main content: strip scripts/styles/nav/footer, then convert.
        let cleaned = stripChrome(body)
        let parser = HTMLToBlocks(baseURL: url)
        let result = parser.parse(cleaned)
        let docTitle = result.title ?? htmlTitle

        var blocks: [DocBlock] = []
        if !result.blocks.isEmpty { blocks = result.blocks }
        else { blocks = fallbackTextBlocks(from: cleaned) }

        return ReaderDocument(title: docTitle, fileType: .url, sourceID: "url:\(target)",
                              fileName: docTitle, previewURL: target,
                              frontmatter: nil, blocks: blocks,
                              outline: result.outline.remapBlocks(blocks.count))
    }

    private static func extractTitle(_ html: String) -> String? {
        guard let re = RegexCache.regex(#"<title[^>]*>(.*?)</title>"#, options: [.caseInsensitive, .dotMatchesLineSeparators]),
              let m = re.firstMatch(in: html, range: NSRange(html.startIndex..., in: html)),
              m.numberOfRanges > 1, let r = Range(m.range(at: 1), in: html) else { return nil }
        let t = HTMLToBlocks.decodeEntities(String(html[r])).trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    private static func stripChrome(_ html: String) -> String {
        var out = html
        for pattern in [#"<script\b[\s\S]*?</script>"#, #"<style\b[\s\S]*?</style>"#,
                        #"<nav\b[\s\S]*?</nav>"#, #"<footer\b[\s\S]*?</footer>"#,
                        #"<!--[\s\S]*?-->"#, #"<!(doctype|DOCTYPE)[^>]*>"#] {
            out = out.replacingOccurrences(of: pattern, with: " ", options: .regularExpression)
        }
        // Keep <body> content when present.
        if let r = out.range(of: #"<body[^>]*>([\s\S]*)</body>"#, options: .regularExpression) {
            let inner = out[r]
            if let start = inner.firstIndex(of: ">") , let end = inner.lastIndex(of: "<") {
                out = String(inner[inner.index(after: start)..<end])
            }
        }
        return out
    }

    private static func fallbackTextBlocks(from html: String) -> [DocBlock] {
        let stripped = html.replacingOccurrences(of: #"<[^>]+>"#, with: "\n\n", options: .regularExpression)
        let decoded = HTMLToBlocks.decodeEntities(stripped)
        return decoded.components(separatedBy: "\n\n")
            .map { $0.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count > 2 }
            .map { DocBlock(content: .paragraph($0)) }
    }
}

private extension Array where Element == OutlineEntry {
    func remapBlocks(_ blockCount: Int) -> [OutlineEntry] { self }
}

// MARK: - Plain text loader

enum TextLoader {
    static func load(text: String, title: String, sourceID: String, fileName: String,
                     markdown: Bool, previewURL: String? = nil) -> ReaderDocument {
        if markdown {
            return MarkdownLoader.load(text: text, title: title, sourceID: sourceID,
                                       fileName: fileName, previewURL: previewURL)
        }
        var blocks: [DocBlock] = []
        for paraRaw in text.components(separatedBy: "\n{2}") {
            let para = paraRaw.trimmingCharacters(in: .whitespacesAndNewlines)
            if para.isEmpty { continue }
            blocks.append(DocBlock(content: .paragraph(para)))
        }
        if blocks.isEmpty { blocks = [DocBlock(content: .paragraph(text))] }
        return ReaderDocument(title: title, fileType: .text, sourceID: sourceID, fileName: fileName,
                              previewURL: previewURL, frontmatter: nil, blocks: blocks, outline: [])
    }
}

// MARK: - Markdown loader (port of loaders.tsx loadMarkdown)

enum MarkdownLoader {

    struct FrontMatter {
        var title: String?
        var description: String?
        var image: String?
    }

    static func load(text: String, title: String, sourceID: String, fileName: String,
                     previewURL: String? = nil) -> ReaderDocument {
        var body = text
        var fm: FrontMatter?

        // Strip a Jina-reader style marker.
        if let r = body.range(of: "Markdown Content:") {
            body = String(body[r.upperBound...])
        }

        // YAML frontmatter
        if body.hasPrefix("---") {
            let lines = body.split(separator: "\n", omittingEmptySubsequences: false)
            if lines.first.map(String.init) == "---" {
                var endIdx: Int?
                for (i, line) in lines.enumerated().dropFirst() where String(line).trimmingCharacters(in: .whitespaces) == "---" {
                    endIdx = i; break
                }
                if let end = endIdx {
                    var t: String?, d: String?, img: String?
                    for i in 1..<end {
                        let line = String(lines[i])
                        if let v = value(of: "title", in: line) { t = v }
                        else if let v = value(of: "description", in: line) { d = v }
                        else if let v = value(of: "image", in: line) { img = v }
                    }
                    fm = FrontMatter(title: t, description: d, image: img)
                    body = lines.dropFirst(end + 1).joined(separator: "\n")
                }
            }
        }

        var blocks: [DocBlock] = []
        var outline: [OutlineEntry] = []
        if let fm = fm {
            blocks.append(DocBlock(content: .frontmatter(title: fm.title, description: fm.description, image: fm.image)))
        }

        let lines = body.components(separatedBy: "\n")
        var i = 0
        var currentPara = ""
        var codeBuffer: [String] = []
        var codeFence: String?
        var tableBuffer: [String] = []

        func flushTable() {
            guard tableBuffer.count >= 2 else { tableBuffer.removeAll(); return }
            var rows: [[String]] = []
            for (idx, line) in tableBuffer.enumerated() {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix("|"), trimmed.hasSuffix("|") else { continue }
                let cells = trimmed.dropFirst().dropLast().components(separatedBy: "|")
                                  .map { $0.trimmingCharacters(in: .whitespaces) }
                if idx > 0 && cells.allSatisfy({ $0.range(of: #"^:?-+:?$"#, options: .regularExpression) != nil }) { continue }
                rows.append(cells)
            }
            if !rows.isEmpty { blocks.append(DocBlock(content: .table(rows: rows))) }
            tableBuffer.removeAll()
        }

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Fenced code
            if let fence = codeFence {
                if trimmed.hasPrefix(fence) {
                    blocks.append(DocBlock(content: .code(codeBuffer.joined(separator: "\n"))))
                    codeBuffer = []; codeFence = nil
                } else {
                    codeBuffer.append(line)
                }
                i += 1; continue
            }
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                flushTable(); closeParagraph(&blocks, current: currentPara); currentPara = ""
                codeFence = String(trimmed.prefix(3)); codeBuffer = []
                i += 1; continue
            }

            // Tables
            if trimmed.hasPrefix("|") {
                tableBuffer.append(trimmed)
                i += 1; continue
            } else if !tableBuffer.isEmpty {
                flushTable()
            }

            // Headings
            if let m = trimmed.range(of: "^\\#{1,6}\\s+(.*)$", options: .regularExpression) {
                closeParagraph(&blocks, current: currentPara); currentPara = ""
                let hashes = trimmed[m.lowerBound...].prefix(while: { $0 == "#" })
                let level = hashes.count
                let headingText = stripInlineMd(
                    trimmed[trimmed.index(m.lowerBound, offsetBy: hashes.count)...]
                        .trimmingCharacters(in: .whitespaces))
                if !headingText.isEmpty {
                    blocks.append(DocBlock(content: .heading(level: level, text: headingText)))
                    outline.append(OutlineEntry(level: level, title: headingText, blockIndex: blocks.count - 1))
                }
                i += 1; continue
            }

            // Horizontal rule
            if trimmed.range(of: #"^([-*_])\s*(?:\1\s*){2,}$"#, options: .regularExpression) != nil {
                closeParagraph(&blocks, current: currentPara); currentPara = ""
                blocks.append(DocBlock(content: .rule))
                i += 1; continue
            }

            // Images (block level)
            if trimmed.range(of: #"^!\[[^\]]*\]\([^)]+\)$"#, options: .regularExpression) != nil {
                closeParagraph(&blocks, current: currentPara); currentPara = ""
                let body = trimmed.dropFirst(2)
                let alt = String(body.prefix(while: { $0 != "]" }))
                // src is everything between the "](" and the closing ")",
                // minus any optional "title" the Markdown syntax allows.
                var src = String(body.drop(while: { $0 != "(" }).dropFirst().dropLast())
                if let space = src.firstIndex(of: " ") { src = String(src[..<space]) }
                blocks.append(DocBlock(content: .image(alt: alt, src: src)))
                i += 1; continue
            }

            // Blockquote
            if trimmed.hasPrefix(">") {
                closeParagraph(&blocks, current: currentPara); currentPara = ""
                let q = stripInlineMd(trimmed.dropFirst().trimmingCharacters(in: .whitespaces))
                if !q.isEmpty { blocks.append(DocBlock(content: .quote(q))) }
                i += 1; continue
            }

            // List item
            if trimmed.range(of: #"^([-*+]|\d+[.)])\s+"#, options: .regularExpression) != nil {
                closeParagraph(&blocks, current: currentPara); currentPara = ""
                let itemBody = trimmed.replacingOccurrences(of: #"^([-*+]|\d+[.)])\s+"#, with: "", options: .regularExpression)
                let t = stripInlineMd(itemBody)
                if !t.isEmpty { blocks.append(DocBlock(content: .listItem(t.hasPunctuationTerminal ? t : t + "."))) }
                i += 1; continue
            }

            // Blank → paragraph break
            if trimmed.isEmpty {
                closeParagraph(&blocks, current: currentPara); currentPara = ""
                i += 1; continue
            }

            currentPara += (currentPara.isEmpty ? "" : " ") + trimmed
            i += 1
        }
        if codeFence != nil, !codeBuffer.isEmpty {
            blocks.append(DocBlock(content: .code(codeBuffer.joined(separator: "\n"))))
        }
        flushTable()
        closeParagraph(&blocks, current: currentPara); currentPara = ""

        let displayTitle = fm?.title ?? title
        return ReaderDocument(title: displayTitle, fileType: .text, sourceID: sourceID, fileName: displayTitle,
                              previewURL: previewURL,
                              frontmatter: fm != nil ? .frontmatter(title: fm!.title, description: fm!.description, image: fm!.image) : nil,
                              blocks: blocks, outline: outline)
    }

    private static func closeParagraph(_ blocks: inout [DocBlock], current: String) {
        let t = stripInlineMd(current)
        if !t.isEmpty { blocks.append(DocBlock(content: .paragraph(t))) }
    }

    private static func value(of key: String, in line: String) -> String? {
        guard let re = RegexCache.regex(#"^\#(key):\s*"?(.*?)"?\s*$"#),
              let m = re.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
              m.numberOfRanges > 1, let r = Range(m.range(at: 1), in: line) else { return nil }
        let v = String(line[r])
        return v.isEmpty ? nil : v
    }

    /// Strip inline markdown for speech/display: bold, italic, strike, links (keep text).
    static func stripInlineMd(_ s: String) -> String {
        var out = s
        out = out.replacingOccurrences(of: #"!\[([^\]]*)\]\([^)]*\)"#, with: "", options: .regularExpression)
        out = out.replacingOccurrences(of: #"\[([^\]]+)\]\([^)]*\)"#, with: "$1", options: .regularExpression)
        out = out.replacingOccurrences(of: #"\*\*([^*]+)\*\*"#, with: "$1", options: .regularExpression)
        out = out.replacingOccurrences(of: #"__([^_]+)__"#, with: "$1", options: .regularExpression)
        out = out.replacingOccurrences(of: #"\*([^*]+)\*"#, with: "$1", options: .regularExpression)
        out = out.replacingOccurrences(of: #"_([^_]+)_"#, with: "$1", options: .regularExpression)
        out = out.replacingOccurrences(of: #"~~([^~]+)~~"#, with: "$1", options: .regularExpression)
        out = out.replacingOccurrences(of: #"`([^`]+)`"#, with: "$1", options: .regularExpression)
        return out
    }
}
