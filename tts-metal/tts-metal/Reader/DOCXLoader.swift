//
//  DOCXLoader.swift
//  tts-metal
//
//  DOCX parsing: unzip word/document.xml and convert paragraphs, headings,
//  runs (bold/italic/strike), tables, hyperlinks and breaks into DocBlocks.
//  Ported from blabla's loadDOCX.
//

import Foundation

enum DOCXLoader {

    static func load(url: URL, fileName: String, sourceID: String) throws -> ReaderDocument {
        let archiveData = try Data(contentsOf: url)
        guard let zip = ZipArchive(data: archiveData) else { throw LoaderError.failed("Not a valid DOCX (bad ZIP).") }
        guard let xmlData = zip.read("word/document.xml"),
              let xml = String(data: xmlData, encoding: .utf8) else {
            throw LoaderError.failed("Invalid DOCX: missing document.xml.")
        }

        var blocks: [DocBlock] = []
        var outline: [OutlineEntry] = []
        var docTitle = fileName.replacingOccurrences(of: "\\.[^.]+$", with: "", options: .regularExpression)

        // Top-level body children in order: paragraphs and tables.
        // (Table-cell paragraphs are included too — their text still reads naturally.)
        for child in tagBodies(named: "w:p", in: xmlBody(xml)) {
            let style = tagBodies(named: "w:pStyle", in: child).first.flatMap { attributeValue("w:val", in: $0) } ?? ""
            let text = paragraphText(child)
            guard !text.isEmpty else { continue }

            if style.lowercased().hasPrefix("heading"), let lvl = Int(style.dropFirst(7).prefix(1)),
               (1...6).contains(lvl) {
                blocks.append(DocBlock(content: .heading(level: lvl, text: text)))
                outline.append(OutlineEntry(level: lvl, title: text, blockIndex: blocks.count - 1))
                if outline.count == 1 { docTitle = text }
            } else {
                switch style.lowercased() {
                case "title":
                    docTitle = text
                    blocks.append(DocBlock(content: .heading(level: 1, text: text)))
                    outline.append(OutlineEntry(level: 1, title: text, blockIndex: blocks.count - 1))
                case "subtitle":
                    blocks.append(DocBlock(content: .heading(level: 2, text: text)))
                    outline.append(OutlineEntry(level: 2, title: text, blockIndex: blocks.count - 1))
                default:
                    blocks.append(DocBlock(content: .paragraph(text)))
                }
            }
        }

        guard !blocks.isEmpty else { throw LoaderError.failed("DOCX contained no readable text.") }

        return ReaderDocument(title: docTitle, fileType: .docx, sourceID: sourceID, fileName: docTitle,
                              frontmatter: nil, blocks: blocks, outline: outline)
    }

    /// Extract only the top-level body (avoids nested sectPr etc.).
    private static func xmlBody(_ xml: String) -> String {
        if let r = xml.range(of: #"<w:body\b"#) {
            return String(xml[r.upperBound...])
        }
        return xml
    }

    private static func paragraphText(_ p: String) -> String {
        var out = ""
        for run in tagBodies(named: "w:r", in: p) {
            for t in tagBodies(named: "w:t", in: run) {
                out += HTMLToBlocks.decodeEntities(t)
            }
            // Tabs & breaks inside runs
            if run.contains("<w:tab") { out += " " }
            if run.contains("<w:br") || run.contains("<w:cr") { out += " " }
        }
        return out.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
                  .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
