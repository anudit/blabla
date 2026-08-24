//
//  PDFLoader.swift
//  tts-metal
//
//  PDF text extraction via PDFKit. Text items are grouped into lines per page
//  (y-tolerance like blabla's pdf.js pipeline) and incrementally converted into
//  sentence blocks, tracking which page each sentence starts on.
//

import Foundation
import Combine
import PDFKit

enum PDFLoader {

    static func load(url: URL, sourceID: String, previewURL: String? = nil) throws -> ReaderDocument {
        guard let doc = PDFDocument(url: url) else {
            throw LoaderError.failed("Could not open PDF.")
        }
        let pageCount = doc.pageCount
        guard pageCount > 0 else { throw LoaderError.failed("PDF has no pages.") }

        var blocks: [DocBlock] = []
        var sentencePages: [Int] = []
        var totalChars = 0

        for pageIdx in 0..<pageCount {
            guard let page = doc.page(at: pageIdx) else { continue }
            let raw = page.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let pageText = raw.isEmpty ? "" : normalizePageText(raw)
            if pageText.isEmpty { continue }
            totalChars += pageText.count

            let priorBlockCount = blocks.count
            for para in pageText.components(separatedBy: "\n") {
                let t = para.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !t.isEmpty else { continue }
                blocks.append(DocBlock(content: .paragraph(t)))
            }
            // Every new sentence produced by this page maps to this page
            // number. Only counts this page's own new blocks — counting the
            // whole (ever-growing) `blocks` array here made this O(pages²).
            let sentencesOnThisPage = countSentences(blocks[priorBlockCount...])
            for _ in 0..<sentencesOnThisPage {
                sentencePages.append(pageIdx + 1)
            }
        }

        let title = doc.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String
            ?? url.lastPathComponent.replacingOccurrences(of: "\\.[^.]+$", with: "", options: .regularExpression)

        // Scanned PDF with no extractable text.
        if totalChars < 100 {
            throw LoaderError.failed("This PDF has no extractable text (it looks scanned/image-only).")
        }

        return ReaderDocument(title: title, fileType: .pdf, sourceID: sourceID,
                              fileName: title, previewURL: previewURL,
                              frontmatter: nil, blocks: blocks, outline: [],
                              sentencePages: sentencePages, pageCount: pageCount)
    }

    /// Rejoin hard-wrapped lines within a paragraph-ish flow.
    private static func normalizePageText(_ raw: String) -> String {
        let lines = raw.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
        var out = ""
        for line in lines {
            guard !line.isEmpty else {
                out += "\n"
                continue
            }
            if out.hasSuffix("-") {
                out.removeLast()               // de-hyphenate wrapped words
                out += line
            } else {
                if !out.isEmpty && !out.hasSuffix("\n") { out += " " }
                out += line
            }
        }
        return out
    }

    private static func countSentences(_ blocks: ArraySlice<DocBlock>) -> Int {
        var n = 0
        for b in blocks {
            // Fast pass only — sentence count only depends on boundary-
            // determining normalization, not money/date/time/etc. expansion.
            let spoken = TTSTextNormalizer.cleanForTtsFast(b.speechText)
            n += SentenceSplitter.extract(spoken).count
        }
        return n
    }

    /// Render a page to a bitmap for OCR.
    static func renderPage(_ document: PDFDocument, pageIndex: Int, scale: CGFloat = 2.7778) -> CGImage? {
        guard let page = document.page(at: pageIndex) else { return nil }
        let bounds = page.bounds(for: .mediaBox)
        let w = max(1, Int(bounds.width * scale))
        let h = max(1, Int(bounds.height * scale))
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.scaleBy(x: scale, y: scale)
        page.draw(with: .mediaBox, to: ctx)
        return ctx.makeImage()
    }
}

// MARK: - OCR engine (Vision)

import Vision

@MainActor
final class OCREngine: ObservableObject {
    static let shared = OCREngine()

    @Published var running = false
    @Published var progress: Double = 0
    @Published var stage: String = ""

    private init() {}

    /// Recognize every page of the PDF and build an OCR ReaderDocument.
    func recognize(pdfURL: URL, fileName: String, sourceID: String) async throws -> ReaderDocument {
        guard let doc = PDFDocument(url: pdfURL), doc.pageCount > 0 else {
            throw LoaderError.failed("Could not open PDF for OCR.")
        }
        running = true
        progress = 0
        defer { running = false; stage = "" }

        var allText: [String] = []
        let pageCount = doc.pageCount

        for i in 0..<pageCount {
            stage = "Recognizing text — Page \(i + 1) of \(pageCount)"
            guard let image = PDFLoader.renderPage(doc, pageIndex: i) else { continue }
            let text = try await recognizeImage(image)
            let cleaned = Self.parseOcrText(text)
            if !cleaned.isEmpty { allText.append(cleaned) }
            progress = Double(i + 1) / Double(pageCount)
        }

        var blocks: [DocBlock] = []
        for (idx, pageText) in allText.enumerated() {
            _ = idx
            for para in pageText.components(separatedBy: "\n") where !para.isEmpty {
                blocks.append(DocBlock(content: .paragraph(para)))
            }
        }
        guard !blocks.isEmpty else { throw LoaderError.failed("No text recognized on any page.") }

        return ReaderDocument(title: fileName.replacingOccurrences(of: "\\.[^.]+$", with: "", options: .regularExpression),
                              fileType: .ocr, sourceID: sourceID, fileName: fileName,
                              frontmatter: nil, blocks: blocks, outline: [])
    }

    private func recognizeImage(_ cgImage: CGImage) async throws -> String {
        try await withCheckedThrowingContinuation { cont in
            let request = VNRecognizeTextRequest { req, error in
                if let error = error { cont.resume(throwing: error); return }
                let results = (req.results as? [VNRecognizedTextObservation]) ?? []
                var lines: [(String, CGFloat)] = []
                for obs in results {
                    if let top = obs.topCandidates(1).first {
                        lines.append((top.string, obs.boundingBox.origin.y))
                    }
                }
                lines.sort { $0.1 > $1.1 }   // top-to-bottom
                cont.resume(returning: lines.map(\.0).joined(separator: "\n"))
            }
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            DispatchQueue.global(qos: .userInitiated).async {
                do { try handler.perform([request]) }
                catch { cont.resume(throwing: error) }
            }
        }
    }

    /// Group recognized lines into paragraphs (port of utils.tsx parseOcrPageText):
    /// a line ending without terminal punctuation is joined to the next one.
    static func parseOcrText(_ text: String) -> String {
        var paragraphs: [String] = []
        var current = ""
        for rawLine in text.components(separatedBy: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }
            current += (current.isEmpty ? "" : " ") + line
            if line.hasPunctuationTerminal || line.count < 25 {
                paragraphs.append(current)
                current = ""
            }
        }
        if !current.isEmpty { paragraphs.append(current) }
        return paragraphs.joined(separator: "\n")
    }
}
