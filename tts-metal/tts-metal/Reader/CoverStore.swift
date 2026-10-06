//
//  CoverStore.swift
//  tts-metal
//
//  Cover art for the library grid. Each book's cover is resolved once, in
//  this order, and cached on disk:
//
//    1. the cover embedded in the file (EPUB manifest cover image);
//    2. Open Library's cover for the title (and author, when the file names
//       one) — accepted only if the match's title actually agrees with ours,
//       because a loose search returns *some* book for almost any string;
//    3. for a PDF with neither, its first page;
//    (an Apple Books title tries Books' own cover cache before all of these;)
//    4. otherwise nothing — the grid draws a typographic cover from the title.
//
//  A miss is cached too (as an empty marker file), so a book with no cover
//  anywhere doesn't hit the network every time the library is shown.
//

import AppKit
import Combine
import CryptoKit
import ImageIO
import PDFKit

@MainActor
final class CoverStore: ObservableObject {
    static let shared = CoverStore()

    /// Resolved covers by entry id. An id that maps to `nil` was looked up
    /// and has no cover image.
    @Published private(set) var covers: [String: NSImage?] = [:]
    private var inFlight: Set<String> = []

    private let directory: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let dir = base.appendingPathComponent(Bundle.main.bundleIdentifier ?? "tts-metal")
            .appendingPathComponent("Covers", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    /// Starts resolving `entry`'s cover if it hasn't been already.
    func request(_ entry: BookmarkEntry) {
        guard covers[entry.id] == nil, !inFlight.contains(entry.id) else { return }
        inFlight.insert(entry.id)
        let file = cacheFile(for: entry.id)
        Task.detached(priority: .utility) {
            let image = await Self.resolve(entry, cacheFile: file)
            await MainActor.run {
                self.covers[entry.id] = .some(image)
                self.inFlight.remove(entry.id)
            }
        }
    }

    private func cacheFile(for id: String) -> URL {
        let hash = SHA256.hash(data: Data(id.utf8)).map { String(format: "%02x", $0) }.joined()
        return directory.appendingPathComponent(String(hash.prefix(32)) + ".img")
    }

    // MARK: - Resolution (off the main actor)

    private nonisolated static func resolve(_ entry: BookmarkEntry, cacheFile: URL) async -> NSImage? {
        if let cached = try? Data(contentsOf: cacheFile) {
            return cached.isEmpty ? nil : NSImage(data: cached)
        }
        let data = await fetch(entry)
        try? (data ?? Data()).write(to: cacheFile, options: .atomic)
        return data.flatMap(NSImage.init(data:))
    }

    private nonisolated static func fetch(_ entry: BookmarkEntry) async -> Data? {
        let path = entry.filePath.flatMap { FileManager.default.fileExists(atPath: $0) ? $0 : nil }
        var author: String?

        // 0. An Apple Books title: the cover Books already rendered.
        if entry.id.hasPrefix("applebooks:"),
           let d = AppleBooksLibrary.cachedCover(assetID: String(entry.id.dropFirst("applebooks:".count))) {
            return d
        }

        // 1. Embedded cover.
        if entry.fileType == "epub", let path,
           let zip = (try? ZipArchive.open(URL(fileURLWithPath: path))) ?? nil {
            let info = EPUBLoader.coverInfo(zip: zip)
            if let img = info.image, let normalized = downscaled(img) { return normalized }
            author = info.author
        }

        // 2. Open Library — books only; articles, pasted text and OCR'd
        //    screenshots have no catalogue entry to find.
        if ["epub", "mobi", "pdf", "docx"].contains(entry.fileType),
           let d = await openLibraryCover(title: entry.fileName, author: author) {
            return d
        }

        // 3. A PDF's own first page.
        if entry.fileType == "pdf", let path,
           let page = PDFDocument(url: URL(fileURLWithPath: path))?.page(at: 0) {
            let bounds = page.bounds(for: .mediaBox)
            let scale = 600 / max(bounds.height, 1)
            let thumb = page.thumbnail(of: CGSize(width: bounds.width * scale, height: 600), for: .mediaBox)
            return jpeg(thumb)
        }
        return nil
    }

    private struct OLSearch: Decodable {
        struct Doc: Decodable { let title: String?; let cover_i: Int? }
        let docs: [Doc]
    }

    private nonisolated static func openLibraryCover(title raw: String, author: String?) async -> Data? {
        let title = searchableTitle(raw)
        guard title.count >= 3 else { return nil }
        var comps = URLComponents(string: "https://openlibrary.org/search.json")!
        comps.queryItems = [
            URLQueryItem(name: "title", value: title),
            URLQueryItem(name: "fields", value: "title,cover_i"),
            URLQueryItem(name: "limit", value: "5"),
        ]
        if let author { comps.queryItems?.append(URLQueryItem(name: "author", value: author)) }
        guard let url = comps.url else { return nil }
        var req = URLRequest(url: url, timeoutInterval: 10)
        req.setValue("BlaBla/1.0 (macOS reader)", forHTTPHeaderField: "User-Agent")
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let result = try? JSONDecoder().decode(OLSearch.self, from: data) else { return nil }

        let want = normalized(title)
        guard let doc = result.docs.first(where: { d in
            guard d.cover_i != nil, let t = d.title.map(normalized), !t.isEmpty else { return false }
            return t == want || want.hasPrefix(t) || t.hasPrefix(want)
        }), let id = doc.cover_i,
              let coverURL = URL(string: "https://covers.openlibrary.org/b/id/\(id)-L.jpg?default=false"),
              let (img, imgResp) = try? await URLSession.shared.data(from: coverURL),
              (imgResp as? HTTPURLResponse)?.statusCode == 200,
              NSImage(data: img) != nil else { return nil }
        return img
    }

    /// File-name titles carry download-site debris: "Author - Title (2021,
    /// Publisher) - libgen.li", "Title -- Author -- ( WeLib.org )",
    /// "Title{Author}(2013…)". Keep the part that is most likely the title.
    nonisolated static func searchableTitle(_ raw: String) -> String {
        var t = raw
        t = t.replacingOccurrences(of: #"\{[^}]*\}|\([^)]*\)|\[[^\]]*\]"#, with: " ", options: .regularExpression)
        t = t.replacingOccurrences(of: #"\b(libgen\.\w+|z-lib\.\w+|anna’s archive|welib\.org)\b"#,
                                   with: " ", options: [.regularExpression, .caseInsensitive])
        let parts = t.components(separatedBy: " -- ").map { $0.trimmingCharacters(in: .whitespaces) }
        if parts.count > 1 { t = parts[0] }
        else {
            // "Author - Title": a short first segment with a comma or two
            // capitalised words is the author.
            let dash = t.components(separatedBy: " - ").map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            if dash.count >= 2 { t = dash[0].split(separator: " ").count <= 4 ? dash[1] : dash[0] }
        }
        // Subtitles: "Title_ Subtitle" (libgen's colon) and "Title: Subtitle".
        if let r = t.range(of: #"[_:]\s"#, options: .regularExpression) { t = String(t[..<r.lowerBound]) }
        return t.trimmingCharacters(in: CharacterSet.whitespaces.union(.punctuationCharacters))
    }

    private nonisolated static func normalized(_ s: String) -> String {
        let base = s.lowercased().folding(options: .diacriticInsensitive, locale: nil)
        let cut = base.range(of: #"[:(]"#, options: .regularExpression).map { String(base[..<$0.lowerBound]) } ?? base
        return cut.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(String.init).joined()
    }

    /// Embedded covers are often multi-megapixel; the grid never shows one
    /// larger than a few hundred points.
    private nonisolated static func downscaled(_ data: Data) -> Data? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: 900,
              ] as CFDictionary) else { return nil }
        return NSBitmapImageRep(cgImage: cg).representation(using: .jpeg, properties: [.compressionFactor: 0.85])
    }

    private nonisolated static func jpeg(_ image: NSImage) -> Data? {
        guard let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85])
    }
}
