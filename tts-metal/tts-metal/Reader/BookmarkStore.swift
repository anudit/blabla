//
//  BookmarkStore.swift
//  tts-metal
//
//  Reading history / auto-resume. Ported from blabla's localStorage
//  `blabla_bookmarks` (max 20 entries), keyed by file identity.
//

import Foundation
import Combine

struct BookmarkEntry: Codable, Identifiable, Equatable {
    var id: String                 // "name:size" or "url:<url>" or "paste:<title>"
    var fileName: String
    var sentenceIndex: Int
    var totalSentences: Int
    var timestamp: Date
    var fileType: String           // pdf | epub | mobi | docx | url | text | ocr
    var preview: String
    var url: String?
    var ocrPage: Int?
    var filePath: String?          // absolute path for auto-reopen on tap

    var progress: Double {
        totalSentences > 0 ? Double(sentenceIndex) / Double(totalSentences) : 0
    }

    func relativeTimeString(now: Date = Date()) -> String {
        let dt = now.timeIntervalSince(timestamp)
        if dt < 60 { return "Just now" }
        if dt < 3600 { return "\(Int(dt / 60))m ago" }
        if dt < 86_400 { return "\(Int(dt / 3600))h ago" }
        return "\(Int(dt / 86_400))d ago"
    }
}

@MainActor
final class BookmarkStore: ObservableObject {
    static let shared = BookmarkStore()
    static let maxEntries = 200

    @Published private(set) var entries: [BookmarkEntry] = []

    private let key = "blabla_bookmarks"

    private init() {
        if let data = UserDefaults.standard.data(forKey: key),
           let decoded = try? JSONDecoder().decode([BookmarkEntry].self, from: data) {
            entries = decoded
        }
    }

    func save(entry: BookmarkEntry) {
        entries.removeAll { $0.id == entry.id }
        entries.insert(entry, at: 0)
        if entries.count > Self.maxEntries {
            entries = Array(entries.prefix(Self.maxEntries))
        }
        persist()
    }

    func remove(id: String) {
        entries.removeAll { $0.id == id }
        persist()
    }

    func entry(for sourceID: String) -> BookmarkEntry? {
        entries.first { $0.id == sourceID }
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(entries) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
}
