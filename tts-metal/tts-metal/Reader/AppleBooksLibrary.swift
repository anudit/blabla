//
//  AppleBooksLibrary.swift
//  tts-metal
//
//  The user's Apple Books library, read from Books' own Core Data store so
//  the home screen can shelve those books (with Books' reading progress)
//  beside BlaBla's.
//
//  Books keeps its library in its sandbox container, which macOS guards:
//  the first read raises the system "would like to access data from other
//  apps" prompt. So nothing is read until the user asks for it from the
//  library, and a refusal is remembered as `.denied` with a way to fix it
//  in System Settings (Full Disk Access).
//
//  The store is copied (with its WAL) before it's opened, so BlaBla never
//  holds a lock on a database Books is writing to.
//

import AppKit
import Combine
import Foundation
import SQLite3

struct AppleBook: Identifiable, Equatable {
    let id: String              // Books' asset id
    let title: String
    let author: String?
    let path: String            // .epub (usually an unpacked folder) or .pdf
    let progress: Double        // 0…1, as Books reports it
    let isFinished: Bool
    let lastOpened: Date?
    /// Store purchases are FairPlay-encrypted; only Books can read them.
    let isProtected: Bool

    var fileType: String { (path as NSString).pathExtension.lowercased() }
    var isAvailable: Bool { !isProtected && FileManager.default.fileExists(atPath: path) }

    /// Presented as a library entry, so the grid and cover pipeline can
    /// treat it like any other book.
    var libraryEntry: BookmarkEntry {
        BookmarkEntry(id: "applebooks:\(id)", fileName: title, sentenceIndex: 0, totalSentences: 0,
                      timestamp: lastOpened ?? .distantPast, fileType: fileType, preview: author ?? "",
                      url: nil, ocrPage: nil, filePath: path)
    }
}

@MainActor
final class AppleBooksLibrary: ObservableObject {
    static let shared = AppleBooksLibrary()

    enum Access: Equatable {
        case notRequested       // the user hasn't been asked yet
        case granted
        case denied             // macOS refused the read
        case unavailable        // Books has no library on this Mac
    }

    @Published private(set) var access: Access = .notRequested
    @Published private(set) var books: [AppleBook] = []
    /// "Not Now" on the invitation hides it for good; the library header
    /// keeps a button to connect later.
    @Published private(set) var promptDismissed: Bool

    private let enabledKey = "appleBooks.enabled"
    private let dismissedKey = "appleBooks.promptDismissed"
    private var refreshing = false
    private var lastRefresh: Date?
    private var activationObserver: NSObjectProtocol?

    nonisolated static let containerURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Containers/com.apple.iBooksX/Data", isDirectory: true)
    private nonisolated static let libraryDirectory = containerURL.appendingPathComponent("Documents/BKLibrary", isDirectory: true)
    nonisolated static let coverCacheDirectory = containerURL
        .appendingPathComponent("Library/Caches/BCCoverCache-1/BICDiskDataStore", isDirectory: true)

    private init() {
        promptDismissed = UserDefaults.standard.bool(forKey: dismissedKey)
        if !FileManager.default.fileExists(atPath: Self.containerURL.path) {
            access = .unavailable
        } else if UserDefaults.standard.bool(forKey: enabledKey) {
            refresh(force: true)
        }
        // Books may have moved on while we were in the background.
        activationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }

    /// Whether the home screen should invite the user to connect Books.
    var shouldInvite: Bool { access == .notRequested && !promptDismissed }

    /// Called from the invitation: reading the store is what raises macOS's
    /// permission prompt.
    func requestAccess() {
        UserDefaults.standard.set(true, forKey: enabledKey)
        refresh(force: true)
    }

    func dismissPrompt() {
        promptDismissed = true
        UserDefaults.standard.set(true, forKey: dismissedKey)
    }

    /// Stops reading Books; the invitation stays dismissed, and the
    /// library header offers the way back.
    func disconnect() {
        UserDefaults.standard.set(false, forKey: enabledKey)
        dismissPrompt()
        books = []
        access = .notRequested
    }

    func openFullDiskAccessSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles") {
            NSWorkspace.shared.open(url)
        }
    }

    func openInBooks(_ book: AppleBook) {
        if let url = URL(string: "ibooks://assetid/\(book.id)") { NSWorkspace.shared.open(url) }
    }

    /// Re-reads the library if the user has connected it. Cheap (one small
    /// SQLite copy), but still throttled since activation fires often.
    func refresh(force: Bool = false) {
        guard UserDefaults.standard.bool(forKey: enabledKey), !refreshing else { return }
        if !force, let last = lastRefresh, Date().timeIntervalSince(last) < 15 { return }
        refreshing = true
        Task.detached(priority: .utility) {
            let result = Self.load()
            await MainActor.run {
                self.refreshing = false
                self.lastRefresh = Date()
                switch result {
                case .success(let books):
                    self.access = .granted
                    if books != self.books { self.books = books }
                case .failure(let error):
                    self.access = error == .missing ? .unavailable : .denied
                    self.books = []
                }
            }
        }
    }

    // MARK: - Reading the store (off the main actor)

    private enum LoadError: Error { case missing, denied }

    private nonisolated static func load() -> Result<[AppleBook], LoadError> {
        let fm = FileManager.default
        guard fm.fileExists(atPath: containerURL.path) else { return .failure(.missing) }
        let names: [String]
        do {
            names = try fm.contentsOfDirectory(atPath: libraryDirectory.path)
        } catch let error as NSError {
            // The folder exists (we just saw the container) but can't be
            // listed: that's macOS refusing access.
            let missing = error.domain == NSCocoaErrorDomain && error.code == NSFileReadNoSuchFileError
            return .failure(missing ? .missing : .denied)
        }
        guard let store = names.filter({ $0.hasPrefix("BKLibrary") && $0.hasSuffix(".sqlite") }).sorted().last else {
            return .failure(.missing)
        }

        // Copy the database and its write-ahead log so we see Books' latest
        // state without opening (and locking) the live file.
        let scratch = fm.temporaryDirectory.appendingPathComponent("AppleBooks-\(UUID().uuidString)", isDirectory: true)
        defer { try? fm.removeItem(at: scratch) }
        do {
            try fm.createDirectory(at: scratch, withIntermediateDirectories: true)
            for suffix in ["", "-wal", "-shm"] {
                let src = libraryDirectory.appendingPathComponent(store + suffix)
                guard suffix.isEmpty || fm.fileExists(atPath: src.path) else { continue }
                try fm.copyItem(at: src, to: scratch.appendingPathComponent(store + suffix))
            }
        } catch {
            return .failure(.denied)
        }
        return .success(query(scratch.appendingPathComponent(store)))
    }

    private nonisolated static func query(_ url: URL) -> [AppleBook] {
        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK else {
            sqlite3_close(db)
            return []
        }
        defer { sqlite3_close(db) }

        // Books' schema drifts between releases; only filter on columns
        // this version actually has.
        let columns = columnNames(db, table: "ZBKLIBRARYASSET")
        guard columns.isSuperset(of: ["ZASSETID", "ZTITLE", "ZPATH"]) else { return [] }
        func col(_ name: String, else fallback: String) -> String { columns.contains(name) ? name : fallback }
        var filters = ["ZPATH IS NOT NULL"]
        for flag in ["ZISHIDDEN", "ZISSAMPLE", "ZISSTOREAUDIOBOOK"] where columns.contains(flag) {
            filters.append("IFNULL(\(flag), 0) = 0")
        }
        let lastOpen = col("ZLASTOPENDATE", else: "NULL")
        let sql = """
            SELECT ZASSETID, ZTITLE, \(col("ZAUTHOR", else: "NULL")), ZPATH,
                   \(col("ZREADINGPROGRESS", else: "0")), \(col("ZISFINISHED", else: "0")), \(lastOpen)
            FROM ZBKLIBRARYASSET
            WHERE \(filters.joined(separator: " AND "))
            ORDER BY \(lastOpen) IS NULL, \(lastOpen) DESC, ZTITLE COLLATE NOCASE
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }

        var out: [AppleBook] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let id = text(stmt, 0), let path = text(stmt, 3) else { continue }
            let ext = (path as NSString).pathExtension.lowercased()
            guard ext == "epub" || ext == "pdf" else { continue }
            let fileTitle = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
            let title = text(stmt, 1).flatMap { $0.isEmpty ? nil : $0 } ?? fileTitle
            let author = text(stmt, 2).map(tidyAuthor).flatMap { $0.isEmpty ? nil : $0 }
            let opened = sqlite3_column_type(stmt, 6) == SQLITE_NULL ? nil
                : Date(timeIntervalSinceReferenceDate: sqlite3_column_double(stmt, 6))
            out.append(AppleBook(
                id: id, title: title, author: author, path: path,
                progress: min(max(sqlite3_column_double(stmt, 4), 0), 1),
                isFinished: sqlite3_column_int(stmt, 5) != 0,
                lastOpened: opened,
                isProtected: ext == "epub" && isFairPlay(path)))
        }
        return out
    }

    private nonisolated static func columnNames(_ db: OpaquePointer?, table: String) -> Set<String> {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "PRAGMA table_info(\(table))", -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }
        var names: Set<String> = []
        while sqlite3_step(stmt) == SQLITE_ROW { if let n = text(stmt, 1) { names.insert(n) } }
        return names
    }

    private nonisolated static func text(_ stmt: OpaquePointer?, _ i: Int32) -> String? {
        guard let c = sqlite3_column_text(stmt, i) else { return nil }
        return String(cString: c)
    }

    /// "Peter H. Diamandis;Steven Kotler;" → "Peter H. Diamandis, Steven Kotler".
    private nonisolated static func tidyAuthor(_ raw: String) -> String {
        raw.split(separator: ";").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }.joined(separator: ", ")
    }

    private nonisolated static func isFairPlay(_ path: String) -> Bool {
        FileManager.default.fileExists(atPath: (path as NSString).appendingPathComponent("META-INF/sinf.xml"))
    }

    // MARK: - Covers

    /// The cover Books itself rendered, from its cover cache: the largest
    /// portrait HEIC (files are named "<asset>|1|<w>|<h>|<flags>.heic").
    nonisolated static func cachedCover(assetID: String) -> Data? {
        let dir = coverCacheDirectory.appendingPathComponent(assetID, isDirectory: true)
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return nil }
        let best = names.compactMap { name -> (name: String, pixels: Int)? in
            guard name.hasSuffix(".heic") else { return nil }
            let parts = name.split(separator: "|")
            guard parts.count >= 4, let w = Int(parts[2]), let h = Int(parts[3]), h > w else { return nil }
            return (name, w * h)
        }.max { $0.pixels < $1.pixels }
        guard let best else { return nil }
        return try? Data(contentsOf: dir.appendingPathComponent(best.name))
    }
}
