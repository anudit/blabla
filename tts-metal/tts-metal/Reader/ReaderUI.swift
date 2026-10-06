//
//  ReaderUI.swift
//  tts-metal
//
//  The BlaBla reader window: landing screen (drag-and-drop / URL / clipboard /
//  paste / continue-reading history), document reader with sentence karaoke
//  highlighting and auto-scroll, TOC outline sidebar, bottom-bar cockpit
//  (transport, speed popover, settings, themes) and scroll-to-current button.
//
//  Visual design follows the BlaBla web app (theme-driven cream/dark palettes,
//  floating bottom pill, Aa theme swatch grid).
//
//  Ported from blabla's App.tsx / LandingCard.tsx / ContentViewer.tsx /
//  BottomBar.tsx / BookmarkHistory.tsx / BookOutline.tsx.
//

import SwiftUI
import UniformTypeIdentifiers
import AppKit

// MARK: - Root

struct ReaderRootView: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader

    var body: some View {
        content
        .frame(minWidth: 760, minHeight: 560)
        .background(Color(hex: reader.theme.bg))
        .onAppear { Self.installSpaceKey() }
        .toolbar {
            // Only while a book is open: the home screen has nothing to
            // outline and nowhere to go back to.
            if reader.state != .empty {
                ToolbarItem(placement: .navigation) {
                    Button {
                        reader.stopPlayback()
                        reader.resetDocument()
                    } label: {
                        Label("Home", systemImage: "house.fill")
                    }
                    .help("Back to home — stop playback")
                }
            }
            if reader.document != nil {
                ToolbarItem(placement: .navigation) {
                    Button {
                        reader.outlineVisible.toggle()
                    } label: {
                        Label("Contents", systemImage: "list.bullet")
                    }
                    .help("Show table of contents")
                    // A popover, not a sidebar: inset as a sidebar it narrowed the
                    // reading column, and the column width is what the book's
                    // layout is built against — so every toggle re-rendered the
                    // whole book and lost the reader's place. Floating it over the
                    // text leaves the column untouched.
                    .popover(isPresented: $reader.outlineVisible, arrowEdge: .bottom) {
                        OutlineSidebar()
                            .frame(width: 300, height: 560)
                    }
                }
            }
        }
    }

    /// Space plays and pauses whenever a book is open — unless the user is
    /// typing (a search field, Ask, the URL box), where it's just a space.
    private static var spaceKeyInstalled = false
    private static func installSpaceKey() {
        guard !spaceKeyInstalled else { return }
        spaceKeyInstalled = true
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            let reader = ReaderControllerHolder.reader
            guard event.keyCode == 49,                 // space
                  event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                      .subtracting([.capsLock, .numericPad, .function]).isEmpty,
                  reader.document != nil,
                  !isTyping(in: event.window)
            else { return event }
            reader.togglePlayPause()
            return nil
        }
    }

    /// A text field's editor and an editable text view both take spaces;
    /// the book's own (read-only, selectable) text view doesn't.
    private static func isTyping(in window: NSWindow?) -> Bool {
        switch window?.firstResponder {
        case let text as NSTextView: return text.isEditable
        case is NSTextField: return true
        default: return false
        }
    }

    @ViewBuilder
    private var content: some View {
        switch reader.state {
        case .empty:
            LandingView()
        case .loadingDoc:
            VStack(spacing: 12) {
                ProgressView()
                Text(reader.statusText.isEmpty ? "Loading…" : reader.statusText)
                    .font(.callout)
                    .foregroundStyle(Color(hex: reader.theme.textMuted))
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed(let message):
            VStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 32))
                    .foregroundStyle(.orange)
                Text("Could not open document").font(.headline)
                    .foregroundStyle(Color(hex: reader.theme.text))
                Text(message).font(.caption)
                    .foregroundStyle(Color(hex: reader.theme.textMuted))
                Button("Back") { reader.resetDocument() }
                    .buttonStyle(.borderedProminent)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        default:
            ZStack(alignment: .bottom) {
                DocumentReaderView()
                BottomBar()
            }
        }
    }
}

// MARK: - Landing

struct LandingView: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @ObservedObject private var bookmarks = BookmarkStore.shared
    @State private var urlString = ""
    @State private var draftText = ""
    @State private var showTextarea = false
    @State private var isDragging = false

    private var theme: ReaderTheme { reader.theme }

    @ObservedObject private var covers = CoverStore.shared
    @ObservedObject private var appleBooks = AppleBooksLibrary.shared
    @State private var filter: LibraryFilter = .all

    /// Anything to shelve — BlaBla's own history or the Apple Books library.
    private var hasShelf: Bool { !bookmarks.entries.isEmpty || !appleBooks.books.isEmpty }

    var body: some View {
        ScrollView {
            if !hasShelf {
                // First run: nothing to show yet, so the drop card is the page.
                VStack(spacing: 26) {
                    card
                    if showTextarea { textarea }
                    appleBooksBanner
                }
                .padding(.horizontal, 28)
                .padding(.top, 46)
                .padding(.bottom, 110)
                .frame(maxWidth: 680)
                .frame(maxWidth: .infinity)
            } else {
                VStack(alignment: .leading, spacing: 22) {
                    libraryHeader
                    if showTextarea { textarea.frame(maxWidth: 680) }
                    appleBooksBanner
                    if !visibleEntries.isEmpty { libraryGrid }
                    if !visibleAppleBooks.isEmpty { appleBooksSection }
                }
                // A centred column, like the reading view: on a wide window
                // the shelf stays a comfortable width with margin either side
                // instead of stretching to the window edges.
                .frame(maxWidth: 1080)
                .padding(.horizontal, 56)
                .padding(.top, 30)
                .padding(.bottom, 110)
                .frame(maxWidth: .infinity)
            }
        }
        .background(Color(hex: theme.bg))
        .overlay { if isDragging && hasShelf { dropOverlay } }
        .onAppear {
            registerPasteHandler()
            appleBooks.refresh()
        }
        .onDrop(of: [UTType.fileURL], isTargeted: $isDragging) { providers in
            handleDrop(providers)
        }
    }

    // The big rounded card: drop zone + URL row + clipboard button.
    private var card: some View {
        VStack(spacing: 0) {
            // Drop zone
            VStack(spacing: 12) {
                Image(systemName: "square.and.arrow.up")
                    .font(.system(size: 20, weight: .medium))
                    .foregroundStyle(Color(hex: theme.textMuted))
                    .frame(width: 52, height: 52)
                    .background(Color(hex: theme.inputBg), in: RoundedRectangle(cornerRadius: 14))

                Text(isDragging ? "Drop to start reading" : "Drop a file to start reading")
                    .font(.system(size: 21, weight: .bold))
                    .foregroundStyle(Color(hex: theme.text))
                Text("PDF · EPUB · MOBI · DOCX · Markdown · TXT")
                    .font(.system(size: 15))
                    .foregroundStyle(Color(hex: theme.textMuted))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 44)
            .contentShape(Rectangle())
            .onTapGesture { pickFile() }

            Divider()
                .overlay(Color(hex: theme.dropBorder).opacity(0.5))

            // URL row
            HStack(spacing: 10) {
                Image(systemName: "globe")
                    .foregroundStyle(Color(hex: theme.textMuted))
                TextField("Paste a URL to read as an article...", text: $urlString)
                    .textFieldStyle(.plain)
                    .font(.system(size: 15))
                    .foregroundStyle(Color(hex: theme.text))
                Button {
                    let s = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !s.isEmpty else { return }
                    reader.loadURL(s)
                } label: {
                    Text("Load URL")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Color(hex: theme.text))
                        .padding(.horizontal, 18)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color(hex: theme.inputBg))
                                .overlay(RoundedRectangle(cornerRadius: 10)
                                    .strokeBorder(Color(hex: theme.inputBorder)))
                        )
                }
                .buttonStyle(.plain)
                .disabled(urlString.trimmingCharacters(in: .whitespaces).isEmpty)
                .opacity(urlString.trimmingCharacters(in: .whitespaces).isEmpty ? 0.5 : 1)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(hex: theme.inputBg))
                    .overlay(RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(Color(hex: theme.inputBorder)))
            )
            .padding(.horizontal, 22)
            .padding(.top, 18)

            // Clipboard button
            Button {
                handleClipboard()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "clipboard")
                    Text("Paste from Clipboard")
                        .font(.system(size: 15, weight: .medium))
                }
                .foregroundStyle(Color(hex: theme.text))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(hex: theme.inputBg))
                        .overlay(RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(Color(hex: theme.inputBorder)))
                )
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 22)
            .padding(.top, 12)
            .padding(.bottom, 22)
        }
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(isDragging ? Color(hex: theme.dropBg).opacity(0.9) : Color(hex: theme.dropBg))
                .overlay(RoundedRectangle(cornerRadius: 20)
                    .strokeBorder(isDragging ? Color.accentColor : Color(hex: theme.dropBorder).opacity(0.7),
                                  lineWidth: isDragging ? 2 : 1))
        )
    }

    private var textarea: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextEditor(text: $draftText)
                .font(.body)
                .scrollContentBackground(.hidden)
                .foregroundStyle(Color(hex: theme.text))
                .frame(height: 120)
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(hex: theme.inputBg))
                        .overlay(RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(Color(hex: theme.inputBorder)))
                )
            HStack {
                Spacer()
                Button("Start Reading") {
                    let t = draftText.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !t.isEmpty else { return }
                    showTextarea = false
                    reader.loadText(t, title: "Pasted text")
                }
                .buttonStyle(.borderedProminent)
                .disabled(draftText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    // MARK: Library

    private var visibleEntries: [BookmarkEntry] {
        bookmarks.entries.filter { filter.includes($0.fileType) }
    }

    /// Apple Books titles not already on BlaBla's shelf (once opened here,
    /// a book's BlaBla entry, with BlaBla's progress, stands in for it).
    private var visibleAppleBooks: [AppleBook] {
        let shelved = Set(bookmarks.entries.compactMap(\.filePath))
        return appleBooks.books.filter { !shelved.contains($0.path) && filter.includes($0.fileType) }
    }

    private var libraryHeader: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 12) {
                Text("Library")
                    .font(.system(size: 32, weight: .bold, design: .serif))
                    .foregroundStyle(Color(hex: theme.headerColor))
                Spacer(minLength: 20)
                urlField
                headerButton("Paste", systemImage: "doc.on.clipboard", action: handleClipboard)
                    .help("Read text or a link from the clipboard (⌘V)")
                if appleBooks.access == .notRequested && appleBooks.promptDismissed {
                    headerButton("Apple Books", systemImage: "books.vertical", action: appleBooks.requestAccess)
                        .help("Show your Apple Books library here")
                }
                headerButton("Add", systemImage: "plus", prominent: true, action: pickFile)
                    .help("Open a PDF, EPUB, MOBI, DOCX, Markdown or text file")
            }
            // Only the kinds actually present get a filter.
            let kinds = LibraryFilter.allCases.filter { f in
                f == .all || bookmarks.entries.contains { f.includes($0.fileType) }
                    || appleBooks.books.contains { f.includes($0.fileType) }
            }
            if kinds.count > 2 {
                HStack(spacing: 6) {
                    ForEach(kinds, id: \.self) { f in
                        Button { filter = f } label: {
                            Text(f.title)
                                .font(.system(size: 13, weight: filter == f ? .semibold : .regular))
                                .foregroundStyle(Color(hex: filter == f ? theme.bg : theme.text))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 5)
                                .background(Capsule().fill(filter == f
                                    ? Color(hex: theme.text)
                                    : Color(hex: theme.inputBg)))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var urlField: some View {
        HStack(spacing: 6) {
            Image(systemName: "globe")
                .font(.system(size: 12))
                .foregroundStyle(Color(hex: theme.textMuted))
            TextField("Open a URL…", text: $urlString)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .foregroundStyle(Color(hex: theme.text))
                .onSubmit {
                    let s = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !s.isEmpty else { return }
                    reader.loadURL(s)
                }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(width: 220)
        .background(
            Capsule().fill(Color(hex: theme.inputBg))
                .overlay(Capsule().strokeBorder(Color(hex: theme.inputBorder).opacity(0.7)))
        )
    }

    private func headerButton(_ title: String, systemImage: String, prominent: Bool = false,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Color(hex: prominent ? theme.bg : theme.text))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    Capsule().fill(Color(hex: prominent ? theme.text : theme.inputBg))
                        .overlay(Capsule().strokeBorder(Color(hex: theme.inputBorder)
                            .opacity(prominent ? 0 : 0.7)))
                )
        }
        .buttonStyle(.plain)
    }

    private var libraryGrid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150, maximum: 200), spacing: 34, alignment: .bottom)],
                  alignment: .leading, spacing: 34) {
            ForEach(visibleEntries) { entry in
                let available = isAvailable(entry)
                LibraryBookCell(entry: entry, theme: theme,
                                cover: covers.covers[entry.id] ?? nil,
                                available: available,
                                status: .init(entry, available: available),
                                open: { open(entry) },
                                remove: { bookmarks.remove(id: entry.id) })
                    .onAppear { covers.request(entry) }
            }
        }
    }

    // MARK: Apple Books

    private var appleBooksSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .center, spacing: 10) {
                if let icon = AppleBooksBanner.booksIcon {
                    Image(nsImage: icon)
                        .resizable()
                        .interpolation(.high)
                        .frame(width: 30, height: 30)
                        .accessibilityHidden(true)
                }
                Text("Books")
                    .font(.system(size: 22, weight: .bold, design: .serif))
                    .foregroundStyle(Color(hex: theme.headerColor))
                Text("\(visibleAppleBooks.count)")
                    .font(.system(size: 13, weight: .medium).monospacedDigit())
                    .foregroundStyle(Color(hex: theme.textMuted))
                Spacer()
                Menu {
                    Button("Open Apple Books") {
                        if let app = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.iBooksX") {
                            NSWorkspace.shared.openApplication(at: app, configuration: .init())
                        }
                    }
                    Button("Refresh") { appleBooks.refresh(force: true) }
                    Divider()
                    Button("Stop Showing Apple Books", action: appleBooks.disconnect)
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 15))
                        .foregroundStyle(Color(hex: theme.textMuted))
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
            }
            .padding(.top, visibleEntries.isEmpty ? 0 : 14)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150, maximum: 200), spacing: 34, alignment: .bottom)],
                      alignment: .leading, spacing: 34) {
                ForEach(visibleAppleBooks) { book in
                    let entry = book.libraryEntry
                    LibraryBookCell(entry: entry, theme: theme,
                                    cover: covers.covers[entry.id] ?? nil,
                                    available: book.isAvailable,
                                    status: .init(book),
                                    open: { open(book) },
                                    remove: nil,
                                    openInBooks: { appleBooks.openInBooks(book) })
                        .onAppear { covers.request(entry) }
                }
            }
        }
    }

    /// Invitation to connect Apple Books, or — once macOS has refused —
    /// the way to grant access in System Settings.
    @ViewBuilder
    private var appleBooksBanner: some View {
        if appleBooks.shouldInvite {
            AppleBooksBanner(
                theme: theme,
                title: "Bring in your Apple Books library",
                message: "See the books you're reading in Apple Books here, with your progress, and listen to any of them. macOS will ask to let BlaBla access Apple Books' data.",
                primary: ("Allow Access", appleBooks.requestAccess),
                secondary: ("Not Now", appleBooks.dismissPrompt))
        } else if appleBooks.access == .denied {
            AppleBooksBanner(
                theme: theme,
                title: "BlaBla can't read your Apple Books library",
                message: "Turn on BlaBla under Privacy & Security › Full Disk Access in System Settings, then come back — your books will appear here.",
                primary: ("Open System Settings", appleBooks.openFullDiskAccessSettings),
                secondary: ("Hide", appleBooks.disconnect))
        }
    }

    private var dropOverlay: some View {
        RoundedRectangle(cornerRadius: 18)
            .strokeBorder(Color(hex: theme.accent), style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
            .background(RoundedRectangle(cornerRadius: 18).fill(Color(hex: theme.bg).opacity(0.75)))
            .overlay(
                VStack(spacing: 10) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 28, weight: .medium))
                    Text("Drop to start reading")
                        .font(.system(size: 19, weight: .semibold))
                }
                .foregroundStyle(Color(hex: theme.text))
            )
            .padding(16)
            .allowsHitTesting(false)
    }

    private func isAvailable(_ entry: BookmarkEntry) -> Bool {
        if let fp = entry.filePath { return FileManager.default.fileExists(atPath: fp) }
        return entry.url != nil
    }

    private func open(_ book: AppleBook) {
        // Protected (store-bought) books can only be read by Books itself.
        guard book.isAvailable else {
            appleBooks.openInBooks(book)
            return
        }
        reader.loadFileURL(URL(fileURLWithPath: book.path),
                           startFraction: book.isFinished ? nil : book.progress)
    }

    private func open(_ entry: BookmarkEntry) {
        PerfLog.log("resume tapped: \(entry.fileName)")
        // Auto-reopen: prefer the saved file path (resume at saved line), else URL.
        if let fp = entry.filePath, FileManager.default.fileExists(atPath: fp) {
            reader.loadFileURL(URL(fileURLWithPath: fp))
        } else if let u = entry.url {
            reader.loadURL(u)
        }
    }

    // MARK: - Actions

    private func pickFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = DocLoader.supportedExtensions.compactMap { UTType(filenameExtension: $0) }
        if panel.runModal() == .OK, let url = panel.url {
            reader.loadFileURL(url)
        }
    }

    private func handleClipboard() {
        guard let content = NSPasteboard.general.string(forType: .string)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !content.isEmpty else {
            showTextarea = true
            return
        }
        if content.range(of: #"^https?://\S+$"#, options: .regularExpression) != nil {
            reader.loadURL(content)
        } else if content.count < 400 {
            reader.loadText(content)
        } else {
            draftText = content
            showTextarea = true
        }
    }

    private func registerPasteHandler() {
        // Register exactly once per app run.
        guard !Self.pasteMonitorInstalled else { return }
        Self.pasteMonitorInstalled = true
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            guard event.keyCode == 9,          // 'V'
                  event.modifierFlags.contains(.command),
                  !(NSApp.keyWindow?.firstResponder is NSTextView || NSApp.keyWindow?.firstResponder is NSTextField)
            else { return event }
            handleClipboard()
            return nil
        }
    }

    private static var pasteMonitorInstalled = false

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let first = providers.first else { return false }
        first.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            var url: URL?
            if let data = item as? Data {
                url = URL(dataRepresentation: data, relativeTo: nil)
            } else if let u = item as? URL {
                url = u
            }
            guard let url else { return }
            Task { @MainActor in reader.loadFileURL(url) }
        }
        return true
    }
}

// MARK: - Library grid

enum LibraryFilter: CaseIterable, Hashable {
    case all, books, pdfs, articles

    var title: String {
        switch self {
        case .all: return "All"
        case .books: return "Books"
        case .pdfs: return "PDFs"
        case .articles: return "Articles & Text"
        }
    }

    func includes(_ fileType: String) -> Bool {
        switch self {
        case .all: return true
        case .books: return ["epub", "mobi", "docx"].contains(fileType)
        case .pdfs: return fileType == "pdf"
        case .articles: return ["url", "text", "ocr"].contains(fileType)
        }
    }
}

/// One book in the library: its cover standing on a shared baseline (covers
/// keep their own proportions, as on a shelf), and beneath it the reading
/// progress and a "…" menu — the Apple Books arrangement.
private struct LibraryBookCell: View {
    let entry: BookmarkEntry
    let theme: ReaderTheme
    let cover: NSImage?
    let available: Bool
    let status: LibraryBookStatus
    let open: () -> Void
    /// Nil for books BlaBla doesn't own the shelf entry for (Apple Books).
    let remove: (() -> Void)?
    var openInBooks: (() -> Void)? = nil
    @State private var hovering = false

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Color.clear
                .aspectRatio(2.0 / 3.0, contentMode: .fit)
                .overlay(alignment: .bottom) { coverView }
                .contentShape(Rectangle())
                .onTapGesture(perform: open)
                .onHover { hovering = $0 }
                // Apple Books entries carry the author in `preview`.
                .help(entry.id.hasPrefix("applebooks:") && !entry.preview.isEmpty
                      ? "\(entry.fileName) — \(entry.preview)" : entry.fileName)

            HStack(spacing: 6) {
                progressLabel
                Spacer(minLength: 4)
                Menu {
                    Button("Open", action: open).disabled(!available)
                    if let openInBooks {
                        Button("Open in Apple Books", action: openInBooks)
                    }
                    if let fp = entry.filePath, available {
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: fp)])
                        }
                    }
                    if let remove {
                        Divider()
                        Button("Remove from Library", role: .destructive, action: remove)
                    }
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Color(hex: theme.textMuted))
                        .frame(width: 22, height: 16)
                        .contentShape(Rectangle())
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
            }
        }
    }

    private var coverView: some View {
        Group {
            if let cover {
                Image(nsImage: cover)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
            } else {
                GeneratedCover(title: entry.fileName, kind: entry.fileType)
                    .aspectRatio(2.0 / 3.0, contentMode: .fit)
            }
        }
        .overlay { BookSurface() }
        .clipShape(Self.coverShape)
        .overlay(Self.coverShape.strokeBorder(Color.black.opacity(theme.isDark ? 0.4 : 0.14), lineWidth: 0.5))
        // Contact shadow (where the book meets the shelf) under a wider,
        // softer one; hovering lifts the book and spreads the shadow.
        .shadow(color: .black.opacity(theme.isDark ? 0.6 : 0.28), radius: hovering ? 2.5 : 1.5,
                x: 0, y: hovering ? 2 : 1)
        .shadow(color: .black.opacity(theme.isDark ? 0.5 : 0.2), radius: hovering ? 18 : 10,
                x: hovering ? 4 : 3, y: hovering ? 14 : 8)
        .scaleEffect(hovering ? 1.03 : 1, anchor: .bottom)
        .offset(y: hovering ? -3 : 0)
        .animation(.spring(response: 0.28, dampingFraction: 0.8), value: hovering)
        // Dimmed rather than faded, so the shadow doesn't show through.
        .saturation(available ? 1 : 0.4)
        .brightness(available ? 0 : (theme.isDark ? -0.1 : 0.1))
    }

    /// Tighter at the spine than at the fore-edge, like a bound board.
    private static let coverShape = UnevenRoundedRectangle(
        topLeadingRadius: 1.5, bottomLeadingRadius: 1.5,
        bottomTrailingRadius: 4, topTrailingRadius: 4, style: .continuous)

    @ViewBuilder
    private var progressLabel: some View {
        switch status {
        case .unavailable(let reason):
            Text(reason)
                .font(.system(size: 11))
                .foregroundStyle(Color(hex: theme.textMuted))
        case .new:
            Text("NEW")
                .font(.system(size: 10, weight: .bold))
                .kerning(0.4)
                .foregroundStyle(Color(hex: theme.accent))
                .padding(.horizontal, 7)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color(hex: theme.accent).opacity(0.16)))
        case .finished:
            Text("Finished")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color(hex: theme.textMuted))
        case .reading(let progress):
            HStack(spacing: 6) {
                // A thin bar, as Books draws it, beside the percentage.
                Capsule().fill(Color(hex: theme.textMuted).opacity(0.25))
                    .frame(width: 34, height: 3)
                    .overlay(alignment: .leading) {
                        Capsule().fill(Color(hex: theme.textMuted))
                            .frame(width: max(3, 34 * progress))
                    }
                Text("\(max(1, Int(progress * 100)))%")
                    .font(.system(size: 11, weight: .medium).monospacedDigit())
                    .foregroundStyle(Color(hex: theme.textMuted))
            }
        }
    }
}

/// What the line under a cover says.
enum LibraryBookStatus: Equatable {
    case unavailable(String)
    case new
    case reading(Double)
    case finished

    init(_ entry: BookmarkEntry, available: Bool) {
        if !available { self = .unavailable("Unavailable") }
        else if entry.sentenceIndex == 0 { self = .new }
        else if entry.progress >= 0.995 { self = .finished }
        else { self = .reading(entry.progress) }
    }

    init(_ book: AppleBook) {
        if book.isProtected { self = .unavailable("Apple Books only") }
        else if !book.isAvailable { self = .unavailable("Unavailable") }
        else if book.isFinished || book.progress >= 0.995 { self = .finished }
        else if book.progress <= 0 { self = .new }
        else { self = .reading(book.progress) }
    }
}

/// A full-width card on the home screen inviting the user to connect Apple
/// Books (or explaining how, once macOS has said no), with Books' own icon.
private struct AppleBooksBanner: View {
    let theme: ReaderTheme
    let title: String
    let message: String
    let primary: (String, () -> Void)
    let secondary: (String, () -> Void)

    static let booksIcon: NSImage? = NSWorkspace.shared
        .urlForApplication(withBundleIdentifier: "com.apple.iBooksX")
        .map { NSWorkspace.shared.icon(forFile: $0.path) }

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            Group {
                if let icon = AppleBooksBanner.booksIcon {
                    Image(nsImage: icon).resizable().interpolation(.high)
                } else {
                    Image(systemName: "books.vertical.fill")
                        .font(.system(size: 26))
                        .foregroundStyle(.orange)
                }
            }
            .frame(width: 48, height: 48)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Color(hex: theme.text))
                Text(message)
                    .font(.system(size: 13))
                    .foregroundStyle(Color(hex: theme.textMuted))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 12)

            Button(action: secondary.1) {
                Text(secondary.0)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Color(hex: theme.textMuted))
            }
            .buttonStyle(.plain)

            Button(action: primary.1) {
                Text(primary.0)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color(hex: theme.bg))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 7)
                    .background(Capsule().fill(Color(hex: theme.text)))
            }
            .buttonStyle(.plain)
            .fixedSize()
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(hex: theme.dropBg))
                .overlay(RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(Color(hex: theme.dropBorder).opacity(0.7)))
        )
    }
}

/// Light and shade laid over a cover so it reads as a bound book rather
/// than a flat picture: the rounded spine, the hinge groove where the board
/// folds, a soft sheen across the face, and catch-lights on the top and
/// fore-edge.
private struct BookSurface: View {
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let hinge = w * 0.055
            ZStack(alignment: .leading) {
                // Spine curvature: darkest at the very edge.
                LinearGradient(stops: [
                    .init(color: .black.opacity(0.42), location: 0),
                    .init(color: .black.opacity(0.10), location: 0.45),
                    .init(color: .white.opacity(0.08), location: 0.8),
                    .init(color: .clear, location: 1),
                ], startPoint: .leading, endPoint: .trailing)
                .frame(width: hinge)

                // Hinge: a pressed groove, a dark crease with a lit lip
                // beside it, feathered so it sits *in* the cover.
                HStack(spacing: 0) {
                    LinearGradient(colors: [.clear, .black.opacity(0.32)],
                                   startPoint: .leading, endPoint: .trailing)
                        .frame(width: 2.5)
                    Rectangle().fill(.black.opacity(0.38)).frame(width: 1)
                    LinearGradient(colors: [.white.opacity(0.22), .clear],
                                   startPoint: .leading, endPoint: .trailing)
                        .frame(width: 3.5)
                }
                .blur(radius: 0.4)
                .offset(x: hinge)

                // Sheen across the face, falling off toward the foot.
                LinearGradient(stops: [
                    .init(color: .white.opacity(0.14), location: 0),
                    .init(color: .white.opacity(0.03), location: 0.35),
                    .init(color: .clear, location: 0.6),
                    .init(color: .black.opacity(0.10), location: 1),
                ], startPoint: .topLeading, endPoint: .bottomTrailing)
                .blendMode(.softLight)

                // Catch-lights: top edge and fore-edge.
                VStack(spacing: 0) {
                    Rectangle().fill(.white.opacity(0.18)).frame(height: 0.75)
                    Spacer(minLength: 0)
                }
                HStack(spacing: 0) {
                    Spacer(minLength: 0)
                    LinearGradient(colors: [.clear, .white.opacity(0.12)],
                                   startPoint: .leading, endPoint: .trailing)
                        .frame(width: 3)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

/// The fallback cover for anything with no artwork: a cloth-bound colour
/// picked from the title (so a book keeps its colour between launches), the
/// title set in a serif, and a small label for the kind of document.
private struct GeneratedCover: View {
    let title: String
    let kind: String

    private static let palette: [(bg: String, ink: String)] = [
        ("#1f3a5f", "#f3e9d2"), ("#6b1f2a", "#f5e6c8"), ("#24483a", "#efe6cf"),
        ("#8a5a14", "#fbf1dc"), ("#3d4451", "#ece7dc"), ("#4a2c55", "#f1e4ef"),
        ("#2f5d62", "#eef2e6"), ("#7a3b1d", "#f8ead6"), ("#1d1d1f", "#e9dfc9"),
    ]

    private var colors: (bg: Color, ink: Color) {
        // Stable across launches, unlike `String.hashValue`.
        let h = title.unicodeScalars.reduce(UInt32(5381)) { ($0 &* 33) &+ $1.value }
        let p = Self.palette[Int(h % UInt32(Self.palette.count))]
        return (Color(hex: p.bg), Color(hex: p.ink))
    }

    private var label: String {
        switch kind {
        case "pdf": return "PDF"
        case "url": return "ARTICLE"
        case "text": return "TEXT"
        case "ocr": return "SCAN"
        case "docx": return "DOCUMENT"
        default: return ""
        }
    }

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let c = colors
            ZStack {
                LinearGradient(colors: [c.bg.opacity(0.92), c.bg], startPoint: .top, endPoint: .bottom)
                // Inset rule, like a blind-stamped border.
                RoundedRectangle(cornerRadius: 1)
                    .strokeBorder(c.ink.opacity(0.35), lineWidth: 0.75)
                    .padding(w * 0.06)
                VStack(spacing: w * 0.05) {
                    Spacer(minLength: 0)
                    Text(title)
                        .font(.system(size: w * 0.115, weight: .semibold, design: .serif))
                        .multilineTextAlignment(.center)
                        .lineLimit(5)
                        .minimumScaleFactor(0.6)
                        .foregroundStyle(c.ink)
                    Rectangle().fill(c.ink.opacity(0.5)).frame(width: w * 0.18, height: 0.75)
                    Spacer(minLength: 0)
                    if !label.isEmpty {
                        Text(label)
                            .font(.system(size: w * 0.055, weight: .semibold))
                            .kerning(w * 0.012)
                            .foregroundStyle(c.ink.opacity(0.7))
                    }
                }
                .padding(.horizontal, w * 0.13)
                .padding(.vertical, w * 0.16)
            }
        }
    }
}

// MARK: - Document reader

struct DocumentReaderView: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @State private var isFindVisible = false
    @State private var isAskVisible = false
    @State private var findQuery = ""
    @State private var findCount = 0
    @State private var findCurrent = 0
    @FocusState private var findFocused: Bool
    /// Every jump the reader makes — bookmark resume, an outline click, a find
    /// hit, the scroll-to-playhead button — goes through this one request. The
    /// book is a single text view, so a jump is just "put this range on
    /// screen"; the old step-by-step scroll walk existed only to coax a
    /// LazyVStack into mounting rows it hadn't reached, and is gone with it.
    @State private var scrollRequest = BookScrollRequest()

    private var theme: ReaderTheme { reader.theme }

    var body: some View {
        Group {
            if let doc = reader.document {
                GeometryReader { geo in
                    BookTextView(
                        document: doc,
                        theme: theme,
                        fontScale: CGFloat(reader.fontSize),
                        columnWidth: columnWidth(for: geo.size.width),
                        activeSentenceID: doc.sentences.indices.contains(reader.currentIndex)
                            ? reader.currentIndex : nil,
                        activeWordFraction: reader.activeWordFraction,
                        isPlaying: reader.isSpeaking,
                        searchQuery: isFindVisible ? findQuery : "",
                        searchCurrent: findCurrent,
                        scrollRequest: scrollRequest,
                        onSearchCount: { count in
                            // Reported from inside the AppKit update pass, so
                            // the SwiftUI state change is deferred a turn
                            // rather than mutating state mid-render.
                            DispatchQueue.main.async {
                                findCount = count
                                if findCurrent >= count { findCurrent = 0 }
                                if count > 0 { bumpScroll() }
                            }
                        },
                        onActivateSentence: { reader.playFrom($0) },
                        onZoom: { reader.fontSize = Double($0) }
                    )
                }
                .ignoresSafeArea(edges: .bottom)
                .onAppear { jump(toSentence: reader.currentIndex) }
                .onAppear { BookAskController.shared.bookLoaded(doc) }
                .onChange(of: doc.sourceID) { _, _ in jump(toSentence: reader.currentIndex) }
                .onChange(of: doc.sourceID) { _, _ in BookAskController.shared.bookLoaded(doc) }
                .onChange(of: reader.currentIndex) { _, newIndex in
                    // While a search is up, find navigation owns the scroll
                    // position — otherwise every hit would be yanked back to
                    // the playhead the moment the next sentence started.
                    guard !isSearchActive, reader.isSpeaking || reader.state == .ready else { return }
                    jump(toSentence: newIndex)
                }
                .onChange(of: reader.state) { old, new in
                    guard !isSearchActive else { return }
                    if (old == .paused && new == .playing) || (old == .loadingDoc && new == .ready) {
                        jump(toSentence: reader.currentIndex)
                    }
                }
                .onChange(of: findCurrent) { _, _ in bumpScroll() }
                .onReceive(NotificationCenter.default.publisher(for: .scrollToCurrentSentence)) { _ in
                    jump(toSentence: reader.currentIndex)
                }
                .onReceive(NotificationCenter.default.publisher(for: .scrollToBlock)) { note in
                    if let bi = note.object as? Int {
                        scrollRequest = BookScrollRequest(token: scrollRequest.token + 1,
                                                          sentenceID: nil, blockIndex: bi)
                    }
                }
                .overlay(alignment: .bottomTrailing) { scrollToPlayheadButton }
            } else {
                LandingView()
            }
        }
        .background(Color(hex: theme.bg))
        .overlay {
            // Hidden ⌘F trigger — registers the keyboard shortcut for the window
            Button { isFindVisible = true; findFocused = true } label: { EmptyView() }
                .keyboardShortcut("f", modifiers: .command)
                .hidden()
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                findToolbarField
            }
            ToolbarItem(placement: .primaryAction) {
                Button("Ask") { isAskVisible.toggle() }
                .help("Ask this book")
                .popover(isPresented: $isAskVisible, arrowEdge: .bottom) {
                    if let document = reader.document {
                        BookAskPanel(document: document) { sentence in
                            isAskVisible = false
                            jump(toSentence: sentence)
                        }
                    }
                }
            }
        }
    }

    /// A measured column rather than the full window: past roughly 90
    /// characters a line is hard to track back from, which is why books are
    /// set in a column and not across the page.
    private func columnWidth(for available: CGFloat) -> CGFloat {
        min(680, max(320, available - 120))
    }

    @ViewBuilder
    private var scrollToPlayheadButton: some View {
        if reader.state == .paused && reader.progress > 0 && reader.progress < 1 {
            Button {
                NotificationCenter.default.post(name: .scrollToCurrentSentence, object: nil)
            } label: {
                Image(systemName: "target")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(Color(hex: theme.barIconColor))
                    .frame(width: 44, height: 44)
                    .background(Color(hex: theme.barBg), in: Circle())
                    .overlay(Circle().strokeBorder(Color(hex: theme.barBorder), lineWidth: 1))
                    .shadow(radius: 5)
            }
            .buttonStyle(.plain)
            .padding(.trailing, 26)
            .padding(.bottom, 96)
        }
    }

    // MARK: - Scrolling

    private func jump(toSentence id: Int) {
        guard let doc = reader.document, doc.sentences.indices.contains(id) else { return }
        scrollRequest = BookScrollRequest(token: scrollRequest.token + 1,
                                          sentenceID: id, blockIndex: nil)
    }

    /// Re-issues the current find hit as a scroll target. The hit's range lives
    /// in the text view (it does the searching), so the request only has to say
    /// "move", not where to.
    private func bumpScroll() {
        scrollRequest = BookScrollRequest(token: scrollRequest.token + 1,
                                          sentenceID: nil, blockIndex: nil)
    }

    private var isSearchActive: Bool {
        isFindVisible && !findQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func findNext() {
        guard findCount > 0 else { return }
        findCurrent = (findCurrent + 1) % findCount
    }

    private func findPrev() {
        guard findCount > 0 else { return }
        findCurrent = (findCurrent - 1 + findCount) % findCount
    }

    private func closeFind() {
        isFindVisible = false
        findQuery = ""
        findCount = 0
        findCurrent = 0
    }

    // MARK: - Find (⌘F) — lives in the toolbar next to the Home button

    @ViewBuilder
    private var findToolbarField: some View {
        if isFindVisible {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").foregroundStyle(Color(hex: theme.textMuted))
                TextField("Find", text: $findQuery)
                    .textFieldStyle(.plain)
                    .focused($findFocused)
                    .frame(width: 160)
                    .onChange(of: findQuery) { _, _ in findCurrent = 0 }
                    .onSubmit { findNext() }
                if !findQuery.isEmpty {
                    Text(findCount == 0 ? "No results" : "\(findCurrent + 1)/\(findCount)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(Color(hex: theme.textMuted))
                    Button { findPrev() } label: {
                        Image(systemName: "chevron.up").font(.system(size: 11, weight: .bold))
                    }.buttonStyle(.plain).disabled(findCount == 0)
                    Button { findNext() } label: {
                        Image(systemName: "chevron.down").font(.system(size: 11, weight: .bold))
                    }.buttonStyle(.plain).disabled(findCount == 0)
                }
                Button { closeFind() } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(Color(hex: theme.textMuted))
                }.buttonStyle(.plain)
            }
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color(hex: theme.dropBorder).opacity(0.5)))
            .onExitCommand { closeFind() }
        } else {
            Button { isFindVisible = true; findFocused = true } label: {
                Image(systemName: "magnifyingglass")
            }
            .help("Find (⌘F)")
        }
    }
}

extension Notification.Name {
    static let scrollToCurrentSentence = Notification.Name("scrollToCurrentSentence")
    static let scrollToBlock = Notification.Name("scrollToBlock")
}

// MARK: - Outline sidebar (TOC)

struct OutlineSidebar: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @State private var filter = ""

    private var theme: ReaderTheme { reader.theme }

    private var entries: [OutlineEntry] {
        let all = reader.document?.outline ?? []
        let q = filter.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return all }
        return all.filter { $0.title.localizedCaseInsensitiveContains(q) }
    }

    /// Block the playhead is in, used to mark the section being read.
    private var currentBlock: Int? {
        guard let doc = reader.document, doc.sentences.indices.contains(reader.currentIndex) else { return nil }
        return doc.sentences[reader.currentIndex].blockIndex
    }

    /// The deepest outline entry at or before the playhead — i.e. the
    /// section currently being read, not merely the nearest title.
    private var activeEntryID: UUID? {
        guard let block = currentBlock else { return nil }
        return (reader.document?.outline ?? []).last { $0.blockIndex <= block }?.id
    }

    /// Publisher TOCs nest arbitrarily deep, but the shallowest level in a
    /// given book isn't always 1 (some books start at 2). Normalising against
    /// the minimum keeps indentation tight instead of pushing everything to
    /// the right.
    private var baseLevel: Int {
        (reader.document?.outline ?? []).map(\.level).min() ?? 1
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("CONTENTS")
                    .font(.system(size: 12, weight: .bold))
                    .kerning(0.8)
                    .foregroundStyle(Color(hex: theme.textMuted))
                Spacer()
                Button {
                    withAnimation { reader.outlineVisible = false }
                } label: {
                    Image(systemName: "sidebar.leading")
                        .foregroundStyle(Color(hex: theme.textMuted))
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 14)
            .padding(.top, 14)
            .padding(.bottom, 10)

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 11))
                    .foregroundStyle(Color(hex: theme.textMuted))
                TextField("Filter", text: $filter)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12))
                    .foregroundStyle(Color(hex: theme.text))
                if !filter.isEmpty {
                    Button { filter = "" } label: {
                        Image(systemName: "xmark.circle.fill").font(.system(size: 11))
                            .foregroundStyle(Color(hex: theme.textMuted))
                    }.buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(Color(hex: theme.inputBg), in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6)
                .strokeBorder(Color(hex: theme.inputBorder).opacity(0.6)))
            .padding(.horizontal, 12)
            .padding(.bottom, 10)

            Divider().overlay(Color(hex: theme.menuBorder))

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(entries) { entry in
                            outlineRow(entry)
                                .id(entry.id)
                        }
                        if entries.isEmpty {
                            Text(filter.isEmpty ? "No table of contents" : "No matches")
                                .font(.caption)
                                .foregroundStyle(Color(hex: theme.textMuted))
                                .padding(14)
                        }
                    }
                    .padding(.vertical, 6)
                }
                .onChange(of: activeEntryID) { _, new in
                    guard let new, filter.isEmpty else { return }
                    withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(new, anchor: .center) }
                }
            }
        }
        .background(Color(hex: theme.menuBg))
    }

    @ViewBuilder
    private func outlineRow(_ entry: OutlineEntry) -> some View {
        let depth = max(0, entry.level - baseLevel)
        let isActive = entry.id == activeEntryID
        Button {
            NotificationCenter.default.post(name: .scrollToBlock, object: entry.blockIndex)
            reader.outlineVisible = false
        } label: {
            HStack(alignment: .top, spacing: 8) {
                // A thin rail marks the current section without shifting the
                // text, so the list doesn't jump as playback moves.
                Rectangle()
                    .fill(isActive ? Color(hex: theme.accent) : .clear)
                    .frame(width: 2.5)
                Text(entry.title)
                    .font(.system(size: depth == 0 ? 13 : (depth == 1 ? 12.5 : 12),
                                  weight: depth == 0 ? .semibold : .regular))
                    .kerning(depth == 0 ? 0.2 : 0)
                    .lineLimit(3)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .foregroundStyle(Color(hex: isActive ? theme.headerColor
                                                         : (depth == 0 ? theme.text : theme.textMuted)))
                    .padding(.leading, CGFloat(depth) * 13)
                Spacer(minLength: 0)
            }
            .padding(.vertical, depth == 0 ? 6 : 4)
            .padding(.trailing, 12)
            .background(isActive ? Color(hex: theme.accent).opacity(0.10) : .clear)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(entry.title)
    }
}

// MARK: - Bottom bar cockpit

struct BottomBar: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @State private var showSettings = false
    @State private var showSpeed = false
    @State private var showThemes = false

    private var theme: ReaderTheme { reader.theme }

    var body: some View {
        GlassEffectContainer(spacing: 0) {
            HStack(spacing: 16) {
                // Menu / settings
                PillButton {
                    showSettings.toggle()
                } label: {
                    Image(systemName: "line.3.horizontal")
                }
                .popover(isPresented: $showSettings, arrowEdge: .bottom) { SettingsPopover() }

                // Speed
                PillButton {
                    showSpeed.toggle()
                } label: {
                    Text(Self.formatSpeed(reader.speed))
                        .font(.system(size: 13, weight: .bold).monospacedDigit())
                }
                .popover(isPresented: $showSpeed, arrowEdge: .bottom) {
                    speedPopover
                }

                // Play / pause — shows a spinner while buffering (generating)
                Button { reader.togglePlayPause() } label: {
                    Group {
                        if reader.state == .generating {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .tint(.white)
                                .scaleEffect(0.85)
                        } else {
                            Image(systemName: primaryIcon)
                                .font(.system(size: 17, weight: .bold))
                        }
                    }
                    .foregroundStyle(.white)
                    .frame(width: 46, height: 46)
                    .background(Color(hex: "#2563eb"), in: Circle())
                    .shadow(color: .black.opacity(0.25), radius: 4, y: 1)
                }
                .buttonStyle(.plain)
                .pointerStyle(.link)
                .disabled(!engineReady || (reader.document?.sentences.isEmpty ?? true))
                .opacity(engineReady && !(reader.document?.sentences.isEmpty ?? true) ? 1 : 0.55)
                .help(primaryHelp)

                // Mini player toggle
                PillButton {
                    reader.miniPlayerVisible.toggle()
                } label: {
                    Image(systemName: reader.miniPlayerVisible
                            ? "rectangle.on.rectangle.fill" : "rectangle.on.rectangle")
                }
                .help("Toggle floating mini player")

                // Theme picker
                PillButton {
                    showThemes.toggle()
                } label: {
                    Image(systemName: "paintpalette")
                }
                .popover(isPresented: $showThemes, arrowEdge: .bottom) { themeGrid }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            // Apple Liquid Glass (macOS 26): adaptive glass capsule with hover
            // interactivity that picks up the content behind it.
            .glassEffect(.regular.interactive(), in: .capsule)
        }
        .shadow(color: .black.opacity(0.18), radius: 10, y: 3)
        .padding(.bottom, 16)
    }

    private var engineReady: Bool { EngineHub.shared.ready }

    private var primaryIcon: String {
        switch reader.state {
        case .playing, .generating: return "pause.fill"
        case .loadingDoc:           return "hourglass"
        default:                    return "play.fill"
        }
    }

    private var primaryHelp: String {
        switch reader.state {
        case .playing, .generating: return "Pause"
        case .paused:               return "Resume"
        default:                    return "Play"
        }
    }

    /// "1×", "1.25×", "1.5×" — `%g` keeps every significant digit
    /// (`%.2g` had been rounding 1.25 to "1.2").
    static func formatSpeed(_ s: Double) -> String {
        String(format: "%g×", (s * 100).rounded() / 100)
    }

    private var speedPopover: some View {
        VStack(alignment: .leading, spacing: 8) {
            PopoverSectionLabel("Speed", theme: theme)
                .padding(.horizontal, 8)
            VStack(spacing: 2) {
                ForEach(ReaderController.speedChoices, id: \.self) { s in
                    PopoverChoiceRow(theme: theme, selected: abs(s - reader.speed) < 0.001) {
                        reader.speed = s
                        showSpeed = false
                    } label: {
                        Text(Self.formatSpeed(s))
                            .font(.system(size: 14, weight: .medium).monospacedDigit())
                    }
                }
            }
        }
        .padding(8)
        .padding(.top, 4)
        .frame(width: 148)
        .background(Color(hex: theme.menuBg))
    }

    /// Aa swatch grid matching the BlaBla theme popover.
    private var themeGrid: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("THEME")
                .font(.system(size: 11, weight: .bold))
                .kerning(1)
                .foregroundStyle(Color(hex: reader.theme.textMuted))
            LazyVGrid(columns: [GridItem(.fixed(78)), GridItem(.fixed(78)), GridItem(.fixed(78))], spacing: 10) {
                ForEach(ReaderTheme.all, id: \.name) { t in
                    Button {
                        reader.themeName = t.name
                        showThemes = false
                    } label: {
                        VStack(spacing: 2) {
                            Text("Aa")
                                .font(.system(size: 20, weight: .bold, design: .serif))
                                .foregroundStyle(Color(hex: t.text))
                            Text(t.name)
                                .font(.system(size: 10))
                                .foregroundStyle(Color(hex: t.textMuted))
                        }
                        .frame(width: 76, height: 58)
                        .background(RoundedRectangle(cornerRadius: 10).fill(Color(hex: t.bg)))
                        .overlay(RoundedRectangle(cornerRadius: 10)
                            .strokeBorder(t.name == reader.themeName
                                          ? Color(hex: t.text).opacity(0.85)
                                          : Color(hex: t.dropBorder).opacity(0.6),
                                          lineWidth: t.name == reader.themeName ? 2 : 1))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(14)
        .background(Color(hex: reader.theme.menuBg))
    }
}

private struct PillButton<Label: View>: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    let action: () -> Void
    @ViewBuilder let label: () -> Label

    private var iconColor: Color {
        // Liquid Glass shows the page behind it, so icon contrast must follow
        // the page theme (not the old solid barBg). Dark pages need light icons.
        reader.theme.isDark
            ? Color.white.opacity(0.88)
            : Color(hex: reader.theme.text).opacity(0.82)
    }

    var body: some View {
        Button(action: action) {
            label()
                .font(.system(size: 14))
                .foregroundStyle(iconColor)
                .frame(minWidth: 32, minHeight: 32)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .pointerStyle(.link)
    }
}

// MARK: - Settings popover

struct SettingsPopover: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @ObservedObject private var hub = EngineHub.shared

    private var theme: ReaderTheme { reader.theme }

    /// Display names, grouped as the voice menu shows them.
    private static let voiceGroups: [(title: String?, voices: [(id: String, name: String)])] = [
        (nil, [("daisy", "Daisy"), ("david-deep", "David")]),
        ("Female", (1...5).map { ("F\($0)", "Female \($0)") }),
        ("Male", (1...5).map { ("M\($0)", "Male \($0)") }),
    ]

    private static func voiceName(_ id: String) -> String {
        voiceGroups.flatMap(\.voices).first { $0.id == id }?.name ?? id
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            nowReading
                .padding(16)

            separator

            VStack(alignment: .leading, spacing: 16) {
                voiceRow
                textSizeRow
                volumeRow
            }
            .padding(16)

            separator

            Button { reader.resetDocument() } label: {
                Label("Close Book", systemImage: "xmark.circle")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Color(hex: theme.textMuted))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Stop reading and go back to the library")
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .frame(width: 300)
        .background(Color(hex: theme.menuBg))
    }

    // MARK: Sections

    /// Where the reader is in the book, and what the player is doing.
    private var nowReading: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                PopoverSectionLabel("Progress", theme: theme)
                Spacer()
                Circle().fill(statusColor).frame(width: 7, height: 7)
                Text(statusText)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color(hex: theme.textMuted))
            }
            HStack(alignment: .firstTextBaseline) {
                Text("\(Int(reader.progress * 100))%")
                    .font(.system(size: 22, weight: .semibold).monospacedDigit())
                    .foregroundStyle(Color(hex: theme.text))
                Spacer()
                if let total = reader.document?.sentences.count {
                    Text("Sentence \(reader.currentIndex + 1) of \(total)")
                        .font(.system(size: 12).monospacedDigit())
                        .foregroundStyle(Color(hex: theme.textMuted))
                }
            }
            Capsule()
                .fill(Color(hex: theme.text).opacity(0.1))
                .frame(height: 4)
                .overlay(alignment: .leading) {
                    GeometryReader { geo in
                        Capsule().fill(Color(hex: theme.accent))
                            .frame(width: max(4, geo.size.width * reader.progress))
                    }
                }
        }
    }

    private var voiceRow: some View {
        settingRow("Voice") {
            HStack(spacing: 8) {
                Menu {
                    ForEach(Self.voiceGroups.indices, id: \.self) { g in
                        let group = Self.voiceGroups[g]
                        if let title = group.title {
                            Section(title) { voiceButtons(group.voices) }
                        } else {
                            voiceButtons(group.voices)
                            Divider()
                        }
                    }
                } label: {
                    HStack {
                        Text(Self.voiceName(reader.voice))
                            .font(.system(size: 13, weight: .medium))
                        Spacer(minLength: 4)
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(Color(hex: theme.textMuted))
                    }
                    .foregroundStyle(Color(hex: theme.text))
                    .padding(.horizontal, 10)
                    .frame(height: 28)
                    .background(fieldBackground)
                    .contentShape(Rectangle())
                }
                .menuStyle(.button)
                .buttonStyle(.plain)
                .menuIndicator(.hidden)

                Button { reader.testVoice() } label: {
                    Image(systemName: "speaker.wave.2.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Color(hex: theme.text))
                        .frame(width: 28, height: 28)
                        .background(fieldBackground)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(!hub.ready)
                .help("Preview this voice")
            }
        }
    }

    private var textSizeRow: some View {
        let range = BookTextView.fontScaleRange
        return settingRow("Text Size") {
            HStack(spacing: 0) {
                stepButton(systemImage: "textformat.size.smaller",
                           disabled: reader.fontSize <= Double(range.lowerBound) + 0.001) {
                    reader.fontSize = max(Double(range.lowerBound), reader.fontSize - 0.05)
                }
                Text("\(Int((reader.fontSize * 100).rounded()))%")
                    .font(.system(size: 12, weight: .medium).monospacedDigit())
                    .foregroundStyle(Color(hex: theme.text))
                    .frame(maxWidth: .infinity)
                stepButton(systemImage: "textformat.size.larger",
                           disabled: reader.fontSize >= Double(range.upperBound) - 0.001) {
                    reader.fontSize = min(Double(range.upperBound), reader.fontSize + 0.05)
                }
            }
            .frame(height: 28)
            .background(fieldBackground)
        }
    }

    private var volumeRow: some View {
        settingRow("Volume") {
            HStack(spacing: 8) {
                Image(systemName: "speaker.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(Color(hex: theme.textMuted))
                Slider(value: $reader.volume, in: 0...1)
                    .controlSize(.small)
                    .tint(Color(hex: theme.accent))
                Image(systemName: "speaker.wave.3.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(Color(hex: theme.textMuted))
            }
            .help("\(Int(reader.volume * 100))%")
        }
    }

    // MARK: Pieces

    private func voiceButtons(_ voices: [(id: String, name: String)]) -> some View {
        ForEach(voices, id: \.id) { v in
            Button {
                reader.voice = v.id
            } label: {
                if v.id == reader.voice {
                    Label(v.name, systemImage: "checkmark")
                } else {
                    Text(v.name)
                }
            }
        }
    }

    private var separator: some View {
        Rectangle().fill(Color(hex: theme.text).opacity(0.08)).frame(height: 1)
    }

    private var fieldBackground: some View {
        RoundedRectangle(cornerRadius: 7, style: .continuous)
            .fill(Color(hex: theme.inputBg))
            .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous)
                .strokeBorder(Color(hex: theme.inputBorder).opacity(0.6)))
    }

    private func settingRow<Content: View>(_ title: String,
                                           @ViewBuilder content: () -> Content) -> some View {
        HStack(spacing: 12) {
            Text(title)
                .font(.system(size: 13))
                .foregroundStyle(Color(hex: theme.textMuted))
                .lineLimit(1)
                .frame(width: 70, alignment: .leading)
            content()
        }
    }

    private func stepButton(systemImage: String, disabled: Bool,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Color(hex: theme.text))
                .frame(width: 36, height: 28)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.35 : 1)
    }

    private var statusColor: Color {
        if hub.failed { return .red }
        if !hub.ready { return .orange }
        switch reader.state {
        case .playing, .generating: return .green
        case .failed: return .red
        default: return Color(hex: theme.textMuted).opacity(0.6)
        }
    }

    /// The player's state once the engine is up; until then, the engine's.
    private var statusText: String {
        if hub.failed { return "Engine failed" }
        if !hub.ready { return hub.statusText.isEmpty ? "Loading voice…" : hub.statusText }
        return stateLabel
    }

    private var stateLabel: String {
        switch reader.state {
        case .empty: return "Idle"
        case .ready: return "Ready"
        case .generating: return "Buffering"
        case .playing: return "Playing"
        case .paused: return "Paused"
        case .loadingDoc: return "Loading"
        case .failed: return "Error"
        }
    }
}

// MARK: - Popover pieces

/// Small caps heading inside a popover ("SPEED", "PROGRESS").
private struct PopoverSectionLabel: View {
    let title: String
    let theme: ReaderTheme

    init(_ title: String, theme: ReaderTheme) {
        self.title = title
        self.theme = theme
    }

    var body: some View {
        Text(title.uppercased())
            .font(.system(size: 10, weight: .bold))
            .kerning(0.8)
            .foregroundStyle(Color(hex: theme.textMuted))
    }
}

/// One choice in a popover list: a rounded highlight on hover, a tinted
/// fill and checkmark when selected.
private struct PopoverChoiceRow<Label: View>: View {
    let theme: ReaderTheme
    let selected: Bool
    let action: () -> Void
    @ViewBuilder let label: () -> Label
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack {
                label()
                    .foregroundStyle(Color(hex: theme.text))
                Spacer(minLength: 8)
                if selected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(Color(hex: theme.accent))
                }
            }
            .padding(.horizontal, 10)
            .frame(height: 30)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(selected ? Color(hex: theme.accent).opacity(0.14)
                          : Color(hex: theme.text).opacity(hovering ? 0.07 : 0))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }
}
