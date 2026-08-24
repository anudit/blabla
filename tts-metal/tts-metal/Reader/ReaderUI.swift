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

// MARK: - Root

struct ReaderRootView: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader

    var body: some View {
        HStack(spacing: 0) {
            if reader.outlineVisible && reader.document != nil {
                OutlineSidebar()
                    .frame(width: 250)
                    .transition(.move(edge: .leading))
            }
            content
        }
        .frame(minWidth: 760, minHeight: 560)
        .background(Color(hex: reader.theme.bg))
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    reader.stopPlayback()
                    reader.resetDocument()
                } label: {
                    Label("Home", systemImage: "house.fill")
                }
                .help("Back to home — stop playback")
                .disabled(reader.state == .empty)
                .opacity(reader.state == .empty ? 0.45 : 1)
            }
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

    var body: some View {
        ScrollView {
            VStack(spacing: 26) {
                card
                if showTextarea { textarea }
                historySection
            }
            .padding(.horizontal, 28)
            .padding(.top, 46)
            .padding(.bottom, 110)
            .frame(maxWidth: 680)
            .frame(maxWidth: .infinity)
        }
        .background(Color(hex: theme.bg))
        .onAppear(perform: registerPasteHandler)
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

    private var historySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "clock")
                    .font(.system(size: 12))
                Text("CONTINUE READING")
                    .font(.system(size: 12, weight: .bold))
                    .kerning(0.8)
            }
            .foregroundStyle(Color(hex: theme.textMuted))

            ForEach(bookmarks.entries) { entry in
                historyCard(entry)
            }

            if bookmarks.entries.contains(where: { $0.url == nil }) {
                Text("Drop the same file again to resume from where you left off")
                    .font(.system(size: 13))
                    .foregroundStyle(Color(hex: theme.textMuted))
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private func historyCard(_ entry: BookmarkEntry) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: iconFor(entry))
                .foregroundStyle(Color(hex: theme.textMuted))
                .frame(width: 22)
                .padding(.top, 3)

            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(entry.fileName)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(Color(hex: theme.text))
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Spacer()
                    Text(entry.relativeTimeString())
                        .font(.system(size: 12))
                        .foregroundStyle(Color(hex: theme.textMuted))
                }
                Text("\"\(entry.preview)\"")
                    .font(.system(size: 13).italic())
                    .foregroundStyle(Color(hex: theme.textMuted))
                    .lineLimit(1)
                    .truncationMode(.tail)

                HStack(spacing: 8) {
                    ProgressView(value: entry.progress)
                        .progressViewStyle(.linear)
                        .tint(Color(hex: "#d4a017"))
                    Text("\(Int(entry.progress * 100))%")
                        .font(.system(size: 12))
                        .foregroundStyle(Color(hex: theme.textMuted))
                    Text(entry.fileType.uppercased())
                        .font(.system(size: 11, weight: .bold))
                        .kerning(0.5)
                        .foregroundStyle(Color(hex: theme.textMuted))
                }
            }

            Button {
                bookmarks.remove(id: entry.id)
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 12))
                    .foregroundStyle(Color(hex: theme.textMuted))
            }
            .buttonStyle(.plain)
            .padding(.top, 3)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(hex: theme.dropBg))
                .overlay(RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(Color(hex: theme.dropBorder).opacity(0.6)))
        )
        .contentShape(RoundedRectangle(cornerRadius: 14))
        .onTapGesture {
            PerfLog.log("resume tapped: \(entry.fileName)")
            // Auto-reopen: prefer the saved file path (resume at saved line), else URL.
            if let fp = entry.filePath, FileManager.default.fileExists(atPath: fp) {
                reader.loadFileURL(URL(fileURLWithPath: fp))
            } else if let u = entry.url {
                reader.loadURL(u)
            }
        }
    }

    // MARK: - Actions

    private func iconFor(_ e: BookmarkEntry) -> String {
        switch e.fileType {
        case "pdf": return "book.pages"
        case "epub": return "book"
        case "mobi": return "text.book.closed"
        case "docx": return "doc.text"
        case "url": return "link"
        case "ocr": return "eye"
        default: return "doc.plaintext"
        }
    }

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

// MARK: - Document reader

struct DocumentReaderView: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @State private var isFindVisible = false
    @State private var findQuery = ""
    @State private var findMatches: [Int] = []          // sentence ids containing query
    @State private var findCurrent = 0
    @FocusState private var findFocused: Bool
    /// Bound to the `ScrollView` via `.scrollPosition(id:)`. Unlike
    /// `ScrollViewReader.scrollTo`, this is honored even when the target row
    /// is deep inside a `LazyVStack` and hasn't been mounted/measured yet —
    /// SwiftUI defers the jump until it can resolve the id — so it's what
    /// actually restores a deep resume position. `scrollTo` retries are kept
    /// alongside as a nudge once content settles, but this is the fix for
    /// "doesn't jump to the resume line" for anything beyond the first screen.
    @State private var initialScrollTarget: String?

    private var theme: ReaderTheme { reader.theme }

    var body: some View {
        Group {
            if let doc = reader.document {
                ScrollViewReader { proxy in
                    GeometryReader { geo in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 18) {
                                findBar(proxy: proxy, doc: doc)
                                titleHeader(doc)
                                // One block per LazyVStack item (each still
                                // wrapped in its own plain VStack so FlowLayout
                                // gets a concrete width to wrap sentences
                                // against). Previously blocks were batched 40
                                // to a lazy item for fewer top-level rows, but
                                // that made scrollTo/scrollPosition unable to
                                // jump anywhere beyond the first batch —
                                // LazyVStack only mounts items near the
                                // current scroll position and can't be told
                                // to jump into a batch it hasn't reached yet.
                                // One-block-per-item is the pattern
                                // ScrollViewReader/scrollPosition are actually
                                // built to handle at book-length row counts.
                                LazyVStack(alignment: .leading, spacing: 18) {
                                    ForEach(Array(doc.blocks.enumerated()), id: \.offset) { bi, block in
                                        VStack(alignment: .leading, spacing: 18) {
                                            blockView(block, blockIndex: bi, doc: doc)
                                                .id(blockID(bi))
                                                .onAppear { PerfLog.log("block \(bi) mounted") }
                                        }
                                    }
                                }
                            }
                            .padding(.horizontal, 56)
                            .padding(.vertical, 30)
                            .frame(width: min(CGFloat(760), geo.size.width - 24))
                            .frame(maxWidth: .infinity, alignment: .center)
                        }
                        .scrollPosition(id: $initialScrollTarget, anchor: .center)
                    }
                    .onAppear { scrollToResume(proxy: proxy, doc: doc) }
                    .onChange(of: doc.sourceID) { _, _ in scrollToResume(proxy: proxy, doc: doc) }
                    .onChange(of: reader.currentIndex) { _, newIndex in
                        guard let target = scrollTargetID(forSentence: newIndex, in: doc) else { return }
                        if reader.isSpeaking || reader.state == .ready {
                            proxy.scrollTo(target, anchor: .center)
                        }
                    }
                    .onChange(of: reader.state) { old, new in
                        if old == .paused && new == .playing,
                           let target = scrollTargetID(forSentence: reader.currentIndex, in: doc) {
                            proxy.scrollTo(target, anchor: .center)
                        }
                        if old == .loadingDoc && new == .ready { scrollToResume(proxy: proxy, doc: doc) }
                    }
                    .onReceive(NotificationCenter.default.publisher(for: .scrollToCurrentSentence)) { _ in
                        // The user may have manually scrolled far from the
                        // current sentence (that's what this button is for),
                        // so the target may be unmounted — walk to it rather
                        // than a direct scrollTo. See scrollToResume.
                        if doc.sentences.indices.contains(reader.currentIndex) {
                            walkScroll(proxy: proxy, toBlock: doc.sentences[reader.currentIndex].blockIndex, from: 0)
                        }
                    }
                    .onReceive(NotificationCenter.default.publisher(for: .scrollToBlock)) { note in
                        if let bi = note.object as? Int {
                            proxy.scrollTo(blockID(bi), anchor: .top)
                        }
                    }
                }
                .safeAreaInset(edge: .bottom) { Color.clear.frame(height: 80) }
                .overlay(alignment: .bottomTrailing) {
                    if reader.state == .paused && reader.progress > 0 && reader.progress < 1 {
                        Button {
                            NotificationCenter.default.post(name: .scrollToCurrentSentence, object: nil)
                        } label: {
                            Image(systemName: "target")
                                .font(.system(size: 16, weight: .bold))
                                .foregroundStyle(.white)
                                .frame(width: 44, height: 44)
                                .background(Color(hex: theme.barBg), in: Circle())
                                .shadow(radius: 5)
                        }
                        .buttonStyle(.plain)
                        .padding(.trailing, 26)
                        .padding(.bottom, 96)
                    }
                }
            } else {
                LandingView()
            }
        }
        .background(Color(hex: theme.bg))
        .overlay(alignment: .top) {
            // Hidden Cmd+F trigger — registers the keyboard shortcut for the window
            Button { isFindVisible = true; findFocused = true } label: { EmptyView() }
                .keyboardShortcut("f", modifiers: .command)
                .hidden()
        }
    }

    // MARK: - Find (Cmd+F)

    @ViewBuilder
    private func findBar(proxy: ScrollViewProxy, doc: ReaderDocument) -> some View {
        if isFindVisible {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(Color(hex: theme.textMuted))
                TextField("Find", text: $findQuery)
                    .textFieldStyle(.plain)
                    .focused($findFocused)
                    .frame(width: 220)
                    .onChange(of: findQuery) { _, new in
                        updateFindMatches(query: new, doc: doc, proxy: proxy)
                    }
                    .onSubmit { findNext(proxy: proxy) }
                if !findQuery.isEmpty {
                    Text(findMatches.isEmpty ? "No results" : "\(findCurrent + 1)/\(findMatches.count)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(Color(hex: theme.textMuted))
                    Button { findPrev(proxy: proxy) } label: {
                        Image(systemName: "chevron.up").font(.system(size: 11, weight: .bold))
                    }.buttonStyle(.plain).disabled(findMatches.isEmpty)
                    Button { findNext(proxy: proxy) } label: {
                        Image(systemName: "chevron.down").font(.system(size: 11, weight: .bold))
                    }.buttonStyle(.plain).disabled(findMatches.isEmpty)
                }
                Button { isFindVisible = false; findQuery = ""; findMatches = [] } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(Color(hex: theme.textMuted))
                }.buttonStyle(.plain)
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color(hex: theme.dropBorder).opacity(0.5)))
            .onExitCommand { isFindVisible = false; findQuery = ""; findMatches = [] }
        }
    }

    private func updateFindMatches(query: String, doc: ReaderDocument, proxy: ScrollViewProxy) {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { findMatches = []; findCurrent = 0; return }
        findMatches = doc.sentences.filter { $0.text.localizedCaseInsensitiveContains(q) }.map(\.id)
        findCurrent = 0
        if let first = findMatches.first, let target = scrollTargetID(forSentence: first, in: doc) {
            proxy.scrollTo(target, anchor: .center)
        }
    }
    private func findNext(proxy: ScrollViewProxy) {
        guard !findMatches.isEmpty, let doc = reader.document else { return }
        findCurrent = (findCurrent + 1) % findMatches.count
        if let target = scrollTargetID(forSentence: findMatches[findCurrent], in: doc) {
            proxy.scrollTo(target, anchor: .center)
        }
    }
    private func findPrev(proxy: ScrollViewProxy) {
        guard !findMatches.isEmpty, let doc = reader.document else { return }
        findCurrent = (findCurrent - 1 + findMatches.count) % findMatches.count
        if let target = scrollTargetID(forSentence: findMatches[findCurrent], in: doc) {
            proxy.scrollTo(target, anchor: .center)
        }
    }

    private func sentenceID(_ index: Int) -> String { "s-\(index)" }
    private func blockID(_ index: Int) -> String { "b-\(index)" }

    /// `ScrollViewReader.scrollTo` can't find ids set inside `FlowLayout`
    /// (a custom `Layout`, whose subviews don't propagate anchor preferences
    /// up to the enclosing `ScrollView` the way a plain stack's children do)
    /// — sentence spans live inside `FlowingParagraph`'s `FlowLayout`, so
    /// `scrollTo(sentenceID(...))` silently no-ops. Block ids are set
    /// directly on the `ForEach` in the outer (non-custom-Layout) `VStack`
    /// and work correctly, so every scroll-to-sentence call resolves to the
    /// sentence's owning block instead.
    private func scrollTargetID(forSentence id: Int, in doc: ReaderDocument) -> String? {
        guard doc.sentences.indices.contains(id) else { return nil }
        return blockID(doc.sentences[id].blockIndex)
    }

    /// Jump to the resume sentence.
    ///
    /// Confirmed by instrumentation: neither `ScrollViewReader.scrollTo` nor
    /// `.scrollPosition(id:)` can reach a block that hasn't been mounted by
    /// `LazyVStack` yet — the id genuinely doesn't exist in the view tree
    /// until the stack has scrolled near it, and there's no API to force
    /// that from far away. So instead of one jump, this walks the scroll
    /// position forward in small steps: each `scrollTo` lands just past the
    /// currently-mounted region (which *is* reachable, since it either
    /// already exists or is right at the mounting edge), which causes
    /// LazyVStack to mount the next stretch, which makes the next step's
    /// target reachable, and so on until the real target block is hit.
    private func scrollToResume(proxy: ScrollViewProxy, doc: ReaderDocument) {
        let index = reader.currentIndex
        PerfLog.log("scrollToResume called, target index=\(index) of \(doc.sentences.count)")
        guard index != 0, doc.sentences.indices.contains(index) else {
            PerfLog.log("scrollToResume skipped (index 0 or out of range)")
            return
        }
        let targetBlock = doc.sentences[index].blockIndex
        guard targetBlock > 0 else { return }
        initialScrollTarget = blockID(targetBlock)
        walkScroll(proxy: proxy, toBlock: targetBlock, from: 0)
    }

    /// Steps the scroll position from `current` toward `toBlock` in a
    /// LazyVStack, one mountable stretch at a time — see `scrollToResume`.
    private func walkScroll(proxy: ScrollViewProxy, toBlock target: Int, from current: Int,
                            step: Int = 24, hop: Int = 0) {
        guard hop < 2000 else {
            PerfLog.log("walkScroll aborted (too many hops)")
            return
        }
        let next = min(current + step, target)
        let reachedTarget = next >= target
        proxy.scrollTo(blockID(next), anchor: reachedTarget ? .center : .bottom)
        PerfLog.log("walkScroll hop \(hop) -> block \(next)\(reachedTarget ? " (target)" : "")")
        guard !reachedTarget else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.03) {
            walkScroll(proxy: proxy, toBlock: target, from: next, step: step, hop: hop + 1)
        }
    }

    @ViewBuilder
    private func titleHeader(_ doc: ReaderDocument) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(doc.title)
                .font(.system(size: 26, weight: .bold))
                .foregroundStyle(Color(hex: theme.headerColor))
            HStack(spacing: 8) {
                Text(doc.fileType.label.uppercased())
                    .font(.system(size: 11, weight: .bold))
                    .kerning(0.6)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color(hex: theme.textMuted).opacity(0.18),
                                in: RoundedRectangle(cornerRadius: 4))
                    .foregroundStyle(Color(hex: theme.textMuted))
                Text("\(doc.sentences.count) sentences")
                    .font(.caption)
                    .foregroundStyle(Color(hex: theme.textMuted))
            }
        }
        .padding(.bottom, 6)
    }

    @ViewBuilder
    private func blocks(_ doc: ReaderDocument) -> some View {
        ForEach(Array(doc.blocks.enumerated()), id: \.offset) { bi, block in
            blockView(block, blockIndex: bi, doc: doc)
                .id(blockID(bi))
        }
    }

    @ViewBuilder
    private func blockView(_ block: DocBlock, blockIndex: Int, doc: ReaderDocument) -> some View {
        switch block.content {
        case .heading(let level, let text):
            heading(level: level, text: text, blockIndex: blockIndex)
        case .paragraph(let text):
            paragraphBlock(text, blockIndex: blockIndex, doc: doc)
        case .code(let code):
            Text(code)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(Color(hex: theme.text))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(Color(hex: theme.textMuted).opacity(0.1),
                            in: RoundedRectangle(cornerRadius: 8))
        case .quote(let text):
            paragraphBlock(text, blockIndex: blockIndex, doc: doc)
                .padding(.leading, 14)
                .overlay(alignment: .leading) {
                    Rectangle().fill(Color(hex: theme.textMuted).opacity(0.5)).frame(width: 3)
                }
        case .listItem(let text):
            HStack(alignment: .top, spacing: 8) {
                Text("•").foregroundStyle(Color(hex: theme.textMuted))
                paragraphBlock(text, blockIndex: blockIndex, doc: doc)
            }
        case .table(let rows):
            tableView(rows)
        case .image(let alt):
            if !alt.isEmpty {
                Text("[image — \(alt)]")
                    .font(.caption.italic())
                    .foregroundStyle(Color(hex: theme.textMuted).opacity(0.7))
            }
        case .rule:
            Rectangle()
                .fill(Color(hex: theme.textMuted).opacity(0.3))
                .frame(height: 1)
                .padding(.vertical, 8)
        case .frontmatter(_, _, _):
            EmptyView()
        }
    }

    private func heading(level: Int, text: String, blockIndex: Int) -> some View {
        let sizes: [CGFloat] = [27, 23, 20, 17.5, 15.5, 14.5]
        return Text(text)
            .font(.system(size: sizes[max(0, min(5, level - 1))], weight: .bold))
            .foregroundStyle(Color(hex: theme.headerColor))
            .padding(.top, level <= 2 ? 12 : 6)
    }

    /// Paragraph rendered from precomputed sentences — equatable so only the
    /// active paragraph re-renders on each word tick. `themeName` is part of
    /// the identity so toggling themes invalidates the cache.
    private func paragraphBlock(_ text: String, blockIndex: Int, doc: ReaderDocument) -> some View {
        let items = doc.sentencesByBlock[blockIndex] ?? []
        if items.isEmpty { return AnyView(EmptyView()) }
        let isActiveBlock = items.contains { $0.id == reader.currentIndex }
        let searchCurrent = (!findQuery.isEmpty && !findMatches.isEmpty) ? findMatches[findCurrent] : nil
        return AnyView(
            FlowingParagraph(
                items: items,
                fontSize: reader.fontSize * 16.5,
                textColor: Color(hex: theme.text),
                textColorHex: theme.text,
                activeColor: Color(hex: "#f5e08a"),
                wordColor: Color(hex: "#b47a32"),
                activeSentenceID: isActiveBlock ? reader.currentIndex : nil,
                activeWordIndex: isActiveBlock ? reader.activeWordIndex : -1,
                isPlaying: reader.isSpeaking,
                themeName: theme.name,
                searchQuery: findQuery,
                searchCurrentID: searchCurrent,
                onTap: { id in reader.playFrom(id) }
            )
            .equatable()
            .id("para-\(blockIndex)")
        )
    }

    private func tableView(_ rows: [[String]]) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(rows.enumerated()), id: \.offset) { ri, row in
                HStack(spacing: 0) {
                    ForEach(Array(row.enumerated()), id: \.offset) { ci, cell in
                        Text(MarkdownLoader.stripInlineMd(cell))
                            .font(.caption)
                            .foregroundStyle(Color(hex: theme.text))
                            .frame(maxWidth: .infinity, minHeight: 24, alignment: .leading)
                            .padding(6)
                            .background(ri % 2 == 0 ? Color(hex: theme.textMuted).opacity(0.06) : .clear)
                        if ci < row.count - 1 {
                            Divider().overlay(Color(hex: theme.textMuted).opacity(0.3))
                        }
                    }
                }
                Divider().overlay(Color(hex: theme.textMuted).opacity(0.3))
            }
        }
        .overlay(RoundedRectangle(cornerRadius: 6)
            .strokeBorder(Color(hex: theme.textMuted).opacity(0.3)))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

/// Renders one paragraph's precomputed sentence items as tappable spans.
/// Equatable — only the paragraph containing the active sentence re-renders on
/// each word tick; all others are skipped via `.equatable()`. `themeName` and
/// `textColorHex` are included so a theme toggle invalidates every paragraph
/// (otherwise the Equatable cache would keep the old white-on-beige colors).
struct FlowingParagraph: View, Equatable {
    let items: [(id: Int, text: String)]
    let fontSize: CGFloat
    let textColor: Color
    let textColorHex: String
    let activeColor: Color
    let wordColor: Color
    let activeSentenceID: Int?
    let activeWordIndex: Int
    let isPlaying: Bool
    let themeName: String
    let searchQuery: String
    let searchCurrentID: Int?
    var onTap: (Int) -> Void = { _ in }

    init(items: [(id: Int, text: String)], fontSize: CGFloat, textColor: Color, textColorHex: String, activeColor: Color, wordColor: Color, activeSentenceID: Int?, activeWordIndex: Int, isPlaying: Bool, themeName: String, searchQuery: String = "", searchCurrentID: Int? = nil, onTap: @escaping (Int) -> Void = { _ in }) {
        self.items = items; self.fontSize = fontSize; self.textColor = textColor; self.textColorHex = textColorHex
        self.activeColor = activeColor; self.wordColor = wordColor
        self.activeSentenceID = activeSentenceID; self.activeWordIndex = activeWordIndex
        self.isPlaying = isPlaying; self.themeName = themeName
        self.searchQuery = searchQuery; self.searchCurrentID = searchCurrentID; self.onTap = onTap
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.items.map(\.id) == rhs.items.map(\.id)
            && lhs.fontSize == rhs.fontSize
            && lhs.textColorHex == rhs.textColorHex
            && lhs.themeName == rhs.themeName
            && lhs.activeSentenceID == rhs.activeSentenceID
            && lhs.activeWordIndex == rhs.activeWordIndex
            && lhs.isPlaying == rhs.isPlaying
            && lhs.searchQuery == rhs.searchQuery
            && lhs.searchCurrentID == rhs.searchCurrentID
    }

    var body: some View {
        FlowLayout(spacing: 6, lineSpacing: 5) {
            ForEach(items, id: \.id) { item in
                sentenceSpan(item)
                    .id("s-\(item.id)")
            }
        }
        .font(.system(size: fontSize))
        .lineSpacing(fontSize * 0.55)
    }

    @ViewBuilder
    private func sentenceSpan(_ item: (id: Int, text: String)) -> some View {
        let isActive = item.id == activeSentenceID
        let isSearchCurrent = item.id == searchCurrentID && !searchQuery.isEmpty
        let isSearchMatch = !searchQuery.isEmpty
            && item.text.localizedCaseInsensitiveContains(searchQuery)
        Group {
            if isActive, isPlaying || activeSentenceID != nil {
                karaokeText(item.text, searchQuery: searchQuery)
            } else if isSearchMatch {
                highlightedText(item.text, query: searchQuery, isCurrent: isSearchCurrent)
                    .foregroundStyle(textColor)
            } else {
                Text(item.text).foregroundStyle(textColor)
            }
        }
        .padding(.horizontal, 2)
        .padding(.vertical, 1)
        .background(
            isActive ? activeColor
            : isSearchCurrent ? Color(hex: "#ff9f0a").opacity(0.35)
            : isSearchMatch ? Color(hex: "#a8d8ff").opacity(0.45)
            : Color.clear,
            in: RoundedRectangle(cornerRadius: 3)
        )
        .contentShape(Rectangle())
        .onTapGesture { onTap(item.id) }
    }

    private func highlightedText(_ text: String, query: String, isCurrent: Bool) -> Text {
        guard !query.isEmpty else { return Text(text) }
        var attr = AttributedString(text)
        var searchRange = attr.startIndex..<attr.endIndex
        let bg: Color = isCurrent ? Color(hex: "#ff9f0a") : Color(hex: "#a8d8ff")
        let bgOpacity: Color = isCurrent ? bg : bg.opacity(0.55)
        while let r = attr[searchRange].range(of: query, options: .caseInsensitive) {
            attr[r].backgroundColor = bgOpacity
            attr[r].foregroundColor = isCurrent ? .white : Color(hex: "#3a3028")
            searchRange = r.upperBound..<attr.endIndex
            if searchRange.lowerBound >= attr.endIndex { break }
        }
        return Text(attr)
    }

    private func karaokeText(_ sentence: String, searchQuery: String = "") -> some View {
        let timings = WordTimingCalculator.timings(for: sentence)
        // FlowLayout wraps words so a long active sentence doesn't force a
        // single-line HStack that overflows the paragraph width (the cause of
        // the giant yellow viewport fill).
        return FlowLayout(spacing: 2, lineSpacing: 3) {
            ForEach(Array(timings.enumerated()), id: \.offset) { i, t in
                let isSearchWord = !searchQuery.isEmpty && t.word.localizedCaseInsensitiveContains(searchQuery)
                Text(t.word)
                    .foregroundStyle(i == activeWordIndex ? .white : Color(hex: "#3a3028"))
                    .padding(.horizontal, 3)
                    .padding(.vertical, 1)
                    .background(
                        i == activeWordIndex ? wordColor
                        : isSearchWord ? Color(hex: "#a8d8ff").opacity(0.6)
                        : Color.clear,
                        in: RoundedRectangle(cornerRadius: 3)
                    )
            }
        }
    }
}

/// Wrapping flow layout: places children left-to-right and wraps to the next
/// line when they exceed the available width (macOS 13+ Layout protocol).
struct FlowLayout: Layout {
    var spacing: CGFloat = 6
    var lineSpacing: CGFloat = 5

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        // Inside a LazyVStack the width proposal can be nil. Use the constrained
        // content width, and measure each sentence Text with that width as a
        // constraint so long sentences wrap instead of overflowing.
        let maxWidth = proposal.width ?? 648
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        let constrained = ProposedViewSize(width: maxWidth, height: nil)
        for subview in subviews {
            let size = subview.sizeThatFits(constrained)
            // Single long sentence that needs the full width should not be
            // treated as fitting beside the previous one — wrap it.
            let w = min(size.width, maxWidth)
            if x > 0, x + w > maxWidth {
                x = 0
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            x += w + spacing
            lineHeight = max(lineHeight, size.height)
        }
        return CGSize(width: maxWidth, height: y + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let maxWidth = bounds.width
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        let constrained = ProposedViewSize(width: maxWidth, height: nil)
        for subview in subviews {
            let size = subview.sizeThatFits(constrained)
            let w = min(size.width, maxWidth)
            if x > 0, x + w > maxWidth {
                x = 0
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: bounds.minX + x, y: bounds.minY + y),
                          anchor: .topLeading,
                          proposal: ProposedViewSize(width: w, height: size.height))
            x += w + spacing
            lineHeight = max(lineHeight, size.height)
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

    private var theme: ReaderTheme { reader.theme }

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
            .padding(14)

            Divider().overlay(Color(hex: theme.menuBorder))

            ScrollViewReader { proxy in
                List {
                    ForEach(reader.document?.outline ?? []) { entry in
                        Button {
                            NotificationCenter.default.post(name: .scrollToBlock, object: entry.blockIndex)
                        } label: {
                            Text(entry.title)
                                .font(entry.level <= 2 ? .callout.bold() : .callout)
                                .foregroundStyle(Color(hex: theme.text))
                                .padding(.leading, CGFloat(max(0, entry.level - 1)) * 12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .listRowSeparator(.hidden)
                        .listRowBackground(Color.clear)
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
            }
        }
        .background(Color(hex: theme.menuBg))
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
                    Text(formatSpeed(reader.speed))
                        .font(.system(size: 13, weight: .bold))
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

    private func formatSpeed(_ s: Double) -> String {
        let v = (s * 100).rounded() / 100
        return v == v.rounded() ? String(format: "%.0fx", v) : String(format: "%.2gx", v)
    }

    private var speedPopover: some View {
        VStack(spacing: 2) {
            ForEach(ReaderController.speedChoices, id: \.self) { s in
                Button {
                    reader.speed = s
                    showSpeed = false
                } label: {
                    HStack {
                        Text(String(format: "%.2g×", s))
                            .monospacedDigit()
                            .foregroundStyle(Color(hex: reader.theme.text))
                        if abs(s - reader.speed) < 0.001 {
                            Spacer()
                            Image(systemName: "checkmark")
                                .foregroundStyle(Color(hex: reader.theme.text))
                        }
                    }
                    .frame(width: 90)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .padding(.vertical, 5)
                .padding(.horizontal, 12)
            }
        }
        .padding(.vertical, 6)
        .background(Color(hex: reader.theme.menuBg))
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
    }
}

// MARK: - Settings popover

struct SettingsPopover: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @ObservedObject private var hub = EngineHub.shared

    private var theme: ReaderTheme { reader.theme }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Engine status row
            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }

            Divider()

            // Voice
            HStack {
                Text("Voice").frame(width: 52, alignment: .leading)
                Picker("", selection: $reader.voice) {
                    ForEach(["M1","M2","M3","M4","M5","F1","F2","F3","F4","F5","david-deep"], id: \.self) {
                        Text($0).tag($0)
                    }
                }
                .labelsHidden()
            }

            // State
            HStack {
                Text("State").frame(width: 52, alignment: .leading)
                Text(stateLabel).font(.caption).foregroundStyle(.secondary)
            }

            // Progress
            HStack {
                Text("Progress").frame(width: 52, alignment: .leading)
                Text("\(Int(reader.progress * 100))%")
                    .font(.caption.monospacedDigit())
                Spacer()
                if let total = reader.document?.sentences.count {
                    Text("\(reader.currentIndex + 1)/\(total)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            Divider()

            // Font size
            HStack {
                Text("Font size").frame(width: 52, alignment: .leading)
                Button("-") { reader.fontSize = max(0.8, reader.fontSize - 0.05) }
                    .buttonStyle(.bordered)
                Text(String(format: "%.2f×", reader.fontSize))
                    .font(.caption.monospacedDigit())
                    .frame(width: 44)
                Button("+") { reader.fontSize = min(1.6, reader.fontSize + 0.05) }
                    .buttonStyle(.bordered)
            }

            // Volume
            HStack {
                Text("Volume").frame(width: 52, alignment: .leading)
                Slider(value: $reader.volume, in: 0...1, step: 0.05)
                Text(String(format: "%d%%", Int(reader.volume * 100)))
                    .font(.caption.monospacedDigit())
                    .frame(width: 36, alignment: .trailing)
            }

            Divider()

            HStack {
                Button("Test Voice") { reader.testVoice() }
                    .buttonStyle(.bordered)
                Spacer()
                Button("Reset Document", role: .destructive) { reader.resetDocument() }
                    .buttonStyle(.bordered)
            }
        }
        .padding(14)
        .frame(width: 320)
        .background(Color(hex: theme.menuBg))
    }

    private var statusColor: Color {
        if hub.failed { return .red }
        return hub.ready ? .green : .orange
    }

    private var statusText: String {
        if hub.failed { return "Engine failed" }
        if hub.ready { return "Engine ready" }
        return hub.statusText
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
