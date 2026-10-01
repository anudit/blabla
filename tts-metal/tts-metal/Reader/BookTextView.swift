//
//  BookTextView.swift
//  tts-metal
//
//  The reading surface: one TextKit text view holding the whole book.
//
//  Everything here exists because the previous surface — a `LazyVStack` with a
//  SwiftUI view per block and a `Text` per sentence — could not do three
//  things the reader needs:
//
//  * **Selection.** Each sentence was its own view, so a drag could never
//    cross from one to the next. Here the book is a single text storage, and
//    selection is whatever AppKit already does.
//  * **Jumping.** `scrollTo` and `.scrollPosition(id:)` only reach rows a lazy
//    stack has already mounted, which is why ⌘F and bookmark-resume silently
//    failed for anything past the first screen and needed a chain of
//    step-by-step scroll hops to fake it. A text view knows the position of
//    every character whether or not it has been drawn, so a jump is one
//    `scroll(to:)`.
//  * **Typography.** Justification, hyphenation, hanging indents and inline
//    figures are paragraph-level decisions; a stack of independently measured
//    views has nowhere to express them.
//

import SwiftUI
import AppKit

/// A one-shot scroll instruction. `token` is what makes it one-shot: the view
/// acts only when the token changes, so re-rendering for an unrelated reason
/// never re-triggers an old jump.
struct BookScrollRequest: Equatable {
    var token = 0
    var sentenceID: Int?
    var blockIndex: Int?
}

struct BookTextView: NSViewRepresentable {
    let document: ReaderDocument
    let theme: ReaderTheme
    let fontScale: CGFloat
    let columnWidth: CGFloat

    let activeSentenceID: Int?
    let activeWordFraction: Double
    let isPlaying: Bool

    let searchQuery: String
    let searchCurrent: Int
    var scrollRequest: BookScrollRequest

    /// Reports how many search hits the text contains, so the find bar can
    /// show "3/57" without duplicating the search.
    var onSearchCount: (Int) -> Void = { _ in }
    /// Double-click anywhere in the prose starts playback from that sentence.
    var onActivateSentence: (Int) -> Void = { _ in }
    /// A trackpad pinch finished; carries the new font scale.
    var onZoom: (CGFloat) -> Void = { _ in }

    static let fontScaleRange: ClosedRange<CGFloat> = 0.8...1.6

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let coordinator = context.coordinator

        let storage = NSTextStorage()
        let layout = NSLayoutManager()
        // Exact positions matter more here than incremental layout cost: the
        // whole point of this view is that "jump to sentence 4,812" lands
        // where it should. With non-contiguous layout the y of an un-laid-out
        // paragraph is an estimate, and a resume or ⌘F jump would arrive near
        // the target rather than on it.
        layout.allowsNonContiguousLayout = false
        // The container follows the text view's width, and the text view fills
        // the scroll view: the reading column is produced by the horizontal
        // inset instead.
        let container = NSTextContainer(size: CGSize(width: columnWidth,
                                                     height: .greatestFiniteMagnitude))
        container.widthTracksTextView = true
        container.lineFragmentPadding = 0
        layout.addTextContainer(container)
        storage.addLayoutManager(layout)

        let textView = BookNSTextView(frame: CGRect(x: 0, y: 0, width: columnWidth, height: 0),
                                      textContainer: container)
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = false
        textView.drawsBackground = true
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        // Without an unbounded maxSize the view cannot grow past its initial
        // frame, so the book has nothing to scroll.
        textView.minSize = CGSize(width: 0, height: 0)
        textView.maxSize = CGSize(width: CGFloat.greatestFiniteMagnitude,
                                  height: CGFloat.greatestFiniteMagnitude)
        textView.autoresizingMask = [.width]
        textView.textContainerInset = CGSize(width: 0, height: 28)
        textView.linkTextAttributes = [:]
        textView.onDoubleClick = { [weak coordinator] location in
            guard let id = coordinator?.rendered.sentence(at: location) else { return }
            coordinator?.parent.onActivateSentence(id)
        }

        let scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.drawsBackground = true
        scrollView.autohidesScrollers = true
        // Pinch zooms the page live, then `pinchEnded` turns the zoom into a
        // font scale so the text re-wraps at the new size instead of staying
        // a scaled bitmap of the old layout.
        scrollView.allowsMagnification = true

        coordinator.textView = textView
        coordinator.scrollView = scrollView
        coordinator.observePinch(on: scrollView)
        coordinator.installClickOutsideMonitor()
        return scrollView
    }

    static func dismantleNSView(_ scrollView: NSScrollView, coordinator: Coordinator) {
        coordinator.teardown()
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        let coordinator = context.coordinator
        coordinator.parent = self
        scrollView.backgroundColor = NSColor(hex: theme.bg)
        // Live pinch stops where the committed font scale would be clamped.
        scrollView.minMagnification = Self.fontScaleRange.lowerBound / fontScale
        scrollView.maxMagnification = Self.fontScaleRange.upperBound / fontScale

        coordinator.applyContent(document: document, theme: theme,
                                 fontScale: fontScale, columnWidth: columnWidth)
        coordinator.applySearch(query: searchQuery, current: searchCurrent, theme: theme)
        coordinator.applyHighlight(sentenceID: activeSentenceID,
                                   fraction: activeWordFraction,
                                   isPlaying: isPlaying,
                                   document: document, theme: theme)
        coordinator.applyScroll(scrollRequest)
    }

    // MARK: - Coordinator

    @MainActor
    final class Coordinator {
        var parent: BookTextView
        weak var textView: BookNSTextView?
        weak var scrollView: NSScrollView?
        private(set) var rendered = RenderedBook.empty

        /// What the current `rendered` was built from. Rebuilding is the one
        /// genuinely expensive operation here, so it happens only when one of
        /// these actually changes — not on every karaoke tick.
        private var builtKey: String?
        private var searchRanges: [NSRange] = []
        private var searchKey: String?
        private var highlightKey: String?
        private var lastScrollToken = -1
        /// The find hit currently stepped onto. A scroll request that names
        /// neither a sentence nor a block means "show the current find hit".
        private var currentSearchRange: NSRange?
        /// Set when a jump arrives before the text it targets exists, so it can
        /// be retried once the book is installed.
        private var pendingScroll: BookScrollRequest?

        private var pinchObserver: NSObjectProtocol?
        private var clickMonitor: Any?

        init(_ parent: BookTextView) { self.parent = parent }

        func teardown() {
            if let pinchObserver { NotificationCenter.default.removeObserver(pinchObserver) }
            if let clickMonitor { NSEvent.removeMonitor(clickMonitor) }
            pinchObserver = nil
            clickMonitor = nil
        }

        // MARK: Pinch zoom

        func observePinch(on scrollView: NSScrollView) {
            pinchObserver = NotificationCenter.default.addObserver(
                forName: NSScrollView.didEndLiveMagnifyNotification, object: scrollView, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.pinchEnded() }
            }
        }

        private func pinchEnded() {
            guard let scrollView else { return }
            let range = BookTextView.fontScaleRange
            let raw = parent.fontScale * scrollView.magnification
            // Same 0.05 steps as the toolbar's −/+ buttons.
            let scale = min(range.upperBound, max(range.lowerBound, (raw * 20).rounded() / 20))
            // Back to 1× before the rebuild: the new font size now carries the
            // zoom, and the rebuild re-anchors on whatever is at the top.
            scrollView.magnification = 1
            if abs(scale - parent.fontScale) > 0.001 { parent.onZoom(scale) }
        }

        // MARK: Click outside

        /// A selection otherwise stays painted after clicking elsewhere in the
        /// window (sidebar, toolbar, empty chrome), because the text view only
        /// greys it out on losing focus. Clicks inside the scroll view —
        /// including its margins and scroller — are left to AppKit.
        func installClickOutsideMonitor() {
            clickMonitor = NSEvent.addLocalMonitorForEvents(matching: .leftMouseDown) { [weak self] event in
                MainActor.assumeIsolated {
                    guard let self, let textView = self.textView, let scrollView = self.scrollView,
                          event.window === textView.window,
                          textView.selectedRange().length > 0 else { return }
                    let hit = event.window?.contentView?.hitTest(event.locationInWindow)
                    if hit.map({ !$0.isDescendant(of: scrollView) }) ?? true {
                        textView.setSelectedRange(NSRange(location: textView.selectedRange().location, length: 0))
                    }
                }
                return event
            }
        }

        // MARK: Content

        func applyContent(document: ReaderDocument, theme: ReaderTheme,
                          fontScale: CGFloat, columnWidth: CGFloat) {
            // Column width is quantized: a live window resize would otherwise
            // rebuild the whole book on every frame of the drag.
            let width = (columnWidth / 20).rounded() * 20
            let key = "\(document.sourceID)|\(document.blocks.count)|\(theme.name)|\(fontScale)|\(width)"
            guard key != builtKey, let textView else { return }
            let isRebuild = builtKey != nil
            builtKey = key

            // A rebuild (theme swap, font step, window resize) must not throw
            // the reader back to page one, so the character at the top of the
            // window is remembered and restored once the new layout exists.
            let anchor = isRebuild ? visibleCharacterIndex() : nil

            rendered = BookRenderer.render(document: document, theme: theme,
                                           fontScale: fontScale, columnWidth: width)
            textView.rendered = rendered
            textView.columnWidth = width
            textView.textStorage?.setAttributedString(rendered.attributed)
            textView.setSelectedRange(NSRange(location: 0, length: 0))
            textView.backgroundColor = NSColor(hex: theme.bg)
            textView.insertionPointColor = NSColor(hex: theme.text)
            textView.highlights = []
            // Re-apply on the next pass; the maps they index into just changed.
            searchKey = nil
            highlightKey = nil

            if let anchor, anchor < rendered.attributed.length {
                scroll(to: NSRange(location: anchor, length: 1), anchorFraction: 0.02)
            }
            // A jump that arrived before this book existed (the resume position
            // travels with the document) now has something to aim at.
            if let held = pendingScroll { applyScroll(held) }
        }

        /// Character index at the top of the visible area, used to hold the
        /// reader's place across a re-layout.
        private func visibleCharacterIndex() -> Int? {
            guard let textView, let layout = textView.layoutManager,
                  let container = textView.textContainer,
                  let clip = scrollView?.contentView else { return nil }
            var rect = clip.bounds
            rect.origin.y -= textView.textContainerInset.height
            let glyphs = layout.glyphRange(forBoundingRect: rect, in: container)
            guard glyphs.length > 0 else { return nil }
            return layout.characterIndexForGlyph(at: glyphs.location)
        }

        // MARK: Search

        func applySearch(query raw: String, current: Int, theme: ReaderTheme) {
            let query = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            let key = "\(query)|\(current)"
            guard key != searchKey, let textView else { return }
            let queryChanged = searchKey?.split(separator: "|", omittingEmptySubsequences: false)
                .first.map(String.init) != query
            searchKey = key

            if queryChanged {
                searchRanges = query.isEmpty ? [] : matches(of: query)
                parent.onSearchCount(searchRanges.count)
            }
            currentSearchRange = searchRanges.indices.contains(current) ? searchRanges[current] : nil
            textView.searchHits = searchRanges
            textView.currentSearchHit = currentSearchRange
            rebuildHighlights(theme: theme)
        }

        private func matches(of query: String) -> [NSRange] {
            let text = rendered.attributed.string as NSString
            guard text.length > 0, !query.isEmpty else { return [] }
            var out: [NSRange] = []
            var from = 0
            while from < text.length {
                let hit = text.range(of: query, options: [.caseInsensitive, .diacriticInsensitive],
                                     range: NSRange(location: from, length: text.length - from))
                guard hit.location != NSNotFound, hit.length > 0 else { break }
                out.append(hit)
                from = hit.upperBound
            }
            return out
        }

        // MARK: Playback highlight

        func applyHighlight(sentenceID: Int?, fraction: Double, isPlaying: Bool,
                            document: ReaderDocument, theme: ReaderTheme) {
            guard let textView else { return }
            // The word, not the raw fraction, is what changes what's on screen
            // — quantizing here is what keeps a 60fps tick from repainting
            // sixty times a second while the word hasn't moved.
            let sentence = sentenceID.flatMap { id in
                document.sentences.indices.contains(id) ? document.sentences[id] : nil
            }
            let word = sentence.map { wordIndex(in: $0.displayText, at: fraction) } ?? -1
            let key = "\(sentenceID ?? -1)|\(isPlaying ? word : -1)|\(theme.name)"
            guard key != highlightKey else { return }
            highlightKey = key

            var sentenceRange: NSRange?
            var wordRangeOut: NSRange?
            if let sentence, let range = rendered.sentenceRanges[sentence.id] {
                sentenceRange = range
                if isPlaying, word >= 0 {
                    wordRangeOut = wordRange(index: word, in: sentence.displayText, base: range.location)
                }
            }
            textView.activeSentence = sentenceRange
            textView.activeWord = wordRangeOut
            rebuildHighlights(theme: theme)
        }

        /// Hands the text view the full highlight set in paint order.
        private func rebuildHighlights(theme: ReaderTheme) {
            guard let textView else { return }
            var out: [BookHighlight] = []
            let wash = NSColor(hex: "#a8d8ff").withAlphaComponent(0.40)
            for r in textView.searchHits { out.append(BookHighlight(range: r, color: wash)) }
            if let r = textView.currentSearchHit {
                out.append(BookHighlight(range: r, color: NSColor(hex: "#ff9f0a")))
            }
            if let r = textView.activeSentence {
                out.append(BookHighlight(range: r,
                                         color: NSColor(hex: theme.isDark ? "#4a4020" : "#f5e08a")))
            }
            if let r = textView.activeWord {
                out.append(BookHighlight(range: r, color: NSColor(hex: "#b47a32")))
            }
            textView.highlights = out
        }

        /// Which word of the *displayed* sentence is being spoken.
        ///
        /// The controller measures progress against the synthesized string,
        /// whose word count differs from the page whenever normalization
        /// expanded something ("125" is one word here and three in the audio),
        /// so a word *index* cannot be carried across — a fraction can.
        private func wordIndex(in display: String, at fraction: Double) -> Int {
            let timings = WordTimingCalculator.timings(for: display)
            guard !timings.isEmpty else { return -1 }
            let f = max(0, min(0.999, fraction))
            for (i, t) in timings.enumerated() where f < t.endFrac { return i }
            return timings.count - 1
        }

        private func wordRange(index: Int, in display: String, base: Int) -> NSRange? {
            let ns = display as NSString
            var wordStart = 0
            var seen = 0
            var i = 0
            while i <= ns.length {
                let isBreak = i == ns.length || ns.character(at: i) == 0x20
                if isBreak {
                    if i > wordStart {
                        if seen == index {
                            return NSRange(location: base + wordStart, length: i - wordStart)
                        }
                        seen += 1
                    }
                    wordStart = i + 1
                }
                i += 1
            }
            return nil
        }

        // MARK: Scrolling

        func applyScroll(_ request: BookScrollRequest) {
            guard request.token != lastScrollToken else { return }
            let range: NSRange?
            if let id = request.sentenceID {
                range = rendered.sentenceRanges[id]
            } else if let bi = request.blockIndex {
                range = rendered.blockRanges[bi]
            } else {
                range = currentSearchRange
            }
            // A jump can be requested before the book has been installed (the
            // resume position arrives with the document). Hold it rather than
            // dropping it; the next update pass retries, since the token still
            // differs from the last one acted on.
            guard let range else {
                pendingScroll = request
                return
            }
            lastScrollToken = request.token
            pendingScroll = nil
            PerfLog.log("scroll -> sentence \(request.sentenceID.map(String.init) ?? "-") "
                        + "range \(range.location)+\(range.length) of \(rendered.attributed.length)")
            scroll(to: range)
        }

        /// Parks `range` a little above the middle of the window by default —
        /// reading continues downward, so the lines that follow are the ones
        /// worth showing.
        private func scroll(to range: NSRange, anchorFraction: CGFloat = 0.38) {
            guard let textView, let layout = textView.layoutManager,
                  let container = textView.textContainer,
                  let clip = scrollView?.contentView else { return }
            layout.ensureLayout(forCharacterRange: range)
            let glyphs = layout.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
            var rect = layout.boundingRect(forGlyphRange: glyphs, in: container)
            rect.origin.y += textView.textContainerInset.height

            let visible = clip.bounds.height
            let maxY = max(0, textView.bounds.height - visible)
            let y = min(max(0, rect.midY - visible * anchorFraction), maxY)
            clip.scroll(to: NSPoint(x: 0, y: y))
            scrollView?.reflectScrolledClipView(clip)
        }
    }
}

/// One painted highlight. Backgrounds are drawn by the view rather than set as
/// a background text attribute — see `BookNSTextView.boxes(for:)`.
struct BookHighlight: Equatable {
    let range: NSRange
    let color: NSColor
}

/// `NSTextView` that paints its own highlight backgrounds and centres the
/// reading column.
///
/// Highlights are drawn here instead of being set as `.backgroundColor`
/// attributes because `NSLayoutManager` fills a background across the whole
/// *line fragment*, and a fragment is much taller than its glyphs: the reader
/// sets a line height well above the font's own so the prose breathes, and all
/// that extra leading is added above the text. The stock fill therefore paints
/// a tall rectangle with the words sitting along its bottom edge — the
/// misaligned highlight. Drawing from the baseline and the font's own ascender
/// and descender puts the box exactly around the letters, whatever the leading.
final class BookNSTextView: NSTextView {
    var onDoubleClick: (Int) -> Void = { _ in }
    var rendered = RenderedBook.empty

    // Ranges the coordinator keeps up to date; `highlights` is the paint list
    // built from them.
    var searchHits: [NSRange] = []
    var currentSearchHit: NSRange?
    var activeSentence: NSRange?
    var activeWord: NSRange?

    var highlights: [BookHighlight] = [] {
        didSet {
            guard highlights != oldValue else { return }
            // Repaint the union of where the highlight was and where it now
            // is, with room for the rounded corners; anything tighter leaves
            // slivers of the previous highlight behind as it moves.
            var dirty = NSRect.zero
            for h in oldValue + highlights {
                for box in boxes(for: h.range) {
                    dirty = dirty.isEmpty ? box : dirty.union(box)
                }
            }
            if !dirty.isEmpty { setNeedsDisplay(dirty.insetBy(dx: -8, dy: -8)) }
        }
    }

    /// Width of the text column. The view itself always spans the scroll view
    /// — so the background, the scroller and a click in the margin all behave
    /// normally — and the column is centred inside it by the container inset,
    /// which `layout()` keeps correct across window resizes.
    var columnWidth: CGFloat = 680 {
        didSet { if columnWidth != oldValue { needsLayout = true } }
    }

    override func layout() {
        let inset = max(24, ((bounds.width - columnWidth) / 2).rounded())
        if abs(textContainerInset.width - inset) > 0.5 {
            textContainerInset = CGSize(width: inset, height: textContainerInset.height)
        }
        super.layout()
    }

    /// The boxes a range occupies, one per line it spans, sized to the text
    /// rather than to the line fragment.
    func boxes(for range: NSRange) -> [NSRect] {
        guard let layout = layoutManager, let container = textContainer,
              range.length > 0, range.upperBound <= (textStorage?.length ?? 0) else { return [] }
        let glyphRange = layout.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
        guard glyphRange.length > 0 else { return [] }
        let origin = textContainerOrigin
        var out: [NSRect] = []
        layout.enumerateLineFragments(forGlyphRange: glyphRange) { fragment, _, _, lineGlyphs, _ in
            let hit = NSIntersectionRange(lineGlyphs, glyphRange)
            guard hit.length > 0 else { return }
            var box = layout.boundingRect(forGlyphRange: hit, in: container)
            // `location(forGlyphAt:)` is relative to the line fragment, and its
            // y *is* the baseline — the one anchor that doesn't move when the
            // paragraph's line height changes.
            let baseline = fragment.minY + layout.location(forGlyphAt: hit.location).y
            let font = self.font(atGlyph: hit.location, layout: layout)
            box.origin.y = baseline - font.ascender
            box.size.height = font.ascender - font.descender
            out.append(box.offsetBy(dx: origin.x, dy: origin.y).insetBy(dx: -2, dy: -1.5))
        }
        return out
    }

    private func font(atGlyph glyph: Int, layout: NSLayoutManager) -> NSFont {
        let fallback = self.font ?? NSFont.systemFont(ofSize: 16)
        guard let storage = textStorage, storage.length > 0 else { return fallback }
        let index = min(layout.characterIndexForGlyph(at: glyph), storage.length - 1)
        return storage.attribute(.font, at: index, effectiveRange: nil) as? NSFont ?? fallback
    }

    override func drawBackground(in rect: NSRect) {
        super.drawBackground(in: rect)
        for highlight in highlights {
            var filled = false
            for box in boxes(for: highlight.range) where box.intersects(rect) {
                if !filled { highlight.color.setFill(); filled = true }
                NSBezierPath(roundedRect: box, xRadius: 5, yRadius: 5).fill()
            }
        }
    }

    /// Double-click, not single, so a plain click stays available for placing
    /// and dragging a selection.
    override func mouseDown(with event: NSEvent) {
        guard event.clickCount == 2 else {
            super.mouseDown(with: event)
            return
        }
        let point = convert(event.locationInWindow, from: nil)
        let index = characterIndexForInsertion(at: point)
        super.mouseDown(with: event)
        onDoubleClick(index)
    }
}
