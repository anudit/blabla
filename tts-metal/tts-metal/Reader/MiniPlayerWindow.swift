//
//  MiniPlayerWindow.swift
//  tts-metal
//
//  Floating always-on-top "now reading" panel — the native replacement for
//  blabla's Document Picture-in-Picture canvas. Mirrors the current sentence
//  with karaoke word highlighting plus transport controls.
//

import AppKit
import SwiftUI

/// Shared ReaderController instance for the whole app (reader window + mini player).
@MainActor
enum ReaderControllerHolder {
    static let reader: ReaderController = ReaderController()
}

final class MiniPlayerWindow {
    static let shared = MiniPlayerWindow()

    private var panel: NSPanel?
    private var hostingView: NSHostingView<MiniPlayerView>?

    private init() {}

    func setVisible(_ visible: Bool) {
        Task { @MainActor in
            if visible {
                show()
            } else {
                panel?.orderOut(nil)
            }
        }
    }

    private func show() {
        if panel == nil {
            let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 560, height: 195),
                            styleMask: [.nonactivatingPanel, .titled, .fullSizeContentView],
                            backing: .buffered, defer: false)
            p.isFloatingPanel = true
            p.level = .floating
            p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            p.titleVisibility = .hidden
            p.titlebarAppearsTransparent = true
            p.isMovableByWindowBackground = true
            p.hidesOnDeactivate = false

            let view = MiniPlayerView()
            hostingView = NSHostingView(rootView: view)
            p.contentView = hostingView
            panel = p

            // Center near the bottom of the screen on first open.
            if let screen = NSScreen.main?.visibleFrame {
                p.setFrameOrigin(NSPoint(x: screen.midX - 280, y: screen.minY + 60))
            }
        }
        panel?.makeKeyAndOrderFront(nil)
    }
}

struct MiniPlayerView: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader

    private var theme: ReaderTheme { reader.theme }
    private var current: RSentence? { reader.currentSentence }
    private var sentences: [RSentence] { reader.document?.sentences ?? [] }

    private var prevText: String? {
        guard let c = current, c.id > 0 else { return nil }
        return sentences.first(where: { $0.id == c.id - 1 })?.text
    }
    private var nextText: String? {
        guard let c = current else { return nil }
        return sentences.first(where: { $0.id == c.id + 1 })?.text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Transport — theme-aware icons
            HStack(spacing: 10) {
                Button { reader.skipSentence(-1) } label: {
                    Image(systemName: "backward.fill").font(.system(size: 13))
                        .foregroundStyle(Color(hex: theme.text).opacity(0.85))
                }
                .buttonStyle(.plain)
                .disabled(reader.document == nil)
                Button { reader.togglePlayPause() } label: {
                    Image(systemName: icon)
                        .font(.system(size: 15, weight: .bold))
                        .frame(width: 34, height: 34)
                        .background(Color(hex: "#b47a32"), in: Circle())
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
                Button { reader.skipSentence(+1) } label: {
                    Image(systemName: "forward.fill").font(.system(size: 13))
                        .foregroundStyle(Color(hex: theme.text).opacity(0.85))
                }
                .buttonStyle(.plain)
                .disabled(reader.document == nil)
                Spacer()
                Text(reader.document?.fileName ?? "Blabla")
                    .font(.caption)
                    .foregroundStyle(Color(hex: theme.textMuted))
                    .lineLimit(1).truncationMode(.tail)
                Button { reader.miniPlayerVisible = false } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(Color(hex: theme.textMuted))
                }
                .buttonStyle(.plain)
                .help("Close mini player")
            }

            // Fixed-height single-line focus (Spotify-like): only the active
            // sentence is shown large in the center with per-word highlight;
            // prev/next are dimmed single lines for context.
            VStack(alignment: .center, spacing: 8) {
                if let prev = prevText {
                    Text(prev)
                        .font(.system(size: 11))
                        .foregroundStyle(Color(hex: theme.textMuted).opacity(0.65))
                        .lineLimit(1).truncationMode(.tail)
                        .frame(maxWidth: .infinity, alignment: .center)
                } else {
                    Text(" ").font(.system(size: 11)).hidden().frame(height: 14)
                }
                Group {
                    if let cur = current {
                        let words = cur.text.split(separator: " ").map(String.init)
                        MiniPlayerCurrentLineView(words: words, activeWordIndex: reader.activeWordIndex, theme: theme)
                    } else {
                        Text("Nothing playing").foregroundStyle(Color(hex: theme.textMuted))
                            .font(.system(size: 14))
                    }
                }
                .frame(height: 48, alignment: .center)

                if let nxt = nextText {
                    Text(nxt)
                        .font(.system(size: 11))
                        .foregroundStyle(Color(hex: theme.textMuted).opacity(0.65))
                        .lineLimit(1).truncationMode(.tail)
                        .frame(maxWidth: .infinity, alignment: .center)
                } else {
                    Text(" ").font(.system(size: 11)).hidden().frame(height: 14)
                }
            }
            .frame(height: 124, alignment: .center)
            .frame(maxWidth: .infinity)
            .padding(12)
            .background(Color(hex: theme.dropBg), in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10)
                .strokeBorder(Color(hex: theme.dropBorder).opacity(0.5)))
        }
        .frame(height: 195)
        .padding(14)
        .background(Color(hex: theme.bg))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var icon: String {
        switch reader.state { case .playing, .generating: return "pause.fill"; default: return "play.fill" }
    }
}

/// Shows the active sentence two lines at a time, always keeping the row
/// containing the karaoke-highlighted word visible. Longer sentences that
/// used to hard-truncate past line 2 (`...`) instead page forward to the
/// next two-line "window" once the highlight moves past what's currently
/// shown — same idea as paging a teleprompter.
private struct MiniPlayerCurrentLineView: View {
    let words: [String]
    let activeWordIndex: Int
    let theme: ReaderTheme

    var body: some View {
        GeometryReader { geo in
            let font = MiniPlayerLineWrap.serifBoldFont(size: 17)
            let lineIndices = MiniPlayerLineWrap.lineIndices(words: words, font: font, width: geo.size.width)
            let activeLine = lineIndices.indices.contains(activeWordIndex) ? lineIndices[activeWordIndex] : 0
            let windowStart = (activeLine / 2) * 2
            let startIdx = lineIndices.firstIndex(where: { $0 >= windowStart }) ?? 0
            let endIdx = lineIndices.firstIndex(where: { $0 > windowStart + 1 }) ?? lineIndices.count
            let visibleWords = Array(words[startIdx..<max(startIdx, endIdx)])
            let localActive = activeWordIndex - startIdx

            Text(karaokeWords(visibleWords, active: localActive, theme: theme))
                .font(.system(size: 17, weight: .bold, design: .serif))
                .lineSpacing(4)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
    }

    private func karaokeWords(_ words: [String], active: Int, theme: ReaderTheme) -> AttributedString {
        var out = AttributedString()
        let accent = Color(hex: "#b47a32")
        for (i, w) in words.enumerated() {
            var chunk = AttributedString(w + (i < words.count - 1 ? " " : ""))
            if i == active {
                chunk.backgroundColor = accent
                chunk.foregroundColor = .white
            } else {
                chunk.foregroundColor = Color(hex: theme.text)
            }
            out += chunk
        }
        return out
    }
}

/// Figures out which wrapped line each word of the mini-player's current
/// sentence lands on, so the view can page forward by whole lines instead
/// of truncating. Uses `NSLayoutManager` (the same line-breaking engine
/// SwiftUI's `Text` sits on top of) to reproduce the exact wrap points for
/// a given width/font rather than approximating with character counts.
enum MiniPlayerLineWrap {
    private static var lastKey: String?
    private static var lastResult: [Int] = []

    static func serifBoldFont(size: CGFloat) -> NSFont {
        let base = NSFont.boldSystemFont(ofSize: size)
        let serifDescriptor = base.fontDescriptor.withDesign(.serif) ?? base.fontDescriptor
        return NSFont(descriptor: serifDescriptor, size: size) ?? base
    }

    /// Returns, for each word, the 0-based wrapped-line index it falls on.
    static func lineIndices(words: [String], font: NSFont, width: CGFloat) -> [Int] {
        guard width > 0, !words.isEmpty else { return Array(repeating: 0, count: words.count) }
        let key = "\(words.joined(separator: " "))|\(Int(width.rounded()))|\(font.pointSize)"
        if key == lastKey { return lastResult }
        let result = compute(words: words, font: font, width: width)
        lastKey = key
        lastResult = result
        return result
    }

    private static func compute(words: [String], font: NSFont, width: CGFloat) -> [Int] {
        var text = ""
        var wordStarts: [Int] = []
        for (i, w) in words.enumerated() {
            wordStarts.append(text.utf16.count)
            text += w
            if i < words.count - 1 { text += " " }
        }

        let storage = NSTextStorage(string: text, attributes: [.font: font])
        let layoutManager = NSLayoutManager()
        storage.addLayoutManager(layoutManager)
        let container = NSTextContainer(size: CGSize(width: width, height: .greatestFiniteMagnitude))
        container.lineFragmentPadding = 0
        layoutManager.addTextContainer(container)
        layoutManager.ensureLayout(for: container)

        var lineIndices = [Int](repeating: 0, count: words.count)
        var lineNumber = 0
        layoutManager.enumerateLineFragments(forGlyphRange: NSRange(location: 0, length: layoutManager.numberOfGlyphs)) { _, _, _, glyphRange, _ in
            let charRange = layoutManager.characterRange(forGlyphRange: glyphRange, actualGlyphRange: nil)
            let lineEnd = charRange.location + charRange.length
            for (i, start) in wordStarts.enumerated() where start >= charRange.location && start < lineEnd {
                lineIndices[i] = lineNumber
            }
            lineNumber += 1
        }
        return lineIndices
    }
}

extension Color {
    // Karaoke highlighting re-renders every visible word on a 60fps timer
    // tick, and each render re-parses the same handful of theme hex strings
    // via Scanner — memoize so steady-state playback doesn't burn CPU on
    // repeat string parsing.
    private static var hexCache: [String: Color] = [:]

    init(hex: String) {
        if let cached = Color.hexCache[hex] {
            self = cached
            return
        }
        var h = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        if h.hasPrefix("#") { h.removeFirst() }
        var v: UInt64 = 0
        Scanner(string: h).scanHexInt64(&v)
        let r = Double((v >> 16) & 0xFF) / 255.0
        let g = Double((v >> 8) & 0xFF) / 255.0
        let b = Double(v & 0xFF) / 255.0
        self.init(red: r, green: g, blue: b)
        Color.hexCache[hex] = self
    }
}
