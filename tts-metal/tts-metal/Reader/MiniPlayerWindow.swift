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

/// Borderless panels refuse key status by default, which would leave the
/// transport buttons needing a first click just to focus the window.
private final class MiniPlayerPanel: NSPanel {
    override var canBecomeKey: Bool { true }
}

final class MiniPlayerWindow {
    static let shared = MiniPlayerWindow()

    static let size = CGSize(width: 520, height: 156)

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
            // Borderless rather than a titled panel with a transparent titlebar:
            // SwiftUI keeps a titlebar-height safe area at the top of a titled
            // window even when the bar is invisible, which left an empty band
            // above the controls.
            let p = MiniPlayerPanel(contentRect: NSRect(origin: .zero, size: Self.size),
                                    styleMask: [.borderless, .nonactivatingPanel],
                                    backing: .buffered, defer: false)
            p.isFloatingPanel = true
            p.level = .floating
            p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            p.isMovableByWindowBackground = true
            p.hidesOnDeactivate = false
            // The rounded card is drawn by SwiftUI; the window itself is clear
            // so the corners and the shadow follow its shape.
            p.isOpaque = false
            p.backgroundColor = .clear
            p.hasShadow = true

            let view = MiniPlayerView()
            hostingView = NSHostingView(rootView: view)
            p.contentView = hostingView
            panel = p

            // Center near the bottom of the screen on first open.
            if let screen = NSScreen.main?.visibleFrame {
                p.setFrameOrigin(NSPoint(x: screen.midX - Self.size.width / 2, y: screen.minY + 60))
            }
        }
        panel?.makeKeyAndOrderFront(nil)
        panel?.invalidateShadow()
    }
}

struct MiniPlayerView: View {
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @State private var isHovering = false

    private var theme: ReaderTheme { reader.theme }
    private var current: RSentence? { reader.currentSentence }
    private var sentences: [RSentence] { reader.document?.sentences ?? [] }
    private let accent = Color(hex: "#b47a32")

    private var prevText: String? {
        guard let c = current, c.id > 0 else { return nil }
        return sentences.first(where: { $0.id == c.id - 1 })?.text
    }
    private var nextText: String? {
        guard let c = current else { return nil }
        return sentences.first(where: { $0.id == c.id + 1 })?.text
    }

    var body: some View {
        VStack(spacing: 0) {
            // Spotify-style lyrics: the active sentence large with per-word
            // highlight, its neighbours as dim single lines for context.
            VStack(spacing: 6) {
                contextLine(prevText)
                Group {
                    if let cur = current {
                        let words = cur.text.split(separator: " ").map(String.init)
                        MiniPlayerCurrentLineView(words: words, activeWordIndex: reader.activeWordIndex, theme: theme)
                    } else {
                        Text("Nothing playing")
                            .font(.system(size: 15, weight: .medium, design: .serif))
                            .foregroundStyle(Color(hex: theme.textMuted))
                    }
                }
                .frame(height: 50)
                contextLine(nextText)
            }
            .padding(.horizontal, 22)
            .padding(.top, 14)

            Spacer(minLength: 0)

            HStack(spacing: 14) {
                transportButton("backward.fill", size: 11) { reader.skipSentence(-1) }
                    .disabled(reader.document == nil)
                Button { reader.togglePlayPause() } label: {
                    Image(systemName: icon)
                        .font(.system(size: 12, weight: .bold))
                        .frame(width: 28, height: 28)
                        .background(accent, in: Circle())
                        .foregroundStyle(.white)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                transportButton("forward.fill", size: 11) { reader.skipSentence(+1) }
                    .disabled(reader.document == nil)

                Text(reader.document?.fileName ?? "Blabla")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color(hex: theme.textMuted))
                    .lineLimit(1).truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .trailing)

                Button { reader.miniPlayerVisible = false } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .frame(width: 18, height: 18)
                        .background(Color(hex: theme.text).opacity(0.08), in: Circle())
                        .foregroundStyle(Color(hex: theme.textMuted))
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Close mini player")
                .opacity(isHovering ? 1 : 0)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
        }
        .frame(width: MiniPlayerWindow.size.width, height: MiniPlayerWindow.size.height)
        .background(WindowDragArea())
        .background(Color(hex: theme.bg), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
            .strokeBorder(Color(hex: theme.dropBorder).opacity(0.45), lineWidth: 1))
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.15)) { isHovering = hovering }
        }
    }

    private func contextLine(_ text: String?) -> some View {
        Text(text ?? " ")
            .font(.system(size: 11))
            .foregroundStyle(Color(hex: theme.textMuted).opacity(0.6))
            .lineLimit(1).truncationMode(.tail)
            .frame(maxWidth: .infinity, alignment: .center)
            .frame(height: 14)
    }

    private func transportButton(_ symbol: String, size: CGFloat,
                                 action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: size))
                .foregroundStyle(Color(hex: theme.text).opacity(0.7))
                .frame(width: 22, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var icon: String {
        switch reader.state { case .playing, .generating: return "pause.fill"; default: return "play.fill" }
    }
}

/// Lets the borderless panel be dragged from anywhere that isn't a control.
/// `isMovableByWindowBackground` alone does nothing here: the SwiftUI hosting
/// view claims every mouse-down, so AppKit never sees a background to drag.
/// This view sits behind the content, so the buttons above it still take
/// their clicks and everything else lands here.
private struct WindowDragArea: NSViewRepresentable {
    final class DragView: NSView {
        override func mouseDown(with event: NSEvent) {
            window?.performDrag(with: event)
        }
    }

    func makeNSView(context: Context) -> NSView { DragView() }
    func updateNSView(_ nsView: NSView, context: Context) {}
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
            // Lyrics-style progress: the word being spoken in the accent,
            // words already read at full strength, the rest of the line dim.
            if i == active {
                chunk.foregroundColor = accent
            } else if i < active {
                chunk.foregroundColor = Color(hex: theme.text)
            } else {
                chunk.foregroundColor = Color(hex: theme.text).opacity(active < 0 ? 1 : 0.4)
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
