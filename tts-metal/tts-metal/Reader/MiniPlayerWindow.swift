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
                        Text(karaokeWords(words, active: reader.activeWordIndex))
                            .font(.system(size: 17, weight: .bold, design: .serif))
                            .lineSpacing(4)
                            .multilineTextAlignment(.center)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .center)
                    } else {
                        Text("Nothing playing").foregroundStyle(Color(hex: theme.textMuted))
                            .font(.system(size: 14))
                    }
                }
                .frame(minHeight: 48, alignment: .center)

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

    private func karaokeWords(_ words: [String], active: Int) -> AttributedString {
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
