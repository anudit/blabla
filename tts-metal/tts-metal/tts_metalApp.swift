//
//  tts_metalApp.swift
//  tts-metal
//
//  Menu-bar agent app hosting two surfaces:
//   1. the menu-bar popover (selection reading via ⌥⌘R), and
//   2. the BlaBla document reader window (PDF/EPUB/MOBI/DOCX/MD/TXT/URL),
//      both powered by the shared Supertonic 3 Metal engine.
//

import SwiftUI

@main
struct tts_metalApp: App {
    @StateObject private var controller = TtsController()

    init() {
        // Diagnostic harness for the main-thread stall report; see MainThreadStallTest.
        if ProcessInfo.processInfo.environment["SUPERTONIC_STALLTEST"] == "1" {
            Task { @MainActor in await MainThreadStallTest.run() }
        }
    }

    var body: some Scene {
        MenuBarExtra {
            ContentView(controller: controller)
        } label: {
            Image(systemName: controller.menuBarIcon)
        }
        .menuBarExtraStyle(.window)

        Window("BlaBla", id: "blabla-reader") {
            ReaderRootView()
                .frame(minWidth: 760, minHeight: 560)
                .onAppear { NSApp.activate(ignoringOtherApps: true) }
        }
        .defaultSize(width: 980, height: 720)
        .defaultLaunchBehavior(.presented)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Document…") { openDocumentPanel() }
                    .keyboardShortcut("o")
                Button("Paste from Clipboard") { pasteIntoReader() }
                    .keyboardShortcut("v", modifiers: [.command, .shift])
            }
        }
    }

    private func openDocumentPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            Task { @MainActor in ReaderControllerHolder.reader.loadFileURL(url) }
        }
    }

    private func pasteIntoReader() {
        guard let text = NSPasteboard.general.string(forType: .string)?
            .trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return }
        Task { @MainActor in
            if text.range(of: #"^https?://\S+$"#, options: .regularExpression) != nil {
                ReaderControllerHolder.reader.loadURL(text)
            } else {
                ReaderControllerHolder.reader.loadText(text)
            }
        }
    }
}
