//
//  SelectionReader.swift
//  tts-metal
//
//  Reads the current system-wide text selection. Prefers the Accessibility API
//  (does not touch the clipboard); falls back to a synthesized ⌘C copy when the
//  focused app doesn't expose its selection via AX.
//
//  Requires the app to be granted Accessibility access in
//  System Settings ▸ Privacy & Security ▸ Accessibility.
//

import AppKit
import ApplicationServices

enum SelectionReader {
    /// Whether the app currently has Accessibility (AX) trust.
    nonisolated static var isTrusted: Bool { AXIsProcessTrusted() }

    /// Prompt the user to grant Accessibility access (opens the system prompt).
    @discardableResult
    nonisolated static func requestTrust() -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        return AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }

    /// Best-effort read of the current selection. May block briefly (clipboard
    /// fallback waits for the copy to land), so call this off the main thread.
    nonisolated static func currentSelection() -> String? {
        if let viaAX = accessibilitySelection(), !viaAX.isEmpty { return viaAX }
        return clipboardSelection()
    }

    // MARK: - Accessibility path

    private nonisolated static func accessibilitySelection() -> String? {
        let system = AXUIElementCreateSystemWide()
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
              let focused else { return nil }
        // CFTypeRef is an AXUIElement here.
        let element = focused as! AXUIElement
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextAttribute as CFString, &value) == .success
        else { return nil }
        return value as? String
    }

    // MARK: - Clipboard fallback

    private nonisolated static func clipboardSelection() -> String? {
        let pasteboard = NSPasteboard.general
        let previous = pasteboard.string(forType: .string)
        let beforeCount = pasteboard.changeCount

        postCommandC()

        // Wait (up to ~400 ms) for the frontmost app to service the copy.
        var waited = 0
        while pasteboard.changeCount == beforeCount && waited < 400_000 {
            usleep(20_000)
            waited += 20_000
        }
        let copied = (pasteboard.changeCount != beforeCount) ? pasteboard.string(forType: .string) : nil

        // Restore the user's previous clipboard contents.
        if let previous {
            pasteboard.clearContents()
            pasteboard.setString(previous, forType: .string)
        }
        return copied
    }

    private nonisolated static func postCommandC() {
        let source = CGEventSource(stateID: .combinedSessionState)
        let cKey: CGKeyCode = 0x08 // ANSI 'c'
        let down = CGEvent(keyboardEventSource: source, virtualKey: cKey, keyDown: true)
        down?.flags = .maskCommand
        let up = CGEvent(keyboardEventSource: source, virtualKey: cKey, keyDown: false)
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
    }
}
