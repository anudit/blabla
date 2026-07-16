//
//  GlobalHotKey.swift
//  tts-metal
//
//  A single system-wide hotkey via the Carbon Hot Key API. Fires its action on the
//  main run loop whenever the registered key combination is pressed in any app.
//

import AppKit
import Carbon.HIToolbox

final class GlobalHotKey {
    private var hotKeyRef: EventHotKeyRef?
    private var eventHandler: EventHandlerRef?
    private nonisolated let action: @Sendable () -> Void

    /// - Parameters:
    ///   - keyCode: a virtual key code (e.g. `kVK_ANSI_R`).
    ///   - modifiers: Carbon modifier mask (`cmdKey`, `optionKey`, `controlKey`, `shiftKey`).
    init?(keyCode: UInt32, modifiers: UInt32, action: @escaping @Sendable () -> Void) {
        self.action = action

        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                                      eventKind: UInt32(kEventHotKeyPressed))
        let selfPtr = Unmanaged.passUnretained(self).toOpaque()
        let installStatus = InstallEventHandler(
            GetEventDispatcherTarget(),
            { _, _, userData -> OSStatus in
                guard let userData else { return OSStatus(eventNotHandledErr) }
                Unmanaged<GlobalHotKey>.fromOpaque(userData).takeUnretainedValue().action()
                return noErr
            },
            1, &eventType, selfPtr, &eventHandler)
        guard installStatus == noErr else { return nil }

        let hotKeyID = EventHotKeyID(signature: OSType(0x54545348), id: 1) // 'TTSH'
        let registerStatus = RegisterEventHotKey(keyCode, modifiers, hotKeyID,
                                                 GetEventDispatcherTarget(), 0, &hotKeyRef)
        guard registerStatus == noErr else {
            if let eventHandler { RemoveEventHandler(eventHandler) }
            return nil
        }
    }

    deinit {
        if let hotKeyRef { UnregisterEventHotKey(hotKeyRef) }
        if let eventHandler { RemoveEventHandler(eventHandler) }
    }
}
