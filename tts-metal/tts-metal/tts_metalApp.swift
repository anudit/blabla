//
//  tts_metalApp.swift
//  tts-metal
//
//  Created by Anudit Nagar on 15/07/26.
//
//  Menu-bar (agent) app: no dock icon, no window — a single status-bar item whose
//  icon reflects playback state and whose popover hosts the controls.
//

import SwiftUI

@main
struct tts_metalApp: App {
    @StateObject private var controller = TtsController()

    var body: some Scene {
        MenuBarExtra {
            ContentView(controller: controller)
        } label: {
            Image(systemName: controller.menuBarIcon)
        }
        .menuBarExtraStyle(.window)
    }
}
