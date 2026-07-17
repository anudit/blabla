//
//  ContentView.swift
//  tts-metal
//
//  Menu-bar popover: transport controls, voice/speed, an optional type-to-speak box,
//  and the Accessibility-permission prompt needed to read the system selection.
//

import SwiftUI

struct ContentView: View {
    @ObservedObject var controller: TtsController
    @State private var draft: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            if !controller.accessibilityGranted {
                accessibilityBanner
            }

            transport

            Divider()

            settings

            Divider()

            typeToSpeak

            footer
        }
        .padding(14)
        .frame(width: 320)
        .onAppear { controller.refreshAccessibility() }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: controller.menuBarIcon)
                .foregroundStyle(controller.statusColor)
            VStack(alignment: .leading, spacing: 1) {
                Text("Kitten TTS · Metal").font(.headline)
                Text(controller.statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
    }

    private var accessibilityBanner: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Accessibility access needed", systemImage: "lock.shield")
                .font(.callout.bold())
            Text("Required to read highlighted text from other apps.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button("Grant Access…") { controller.requestAccessibility() }
                .controlSize(.small)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.yellow.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
    }

    private var transport: some View {
        HStack(spacing: 10) {
            Button(action: { controller.togglePlayPause() }) {
                Label(primaryLabel, systemImage: primaryIcon)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!controller.canControl)

            Button(action: { controller.stop() }) {
                Image(systemName: "stop.fill")
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(!controller.isBusy)
        }
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Voice").frame(width: 48, alignment: .leading)
                Picker("", selection: $controller.voiceKey) {
                    ForEach(TtsConfig.voiceKeys, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden()
            }
            HStack {
                Text("Speed").frame(width: 48, alignment: .leading)
                Slider(value: $controller.speed, in: 0.5...2.0, step: 0.1)
                Text(String(format: "%.1f×", controller.speed))
                    .monospacedDigit()
                    .frame(width: 40, alignment: .trailing)
            }
            Toggle(isOn: $controller.enhanceEnabled) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Enhance audio (RE-USE)")
                    Text(controller.enhancerReady ? "Denoise + upsample to 48 kHz"
                                                   : "Loading model…")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .toggleStyle(.switch)
            .controlSize(.small)
            .disabled(!controller.enhancerReady)
        }
    }

    private var typeToSpeak: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Type to speak").font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .top, spacing: 6) {
                TextEditor(text: $draft)
                    .font(.callout)
                    .frame(height: 54)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.tertiary))
                Button {
                    controller.speak(draft)
                } label: {
                    Image(systemName: "play.fill")
                }
                .buttonStyle(.bordered)
                .disabled(!controller.canControl || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    private var footer: some View {
        HStack {
            Label("Read selection", systemImage: "text.viewfinder")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("⌥⌘R")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            Spacer()
            Button("Quit") { NSApp.terminate(nil) }
                .controlSize(.small)
        }
    }

    // MARK: - Labels

    private var primaryLabel: String {
        switch controller.phase {
        case .speaking, .generating: return "Pause"
        case .paused:                return "Resume"
        default:                     return "Read Selection"
        }
    }

    private var primaryIcon: String {
        switch controller.phase {
        case .speaking, .generating: return "pause.fill"
        case .paused:                return "play.fill"
        default:                     return "text.viewfinder"
        }
    }
}
