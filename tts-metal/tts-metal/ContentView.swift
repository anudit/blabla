//
//  ContentView.swift
//  tts-metal
//
//  Native Kitten TTS UI. Loads the largest (mini) model from the app bundle on launch,
//  phonemizes the text, runs the full inference on Metal, and plays back the waveform.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var controller = TtsController()
    @State private var text: String = "Hello! This is Kitten TTS running natively on Metal."
    @State private var voice: String = "Bella"
    @State private var speed: Double = 1.0

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Kitten TTS · Metal")
                .font(.title2.bold())

            HStack {
                statusDot
                Text(controller.statusText)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            HStack {
                Text("Voice")
                Picker("", selection: $voice) {
                    ForEach(TtsConfig.voiceKeys, id: \.self) { v in Text(v).tag(v) }
                }
                .pickerStyle(.menu)
                .frame(width: 130)

                Text("Speed")
                Slider(value: $speed, in: 0.5...2.0, step: 0.1)
                    .frame(width: 200)
                Text(String(format: "%.1f", speed))
                    .monospacedDigit()
                    .frame(width: 30, alignment: .trailing)
            }

            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 120, maxHeight: 180)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.tertiary))

            HStack {
                Button(action: { Task { await controller.generate(text: text, voice: voice, speed: Float(speed)) } }) {
                    if controller.isPlaying {
                        Label("Generating…", systemImage: "circle.circle")
                            .labelStyle(.titleAndIcon)
                            .opacity(0.8)
                    } else {
                        Label("Generate & Play", systemImage: "play.fill")
                            .labelStyle(.titleAndIcon)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!controller.canGenerate)
                .animation(.easeInOut(duration: 0.2), value: controller.isPlaying)

                Button(action: { controller.stop() }) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(.bordered)
                .disabled(!controller.isPlaying)
            }

            if !controller.timingLog.isEmpty {
                timingLogView
            }
        }
        .padding(24)
        .frame(minWidth: 640, minHeight: 480)
    }

    private var statusDot: some View {
        Circle()
            .fill(controller.statusColor)
            .frame(width: 10, height: 10)
    }

    private var timingLogView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Last generation")
                .font(.headline)
            Text("Total: \(String(format: "%.0f ms", controller.lastTotalMs))")
                .font(.callout)
                .monospacedDigit()
            ForEach(controller.timingLog, id: \.name) { stage in
                HStack {
                    Text(stage.name).font(.system(.caption, design: .monospaced))
                    Spacer()
                    Text(String(format: "%.1f ms", stage.ms))
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(12)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
    }
}

#Preview {
    ContentView()
}