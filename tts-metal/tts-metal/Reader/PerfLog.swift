//
//  PerfLog.swift
//  tts-metal
//
//  Temporary timing instrumentation for the "click resume → book renders →
//  scrolls to the resume line → playback starts" pipeline. Prints to stdout
//  with a timestamp relative to app launch so consecutive lines show elapsed
//  duration directly. Remove once the reported slowness is diagnosed.
//

import Foundation

enum PerfLog {
    private static let start = Date()

    static func log(_ event: String) {
        let elapsed = Date().timeIntervalSince(start)
        print(String(format: "[PERF] +%8.3fs  %@", elapsed, event))
    }
}
