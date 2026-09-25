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
    /// Process start time from the kernel, so timestamps include everything before the
    /// first log call (dyld, SwiftUI scene setup) — a lazily-initialized `Date()` here
    /// used to start the clock at the first log line and hide that part of launch.
    private static let start: Date = {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard sysctl(&mib, 4, &info, &size, nil, 0) == 0 else { return Date() }
        let tv = info.kp_proc.p_un.__p_starttime
        return Date(timeIntervalSince1970: Double(tv.tv_sec) + Double(tv.tv_usec) / 1e6)
    }()

    /// Seconds since the process was launched.
    static var sinceLaunch: Double { Date().timeIntervalSince(start) }

    static func log(_ event: String) {
        print(String(format: "[PERF] +%8.3fs  %@", sinceLaunch, event))
    }
}
