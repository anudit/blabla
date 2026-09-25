//
//  Benchmark.swift
//  tts-metal
//
//  End-to-end latency / RTF / memory benchmark (SUPERTONIC_BENCH=1).
//
//  Drives EngineHub.generate exactly as the readers do (engine queue hop, synthesis,
//  resample to 48 kHz), so "latency" here is time-to-first-audio for a sentence:
//  the moment its buffer could be handed to AVAudioEngine.
//
//     SUPERTONIC_BENCH=1 Blabla                       launch → ready → first sentence,
//                                                     then steady state once ANE is warm
//     SUPERTONIC_BENCH=1 SUPERTONIC_ANE=0 Blabla      Metal-only steady state
//     SUPERTONIC_BENCH_REPS=N                         steady-state reps per text (default 6)
//     SUPERTONIC_BENCH_DUMP=/path/prefix              write each text's first steady-state
//                                                     output as raw float32 (A/B checks;
//                                                     pair with SUPERTONIC_SEED)
//
//  Prints one "[BENCH] ..." line per measurement and exits.
//
//  Energy is what the kernel bills to this process: CPU energy (rusage ri_energy_nj) and
//  CPU time. GPU and ANE energy are not attributed per process on macOS; measure
//  whole-system power with `sudo powermetrics --samplers cpu_power,gpu_power,ane_power`.
//

import Foundation

@MainActor
enum Benchmark {

    static let texts: [(String, String)] = [
        ("short", "Short and sweet."),
        ("medium", "A gentle breeze moved through the open window while everyone listened to the story."),
        ("long", "The afternoon light fell across the floorboards in long pale stripes, and somewhere outside a dog barked twice and then fell silent again, leaving the room quieter than before it had started."),
    ]

    static func run() async {
        setvbuf(stdout, nil, _IONBF, 0)
        let env = ProcessInfo.processInfo.environment
        let reps = env["SUPERTONIC_BENCH_REPS"].flatMap { Int($0) } ?? 6
        let hub = EngineHub.shared
        hub.ensureLoaded()
        while !hub.ready && !hub.failed { try? await Task.sleep(nanoseconds: 2_000_000) }
        guard hub.ready, let eng = hub.engine else { print("[BENCH] engine failed to load"); exit(1) }
        let readyAt = PerfLog.sinceLaunch
        print(String(format: "[BENCH] launch_to_ready_s=%.3f", readyAt))
        var mark = Usage.now()
        report("launch_to_ready", Usage.zero, mark)

        // A sentence requested the instant the engine is ready: what a user who presses
        // play right after launch waits for.
        let t0 = PerfLog.sinceLaunch
        let first = (try? await hub.generate(texts[1].1, voice: "M1", speed: 1.0)) ?? []
        let t1 = PerfLog.sinceLaunch
        print(String(format: "[BENCH] first_sentence latency_ms=%.1f launch_to_first_audio_s=%.3f backend=%@ audio_s=%.2f",
                     (t1 - t0) * 1000, t1, eng.lastFlowBackend, Double(first.count) / TtsConfig.enhancedSampleRate))

        if env["SUPERTONIC_ANE"] != "0" {
            while !eng.aneWarmupComplete { try? await Task.sleep(nanoseconds: 20_000_000) }
            print(String(format: "[BENCH] ane_launch_cells_warm_s=%.3f", PerfLog.sinceLaunch))
            // Then the cells for the benchmark texts, as the readers request for look-ahead.
            await hub.prefetch(texts.map { $0.1 }, speed: 1.0)
            while !eng.aneWarmupComplete { try? await Task.sleep(nanoseconds: 20_000_000) }
            print(String(format: "[BENCH] ane_bench_cells_warm_s=%.3f", PerfLog.sinceLaunch))
        }
        let beforeWarmup = mark
        mark = Usage.now()
        report("first_sentence+ane_warmup", beforeWarmup, mark)
        var steadyAudio = 0.0

        for (name, text) in texts {
            var lat: [Double] = []
            var audio = 0.0
            var stages: [String: Double] = [:]
            for i in 0..<(reps + 1) {
                let a = PerfLog.sinceLaunch
                let w = (try? await hub.generate(text, voice: "M1", speed: 1.0)) ?? []
                let ms = (PerfLog.sinceLaunch - a) * 1000
                if i == 0 { continue }   // first run of each shape pays one-time costs
                if i == 1, let prefix = env["SUPERTONIC_BENCH_DUMP"] {
                    try? w.withUnsafeBytes { Data($0) }.write(to: URL(fileURLWithPath: "\(prefix)_\(name).f32"))
                }
                lat.append(ms)
                audio = Double(w.count) / TtsConfig.enhancedSampleRate
                steadyAudio += audio
                for s in eng.lastTimings { stages[s.name, default: 0] += s.ms / Double(reps) }
            }
            lat.sort()
            let med = lat[lat.count / 2]
            let stageStr = ["text_encoder", "duration_predictor", "flow_matching", "vocoder"]
                .map { String(format: "%@=%.1f", $0, stages[$0] ?? 0) }.joined(separator: " ")
            print(String(format: "[BENCH] steady %@ chars=%d audio_s=%.2f median_ms=%.1f min_ms=%.1f max_ms=%.1f rtf=%.1fx backend=%@ %@",
                         name, text.count, audio, med, lat.first!, lat.last!, audio / (med / 1000),
                         eng.lastFlowBackend, stageStr))
        }

        let end = Usage.now()
        report("steady_state", mark, end, audioSeconds: steadyAudio)

        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        _ = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        print(String(format: "[BENCH] memory footprint_mb=%.0f peak_footprint_mb=%.0f",
                     Double(info.phys_footprint) / 1_048_576, Double(info.ledger_phys_footprint_peak) / 1_048_576))
        exit(0)
    }

    private static func report(_ phase: String, _ a: Usage, _ b: Usage, audioSeconds: Double = 0) {
        let mj = Double(b.energyNJ - a.energyNJ) / 1e6
        var line = String(format: "[BENCH] energy %@ cpu_energy_mj=%.0f cpu_time_ms=%.0f",
                          phase, mj, (b.cpuSec - a.cpuSec) * 1000)
        if audioSeconds > 0 {
            line += String(format: " audio_s=%.1f cpu_mj_per_audio_s=%.1f", audioSeconds, mj / audioSeconds)
        }
        print(line)
    }

    struct Usage {
        var cpuSec = 0.0
        var energyNJ: UInt64 = 0
        static let zero = Usage()

        static func now() -> Usage {
            var u = Usage()
            var ri = rusage_info_v6()
            let ok = withUnsafeMutablePointer(to: &ri) {
                $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                    proc_pid_rusage(getpid(), RUSAGE_INFO_V6, $0)
                }
            }
            if ok == 0 {
                var tb = mach_timebase_info_data_t(); mach_timebase_info(&tb)
                let ticks = Double(ri.ri_user_time + ri.ri_system_time)
                u.cpuSec = ticks * Double(tb.numer) / Double(tb.denom) / 1e9
                u.energyNJ = ri.ri_energy_nj
            }
            return u
        }
    }
}
