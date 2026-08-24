//
//  ReaderController.swift
//  tts-metal
//
//  Drives the BlaBla document reader: loads documents (file / URL / clipboard /
//  OCR), streams sentences through the Supertonic Metal engine with a 3-sentence
//  look-ahead, schedules gapless tagged playback, tracks the active sentence for
//  karaoke highlighting + auto-scroll, and persists auto-resume bookmarks.
//
//  Ported from blabla's App.tsx playback orchestration.
//

import Foundation
import SwiftUI
import Combine
import AppKit

@MainActor
final class ReaderController: ObservableObject {

    // MARK: - Published state

    enum TransportState: Equatable {
        case empty          // no document
        case ready          // doc loaded, not playing
        case generating     // synthesizing ahead of playback
        case playing
        case paused
        case loadingDoc     // parsing/fetching/OCR in progress
        case failed(String)
    }

    @Published private(set) var state: TransportState = .empty
    @Published private(set) var statusText = ""
    @Published private(set) var document: ReaderDocument?
    /// Sentence id currently being spoken (or next to be spoken when paused).
    @Published private(set) var currentIndex = 0
    /// Word index within the active sentence for karaoke highlighting.
    @Published private(set) var activeWordIndex = -1
    @Published var outlineVisible = false
    @Published var miniPlayerVisible = false {
        didSet { MiniPlayerWindow.shared.setVisible(miniPlayerVisible) }
    }

    // Persisted settings.
    @Published var speed: Double {
        didSet {
            UserDefaults.standard.set(speed, forKey: "readerSpeed")
            if oldValue != speed && isSpeaking { restartCurrentSentence() }
        }
    }
    @Published var volume: Double {
        didSet {
            UserDefaults.standard.set(volume, forKey: "readerVolume")
            AudioPlayer.shared.volume = Float(volume)
        }
    }
    @Published var fontSize: Double {
        didSet { UserDefaults.standard.set(fontSize, forKey: "readerFontSize") }
    }
    @Published var voice: String {
        didSet {
            UserDefaults.standard.set(voice, forKey: "readerVoice")
            if oldValue != voice && isSpeaking { restartCurrentSentence() }
        }
    }
    @Published var themeName: String {
        didSet { UserDefaults.standard.set(themeName, forKey: "readerTheme") }
    }
    var theme: ReaderTheme { ReaderTheme.named(themeName) }

    let engineHub = EngineHub.shared
    private let audio = AudioPlayer.shared
    private let bookmarks = BookmarkStore.shared

    // MARK: - Playback internals

    private var speakTask: Task<Void, Never>?
    private var sessionGen = 0
    private var cache: [Int: [Float]] = [:]
    private var wordTimings: [WordTiming] = []
    private var karaokeTimer: Timer?
    private var sentenceStart: Date?
    private var sentenceDuration: Double = 0
    private var pendingResumeIndex: Int?
    private var bookmarkSaveTask: Task<Void, Never>?

    static let speedChoices: [Double] = [1.0, 1.25, 1.5, 1.75, 2.0]
    static let lookahead = 3           // prefetch window (blabla's LOOKAHEAD)
    static let backpressureLimit = 5   // sliding-window queue depth

    var isSpeaking: Bool { state == .playing || state == .generating }

    init() {
        speed = UserDefaults.standard.object(forKey: "readerSpeed") as? Double ?? 1.0
        volume = UserDefaults.standard.object(forKey: "readerVolume") as? Double ?? 0.8
        fontSize = UserDefaults.standard.object(forKey: "readerFontSize") as? Double ?? 1.0
        voice = UserDefaults.standard.string(forKey: "readerVoice") ?? "M1"
        themeName = UserDefaults.standard.string(forKey: "readerTheme") ?? "Original"

        AudioPlayer.shared.volume = Float(volume)

        // Media keys / Now Playing
        NowPlayingManager.shared.register()
        NowPlayingManager.shared.onTogglePlayback = { [weak self] in self?.togglePlayPause() }
        NowPlayingManager.shared.onNextSentence = { [weak self] in self?.skipSentence(+1) }
        NowPlayingManager.shared.onPrevSentence = { [weak self] in self?.skipSentence(-1) }

        Task { await waitForEngine() }
    }

    private func waitForEngine() async {
        engineHub.ensureLoaded()
        if !engineHub.ready {
            statusText = engineHub.statusText
            while !engineHub.ready && !engineHub.failed {
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
            statusText = engineHub.statusText
        }
    }

    var progress: Double {
        guard let doc = document, !doc.sentences.isEmpty else { return 0 }
        return Double(currentIndex) / Double(doc.sentences.count)
    }

    var currentSentence: RSentence? {
        guard let doc = document,
              currentIndex >= 0, currentIndex < doc.sentences.count else { return nil }
        return doc.sentences[currentIndex]
    }

    // MARK: - Document loading

    func loadFileURL(_ url: URL) {
        Task { await loadDocument(.file(url)) }
    }

    func loadURL(_ urlString: String) {
        Task { await loadDocument(.url(urlString)) }
    }

    func loadText(_ text: String, title: String = "Pasted text") {
        Task { await loadDocument(.text(text, title)) }
    }

    enum LoadRequest {
        case file(URL)
        case url(String)
        case text(String, String)
    }

    private func loadDocument(_ request: LoadRequest) async {
        PerfLog.log("loadDocument start")
        stopPlayback()
        state = .loadingDoc
        statusText = "Loading…"

        do {
            var doc: ReaderDocument

            switch request {
            case .file(let url):
                // Parsing large books (EPUB/PDF) is CPU-heavy — keep it off the main actor.
                doc = try await Task.detached(priority: .userInitiated) {
                    PerfLog.log("DocLoader.load start (background thread=\(!Thread.isMainThread))")
                    let d = try DocLoader.load(fileURL: url)
                    PerfLog.log("DocLoader.load done — \(d.sentences.count) sentences")
                    return d
                }.value
            case .url(let s):
                statusText = "Fetching page…"
                doc = try await DocLoader.load(urlString: s)
            case .text(let t, let title):
                doc = DocLoader.loadText(t, title: title)
            }

            setDocument(doc)
            statusText = "\(doc.sentences.count) sentences · \(doc.fileType.label)"
            state = .ready
            PerfLog.log("state = .ready, resume index = \(currentIndex)")
            upgradeDocumentInBackground(sourceID: doc.sourceID)
        } catch {
            state = .failed(error.localizedDescription)
            statusText = error.localizedDescription
        }
    }

    /// Fills in fully-normalized (money/date/time/phone/version/ordinal
    /// expanded) sentence text after the fast-normalized document is already
    /// on screen. Playback never depends on this finishing — it applies
    /// `expandNumericForms` itself at the point of synthesis — this only
    /// upgrades what's displayed. Ignored if a different document has since
    /// been opened.
    private func upgradeDocumentInBackground(sourceID: String) {
        Task.detached(priority: .utility) { [weak self] in
            guard let self, let base = await self.document, base.sourceID == sourceID else { return }
            let upgraded = base.withFullyNormalizedSentences()
            await MainActor.run {
                guard self.document?.sourceID == sourceID else { return }
                self.document = upgraded
                PerfLog.log("background full-normalize upgrade applied")
            }
        }
    }

    private func sourceIDForFile(_ url: URL) -> String {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
        return "\(url.lastPathComponent):\(size)"
    }

    func resetDocument() {
        stopPlayback()
        document = nil
        currentIndex = 0
        activeWordIndex = -1
        cache.removeAll()
        state = .empty
        statusText = ""
        NowPlayingManager.shared.clear()
    }

    private func setDocument(_ doc: ReaderDocument) {
        document = doc
        cache.removeAll()
        activeWordIndex = -1
        currentIndex = 0

        // Auto-resume from bookmark history.
        if let entry = bookmarks.entry(for: doc.sourceID),
           entry.sentenceIndex < doc.sentences.count {
            pendingResumeIndex = entry.sentenceIndex
            currentIndex = entry.sentenceIndex
            statusText = "Resuming at sentence \(entry.sentenceIndex + 1)"
        }
    }

    // MARK: - Transport

    func togglePlayPause() {
        switch state {
        case .playing, .generating:
            pause()
        case .paused:
            resume()
        case .ready:
            startPlaying(from: currentIndex)
        case .empty, .loadingDoc:
            break
        case .failed:
            break
        }
    }

    func playFrom(_ index: Int) {
        guard let doc = document, doc.sentences.indices.contains(index), engineHub.ready else { return }
        if isSpeaking || state == .paused {
            restartAt(index)
        } else {
            startPlaying(from: index)
        }
    }

    func pause() {
        audio.pause()
        state = .paused
        statusText = "Paused"
        stopKaraoke()
        saveBookmarkNow()
        NowPlayingManager.shared.setPaused(true)
    }

    func resume() {
        audio.resume()
        state = audio.isActive ? .playing : .generating
        statusText = "Reading…"
        startKaraoke()
        NowPlayingManager.shared.setPaused(false)
    }

    func skipSentence(_ delta: Int) {
        guard let doc = document else { return }
        let target = max(0, min(doc.sentences.count - 1, currentIndex + delta))
        playFrom(target)
    }

    func stopPlayback() {
        speakTask?.cancel()
        speakTask = nil
        audio.stop()
        sessionGen = audio.currentGeneration
        cache.removeAll()
        stopKaraoke()
        NowPlayingManager.shared.clear()
        if let doc = document {
            state = .ready
            statusText = "\(doc.sentences.count) sentences"
        } else {
            state = .empty
        }
    }

    /// Voice/speed changed while speaking → resynthesize current sentence.
    private func restartCurrentSentence() {
        restartAt(currentIndex)
    }

    private func restartAt(_ index: Int) {
        speakTask?.cancel()
        audio.stop()
        sessionGen = audio.currentGeneration
        cache.removeValue(forKey: index)   // force regeneration with new settings
        stopKaraoke()
        currentIndex = index
        startPlaying(from: index)
    }

    private func startPlaying(from index: Int) {
        PerfLog.log("startPlaying requested at index \(index) (engine ready=\(engineHub.ready))")
        guard let doc = document, doc.sentences.indices.contains(index) else { return }
        guard engineHub.ready else {
            statusText = "Model still loading…"
            return
        }

        speakTask?.cancel()
        audio.stop()
        sessionGen = audio.currentGeneration
        stopKaraoke()
        currentIndex = index
        state = .generating
        statusText = "Reading…"
        NowPlayingManager.shared.update(title: doc.fileName,
                                        artist: "\(voice) · supertonic",
                                        index: index, total: doc.sentences.count)
        NowPlayingManager.shared.setPaused(false)

        let gen = sessionGen
        let stVoice = voice
        let rate = Float(speed)
        let outRate = TtsConfig.enhancedSampleRate

        speakTask = Task { @MainActor [weak self] in
            guard let self else { return }
            var nextToSchedule = index

            while !Task.isCancelled {
                // All scheduled buffers done & nothing more to synthesize → finished.
                if nextToSchedule >= doc.sentences.count && !self.audio.isActive { break }

                if nextToSchedule < doc.sentences.count {
                    // Sliding-window backpressure against playback.
                    await self.audio.reserveSlot(limit: Self.backpressureLimit)
                    if Task.isCancelled || gen != self.audio.currentGeneration { return }

                    let idx = nextToSchedule
                    nextToSchedule += 1

                    // Applied here (not relied upon from the background
                    // upgrade pass) so synthesis and karaoke are always
                    // fully-normalized regardless of that pass's timing —
                    // idempotent, so a no-op once the background pass catches up.
                    let ttsText = TTSTextNormalizer.expandNumericForms(doc.sentences[idx].text)

                    var wave = self.cache[idx] ?? []
                    if wave.isEmpty {
                        if idx == index { PerfLog.log("synthesizing first sentence (idx \(idx))") }
                        let pieces = SentenceSplitter.splitChunk(ttsText, limit: 280)
                        do {
                            var joined: [Float] = []
                            for piece in pieces {
                                let w = try await self.engineHub.generate(piece, voice: stVoice, speed: rate)
                                joined.append(contentsOf: w)
                                joined.append(contentsOf: silence(forBoundaryAfter: piece, sampleRate: outRate))
                                if Task.isCancelled || gen != self.audio.currentGeneration { return }
                            }
                            edgeFade(&joined)
                            wave = joined
                            self.cache[idx] = wave
                            self.pruneCache(around: idx)
                        } catch {
                            self.statusText = "Generation error: \(error.localizedDescription)"
                            continue
                        }
                    }

                    if !wave.isEmpty {
                        let isFirst = (idx == self.currentIndex && !self.audio.isActive)
                        self.audio.enqueueTagged(wave, tag: idx) { [weak self] finishedTag in
                            self?.onSentenceFinished(finishedTag, total: doc.sentences.count, gen: gen)
                        }
                        if isFirst {
                            PerfLog.log("playback started (first buffer enqueued, idx \(idx))")
                            self.state = .playing
                            self.beginKaraoke(text: ttsText, samples: wave.count, rate: outRate)
                        }
                        self.statusProgress(idx, total: doc.sentences.count)
                        self.saveBookmarkDebounced()
                    }
                } else {
                    // Waiting for tail playback to finish.
                    try? await Task.sleep(nanoseconds: 120_000_000)
                    if gen != self.audio.currentGeneration { return }
                }
            }

            if !Task.isCancelled && gen == self.audio.currentGeneration && !self.isPausedState {
                self.finish()
            }
        }
    }

    private var isPausedState: Bool { state == .paused }

    private func onSentenceFinished(_ tag: Int, total: Int, gen: Int) {
        guard gen == sessionGen, tag == currentIndex else { return }
        let next = tag + 1
        if next >= total {
            finish()
            return
        }
        currentIndex = next
        activeWordIndex = -1

        if let doc = document, doc.sentences.indices.contains(next) {
            let ttsText = TTSTextNormalizer.expandNumericForms(doc.sentences[next].text)
            beginKaraoke(text: ttsText,
                         samples: cache[next]?.count ?? 0,
                         rate: TtsConfig.enhancedSampleRate)
        }
        NowPlayingManager.shared.update(title: document?.fileName ?? "",
                                        artist: "\(voice) · supertonic",
                                        index: next, total: total)
        saveBookmarkDebounced()
    }

    private func finish() {
        stopKaraoke()
        state = .ready
        statusText = "Finished — \(document?.sentences.count ?? 0) sentences"
        activeWordIndex = -1
        NowPlayingManager.shared.clear()
    }

    // MARK: - Karaoke highlighting

    /// `text` must be the exact fully-normalized text used for synthesis
    /// (not necessarily `sentence.text`, whose displayed form may still be
    /// mid-upgrade in the background) so word timings stay in sync with audio.
    private func beginKaraoke(text: String, samples: Int, rate: Double) {
        wordTimings = WordTimingCalculator.timings(for: text)
        sentenceDuration = samples > 0 ? Double(samples) / rate : estimateDuration(text)
        sentenceStart = Date()
        startKaraoke()
    }

    private func estimateDuration(_ text: String) -> Double {
        max(0.4, Double(text.count) * 0.062 / speed)
    }

    private func startKaraoke() {
        karaokeTimer?.invalidate()
        // 60 fps for smooth word-to-word progression (avoids skipping short words)
        let timer = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tickKaraoke() }
        }
        RunLoop.main.add(timer, forMode: .common)
        karaokeTimer = timer
    }

    private func stopKaraoke() {
        karaokeTimer?.invalidate()
        karaokeTimer = nil
        sentenceStart = nil
    }

    private func tickKaraoke() {
        guard state == .playing, let start = sentenceStart else { return }
        let frac = sentenceDuration > 0 ? Date().timeIntervalSince(start) / sentenceDuration : 0
        let clampedFrac = max(0, min(0.999, frac))
        // Quantize to word index by startFrac — advance one word at a time
        // (no skipping) even if a tick straddles a boundary.
        var idx = wordTimings.count - 1
        for (i, t) in wordTimings.enumerated() where clampedFrac < t.endFrac {
            idx = i
            break
        }
        // No +1-per-tick clamp here: at higher playback speeds a sentence's
        // real duration can shrink below wordCount/60s, so a hard per-tick
        // cap would make the highlight permanently unable to catch up to the
        // real audio position — it would trail off and never reach the final
        // word(s) before the sentence ends and activeWordIndex resets.
        if activeWordIndex != idx {
            activeWordIndex = idx
        }
    }

    // MARK: - Audio helpers (port of blabla's boundary pauses + edge fade)

    /// Silence inserted after a chunk based on its trailing punctuation.
    private func silence(forBoundaryAfter piece: String, sampleRate: Double) -> [Float] {
        let seconds: Double
        if piece.hasSuffix("?") { seconds = 0.28 }
        else if piece.hasSuffix("!") { seconds = 0.24 }
        else if piece.hasSuffix(".") || piece.hasSuffix("。") { seconds = 0.22 }
        else if piece.hasSuffix(";") { seconds = 0.16 }
        else if piece.hasSuffix(":") { seconds = 0.13 }
        else if piece.hasSuffix(",") { seconds = 0.09 }
        else { seconds = 0.08 }
        let n = Int(seconds * sampleRate)
        return n > 0 ? [Float](repeating: 0, count: n) : []
    }

    /// 20 ms raised-cosine (equal-power) fade at both edges so consecutive
    /// sentence buffers blend into each other instead of clicking at the
    /// splice point. A short linear ramp still has an abrupt slope change at
    /// its ends, which is audible as a click; the cosine curve tapers the
    /// envelope's derivative to zero too, so back-to-back buffers (one fading
    /// out, the next fading in) meet smoothly.
    private func edgeFade(_ samples: inout [Float]) {
        let fadeLen = min(Int(0.020 * TtsConfig.enhancedSampleRate), samples.count / 2)
        guard fadeLen > 0 else { return }
        for i in 0..<fadeLen {
            let g = Float(0.5 * (1 - cos(Double.pi * Double(i) / Double(fadeLen))))
            samples[i] *= g
            samples[samples.count - 1 - i] *= g
        }
    }

    private func pruneCache(around index: Int) {
        let keep = Set((index - 1)...(index + Self.lookahead))
        cache = cache.filter { keep.contains($0.key) }
    }

    private func statusProgress(_ index: Int, total: Int) {
        statusText = "Sentence \(index + 1) of \(total)"
    }

    // MARK: - Bookmarks

    private func saveBookmarkDebounced() {
        bookmarkSaveTask?.cancel()
        bookmarkSaveTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            guard !Task.isCancelled else { return }
            self?.saveBookmarkNow()
        }
    }

    private func saveBookmarkNow() {
        guard let doc = document else { return }
        let preview = doc.sentences.map(\.text).prefix(3).joined(separator: " ")
        BookmarkStore.shared.save(entry: BookmarkEntry(
            id: doc.sourceID,
            fileName: doc.fileName,
            sentenceIndex: currentIndex,
            totalSentences: doc.sentences.count,
            timestamp: Date(),
            fileType: doc.fileType.rawValue,
            preview: String(preview.prefix(80)),
            url: doc.previewURL,
            ocrPage: doc.sentencePages.indices.contains(currentIndex) ? doc.sentencePages[currentIndex] : nil,
            filePath: doc.sourceFilePath
        ))
    }

    // MARK: - Test voice

    func testVoice() {
        guard engineHub.ready else { return }
        speakTask?.cancel()
        audio.stop()
        sessionGen = audio.currentGeneration
        state = .ready
        Task { @MainActor in
            if let wave = try? await engineHub.generate("Hello! I am ready to read.",
                                                        voice: voice,
                                                        speed: Float(speed)), !wave.isEmpty {
                audio.enqueueTagged(wave, tag: -1)
            }
        }
    }
}
