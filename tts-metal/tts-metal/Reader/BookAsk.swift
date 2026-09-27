import Foundation
import SwiftUI
import Combine
import CoreML
import CryptoKit
import FoundationModels
import USearch

private nonisolated struct BookPassage: Codable {
    let firstSentence: Int
    let lastSentence: Int
    let text: String
}

private nonisolated struct StoredPassages: Codable {
    let version: Int
    let sourceID: String
    let passages: [BookPassage]
}

nonisolated struct AskSource: Identifiable {
    let id: Int
    let sentenceID: Int
    let excerpt: String
}

private nonisolated enum AskError: LocalizedError {
    case missingAsset(String)
    case emptyBook
    case modelUnavailable
    case emptyResponse

    var errorDescription: String? {
        switch self {
        case .missingAsset(let name): return "Missing embedding asset: \(name)"
        case .emptyBook: return "This book has no searchable text."
        case .modelUnavailable: return "Apple Intelligence is unavailable on this Mac."
        case .emptyResponse: return "The model returned an empty answer. Please try again."
        }
    }
}

/// The model's BertNormalizer + WordPiece tokenizer, using its bundled vocabulary.
private nonisolated final class PotionTokenizer {
    private let vocabulary: [String: Int]
    private let unknown: Int

    init() throws {
        guard let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json") else {
            throw AskError.missingAsset("tokenizer.json")
        }
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        let model = root?["model"] as? [String: Any]
        guard let vocabulary = model?["vocab"] as? [String: Int] else {
            throw AskError.missingAsset("tokenizer vocabulary")
        }
        self.vocabulary = vocabulary
        self.unknown = vocabulary["[UNK]"] ?? 100
    }

    func encode(_ text: String) -> [Int32] {
        let normalized = text.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: Locale(identifier: "en_US_POSIX"))
        let pattern = #"[\p{Han}]|[\p{L}\p{N}]+|[^\s\p{L}\p{N}]"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let ns = normalized as NSString
        var ids: [Int32] = []
        for match in regex.matches(in: normalized, range: NSRange(location: 0, length: ns.length)) {
            let word = ns.substring(with: match.range)
            if let id = vocabulary[word] { if id != unknown { ids.append(Int32(id)) }; continue }
            let scalars = Array(word.unicodeScalars.map(String.init))
            if scalars.count > 100 { continue }
            var position = 0
            var pieces: [Int32] = []
            while position < scalars.count {
                var found: (Int32, Int)?
                for end in stride(from: scalars.count, through: position + 1, by: -1) {
                    let piece = (position == 0 ? "" : "##") + scalars[position..<end].joined()
                    if let id = vocabulary[piece] { found = (Int32(id), end); break }
                }
                guard let (id, end) = found else { pieces = [Int32(unknown)]; break }
                pieces.append(id)
                position = end
            }
            ids.append(contentsOf: pieces.filter { $0 != unknown })
        }
        return ids
    }
}

private nonisolated final class PotionEmbedder {
    private let tokenizer: PotionTokenizer
    private let model: MLModel

    init() throws {
        tokenizer = try PotionTokenizer()
        guard let url = Bundle.main.url(forResource: "PotionEmbedding", withExtension: "mlmodelc") else {
            throw AskError.missingAsset("PotionEmbedding.mlmodelc")
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: config)
    }

    func embed(_ text: String) throws -> [Float] {
        let tokens = tokenizer.encode(text)
        let counts = try MLMultiArray(shape: [1, 29528], dataType: .float16)
        for token in tokens {
            let n = Int(token)
            counts[n] = NSNumber(value: counts[n].floatValue + 1)
        }
        let input = try MLDictionaryFeatureProvider(dictionary: ["token_counts": counts])
        let output = try model.prediction(from: input)
        guard let vector = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw AskError.missingAsset("embedding output")
        }
        return (0..<256).map { vector[$0].floatValue }
    }
}

private actor BookSemanticIndex {
    private struct Unit {
        let sentenceID: Int
        let blockIndex: Int
        let text: String
        let words: Int
    }

    private static let indexVersion = 2
    private static let searchStopWords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does",
        "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "the", "this",
        "to", "was", "were", "what", "when", "where", "which", "who", "why", "with"
    ]
    private var embedder: PotionEmbedder?
    private var index: USearchIndex?
    private var passages: [BookPassage] = []
    private var loadedID: String?

    private func directory(for sourceID: String) throws -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("BlaBla/BookIndexes", isDirectory: true)
        let digest = SHA256.hash(data: Data(sourceID.utf8)).map { String(format: "%02x", $0) }.joined()
        let dir = base.appendingPathComponent(digest, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func makePassages(_ sentences: [RSentence]) -> [BookPassage] {
        // Keep a passage near 90 words, at most 125. A short sentence overlap
        // preserves context across windows without flooding retrieval with copies.
        var units: [Unit] = []
        for sentence in sentences {
            let words = sentence.displayText.split(whereSeparator: \.isWhitespace).map(String.init)
            guard !words.isEmpty else { continue }
            if words.count <= 125 {
                units.append(Unit(sentenceID: sentence.id, blockIndex: sentence.blockIndex,
                                  text: words.joined(separator: " "), words: words.count))
            } else {
                // A single long sentence still needs bounded embedding windows.
                var start = 0
                while start < words.count {
                    let end = min(start + 100, words.count)
                    units.append(Unit(sentenceID: sentence.id, blockIndex: sentence.blockIndex,
                                      text: words[start..<end].joined(separator: " "), words: end - start))
                    if end == words.count { break }
                    start = end - 20
                }
            }
        }

        var result: [BookPassage] = []
        var window: [Unit] = []
        var freshWords = 0

        func emit() {
            guard !window.isEmpty, freshWords > 0 else { return }
            result.append(BookPassage(firstSentence: window[0].sentenceID,
                                      lastSentence: window[window.count - 1].sentenceID,
                                      text: window.map(\.text).joined(separator: " ")))
            var overlap: [Unit] = []
            var overlapWords = 0
            for unit in window.reversed() {
                if overlapWords + unit.words > 25 { break }
                overlap.insert(unit, at: 0)
                overlapWords += unit.words
            }
            window = overlap
            freshWords = 0
        }

        for unit in units {
            let currentWords = window.reduce(0) { $0 + $1.words }
            if let previous = window.last,
               previous.blockIndex != unit.blockIndex, freshWords >= 45 {
                emit()
            } else if currentWords + unit.words > 125, freshWords > 0 {
                emit()
            }
            if window.reduce(0, { $0 + $1.words }) + unit.words > 125 {
                window.removeAll()
            }
            window.append(unit)
            freshWords += unit.words
            if freshWords >= 90 { emit() }
        }
        emit()
        return result
    }

    func prepare(sourceID: String, sentences: [RSentence]) throws -> Int {
        if loadedID == sourceID { return passages.count }
        let embedder = try self.embedder ?? PotionEmbedder()
        self.embedder = embedder
        let dir = try directory(for: sourceID)
        let indexURL = dir.appendingPathComponent("passages.usearch")
        let textURL = dir.appendingPathComponent("passages.json")
        var index = try USearchIndex.make(metric: .cos, dimensions: 256, connectivity: 16, quantization: .f32)
        if let data = try? Data(contentsOf: textURL),
           let stored = try? JSONDecoder().decode(StoredPassages.self, from: data),
           stored.sourceID == sourceID, stored.version == Self.indexVersion,
           FileManager.default.fileExists(atPath: indexURL.path) {
            do {
                try index.load(path: indexURL.path)
                if (try index.count) == stored.passages.count {
                    self.index = index
                    passages = stored.passages
                    loadedID = sourceID
                    return stored.passages.count
                }
            } catch { /* Rebuild an incomplete or outdated index. */ }
            index = try USearchIndex.make(metric: .cos, dimensions: 256, connectivity: 16, quantization: .f32)
        }
        let chunks = makePassages(sentences)
        guard !chunks.isEmpty else { throw AskError.emptyBook }
        try index.reserve(UInt32(chunks.count))
        for (number, chunk) in chunks.enumerated() {
            let vector = try embedder.embed(chunk.text)
            try index.add(key: UInt64(number), vector: vector)
        }
        let tempIndex = dir.appendingPathComponent("passages.tmp.usearch")
        try index.save(path: tempIndex.path)
        let encoded = try JSONEncoder().encode(StoredPassages(version: Self.indexVersion,
                                                               sourceID: sourceID, passages: chunks))
        try encoded.write(to: textURL, options: .atomic)
        if FileManager.default.fileExists(atPath: indexURL.path) { try FileManager.default.removeItem(at: indexURL) }
        try FileManager.default.moveItem(at: tempIndex, to: indexURL)
        self.index = index
        passages = chunks
        loadedID = sourceID
        return chunks.count
    }

    func retrieve(_ question: String, count: Int = 20) throws -> [BookPassage] {
        guard let embedder, let index else { throw AskError.emptyBook }
        let vector = try embedder.embed(question)
        let (keys, _) = try index.search(vector: vector, count: min(100, passages.count))
        let semantic = keys.map(Int.init).filter { passages.indices.contains($0) }
        let lexical = lexicalMatches(for: question)

        // Static word embeddings are good for themes, but can miss rare names and
        // exact facts. Put the strongest text matches first, then fill with
        // semantic neighbors so both kinds of question have useful evidence.
        var candidates: [Int] = []
        var seen = Set<Int>()
        for number in lexical.prefix(min(6, count)) + semantic + lexical {
            if seen.insert(number).inserted { candidates.append(number) }
        }
        var chosen: [BookPassage] = []
        var chosenWords: [Set<String>] = []
        var chosenIndices = Set<Int>()
        for number in candidates {
            let candidate = passages[number]
            let words = Set(Self.searchTerms(candidate.text))
            let tooSimilar = chosenWords.contains { previous in
                let union = words.union(previous).count
                return union > 0 && Double(words.intersection(previous).count) / Double(union) > 0.55
            }
            if tooSimilar { continue }
            chosen.append(candidate)
            chosenWords.append(words)
            chosenIndices.insert(number)
            if chosen.count == count { break }
        }
        if chosen.count < count {
            for number in candidates where !chosenIndices.contains(number) {
                chosen.append(passages[number])
                if chosen.count == count { break }
            }
        }
        return chosen
    }

    private static func searchTerms(_ text: String) -> [String] {
        text.folding(options: [.diacriticInsensitive, .caseInsensitive],
                     locale: Locale(identifier: "en_US_POSIX"))
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
    }

    private func lexicalMatches(for question: String) -> [Int] {
        let query = Array(Set(Self.searchTerms(question).filter {
            !Self.searchStopWords.contains($0)
        }))
        guard !query.isEmpty else { return [] }
        let documentTerms = passages.map { Self.searchTerms($0.text) }
        let averageLength = max(1.0, Double(documentTerms.reduce(0) { $0 + $1.count }) / Double(passages.count))
        let documentFrequency = Dictionary(uniqueKeysWithValues: query.map { term in
            (term, documentTerms.reduce(0) { $0 + ($1.contains(term) ? 1 : 0) })
        })
        let ranked: [(Int, Double)] = documentTerms.enumerated().compactMap { number, words in
            let frequencies = words.reduce(into: [String: Int]()) { $0[$1, default: 0] += 1 }
            var score = 0.0
            var matched = 0
            for term in query {
                guard let frequency = frequencies[term] else { continue }
                matched += 1
                let frequencyInBook = Double(documentFrequency[term] ?? 0)
                let idf = log(1 + (Double(passages.count) - frequencyInBook + 0.5) / (frequencyInBook + 0.5))
                let normalizedLength = 1.2 * (0.25 + 0.75 * Double(words.count) / averageLength)
                score += idf * Double(frequency) * 2.2 / (Double(frequency) + normalizedLength)
            }
            guard matched > 0 else { return nil }
            // A passage containing every specific word in a factual question
            // should outrank one that only shares the general topic.
            if matched == query.count { score *= 1.6 }
            return (number, score)
        }
        return ranked.sorted { $0.1 == $1.1 ? $0.0 < $1.0 : $0.1 > $1.1 }.map(\.0)
    }
}

@MainActor
final class BookAskController: ObservableObject {
    static let shared = BookAskController()
    @Published private(set) var status = "Indexing starts shortly…"
    @Published private(set) var answer = ""
    @Published private(set) var sources: [AskSource] = []
    @Published private(set) var embeddingCount = 0
    @Published private(set) var errorMessage: String?
    @Published private(set) var isAnswering = false
    @Published private(set) var processingMessage: String?
    private let semanticIndex = BookSemanticIndex()
    private var indexingTask: Task<Void, Never>?
    private var sourceID: String?

    func bookLoaded(_ document: ReaderDocument) {
        guard sourceID != document.sourceID else { return }
        indexingTask?.cancel()
        sourceID = document.sourceID
        answer = ""
        sources = []
        embeddingCount = 0
        errorMessage = nil
        processingMessage = nil
        status = "Indexing starts shortly…"
        let id = document.sourceID
        let sentences = document.sentences
        indexingTask = Task(priority: .utility) {
            try? await Task.sleep(for: .seconds(1))
            guard !Task.isCancelled, sourceID == id else { return }
            status = "Indexing book…"
            do {
                let count = try await semanticIndex.prepare(sourceID: id, sentences: sentences)
                if sourceID == id {
                    embeddingCount = count
                    status = "Ready to ask"
                }
            } catch {
                if sourceID == id { status = error.localizedDescription }
            }
        }
    }

    func ask(_ question: String, document: ReaderDocument) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        bookLoaded(document)
        isAnswering = true
        processingMessage = embeddingCount == 0 ? "Preparing the book index…" : "Finding relevant passages…"
        answer = ""
        sources = []
        errorMessage = nil
        let id = document.sourceID
        Task {
            defer {
                isAnswering = false
                processingMessage = nil
            }
            await indexingTask?.value
            guard sourceID == id else { return }
            do {
                processingMessage = "Finding relevant passages…"
                let matches = try await semanticIndex.retrieve(trimmed)
                guard !matches.isEmpty else { throw AskError.emptyBook }
                sources = matches.enumerated().map { index, passage in
                    AskSource(id: index + 1, sentenceID: passage.firstSentence,
                              excerpt: passage.text)
                }
                guard SystemLanguageModel.default.isAvailable else { throw AskError.modelUnavailable }
                let context = matches.enumerated().map { i, passage in
                    "[\(i + 1)] \(promptExcerpt(passage.text))"
                }.joined(separator: "\n\n")
                processingMessage = "Writing an answer…"
                let session = LanguageModelSession(instructions: "Answer the user's question using only the supplied book passages, ordered by relevance. Check the first passages for an explicit answer before concluding it is absent. For a direct factual question, give the specific answer in the book's words. Cite the supporting passage as [1], [2], etc. next to the claim. Say the answer is unavailable only if no passage supports it. Be concise.")
                let response = try await session.respond(to: "Question: \(trimmed)\n\nBook passages:\n\(context)")
                let content = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !content.isEmpty else { throw AskError.emptyResponse }
                if sourceID == id { answer = content }
            } catch {
                if sourceID == id { errorMessage = error.localizedDescription }
            }
        }
    }

    /// Twenty passages must leave room in the 4,096-token model context for
    /// the question, instructions, and answer. Roughly budget 100 tokens per
    /// excerpt (three ASCII characters or one non-ASCII character per token).
    private func promptExcerpt(_ text: String) -> String {
        var result = ""
        var units = 0
        for character in text {
            let cost = character.unicodeScalars.allSatisfy { $0.value < 128 } ? 1 : 3
            if units + cost > 300 { break }
            result.append(character)
            units += cost
        }
        if result.count < text.count, let boundary = result.lastIndex(where: \.isWhitespace) {
            result = String(result[..<boundary])
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct BookAskPanel: View {
    let document: ReaderDocument
    let onJump: (Int) -> Void
    @ObservedObject private var ask = BookAskController.shared
    @ObservedObject private var reader = ReaderControllerHolder.reader
    @State private var question = ""
    @FocusState private var questionFocused: Bool

    private var theme: ReaderTheme { reader.theme }
    private var foreground: Color { Color(hex: theme.text) }
    private var muted: Color { Color(hex: theme.textMuted) }
    private var surface: Color { Color(hex: theme.inputBg) }
    private var border: Color { Color(hex: theme.inputBorder) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "sparkles")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Color(hex: theme.accent))
                    .frame(width: 34, height: 34)
                    .background(Color(hex: theme.accent).opacity(0.12),
                                in: RoundedRectangle(cornerRadius: 10))
                VStack(alignment: .leading, spacing: 3) {
                    Text("Ask this book")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(foreground)
                    Text(document.title)
                        .font(.system(size: 12))
                        .foregroundStyle(muted)
                        .lineLimit(1)
                }
                Spacer(minLength: 8)
                if ask.embeddingCount > 0 {
                    Text("\(ask.embeddingCount.formatted()) embeddings")
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(muted)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 6)
                        .background(surface, in: Capsule())
                } else if ask.status == "Indexing book…" {
                    ProgressView().controlSize(.small)
                }
            }
            .padding(.bottom, 14)

            HStack(spacing: 6) {
                Circle()
                    .fill(ask.embeddingCount > 0 ? Color.green : muted)
                    .frame(width: 6, height: 6)
                Text(ask.status)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(muted)
                    .lineLimit(1)
            }
            .padding(.bottom, 14)

            HStack(spacing: 8) {
                TextField("What would you like to know?", text: $question)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .foregroundStyle(foreground)
                    .focused($questionFocused)
                    .onSubmit(submit)
                    .padding(.horizontal, 12)
                    .frame(height: 38)
                    .background(surface, in: RoundedRectangle(cornerRadius: 9))
                    .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(border.opacity(0.7)))
                Button("Ask", action: submit)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)
                    .disabled(question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || ask.isAnswering)
            }
            .padding(.bottom, 15)

            if let error = ask.errorMessage {
                Label(error, systemImage: "exclamationmark.circle")
                    .font(.system(size: 12))
                    .foregroundStyle(.orange)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 9))
            }
            if ask.isAnswering || !ask.answer.isEmpty {
                VStack(alignment: .leading, spacing: 9) {
                    Label("Answer", systemImage: "text.quote")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(muted)
                    if ask.isAnswering {
                        HStack(spacing: 10) {
                            ProgressView().controlSize(.small)
                            Text(ask.processingMessage ?? "Working on your answer…")
                                .font(.system(size: 12))
                                .foregroundStyle(muted)
                        }
                        .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
                    } else {
                        ScrollView {
                            Text(ask.answer)
                                .font(.system(size: 13))
                                .lineSpacing(4)
                                .foregroundStyle(foreground)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        }
                        .frame(height: 150)
                    }
                }
                .padding(13)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(surface, in: RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(border.opacity(0.55)))
            }
            if !ask.sources.isEmpty {
                HStack {
                    Text("Sources")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(muted)
                    Spacer()
                    Text("\(ask.sources.count) passages")
                        .font(.system(size: 11))
                        .foregroundStyle(muted)
                }
                .padding(.top, 16)
                .padding(.bottom, 7)
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(ask.sources) { source in
                            Button { onJump(source.sentenceID) } label: {
                                HStack(alignment: .center, spacing: 10) {
                                    Text("\(source.id)")
                                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                        .foregroundStyle(Color(hex: theme.accent))
                                        .frame(width: 24, height: 24)
                                        .background(Color(hex: theme.accent).opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
                                    Text(source.excerpt)
                                        .font(.system(size: 11))
                                        .foregroundStyle(foreground)
                                        .lineLimit(2)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                    Image(systemName: "arrow.up.right")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(muted)
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .background(surface, in: RoundedRectangle(cornerRadius: 8))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .frame(height: min(CGFloat(ask.sources.count) * 58, 174))
            }
        }
        .padding(18)
        .frame(width: 470)
        .background(Color(hex: theme.menuBg))
        .onAppear { ask.bookLoaded(document) }
    }

    private func submit() { ask.ask(question, document: document) }
}
