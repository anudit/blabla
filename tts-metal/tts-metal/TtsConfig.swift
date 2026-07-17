//
//  TtsConfig.swift
//  tts-metal
//

import Foundation

enum TtsConfig {
    static let sampleRate: Double = 24000

    /// Output rate after the RE-USE cleanup/upsampling layer.
    static let enhancedSampleRate: Double = 48000

    static let symbols: [String] = {
        var s = ""
        s += "$;:,.!?¡¿—…“«»”„ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        s += "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃ"
        s += "ʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        return Array(s).map { String($0) }
    }()

    static let symbolToIndex: [String: Int] = {
        var map: [String: Int] = [:]
        for (i, sym) in symbols.enumerated() {
            map[sym] = i
        }
        return map
    }()

    static let voiceAliases: [String: String] = [
        "Bella": "expr-voice-2-f",
        "Jasper": "expr-voice-2-m",
        "Luna": "expr-voice-3-f",
        "Bruno": "expr-voice-3-m",
        "Rosie": "expr-voice-4-f",
        "Hugo": "expr-voice-4-m",
        "Kiki": "expr-voice-5-f",
        "Leo": "expr-voice-5-m"
    ]

    static let voiceKeys: [String] = Array(voiceAliases.keys.sorted())

    struct ModelURL {
        let name: String
        let onnx: String
        let voices: String
        let size: String
        let params: String
    }

    static let mini = ModelURL(
        name: "mini (80M, best quality)",
        onnx: "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/kitten_tts_mini_v0_8.onnx",
        voices: "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/voices.npz",
        size: "78 MB",
        params: "80M"
    )
}