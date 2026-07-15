//
//  ModelLoader.swift
//  tts-metal
//
//  The Kitten TTS mini model and voices archive are bundled as app resources.
//  This helper resolves their URLs from the main bundle.
//

import Foundation

enum ModelLoader {
    struct EmbeddedModel {
        let onnxURL: URL
        let voicesURL: URL
    }

    static func loadEmbedded() throws -> EmbeddedModel {
        guard let onnxURL = Bundle.main.url(forResource: "kitten_tts_mini_v0_8", withExtension: "onnx") else {
            throw NSError(domain: "ModelLoader", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "kitten_tts_mini_v0_8.onnx not found in bundle"])
        }
        guard let voicesURL = Bundle.main.url(forResource: "voices", withExtension: "npz") else {
            throw NSError(domain: "ModelLoader", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "voices.npz not found in bundle"])
        }
        return EmbeddedModel(onnxURL: onnxURL, voicesURL: voicesURL)
    }
}