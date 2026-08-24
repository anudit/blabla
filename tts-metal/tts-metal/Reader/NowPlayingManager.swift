//
//  NowPlayingManager.swift
//  tts-metal
//
//  Native macOS replacement for blabla's MediaSession hack: system media keys
//  (play/pause/next/prev), Control Center and the Now Playing pane are driven via
//  MPNowPlayingInfoCenter / MPRemoteCommandCenter.
//

import Foundation
import MediaPlayer

@MainActor
final class NowPlayingManager {
    static let shared = NowPlayingManager()

    var onTogglePlayback: (() -> Void)?
    var onNextSentence: (() -> Void)?
    var onPrevSentence: (() -> Void)?

    private var registered = false

    private init() {}

    func register() {
        guard !registered else { return }
        registered = true

        let center = MPRemoteCommandCenter.shared()
        center.playCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.onTogglePlayback?() }
            return .success
        }
        center.pauseCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.onTogglePlayback?() }
            return .success
        }
        center.togglePlayPauseCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.onTogglePlayback?() }
            return .success
        }
        center.nextTrackCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.onNextSentence?() }
            return .success
        }
        center.previousTrackCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.onPrevSentence?() }
            return .success
        }
    }

    /// Update metadata while reading.
    func update(title: String, artist: String, index: Int, total: Int) {
        var info: [String: Any] = [
            MPMediaItemPropertyTitle: title,
            MPMediaItemPropertyArtist: artist,
            MPNowPlayingInfoPropertyMediaType: MPNowPlayingInfoMediaType.audio.rawValue,
        ]
        if total > 0 {
            info[MPMediaItemPropertyAlbumTrackCount] = total
            info[MPMediaItemPropertyAlbumTrackNumber] = index + 1
        }
        MPNowPlayingInfoCenter.default().nowPlayingInfo = info
    }

    func setPaused(_ paused: Bool) {
        MPNowPlayingInfoCenter.default().playbackState = paused ? .paused : .playing
    }

    func clear() {
        MPNowPlayingInfoCenter.default().nowPlayingInfo = nil
        MPNowPlayingInfoCenter.default().playbackState = .stopped
    }
}
