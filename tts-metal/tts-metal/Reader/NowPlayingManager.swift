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
    var onPlay: (() -> Void)?
    var onPause: (() -> Void)?
    var onNextSentence: (() -> Void)?
    var onPrevSentence: (() -> Void)?

    private var registered = false

    private init() {}

    func register() {
        guard !registered else { return }
        registered = true

        let center = MPRemoteCommandCenter.shared()
        center.playCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.onPlay?() }
            return .success
        }
        center.pauseCommand.addTarget { [weak self] _ in
            Task { @MainActor in self?.onPause?() }
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
        setCommandsEnabled(false)
    }

    /// The media keys belong to BlaBla only while a book is playing or
    /// paused; otherwise they're left to Music and the rest.
    private func setCommandsEnabled(_ enabled: Bool) {
        let center = MPRemoteCommandCenter.shared()
        for command in [center.playCommand, center.pauseCommand, center.togglePlayPauseCommand,
                        center.nextTrackCommand, center.previousTrackCommand] {
            command.isEnabled = enabled
        }
    }

    /// Update metadata while reading.
    func update(title: String, artist: String, index: Int, total: Int) {
        var info: [String: Any] = [
            MPMediaItemPropertyTitle: title,
            MPMediaItemPropertyArtist: artist,
            MPNowPlayingInfoPropertyMediaType: MPNowPlayingInfoMediaType.audio.rawValue,
            // macOS routes the media keys to the app it believes is playing,
            // which it judges by the rate.
            MPNowPlayingInfoPropertyPlaybackRate: paused ? 0.0 : 1.0,
        ]
        if total > 0 {
            info[MPMediaItemPropertyAlbumTrackCount] = total
            info[MPMediaItemPropertyAlbumTrackNumber] = index + 1
        }
        MPNowPlayingInfoCenter.default().nowPlayingInfo = info
        setCommandsEnabled(true)
    }

    private var paused = false

    func setPaused(_ paused: Bool) {
        self.paused = paused
        let center = MPNowPlayingInfoCenter.default()
        center.playbackState = paused ? .paused : .playing
        center.nowPlayingInfo?[MPNowPlayingInfoPropertyPlaybackRate] = paused ? 0.0 : 1.0
        setCommandsEnabled(true)
    }

    func clear() {
        paused = false
        MPNowPlayingInfoCenter.default().nowPlayingInfo = nil
        MPNowPlayingInfoCenter.default().playbackState = .stopped
        setCommandsEnabled(false)
    }
}
