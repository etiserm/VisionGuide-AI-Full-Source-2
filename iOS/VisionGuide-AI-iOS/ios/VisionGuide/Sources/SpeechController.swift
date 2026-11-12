import AVFoundation
import Combine

enum PreferredSpeaker: String {
    case female
    case male

    var displayName: String {
        switch self {
        case .female: return "Female"
        case .male: return "Male"
        }
    }

    var systemGender: AVSpeechSynthesisVoiceGender {
        switch self {
        case .female: return .female
        case .male: return .male
        }
    }
}

@MainActor
final class SpeechController: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var isEnabled = true
    @Published private(set) var volume: Float
    @Published private(set) var preferredSpeaker: PreferredSpeaker
    @Published private(set) var audioStatus = "Voice is ready."

    private let synthesizer = AVSpeechSynthesizer()
    private let volumeKey = "speechVolume"
    private let speakerKey = "preferredSpeakerGender"

    override init() {
        let savedVolume = UserDefaults.standard.object(forKey: volumeKey) as? Double
        volume = min(1.0, max(0.1, Float(savedVolume ?? 0.8)))
        let savedSpeaker = UserDefaults.standard.string(forKey: speakerKey)
        preferredSpeaker = PreferredSpeaker(rawValue: savedSpeaker ?? "") ?? .female
        super.init()
        synthesizer.delegate = self
        synthesizer.usesApplicationAudioSession = true
    }

    private var availableEnglishVoices: [AVSpeechSynthesisVoice] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix("en") }
    }

    private func bestVoice(for speaker: PreferredSpeaker) -> AVSpeechSynthesisVoice? {
        let matchingVoices = availableEnglishVoices
            .filter { $0.gender == speaker.systemGender }
            .sorted {
                let firstRank = qualityRank($0.quality)
                let secondRank = qualityRank($1.quality)
                if firstRank != secondRank { return firstRank < secondRank }

                let firstIsUS = $0.language == "en-US"
                let secondIsUS = $1.language == "en-US"
                if firstIsUS != secondIsUS { return firstIsUS }

                let preferredName = speaker == .female ? "Ava" : "Evan"
                let firstPreferred = $0.name.localizedCaseInsensitiveContains(preferredName)
                let secondPreferred = $1.name.localizedCaseInsensitiveContains(preferredName)
                if firstPreferred != secondPreferred { return firstPreferred }

                return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
            }

        if let exactUSVoice = matchingVoices.first(where: { $0.language == "en-US" }) {
            return exactUSVoice
        }
        if let otherEnglishVoice = matchingVoices.first {
            return otherEnglishVoice
        }
        return AVSpeechSynthesisVoice(language: "en-US")
            ?? AVSpeechSynthesisVoice(language: "en-GB")
    }

    private var selectedVoice: AVSpeechSynthesisVoice? {
        bestVoice(for: preferredSpeaker)
    }

    var voiceName: String {
        guard let voice = selectedVoice else { return "English voice" }
        return voice.name
    }

    var voiceQuality: String {
        guard let voice = selectedVoice else { return "English voice unavailable" }
        return "\(qualityName(voice.quality)) quality"
    }

    var voiceSetupHint: String? {
        guard let voice = selectedVoice, voice.quality != .premium else { return nil }
        return "For the most natural sound, download a Premium English voice in iPhone Settings → Accessibility → Read & Speak → Voices. Current voice quality: \(qualityName(voice.quality))."
    }

    @discardableResult
    func speak(_ text: String, interrupt: Bool = false) -> Bool {
        guard isEnabled, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            if !isEnabled { audioStatus = "Voice is muted." }
            return false
        }

        if interrupt {
            synthesizer.stopSpeaking(at: .immediate)
        } else if synthesizer.isSpeaking {
            return false
        }

        guard activateAudioSession() else { return false }

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = selectedVoice ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.volume = volume
        utterance.rate = 0.49
        utterance.pitchMultiplier = 1.0
        utterance.preUtteranceDelay = 0.04
        utterance.postUtteranceDelay = 0.08
        audioStatus = "Starting voice…"
        synthesizer.speak(utterance)
        return true
    }

    func speakInstructions() {
        speak(
            "Welcome to VisionGuide. Volume up is in the upper left corner. "
                + "Volume down is in the upper right. Female voice is in the lower left. "
                + "Male voice is in the lower right. Tap the center to start the camera. "
                + "When the camera is on, tap the center again to hear a description.",
            interrupt: true
        )
    }

    func adjustVolume(by change: Float) {
        volume = min(1.0, max(0.1, volume + change))
        UserDefaults.standard.set(Double(volume), forKey: volumeKey)
        speak("Volume \(Int((volume * 100).rounded())) percent.", interrupt: true)
    }

    func selectFemaleVoice() {
        selectVoice(.female)
    }

    func selectMaleVoice() {
        selectVoice(.male)
    }

    func toggle() {
        isEnabled.toggle()
        if isEnabled {
            speak("Voice is on.", interrupt: true)
        } else {
            synthesizer.stopSpeaking(at: .immediate)
            deactivateAudioSessionWhenIdle()
            audioStatus = "Voice is muted."
        }
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        deactivateAudioSessionWhenIdle()
        audioStatus = isEnabled ? "Voice is ready." : "Voice is muted."
    }

    private func selectVoice(_ speaker: PreferredSpeaker) {
        preferredSpeaker = speaker
        UserDefaults.standard.set(speaker.rawValue, forKey: speakerKey)

        guard let voice = selectedVoice else {
            audioStatus = "No English voice is available on this iPhone."
            return
        }

        speak(
            "Hello. I'm \(voice.name). I'll describe what I can see, clearly and calmly.",
            interrupt: true
        )
    }

    private func qualityRank(_ quality: AVSpeechSynthesisVoiceQuality) -> Int {
        if quality == .premium { return 0 }
        if quality == .enhanced { return 1 }
        return 2
    }

    private func qualityName(_ quality: AVSpeechSynthesisVoiceQuality) -> String {
        if quality == .premium { return "Premium" }
        if quality == .enhanced { return "Enhanced" }
        return "Standard"
    }

    private func activateAudioSession() -> Bool {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(
                .playback,
                mode: .spokenAudio,
                options: [.duckOthers, .allowBluetoothA2DP]
            )
        } catch {
            do {
                try audioSession.setCategory(.playback, mode: .default)
            } catch {
                audioStatus = "Audio could not start: \(error.localizedDescription)"
                return false
            }
        }

        do {
            try audioSession.setActive(true)
            return true
        } catch {
            audioStatus = "Audio could not start: \(error.localizedDescription)"
            return false
        }
    }

    private func deactivateAudioSessionWhenIdle() {
        guard !synthesizer.isSpeaking else { return }
        do {
            try AVAudioSession.sharedInstance().setActive(
                false,
                options: .notifyOthersOnDeactivation
            )
        } catch {
            // Deactivation can fail during an iOS audio interruption; the next
            // utterance activates the session again before speaking.
        }
        audioStatus = isEnabled ? "Voice is ready." : "Voice is muted."
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didStart utterance: AVSpeechUtterance
    ) {
        let voiceName = utterance.voice?.name ?? "English voice"
        Task { @MainActor [weak self] in
            self?.audioStatus = "Speaking with \(voiceName)."
        }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        Task { @MainActor [weak self] in
            self?.deactivateAudioSessionWhenIdle()
        }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        Task { @MainActor [weak self] in
            self?.deactivateAudioSessionWhenIdle()
        }
    }
}
