import SwiftUI

private enum VisionGuidePalette {
    static let background = Color(red: 0.96, green: 0.94, blue: 0.90)
    static let surface = Color(red: 0.90, green: 0.87, blue: 0.81)
    static let charcoal = Color(red: 0.22, green: 0.23, blue: 0.24)
    static let text = Color(red: 0.18, green: 0.19, blue: 0.20)
}

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var camera = CameraManager()
    @StateObject private var speech = SpeechController()
    @State private var hasRequestedCamera = false
    @State private var hasScheduledInstructions = false
    @State private var instructionTask: Task<Void, Never>?
    @State private var isSettingsExpanded = false

    private var sensitivityBinding: Binding<Double> {
        Binding(
            get: { camera.sensitivity },
            set: { camera.sensitivity = $0 }
        )
    }

    var body: some View {
        VStack(spacing: 12) {
            header

            centerCameraAction
                .layoutPriority(1)

            recognitionStatus
            speechOutputStatus
            quickActions
            if let voiceSetupHint = speech.voiceSetupHint {
                Text(voiceSetupHint)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityHint("Instructions for downloading a higher-quality English voice")
            }
            settingsDisclosure
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .safeAreaInset(edge: .top, spacing: 0) {
            volumeHotkeys
                .padding(.horizontal, 10)
                .padding(.top, 4)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            voiceHotkeys
                .padding(.horizontal, 10)
                .padding(.bottom, 4)
        }
        .foregroundStyle(VisionGuidePalette.text)
        .background(VisionGuidePalette.background.ignoresSafeArea())
        .onAppear {
            camera.onEvent = { message in
                speech.speak(message)
            }
            scheduleInstructions()
        }
        .onChange(of: camera.isCameraRunning) { _, isRunning in
            if isRunning {
                if camera.modelAvailable {
                    speech.speak("The camera and object recognition are ready. Tap the center to hear a description.", interrupt: true)
                } else if camera.isModelLoading {
                    speech.speak("The camera is on. Object recognition is loading.", interrupt: true)
                } else {
                    speech.speak("The camera is on, but object recognition is unavailable.", interrupt: true)
                }
            }
        }
        .onChange(of: camera.modelAvailable) { _, isAvailable in
            if isAvailable && camera.isCameraRunning {
                speech.speak("Object recognition is ready.")
            }
        }
        .onChange(of: camera.statusMessage) { _, message in
            guard hasRequestedCamera else { return }
            let isCameraProblem = message.contains("not allowed")
                || message.contains("Allow camera access")
                || message.contains("unavailable")
                || message.contains("could not")
            if isCameraProblem {
                speech.speak(message, interrupt: true)
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                if hasRequestedCamera {
                    camera.start()
                }
                scheduleInstructions()
            } else {
                cancelPendingInstructions()
                camera.stop()
                if phase == .background {
                    speech.stop()
                }
            }
        }
        .onDisappear {
            cancelPendingInstructions()
            camera.stop()
            speech.stop()
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("VisionGuide")
                    .font(.title2.bold())
                Text(camera.isStartingCamera ? "Starting camera…" : camera.statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .accessibilityLabel("Camera status: \(camera.statusMessage)")
            }
            Spacer()
            Circle()
                .fill(camera.isCameraRunning ? Color.green : (camera.isStartingCamera ? Color.yellow : Color.orange))
                .frame(width: 12, height: 12)
                .accessibilityHidden(true)
        }
        .padding(.horizontal, 2)
    }

    private var centerCameraAction: some View {
        Button(action: performCenterAction) {
            ZStack {
                if camera.isCameraRunning {
                    CameraPreview(session: camera.session)
                    LinearGradient(
                        colors: [.clear, .black.opacity(0.68)],
                        startPoint: .center,
                        endPoint: .bottom
                    )

                    VStack {
                        Spacer()
                        Label("Tap the center to describe this view", systemImage: "waveform")
                            .font(.headline)
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 18)
                            .padding(.vertical, 12)
                    }
                } else {
                    VisionGuidePalette.charcoal

                    VStack(spacing: 16) {
                        Image(systemName: camera.isStartingCamera ? "camera.aperture" : "camera.fill")
                            .font(.system(size: 46, weight: .regular))
                        Text(camera.isStartingCamera ? "Starting the camera…" : "Tap the center to start the camera")
                            .font(.title3.bold())
                            .multilineTextAlignment(.center)
                        Text("The camera starts only when you choose to turn it on.")
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.78))
                            .multilineTextAlignment(.center)
                    }
                    .foregroundStyle(.white)
                    .padding(28)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(minHeight: 270, maxHeight: .infinity)
            .clipShape(RoundedRectangle(cornerRadius: 24))
            .overlay {
                RoundedRectangle(cornerRadius: 24)
                    .strokeBorder(VisionGuidePalette.charcoal.opacity(0.16), lineWidth: 1)
            }
            .contentShape(RoundedRectangle(cornerRadius: 24))
        }
        .buttonStyle(.plain)
        .disabled(camera.isStartingCamera)
        .accessibilityLabel(camera.isCameraRunning ? "Describe current camera view" : "Start camera")
        .accessibilityHint(camera.isCameraRunning
            ? "Speaks a description of the important objects currently visible"
            : "Turns on the iPhone back camera")
    }

    private var recognitionStatus: some View {
        Group {
            if camera.isModelLoading || (camera.isCameraRunning && !camera.modelAvailable) {
                Text(camera.modelMessage)
                    .foregroundStyle(.orange)
                    .accessibilityLabel("Recognition model status: \(camera.modelMessage)")
            } else if camera.isCameraRunning {
                Text("Objects currently detected: \(camera.detectedObjectCount)")
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("\(camera.detectedObjectCount) objects currently detected")
            } else {
                Text("Tap the center when you are ready.")
                    .foregroundStyle(.secondary)
            }
        }
        .font(.footnote)
        .multilineTextAlignment(.center)
        .lineLimit(2)
    }

    private var quickActions: some View {
        HStack(spacing: 10) {
            Button {
                speech.speakInstructions()
            } label: {
                Label("Repeat guide", systemImage: "speaker.wave.2")
                    .font(.headline.weight(.semibold))
                    .frame(maxWidth: .infinity, minHeight: 62)
                    .foregroundStyle(.white)
                    .background(VisionGuidePalette.charcoal, in: RoundedRectangle(cornerRadius: 18))
            }
            .buttonStyle(.plain)
            .accessibilityHint("Repeats the spoken instructions for the corner controls and camera")

            Button {
                speech.toggle()
            } label: {
                Label(speech.isEnabled ? "Mute voice" : "Enable voice",
                      systemImage: speech.isEnabled ? "speaker.slash" : "speaker.wave.2")
                    .font(.headline.weight(.semibold))
                    .frame(maxWidth: .infinity, minHeight: 62)
                    .foregroundStyle(.white)
                    .background(VisionGuidePalette.charcoal, in: RoundedRectangle(cornerRadius: 18))
            }
            .buttonStyle(.plain)
            .accessibilityHint("Turns spoken descriptions on or off")
        }
    }

    private var speechOutputStatus: some View {
        let hasAudioError = speech.audioStatus.hasPrefix("Audio could not")
            || speech.audioStatus.hasPrefix("No English voice")

        return Label(
            speech.audioStatus,
            systemImage: hasAudioError ? "exclamationmark.triangle.fill" : "waveform"
        )
        .font(.footnote)
        .foregroundStyle(hasAudioError ? .orange : .secondary)
        .lineLimit(2)
        .multilineTextAlignment(.center)
        .accessibilityLabel("Speech output status: \(speech.audioStatus)")
    }

    private var settingsDisclosure: some View {
        DisclosureGroup("More settings", isExpanded: $isSettingsExpanded) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Sensitivity")
                    Spacer()
                    Text(String(format: "%.2f", camera.sensitivity))
                        .monospacedDigit()
                }
                Slider(value: sensitivityBinding, in: 0.20...0.50, step: 0.05)
                    .tint(VisionGuidePalette.charcoal)
                    .accessibilityLabel("Detection sensitivity")
                    .accessibilityHint("Higher sensitivity filters more uncertain detections")

                Toggle("Rare uncertain-object alerts", isOn: $camera.uncertaintyAlertsEnabled)
                    .tint(VisionGuidePalette.charcoal)
                    .accessibilityHint("Only speaks after a possible object remains in view")

                Text("Distance is a rough estimate based on typical object sizes. It may be inaccurate; use it as an additional cue.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if camera.isCameraRunning {
                    Button {
                        camera.togglePause()
                    } label: {
                        Label(camera.isPaused ? "Resume analysis" : "Pause analysis",
                              systemImage: camera.isPaused ? "play.fill" : "pause.fill")
                    }
                    .accessibilityHint("Stops or resumes live camera analysis; preview remains visible")
                }
            }
            .padding(.top, 8)
        }
        .font(.subheadline.weight(.semibold))
        .padding(12)
        .background(VisionGuidePalette.surface, in: RoundedRectangle(cornerRadius: 16))
    }

    private var volumeHotkeys: some View {
        HStack(spacing: 8) {
            Button {
                speech.adjustVolume(by: 0.1)
            } label: {
                hotkeyLabel("Volume +", symbol: "speaker.wave.3.fill")
            }
            .accessibilityLabel("Increase speech volume")
            .accessibilityHint("Raises spoken audio volume")

            Spacer(minLength: 2)

            Text("VOLUME \(Int((speech.volume * 100).rounded()))%")
                .font(.footnote.bold())
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .accessibilityLabel("Speech volume \(Int((speech.volume * 100).rounded())) percent")

            Spacer(minLength: 2)

            Button {
                speech.adjustVolume(by: -0.1)
            } label: {
                hotkeyLabel("Volume −", symbol: "speaker.wave.1.fill")
            }
            .accessibilityLabel("Decrease speech volume")
            .accessibilityHint("Lowers spoken audio volume")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(VisionGuidePalette.surface, in: RoundedRectangle(cornerRadius: 20))
    }

    private var voiceHotkeys: some View {
        HStack(spacing: 8) {
            Button {
                speech.selectFemaleVoice()
            } label: {
                hotkeyLabel("Female", symbol: "person.crop.circle")
            }
            .accessibilityLabel("Choose female voice")
            .accessibilityHint("Plays a sample of the best installed female English voice")

            Spacer(minLength: 2)

            VStack(spacing: 2) {
                Text(speech.preferredSpeaker.displayName.uppercased())
                    .font(.caption.bold())
                Text(speech.voiceQuality)
                    .font(.caption2)
            }
            .frame(minWidth: 74)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Preferred speaker: \(speech.preferredSpeaker.displayName), \(speech.voiceName)")

            Spacer(minLength: 2)

            Button {
                speech.selectMaleVoice()
            } label: {
                hotkeyLabel("Male", symbol: "person.crop.circle")
            }
            .accessibilityLabel("Choose male voice")
            .accessibilityHint("Plays a sample of the best installed male English voice")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(VisionGuidePalette.surface, in: RoundedRectangle(cornerRadius: 20))
    }

    private func hotkeyLabel(_ title: String, symbol: String) -> some View {
        Label(title, systemImage: symbol)
            .font(.subheadline.bold())
            .lineLimit(1)
            .minimumScaleFactor(0.8)
            .frame(minWidth: 88, minHeight: 58)
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .background(VisionGuidePalette.charcoal, in: RoundedRectangle(cornerRadius: 16))
    }

    private func performCenterAction() {
        if camera.isCameraRunning {
            speech.speak(camera.describeCurrentView(), interrupt: true)
        } else if !camera.isStartingCamera {
            hasRequestedCamera = true
            camera.start()
        }
    }

    private func scheduleInstructions() {
        guard !hasScheduledInstructions, scenePhase == .active else { return }
        hasScheduledInstructions = true
        instructionTask = Task { @MainActor in
            do {
                try await Task.sleep(nanoseconds: 700_000_000)
            } catch {
                return
            }
            guard !Task.isCancelled, scenePhase == .active else {
                hasScheduledInstructions = false
                instructionTask = nil
                return
            }
            speech.speakInstructions()
            instructionTask = nil
        }
    }

    private func cancelPendingInstructions() {
        guard instructionTask != nil else { return }
        instructionTask?.cancel()
        instructionTask = nil
        hasScheduledInstructions = false
    }
}
