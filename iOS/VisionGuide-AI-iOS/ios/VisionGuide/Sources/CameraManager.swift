import AVFoundation
import Combine
import CoreVideo

final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published private(set) var statusMessage = "Camera is not running."
    @Published private(set) var isCameraRunning = false
    @Published private(set) var isStartingCamera = false
    @Published private(set) var isPaused = false
    @Published private(set) var modelAvailable = false
    @Published private(set) var isModelLoading = false
    @Published private(set) var modelMessage = "Recognition model loads when the camera starts."
    // The latest boxes are internal to the camera flow; publishing every frame would
    // make SwiftUI rebuild the screen for each inference result.
    private(set) var latestDetections: [ObjectDetection] = []
    @Published private(set) var detectedObjectCount = 0

    @Published var sensitivity: Double {
        didSet { UserDefaults.standard.set(sensitivity, forKey: "confidenceThreshold") }
    }

    @Published var uncertaintyAlertsEnabled: Bool {
        didSet { UserDefaults.standard.set(uncertaintyAlertsEnabled, forKey: "uncertaintyAlertsEnabled") }
    }

    let session = AVCaptureSession()
    var onEvent: ((String) -> Bool)?

    private let sessionQueue = DispatchQueue(label: "visionguide.camera.session")
    private let inferenceQueue = DispatchQueue(label: "visionguide.camera.inference")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let stateLock = NSLock()
    private var frameProcessor: FrameProcessor?
    private var isConfigured = false
    private var pausedStorage = false
    private var analysisEnabledStorage = false
    private var modelGeneration = 0
    private var lastInferenceTime: TimeInterval = 0
    private let inferenceInterval: TimeInterval = 0.25
    private var horizontalFieldOfViewDegrees: CGFloat?
    // Accessed only on inferenceQueue.
    private var isLoadingModel = false

    private struct TrackState {
        var consecutiveFrames = 0
        var lastSeenFrame = 0
        var announced = false
    }

    private var frameNumber = 0
    private var objectTracks: [String: TrackState] = [:]
    private var uncertainTracks: [String: TrackState] = [:]
    private var lastAutomaticSpeechTime: TimeInterval = 0
    private var lastUncertaintySpeechTime: TimeInterval = 0

    private let urgentLabels: Set<String> = [
        "car", "truck", "bus", "motorcycle", "bicycle", "scooter",
        "stairs", "staircase", "pothole", "curb", "kerb",
        "construction barrier", "barrier", "traffic cone", "bollard",
        "pole", "manhole", "person", "pedestrian", "wheelchair", "dog",
    ]

    private var wantsCameraRunning: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return analysisEnabledStorage
    }

    private var currentModelGeneration: Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        return modelGeneration
    }

    override init() {
        sensitivity = UserDefaults.standard.object(forKey: "confidenceThreshold") as? Double ?? 0.25
        uncertaintyAlertsEnabled = UserDefaults.standard.object(forKey: "uncertaintyAlertsEnabled") as? Bool ?? true
        super.init()
    }

    func start() {
        guard !isCameraRunning, !isStartingCamera else { return }
        stateLock.lock()
        analysisEnabledStorage = true
        let generation = modelGeneration
        stateLock.unlock()
        isStartingCamera = true
        statusMessage = "Starting camera…"

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStartSession(generation: generation)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                DispatchQueue.main.async {
                    guard self.wantsCameraRunning,
                          self.currentModelGeneration == generation
                    else {
                        self.isStartingCamera = false
                        return
                    }
                    if granted {
                        self.configureAndStartSession(generation: generation)
                    } else {
                        self.isStartingCamera = false
                        self.statusMessage = "Camera access was not allowed."
                    }
                }
            }
        case .denied, .restricted:
            isStartingCamera = false
            statusMessage = "Allow camera access in iPhone Settings to start."
        @unknown default:
            isStartingCamera = false
            statusMessage = "Camera permission status is unavailable."
        }
    }

    func stop() {
        isStartingCamera = false
        isCameraRunning = false
        isPaused = false
        stateLock.lock()
        pausedStorage = false
        analysisEnabledStorage = false
        modelGeneration += 1
        stateLock.unlock()
        statusMessage = "Camera is not running."
        latestDetections = []
        detectedObjectCount = 0
        objectTracks.removeAll(keepingCapacity: false)
        uncertainTracks.removeAll(keepingCapacity: false)
        modelAvailable = false
        isModelLoading = false
        modelMessage = "Recognition model loads when the camera starts."

        inferenceQueue.async { [weak self] in
            guard let self else { return }
            self.frameProcessor = nil
            self.isLoadingModel = false
        }

        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }

    func togglePause() {
        stateLock.lock()
        pausedStorage.toggle()
        let paused = pausedStorage
        stateLock.unlock()

        DispatchQueue.main.async {
            self.isPaused = paused
            self.statusMessage = paused ? "Live analysis is paused." : "Camera is running."
        }
    }

    func describeCurrentView() -> String {
        if isPaused {
            return "Camera analysis is paused. Resume it to describe the current view."
        }
        guard modelAvailable else {
            return isModelLoading
                ? "Object recognition is loading. Please try again in a moment."
                : "Object recognition is currently unavailable."
        }
        return SceneDescription.make(from: latestDetections)
    }

    private func configureAndStartSession(generation: Int) {
        loadModelIfNeeded(generation: generation)
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard self.wantsCameraRunning,
                  self.currentModelGeneration == generation
            else { return }
            guard !self.isConfigured else {
                if self.wantsCameraRunning,
                   self.currentModelGeneration == generation,
                   !self.session.isRunning {
                    self.session.startRunning()
                }
                if (!self.wantsCameraRunning || self.currentModelGeneration != generation),
                   self.session.isRunning {
                    self.session.stopRunning()
                }
                DispatchQueue.main.async {
                    guard self.wantsCameraRunning,
                          self.currentModelGeneration == generation
                    else { return }
                    self.isStartingCamera = false
                    self.isCameraRunning = true
                    self.statusMessage = self.cameraRunningStatus
                }
                return
            }

            self.session.beginConfiguration()
            self.session.sessionPreset = .vga640x480
            var cameraInput: AVCaptureDeviceInput?

            do {
                guard let camera = AVCaptureDevice.default(
                    .builtInWideAngleCamera,
                    for: .video,
                    position: .back
                ) else {
                    throw CameraError.backCameraUnavailable
                }

                let input = try AVCaptureDeviceInput(device: camera)
                cameraInput = input
                guard self.session.canAddInput(input) else {
                    throw CameraError.cannotAddCameraInput
                }
                self.session.addInput(input)

                self.videoOutput.alwaysDiscardsLateVideoFrames = true
                self.videoOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String:
                        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                ]
                self.videoOutput.setSampleBufferDelegate(
                    self,
                    queue: self.inferenceQueue
                )

                guard self.session.canAddOutput(self.videoOutput) else {
                    throw CameraError.cannotAddVideoOutput
                }
                self.session.addOutput(self.videoOutput)

                self.session.commitConfiguration()
                self.isConfigured = true
                // The session may select a cropped VGA format when configuration is committed.
                let fieldOfView = CGFloat(camera.activeFormat.videoFieldOfView)
                self.stateLock.lock()
                self.horizontalFieldOfViewDegrees = fieldOfView > 0 ? fieldOfView : nil
                self.stateLock.unlock()
                if self.wantsCameraRunning,
                   self.currentModelGeneration == generation {
                    self.session.startRunning()
                }
                if (!self.wantsCameraRunning || self.currentModelGeneration != generation),
                   self.session.isRunning {
                    self.session.stopRunning()
                }

                DispatchQueue.main.async {
                    guard self.wantsCameraRunning,
                          self.currentModelGeneration == generation
                    else { return }
                    self.isStartingCamera = false
                    self.isCameraRunning = true
                    self.statusMessage = self.cameraRunningStatus
                }
            } catch {
                if let cameraInput,
                   self.session.inputs.contains(where: { $0 === cameraInput }) {
                    self.session.removeInput(cameraInput)
                }
                if self.session.outputs.contains(where: { $0 === self.videoOutput }) {
                    self.session.removeOutput(self.videoOutput)
                }
                self.videoOutput.setSampleBufferDelegate(nil, queue: nil)
                self.isConfigured = false
                self.session.commitConfiguration()
                self.stateLock.lock()
                let requestIsCurrent = self.modelGeneration == generation
                if requestIsCurrent {
                    self.analysisEnabledStorage = false
                    self.modelGeneration += 1
                }
                let failedGeneration = requestIsCurrent ? self.modelGeneration : nil
                self.stateLock.unlock()

                if requestIsCurrent {
                    self.inferenceQueue.async { [weak self] in
                        self?.frameProcessor = nil
                        self?.isLoadingModel = false
                    }
                }
                DispatchQueue.main.async {
                    guard let failedGeneration,
                          self.currentModelGeneration == failedGeneration
                    else { return }
                    self.isStartingCamera = false
                    self.isCameraRunning = false
                    self.statusMessage = error.localizedDescription
                    self.modelAvailable = false
                    self.isModelLoading = false
                    self.modelMessage = "Recognition model loads when the camera starts."
                }
            }
        }
    }

    private var cameraRunningStatus: String {
        if modelAvailable { return "Camera is running." }
        return isModelLoading
            ? "Camera is running; object recognition is loading."
            : "Camera is running; object recognition is unavailable."
    }

    private func loadModelIfNeeded(generation: Int) {
        guard !modelAvailable, !isModelLoading else { return }
        isModelLoading = true
        modelMessage = "Loading object recognition…"

        inferenceQueue.async { [weak self] in
            guard let self,
                  self.frameProcessor == nil,
                  !self.isLoadingModel
            else {
                return
            }

            self.isLoadingModel = true
            do {
                self.frameProcessor = try FrameProcessor()
                self.isLoadingModel = false
                DispatchQueue.main.async {
                    guard self.wantsCameraRunning,
                          self.currentModelGeneration == generation
                    else { return }
                    self.modelAvailable = true
                    self.isModelLoading = false
                    self.modelMessage = "Object model loaded."
                    if self.isCameraRunning {
                        self.statusMessage = "Camera is running."
                    }
                }
            } catch {
                self.isLoadingModel = false
                DispatchQueue.main.async {
                    guard self.wantsCameraRunning,
                          self.currentModelGeneration == generation
                    else { return }
                    self.modelAvailable = false
                    self.isModelLoading = false
                    self.modelMessage = error.localizedDescription
                    if self.isCameraRunning {
                        self.statusMessage = "Camera is running; object recognition is unavailable."
                    }
                }
            }
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        stateLock.lock()
        let isPausedNow = pausedStorage
        let analysisEnabledNow = analysisEnabledStorage
        let fieldOfView = horizontalFieldOfViewDegrees
        stateLock.unlock()

        guard analysisEnabledNow,
              !isPausedNow,
              let processor = frameProcessor,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else {
            return
        }

        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastInferenceTime >= inferenceInterval else { return }
        lastInferenceTime = now

        do {
            let detections = try processor.process(
                pixelBuffer,
                horizontalFieldOfViewDegrees: fieldOfView
            )
            DispatchQueue.main.async { [weak self] in
                guard let self, self.wantsCameraRunning else { return }
                self.handle(detections)
            }
        } catch {
            DispatchQueue.main.async { [weak self] in
                self?.statusMessage = "Frame analysis failed: \(error.localizedDescription)"
            }
        }
    }

    private func handle(_ detections: [ObjectDetection]) {
        frameNumber += 1
        let accepted = detections
            .filter { Double($0.confidence) >= sensitivity }
            .sorted { $0.priorityScore < $1.priorityScore }
        latestDetections = accepted
        if detectedObjectCount != accepted.count {
            detectedObjectCount = accepted.count
        }

        let now = ProcessInfo.processInfo.systemUptime
        var newEventDetections: [ObjectDetection] = []
        var seenObjectKeys = Set<String>()

        for detection in accepted {
            let key = trackKey(for: detection)
            guard seenObjectKeys.insert(key).inserted else { continue }
            var state = objectTracks[key] ?? TrackState()
            state.consecutiveFrames = state.lastSeenFrame == frameNumber - 1
                ? state.consecutiveFrames + 1
                : 1
            state.lastSeenFrame = frameNumber

            // Keep general household detections available for on-demand descriptions,
            // but don't speak every newly noticed item while the camera is running.
            if state.consecutiveFrames >= 3,
               !state.announced,
               urgentLabels.contains(detection.label) {
                newEventDetections.append(detection)
            }
            objectTracks[key] = state
        }

        removeStaleTracks(from: &objectTracks)

        if now - lastAutomaticSpeechTime >= 2.5,
           let mostImportant = newEventDetections.min(by: {
               $0.priorityScore < $1.priorityScore
           }) {
            let message = automaticMessage(for: mostImportant)
            if onEvent?(message) == true {
                lastAutomaticSpeechTime = now
                let key = trackKey(for: mostImportant)
                if var state = objectTracks[key] {
                    state.announced = true
                    objectTracks[key] = state
                }
            }
        }

        updateUncertaintyAlerts(
            detections: detections,
            now: now
        )
    }

    private func updateUncertaintyAlerts(
        detections: [ObjectDetection],
        now: TimeInterval
    ) {
        guard uncertaintyAlertsEnabled else {
            uncertainTracks.removeAll()
            return
        }

        removeStaleTracks(from: &uncertainTracks)

        let confirmedHazardInPath = latestDetections.contains {
            urgentLabels.contains($0.label) && isInForwardZone($0)
        }
        guard !confirmedHazardInPath else { return }

        var stableCandidates: [ObjectDetection] = []
        var seenUncertainKeys = Set<String>()

        for detection in detections where
            urgentLabels.contains(detection.label)
                && Double(detection.confidence) >= 0.18
                && Double(detection.confidence) < min(sensitivity, 0.30)
                && detection.areaRatio >= 0.004
                && isInForwardZone(detection) {
            let key = trackKey(for: detection)
            guard seenUncertainKeys.insert(key).inserted else { continue }
            var state = uncertainTracks[key] ?? TrackState()
            state.consecutiveFrames = state.lastSeenFrame == frameNumber - 1
                ? state.consecutiveFrames + 1
                : 1
            state.lastSeenFrame = frameNumber

            if state.consecutiveFrames >= 18 && !state.announced {
                stableCandidates.append(detection)
            }
            uncertainTracks[key] = state
        }

        removeStaleTracks(from: &uncertainTracks)

        guard now - lastUncertaintySpeechTime >= 25,
              let candidate = stableCandidates.min(by: {
                  $0.priorityScore < $1.priorityScore
              })
        else {
            return
        }

        let candidateKey = trackKey(for: candidate)
        let message = "I may have noticed something in your path, but I'm not certain."
        if onEvent?(message) == true {
            if var state = uncertainTracks[candidateKey] {
                state.announced = true
                uncertainTracks[candidateKey] = state
            }
            lastUncertaintySpeechTime = now
        }
    }

    private func removeStaleTracks(from tracks: inout [String: TrackState]) {
        tracks = tracks.filter { frameNumber - $0.value.lastSeenFrame <= 10 }
    }

    private func trackKey(for detection: ObjectDetection) -> String {
        "\(detection.label)|\(detection.zone)"
    }

    private func isInForwardZone(_ detection: ObjectDetection) -> Bool {
        (0.25...0.75).contains(detection.centerX)
            && detection.bottomRatio <= 0.32
    }

    private func automaticMessage(for detection: ObjectDetection) -> String {
        let label: String
        if detection.label == "person" {
            label = "A person"
        } else if detection.label == "stairs" {
            label = "Stairs"
        } else {
            let article = "aeiou".contains(detection.label.first ?? "z") ? "An" : "A"
            label = "\(article) \(detection.label)"
        }

        let verb = detection.label == "stairs" ? "have appeared" : "has appeared"
        let position = SceneDescription.positionPhrase(for: detection.zone)
        let distance = SceneDescription.distancePhrase(for: detection.distanceSteps)
        return "\(label) \(verb) \(position)\(distance)."
    }
}

private enum CameraError: LocalizedError {
    case backCameraUnavailable
    case cannotAddCameraInput
    case cannotAddVideoOutput

    var errorDescription: String? {
        switch self {
        case .backCameraUnavailable:
            return "The iPhone back camera is unavailable."
        case .cannotAddCameraInput:
            return "The camera input could not be started."
        case .cannotAddVideoOutput:
            return "The camera video stream could not be started."
        }
    }
}
