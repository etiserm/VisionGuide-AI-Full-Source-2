import CoreML
import CoreVideo
import ImageIO
import Vision

enum FrameProcessorError: LocalizedError {
    case modelMissing
    case labelsMissing
    case invalidModelOutput

    var errorDescription: String? {
        switch self {
        case .modelMissing:
            return "The VisionGuide Core ML model is not included in this app yet."
        case .labelsMissing:
            return "The VisionGuide object labels are missing from this app."
        case .invalidModelOutput:
            return "The VisionGuide model returned an unsupported result format."
        }
    }
}

final class FrameProcessor {
    private let request: VNCoreMLRequest
    private let labels: [String]
    private let inputSize: CGSize

    init(modelName: String = "VisionGuide") throws {
        guard let modelURL = Bundle.main.url(
            forResource: modelName,
            withExtension: "mlmodelc"
        ) else {
            throw FrameProcessorError.modelMissing
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        guard let labelsURL = Bundle.main.url(forResource: "VisionGuideClasses", withExtension: "json") else {
            throw FrameProcessorError.labelsMissing
        }
        labels = try JSONDecoder().decode([String].self, from: Data(contentsOf: labelsURL))
        guard !labels.isEmpty,
              let imageConstraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint
        else {
            throw FrameProcessorError.invalidModelOutput
        }
        inputSize = CGSize(
            width: CGFloat(imageConstraint.pixelsWide),
            height: CGFloat(imageConstraint.pixelsHigh)
        )

        let visionModel = try VNCoreMLModel(for: model)
        request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFit
    }

    func process(
        _ pixelBuffer: CVPixelBuffer,
        horizontalFieldOfViewDegrees: CGFloat? = nil
    ) throws -> [ObjectDetection] {
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .right,
            options: [:]
        )
        try handler.perform([request])

        if let observations = request.results as? [VNRecognizedObjectObservation] {
            return observations.compactMap { observation in
                guard let label = observation.labels.first else { return nil }
                let normalizedLabel = label.identifier.lowercased()
                let box = observation.boundingBox
                return ObjectDetection(
                    label: normalizedLabel,
                    confidence: label.confidence,
                    boundingBox: box,
                    distanceSteps: DistanceEstimator.estimateSteps(
                        for: normalizedLabel,
                        boundingBox: box,
                        frameHeightPixels: CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
                        horizontalFieldOfViewDegrees: horizontalFieldOfViewDegrees,
                        confidence: label.confidence
                    )
                )
            }
        }

        guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else {
            throw FrameProcessorError.invalidModelOutput
        }

        var outputs: [String: MLMultiArray] = [:]
        for observation in observations {
            if let array = observation.featureValue.multiArrayValue {
                outputs[observation.featureName] = array
            }
        }
        guard let confidence = outputs["confidence"],
              let coordinates = outputs["coordinates"],
              confidence.shape.count == 2,
              coordinates.shape.count == 2,
              coordinates.shape[1].intValue == 4
        else {
            throw FrameProcessorError.invalidModelOutput
        }

        let boxCount = min(confidence.shape[0].intValue, coordinates.shape[0].intValue)
        let classCount = min(labels.count, confidence.shape[1].intValue)
        let contentRect = fittedImageRect(for: pixelBuffer)
        guard contentRect.width > 0, contentRect.height > 0 else { return [] }

        var detections: [ObjectDetection] = []
        // NMS can return a variable number of boxes. Avoid reserving the full upper bound
        // on every frame when only a few objects are normally visible.
        detections.reserveCapacity(min(boxCount, 32))

        for boxIndex in 0..<boxCount {
            var bestClass = -1
            var bestConfidence: Float = 0
            for classIndex in 0..<classCount {
                let score = confidence[[NSNumber(value: boxIndex), NSNumber(value: classIndex)]].floatValue
                if score > bestConfidence {
                    bestConfidence = score
                    bestClass = classIndex
                }
            }
            guard bestClass >= 0, bestConfidence > 0 else { continue }

            let centerX = CGFloat(coordinates[[NSNumber(value: boxIndex), NSNumber(value: 0)]].doubleValue)
            let centerY = CGFloat(coordinates[[NSNumber(value: boxIndex), NSNumber(value: 1)]].doubleValue)
            let width = CGFloat(coordinates[[NSNumber(value: boxIndex), NSNumber(value: 2)]].doubleValue)
            let height = CGFloat(coordinates[[NSNumber(value: boxIndex), NSNumber(value: 3)]].doubleValue)
            guard width > 0, height > 0 else { continue }

            // The exported YOLO model returns normalized center-based boxes in its square input.
            // Vision's .scaleFit adds letterboxing, so map boxes back to the oriented camera frame.
            let mappedCenterX = (centerX * inputSize.width - contentRect.minX) / contentRect.width
            let mappedCenterY = (centerY * inputSize.height - contentRect.minY) / contentRect.height
            let mappedWidth = width * inputSize.width / contentRect.width
            let mappedHeight = height * inputSize.height / contentRect.height
            let minX = max(0, mappedCenterX - mappedWidth / 2)
            let maxX = min(1, mappedCenterX + mappedWidth / 2)
            let minTopY = max(0, mappedCenterY - mappedHeight / 2)
            let maxTopY = min(1, mappedCenterY + mappedHeight / 2)
            guard maxX > minX, maxTopY > minTopY else { continue }

            let label = labels[bestClass].lowercased()
            let box = CGRect(
                x: minX,
                y: 1 - maxTopY,
                width: maxX - minX,
                height: maxTopY - minTopY
            )
            detections.append(
                ObjectDetection(
                    label: label,
                    confidence: bestConfidence,
                    boundingBox: box,
                    distanceSteps: DistanceEstimator.estimateSteps(
                        for: label,
                        boundingBox: box,
                        frameHeightPixels: CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
                        horizontalFieldOfViewDegrees: horizontalFieldOfViewDegrees,
                        confidence: bestConfidence
                    )
                )
            )
        }

        return detections
    }

    private func fittedImageRect(for pixelBuffer: CVPixelBuffer) -> CGRect {
        // .right rotates the landscape sensor buffer into portrait before Vision resizes it.
        let orientedSize = CGSize(
            width: CGFloat(CVPixelBufferGetHeight(pixelBuffer)),
            height: CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        )
        let scale = min(inputSize.width / orientedSize.width, inputSize.height / orientedSize.height)
        let width = orientedSize.width * scale
        let height = orientedSize.height * scale
        return CGRect(
            x: (inputSize.width - width) / 2,
            y: (inputSize.height - height) / 2,
            width: width,
            height: height
        )
    }
}
