import Foundation
import Vision

struct ObjectDetection: Identifiable, Equatable {
    let label: String
    let confidence: Float
    let boundingBox: CGRect
    let distanceSteps: ClosedRange<Int>?

    init(
        label: String,
        confidence: Float,
        boundingBox: CGRect,
        distanceSteps: ClosedRange<Int>? = nil
    ) {
        self.label = label
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.distanceSteps = distanceSteps
    }

    var id: String {
        "\(label)-\(zone)"
    }

    var centerX: CGFloat {
        boundingBox.midX
    }

    var zone: String {
        if centerX < 1.0 / 3.0 {
            return "left"
        }
        if centerX > 2.0 / 3.0 {
            return "right"
        }
        return "center"
    }

    var bottomRatio: CGFloat {
        boundingBox.minY
    }

    var areaRatio: CGFloat {
        boundingBox.width * boundingBox.height
    }

    var priorityScore: Double {
        let urgentVehicles: Set<String> = ["car", "truck", "bus", "motorcycle"]
        let urgentObstacles: Set<String> = [
            "stairs", "staircase", "pothole", "curb", "kerb",
            "construction barrier", "barrier", "traffic cone", "bollard",
            "pole", "manhole",
        ]

        var score: Double
        if urgentVehicles.contains(label) {
            score = 0
        } else if urgentObstacles.contains(label) {
            score = 1
        } else if ["person", "pedestrian", "wheelchair", "bicycle", "scooter"].contains(label) {
            score = 2
        } else if ["traffic light", "traffic sign", "crosswalk"].contains(label) {
            score = 3
        } else if ["dog", "cat", "bird", "horse"].contains(label) {
            score = 5
        } else {
            score = 6
        }

        if (0.30...0.70).contains(centerX), bottomRatio <= 0.35 {
            score -= 1.0
        } else if (0.20...0.80).contains(centerX), bottomRatio <= 0.48 {
            score -= 0.4
        }

        score -= min(Double(areaRatio), 0.20) * 1.5
        score -= Double(confidence) * 0.05
        return score
    }
}

enum DistanceEstimator {
    // These are typical real-world object heights, used only as rough visual references.
    // Large variation between objects of the same class is why the result is a range.
    private static let referenceHeights: [String: CGFloat] = [
        "person": 1.70, "wheelchair": 0.95, "bicycle": 1.05, "scooter": 0.95,
        "motorcycle": 1.20, "car": 1.45, "bus": 3.00, "truck": 2.60,
        "train": 3.60, "tram": 3.00, "dog": 0.55, "cat": 0.28,
        "bench": 0.85, "chair": 0.85, "table": 0.75, "trash can": 0.75,
        "bicycle rack": 0.95, "fire hydrant": 0.75, "parking meter": 1.40,
        "bus shelter": 2.20, "shopping cart": 1.00, "traffic light": 0.80,
        "traffic sign": 0.55, "traffic cone": 0.70, "bollard": 0.80,
        "pole": 2.00, "construction barrier": 0.80, "manhole cover": 0.70,
        "fence": 1.20, "handrail": 0.90, "door": 2.00, "window": 1.00,
        "couch": 0.90, "bed": 0.65, "pillow": 0.25, "cabinet": 1.40,
        "bookshelf": 1.70, "lamp": 0.50, "television": 0.60, "laptop": 0.22,
        "cell phone": 0.16, "remote control": 0.18, "keyboard": 0.15,
        "headphones": 0.18, "bottle": 0.28, "cup": 0.11, "glass": 0.12,
        "bowl": 0.12, "kettle": 0.28, "microwave": 0.35, "oven": 0.90,
        "stove": 0.90, "refrigerator": 1.70, "sink": 0.90, "toilet": 0.75,
        "bathtub": 0.60, "shower": 2.00, "washing machine": 0.85,
        "dishwasher": 0.85, "book": 0.23, "clock": 0.25, "mirror": 0.80,
        "vase": 0.35, "potted plant": 0.60, "backpack": 0.50, "handbag": 0.30,
        "suitcase": 0.60, "umbrella": 0.85, "shoes": 0.15, "box": 0.50,
        "medicine bottle": 0.12, "toothbrush": 0.19, "towel": 0.30,
        "fire extinguisher": 0.55, "baby stroller": 1.05,
    ]

    static func estimateSteps(
        for label: String,
        boundingBox: CGRect,
        frameHeightPixels: CGFloat,
        horizontalFieldOfViewDegrees: CGFloat?,
        confidence: Float
    ) -> ClosedRange<Int>? {
        guard confidence >= 0.45,
              let referenceHeight = referenceHeights[label],
              let horizontalFieldOfViewDegrees,
              (35...100).contains(horizontalFieldOfViewDegrees),
              frameHeightPixels > 0,
              boundingBox.height > 0,
              // A clipped object or a tiny box gives an especially unreliable estimate.
              boundingBox.minY > 0.01,
              boundingBox.maxY < 0.99
        else {
            return nil
        }

        let objectHeightPixels = boundingBox.height * frameHeightPixels
        guard objectHeightPixels >= 24 else { return nil }

        // The back-camera image is rotated to portrait. Its vertical axis corresponds to
        // the sensor's horizontal axis, which is the axis used by videoFieldOfView.
        let halfFieldOfViewRadians = horizontalFieldOfViewDegrees * .pi / 360
        let focalLengthPixels = frameHeightPixels / (2 * tan(halfFieldOfViewRadians))
        let distanceMeters = referenceHeight * focalLengthPixels / objectHeightPixels
        guard (0.4...15).contains(distanceMeters) else { return nil }

        // A step length is highly personal; return a broad range around a typical estimate.
        let estimatedSteps = distanceMeters / 0.68
        let lowerBound = max(1, Int(floor(estimatedSteps * 0.65)))
        let upperBound = max(lowerBound + 1, Int(ceil(estimatedSteps * 1.35)))
        return lowerBound...upperBound
    }
}

enum SceneDescription {
    static func make(from detections: [ObjectDetection], limit: Int = 4) -> String {
        guard !detections.isEmpty else {
            return "I can't clearly identify any objects in the current view."
        }

        var groups: [String: (detection: ObjectDetection, count: Int)] = [:]
        for detection in detections {
            let key = "\(detection.label)|\(detection.zone)"
            if let existing = groups[key] {
                groups[key] = (existing.detection, existing.count + 1)
            } else {
                groups[key] = (detection, 1)
            }
        }

        let ordered = groups.values.sorted {
            if $0.detection.priorityScore == $1.detection.priorityScore {
                return $0.detection.label < $1.detection.label
            }
            return $0.detection.priorityScore < $1.detection.priorityScore
        }

        let phrases = ordered.prefix(limit).map { group in
            let objectName = name(for: group.detection.label, count: group.count)
            let distance = distancePhrase(for: group.detection.distanceSteps)
            return "\(objectName) \(positionPhrase(for: group.detection.zone))\(distance)"
        }

        let list: String
        if phrases.count == 1 {
            list = phrases[0]
        } else {
            list = phrases.dropLast().joined(separator: ", ") + ", and \(phrases[phrases.count - 1])"
        }

        return "I can see \(list)."
    }

    static func positionPhrase(for zone: String) -> String {
        switch zone {
        case "left": return "on the left"
        case "right": return "on the right"
        default: return "near the center"
        }
    }

    static func distancePhrase(for steps: ClosedRange<Int>?) -> String {
        guard let steps else { return "" }
        return ", roughly \(steps.lowerBound) to \(steps.upperBound) steps away"
    }

    private static func name(for label: String, count: Int) -> String {
        if count == 1 {
            if label == "person" { return "a person" }
            if label == "stairs" { return "stairs" }
            if label == "shoes" { return "a pair of shoes" }
            if label == "headphones" { return "a pair of headphones" }
            if label == "keys" { return "a set of keys" }
            let article = "aeiou".contains(label.first ?? "z") ? "an" : "a"
            return "\(article) \(label)"
        }

        let irregularPlurals = [
            "person": "people",
            "bus": "buses",
            "knife": "knives",
            "shelf": "shelves",
            "mouse": "mice",
            "computer mouse": "computer mice",
            "glass": "glasses",
            "tooth": "teeth",
            "foot": "feet",
        ]
        if let plural = irregularPlurals[label] {
            return "\(count) \(plural)"
        }
        if ["shoes", "headphones", "keys", "stairs", "scissors"].contains(label) {
            return "\(count) \(label)"
        }
        if label == "clothing" {
            return "\(count) clothing items"
        }
        if ["ch", "sh", "x", "z"].contains(where: { label.hasSuffix($0) }) {
            return "\(count) \(label)es"
        }
        if label.hasSuffix("s") {
            return "\(count) \(label)"
        }
        return "\(count) \(label)s"
    }
}
