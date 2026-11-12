from typing import Any


OBJECT_CATEGORIES = {
    "people": {
        "person",
        "pedestrian",
        "people",
        "child",
        "wheelchair",
    },
    "vehicles": {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "scooter",
        "train",
        "tram",
    },
    "animals": {
        "dog",
        "cat",
        "bird",
        "horse",
    },
    "street_furniture": {
        "bench",
        "chair",
        "table",
        "trash bin",
        "waste bin",
        "mailbox",
        "bicycle rack",
        "street lamp",
        "fire hydrant",
        "bus shelter",
        "parking meter",
    },
    "road_infrastructure": {
        "traffic light",
        "stop sign",
        "street sign",
        "crosswalk",
        "pedestrian crossing",
        "traffic sign",
        "traffic cone",
        "barrier",
        "road barrier",
    },
    "obstacles": {
        "pothole",
        "stairs",
        "staircase",
        "curb",
        "kerb",
        "ramp",
        "fence",
        "pole",
        "bollard",
        "handrail",
        "manhole",
        "shopping cart",
        "construction barrier",
    },
    "buildings": {
        "building",
        "door",
        "window",
        "wall",
        "garage",
        "shop",
        "entrance",
    },
    "nature": {
        "tree",
        "plant",
        "bush",
        "grass",
        "flower",
    },
}


ALIASES = {
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "person": "person",
    "people": "people",
    "bin": "trash bin",
    "trash can": "trash bin",
    "garbage can": "trash bin",
    "garbage bin": "trash bin",
    "rubbish bin": "trash bin",
    "wastebasket": "trash bin",
    "traffic signal": "traffic light",
    "cross walk": "crosswalk",
    "steps": "stairs",
    "stair": "stairs",
    "stairway": "stairs",
    "kerb": "curb",
    "curb stone": "curb",
    "kerb stone": "curb",
    "bollards": "bollard",
    "manhole cover": "manhole",
    "crosswalk marking": "crosswalk",
    "pedestrian crossing": "crosswalk",
    "wheel chair": "wheelchair",
    "streetlight": "street lamp",
    "lamp post": "street lamp",
    "bus stop shelter": "bus shelter",
}


CATEGORY_PRIORITY = {
    "people": 1,
    "vehicles": 2,
    "obstacles": 3,
    "road_infrastructure": 4,
    "animals": 5,
    "street_furniture": 6,
    "buildings": 7,
    "nature": 8,
    "unknown": 9,
}


LABEL_PRIORITY = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "stairs": 1,
    "staircase": 1,
    "pothole": 1,
    "curb": 1,
    "kerb": 1,
    "construction barrier": 1,
    "barrier": 1,
    "traffic cone": 1,
    "bollard": 1,
    "pole": 1,
    "manhole": 1,
    "bicycle": 2,
    "scooter": 2,
    "person": 2,
    "pedestrian": 2,
    "wheelchair": 2,
    "traffic light": 3,
    "traffic sign": 3,
    "crosswalk": 3,
}


def normalize_label(label: str) -> str:
    label = str(label).lower().strip()
    label = label.replace("_", " ")
    label = " ".join(label.split())

    return ALIASES.get(label, label)


def get_category(label: str) -> str:
    normalized = normalize_label(label)

    for category, labels in OBJECT_CATEGORIES.items():
        if normalized in labels:
            return category

    return "unknown"


def get_position(box: list, image_width: int) -> str:
    x1, _, x2, _ = box
    center_x = (x1 + x2) / 2

    if center_x < image_width / 3:
        return "left"

    if center_x > image_width * 2 / 3:
        return "right"

    return "center"


def classify_detection(
    detection: dict[str, Any],
    image_width: int,
    image_height: int | None = None,
) -> dict[str, Any]:

    label = normalize_label(detection.get("class_name", "unknown"))
    confidence = float(detection.get("confidence", 0.0))
    box = detection.get("box")

    if box is not None and len(box) == 4:
        position = get_position(box, image_width)
    else:
        position = "unknown"

    category = get_category(label)

    center_x_ratio = None
    bottom_ratio = None
    area_ratio = None

    if box is not None and len(box) == 4 and image_width > 0:
        x1, y1, x2, y2 = [float(value) for value in box]
        center_x_ratio = ((x1 + x2) / 2) / image_width

        if image_height is not None and image_height > 0:
            bottom_ratio = y2 / image_height
            area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio /= image_width * image_height

    return {
        "label": label,
        "category": category,
        "confidence": confidence,
        "position": position,
        "box": box,
        "track_id": detection.get("track_id"),
        "center_x_ratio": center_x_ratio,
        "bottom_ratio": bottom_ratio,
        "area_ratio": area_ratio,
    }


def detection_priority_score(detection: dict[str, Any]) -> float:
    """Lower scores are more important for spoken live-camera alerts."""
    label = normalize_label(detection.get("label", "unknown"))
    category = detection.get("category", get_category(label))
    priority = float(
        LABEL_PRIORITY.get(label, CATEGORY_PRIORITY.get(category, 9))
    )

    center_x = detection.get("center_x_ratio")
    bottom = detection.get("bottom_ratio")
    area = detection.get("area_ratio") or 0.0

    # Frame position and box size are rough relevance cues, not range estimates.
    if center_x is not None and bottom is not None:
        if 0.30 <= center_x <= 0.70 and bottom >= 0.65:
            priority -= 1.0
        elif 0.20 <= center_x <= 0.80 and bottom >= 0.52:
            priority -= 0.4

    priority -= min(max(float(area), 0.0), 0.20) * 1.5
    priority -= float(detection.get("confidence", 0.0)) * 0.05
    return priority


def classify_detections(
    detections: list[dict[str, Any]],
    image_width: int,
    confidence_threshold: float = 0.35,
    image_height: int | None = None,
) -> list[dict[str, Any]]:

    results = []

    for detection in detections:
        confidence = float(detection.get("confidence", 0.0))

        if confidence < confidence_threshold:
            continue

        classified = classify_detection(
            detection,
            image_width,
            image_height,
        )

        results.append(classified)

    results.sort(
        key=lambda item: (
            CATEGORY_PRIORITY.get(item["category"], 99),
            -item["confidence"],
        )
    )

    return results


def group_by_category(
    detections: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:

    grouped = {}

    for detection in detections:
        category = detection["category"]

        if category not in grouped:
            grouped[category] = []

        grouped[category].append(detection)

    return grouped


def create_scene_summary(
    detections: list[dict[str, Any]],
) -> dict[str, Any]:

    grouped = group_by_category(detections)

    return {
        "total_objects": len(detections),
        "categories": {
            category: len(objects)
            for category, objects in grouped.items()
        },
        "objects": detections,
    }
