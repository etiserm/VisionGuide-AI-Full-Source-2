
from ultralytics import YOLOWorld
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from pathlib import Path
from collections import Counter
import asyncio
import os
import tempfile
import torch
import cv2
import numpy as np
import subprocess
import sys
from perception.object_classifier import (
    classify_detections,
    create_scene_summary,
)

YOLO_MODEL_PATH = "yolov8s-world.pt"
YOLO_CLASSES = [
    "person",
    "wheelchair",
    "bicycle",
    "scooter",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "train",
    "tram",
    "dog",
    "cat",
    "bench",
    "chair",
    "table",
    "trash can",
    "bicycle rack",
    "fire hydrant",
    "parking meter",
    "bus shelter",
    "shopping cart",
    "traffic light",
    "traffic sign",
    "traffic cone",
    "crosswalk",
    "stairs",
    "curb",
    "pothole",
    "bollard",
    "pole",
    "construction barrier",
    "manhole cover",
    "ramp",
    "fence",
    "handrail",
    "door",
    "building",
    "wall",
    "shop",
    "entrance",
    "window",
    "tree",
    # Everyday indoor objects (open-vocabulary YOLO-World prompts).
    "couch",
    "bed",
    "pillow",
    "cabinet",
    "bookshelf",
    "lamp",
    "television",
    "laptop",
    "cell phone",
    "remote control",
    "keyboard",
    "computer mouse",
    "headphones",
    "bottle",
    "cup",
    "glass",
    "bowl",
    "plate",
    "fork",
    "knife",
    "spoon",
    "kettle",
    "microwave",
    "oven",
    "stove",
    "refrigerator",
    "sink",
    "toilet",
    "bathtub",
    "shower",
    "washing machine",
    "dishwasher",
    "book",
    "clock",
    "mirror",
    "vase",
    "potted plant",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "shoes",
    "clothing",
    "box",
    "keys",
    "medicine bottle",
    "toothbrush",
    "towel",
    "fire extinguisher",
    "smoke detector",
    "baby stroller",
]
SEGMENTATION_MODEL = "facebook/mask2former-swin-small-cityscapes-semantic"

DEFAULT_IMAGE = "research/gbhjn.jpg"
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45
VOICE_NAME = "en-US-EmmaMultilingualNeural"
VOICE_RATE = "-3%"
VOICE_PITCH = "+0Hz"

BOTTOM_REGION_START = 0.50
OBSTACLE_PADDING = 0.15
OBSTACLE_DILATION = 15
ROAD_BUFFER_SIZE = 9

ZONE_NAMES = ["left", "center", "right"]
ZONE_SPOKEN_LABELS = {
    "left": "On the left",
    "center": "Near the center",
    "right": "On the right",
}

CATEGORY_COLORS = {
    "people": (0, 165, 255),
    "vehicles": (0, 0, 255),
    "animals": (0, 200, 0),
    "street_furniture": (0, 255, 255),
    "road_infrastructure": (255, 255, 0),
    "obstacles": (255, 0, 255),
    "buildings": (180, 120, 60),
    "nature": (80, 180, 80),
    "unknown": (190, 190, 190),
}

OBSTACLE_CLASSES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "pole",
    "traffic light",
    "traffic sign"
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_detection_model():
    print(
        f"\nLoading YOLO-World with {len(YOLO_CLASSES)} object prompts..."
    )
    model = YOLOWorld(YOLO_MODEL_PATH)
    model.set_classes(YOLO_CLASSES)
    return model


def normalize_label(label):
    return str(label).lower().replace("_", " ").strip()


def get_class_name(model, class_id):
    labels = model.config.id2label
    name = labels.get(class_id, labels.get(str(class_id), str(class_id)))
    return normalize_label(name)


def find_class_id(model, target_name):
    for class_id in range(model.config.num_labels):
        if get_class_name(model, class_id) == target_name:
            return class_id
    return None


def get_position(center_x, width):
    if center_x < width / 3:
        return "left"
    if center_x > 2 * width / 3:
        return "right"
    return "center"


def classify_yolo_result(
    yolo_result,
    yolo_model,
    image_width,
    image_height=None,
    confidence_threshold=CONFIDENCE,
):
    boxes = getattr(yolo_result, "boxes", None)

    if boxes is None:
        return []

    detections = []

    for box in boxes:
        class_id = int(box.cls[0])
        track_ids = getattr(box, "id", None)
        track_id = None

        if track_ids is not None:
            track_value = track_ids[0]
            track_id = int(
                track_value.item()
                if hasattr(track_value, "item")
                else track_value
            )

        detections.append({
            "class_name": yolo_model.names[class_id],
            "confidence": float(box.conf[0]),
            "box": [float(value) for value in box.xyxy[0].tolist()],
            "track_id": track_id,
        })

    return classify_detections(
        detections,
        image_width=image_width,
        image_height=image_height,
        confidence_threshold=confidence_threshold,
    )


def draw_classified_detections(image, detections):
    output = image.copy()
    height, width = output.shape[:2]

    for detection in detections:
        box = detection.get("box")

        if box is None or len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, 0), width - 1)
        y2 = min(max(y2, 0), height - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        category = detection.get("category", "unknown")
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["unknown"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = detection.get("label", "unknown")
        category_label = category.replace("_", " ")
        confidence = float(detection.get("confidence", 0.0))
        track_id = detection.get("track_id")
        track_label = f" #{track_id}" if track_id is not None else ""
        text = f"{label} [{category_label}] {confidence:.2f}{track_label}"

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )
        text_x = min(x1, max(0, width - text_width - 8))
        text_y = max(text_height + baseline + 6, y1 - 6)

        if text_y + baseline >= height:
            text_y = max(text_height + baseline + 6, height - baseline - 2)

        cv2.rectangle(
            output,
            (text_x, text_y - text_height - baseline - 4),
            (
                min(width - 1, text_x + text_width + 8),
                min(height - 1, text_y + baseline),
            ),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            text,
            (text_x + 4, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def format_object(name, count):
    if name == "person":
        return "a person" if count == 1 else f"{count} people"
    if count == 1:
        article = "an" if name[:1].lower() in "aeiou" else "a"
        return f"{article} {name}"
    if name == "bus":
        return f"{count} buses"
    if name.endswith("s"):
        return f"{count} {name}"
    return f"{count} {name}s"


def describe_objects(detections):
    parts = []
    counts = Counter(
        (detection["position"], detection["label"])
        for detection in detections
        if detection.get("position") in ZONE_NAMES
    )

    for position in ZONE_NAMES:
        objects = [
            (name, count)
            for (zone, name), count in counts.items()
            if zone == position
        ]

        if objects:
            description = ", ".join(
                format_object(name, count)
                for name, count in objects
            )
            parts.append(f"{ZONE_SPOKEN_LABELS[position]}: {description}")

    if not parts:
        return "No objects detected."

    return ". ".join(parts) + "."


def load_segmentation_model(device):
    print("\nLoading Mask2Former...")

    processor = AutoImageProcessor.from_pretrained(
        SEGMENTATION_MODEL
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        SEGMENTATION_MODEL
    )

    model.to(device)
    model.eval()

    return processor, model


def run_segmentation(image, processor, model, device):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    inputs = processor(
        images=rgb_image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    segmentation = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.shape[:2]]
    )[0]

    return segmentation.cpu().numpy()


def print_class_statistics(segmentation, model):
    print("\nMask2Former segmentation:")

    for class_id in np.unique(segmentation):
        name = get_class_name(model, int(class_id))
        percentage = np.mean(segmentation == class_id) * 100
        print(f"{name}: {percentage:.1f}%")


def create_segmentation_visualization(image, segmentation, model):
    color_mask = np.zeros_like(image)

    for class_id in np.unique(segmentation):
        name = get_class_name(model, int(class_id))

        if name == "sidewalk":
            color = (0, 165, 255)
        elif name == "road":
            color = (0, 0, 255)
        elif name == "person":
            color = (255, 0, 0)
        elif name in {"car", "truck", "bus", "motorcycle"}:
            color = (0, 0, 180)
        else:
            color = (
                (int(class_id) * 37 + 70) % 256,
                (int(class_id) * 67 + 120) % 256,
                (int(class_id) * 97 + 180) % 256
            )

        color_mask[segmentation == class_id] = color

    return cv2.addWeighted(image, 0.55, color_mask, 0.45, 0)


def create_obstacle_mask(image, segmentation, model, yolo_result):
    height, width = image.shape[:2]

    obstacle_mask = np.zeros((height, width), dtype=np.uint8)

    for class_id in np.unique(segmentation):
        name = get_class_name(model, int(class_id))

        if name in OBSTACLE_CLASSES:
            obstacle_mask[segmentation == class_id] = 255

    for box in yolo_result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        box_width = x2 - x1
        box_height = y2 - y1

        padding_x = int(box_width * OBSTACLE_PADDING)
        padding_y = int(box_height * OBSTACLE_PADDING)

        x1 = max(0, int(x1 - padding_x))
        y1 = max(0, int(y1 - padding_y))
        x2 = min(width, int(x2 + padding_x))
        y2 = min(height, int(y2 + padding_y))

        obstacle_mask[y1:y2, x1:x2] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (OBSTACLE_DILATION, OBSTACLE_DILATION)
    )

    return cv2.dilate(obstacle_mask, kernel, iterations=1)


def create_pedestrian_masks(image, segmentation, model, yolo_result):
    sidewalk_id = find_class_id(model, "sidewalk")
    road_id = find_class_id(model, "road")

    if sidewalk_id is None or road_id is None:
        raise RuntimeError(
            "Mask2Former model must contain road and sidewalk classes."
        )

    sidewalk_mask = (
        segmentation == sidewalk_id
    ).astype(np.uint8) * 255

    road_mask = (
        segmentation == road_id
    ).astype(np.uint8) * 255

    obstacle_mask = create_obstacle_mask(
        image,
        segmentation,
        model,
        yolo_result
    )

    road_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ROAD_BUFFER_SIZE, ROAD_BUFFER_SIZE)
    )

    road_buffer = cv2.dilate(
        road_mask,
        road_kernel,
        iterations=1
    )

    free_mask = sidewalk_mask.copy()
    free_mask[road_buffer > 0] = 0
    free_mask[obstacle_mask > 0] = 0

    return sidewalk_mask, road_mask, obstacle_mask, free_mask


def analyze_sidewalk(sidewalk_mask, road_mask, free_mask):
    height, width = sidewalk_mask.shape
    start_y = int(height * BOTTOM_REGION_START)

    sidewalk_bottom = sidewalk_mask[start_y:, :]
    road_bottom = road_mask[start_y:, :]
    free_bottom = free_mask[start_y:, :]

    zone_width = width // 3
    results = {}

    for index, zone in enumerate(ZONE_NAMES):
        x1 = index * zone_width
        x2 = width if index == 2 else (index + 1) * zone_width

        sidewalk_zone = sidewalk_bottom[:, x1:x2]
        road_zone = road_bottom[:, x1:x2]
        free_zone = free_bottom[:, x1:x2]

        total_pixels = sidewalk_zone.size
        sidewalk_pixels = np.count_nonzero(sidewalk_zone)
        road_pixels = np.count_nonzero(road_zone)
        free_pixels = np.count_nonzero(free_zone)

        results[zone] = {
            "sidewalk": sidewalk_pixels / total_pixels * 100,
            "road": road_pixels / total_pixels * 100,
            "potential_free": free_pixels / total_pixels * 100,
            "remaining": (
                free_pixels / sidewalk_pixels * 100
                if sidewalk_pixels else 0
            )
        }

    return results


def add_analysis_boundary(image):
    output = image.copy()
    height, width = image.shape[:2]
    boundary_y = int(height * BOTTOM_REGION_START)

    cv2.line(
        output,
        (0, boundary_y),
        (width, boundary_y),
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "Analysis region below this line",
        (15, boundary_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return output


def create_mask_visualization(image, mask, color):
    output = image.copy()
    overlay = np.zeros_like(image)
    overlay[:] = color

    selected = mask > 0

    output[selected] = cv2.addWeighted(
        image[selected],
        0.45,
        overlay[selected],
        0.55,
        0
    )

    return add_analysis_boundary(output)


def create_free_space_visualization(image, sidewalk_mask, free_mask):
    output = image.copy()

    sidewalk_layer = np.zeros_like(image)
    sidewalk_layer[:] = (0, 165, 255)

    free_layer = np.zeros_like(image)
    free_layer[:] = (0, 220, 0)

    sidewalk_selected = sidewalk_mask > 0
    free_selected = free_mask > 0

    output[sidewalk_selected] = cv2.addWeighted(
        image[sidewalk_selected],
        0.45,
        sidewalk_layer[sidewalk_selected],
        0.55,
        0
    )

    output[free_selected] = cv2.addWeighted(
        image[free_selected],
        0.35,
        free_layer[free_selected],
        0.65,
        0
    )

    return add_analysis_boundary(output)


def create_combined_visualization(image, free_mask, detections):
    output = image.copy()

    green_layer = np.zeros_like(image)
    green_layer[:] = (0, 220, 0)

    selected = free_mask > 0

    output[selected] = cv2.addWeighted(
        image[selected],
        0.4,
        green_layer[selected],
        0.6,
        0
    )

    output = draw_classified_detections(output, detections)
    return add_analysis_boundary(output)


def build_report(results):
    parts = []

    for zone in ZONE_NAMES:
        data = results[zone]

        parts.append(
            f"{zone}: {data['sidewalk']:.0f} percent sidewalk, "
            f"{data['road']:.0f} percent road, "
            f"{data['potential_free']:.0f} percent potential free pixels"
        )

    return (
        "Experimental pedestrian area analysis. "
        + ". ".join(parts)
        + ". This is not a safety assessment."
    )


def speak_text(text):
    audio_path = None

    try:
        import edge_tts

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False,
        ) as audio_file:
            audio_path = audio_file.name

        async def generate_audio():
            communicate = edge_tts.Communicate(
                text,
                VOICE_NAME,
                rate=VOICE_RATE,
                pitch=VOICE_PITCH,
            )
            await communicate.save(audio_path)

        asyncio.run(generate_audio())
        subprocess.run(["afplay", audio_path], check=False)

    except Exception as error:
        print(f"Neural voice error: {error}")

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


def main():
    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_IMAGE
    )

    image_file = Path(image_path)

    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_file))

    if image is None:
        raise ValueError("Unable to read image")

    height, width = image.shape[:2]
    device = get_device()

    print("\n--- VISIONGUIDE AI ---")
    print(f"Image: {image_path}")
    print(f"Dimensions: {width}x{height}")
    print(f"Device: {device}")

    yolo_model = load_detection_model()

    yolo_results = yolo_model.predict(
        source=image,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        device=str(device),
        verbose=False
    )

    yolo_result = yolo_results[0]
    classified_detections = classify_yolo_result(
        yolo_result,
        yolo_model,
        image_width=width,
    )

    print("\nYOLO detections:")

    for detection in classified_detections:
        print(
            f"{detection['label']} | {detection['category']} | "
            f"{detection['position']} | {detection['confidence']:.2f}"
        )

    scene_summary = create_scene_summary(classified_detections)
    category_summary = scene_summary["categories"]
    category_text = ", ".join(
        f"{category}: {count}"
        for category, count in category_summary.items()
    ) or "none"
    print(
        f"Classified objects: {scene_summary['total_objects']} "
        f"({category_text})"
    )

    processor, segmentation_model = load_segmentation_model(device)

    print("\nRunning Mask2Former segmentation...")

    segmentation = run_segmentation(
        image,
        processor,
        segmentation_model,
        device
    )

    print_class_statistics(segmentation, segmentation_model)

    sidewalk_mask, road_mask, obstacle_mask, free_mask = (
        create_pedestrian_masks(
            image,
            segmentation,
            segmentation_model,
            yolo_result
        )
    )

    results = analyze_sidewalk(
        sidewalk_mask,
        road_mask,
        free_mask
    )

    object_report = describe_objects(classified_detections)
    free_space_report = build_report(results)

    print("\nObject report:")
    print(object_report)

    print("\nMask2Former sidewalk analysis:")

    for zone, data in results.items():
        print(
            f"{zone}: "
            f"sidewalk={data['sidewalk']:.1f}%, "
            f"road={data['road']:.1f}%, "
            f"potential_free={data['potential_free']:.1f}%, "
            f"remaining={data['remaining']:.1f}%"
        )

    print("\nFree space report:")
    print(free_space_report)

    speech = object_report

    print("\nVoice output:")

    speak_text(speech)

    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_file.stem

    outputs = {
        output_dir / f"{stem}_detected.jpg": draw_classified_detections(
            image,
            classified_detections,
        ),
        output_dir / f"{stem}_segmentation.jpg": (
            create_segmentation_visualization(
                image,
                segmentation,
                segmentation_model
            )
        ),
        output_dir / f"{stem}_sidewalk.jpg": (
            create_mask_visualization(
                image,
                sidewalk_mask,
                (0, 165, 255)
            )
        ),
        output_dir / f"{stem}_free_space.jpg": (
            create_free_space_visualization(
                image,
                sidewalk_mask,
                free_mask
            )
        ),
        output_dir / f"{stem}_combined.jpg": (
            create_combined_visualization(
                image,
                free_mask,
                classified_detections,
            )
        )
    }

    for path, output_image in outputs.items():
        if not cv2.imwrite(str(path), output_image):
            raise RuntimeError(f"Unable to save image: {path}")

    print("\nSaved results:")

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
