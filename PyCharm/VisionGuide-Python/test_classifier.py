from perception.object_classifier import (
    classify_detections,
    create_scene_summary,
)


detections = [
    {
        "class_name": "person",
        "confidence": 0.94,
        "box": [400, 100, 550, 500],
        "track_id": 1,
    },
    {
        "class_name": "bench",
        "confidence": 0.87,
        "box": [50, 200, 250, 400],
        "track_id": 2,
    },
    {
        "class_name": "traffic light",
        "confidence": 0.91,
        "box": [700, 20, 780, 150],
        "track_id": 3,
    },
    {
        "class_name": "stairs",
        "confidence": 0.82,
        "box": [550, 300, 750, 600],
        "track_id": 4,
    },
]

classified = classify_detections(
    detections,
    image_width=800,
)

scene = create_scene_summary(classified)

print(scene)