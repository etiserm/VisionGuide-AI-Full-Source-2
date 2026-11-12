"""Export the VisionGuide YOLO-World model for the Android app."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

ANDROID_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ANDROID_DIR / "app" / "src" / "main" / "assets"
LABELS_FILE = ASSETS_DIR / "classes.json"
TARGET_MODEL = ASSETS_DIR / "visionguide.onnx"


def main() -> None:
    classes = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not classes or not all(
        isinstance(label, str) and label.strip() for label in classes
    ):
        raise ValueError(f"Invalid object-label file: {LABELS_FILE}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="visionguide-android-export-") as temp:
            staging = Path(temp)
            os.chdir(staging)
            model = YOLO("yolov8s-worldv2.pt")
            model.set_classes(classes)
            exported = model.export(
                format="onnx",
                imgsz=640,
                nms=True,
                conf=0.15,
                iou=0.5,
                max_det=100,
                simplify=True,
                opset=17,
                dynamic=False,
                half=False,
                device="cpu",
            )
            exported_file = Path(exported).resolve()
            staged_target = ASSETS_DIR / ".visionguide.onnx.tmp"
            shutil.copy2(exported_file, staged_target)
            os.replace(staged_target, TARGET_MODEL)
    finally:
        os.chdir(old_cwd)

    print(f"Saved {len(classes)} Android object classes to {LABELS_FILE}")
    print(f"Saved ONNX model to {TARGET_MODEL}")


if __name__ == "__main__":
    main()
