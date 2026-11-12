from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "ios" / "VisionGuide" / "Models"
TOOL_DEPENDENCIES = PROJECT_ROOT / "ios" / "tooldeps"
TARGET_MODEL = OUTPUT_DIR / "VisionGuide.mlpackage"
TARGET_LABELS = OUTPUT_DIR / "VisionGuideClasses.json"
COMPILED_MODEL = OUTPUT_DIR / "VisionGuide.mlmodelc"

sys.path.insert(0, str(TOOL_DEPENDENCIES))

from ultralytics import YOLOWorld

try:
    import coremltools  # noqa: F401
except ImportError as error:
    raise SystemExit(
        "Core ML export tools are missing. Run the install command in ios/README.md first."
    ) from error


def main():
    if not TARGET_LABELS.exists():
        raise FileNotFoundError(f"Object prompt list is missing: {TARGET_LABELS}")
    object_classes = json.loads(TARGET_LABELS.read_text(encoding="utf-8"))
    if not isinstance(object_classes, list) or not object_classes or not all(
        isinstance(label, str) and label.strip() for label in object_classes
    ):
        raise ValueError(f"Object prompt list is invalid: {TARGET_LABELS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_directory = Path.cwd()

    # Export and compile beside the live app resources first. A failed conversion
    # must never remove the model that is currently used by the iOS target.
    with tempfile.TemporaryDirectory(prefix=".visionguide-export-", dir=OUTPUT_DIR) as temporary:
        staging_dir = Path(temporary)
        try:
            os.chdir(staging_dir)
            print("Loading exportable YOLO-World v2 model...")
            model = YOLOWorld("yolov8s-worldv2.pt")
            model.set_classes(object_classes)

            print(f"Embedding {len(object_classes)} VisionGuide object classes.")
            exported = model.export(
                format="coreml",
                imgsz=640,
                nms=True,
                conf=0.15,
            )
            exported_path = Path(exported).resolve()

            if not exported_path.exists():
                raise FileNotFoundError(f"Core ML export was not created: {exported_path}")

            staged_model = staging_dir / TARGET_MODEL.name
            shutil.move(str(exported_path), str(staged_model))
            staged_labels = staging_dir / TARGET_LABELS.name
            staged_labels.write_text(
                json.dumps(object_classes, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "xcrun",
                    "coremlcompiler",
                    "compile",
                    str(staged_model),
                    str(staging_dir),
                    "--platform",
                    "iOS",
                    "--deployment-target",
                    "17.0",
                ],
                check=True,
            )

            staged_compiled_model = staging_dir / COMPILED_MODEL.name
            if not staged_compiled_model.exists():
                raise FileNotFoundError(
                    f"Compiled Core ML model was not created: {staged_compiled_model}"
                )

            backup_dir = staging_dir / "previous-resources"
            backup_dir.mkdir()
            targets_and_staged = [
                (TARGET_MODEL, staged_model),
                (COMPILED_MODEL, staged_compiled_model),
                (TARGET_LABELS, staged_labels),
            ]
            backed_up = []
            installed = []
            try:
                for target, _ in targets_and_staged:
                    if target.exists():
                        os.replace(target, backup_dir / target.name)
                        backed_up.append(target)
                for target, staged in targets_and_staged:
                    os.replace(staged, target)
                    installed.append(target)
            except Exception:
                for target in installed:
                    if target.is_dir():
                        shutil.rmtree(target)
                    elif target.exists():
                        target.unlink()
                for target in backed_up:
                    os.replace(backup_dir / target.name, target)
                raise
        finally:
            os.chdir(original_directory)

    print(f"Saved iPhone model: {TARGET_MODEL}")
    print(f"Compiled iPhone model: {COMPILED_MODEL}")
    print(f"Saved object labels: {TARGET_LABELS}")
    print("The Xcode target includes both files as build resources.")


if __name__ == "__main__":
    main()
