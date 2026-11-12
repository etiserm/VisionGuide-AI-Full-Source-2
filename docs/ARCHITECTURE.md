# Architecture and run guide

## Processing flow

**Mobile apps:** camera frame → platform image preprocessing → object detector (Core ML on iOS / ONNX Runtime on Android) → confidence and event filtering → short natural-language description → platform text-to-speech.

The iOS and Android apps use separate native implementations and packaged model formats. They do not send camera frames to a project server. The iOS guide reports analysis up to four frames per second; the Android guide reports a 2.5 FPS cap. Actual throughput depends on hardware and conditions.

**Python/macOS prototype:** camera or still image → Python/OpenCV and Ultralytics components → detections and experimental scene visualizations → speech utilities. The segmentation display is experimental and is not documented as a source of spoken navigation instructions.

## Run locally

Use the platform-specific instructions in [README](../README.md), [iOS README](../iOS/VisionGuide-AI-iOS/ios/README.md), and [Android README](../Android/VisionGuide-AI-Android/README.md). Python dependencies are listed in `PyCharm/VisionGuide-Python/requirements.txt`.

The Android project specifies Android Gradle Plugin 8.13.2, Kotlin 2.2.20, Java 17, compile SDK 36, and minimum API 26 in its project files. The iOS project requires Xcode on macOS and a signing team to install on a device. Toolchains and SDKs change; consult the project files and platform IDE for the versions supported on your machine.

## Model artifacts

The repository includes class-label lists but omits pretrained and exported model binaries while redistribution rights are being checked. The mobile apps need a model asset at runtime; the iOS Xcode project can build without the compiled model, but inference reports a missing-model error. The Android app also needs its ONNX asset before camera inference can run. Python detection needs a YOLO checkpoint and local input image.

Platform guides describe export steps. Export scripts can download upstream weights; review model and dependency terms before downloading, exporting, or distributing artifacts. See [Third-party notices](../THIRD_PARTY_NOTICES.md).

## Diagnostics

When reporting an issue, include platform, device/OS, app revision, model artifact, reproduction steps, and relevant error output. Do not attach camera images or voice recordings containing personal information unless the people depicted or recorded have agreed.
