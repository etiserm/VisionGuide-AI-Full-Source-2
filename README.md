# VisionGuide AI

**An experimental, on-device computer-vision assistant that describes nearby objects with speech.** VisionGuide is a portfolio project exploring accessible interaction across iOS, Android, and a macOS/Python prototype.

> **Safety:** This is an experimental prototype, not a mobility aid or safety system. It can miss objects, misclassify them, or speak too late. Do not use it to navigate, cross streets, or avoid hazards. Continue using established mobility tools and skills.

[Русская версия](README_RU.md) · [Project overview](docs/PROJECT_OVERVIEW.md) · [Architecture](docs/ARCHITECTURE.md) · [Safety & privacy](docs/SAFETY_AND_PRIVACY.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## What’s in this repository

| Platform | Project | What it contains |
| --- | --- | --- |
| iOS | [`iOS/VisionGuide-AI-iOS/ios`](iOS/VisionGuide-AI-iOS/ios) | SwiftUI camera app, Core ML object detection, spoken descriptions |
| Android | [`Android/VisionGuide-AI-Android`](Android/VisionGuide-AI-Android) | Kotlin/Compose camera app, ONNX Runtime detection, spoken descriptions |
| macOS / Python | [`PyCharm/VisionGuide-Python`](PyCharm/VisionGuide-Python) | Camera and image experiments, object detection, segmentation visualization and speech utilities |

The mobile apps are designed for on-device inference. The repository contains source code and class labels, but omits pretrained and exported model binaries and demo images while their redistribution rights are being checked. See the platform guides for model setup. The apps provide on-demand scene descriptions, speech controls, and selected automatic announcements. Distance/range language, where present, is a rough estimate based on object size and camera geometry; it can be wrong and is not a measurement.

## Run the apps

### iOS

1. Open [`VisionGuide.xcodeproj`](iOS/VisionGuide-AI-iOS/ios/VisionGuide/VisionGuide.xcodeproj) in Xcode on macOS.
2. Select the VisionGuide target, set a signing team, connect an iPhone, and run.
3. Grant camera access when prompted.

See the [iOS guide](iOS/VisionGuide-AI-iOS/ios/README.md) for controls, voice setup, model export, and implementation notes. The compiled Core ML model is not included. Follow the iOS guide to generate it after reviewing the model terms; the Xcode project builds without it, but inference needs the model at runtime.

### Android

1. Open [`Android/VisionGuide-AI-Android`](Android/VisionGuide-AI-Android) in Android Studio and allow Gradle sync.
2. Install the Android SDK platform required by the project (API 36), connect a device running Android 8.0/API 26 or later, and run the app.
3. Grant camera access.

See the [Android guide](Android/VisionGuide-AI-Android/README.md). The ONNX model is not included. Follow the Android guide to generate it after reviewing the model terms; the app requires that asset for inference.

### Python prototype (macOS)

```bash
cd PyCharm/VisionGuide-Python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python live_camera.py
```

Python inference also needs a legally obtained YOLO checkpoint at the filename expected by the script. For a still-image demo, place an image at `research/gbhjn.jpg` and run `python detect.py` from that directory; sample images are not included. macOS may ask the terminal or IDE for camera permission. The Python prototype has separate experimental components and is not identical to either mobile app.

## Documentation

- [Project overview and scope](docs/PROJECT_OVERVIEW.md)
- [Architecture and run guide](docs/ARCHITECTURE.md)
- [Experiments and metrics](docs/EXPERIMENTS.md)
- [Safety, privacy, and user feedback](docs/SAFETY_AND_PRIVACY.md)
- [Third-party notices and licensing status](THIRD_PARTY_NOTICES.md)
- [Contributing](CONTRIBUTING.md)

## Project status

This repository contains prototypes and research artifacts. A packaged source archive does not establish that every platform was freshly built or tested. No benchmark or usability claim is made here. See [experiments](docs/EXPERIMENTS.md) for the available evidence and its limits.

## License

No project-wide open-source license has been provided. Until contributors add one and confirm rights for all included material, reuse and redistribution permissions are not granted by this repository. Model and dependency terms are separate; see [Third-party notices](THIRD_PARTY_NOTICES.md).
