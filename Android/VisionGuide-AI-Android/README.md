# VisionGuide AI for Android

This is a native Android Studio project for the VisionGuide iPhone prototype. It
uses 93 YOLO-World object prompts with ONNX Runtime Android. The exported
ONNX model is not included while redistribution terms are being reviewed.

## Open and run

1. Install Android Studio and an Android SDK for API 36. Android Studio includes
   the JDK used by Gradle.
2. Extract the archive, open the `VisionGuide-AI-Android` folder in Android
   Studio, and allow Gradle to sync.
3. Connect an Android phone, select it in the run-device menu, and press **Run**.
4. Add a legally obtained `visionguide.onnx` to `app/src/main/assets/` as described below.
5. Allow camera access. The app analyzes images locally on the phone and does not
   save or upload camera frames.

The first camera start takes longer while the ONNX model is opened. The model
session is released when the camera stops. Camera analysis is capped at 2.5
frames per second, keeps only the newest pending frame, and uses two ONNX
intra-op CPU threads. This limits queued work and avoids processing old frames.

## Features

- Large center control starts the rear camera; tapping again describes the view.
- English speech output with female and male voice choices where the installed
  Android text-to-speech engine identifies voices by gender.
- Adjustable speech volume and recognition sensitivity.
- Optional repeated, important-object announcements with a cooldown.
- Automatic warnings require the same important object in three analyzed frames
  and at least 45% confidence; a higher sensitivity setting raises that bar.
- Approximate step ranges for supported objects, derived from the detected box,
  camera focal length and typical object sizes. No distance is spoken when the
  camera information or detection is insufficient. The estimate is not a
  measurement and can be wrong.
- Beige background, large charcoal buttons, pause/resume and stop controls.
- Camera errors and repeated inference errors stop cleanly and can be retried;
  camera permission denial includes a shortcut to Android app settings.
- Voice instructions wait for text-to-speech initialization instead of being
  dropped, and speech is routed as accessibility output.

Android text-to-speech engines label voices differently. If the phone does not
identify separate male and female English voices, both buttons may select the
same installed English voice; the status line explains this.

## Prepare the object model

The project includes the matching `classes.json`; the ONNX model is omitted. To export it into app assets from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "ultralytics>=8.4.0" "onnx>=1.17" "onnxslim>=0.1"
python tools/export_onnx_model.py
```

The exporter downloads the YOLO-World v2 checkpoint, sets the 93 classes from
`classes.json`, and writes a fixed 640 × 640 ONNX model with embedded NMS. Check
the upstream model and Ultralytics license terms before publishing or
distributing this prototype.

## Build notes

The project uses Android Gradle Plugin 8.13.2, Kotlin 2.2.20, CameraX 1.6.2,
Compose BOM 2026.09.00 and ONNX Runtime Android 1.31.1. Minimum Android version
is API 26. An Android platform SDK is not configured in this workspace, so an
APK has not been built or launched on an emulator or phone. Open the project in
Android Studio to install the SDK and build it there.
