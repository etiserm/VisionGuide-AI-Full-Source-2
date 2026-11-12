# VisionGuide for iPhone

This native iOS app keeps camera analysis and speech on the iPhone. It provides
a live camera preview, large accessible controls, on-demand scene descriptions,
prioritized spoken events, adjustable sensitivity, and rare uncertainty alerts.
The open-vocabulary detector includes both street-safety prompts and common
household items such as furniture, kitchenware, appliances, electronics and
bathroom objects.

## Refresh the Core ML model (optional)

The compiled model and its matching object-label list are already included in
this iOS project. If you edit `VisionGuide/Models/VisionGuideClasses.json`, the
model must also be exported again. The exporter reads that JSON file directly
and stages the replacement before swapping app resources, so a failed export
leaves the previous model in place.

From the project root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "ultralytics>=8.4.0" "coremltools>=9.0" "numpy<=2.3.5"
python ios/tools/export_coreml_model.py
```

The script downloads the exportable `yolov8s-worldv2.pt` checkpoint, saves the
editable `VisionGuide.mlpackage`, compiles it for iOS as `VisionGuide.mlmodelc`,
and updates the matching class-label list. The export keeps lower-confidence
candidates so the sensitivity control and rare uncertainty alerts can work.

## Run on an iPhone

Open `ios/VisionGuide/VisionGuide.xcodeproj` in Xcode, select the VisionGuide
target, choose a signing team, connect an iPhone, and press Run. Camera access
is requested by iOS the first time the app starts.

When the app opens, it speaks a short guide. The upper-left and upper-right
buttons raise and lower speech volume. The lower-left button selects the best
installed female English voice; the lower-right selects the best installed
male English voice. Tapping either plays a sample. These preferences are
saved on the iPhone.

Tap the large center area to start the camera. Once it is running, tap the same
area to hear a description of the current view. **Repeat guide** replays the
instructions. **More settings** contains the analysis pause control, detection
sensitivity, and the optional rare uncertain-object alerts.

## Make the English voice more natural

`SpeechController` offers exactly two choices: the best installed female
English voice and the best installed male English voice. It prefers Premium,
then Enhanced, then Standard quality; Ava and Evan are preferred when their
quality matches other voices. Tapping either lower corner plays a short sample
in that voice. The selected voice is remembered on the iPhone.

For the most natural sound, download an Enhanced or Premium English voice for each gender in
**Settings → Accessibility → Read & Speak → Voices → English**. The app shows
the active voice and its quality at the bottom of the screen. On older iOS
versions, the menu may be called **Spoken Content**.

Speech uses iOS playback audio so it can be heard when the ringer/silent switch
is on. The app ducks other audio while speaking, then releases the audio session.
The status line under the camera status shows whether a voice is starting,
speaking, muted or unavailable.

The camera preview runs at the low VGA resolution. The model analyzes up to
four frames per second and drops late frames rather than building up stale work.
The live object list stays internal to the camera flow; SwiftUI is notified only
when the displayed object count changes. These changes reduce repeated model
work and screen updates. Core ML loads when the camera starts and is released
when the camera stops, so the idle screen does not retain the model in memory.
The first camera start can take a little longer. Actual battery savings depend
on the iPhone and usage. The app does not save or upload camera frames.

For supported object classes, descriptions may include a broad step range. It is
estimated from the back camera's horizontal field of view and a typical real-
world object height. Small, clipped, uncertain, or unsupported detections have
no distance estimate. Object sizes and walking step lengths vary, so use this
as an extra cue rather than a measurement.

## Current scope

The iPhone build starts with live object detection and speech. Descriptions of
all detected classes, including household items, are spoken when requested.
Automatic speech is reserved for stable priority classes so ordinary furniture
and home items do not trigger constant announcements.
The desktop Mask2Former window is a visualization in `live_camera.py`; it does
not drive that script's spoken object events. Segmentation can be considered
after the Core ML detector is profiled on the target iPhone.

The exported model uses the Ultralytics YOLO-World v2 checkpoint and its
license should be reviewed before distributing the app.
