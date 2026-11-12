# Project overview

## Purpose

VisionGuide AI is a portfolio prototype exploring how on-device computer vision and speech might provide additional descriptions of nearby objects to blind and low-vision people. It is not a validated assistive product.

## Current scope

- Native iOS and Android camera applications with object detection and spoken output.
- A macOS/Python prototype containing camera, detection, segmentation-visualization, speech, and research scripts.
- The mobile apps are designed for local inference; model binaries are omitted from this repository pending redistribution review.
- English speech and object labels in the current application materials.

The desktop code and mobile apps are separate implementations. Segmentation is described in the Python research prototype; it should not be assumed to power speech or mobile navigation.

## Out of scope

- Guaranteed detection of obstacles, hazards, traffic signals, or signs.
- Route planning, autonomous navigation, or replacement of a white cane, guide dog, orientation and mobility training, or other established mobility tools.
- Medical, safety-certification, or general accuracy claims.

## Known limitations

Results can change with lighting, weather, viewpoint, camera characteristics, distance, occlusion, motion, and device workload. Object detections can be false or missed. Speech can be delayed or distracting. Any spoken distance/range is approximate and can be wrong.

## Intended audience

Developers, reviewers, and portfolio readers interested in mobile computer vision and accessible interaction. Any user feedback involving people should use a controlled, safe setting and an appropriate consent process; see [Safety and privacy](SAFETY_AND_PRIVACY.md).
