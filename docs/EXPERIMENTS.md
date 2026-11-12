# Experiments and evidence

## What is present

The repository contains `PyCharm/VisionGuide-Python/research/results/baseline_experiment.csv`, an exploratory artifact rather than a representative benchmark or validated dataset. Demonstration images and generated visualizations were omitted pending sharing-rights review.

## What is not established

The repository does not establish general precision/recall, obstacle-detection coverage, end-to-end latency, comparative FPS across devices, usability, or safety. A small number of demonstration scenes cannot support such claims. Do not quote a best observed frame rate as general performance.

## How to report future measurements

For reproducible measurements, record the device and OS, code revision, model and version, input resolution, confidence threshold, lighting, distance, scene density, camera motion, warm-up, run duration, and sample count. Report median and distribution (for example, p5–p95), not only the best value. Measure capture-to-speech latency separately from inference and TTS. Calculate precision/recall only against a clearly described, labeled dataset and explicit classes.

Usability feedback should report the procedure and participant count without exposing identifying information. Follow [Safety and privacy](SAFETY_AND_PRIVACY.md); do not ask participants to rely on VisionGuide to avoid hazards.
