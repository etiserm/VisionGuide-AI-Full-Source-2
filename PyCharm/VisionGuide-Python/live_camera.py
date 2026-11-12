import cv2
import time
import threading
import subprocess
import asyncio
import tempfile
import os
import queue
import json
from pathlib import Path

from detect import (
    CONFIDENCE,
    IOU_THRESHOLD,
    VOICE_NAME,
    VOICE_RATE,
    VOICE_PITCH,
    get_device,
    load_detection_model,
    classify_yolo_result,
    draw_classified_detections,
    format_object,
    load_segmentation_model,
    run_segmentation,
    create_segmentation_visualization
)
from perception.object_classifier import detection_priority_score

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

TARGET_FPS = 15
YOLO_IMAGE_SIZE = 640

SEGMENTATION_INTERVAL = 30
SEGMENTATION_WIDTH = 320
SEGMENTATION_HEIGHT = 240

VOICE_COOLDOWN = 6
VOICE_MIN_INTERVAL = 3.0
VOICE_TRACK_CONFIRMATION_FRAMES = 3
VOICE_ZONE_CONFIRMATION_FRAMES = 4
VOICE_TRACK_MEMORY_FRAMES = 10

MODEL_MIN_CONFIDENCE = 0.15
MIN_CONFIDENCE_THRESHOLD = 0.20
MAX_CONFIDENCE_THRESHOLD = 0.50
CONFIDENCE_STEP = 0.05

UNCERTAINTY_MIN_CONFIDENCE = 0.18
UNCERTAINTY_MAX_CONFIDENCE = 0.30
UNCERTAINTY_CONFIRMATION_FRAMES = 18
UNCERTAINTY_COOLDOWN_SECONDS = 25
UNCERTAINTY_CENTER_MIN = 0.25
UNCERTAINTY_CENTER_MAX = 0.75
UNCERTAINTY_BOTTOM_MIN = 0.68
UNCERTAINTY_MIN_AREA = 0.004
UNCERTAINTY_LABELS = {
    "car", "truck", "bus", "motorcycle", "bicycle", "scooter",
    "stairs", "staircase", "pothole", "curb", "kerb",
    "construction barrier", "barrier", "traffic cone", "bollard",
    "pole", "manhole", "person", "pedestrian", "wheelchair", "dog",
}

SETTINGS_PATH = Path(__file__).with_name("visionguide_settings.json")

VOICE_POSITION_PHRASES = {
    "left": "on the left",
    "center": "near the center",
    "right": "on the right",
}

LIVE_WINDOW = "VisionGuide AI - Live"
SEGMENTATION_WINDOW = "VisionGuide AI - Segmentation"


class LiveSettings:
    def __init__(self):
        saved = {}

        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if not isinstance(saved, dict):
            saved = {}

        self.confidence_threshold = self._clamp_threshold(
            saved.get("confidence_threshold", CONFIDENCE)
        )
        self.uncertainty_alerts = bool(
            saved.get("uncertainty_alerts", True)
        )

    @staticmethod
    def _clamp_threshold(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = CONFIDENCE

        value = min(
            MAX_CONFIDENCE_THRESHOLD,
            max(MIN_CONFIDENCE_THRESHOLD, value),
        )
        return round(round(value / CONFIDENCE_STEP) * CONFIDENCE_STEP, 2)

    def adjust_confidence(self, delta):
        updated = self._clamp_threshold(
            self.confidence_threshold + delta
        )
        if updated == self.confidence_threshold:
            return False

        self.confidence_threshold = updated
        self.save()
        return True

    def toggle_uncertainty_alerts(self):
        self.uncertainty_alerts = not self.uncertainty_alerts
        self.save()

    def save(self):
        try:
            SETTINGS_PATH.write_text(
                json.dumps(
                    {
                        "confidence_threshold": self.confidence_threshold,
                        "uncertainty_alerts": self.uncertainty_alerts,
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            print(f"Unable to save settings: {error}")


class CameraStream:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(
            camera_index,
            cv2.CAP_AVFOUNDATION
        )

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError(
                "Unable to open camera. Check macOS camera permissions."
            )

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(
            target=self._update,
            daemon=True
        )
        self.thread.start()

    def _update(self):
        while self.running:
            success, frame = self.camera.read()

            if not success:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.camera.release()


class SegmentationWorker:
    def __init__(self, processor, model, device):
        self.processor = processor
        self.model = model
        self.device = device

        self.lock = threading.Lock()
        self.new_frame = threading.Event()

        self.frame = None
        self.result = None
        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self.thread.start()

    def submit(self, frame):
        small_frame = cv2.resize(
            frame,
            (SEGMENTATION_WIDTH, SEGMENTATION_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        with self.lock:
            self.frame = small_frame

        self.new_frame.set()

    def get_result(self):
        with self.lock:
            if self.result is None:
                return None

            return self.result.copy()

    def _run(self):
        while self.running:
            self.new_frame.wait(timeout=0.5)

            if not self.running:
                break

            self.new_frame.clear()

            with self.lock:
                if self.frame is None:
                    continue

                frame = self.frame.copy()

            try:
                segmentation = run_segmentation(
                    frame,
                    self.processor,
                    self.model,
                    self.device
                )

                visualization = create_segmentation_visualization(
                    frame,
                    segmentation,
                    self.model
                )

                with self.lock:
                    self.result = visualization

            except Exception as error:
                print(f"Segmentation error: {error}")

    def stop(self):
        self.running = False
        self.new_frame.set()
        self.thread.join(timeout=3)


class VoiceManager:
    def __init__(self):
        self.enabled = True
        self.voice = VOICE_NAME

        self.last_message = ""
        self.last_time = 0.0
        self.last_speech_start = 0.0

        self.speech_queue = queue.Queue(maxsize=1)
        self.running = True
        self.process = None
        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._speech_worker,
            daemon=True
        )
        self.thread.start()

        print(f"Neural voice: {self.voice}")

    def speak(self, message, force=False, interrupt=False):
        if not self.enabled or not message:
            return False

        now = time.time()

        with self.lock:
            if not force and (
                message == self.last_message
                and now - self.last_time < VOICE_COOLDOWN
            ):
                return False

            if (
                not force
                and now - self.last_speech_start < VOICE_MIN_INTERVAL
            ):
                return False

            if self.process is not None:
                if self.process.poll() is None:
                    if not interrupt:
                        return False
                    self.process.terminate()

            try:
                self.speech_queue.put_nowait(message)
            except queue.Full:
                if not interrupt:
                    return False

                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except queue.Empty:
                    return False

                try:
                    self.speech_queue.put_nowait(message)
                except queue.Full:
                    return False

            self.last_message = message
            self.last_time = now
            self.last_speech_start = now
            return True

    def _speech_worker(self):
        while self.running:
            try:
                message = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if message is None:
                break

            audio_path = None

            try:
                import edge_tts

                with tempfile.NamedTemporaryFile(
                    suffix=".mp3",
                    delete=False
                ) as audio_file:
                    audio_path = audio_file.name

                async def generate_audio():
                    communicate = edge_tts.Communicate(
                        message,
                        self.voice,
                        rate=VOICE_RATE,
                        pitch=VOICE_PITCH
                    )

                    await communicate.save(audio_path)

                asyncio.run(generate_audio())

                if not self.running:
                    break

                with self.lock:
                    self.process = subprocess.Popen(
                        ["afplay", audio_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    process = self.process

                process.wait()

                with self.lock:
                    if self.process is process:
                        self.process = None

            except Exception as error:
                print(f"Neural voice error: {error}")

            finally:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)

                self.speech_queue.task_done()

    def stop(self):
        self.running = False

        with self.lock:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()

        try:
            self.speech_queue.put_nowait(None)
        except queue.Full:
            pass

        self.thread.join(timeout=3)


class SceneEventNarrator:
    def __init__(self):
        self.frame_number = 0
        self.tracks = {}
        self.pending_events = {}

    @staticmethod
    def _track_key(detection):
        track_id = detection.get("track_id")

        if track_id is not None:
            return ("track", track_id)

        return (
            "fallback",
            detection.get("label", "unknown"),
            detection.get("position", "unknown"),
        )

    def _queue_event(self, kind, track_key, state, position):
        event_key = (kind, state["label"], position)
        event = self.pending_events.setdefault(
            event_key,
            {
                "kind": kind,
                "label": state["label"],
                "category": state["category"],
                "position": position,
                "priority": state["priority"],
                "track_keys": set(),
                "created_at": self.frame_number,
            },
        )
        event["priority"] = min(event["priority"], state["priority"])
        event["track_keys"].add(track_key)
        state["pending_events"].add(event_key)

    def update(self, detections):
        self.frame_number += 1
        seen_tracks = set()

        for detection in detections:
            position = detection.get("position")

            if position not in VOICE_POSITION_PHRASES:
                continue

            track_key = self._track_key(detection)
            seen_tracks.add(track_key)
            state = self.tracks.get(track_key)
            label = detection.get("label", "unknown")
            category = detection.get("category", "unknown")

            if state is None:
                self.tracks[track_key] = {
                    "label": label,
                    "category": category,
                    "priority": detection_priority_score(detection),
                    "position": None,
                    "candidate_position": position,
                    "candidate_frames": 1,
                    "missed_frames": 0,
                    "announced": False,
                    "pending_events": set(),
                }
                continue

            missed_frames = state["missed_frames"]
            state["missed_frames"] = 0
            state["label"] = label
            state["category"] = category
            state["priority"] = detection_priority_score(detection)

            if missed_frames:
                state["candidate_frames"] = 0
                state["candidate_position"] = position

            if not state["announced"]:
                if state["candidate_position"] == position:
                    state["candidate_frames"] += 1
                else:
                    state["candidate_position"] = position
                    state["candidate_frames"] = 1

                if state["candidate_frames"] >= VOICE_TRACK_CONFIRMATION_FRAMES:
                    state["announced"] = True
                    state["position"] = position
                    self._queue_event("new", track_key, state, position)
                continue

            if position == state["position"]:
                state["candidate_position"] = position
                state["candidate_frames"] = 0
                continue

            if state["candidate_position"] == position:
                state["candidate_frames"] += 1
            else:
                state["candidate_position"] = position
                state["candidate_frames"] = 1

            if state["candidate_frames"] >= VOICE_ZONE_CONFIRMATION_FRAMES:
                state["position"] = position
                state["candidate_frames"] = 0
                self._queue_event("moved", track_key, state, position)

        for track_key, state in list(self.tracks.items()):
            if track_key in seen_tracks:
                continue

            state["missed_frames"] += 1

            if state["missed_frames"] > VOICE_TRACK_MEMORY_FRAMES:
                self._forget_track(track_key, state)

    def _forget_track(self, track_key, state):
        for event_key in list(state["pending_events"]):
            event = self.pending_events.get(event_key)

            if event is None:
                continue

            event["track_keys"].discard(track_key)

            if not event["track_keys"]:
                self.pending_events.pop(event_key, None)

        self.tracks.pop(track_key, None)

    def next_event(self):
        if not self.pending_events:
            return None

        event_key, event = min(
            self.pending_events.items(),
            key=lambda item: (
                item[1]["priority"],
                0 if item[1]["kind"] == "new" else 1,
                item[1]["created_at"],
            ),
        )

        count = len(event["track_keys"])
        object_text = format_object(event["label"], count).capitalize()
        position_text = VOICE_POSITION_PHRASES[event["position"]]

        if event["kind"] == "new":
            verb = "has appeared" if count == 1 else "have appeared"
            message = f"{object_text} {verb} {position_text}."
        else:
            verb = "is" if count == 1 else "are"
            message = f"{object_text} {verb} now {position_text}."

        return event_key, message

    def consume_event(self, event_key):
        event = self.pending_events.pop(event_key, None)

        if event is None:
            return

        for track_key in event["track_keys"]:
            state = self.tracks.get(track_key)

            if state is not None:
                state["pending_events"].discard(event_key)


def describe_scene(detections, max_groups=4):
    if not detections:
        return "I can't clearly identify any objects in the current view."

    groups = {}

    for detection in detections:
        key = (detection.get("label", "object"), detection.get("position"))
        group = groups.setdefault(
            key,
            {
                "label": key[0],
                "position": key[1],
                "count": 0,
                "priority": detection_priority_score(detection),
            },
        )
        group["count"] += 1
        group["priority"] = min(
            group["priority"], detection_priority_score(detection)
        )

    ordered_groups = sorted(
        groups.values(),
        key=lambda group: (group["priority"], group["label"]),
    )[:max_groups]
    parts = []

    for group in ordered_groups:
        object_text = format_object(group["label"], group["count"])
        position = VOICE_POSITION_PHRASES.get(group["position"])
        parts.append(
            f"{object_text} {position}" if position else object_text
        )

    if len(parts) == 1:
        description = parts[0]
    else:
        description = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return f"Right now, I can see {description}."


class UncertaintyMonitor:
    def __init__(self):
        self.frame_number = 0
        self.candidates = {}
        self.last_alert_time = 0.0
        self.pending_alert = False
        self.pending_track_keys = set()

    @staticmethod
    def _track_key(detection):
        track_id = detection.get("track_id")
        if track_id is not None:
            return ("track", track_id)

        return (
            "fallback",
            detection.get("label", "unknown"),
            detection.get("position", "unknown"),
        )

    @staticmethod
    def _in_forward_zone(detection):
        center_x = detection.get("center_x_ratio")
        bottom = detection.get("bottom_ratio")

        return (
            center_x is not None
            and bottom is not None
            and UNCERTAINTY_CENTER_MIN <= center_x <= UNCERTAINTY_CENTER_MAX
            and bottom >= UNCERTAINTY_BOTTOM_MIN
        )

    def update(
        self,
        detections,
        confirmed_detections,
        confidence_threshold,
        enabled,
    ):
        self.frame_number += 1
        seen_tracks = set()

        for detection in detections:
            label = detection.get("label", "unknown")
            confidence = detection.get("confidence", 0.0)
            area = detection.get("area_ratio") or 0.0

            if (
                label not in UNCERTAINTY_LABELS
                or confidence < UNCERTAINTY_MIN_CONFIDENCE
                or confidence
                >= min(confidence_threshold, UNCERTAINTY_MAX_CONFIDENCE)
                or area < UNCERTAINTY_MIN_AREA
                or not self._in_forward_zone(detection)
            ):
                continue

            track_key = self._track_key(detection)
            seen_tracks.add(track_key)
            state = self.candidates.setdefault(
                track_key,
                {"frames": 0, "missed_frames": 0, "announced": False},
            )
            state["frames"] += 1
            state["missed_frames"] = 0

        for track_key, state in list(self.candidates.items()):
            if track_key in seen_tracks:
                continue

            state["missed_frames"] += 1
            if state["missed_frames"] > 2:
                self.candidates.pop(track_key, None)

        if not enabled:
            self.pending_alert = False
            self.pending_track_keys.clear()
            return None

        confirmed_object_in_path = any(
            detection.get("label") in UNCERTAINTY_LABELS
            and self._in_forward_zone(detection)
            for detection in confirmed_detections
        )

        if confirmed_object_in_path:
            self.pending_alert = False
            self.pending_track_keys.clear()
            return None

        if self.pending_alert:
            self.pending_track_keys = {
                track_key
                for track_key in self.pending_track_keys
                if track_key in self.candidates
                and self.candidates[track_key]["missed_frames"] <= 2
            }
            if not self.pending_track_keys:
                self.pending_alert = False
                return None
            return "I may have noticed something in your path, but I'm not certain."

        if time.time() - self.last_alert_time < UNCERTAINTY_COOLDOWN_SECONDS:
            return None

        stable_candidates = {
            track_key
            for track_key, state in self.candidates.items()
            if (
                state["frames"] >= UNCERTAINTY_CONFIRMATION_FRAMES
                and not state["announced"]
            )
        }

        if not stable_candidates:
            return None

        self.pending_alert = True
        self.pending_track_keys = stable_candidates
        return "I may have noticed something in your path, but I'm not certain."

    def acknowledge_alert(self):
        for track_key in self.pending_track_keys:
            state = self.candidates.get(track_key)
            if state is not None:
                state["announced"] = True

        self.last_alert_time = time.time()
        self.pending_alert = False
        self.pending_track_keys.clear()


def draw_status(
    frame,
    fps,
    yolo_ms,
    device,
    segmentation_enabled,
    voice_enabled,
    settings,
):
    output = frame.copy()

    lines = [
        f"FPS: {fps:.1f}",
        f"YOLO: {yolo_ms:.1f} ms",
        f"Device: {device}",
        f"Segmentation: {'ON' if segmentation_enabled else 'OFF'}",
        f"Neural voice: {'ON' if voice_enabled else 'OFF'}",
        f"Detection sensitivity: {settings.confidence_threshold:.2f}",
        f"Uncertain alerts: {'ON' if settings.uncertainty_alerts else 'OFF'}",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (15, 30 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2
        )

    cv2.putText(
        output,
        "Q quit | D describe | - more detections | = fewer | U uncertain | V voice | S segmentation",
        (15, output.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1
    )

    return output


def main():
    segmentation_enabled = True
    settings = LiveSettings()

    camera = None
    segmentation_worker = None
    voice = VoiceManager()

    device = get_device()

    print("\nVISIONGUIDE AI LIVE")
    print(f"Device: {device}")
    print(f"Detection sensitivity: {settings.confidence_threshold:.2f}")
    print(f"Uncertain alerts: {'ON' if settings.uncertainty_alerts else 'OFF'}")
    print("Loading YOLO-World...")

    yolo_model = load_detection_model()

    print("Loading segmentation model...")

    processor, segmentation_model = load_segmentation_model(
        device
    )

    segmentation_worker = SegmentationWorker(
        processor,
        segmentation_model,
        device
    )

    try:
        camera = CameraStream(CAMERA_INDEX)

        print("Camera started.")
        print("Q - quit")
        print("D - describe the current view once")
        print("- - more detections; = - fewer detections")
        print("U - toggle rare uncertain-object alerts")
        print("V - toggle voice")
        print("S - toggle segmentation")

        frame_count = 0
        narrator = SceneEventNarrator()
        uncertainty_monitor = UncertaintyMonitor()
        fps = 0.0
        fps_frames = 0
        fps_started = time.perf_counter()
        yolo_ms = 0.0

        while True:
            loop_started = time.perf_counter()

            frame = camera.read()

            if frame is None:
                continue

            frame_count += 1

            yolo_started = time.perf_counter()

            results = yolo_model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=MODEL_MIN_CONFIDENCE,
                iou=IOU_THRESHOLD,
                imgsz=YOLO_IMAGE_SIZE,
                device=str(device),
                verbose=False
            )

            yolo_ms = (
                time.perf_counter() - yolo_started
            ) * 1000

            result = results[0]
            model_detections = classify_yolo_result(
                result,
                yolo_model,
                image_width=frame.shape[1],
                image_height=frame.shape[0],
                confidence_threshold=MODEL_MIN_CONFIDENCE,
            )
            classified_detections = [
                detection
                for detection in model_detections
                if detection["confidence"] >= settings.confidence_threshold
            ]
            narrator.update(classified_detections)
            uncertainty_message = uncertainty_monitor.update(
                model_detections,
                classified_detections,
                settings.confidence_threshold,
                enabled=settings.uncertainty_alerts and voice.enabled,
            )
            annotated = draw_classified_detections(
                frame,
                classified_detections,
            )

            if (
                segmentation_enabled
                and frame_count % SEGMENTATION_INTERVAL == 0
            ):
                segmentation_worker.submit(frame)

            if segmentation_enabled:
                segmentation_image = segmentation_worker.get_result()

                if segmentation_image is not None:
                    cv2.imshow(
                        SEGMENTATION_WINDOW,
                        segmentation_image
                    )

            fps_frames += 1
            now = time.perf_counter()

            if now - fps_started >= 1.0:
                fps = fps_frames / (now - fps_started)
                fps_frames = 0
                fps_started = now

            display = draw_status(
                annotated,
                fps,
                yolo_ms,
                device,
                segmentation_enabled,
                voice.enabled,
                settings,
            )

            cv2.imshow(
                LIVE_WINDOW,
                display
            )

            pending_event = narrator.next_event()

            if pending_event is not None:
                event_key, message = pending_event

                if voice.speak(message):
                    narrator.consume_event(event_key)
            elif uncertainty_message is not None:
                if voice.speak(uncertainty_message, force=True):
                    uncertainty_monitor.acknowledge_alert()

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("d"):
                description = describe_scene(classified_detections)
                if voice.speak(description, force=True, interrupt=True):
                    print(f"Scene description: {description}")
                else:
                    print("Voice is unavailable or currently busy.")

            elif key == ord("-"):
                if settings.adjust_confidence(-CONFIDENCE_STEP):
                    print(
                        "Detection sensitivity: "
                        f"{settings.confidence_threshold:.2f} "
                        "(more detections)"
                    )

            elif key == ord("="):
                if settings.adjust_confidence(CONFIDENCE_STEP):
                    print(
                        "Detection sensitivity: "
                        f"{settings.confidence_threshold:.2f} "
                        "(fewer detections)"
                    )

            elif key == ord("u"):
                settings.toggle_uncertainty_alerts()
                print(
                    "Uncertain alerts: "
                    f"{'ON' if settings.uncertainty_alerts else 'OFF'}"
                )

            elif key == ord("v"):
                voice.enabled = not voice.enabled

                if not voice.enabled:
                    voice.stop()

                else:
                    voice = VoiceManager()

                print(
                    f"Voice: {'ON' if voice.enabled else 'OFF'}"
                )

            elif key == ord("s"):
                segmentation_enabled = not segmentation_enabled

                if not segmentation_enabled:
                    cv2.destroyWindow(
                        SEGMENTATION_WINDOW
                    )

                print(
                    f"Segmentation: "
                    f"{'ON' if segmentation_enabled else 'OFF'}"
                )

            elapsed = time.perf_counter() - loop_started
            delay = (1.0 / TARGET_FPS) - elapsed

            if delay > 0:
                time.sleep(delay)

    finally:
        voice.stop()

        if camera is not None:
            camera.stop()

        if segmentation_worker is not None:
            segmentation_worker.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
