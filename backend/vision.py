"""
Vision AI Module — Real-time spatial object detection, tracking, distance calibration & emotion recognition.

Features:
  - YOLOv11 object detection (thread-safe lazy load + per-session ByteTrack isolation)
  - Orientation-invariant monocular distance estimation (meters + steps + EMA smoothing)
  - Spatial positioning (left / ahead / right) & priority scoring
  - Scene change detection (only narrate when scene meaningfully changes)
  - Emotion recognition (DeepFace SSD with clean cache eviction & non-silent error logging)
  - Thread-safe per-client VisionSession preventing cross-client contamination
"""

import base64
import logging
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("assistive-backend")

_deepface = None
_deepface_load_lock = threading.Lock()
_deepface_infer_lock = threading.Lock()


def get_deepface():
    """Thread-safe lazy import of DeepFace."""
    global _deepface
    if _deepface is not None:
        return _deepface
    with _deepface_load_lock:
        if _deepface is None:
            from deepface import DeepFace
            _deepface = DeepFace
    return _deepface


_yolo_model = None
_yolo_load_lock = threading.Lock()
_yolo_infer_lock = threading.Lock()


def get_yolo():
    """Thread-safe lazy load of YOLO11m."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    with _yolo_load_lock:
        if _yolo_model is None:
            logger.info("Loading YOLO11m model...")
            _yolo_model = YOLO("yolo11m.pt")
            logger.info("YOLO11m model loaded successfully")
    return _yolo_model


def _create_byte_tracker():
    """Create an isolated per-session ByteTrack instance."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    return BYTETracker(args=cfg)


# ─── Known real-world physical dimensions (meters) for COCO classes ──────────
REAL_OBJECT_METRICS = {
    # People & Living
    "person":        {"w": 0.45, "h": 1.70, "mode": "person"},
    "dog":           {"w": 0.30, "h": 0.55, "mode": "h"},
    "cat":           {"w": 0.20, "h": 0.30, "mode": "h"},
    "horse":         {"w": 0.60, "h": 1.60, "mode": "h"},
    "sheep":         {"w": 0.40, "h": 0.70, "mode": "h"},
    "cow":           {"w": 0.70, "h": 1.50, "mode": "h"},
    "bird":          {"w": 0.15, "h": 0.20, "mode": "h"},

    # Vehicles & Transportation
    "car":           {"w": 1.80, "h": 1.50, "mode": "w"},
    "bicycle":       {"w": 0.60, "h": 1.05, "mode": "h"},
    "motorcycle":    {"w": 0.80, "h": 1.10, "mode": "h"},
    "bus":           {"w": 2.50, "h": 3.20, "mode": "w"},
    "truck":         {"w": 2.50, "h": 3.50, "mode": "w"},
    "train":         {"w": 3.00, "h": 3.80, "mode": "w"},
    "boat":          {"w": 2.00, "h": 1.50, "mode": "w"},
    "airplane":      {"w": 8.00, "h": 3.00, "mode": "w"},

    # Outdoor & Street Furniture
    "traffic light": {"w": 0.35, "h": 0.85, "mode": "h"},
    "fire hydrant":  {"w": 0.35, "h": 0.65, "mode": "h"},
    "stop sign":     {"w": 0.75, "h": 0.75, "mode": "w"},
    "parking meter": {"w": 0.30, "h": 1.40, "mode": "h"},
    "bench":         {"w": 1.40, "h": 0.80, "mode": "w"},

    # Indoor Furniture & Obstacles
    "chair":         {"w": 0.50, "h": 0.85, "mode": "avg"},
    "couch":         {"w": 1.80, "h": 0.85, "mode": "w"},
    "dining table":  {"w": 1.20, "h": 0.75, "mode": "w"},
    "bed":           {"w": 1.50, "h": 0.70, "mode": "w"},
    "toilet":        {"w": 0.45, "h": 0.75, "mode": "h"},
    "potted plant":  {"w": 0.30, "h": 0.50, "mode": "h"},

    # Electronics & Office
    "tv":            {"w": 0.90, "h": 0.55, "mode": "w"},
    "laptop":        {"w": 0.33, "h": 0.23, "mode": "w"},
    "mouse":         {"w": 0.065, "h": 0.11, "mode": "w"},
    "remote":        {"w": 0.05, "h": 0.18, "mode": "h"},
    "keyboard":      {"w": 0.44, "h": 0.14, "mode": "w"},
    "cell phone":    {"w": 0.075, "h": 0.15, "mode": "h"},
    "book":          {"w": 0.16, "h": 0.24, "mode": "h"},
    "clock":         {"w": 0.30, "h": 0.30, "mode": "avg"},

    # Kitchen & Household
    "bottle":        {"w": 0.075, "h": 0.25, "mode": "h"},
    "wine glass":    {"w": 0.08, "h": 0.20, "mode": "h"},
    "cup":           {"w": 0.085, "h": 0.10, "mode": "w"},
    "fork":          {"w": 0.03, "h": 0.18, "mode": "h"},
    "knife":         {"w": 0.03, "h": 0.22, "mode": "h"},
    "spoon":         {"w": 0.04, "h": 0.18, "mode": "h"},
    "bowl":          {"w": 0.16, "h": 0.08, "mode": "w"},
    "microwave":     {"w": 0.50, "h": 0.30, "mode": "w"},
    "oven":          {"w": 0.75, "h": 0.85, "mode": "avg"},
    "toaster":       {"w": 0.28, "h": 0.20, "mode": "w"},
    "sink":          {"w": 0.55, "h": 0.40, "mode": "w"},
    "refrigerator":  {"w": 0.75, "h": 1.75, "mode": "h"},
    "vase":          {"w": 0.18, "h": 0.35, "mode": "h"},

    # Bags & Personal Items
    "backpack":      {"w": 0.35, "h": 0.45, "mode": "h"},
    "umbrella":      {"w": 0.90, "h": 0.80, "mode": "w"},
    "handbag":       {"w": 0.30, "h": 0.25, "mode": "w"},
    "tie":           {"w": 0.08, "h": 0.45, "mode": "h"},
    "suitcase":      {"w": 0.45, "h": 0.65, "mode": "h"},
}

_IRREGULAR_PLURALS = {
    "person": "people",
    "bus": "buses",
    "knife": "knives",
    "mouse": "mice",
    "sheep": "sheep",
}


def _pluralize(label: str, count: int) -> str:
    """Grammatically correct pluralization for COCO class labels."""
    if count <= 1:
        return label
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    return label + "s"


_MAX_TRACK_HISTORY = 150


class _LRUDistanceHistory:
    """LRU cache for distance smoothing. Evicts oldest entries cleanly."""

    def __init__(self, maxsize=_MAX_TRACK_HISTORY):
        self._data = OrderedDict()
        self._maxsize = maxsize

    def get(self, track_id):
        if track_id in self._data:
            self._data.move_to_end(track_id)
            return self._data[track_id]
        return None

    def set(self, track_id, value):
        if track_id in self._data:
            self._data.move_to_end(track_id)
        self._data[track_id] = value
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)


def decode_frame(base64_data: str) -> np.ndarray | None:
    """Decode a base64-encoded image string into an OpenCV BGR frame safely."""
    if not base64_data or not isinstance(base64_data, str):
        return None
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]
    base64_data = base64_data.strip()
    if not base64_data:
        return None

    try:
        img_bytes = base64.b64decode(base64_data, validate=False)
    except Exception:
        return None

    if not img_bytes:
        return None

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    if img_array.size == 0:
        return None

    try:
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except cv2.error:
        return None

    return frame


def get_position(box_center_x: float, frame_width: float) -> str:
    """Determine if object is on the left, center, or right of the frame."""
    ratio = box_center_x / frame_width if frame_width > 0 else 0.5
    if ratio < 0.33:
        return "on your left"
    elif ratio < 0.66:
        return "ahead"
    else:
        return "on your right"


def get_urgency(distance_m: float) -> str:
    """Classify urgency based on actual metric distance."""
    if distance_m <= 1.2:
        return "critical"
    elif distance_m <= 2.5:
        return "warning"
    elif distance_m <= 5.0:
        return "info"
    else:
        return "background"


def estimate_distance(
    label: str,
    box_w: float,
    box_h: float,
    frame_w: float,
    frame_h: float,
    track_id: int = -1,
    track_history: _LRUDistanceHistory | None = None,
) -> dict:
    """
    Orientation-invariant calibrated monocular distance estimation:
    - Uses max(frame_w, frame_h) so portrait (360x640) and landscape (640x360) keep equal focal length (~589px).
    - Uses dual-axis (width + height) for perspective stability.
    - Smooths per-track distance via LRU EMA when track_history is provided.
    """
    focal_px = max(frame_w, frame_h) * 0.92 if max(frame_w, frame_h) > 0 else 588.8

    dim = REAL_OBJECT_METRICS.get(label)
    if dim and box_w > 4 and box_h > 4:
        mode = dim["mode"]
        w_real = dim["w"]
        h_real = dim["h"]

        if mode == "person":
            aspect = box_h / max(box_w, 1.0)
            if aspect < 2.0:
                h_real = 0.88  # Upper body / seated person
            dist_h = (h_real * focal_px) / box_h
            dist_w = (w_real * focal_px) / box_w
            raw_dist = (dist_h * 0.6) + (dist_w * 0.4)
        elif mode == "w":
            raw_dist = (w_real * focal_px) / box_w
        elif mode == "h":
            raw_dist = (h_real * focal_px) / box_h
        else:
            dist_w = (w_real * focal_px) / box_w
            dist_h = (h_real * focal_px) / box_h
            raw_dist = (dist_w + dist_h) / 2.0
    else:
        box_max = max(box_w, box_h)
        f_max = max(frame_w, frame_h)
        ratio = box_max / f_max if f_max > 0 else 0.1
        raw_dist = 0.55 / max(ratio, 0.04)

    raw_dist = max(0.2, min(raw_dist, 15.0))

    if track_id != -1 and track_history is not None:
        prev = track_history.get(track_id)
        if prev is not None:
            raw_dist = (0.7 * raw_dist) + (0.3 * prev)
        track_history.set(track_id, raw_dist)

    steps = max(1, round(raw_dist / 0.75))
    return {
        "distance_m": round(raw_dist, 1),
        "distance_steps": steps,
    }


class VisionSession:
    """
    Per-WebSocket isolated computer vision pipeline.
    Prevents ByteTrack IDs, EMA smoothing history, scene throttling, and emotion cache
    from leaking across simultaneous clients.
    """

    def __init__(self):
        self._tracker = None
        self._track_history = _LRUDistanceHistory()
        self._last_scene_key = ""
        self._last_scene_time = 0.0
        self._last_emotion_time = 0.0
        self._cached_emotions: list = []

    def _get_tracker(self):
        if self._tracker is None:
            self._tracker = _create_byte_tracker()
        return self._tracker

    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.45) -> list:
        """Run YOLO11 prediction + isolated per-session ByteTrack tracking."""
        if frame is None:
            return []

        h, w = frame.shape[:2]
        model = get_yolo()

        with _yolo_infer_lock:
            results = model.predict(frame, verbose=False, conf=conf_threshold)

        tracker = self._get_tracker()
        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                # Advance empty frame in tracker so stale tracks age out properly
                try:
                    empty_boxes = result.boxes.cpu().numpy()
                    tracker.update(empty_boxes, frame)
                except Exception:
                    pass
                continue

            det_np = result.boxes.cpu().numpy()
            tracks = tracker.update(det_np, frame)

            if tracks is not None and len(tracks) > 0:
                for row in tracks:
                    # STrack.result: [x1, y1, x2, y2, track_id, score, cls, idx]
                    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    track_id = int(row[4])
                    confidence = float(row[5])
                    class_id = int(row[6])
                    label = model.names[class_id]

                    box_center_x = (x1 + x2) / 2
                    box_width = max(1.0, x2 - x1)
                    box_height = max(1.0, y2 - y1)

                    dist = estimate_distance(
                        label, box_width, box_height, w, h, track_id, self._track_history
                    )
                    detections.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "track_id": track_id,
                        "box": {
                            "x1": round(x1), "y1": round(y1),
                            "x2": round(x2), "y2": round(y2),
                        },
                        "position": get_position(box_center_x, w),
                        "distance_m": dist["distance_m"],
                        "distance_steps": dist["distance_steps"],
                        "urgency": get_urgency(dist["distance_m"]),
                    })
            else:
                # Fallback to immediate raw detections on frame 1 or unconfirmed sudden obstacles
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = model.names[class_id]

                    box_center_x = (x1 + x2) / 2
                    box_width = max(1.0, x2 - x1)
                    box_height = max(1.0, y2 - y1)

                    dist = estimate_distance(
                        label, box_width, box_height, w, h, -1, self._track_history
                    )
                    detections.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "track_id": -1,
                        "box": {
                            "x1": round(x1), "y1": round(y1),
                            "x2": round(x2), "y2": round(y2),
                        },
                        "position": get_position(box_center_x, w),
                        "distance_m": dist["distance_m"],
                        "distance_steps": dist["distance_steps"],
                        "urgency": get_urgency(dist["distance_m"]),
                    })

        detections.sort(key=lambda d: d["distance_steps"])
        return detections

    def detect_emotions(self, frame: np.ndarray, has_person: bool = True) -> list:
        """
        Detect faces and emotions using DeepFace (SSD backend).
        Clears cached emotions when no person/face is present and logs unexpected failures.
        """
        if frame is None or not has_person:
            self._cached_emotions = []
            return []

        now = time.time()
        if now - self._last_emotion_time < 3.0:
            return self._cached_emotions

        self._last_emotion_time = now

        try:
            DeepFace = get_deepface()
            with _deepface_infer_lock:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=True,
                    silent=True,
                    detector_backend="opencv",
                )

            emotions = []
            for face in results:
                dominant = face.get("dominant_emotion", "unknown")
                confidence = face.get("emotion", {}).get(dominant, 0)
                region = face.get("region", {})

                emotions.append({
                    "emotion": dominant,
                    "confidence": round(float(confidence), 1),
                    "box": {
                        "x": int(region.get("x", 0)),
                        "y": int(region.get("y", 0)),
                        "w": int(region.get("w", 0)),
                        "h": int(region.get("h", 0)),
                    },
                })

            self._cached_emotions = emotions
            return emotions
        except ValueError as e:
            # Expected when person is facing away or face is occluded
            if "face could not be detected" in str(e).lower():
                self._cached_emotions = []
                return []
            logger.warning("DeepFace emotion value error: %s", e)
            self._cached_emotions = []
            return []
        except Exception as e:
            logger.warning("DeepFace emotion detection error: %s", e)
            self._cached_emotions = []
            return []

    def build_scene_narration(self, objects: list, emotions: list) -> dict:
        """Build crisp TTS hazard narration & scene-change metadata."""
        if not objects and not emotions:
            self._last_scene_key = "clear"
            return {
                "narration": "",
                "scene_key": "clear",
                "changed": False,
                "is_critical_hazard": False,
                "closest_distance_m": None,
            }

        groups = {}
        for obj in objects:
            key = f"{obj['label']}_{obj['position']}"
            if key not in groups:
                groups[key] = {
                    "label": obj["label"],
                    "position": obj["position"],
                    "closest_steps": obj["distance_steps"],
                    "closest_m": obj["distance_m"],
                    "urgency": obj["urgency"],
                    "count": 0,
                }
            groups[key]["count"] += 1
            if obj["distance_steps"] < groups[key]["closest_steps"]:
                groups[key]["closest_steps"] = obj["distance_steps"]
                groups[key]["closest_m"] = obj["distance_m"]
                groups[key]["urgency"] = obj["urgency"]

        sorted_groups = sorted(groups.values(), key=lambda g: g["closest_m"])

        hazard = None
        for g in sorted_groups:
            if g["urgency"] == "critical" or g["closest_m"] <= 1.2:
                hazard = g
                break
            elif g["position"] == "ahead" and g["closest_m"] <= 2.2:
                hazard = g
                break
            elif g["closest_m"] <= 1.5:
                hazard = g
                break

        is_critical_hazard = False
        narration = ""

        if hazard:
            label = hazard["label"]
            count = hazard["count"]
            pos = hazard["position"]
            m = hazard["closest_m"]
            name = f"{count} {_pluralize(label, count)}" if count > 1 else label

            if hazard["urgency"] == "critical" or m <= 1.2:
                is_critical_hazard = True
                narration = f"Watch out! {name} {m} meters {pos}."
            else:
                narration = f"{name} {m} meters {pos}."

        now = time.time()
        scene_key = (
            f"{hazard['label']}:{round(hazard['closest_m'], 1)}:{hazard['position']}"
            if hazard
            else "clear"
        )
        time_threshold = 3.0 if is_critical_hazard else 6.0
        changed = bool(narration) and (
            (scene_key != self._last_scene_key) or (now - self._last_scene_time > time_threshold)
        )

        if changed or scene_key == "clear":
            self._last_scene_key = scene_key
            if changed:
                self._last_scene_time = now

        closest_m = min((o["distance_m"] for o in objects), default=None)

        return {
            "narration": narration,
            "scene_key": scene_key,
            "changed": changed,
            "is_critical_hazard": is_critical_hazard,
            "closest_distance_m": closest_m,
        }

    def process_frame(self, base64_data: str) -> dict:
        """Main processing pipeline for a single WebSocket vision frame."""
        start_t = time.time()

        frame = decode_frame(base64_data)
        if frame is None:
            return {"error": "Failed to decode frame"}

        h, w = frame.shape[:2]

        objects = self.detect_objects(frame)
        has_person = any(o["label"] == "person" for o in objects)
        emotions = self.detect_emotions(frame, has_person)
        narration = self.build_scene_narration(objects, emotions)

        elapsed = round((time.time() - start_t) * 1000)
        obj_summary = ", ".join(f"{o['label']}({o['distance_steps']}steps)" for o in objects[:3])
        logger.info(
            "Frame %dms | %d objects [%s] | %d faces | changed=%s",
            elapsed, len(objects), obj_summary, len(emotions), narration["changed"],
        )

        return {
            "frame_size": {"width": w, "height": h},
            "objects": objects,
            "emotions": emotions,
            "narration": narration["narration"],
            "scene_changed": narration["changed"],
            "is_critical_hazard": narration["is_critical_hazard"],
            "closest_distance_m": narration["closest_distance_m"],
            "object_count": len(objects),
            "face_count": len(emotions),
        }


# Default session for standalone calls / tests
_default_vision_session = VisionSession()


def detect_objects(frame: np.ndarray, conf_threshold: float = 0.45) -> list:
    return _default_vision_session.detect_objects(frame, conf_threshold)


def detect_emotions(frame: np.ndarray, has_person: bool = True) -> list:
    return _default_vision_session.detect_emotions(frame, has_person)


def build_scene_narration(objects: list, emotions: list) -> dict:
    return _default_vision_session.build_scene_narration(objects, emotions)


def process_frame(base64_data: str) -> dict:
    return _default_vision_session.process_frame(base64_data)
