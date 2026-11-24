"""
Vision AI Module — Enhanced with Visiona-inspired spatial intelligence.

Features:
  - YOLOv11 object detection (lazy-loaded)
  - Known-size distance estimation (meters + steps)
  - Spatial positioning (left/ahead/right)
  - Scene change detection (only narrate when scene changes)
  - Smart grouping ("3 chairs on your left")
  - Priority scoring (closest/most dangerous first)
  - Emotion recognition (DeepFace)
  - Scene summary generation for TTS narration
"""

import logging
import cv2
import numpy as np
import base64
import time
from collections import Counter, OrderedDict
from ultralytics import YOLO

logger = logging.getLogger("assistive-backend")

# Lazy-load DeepFace to speed up startup
_deepface = None

def get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


# Lazy-load YOLO model (only on first frame, not at import time)
_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        logger.info("Loading YOLO11m model...")
        _yolo_model = YOLO("yolo11m.pt")
        logger.info("YOLO11m model loaded successfully")
    return _yolo_model


# ─── Known real-world physical dimensions (meters) for calibrated distance estimation ───
# Calibrated for monocular perspective geometry (width + height dual-axis)
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
    "door":          {"w": 0.85, "h": 2.05, "mode": "h"},
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

# Simple irregular plural mappings for COCO class names
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


# ─── LRU Distance History (per-track EMA smoothing) ─────────────────────────
_MAX_TRACK_HISTORY = 150

class _LRUDistanceHistory:
    """LRU cache for distance smoothing. Evicts oldest entries instead of clearing all."""
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
            self._data.popitem(last=False)  # evict oldest

_track_distance_history = _LRUDistanceHistory()

# Scene memory for change detection
_last_scene_key = ""
_last_scene_time = 0.0


def decode_frame(base64_data: str) -> np.ndarray:
    """Decode a base64-encoded image string into an OpenCV frame."""
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]
    try:
        img_bytes = base64.b64decode(base64_data)
    except Exception:
        return None
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame


def estimate_distance(label: str, box_w: float, box_h: float, frame_w: float, frame_h: float, track_id: int = -1) -> dict:
    """
    Calibrated monocular distance estimation:
    - Automatically scales focal length with frame resolution (FOV ~55°)
    - Uses dual-axis (width + height) to handle arbitrary camera angles and perspectives
    - Automatically detects standing vs sitting / torso for people
    - Smooths distance across frames for tracked objects via LRU EMA
    """
    focal_px = frame_h * 0.98  # Standard phone camera focal length in pixels

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
            # Blend height & width
            raw_dist = (dist_h * 0.6) + (dist_w * 0.4)
        elif mode == "w":
            raw_dist = (w_real * focal_px) / box_w
        elif mode == "h":
            raw_dist = (h_real * focal_px) / box_h
        else:  # avg
            dist_w = (w_real * focal_px) / box_w
            dist_h = (h_real * focal_px) / box_h
            raw_dist = (dist_w + dist_h) / 2.0
    else:
        # Dynamic fallback: relative bounding box size against screen dimensions
        box_max = max(box_w, box_h)
        f_max = max(frame_w, frame_h)
        ratio = box_max / f_max if f_max > 0 else 0.1
        raw_dist = 0.55 / max(ratio, 0.04)

    raw_dist = max(0.2, min(raw_dist, 15.0))

    # Apply temporal exponential moving average (EMA) smoothing for tracked objects
    if track_id != -1:
        prev = _track_distance_history.get(track_id)
        if prev is not None:
            raw_dist = (0.7 * raw_dist) + (0.3 * prev)
        _track_distance_history.set(track_id, raw_dist)

    steps = max(1, round(raw_dist / 0.75))
    return {
        "distance_m": round(raw_dist, 1),
        "distance_steps": steps,
    }


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
        return "critical"   # Very close — collision hazard
    elif distance_m <= 2.5:
        return "warning"    # Getting close
    elif distance_m <= 5.0:
        return "info"       # Nearby — awareness
    else:
        return "background" # Far — low priority


def detect_objects(frame: np.ndarray, conf_threshold: float = 0.45) -> list:
    """
    Run YOLO11 object detection with tracking + calibrated spatial awareness.
    Tracking assigns consistent IDs to objects across frames for smoother narration.
    """
    if frame is None:
        return []

    h, w = frame.shape[:2]
    model = get_yolo()
    results = model.track(frame, verbose=False, conf=conf_threshold, persist=True, tracker="bytetrack.yaml")
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]
            track_id = int(box.id[0]) if box.id is not None else -1

            box_center_x = (x1 + x2) / 2
            box_width = max(1.0, x2 - x1)
            box_height = max(1.0, y2 - y1)

            dist = estimate_distance(label, box_width, box_height, w, h, track_id)
            position = get_position(box_center_x, w)
            urgency = get_urgency(dist["distance_m"])

            detections.append({
                "label": label,
                "confidence": round(confidence, 2),
                "track_id": track_id,
                "box": {
                    "x1": round(x1), "y1": round(y1),
                    "x2": round(x2), "y2": round(y2),
                },
                "position": position,
                "distance_m": dist["distance_m"],
                "distance_steps": dist["distance_steps"],
                "urgency": urgency,
            })

    # Sort by distance (closest first)
    detections.sort(key=lambda d: d["distance_steps"])
    return detections


def build_scene_narration(objects: list, emotions: list) -> dict:
    """
    Build an intelligent, crisp scene narration for TTS.
    Prioritizes the single closest/most critical obstacle in the walking path.
    Prevents verbal clutter and cognitive overload.
    Returns: {narration: str, scene_key: str, changed: bool, is_critical_hazard: bool, closest_distance_m: float}
    """
    global _last_scene_key, _last_scene_time

    if not objects and not emotions:
        return {
            "narration": "",
            "scene_key": "",
            "changed": False,
            "is_critical_hazard": False,
            "closest_distance_m": None
        }

    # Group objects by label+position to avoid repetition
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
        # Keep track of the closest one
        if obj["distance_steps"] < groups[key]["closest_steps"]:
            groups[key]["closest_steps"] = obj["distance_steps"]
            groups[key]["closest_m"] = obj["distance_m"]
            groups[key]["urgency"] = obj["urgency"]

    # Sort groups by closest metric distance
    sorted_groups = sorted(groups.values(), key=lambda g: g["closest_m"])

    # Find the single most urgent hazard in the path:
    # 1. Any obstacle <= 1.2m (critical collision hazard)
    # 2. Obstacle directly ahead <= 2.2m (warning)
    # 3. Obstacle on sides <= 1.5m
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
        elif hazard["urgency"] == "warning" or m <= 2.2:
            narration = f"{name} {m} meters {pos}."
        else:
            narration = f"{name} {m} meters {pos}."

    # Scene change detection with throttle
    now = time.time()
    scene_key = f"{hazard['label']}:{round(hazard['closest_m'], 1)}:{hazard['position']}" if hazard else "clear"
    time_threshold = 3.0 if is_critical_hazard else 6.0
    changed = (scene_key != _last_scene_key) or (now - _last_scene_time > time_threshold)

    if changed:
        _last_scene_key = scene_key
        _last_scene_time = now

    closest_m = min((o["distance_m"] for o in objects), default=None)

    return {
        "narration": narration,
        "scene_key": scene_key,
        "changed": changed,
        "is_critical_hazard": is_critical_hazard,
        "closest_distance_m": closest_m,
    }


# Emotion throttle state
_last_emotion_time = 0.0
_cached_emotions = []

def detect_emotions(frame: np.ndarray, has_person: bool = True) -> list:
    """
    Detect faces and emotions using DeepFace.
    Optimized: only runs if a person is present, throttled to once every 3s.
    """
    global _last_emotion_time, _cached_emotions

    if not has_person:
        return []

    now = time.time()
    if now - _last_emotion_time < 3.0:
        return _cached_emotions

    _last_emotion_time = now

    try:
        DeepFace = get_deepface()
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=True,
            silent=True,
            detector_backend="ssd",
        )

        emotions = []
        for face in results:
            dominant = face.get("dominant_emotion", "unknown")
            confidence = face.get("emotion", {}).get(dominant, 0)
            region = face.get("region", {})

            emotions.append({
                "emotion": dominant,
                "confidence": round(confidence, 1),
                "box": {
                    "x": region.get("x", 0),
                    "y": region.get("y", 0),
                    "w": region.get("w", 0),
                    "h": region.get("h", 0),
                }
            })

        _cached_emotions = emotions
        return emotions
    except Exception:
        # Silently fail if face is not clear or backend dependencies are missing
        return _cached_emotions


def process_frame(base64_data: str) -> dict:
    """
    Main processing pipeline:
    1. Decode the base64 frame
    2. Run object detection with spatial awareness
    3. Run emotion recognition
    4. Build scene narration
    5. Return everything as JSON
    """
    start_t = time.time()

    frame = decode_frame(base64_data)
    if frame is None:
        return {"error": "Failed to decode frame"}

    h, w = frame.shape[:2]

    objects = detect_objects(frame)
    has_person = any(o["label"] == "person" for o in objects)
    emotions = detect_emotions(frame, has_person)
    narration = build_scene_narration(objects, emotions)

    elapsed = round((time.time() - start_t) * 1000)
    obj_summary = ", ".join(f"{o['label']}({o['distance_steps']}steps)" for o in objects[:3])
    logger.info("Frame %dms | %d objects [%s] | %d faces | changed=%s",
                elapsed, len(objects), obj_summary, len(emotions), narration['changed'])

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
