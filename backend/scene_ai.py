"""
Scene AI — Gemini Vision API for intelligent scene understanding.

Uses google.genai SDK with gemini-3.5-flash-lite primary + gemini-3.8-flash fallback.
Sends camera frames + YOLO detection context at an adaptive rate-limited interval.
Isolates per-client timing/descriptions via SceneSession while sharing rate-limit backoff cleanly.
"""

import os
import time
import base64
import logging
import threading
from typing import Optional, List

logger = logging.getLogger("assistive-backend")

_client = None
_client_lock = threading.Lock()
_initialized = False

MIN_INTERVAL = 3.5
MAX_INTERVAL = 35.0

# Shared API-key backoff state across sessions
_api_lock = threading.Lock()
_shared_interval = MIN_INTERVAL
_rate_limited_until = 0.0

SCENE_PROMPT = """You are a real-time live navigation guide for a visually impaired person walking forward with their phone.

DETECTED OBJECTS:
{detections}

Give an extremely concise directional walking cue in STRICTLY UNDER 10 WORDS.
Rules:
1. State walking path status first: e.g. "Path clear ahead", "Obstacle ahead, steer right", or "Clear path to left".
2. If path is clear, state the main opening or target (e.g. "Doorway 3 meters straight", "Hallway open ahead").
3. NEVER exceed 10 words. Every extra word delays and confuses a walking user.
4. Do NOT say "I see", "There is", or "In this image". Speak direct and actionable navigation cues.

Example: "Path clear ahead. Doorway 3 meters straight."
Example: "Obstacle ahead. Clear path to your right."
Example: "Path open. Person approaching 3 meters ahead." """


def _get_client():
    """Thread-safe lazy initialization of the Gemini client."""
    global _client, _initialized
    if _initialized:
        return _client

    with _client_lock:
        if _initialized:
            return _client

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY", "")
            except ImportError:
                pass

        if not api_key:
            logger.warning("GEMINI_API_KEY not set. Scene AI disabled.")
            _initialized = True
            return None

        try:
            from google import genai
            _client = genai.Client(api_key=api_key)
            logger.info("Gemini Scene AI initialized (gemini-3.5-flash-lite / gemini-3.8-flash)")
        except Exception as e:
            logger.error("Failed to initialize Gemini: %s", e)
            _client = None

        _initialized = True
        return _client


def _format_detections(objects: List[dict]) -> str:
    """Format YOLO detections into text context for Gemini."""
    if not objects:
        return "No objects detected by YOLO."

    lines = []
    for obj in objects[:8]:
        label = obj.get("label", "unknown")
        dist = obj.get("distance_m", "?")
        pos = obj.get("position", "ahead")
        urgency = obj.get("urgency", "info")
        prefix = "⚠️ VERY CLOSE: " if urgency == "critical" else ""
        lines.append(f"- {prefix}{label} at {dist}m {pos}")

    return "\n".join(lines)


class SceneSession:
    """Per-connection scene narration state to prevent cross-client starvation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_call_time = 0.0
        self._last_description = ""

    def describe_scene(self, frame_base64: str, objects: List[dict] = None) -> Optional[str]:
        """Send a frame + YOLO detections to Gemini for scene understanding."""
        global _shared_interval, _rate_limited_until

        if not frame_base64:
            return None

        now = time.time()

        with _api_lock:
            if now < _rate_limited_until:
                return None
            interval = _shared_interval

        with self._lock:
            if now - self._last_call_time < interval:
                return None
            self._last_call_time = now

        client = _get_client()
        if client is None:
            return None

        try:
            from google.genai import types

            payload = frame_base64.split(",", 1)[1] if "," in frame_base64 else frame_base64
            payload = payload.strip()
            if not payload:
                return None

            image_bytes = base64.b64decode(payload)
            if not image_bytes:
                return None

            det_text = _format_detections(objects or [])
            prompt = SCENE_PROMPT.format(detections=det_text)

            description = None
            for model_name in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            prompt,
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        ],
                        config=types.GenerateContentConfig(
                            max_output_tokens=35,
                            temperature=0.1,
                        ),
                    )
                    description = (response.text or "").strip()
                    if description:
                        break
                except Exception as model_err:
                    err_str = str(model_err).lower()
                    if "503" in err_str or "unavailable" in err_str or "not_found" in err_str:
                        continue
                    raise model_err

            if not description:
                return None

            with self._lock:
                self._last_description = description

            with _api_lock:
                _shared_interval = MIN_INTERVAL

            logger.info("Gemini: %s", description[:100])
            return description

        except Exception as e:
            error_str = str(e).lower()

            if "429" in str(e) or "rate" in error_str or "quota" in error_str or "resource" in error_str:
                with _api_lock:
                    _shared_interval = min(_shared_interval * 2, MAX_INTERVAL)
                    _rate_limited_until = now + _shared_interval
                    backoff = _shared_interval
                logger.warning("Gemini rate limited. Backing off to %.0fs", backoff)
            else:
                logger.error("Gemini API error: %s", e)

            return None

    def get_last_description(self) -> str:
        with self._lock:
            return self._last_description


_default_session = SceneSession()


def describe_scene(frame_base64: str, objects: List[dict] = None) -> Optional[str]:
    """Module-level helper delegating to the default SceneSession."""
    return _default_session.describe_scene(frame_base64, objects)


def get_last_description() -> str:
    """Return last description from the default SceneSession."""
    return _default_session.get_last_description()
