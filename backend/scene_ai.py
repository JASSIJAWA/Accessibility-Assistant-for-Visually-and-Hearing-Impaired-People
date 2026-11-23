"""
Scene AI — Gemini Vision API for intelligent scene understanding.

Uses google.genai SDK with gemini-3.8-flash model.
Sends camera frames + YOLO detection context every ~8 seconds.
Handles rate limits with automatic backoff and recovery.
"""

import os
import time
import base64
import threading
from typing import Optional, List

_client = None
_last_call_time = 0.0
_last_description = ""
_lock = threading.Lock()
_initialized = False

# Start at 4.5s interval for responsive navigation guidance
_current_interval = 4.5
MIN_INTERVAL = 4.5
MAX_INTERVAL = 45.0  # Back off up to 45s on rate limit
_rate_limited_until = 0.0  # Timestamp when rate limit expires

SCENE_PROMPT = """You are a real-time live navigation guide for a visually impaired person holding their phone forward.

DETECTED OBJECTS (YOLO AI):
{detections}

Give a 1 to 2 sentence immediate spoken guide (under 30 words). Rules:
1. START with the path ahead: state if path is clear or if there is an obstacle.
2. If an obstacle is closer than 1.5m, immediately warn them (e.g. "Careful, chair 1 meter on your left").
3. Use exact distance and position from the detections (e.g. "2 meters ahead", "to your right").
4. Mention key landmarks or openings (doorway, hallway, stairs, sidewalk).
5. Speak directly, concisely, and naturally as a walking guide. Do not say "In this image" or "I can see".

Example: "Path is clear ahead. Doorway is straight ahead about 3 meters."
Example: "Watch out, chair 1 meter on your left. Clear path to your right."
Example: "Two people approaching 3 meters ahead on the walkway." """


def _get_client():
    global _client, _initialized
    if _initialized:
        return _client

    _initialized = True

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY", "")
        except ImportError:
            pass

    if not api_key:
        print("⚠️  GEMINI_API_KEY not set. Scene AI disabled.")
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        print("✅ Gemini Scene AI initialized (gemini-3.8-flash)")
        return _client
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        return None


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


def describe_scene(frame_base64: str, objects: List[dict] = None) -> Optional[str]:
    """
    Send a frame + YOLO detections to Gemini for scene understanding.
    Rate-limited with automatic backoff on errors.
    """
    global _last_call_time, _last_description, _current_interval, _rate_limited_until

    now = time.time()

    # If rate limited, wait it out
    if now < _rate_limited_until:
        return None

    if now - _last_call_time < _current_interval:
        return None

    client = _get_client()
    if client is None:
        return None

    with _lock:
        if now - _last_call_time < _current_interval:
            return None
        _last_call_time = now

    try:
        from google.genai import types

        if "," in frame_base64:
            frame_base64 = frame_base64.split(",")[1]

        image_bytes = base64.b64decode(frame_base64)

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
                        max_output_tokens=90,
                        temperature=0.2,
                    ),
                )
                description = response.text.strip()
                if description:
                    break
            except Exception as model_err:
                err_str = str(model_err).lower()
                if "503" in err_str or "unavailable" in err_str or "not_found" in err_str:
                    continue
                raise model_err

        if not description:
            return None

        _last_description = description

        # Success — reset interval to normal
        _current_interval = MIN_INTERVAL
        print(f"🧠 Gemini: {description[:100]}...", flush=True)
        return description

    except Exception as e:
        error_str = str(e).lower()

        if "429" in str(e) or "rate" in error_str or "quota" in error_str or "resource" in error_str:
            # Rate limited — back off
            _current_interval = min(_current_interval * 2, MAX_INTERVAL)
            _rate_limited_until = now + _current_interval
            print(f"⚠️ Gemini rate limited. Backing off to {_current_interval:.0f}s", flush=True)
        else:
            print(f"❌ Gemini API error: {e}", flush=True)

        return None


def get_last_description() -> str:
    return _last_description
