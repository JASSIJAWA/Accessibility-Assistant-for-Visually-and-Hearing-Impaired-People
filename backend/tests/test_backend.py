"""
Automated test suite for AI Accessibility Assistant backend.
Tests vision math/edge cases, session isolation, audio resampling, OCR safety, REST & WebSocket endpoints.
"""

import base64
import io
import os
import sys
import wave
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

# Ensure backend directory is on import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hearing_ai
import main
import vision


def _encode_jpeg_b64(img: np.ndarray, prefix: bool = True) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}" if prefix else b64


def _make_wav_bytes(sample_rate: int = 16000, channels: int = 1, duration_sec: float = 0.5) -> bytes:
    buf = io.BytesIO()
    n_samples = int(sample_rate * duration_sec)
    tone = (np.sin(np.linspace(0, 440 * 2 * np.pi * duration_sec, n_samples)) * 16000).astype(np.int16)
    if channels == 2:
        tone = np.column_stack((tone, tone)).flatten()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(tone.tobytes())
    return buf.getvalue()


# ─── Vision Unit Tests ───────────────────────────────────────────────────────
def test_decode_frame_handles_empty_and_malformed():
    assert vision.decode_frame("") is None
    assert vision.decode_frame("data:image/jpeg;base64,") is None
    assert vision.decode_frame("not-valid-base64-!!!") is None
    assert vision.decode_frame(None) is None


def test_decode_frame_valid():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    decoded = vision.decode_frame(_encode_jpeg_b64(img))
    assert decoded is not None
    assert decoded.shape[:2] == (120, 160)


def test_distance_orientation_consistency():
    # Rotating between 640x360 landscape and 360x640 portrait keeps focal length invariant
    landscape = vision.estimate_distance("car", box_w=180, box_h=150, frame_w=640, frame_h=360)
    portrait = vision.estimate_distance("car", box_w=180, box_h=150, frame_w=360, frame_h=640)
    assert landscape["distance_m"] == portrait["distance_m"]
    assert landscape["distance_steps"] == portrait["distance_steps"]


def test_vision_session_isolation_and_emotion_cache_reset():
    s1 = vision.VisionSession()
    s2 = vision.VisionSession()

    # Different LRU track histories do not leak across sessions
    d1 = vision.estimate_distance("chair", 100, 170, 640, 480, track_id=42, track_history=s1._track_history)
    assert s1._track_history.get(42) is not None
    assert s2._track_history.get(42) is None
    assert d1["distance_m"] > 0

    # Calling detect_emotions with has_person=False must clear cached faces
    s1._cached_emotions = [{"emotion": "happy", "confidence": 90.0, "box": {"x": 0, "y": 0, "w": 10, "h": 10}}]
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert s1.detect_emotions(img, has_person=False) == []
    assert s1._cached_emotions == []


def test_deepface_ssd_runs_without_opencv_dnn_crash():
    s = vision.VisionSession()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Blank frame raises ValueError('Face could not be detected') inside DeepFace, caught cleanly -> []
    emotions = s.detect_emotions(img, has_person=True)
    assert emotions == []


# ─── Hearing & Audio Resampling Tests ────────────────────────────────────────
def test_convert_to_wav_passthrough_and_resample():
    wav_16k = _make_wav_bytes(sample_rate=16000, channels=1)
    assert hearing_ai.convert_to_wav(wav_16k) == wav_16k

    wav_48k_stereo = _make_wav_bytes(sample_rate=48000, channels=2)
    converted = hearing_ai.convert_to_wav(wav_48k_stereo)
    with wave.open(io.BytesIO(converted), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2


def test_hearing_session_isolation_and_offline_fallback():
    h1 = hearing_ai.HearingSession()
    h2 = hearing_ai.HearingSession()

    res1 = h1.analyze_conversation("Watch out, fire near the exit!", skip_gemini=True)
    assert res1["is_urgent"] is True
    assert res1["tone"] == "urgent"

    assert "fire" in h1.get_conversation_text().lower()
    assert h2.get_conversation_text() == ""

    h1.reset_conversation()
    assert h1.get_conversation_text() == ""


# ─── OCR & FastAPI Endpoint Tests ────────────────────────────────────────────
def test_run_ocr_empty_and_valid():
    empty_res = main.run_ocr("data:image/jpeg;base64,")
    assert "error" in empty_res

    img = np.zeros((160, 320, 3), dtype=np.uint8)
    cv2.putText(img, "EXIT 123", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    res = main.run_ocr(_encode_jpeg_b64(img))
    assert "EXIT" in res["text"]
    assert res["accepted"] is True


def test_rest_endpoints_validation_and_health():
    client = TestClient(main.app)

    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "healthy"}

    # Invalid OCR inputs return proper 4xx status codes
    assert client.post("/api/ocr", content="not-json").status_code == 400
    assert client.post("/api/ocr", json={"image": ""}).status_code == 400
    assert client.post("/api/ocr", json={"image": "data:image/jpeg;base64,"}).status_code == 422

    # Hearing REST validation
    assert client.post("/api/hearing/analyze", content="bad").status_code == 400
    assert client.post("/api/hearing/analyze", json={"text": "   "}).status_code == 400


def test_websocket_origin_guard_and_protocol():
    assert main.is_ws_origin_allowed(None) is True
    assert main.is_ws_origin_allowed("https://localhost:5173") is True
    assert main.is_ws_origin_allowed("https://192.168.1.45:5173") is True
    assert main.is_ws_origin_allowed("https://evil-attacker.example") is False

    client = TestClient(main.app)

    with client.websocket_connect("/ws/hearing") as ws:
        ws.send_text("invalid-json")
        assert ws.receive_json() == {"error": "Invalid JSON"}

        ws.send_text('{"action": "reset"}')
        assert ws.receive_json() == {"status": "reset"}

        oversized = "A" * (main.MAX_AUDIO_BASE64_LEN + 10)
        ws.send_json({"action": "audio", "data": oversized, "mime_type": "audio/wav"})
        resp = ws.receive_json()
        assert "error" in resp

    with patch.object(vision.VisionSession, "process_frame", return_value={"error": "Failed to decode frame"}):
        with client.websocket_connect("/ws/vision") as ws:
            ws.send_text("bad-frame")
            resp = ws.receive_json()
            assert resp["error"] == "Failed to decode frame"
