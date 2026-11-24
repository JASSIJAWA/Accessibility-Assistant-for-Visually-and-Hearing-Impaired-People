"""
AI Accessibility Assistant — Backend Server
FastAPI + WebSocket server for real-time vision, OCR, and hearing/speech intelligence.
"""

import asyncio
import base64
import json
import logging
import os
import re
import threading

import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hearing_ai import HearingSession, analyze_conversation
from scene_ai import SceneSession
from vision import VisionSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("assistive-backend")

load_dotenv()

app = FastAPI(title="AI Assistive Backend")

# ─── CORS & WebSocket Origin Security ────────────────────────────────────────
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://localhost:5173",
    "http://127.0.0.1:5173",
    "https://127.0.0.1:5173",
]
_env_origins = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
]
ALLOWED_ORIGINS = list(dict.fromkeys(_DEFAULT_ORIGINS + _env_origins))

# Allow local RFC1918 Wi-Fi IPs on Vite dev ports so phone testing works even after DHCP renewals,
# plus an optional CORS_ORIGIN_REGEX env var for cloud preview domains (e.g. ^https://.*\.vercel\.app$)
_LAN_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?$"
_extra_origin_regex = os.environ.get("CORS_ORIGIN_REGEX", "").strip()
_ALLOWED_ORIGIN_REGEX = f"(?:{_LAN_ORIGIN_REGEX})|(?:{_extra_origin_regex})" if _extra_origin_regex else _LAN_ORIGIN_REGEX
_origin_pattern = re.compile(_ALLOWED_ORIGIN_REGEX)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=_ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_ws_origin_allowed(origin: str | None) -> bool:
    """Validate browser Origin header on WebSocket handshakes."""
    if origin is None:
        return True  # Native/CLI/proxied local test clients without Origin header
    if "*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
        return True
    return bool(_origin_pattern.fullmatch(origin))


# ─── OCR (Thread-safe lazy-loaded EasyOCR) ───────────────────────────────────
_ocr_reader = None
_ocr_load_lock = threading.Lock()
_ocr_infer_lock = threading.Lock()

# Max base64 payload size (~10MB decoded ≈ ~13.3MB base64)
MAX_OCR_BASE64_LEN = 14_000_000
MAX_AUDIO_BASE64_LEN = 7_000_000


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader
    with _ocr_load_lock:
        if _ocr_reader is None:
            import easyocr
            import torch
            _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            logger.info("EasyOCR loaded (GPU=%s)", torch.cuda.is_available())
    return _ocr_reader


def run_ocr(base64_data: str) -> dict:
    """Decode base64 image safely and run EasyOCR on it."""
    if not base64_data or not isinstance(base64_data, str):
        return {"error": "Invalid image payload", "results": [], "text": ""}

    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]
    base64_data = base64_data.strip()

    if not base64_data:
        return {"error": "Empty image data", "results": [], "text": ""}

    if len(base64_data) > MAX_OCR_BASE64_LEN:
        return {"error": "Image too large (>10MB)", "results": [], "text": ""}

    try:
        img_bytes = base64.b64decode(base64_data, validate=False)
    except Exception:
        return {"error": "Invalid base64 data", "results": [], "text": ""}

    if not img_bytes:
        return {"error": "Empty decoded image buffer", "results": [], "text": ""}

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    if img_array.size == 0:
        return {"error": "Empty image buffer", "results": [], "text": ""}

    try:
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except cv2.error:
        frame = None

    if frame is None:
        return {"error": "Failed to decode image", "results": [], "text": ""}

    reader = get_ocr_reader()
    with _ocr_infer_lock:
        raw_results = reader.readtext(frame)

    results = []
    accepted_texts = []

    for (_bbox, text, confidence) in raw_results:
        conf_pct = round(confidence * 100)
        stripped = text.strip()
        is_accepted = conf_pct >= 60 and len(stripped) >= 1

        if stripped:
            alpha_num = sum(1 for c in stripped if c.isalnum())
            if alpha_num / len(stripped) < 0.4:
                is_accepted = False

        results.append({
            "text": stripped,
            "confidence": conf_pct,
            "accepted": is_accepted,
        })

        if is_accepted:
            accepted_texts.append(stripped)

    final_text = " ".join(accepted_texts).strip()

    return {
        "results": results,
        "text": final_text,
        "accepted": len(final_text) >= 2,
    }


@app.get("/")
def read_root():
    return {"status": "Backend AI is running!", "modules": ["vision", "ocr", "hearing"]}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/ocr")
async def ocr_endpoint(request: Request):
    """OCR endpoint: receives base64 image, returns detected text with confidence."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body", "results": [], "text": ""})

    if not isinstance(body, dict):
        return JSONResponse(status_code=422, content={"error": "JSON object expected", "results": [], "text": ""})

    image_data = body.get("image", "")
    if not image_data or not isinstance(image_data, str):
        return JSONResponse(status_code=400, content={"error": "No image provided", "results": [], "text": ""})

    if len(image_data) > MAX_OCR_BASE64_LEN:
        return JSONResponse(status_code=413, content={"error": "Image too large (>10MB)", "results": [], "text": ""})

    try:
        result = await asyncio.to_thread(run_ocr, image_data)
        if result.get("error"):
            return JSONResponse(status_code=422, content=result)
        return result
    except Exception as e:
        logger.error("OCR error: %s", e)
        return JSONResponse(status_code=500, content={"error": "OCR processing failed", "results": [], "text": ""})


# ─── Vision WebSocket ────────────────────────────────────────────────────────
@app.websocket("/ws/vision")
async def vision_endpoint(websocket: WebSocket):
    """
    Per-connection isolated WebSocket endpoint for real-time vision processing.
    - YOLO + emotions run per frame via dedicated VisionSession
    - Gemini scene narration runs asynchronously via dedicated SceneSession
    """
    origin = websocket.headers.get("origin")
    if not is_ws_origin_allowed(origin):
        logger.warning("Rejected unauthorized Vision WebSocket origin: %s", origin)
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("Vision WebSocket connected")

    vision_session = VisionSession()
    scene_session = SceneSession()

    processing = False
    gemini_task = None
    last_gemini_description = ""
    new_gemini_available = False
    frames_dropped = 0
    frames_processed = 0

    async def run_gemini_background(frame_data, objects):
        nonlocal last_gemini_description, new_gemini_available
        try:
            desc = await asyncio.to_thread(scene_session.describe_scene, frame_data, objects)
            if desc and desc != last_gemini_description:
                last_gemini_description = desc
                new_gemini_available = True
        except Exception as e:
            logger.warning("Gemini background error: %s", e)

    try:
        while True:
            data = await websocket.receive_text()

            if data.startswith("{"):
                try:
                    cfg = json.loads(data)
                    if isinstance(cfg, dict) and cfg.get("type") == "config":
                        vision_session.update_camera_config(
                            is_mobile=bool(cfg.get("is_mobile", False)),
                            facing_mode=str(cfg.get("facing_mode", "environment")),
                            focal_scale=float(cfg.get("focal_scale", 1.0)),
                        )
                        await websocket.send_text(json.dumps({
                            "type": "config_ack",
                            "focal_factor": round(vision_session._focal_factor, 3),
                        }))
                        continue
                except Exception:
                    pass

            if processing:
                frames_dropped += 1
                if frames_dropped % 50 == 0:
                    logger.info(
                        "Frames dropped so far: %d (processed: %d)",
                        frames_dropped, frames_processed,
                    )
                continue

            processing = True
            try:
                result = await asyncio.to_thread(vision_session.process_frame, data)
                frames_processed += 1

                if not result.get("error") and (gemini_task is None or gemini_task.done()):
                    gemini_task = asyncio.create_task(
                        run_gemini_background(data, result.get("objects", []))
                    )

                result["scene_description"] = last_gemini_description
                result["new_gemini_guidance"] = new_gemini_available
                if new_gemini_available:
                    new_gemini_available = False

                await websocket.send_text(json.dumps(result))
            except WebSocketDisconnect:
                logger.info("Vision WebSocket disconnected (client closed)")
                break
            except Exception as e:
                logger.error("Frame processing error: %s", e)
                break
            finally:
                processing = False

    except WebSocketDisconnect:
        logger.info("Vision WebSocket disconnected")
    except Exception as e:
        logger.error("Vision WebSocket error: %s", e)
    finally:
        if gemini_task and not gemini_task.done():
            gemini_task.cancel()
        logger.info(
            "Vision session ended. Processed: %d, Dropped: %d",
            frames_processed, frames_dropped,
        )


# ─── Hearing/Speech Analyze REST ─────────────────────────────────────────────
@app.post("/api/hearing/analyze")
async def hearing_analyze_text(request: Request):
    """Analyze conversation text via REST API."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=422, content={"error": "JSON object expected"})

    text = body.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return JSONResponse(status_code=400, content={"error": "Non-empty 'text' field required"})

    try:
        result = await asyncio.to_thread(analyze_conversation, text.strip())
        return result
    except Exception as e:
        logger.error("Hearing REST error: %s", e)
        return JSONResponse(status_code=500, content={"error": "Hearing analysis failed"})


# ─── Hearing WebSocket ───────────────────────────────────────────────────────
@app.websocket("/ws/hearing")
async def hearing_endpoint(websocket: WebSocket):
    """
    Per-connection isolated WebSocket endpoint for hearing impaired module.
    Tracks all spawned tasks and cancels them cleanly upon disconnect.
    """
    origin = websocket.headers.get("origin")
    if not is_ws_origin_allowed(origin):
        logger.warning("Rejected unauthorized Hearing WebSocket origin: %s", origin)
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("Hearing WebSocket connected")

    hearing_session = HearingSession()
    pending_tasks: set[asyncio.Task] = set()

    def _track_task(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)
        return task

    async def _safe_send(payload: dict):
        try:
            await websocket.send_text(json.dumps(payload))
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def run_hearing_analysis(text: str):
        try:
            result = await asyncio.to_thread(hearing_session.analyze_conversation, text)
            await _safe_send(result)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Hearing analysis error: %s", e)

    async def run_audio_analysis(audio_bytes: bytes, mime_type: str):
        try:
            clean_mime = mime_type.split(";")[0].strip() if mime_type else "audio/webm"
            result = await asyncio.to_thread(
                hearing_session.process_audio_chunk, audio_bytes, clean_mime
            )
            await _safe_send(result)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Audio analysis error: %s", e)
            await _safe_send({
                "error": str(e),
                "summary": "Could not transcribe audio.",
                "transcription": "",
            })

    try:
        while True:
            data = await websocket.receive_text()

            try:
                msg = json.loads(data)
                if not isinstance(msg, dict):
                    continue
            except json.JSONDecodeError:
                await _safe_send({"error": "Invalid JSON"})
                continue

            action = msg.get("action", "transcript")

            if action == "reset":
                hearing_session.reset_conversation()
                await _safe_send({"status": "reset"})
                continue

            if action == "audio":
                audio_base64 = msg.get("data", "")
                mime_type = msg.get("mime_type", "audio/webm")
                if not audio_base64 or not isinstance(audio_base64, str):
                    continue
                if len(audio_base64) > MAX_AUDIO_BASE64_LEN:
                    await _safe_send({
                        "error": "Audio segment too long. Please keep recordings under 90 seconds.",
                        "summary": "Recording exceeded max upload size.",
                        "transcription": "",
                    })
                    continue
                if "," in audio_base64:
                    audio_base64 = audio_base64.split(",", 1)[1]
                try:
                    audio_bytes = base64.b64decode(audio_base64)
                except Exception:
                    await _safe_send({"error": "Invalid base64 audio payload", "transcription": ""})
                    continue
                _track_task(run_audio_analysis(audio_bytes, mime_type))
                continue

            if action == "transcript":
                text = str(msg.get("text", "")).strip()
                if text:
                    _track_task(run_hearing_analysis(text))

    except WebSocketDisconnect:
        logger.info("Hearing WebSocket disconnected")
    except Exception as e:
        logger.error("Hearing WebSocket error: %s", e)
    finally:
        for task in list(pending_tasks):
            if not task.done():
                task.cancel()
        logger.info("Hearing session ended")


if __name__ == "__main__":
    import sys

    reload_mode = "--reload" in sys.argv
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_mode,
        ws_max_size=6 * 1024 * 1024,  # 6MB max WebSocket message
    )
