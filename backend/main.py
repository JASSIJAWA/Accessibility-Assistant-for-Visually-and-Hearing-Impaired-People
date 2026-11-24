"""
AI Accessibility Assistant — Backend Server
FastAPI + WebSocket server for real-time vision & audio processing.
"""

import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
import base64
import numpy as np
import cv2
from dotenv import load_dotenv
from vision import process_frame
from hearing_ai import analyze_conversation, reset_conversation, add_transcript_line, process_audio_chunk

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("assistive-backend")

# Load .env file for GEMINI_API_KEY
load_dotenv()

app = FastAPI(title="AI Assistive Backend")

# ─── CORS — Allow frontend origins (dev + mobile LAN) ────────────────────────
# In production, replace with actual deployed frontend domain
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://localhost:5173",
    "http://192.168.29.110:5173",
    "https://192.168.29.110:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── OCR (Lazy-loaded EasyOCR) ───────────────────────────────────────────────
_ocr_reader = None

# Max base64 payload size for OCR (10MB decoded ≈ ~13.3MB base64)
MAX_OCR_BASE64_LEN = 14_000_000


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        import torch
        _ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        logger.info("EasyOCR loaded (GPU=%s)", torch.cuda.is_available())
    return _ocr_reader


def run_ocr(base64_data: str) -> dict:
    """Decode base64 image and run EasyOCR on it."""
    # Strip data URL prefix if present
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]

    if len(base64_data) > MAX_OCR_BASE64_LEN:
        return {"error": "Image too large (>10MB)", "results": [], "text": ""}

    try:
        img_bytes = base64.b64decode(base64_data)
    except Exception:
        return {"error": "Invalid base64 data", "results": [], "text": ""}

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Failed to decode image", "results": [], "text": ""}

    reader = get_ocr_reader()
    # EasyOCR returns list of (bbox, text, confidence)
    raw_results = reader.readtext(frame)

    results = []
    accepted_texts = []

    for (bbox, text, confidence) in raw_results:
        conf_pct = round(confidence * 100)
        is_accepted = conf_pct >= 60 and len(text.strip()) >= 1

        # Garbage filter: reject if mostly non-alphanumeric
        stripped = text.strip()
        if stripped:
            alpha_num = sum(1 for c in stripped if c.isalnum())
            if alpha_num / len(stripped) < 0.4:
                is_accepted = False

        results.append({
            "text": text.strip(),
            "confidence": conf_pct,
            "accepted": is_accepted
        })

        if is_accepted:
            accepted_texts.append(text.strip())

    final_text = " ".join(accepted_texts).strip()

    return {
        "results": results,
        "text": final_text,
        "accepted": len(final_text) >= 2
    }


@app.get("/")
def read_root():
    return {"status": "Backend AI is running!", "modules": ["vision", "ocr"]}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/ocr")
async def ocr_endpoint(request: Request):
    """OCR endpoint: receives base64 image, returns detected text with confidence."""
    try:
        body = await request.json()
        image_data = body.get("image", "")
        if not image_data:
            return {"error": "No image provided", "results": [], "text": ""}

        result = await asyncio.to_thread(run_ocr, image_data)
        return result
    except Exception as e:
        logger.error("OCR error: %s", e)
        return {"error": str(e), "results": [], "text": ""}


# ─── Vision WebSocket ────────────────────────────────────────────────────────
@app.websocket("/ws/vision")
async def vision_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time vision processing.
    - YOLO + emotions run on every frame (fast)
    - Gemini scene description runs every ~6 seconds (rich)
    """
    await websocket.accept()
    logger.info("Vision WebSocket connected")

    processing = False
    gemini_task = None
    last_gemini_description = ""
    new_gemini_available = False
    frames_dropped = 0
    frames_processed = 0
    client_gone = False

    async def run_gemini_background(frame_data, objects):
        """Run Gemini in background without blocking YOLO."""
        nonlocal last_gemini_description, new_gemini_available
        try:
            desc = await asyncio.to_thread(describe_scene, frame_data, objects)
            if desc and desc != last_gemini_description:
                last_gemini_description = desc
                new_gemini_available = True
        except Exception as e:
            logger.warning("Gemini background error: %s", e)

    try:
        while True:
            data = await websocket.receive_text()

            if processing:
                frames_dropped += 1
                if frames_dropped % 50 == 0:
                    logger.info("Frames dropped so far: %d (processed: %d)", frames_dropped, frames_processed)
                continue

            processing = True
            try:
                # Run YOLO + emotions (fast, every frame)
                result = await asyncio.to_thread(process_frame, data)
                frames_processed += 1

                # Kick off Gemini with YOLO results as context
                if gemini_task is None or gemini_task.done():
                    gemini_task = asyncio.create_task(
                        run_gemini_background(data, result.get("objects", []))
                    )

                # Attach latest Gemini description to response
                result["scene_description"] = last_gemini_description
                result["new_gemini_guidance"] = new_gemini_available
                if new_gemini_available:
                    new_gemini_available = False

                await websocket.send_text(json.dumps(result))
            except WebSocketDisconnect:
                logger.info("Vision WebSocket disconnected (client closed)")
                client_gone = True
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
        # Cancel any pending Gemini task
        if gemini_task and not gemini_task.done():
            gemini_task.cancel()
        logger.info("Vision session ended. Processed: %d, Dropped: %d", frames_processed, frames_dropped)


# ─── Hearing/Speech Analyze REST ─────────────────────────────────────────────
@app.post("/api/hearing/analyze")
async def hearing_analyze_text(request: Request):
    """Analyze conversation text via REST API."""
    try:
        body = await request.json()
        text = body.get("text", "")
        result = await asyncio.to_thread(analyze_conversation, text)
        return result
    except Exception as e:
        return {"error": str(e)}


# ─── Hearing WebSocket ───────────────────────────────────────────────────────
@app.websocket("/ws/hearing")
async def hearing_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for hearing impaired module.
    - Receives transcript lines from browser Web Speech API
    - Receives raw audio chunks from MediaRecorder
    - Sends to Gemini for transcription + summarization + tone analysis
    - Returns: summary, tone, key_points, is_urgent, transcription
    """
    await websocket.accept()
    logger.info("Hearing WebSocket connected")

    hearing_task = None
    client_gone = False

    async def run_hearing_analysis(text: str):
        try:
            result = await asyncio.to_thread(analyze_conversation, text)
            try:
                await websocket.send_text(json.dumps(result))
            except (WebSocketDisconnect, RuntimeError):
                pass  # client disconnected while we were processing
        except Exception as e:
            logger.error("Hearing analysis error: %s", e)

    async def run_audio_analysis(audio_bytes: bytes, mime_type: str):
        try:
            clean_mime = mime_type.split(";")[0].strip() if mime_type else "audio/webm"
            result = await asyncio.to_thread(process_audio_chunk, audio_bytes, clean_mime)
            try:
                await websocket.send_text(json.dumps(result))
            except (WebSocketDisconnect, RuntimeError):
                pass  # client disconnected
        except Exception as e:
            logger.error("Audio analysis error: %s", e)
            try:
                await websocket.send_text(json.dumps({
                    "error": str(e),
                    "summary": "Could not transcribe audio.",
                    "transcription": ""
                }))
            except (WebSocketDisconnect, RuntimeError):
                pass

    try:
        while True:
            data = await websocket.receive_text()

            # Validate JSON
            try:
                msg = json.loads(data)
                if not isinstance(msg, dict):
                    continue
            except json.JSONDecodeError:
                try:
                    await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                except (WebSocketDisconnect, RuntimeError):
                    break
                continue

            action = msg.get("action", "transcript")

            if action == "reset":
                reset_conversation()
                try:
                    await websocket.send_text(json.dumps({"status": "reset"}))
                except (WebSocketDisconnect, RuntimeError):
                    break
                continue

            if action == "audio":
                audio_base64 = msg.get("data", "")
                mime_type = msg.get("mime_type", "audio/webm")
                if audio_base64:
                    # Size guard: reject audio payloads > 10MB base64
                    if len(audio_base64) > MAX_OCR_BASE64_LEN:
                        continue
                    if "," in audio_base64:
                        audio_base64 = audio_base64.split(",")[1]
                    try:
                        audio_bytes = base64.b64decode(audio_base64)
                    except Exception:
                        continue
                    asyncio.create_task(run_audio_analysis(audio_bytes, mime_type))
                continue

            if action == "transcript":
                text = msg.get("text", "").strip()
                if text:
                    add_transcript_line(text)
                    if hearing_task is None or hearing_task.done():
                        hearing_task = asyncio.create_task(
                            run_hearing_analysis(text)
                        )

    except WebSocketDisconnect:
        logger.info("Hearing WebSocket disconnected")
    except Exception as e:
        logger.error("Hearing WebSocket error: %s", e)
    finally:
        # Cancel any pending hearing task
        if hearing_task and not hearing_task.done():
            hearing_task.cancel()
        logger.info("Hearing session ended")


# ─── Import describe_scene here (after app setup) for the vision endpoint ────
from scene_ai import describe_scene


if __name__ == "__main__":
    import sys
    reload = "--reload" in sys.argv
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        ws_max_size=5 * 1024 * 1024,  # 5MB max WebSocket message
    )
