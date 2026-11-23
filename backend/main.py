"""
AI Accessibility Assistant — Backend Server
FastAPI + WebSocket server for real-time vision & audio processing.
"""

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
from scene_ai import describe_scene, get_last_description
from hearing_ai import analyze_conversation, reset_conversation, add_transcript_line, process_audio_chunk

# Load .env file for GEMINI_API_KEY
load_dotenv()

app = FastAPI(title="AI Assistive Backend")

# Allow React frontend to communicate with Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load EasyOCR (downloads model on first use, ~100MB)
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR loaded")
    return _ocr_reader


def run_ocr(base64_data: str) -> dict:
    """Decode base64 image and run EasyOCR on it."""
    # Strip data URL prefix if present
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]

    img_bytes = base64.b64decode(base64_data)
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
        print(f"❌ OCR error: {e}")
        return {"error": str(e), "results": [], "text": ""}


@app.websocket("/ws/vision")
async def vision_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time vision processing.
    - YOLO + emotions run on every frame (fast)
    - Gemini scene description runs every ~6 seconds (rich)
    """
    await websocket.accept()
    print("✅ Vision WebSocket connected")

    processing = False
    gemini_task = None
    last_gemini_description = ""

    async def run_gemini_background(frame_data, objects):
        """Run Gemini in background without blocking YOLO."""
        nonlocal last_gemini_description
        try:
            desc = await asyncio.to_thread(describe_scene, frame_data, objects)
            if desc:
                last_gemini_description = desc
        except Exception as e:
            print(f"❌ Gemini background error: {e}")

    try:
        while True:
            data = await websocket.receive_text()

            if processing:
                continue

            processing = True
            try:
                # Run YOLO + emotions (fast, every frame)
                result = await asyncio.to_thread(process_frame, data)

                # Kick off Gemini with YOLO results as context
                if gemini_task is None or gemini_task.done():
                    gemini_task = asyncio.create_task(
                        run_gemini_background(data, result.get("objects", []))
                    )

                # Attach latest Gemini description to response
                result["scene_description"] = last_gemini_description

                await websocket.send_text(json.dumps(result))
            except WebSocketDisconnect:
                print("❌ Vision WebSocket disconnected (client closed)")
                break
            except Exception as e:
                print(f"❌ Frame processing error: {e}", flush=True)
                break
            finally:
                processing = False

    except WebSocketDisconnect:
        print("❌ Vision WebSocket disconnected")
    except Exception as e:
        print(f"❌ Vision WebSocket error: {e}")


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
    print("✅ Hearing WebSocket connected")

    hearing_task = None

    async def run_hearing_analysis(text: str):
        try:
            result = await asyncio.to_thread(analyze_conversation, text)
            await websocket.send_text(json.dumps(result))
        except Exception as e:
            print(f"❌ Hearing analysis error: {e}")

    async def run_audio_analysis(audio_bytes: bytes, mime_type: str):
        try:
            clean_mime = mime_type.split(";")[0].strip() if mime_type else "audio/webm"
            result = await asyncio.to_thread(process_audio_chunk, audio_bytes, clean_mime)
            await websocket.send_text(json.dumps(result))
        except Exception as e:
            print(f"❌ Audio analysis error: {e}")
            await websocket.send_text(json.dumps({
                "error": str(e),
                "summary": "Could not transcribe audio.",
                "transcription": ""
            }))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            action = msg.get("action", "transcript")

            if action == "reset":
                reset_conversation()
                await websocket.send_text(json.dumps({"status": "reset"}))
                continue

            if action == "audio":
                audio_base64 = msg.get("data", "")
                mime_type = msg.get("mime_type", "audio/webm")
                if audio_base64:
                    if "," in audio_base64:
                        audio_base64 = audio_base64.split(",")[1]
                    audio_bytes = base64.b64decode(audio_base64)
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
        print("❌ Hearing WebSocket disconnected")
    except Exception as e:
        print(f"❌ Hearing WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
