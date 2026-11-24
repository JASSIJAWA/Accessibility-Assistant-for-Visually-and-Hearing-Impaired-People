# OMNI-ACCESS — AI Accessibility Assistant for Visually & Hearing Impaired People

**7th Semester Capstone Major Project — Real-Time Multimodal Accessibility Assistant**

---

## Overview

OMNI-ACCESS is a full-stack, mobile-first assistive platform engineered for both **visually impaired** and **hearing impaired** users. Running over HTTPS directly in a smartphone or desktop browser, it combines edge computer vision, monocular metric distance estimation, acoustic proximity sonar, OCR text reading, multimodal scene understanding, and bidirectional speech-to-text/TTS.

---

## Features

### Vision Guide (For Visually Impaired Users)
- **Real-Time Object Detection & Isolated Tracking** — YOLO11m (`yolo11m.pt`) paired with per-session ByteTrack IDs and EMA distance smoothing.
- **Orientation-Invariant Distance Calibration** — Estimates distance in meters and steps consistently across portrait (`720x1280`) and landscape (`1280x720`) mobile sensors.
- **Acoustic Proximity Sonar** — Continuous Web Audio parking-sensor pings that accelerate (`750ms` → `380ms` → `180ms`) as obstacles approach `< 1.8m`.
- **Gemini Scene Understanding** — Generates concise `< 10 word` directional cues via `gemini-3.5-flash-lite` (with automatic `gemini-3.8-flash` fallback and exponential rate-limit backoff).
- **Text Reading (OCR)** — EasyOCR extracts signage, labels, and printed text on demand and reads it aloud without interruption.
- **Facial Emotion Recognition** — DeepFace (`ssd` backend) detects facial expressions (`happy`, `sad`, `angry`, etc.) with spoken cues and telemetry alerts.
- **Mobile Fullscreen HUD & Hardware Torch** — Dedicated touch-friendly Fullscreen Guidance mode with rear/front camera flipping and hardware torch / screen light fallback.

### Hearing & Speech Assistant (For Hearing Impaired Users)
- **Multimodal Audio Transcription** — Captures raw PCM audio, resamples cleanly to 16 kHz mono WAV across iOS Safari, Android Chrome, and Desktop, and transcribes speech + environmental sounds (`[Car horn honking]`, `[Siren wailing]`) via Gemini Audio AI.
- **Offline Speech Fallback** — Automatically falls back to local **Faster-Whisper (`base.en`)** + urgent-keyword safety heuristics when offline or rate-limited.
- **Tone & Cognitive Summary** — Extracts speaker emotion, concise summary, key bullet points, and triggers haptic vibration alerts (`navigator.vibrate`) on emergencies.
- **Two-Way Speech Communication** — Quick-reply buttons and custom text-to-speech playback through the device speaker.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | React 19, Vite 8, Tailwind CSS v4, Web Audio API, Web Speech API |
| **Backend** | Python 3, FastAPI, Uvicorn (WebSockets + REST) |
| **Object Detection & Tracking** | Ultralytics YOLO11m + isolated per-session ByteTrack |
| **Scene & Audio AI** | Google GenAI SDK (`gemini-3.5-flash-lite` primary, `gemini-3.8-flash` fallback) |
| **Local Speech Fallback** | Faster-Whisper (`base.en`, `int8` CPU) + PyAV 16kHz resampler |
| **OCR & Emotion** | EasyOCR (`en`), DeepFace (`ssd` + `tf-keras`), OpenCV 4.x |

---

## Architecture

```
Mobile / Desktop Browser (React + HTTPS)
    │
    ├── /ws/vision (640px JPEG frames w/ backpressure)
    │       ▼
    │   FastAPI VisionSession + SceneSession
    │       ├── YOLO11m + ByteTrack → Distance, urgency & TTS reflexes
    │       ├── DeepFace (SSD)      → Face & emotion recognition
    │       └── Gemini Vision AI    → Directional navigation cues
    │
    ├── /api/ocr (POST full-res frame)
    │       ▼
    │   EasyOCR → Filtered text readout
    │
    └── /ws/hearing (16kHz mono WAV PCM + typed dialogue)
            ▼
        FastAPI HearingSession
            ├── Gemini Audio AI     → Verbatim STT, events, tone & summary
            └── Faster-Whisper AI   → Local offline speech fallback
```

---

## Local Setup & Running

### 1. Backend
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env` (see [backend/.env.example](backend/.env.example)):
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Start the backend server (downloads `yolo11m.pt` automatically on first run if missing):
```powershell
python main.py
```

### 2. Frontend
```powershell
cd frontend
npm install
npm run dev
```

Open `https://localhost:5173/` locally, or the Vite **Network URL** (`https://<your-lan-ip>:5173/`) on a mobile phone over the same Wi-Fi (accept the local HTTPS certificate warning once so camera & microphone permissions work).

---

## Testing & Linting

```powershell
# Backend automated test suite
cd backend
..\backend\venv\Scripts\python.exe -m pytest -v

# Frontend lint & production build
cd frontend
npm run lint
npm run build
```

---

## Free Deployment Guide

* **Recommended for Viva / Live Demos:** Deploy `frontend/` to **Vercel** (set `VITE_API_URL` and `VITE_WS_URL` per [frontend/.env.example](frontend/.env.example)), run `backend/` locally, and expose port `8000` via **Cloudflare Tunnel** (`cloudflared tunnel --url http://localhost:8000`). Set `CORS_ORIGINS` in `backend/.env` to your Vercel URL.
* **100% Cloud Option:** Deploy `backend/` using [backend/Dockerfile](backend/Dockerfile) on **Hugging Face Spaces** (16 GB free RAM) with `GEMINI_API_KEY` + `CORS_ORIGINS` configured in Space Secrets, and connect `frontend/` on **Vercel**.

---

## Repository
GitHub: https://github.com/JASSIJAWA/Accessibility-Assistant-for-Visually-and-Hearing-Impaired-People
