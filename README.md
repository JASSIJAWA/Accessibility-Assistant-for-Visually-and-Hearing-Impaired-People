# AI Accessibility Assistant
**7th Semester Major Project — Real-Time Visual Assistance for the Visually Impaired**

---

## Overview

A real-time, mobile-accessible AI assistant designed to help visually impaired users navigate their surroundings. The system continuously watches through the phone camera and speaks aloud what it sees — objects, distances, scene context — acting as an intelligent guide.

---

## Features

- 🟡 **Real-Time Object Detection** — Powered by YOLO11m (80 object classes), with distance estimation and urgency classification (Critical / Warning / Safe)
- 🧠 **AI Scene Understanding** — Google Gemini 3.8 Flash generates spatial, navigation-focused descriptions (e.g. "Chair 1 meter on your left, clear path ahead")
- 🗣️ **Continuous Voice Narration** — Text-to-speech speaks both YOLO detections and Gemini descriptions; gracefully falls back to YOLO if Gemini rate limits are hit
- 📝 **Text Reading (OCR)** — EasyOCR reads printed text visible in the camera frame
- 😊 **Face & Emotion Detection** — DeepFace detects faces and recognizes emotions in real time
- 📱 **Mobile-First Design** — Runs in any mobile browser, rear/front camera switching, HTTPS enabled

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React.js, Vite, WebSocket, Web Speech API |
| Backend | Python, FastAPI, Uvicorn |
| Object Detection | YOLO11m (Ultralytics) |
| Scene AI | Google Gemini 3.8 Flash API |
| OCR | EasyOCR |
| Face/Emotion | DeepFace (SSD backend) |
| Communication | WebSocket (real-time frames), REST API (OCR) |

---

## Architecture

```
Mobile Browser (React)
    │
    │  WebSocket (base64 JPEG frames)
    ▼
FastAPI Backend
    ├── YOLO11m → Object detection + distance estimation
    ├── DeepFace → Face & emotion detection
    ├── EasyOCR → Text recognition
    └── Gemini API → Semantic scene description (every ~8s)
    │
    │  JSON response (objects, narration, scene_description)
    ▼
Mobile Browser → Text-to-Speech narration spoken aloud
```

---

## Setup Instructions

### Backend
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` folder:
```
GEMINI_API_KEY=your_api_key_here
```

Then run:
```bash
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Mobile Testing (Local Network)

1. Run both servers (backend on port 8000, frontend on port 5173)
2. Find the **Network URL** from Vite output (e.g. `https://192.168.x.x:5173/`)
3. Open that URL on your phone (must be on the same Wi-Fi)
4. Accept the security warning (self-signed certificate for local HTTPS)
5. Grant camera permission when prompted

---

## Project Status

| Module | Status |
|--------|--------|
| Real-time camera streaming | ✅ Complete |
| YOLO11m object detection | ✅ Complete |
| Distance & urgency estimation | ✅ Complete |
| Gemini AI scene narration | ✅ Complete |
| Voice TTS narrator | ✅ Complete |
| EasyOCR text reading | ✅ Complete |
| Face & emotion detection | ✅ Complete |
| Rate-limit backoff (Gemini) | ✅ Complete |

---

## Repository
GitHub: https://github.com/JASSIJAWA/ai-accessibility-assistant
