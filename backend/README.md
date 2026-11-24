---
title: AVHA Backend
emoji: 👁️
colorFrom: zinc
colorTo: green
sdk: docker
app_port: 7860
---

# AVHA — Assistant for Visual & Hearing Accessibility (Backend)

FastAPI + WebSocket AI server powering **AVHA**:
- Real-time YOLO11m object detection, ByteTrack tracking & calibrated metric distance estimation (`/ws/vision`)
- Gemini Vision scene cues & DeepFace emotion recognition (`/ws/vision`)
- EasyOCR on-demand text reading (`/api/ocr`)
- Multimodal Gemini Audio + Faster-Whisper offline speech transcription (`/ws/hearing`)

## Required Space Secrets / Variables
- `GEMINI_API_KEY`: Google Gemini API key
- `CORS_ORIGINS`: Your Vercel frontend URL (e.g. `https://avha-app.vercel.app`)
- `CORS_ORIGIN_REGEX` *(optional)*: `^https://.*\.vercel\.app$` for Vercel preview branches
