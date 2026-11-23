"""
Hearing AI Module — Powered by local Faster-Whisper + Gemini Cognitive Analysis.

Features:
  - Local speech-to-text with faster-whisper (tiny.en, runs in ~80ms on CPU)
  - Zero Google speech cloud errors or network drops
  - Gemini conversation summarization and emotional tone classification
  - Emergency/distress detection with visual and haptic alerts
"""

import os
import io
import time
import json
import base64
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

# Lazy load Whisper model
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("⚡ Loading Faster-Whisper base.en model on CPU...")
        _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("✅ Faster-Whisper base.en ready for accurate speech transcription!")
    return _whisper_model


PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.8-flash"

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client


# ─── Rate limiting ────────────────────────────────────────────────────────────
_last_call_time = 0.0
_min_interval = 3.0
_current_interval = 3.0
_MAX_INTERVAL = 30.0

def _should_call_gemini() -> bool:
    global _last_call_time
    return (time.time() - _last_call_time) >= _current_interval

def _on_success():
    global _current_interval, _last_call_time
    _current_interval = _min_interval
    _last_call_time = time.time()

def _on_rate_limit():
    global _current_interval, _last_call_time
    _current_interval = min(_current_interval * 1.5, _MAX_INTERVAL)
    _last_call_time = time.time()


# ─── Rolling conversation buffer ──────────────────────────────────────────────
_conversation_buffer = []
_MAX_BUFFER_LINES = 30

def add_transcript_line(text: str):
    global _conversation_buffer
    text = text.strip()
    if text:
        _conversation_buffer.append(text)
        if len(_conversation_buffer) > _MAX_BUFFER_LINES:
            _conversation_buffer = _conversation_buffer[-_MAX_BUFFER_LINES:]

def get_conversation_text() -> str:
    return "\n".join(_conversation_buffer)


_last_result = {
    "summary": "",
    "tone": "neutral",
    "tone_emoji": "😐",
    "key_points": [],
    "is_urgent": False,
    "transcription": ""
}

TONE_EMOJI = {
    "urgent": "🚨",
    "alarmed": "🚨",
    "emergency": "🚨",
    "angry": "😠",
    "happy": "😊",
    "sad": "😢",
    "confused": "😕",
    "calm": "😐",
    "neutral": "😐",
    "excited": "😃",
    "worried": "😟",
}


def _parse_gemini_json(raw_text: str) -> dict:
    raw = raw_text.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    start_idx = raw.find("{")
    end_idx = raw.rfind("}")
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx+1]
    return json.loads(raw)


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Transcribes audio bytes locally using faster-whisper base.en model.
    Uses beam search (beam_size=5) and VAD filtering for high accuracy.
    """
    try:
        whisper = get_whisper()
        buf = io.BytesIO(audio_bytes)
        segments, _ = whisper.transcribe(
            buf,
            beam_size=5,
            best_of=5,
            language="en",
            temperature=[0.0, 0.2],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            initial_prompt="A clear spoken conversation about daily life, navigation, questions, directions, and emergencies."
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
    except Exception as e:
        print(f"❌ Whisper transcription error: {e}")
        return ""


def analyze_conversation(transcript_text: str = None) -> dict:
    """
    Analyze conversation text using Gemini for tone, summary, and urgency.
    """
    global _last_result

    if transcript_text:
        add_transcript_line(transcript_text)

    conversation = get_conversation_text()
    if not conversation.strip():
        return _last_result

    if not _should_call_gemini():
        return _last_result

    prompt = f"""You are an assistive AI for a deaf or hard-of-hearing person.
Here is the recent conversation transcript:
---
{conversation}
---

Analyze this and output ONLY valid JSON in this exact structure:
{{
  "summary": "1-2 sentence simple clear summary of what was said",
  "tone": "one word: urgent, angry, happy, sad, confused, calm, neutral, excited, or worried",
  "key_points": ["key takeaway 1", "key takeaway 2"],
  "is_urgent": true or false
}}"""

    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            client = get_client()
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=250,
                ),
            )
            parsed = _parse_gemini_json(response.text)
            tone = parsed.get("tone", "neutral").lower()

            _last_result = {
                "summary": parsed.get("summary", ""),
                "tone": tone,
                "tone_emoji": TONE_EMOJI.get(tone, "😐"),
                "key_points": parsed.get("key_points", []),
                "is_urgent": bool(parsed.get("is_urgent", False)),
                "transcription": transcript_text or "",
            }
            _on_success()
            print(f"🎙️ Hearing AI: {_last_result['tone']} — {_last_result['summary'][:60]}...")
            return _last_result

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err or "503" in err:
                _on_rate_limit()
            else:
                print(f"❌ Hearing AI error with {model_name}: {e}")

    return _last_result


def process_audio_chunk(audio_bytes: bytes, mime_type: str = "audio/webm") -> dict:
    """
    1. Transcribes audio using local faster-whisper.
    2. Sends transcription to Gemini for summary & tone analysis.
    """
    global _last_result

    if not audio_bytes or len(audio_bytes) < 500:
        return _last_result

    # Step 1: Transcribe with local Whisper
    t0 = time.time()
    transcription = transcribe_audio_bytes(audio_bytes)
    t_whisper = round((time.time() - t0) * 1000)

    if not transcription:
        print(f"🎙️ Whisper ({t_whisper}ms): No speech detected")
        return {
            "summary": "No speech detected in this audio segment.",
            "tone": "neutral",
            "tone_emoji": "😐",
            "key_points": [],
            "is_urgent": False,
            "transcription": ""
        }

    print(f"🎙️ Whisper ({t_whisper}ms): \"{transcription}\"")

    # Step 2: Analyze with Gemini
    result = analyze_conversation(transcription)
    result["transcription"] = transcription
    return result


def reset_conversation():
    global _conversation_buffer, _last_result
    _conversation_buffer = []
    _last_result = {
        "summary": "",
        "tone": "neutral",
        "tone_emoji": "😐",
        "key_points": [],
        "is_urgent": False,
        "transcription": ""
    }
    print("🔄 Hearing conversation reset")
