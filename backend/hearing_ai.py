"""
Hearing AI Module — Conversation analysis for hearing impaired users.

Features:
  - Receives speech transcript text OR raw audio recordings
  - Transcribes audio using Gemini
  - Generates plain-language conversation summary
  - Detects conversation tone (urgent, angry, calm, happy, sad, etc.)
  - Extracts key takeaways
  - Flags urgent/emergency situations
"""

import os
import time
import json
import base64
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

# Primary model (fast, reliable, high rate limit)
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
            raise RuntimeError("GEMINI_API_KEY not set in environment or backend/.env")
        _client = genai.Client(api_key=api_key)
    return _client


# ─── Rate limiting & backoff ──────────────────────────────────────────────────
_last_call_time = 0.0
_min_interval = 4.0   # minimum seconds between calls
_current_interval = 4.0
_MAX_INTERVAL = 45.0

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
    print(f"⚠️ Hearing AI rate limited — backing off to {_current_interval:.0f}s")


# ─── Conversation history buffer ──────────────────────────────────────────────
_conversation_buffer = []
_MAX_BUFFER_LINES = 25

def add_transcript_line(text: str):
    """Add a new transcript line to the rolling buffer."""
    global _conversation_buffer
    text = text.strip()
    if text:
        _conversation_buffer.append(text)
        if len(_conversation_buffer) > _MAX_BUFFER_LINES:
            _conversation_buffer = _conversation_buffer[-_MAX_BUFFER_LINES:]


def get_conversation_text() -> str:
    return "\n".join(_conversation_buffer)


# ─── Last result cache ────────────────────────────────────────────────────────
_last_result = {
    "summary": "",
    "tone": "neutral",
    "tone_emoji": "😐",
    "key_points": [],
    "is_urgent": False,
    "transcription": ""
}

def get_last_result() -> dict:
    return _last_result


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
    
    # In case there's any surrounding text
    start_idx = raw.find("{")
    end_idx = raw.rfind("}")
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx+1]
    
    return json.loads(raw)


def analyze_conversation(transcript_text: str = None) -> dict:
    """
    Analyze conversation transcript using Gemini.
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
Here is what has been spoken recently:
---
{conversation}
---

Analyze this and output ONLY valid JSON in this exact structure:
{{
  "summary": "1-2 sentence simple clear summary of what was said",
  "tone": "one word: urgent, angry, happy, sad, confused, calm, neutral, excited, or worried",
  "key_points": ["bullet 1", "bullet 2"],
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
            print(f"🎙️ Hearing AI ({model_name}): {_last_result['tone']} — {_last_result['summary'][:60]}...")
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
    Transcribes raw audio from user's microphone, updates conversation buffer,
    and returns full analysis.
    """
    global _last_result

    if not audio_bytes or len(audio_bytes) < 100:
        return _last_result

    prompt = """You are an assistive AI for a deaf person.
Listen carefully to this audio recording:
1. Transcribe the spoken speech accurately. If no speech is heard, return empty string for transcription.
2. Provide a 1-2 sentence simple summary of what the speaker said.
3. Detect the emotional tone (urgent, angry, happy, sad, confused, calm, neutral, excited, worried).
4. Extract 1-3 key points.
5. Set is_urgent to true if someone is warning of danger, shouting, or needs immediate attention.

Return ONLY JSON:
{
  "transcription": "exact spoken words",
  "summary": "plain-language summary",
  "tone": "one word",
  "key_points": ["point 1"],
  "is_urgent": false
}"""

    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            client = get_client()
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, audio_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=300,
                ),
            )
            parsed = _parse_gemini_json(response.text)
            transcription = parsed.get("transcription", "").strip()

            if transcription:
                add_transcript_line(transcription)

            tone = parsed.get("tone", "neutral").lower()
            _last_result = {
                "summary": parsed.get("summary", ""),
                "tone": tone,
                "tone_emoji": TONE_EMOJI.get(tone, "😐"),
                "key_points": parsed.get("key_points", []),
                "is_urgent": bool(parsed.get("is_urgent", False)),
                "transcription": transcription,
            }
            _on_success()
            print(f"🎙️ Audio transcribed ({model_name}): \"{transcription}\" | {tone}")
            return _last_result

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err or "503" in err:
                _on_rate_limit()
            else:
                print(f"❌ Audio transcription error with {model_name}: {e}")

    return _last_result


def reset_conversation():
    """Clear the conversation buffer."""
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
    print("🔄 Hearing conversation buffer reset")
