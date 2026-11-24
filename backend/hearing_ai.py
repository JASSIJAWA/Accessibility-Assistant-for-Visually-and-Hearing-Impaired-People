"""
Hearing AI Module — Powered by Gemini Multimodal Audio Intelligence + Local Faster-Whisper Fallback.

Capabilities:
  1. Ultra-high-accuracy speech-to-text (transcribes verbatim, handles accents, slang, noise)
  2. Acoustic event detection (car horns, sirens, alarms, screaming in brackets [like this])
  3. Emotional tone classification (urgent, angry, happy, sad, calm, etc.)
  4. Real-time cognitive summary & key takeaways for hearing-impaired users
  5. Emergency / distress detection with instant visual & haptic alerts
  6. Robust local fallback to Faster-Whisper base.en if offline
"""

import os
import io
import time
import json
import logging
import base64
import wave
from dotenv import load_dotenv

logger = logging.getLogger("assistive-backend")

load_dotenv()

from google import genai
from google.genai import types

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


# Lazy load Faster-Whisper model for fallback
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("Loading Faster-Whisper base.en model on CPU (fallback)...")
        _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        logger.info("Faster-Whisper base.en ready as fallback engine")
    return _whisper_model


# ─── Audio Format Converter (Ensures 16kHz mono WAV) ────────────────────────
def convert_to_wav(audio_bytes: bytes) -> bytes:
    """
    Ensures audio is in 16kHz mono 16-bit PCM WAV.
    If already WAV, returns as-is.
    If WebM, MP4, OGG, or AAC, converts using PyAV.
    """
    if not audio_bytes or len(audio_bytes) < 44:
        return audio_bytes

    # Check if already a valid WAV
    if audio_bytes[:4] == b"RIFF" and b"WAVE" in audio_bytes[:16]:
        return audio_bytes

    try:
        import av
        input_buf = io.BytesIO(audio_bytes)
        container = av.open(input_buf)
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        output_buf = io.BytesIO()
        with wave.open(output_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    wf.writeframes(rf.to_ndarray().tobytes())
        result = output_buf.getvalue()
        if len(result) > 44:
            return result
    except Exception as e:
        logger.debug("Audio converter notice: %s", e)

    # Conversion failed — return original bytes (may cause Gemini parsing issues)
    logger.warning("Audio conversion to WAV failed; sending raw audio to Gemini")
    return audio_bytes


# ─── Rate Limiting & Rolling Buffer ──────────────────────────────────────────
_last_call_time = 0.0
_min_interval = 1.5
_current_interval = 1.5
_MAX_INTERVAL = 20.0

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

_last_result = {
    "summary": "",
    "tone": "neutral",
    "tone_emoji": "😐",
    "key_points": [],
    "is_urgent": False,
    "transcription": ""
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


# ─── Primary Engine: Gemini Multimodal Audio Perception ───────────────────────
GEMINI_AUDIO_PROMPT = """You are the audio and speech perception intelligence of an assistive AI for deaf and hard-of-hearing people.
Listen to this audio recording carefully and provide:
1. "transcription": Exact verbatim transcript of all spoken words. Accurately capture Indian and global accents, casual speech, questions, and instructions. If there are important environmental sounds (e.g. car honking, siren wailing, glass breaking, dog barking, alarm beeping, screams for help), include them clearly in square brackets like [Car horn honking] or [Siren wailing]. If no words or meaningful sounds are heard, return "".
2. "summary": A concise 1-2 sentence summary of what was communicated or heard for the deaf user.
3. "tone": The speaker's primary emotion or tone. Choose one: urgent, angry, happy, sad, confused, calm, neutral, excited, or worried.
4. "key_points": A list of 1 to 3 concise takeaway bullet points.
5. "is_urgent": true if this audio indicates danger, warning, emergency, accident, siren, screams for help, or aggressive hostility; false otherwise.

Output ONLY valid JSON in this exact structure:
{
  "transcription": "verbatim text or sound description",
  "summary": "1-2 sentence clear summary",
  "tone": "urgent/angry/happy/sad/confused/calm/neutral/excited/worried",
  "key_points": ["point 1", "point 2"],
  "is_urgent": true or false
}"""


def transcribe_and_analyze_with_gemini(audio_bytes: bytes, mime_type: str = "audio/wav") -> dict:
    """
    Sends raw audio directly to Gemini Multimodal Audio model.
    Transcribes words + environmental sounds AND performs cognitive analysis in one fast call.
    """
    global _last_result

    client = get_client()
    wav_bytes = convert_to_wav(audio_bytes)

    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            t0 = time.time()
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    GEMINI_AUDIO_PROMPT
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=350,
                ),
            )
            dt = round((time.time() - t0) * 1000)
            parsed = _parse_gemini_json(response.text)

            transcription = parsed.get("transcription", "").strip()
            summary = parsed.get("summary", "").strip()
            tone = parsed.get("tone", "neutral").lower()
            key_points = parsed.get("key_points", [])
            is_urgent = bool(parsed.get("is_urgent", False))

            if transcription:
                add_transcript_line(transcription)

            _last_result = {
                "transcription": transcription,
                "summary": summary,
                "tone": tone,
                "tone_emoji": TONE_EMOJI.get(tone, "😐"),
                "key_points": key_points,
                "is_urgent": is_urgent,
            }
            _on_success()
            logger.info("Gemini Audio (%dms) [%s]: \"%s\" | %s | Urgent=%s",
                        dt, model_name, transcription, tone, is_urgent)
            return _last_result

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err or "503" in err:
                _on_rate_limit()
            logger.warning("Gemini audio failed on %s: %s", model_name, e)

    return None


# ─── Fallback Engine: Local Faster-Whisper base.en ────────────────────────────
def transcribe_audio_whisper_fallback(audio_bytes: bytes) -> str:
    """Fallback local transcription if Gemini API is unreachable or rate limited."""
    try:
        whisper = get_whisper()
        wav_bytes = convert_to_wav(audio_bytes)
        buf = io.BytesIO(wav_bytes)
        segments, _ = whisper.transcribe(
            buf,
            beam_size=5,
            best_of=5,
            language="en",
            temperature=[0.0, 0.2],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            initial_prompt="A clear spoken conversation, questions, directions, and emergencies."
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        logger.error("Whisper fallback error: %s", e)
        return ""


# ─── Main Audio Processing Entry Point ───────────────────────────────────────
def process_audio_chunk(audio_bytes: bytes, mime_type: str = "audio/webm") -> dict:
    """
    Main audio handler for incoming microphone recordings.
    1. Attempts Gemini Multimodal Audio (fast, ultra-accurate, tone+summary included).
    2. Falls back to Faster-Whisper base.en if Gemini is offline/rate-limited.
    """
    global _last_result

    if not audio_bytes or len(audio_bytes) < 400:
        return _last_result

    # 1. Primary: Direct Gemini Audio Understanding
    result = transcribe_and_analyze_with_gemini(audio_bytes, mime_type)
    if result is not None and (result.get("transcription") or result.get("summary")):
        return result

    # 2. Fallback: Local Whisper + Separate Gemini Text Analysis
    logger.info("Falling back to local Faster-Whisper base.en...")
    t0 = time.time()
    transcription = transcribe_audio_whisper_fallback(audio_bytes)
    t_whisper = round((time.time() - t0) * 1000)

    if not transcription:
        logger.info("Whisper (%dms): No speech detected", t_whisper)
        return {
            "summary": "No speech detected in this audio segment.",
            "tone": "neutral",
            "tone_emoji": "😐",
            "key_points": [],
            "is_urgent": False,
            "transcription": ""
        }

    logger.info("Whisper (%dms): \"%s\"", t_whisper, transcription)
    result = analyze_conversation(transcription)
    result["transcription"] = transcription
    return result


# ─── Text-Based Cognitive Analysis (For Simulation buttons & Web Speech) ─────
def analyze_conversation(transcript_text: str = None) -> dict:
    """
    Analyze conversation text using Gemini for tone, summary, and urgency.
    Called when text is received from simulation buttons, typed input, or Web Speech API.
    """
    global _last_result

    if transcript_text:
        add_transcript_line(transcript_text)

    conversation = get_conversation_text()
    if not conversation.strip():
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
            logger.info("Hearing Text AI: %s — %s", _last_result['tone'], _last_result['summary'][:60])
            return _last_result

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err or "503" in err:
                _on_rate_limit()
            else:
                logger.error("Hearing AI error with %s: %s", model_name, e)

    # Offline heuristic fallback
    urgent_words = ["danger", "watch out", "fire", "car", "stop", "help", "emergency", "run", "careful", "hospital"]
    lower_conv = conversation.lower()
    is_urgent = any(w in lower_conv for w in urgent_words)
    _last_result = {
        "summary": transcript_text or "Conversation in progress.",
        "tone": "urgent" if is_urgent else "neutral",
        "tone_emoji": "🚨" if is_urgent else "😐",
        "key_points": [transcript_text] if transcript_text else [],
        "is_urgent": is_urgent,
        "transcription": transcript_text or "",
    }
    return _last_result


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
    logger.info("Hearing conversation reset")
