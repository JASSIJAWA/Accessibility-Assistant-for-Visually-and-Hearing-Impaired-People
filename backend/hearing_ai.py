"""
Hearing AI Module — Conversation analysis for hearing impaired users.

Features:
  - Receives speech transcript text from browser Web Speech API
  - Sends to Gemini for conversation summarization
  - Detects conversation tone (urgent, angry, calm, happy, sad, confused)
  - Returns summary + tone + key points
"""

import os
import time
from google import genai
from google.genai import types

# ─── Gemini client ───────────────────────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        _client = genai.Client(api_key=api_key)
    return _client


# ─── Rate limiting ────────────────────────────────────────────────────────────
_last_call_time = 0
_min_interval = 10.0   # minimum seconds between Gemini calls
_current_interval = 10.0
_MAX_INTERVAL = 60.0

def _should_call_gemini() -> bool:
    global _last_call_time
    return (time.time() - _last_call_time) >= _current_interval

def _on_success():
    global _current_interval, _last_call_time
    _current_interval = _min_interval
    _last_call_time = time.time()

def _on_rate_limit():
    global _current_interval, _last_call_time
    _current_interval = min(_current_interval * 2, _MAX_INTERVAL)
    _last_call_time = time.time()
    print(f"⚠️ Hearing AI rate limited — backing off to {_current_interval:.0f}s")


# ─── Conversation history buffer ──────────────────────────────────────────────
_conversation_buffer = []
_MAX_BUFFER_LINES = 30   # keep last 30 lines of transcript

def add_transcript_line(text: str):
    """Add a new transcript line to the rolling buffer."""
    global _conversation_buffer
    if text.strip():
        _conversation_buffer.append(text.strip())
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
    "is_urgent": False
}

def get_last_result() -> dict:
    return _last_result


# ─── Tone emoji map ───────────────────────────────────────────────────────────
TONE_EMOJI = {
    "urgent": "🚨",
    "angry": "😠",
    "happy": "😊",
    "sad": "😢",
    "confused": "😕",
    "calm": "😐",
    "neutral": "😐",
    "excited": "😃",
    "worried": "😟",
}


# ─── Main analysis function ───────────────────────────────────────────────────
def analyze_conversation(transcript_text: str = None) -> dict:
    """
    Analyze the conversation transcript using Gemini.
    Returns summary, tone, key_points, is_urgent.
    Uses cached result if rate limited.
    """
    global _last_result

    # Add new text to buffer if provided
    if transcript_text:
        add_transcript_line(transcript_text)

    conversation = get_conversation_text()
    if not conversation.strip():
        return _last_result

    # Rate limit check
    if not _should_call_gemini():
        return _last_result

    prompt = f"""You are an AI assistant helping a hearing-impaired person understand a conversation happening around them.

Here is the recent conversation transcript:
---
{conversation}
---

Analyze this conversation and respond in this exact JSON format (no markdown, just JSON):
{{
  "summary": "One or two sentence plain-language summary of what is being discussed",
  "tone": "one word from: urgent, angry, happy, sad, confused, calm, neutral, excited, worried",
  "key_points": ["point 1", "point 2", "point 3"],
  "is_urgent": true or false
}}

Rules:
- summary must be short and simple, written for someone who cannot hear
- tone must be exactly one word from the list above
- key_points must be 1-3 bullet points of the most important things said
- is_urgent is true if someone sounds distressed, is asking for help, or the conversation is an emergency
- Do not add any text outside the JSON"""

    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        import json
        parsed = json.loads(raw)

        tone = parsed.get("tone", "neutral").lower()
        _last_result = {
            "summary": parsed.get("summary", ""),
            "tone": tone,
            "tone_emoji": TONE_EMOJI.get(tone, "😐"),
            "key_points": parsed.get("key_points", []),
            "is_urgent": parsed.get("is_urgent", False),
        }
        _on_success()
        print(f"🎙️ Hearing AI: {_last_result['tone']} — {_last_result['summary'][:60]}...")
        return _last_result

    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err or "rate" in err:
            _on_rate_limit()
        else:
            print(f"❌ Hearing AI error: {e}")
        return _last_result


def reset_conversation():
    """Clear the conversation buffer (e.g., when user starts a new session)."""
    global _conversation_buffer, _last_result
    _conversation_buffer = []
    _last_result = {
        "summary": "",
        "tone": "neutral",
        "tone_emoji": "😐",
        "key_points": [],
        "is_urgent": False
    }
    print("🔄 Conversation buffer cleared")
