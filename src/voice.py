"""
Voice layer: Gemini 2.5 Flash for STT + TTS.

Three stage pipeline:
  audio in -> STT (Gemini 2.5 Flash, audio input) -> transcript
  transcript -> existing LangGraph (Orchestrator + agents + Synthesizer)
  final_answer -> TTS (Gemini 2.5 Flash preview TTS, Kore voice) -> audio out

Same GOOGLE_API_KEY as the rest of the app. Batch only, no streaming,
because mic input is capped at 5 seconds.
"""

from __future__ import annotations

import time

import numpy as np
from google import genai
from google.genai import types

from .config import GOOGLE_API_KEY, logger


_client = genai.Client(api_key=GOOGLE_API_KEY)

STT_MODEL = "gemini-2.5-flash"
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_VOICE = "Kore"  # firm, confident tone suited for a financial assistant
MAX_TTS_CHARS = 1000  # keeps TTS calls fast and inside preview quotas
TTS_SAMPLE_RATE = 24000  # Gemini TTS returns 24kHz PCM
TTS_MAX_ATTEMPTS = 3  # Gemini preview TTS occasionally 500s; a retry usually clears it


def transcribe(wav_bytes: bytes, mime: str = "audio/wav") -> str:
    """Transcribe a short audio clip with Gemini 2.5 Flash.

    Pass raw audio bytes plus a transcription instruction. Returns a plain
    string with leading/trailing whitespace trimmed.
    """
    try:
        response = _client.models.generate_content(
            model=STT_MODEL,
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type=mime),
                "Transcribe this audio verbatim. Output only the transcript, "
                "no preamble, no quotes, no commentary.",
            ],
        )
        return (response.text or "").strip()
    except Exception:
        logger.exception("STT failed")
        raise


def synthesize(text: str, voice: str = TTS_VOICE) -> tuple[int, np.ndarray]:
    """Synthesize speech with Gemini TTS. Returns (sample_rate, int16 numpy array).

    Gemini TTS emits 24kHz mono PCM16. Gradio's gr.Audio with type="numpy"
    expects exactly this tuple shape.
    """
    text = (text or "").strip()[:MAX_TTS_CHARS]
    if not text:
        return TTS_SAMPLE_RATE, np.zeros(1, dtype=np.int16)

    styled = (
        "Say in a calm, professional, slightly warm tone suitable for a "
        f"financial assistant: {text}"
    )

    # Gemini's preview TTS endpoint is under capacity pressure and occasionally
    # returns 500 / 503 / 429 on otherwise valid requests. A quick retry with
    # exponential backoff (1s, 2s) clears the vast majority. Non transient
    # errors (400 bad request, auth, etc.) bail out immediately.
    last_err: Exception | None = None
    response = None
    for attempt in range(TTS_MAX_ATTEMPTS):
        try:
            response = _client.models.generate_content(
                model=TTS_MODEL,
                contents=styled,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    ),
                ),
            )
            break
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            transient = (
                "500" in msg
                or "internal" in msg
                or "503" in msg
                or "unavailable" in msg
                or "429" in msg
                or "rate" in msg
                or "too many" in msg
            )
            if transient and attempt < TTS_MAX_ATTEMPTS - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"TTS transient error (attempt {attempt + 1}/{TTS_MAX_ATTEMPTS}), "
                    f"retrying in {wait}s: {exc}"
                )
                time.sleep(wait)
                continue
            logger.exception("TTS failed")
            raise

    if response is None:
        raise RuntimeError(f"TTS failed after {TTS_MAX_ATTEMPTS} attempts: {last_err}")

    part = response.candidates[0].content.parts[0]
    pcm_bytes = part.inline_data.data
    # 16-bit signed PCM, little-endian, mono
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    return TTS_SAMPLE_RATE, audio
