import asyncio
import aiohttp
from edge_tts import Communicate
from io import BytesIO
from typing import Optional
import logging
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)

VOICES = {
    "es": "es-ES-AlvaroNeural",
    "en": "en-US-JennyNeural",
}

AUDIO_FORMATS = {
    "audio-24khz-48kbitrate-mono-mp3": "audio/mpeg",
    "audio-24khz-96kbitrate-mono-mp3": "audio/mpeg",
    "audio-16khz-128kbitrate-mono-mp3": "audio/mpeg",
}


class TTSService:
    def __init__(
        self, lang: str = "es", voice_format: str = "audio-24khz-48kbitrate-mono-mp3"
    ):
        self.lang = lang
        self.voice = VOICES.get(lang, VOICES["es"])
        self.voice_format = voice_format

    async def text_to_speech(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            logger.warning("[TTS] Empty text, skipping")
            return None

        try:
            text = text.strip()
            logger.info(f"[TTS] Using voice: {self.voice}")
            logger.info(f"[TTS] Generating audio for: '{text}'")

            communicate = Communicate(text, self.voice)
            audio_buffer = BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

            audio_data = audio_buffer.getvalue()
            logger.info(f"[TTS] Audio data length: {len(audio_data)} bytes")

            if not audio_data or len(audio_data) == 0:
                logger.error("Edge TTS returned empty audio data")
                return None

            storage = get_storage_service()
            filename = f"tts_{text[:10].replace(' ', '_')}.mp3"
            audio_url = await storage.upload_file(
                audio_data, filename, AUDIO_FORMATS.get(self.voice_format, "audio/mpeg")
            )

            logger.info(f"[TTS] Audio URL: {audio_url}")
            return audio_url

        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            import traceback

            logger.error(f"[TTS] Traceback: {traceback.format_exc()}")
            return None

        try:
            communicate = Communicate(text, self.voice)
            audio_buffer = BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

            audio_data = audio_buffer.getvalue()

            if not audio_data:
                logger.error("Edge TTS returned empty audio data")
                return None

            storage = get_storage_service()
            filename = f"tts_{text[:10].replace(' ', '_')}.mp3"
            audio_url = await storage.upload_file(
                audio_data, filename, AUDIO_FORMATS.get(self.voice_format, "audio/mpeg")
            )

            return audio_url

        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return None


tts_service = None


def get_tts_service() -> TTSService:
    global tts_service
    if tts_service is None:
        tts_service = TTSService()
    return tts_service
