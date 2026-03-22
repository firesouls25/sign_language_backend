from gtts import gTTS
from io import BytesIO
import base64
from typing import Optional
import logging
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self, lang: str = "es"):
        self.lang = lang

    async def text_to_speech(self, text: str) -> Optional[str]:
        if not text:
            return None
        
        try:
            tts = gTTS(text=text, lang=self.lang)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_data = audio_buffer.getvalue()
            
            # Save to storage
            storage = get_storage_service()
            filename = f"tts_{text[:10].replace(' ', '_')}.mp3"
            audio_url = await storage.upload_file(audio_data, filename, "audio/mpeg")
            
            return audio_url
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None


tts_service = None


def get_tts_service() -> TTSService:
    global tts_service
    if tts_service is None:
        tts_service = TTSService()
    return tts_service
