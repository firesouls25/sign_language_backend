from gtts import gTTS
from io import BytesIO
import base64
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self, lang: str = "es"):
        self.lang = lang

    def text_to_speech(self, text: str) -> Optional[str]:
        if not text:
            return None
        
        try:
            tts = gTTS(text=text, lang=self.lang)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
            return f"data:audio/mp3;base64,{audio_base64}"
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None


tts_service = None


def get_tts_service() -> TTSService:
    global tts_service
    if tts_service is None:
        tts_service = TTSService()
    return tts_service
