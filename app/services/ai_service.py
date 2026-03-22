import numpy as np
import cv2
import base64
from typing import Dict
import logging
from ml.processor import get_sign_recognizer
from app.services.tts_service import get_tts_service
from app.utils.redis_client import redis_client

logger = logging.getLogger(__name__)


class AIService:
    def __init__(self):
        self.recognizer = get_sign_recognizer()
        self.tts = get_tts_service()
        self.last_audio_text = ""
        logger.info("AIService initialized with TTS")

    async def process_frame(self, frame_data: str) -> Dict:
        try:
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {
                    "type": "error",
                    "message": "Could not decode frame",
                    "code": "DECODE_ERROR"
                }
            
            result = self.recognizer.process_frame(img)
            text = result.get("text", "")
            
            audio_data = None
            # Only generate audio if text is not empty and different from last one
            if text and text != self.last_audio_text:
                # 1. Check Cache first
                cache_key = f"tts:cache:es:{text}"
                cached_url = await redis_client.get_value(cache_key)
                
                if cached_url:
                    audio_data = cached_url
                    logger.info(f"TTS Cache hit for: {text}")
                else:
                    # 2. Generate new audio
                    audio_data = await self.tts.text_to_speech(text)
                    if audio_data:
                        # 3. Save to Cache (1 week expiration)
                        await redis_client.set_value(cache_key, audio_data, 604800)
                
                self.last_audio_text = text
            elif not text:
                self.last_audio_text = ""

            return {
                "type": "translation",
                "text": text,
                "confidence": result.get("confidence", 0.0),
                "has_keypoints": result.get("keypoints") is not None,
                "audio": audio_data
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                "type": "error",
                "message": str(e),
                "code": "PROCESSING_ERROR"
            }

    def reset(self):
        self.recognizer.reset_sequence()


ai_service = None


def get_ai_service() -> AIService:
    global ai_service
    if ai_service is None:
        ai_service = AIService()
    return ai_service
