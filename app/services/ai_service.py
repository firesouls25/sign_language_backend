import numpy as np
import cv2
import base64
from typing import Dict
import logging
from ml.processor import get_sign_recognizer
from app.services.tts_service import get_tts_service

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
                audio_data = self.tts.text_to_speech(text)
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
