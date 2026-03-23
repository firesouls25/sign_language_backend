import numpy as np
import cv2
import base64
from typing import Dict, Optional
import logging
from ml.processor import get_sign_recognizer
from app.services.tts_service import get_tts_service
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)


class AIService:
    def __init__(self):
        self.recognizer = get_sign_recognizer()
        self.tts = get_tts_service()
        self.storage = get_storage_service()
        self.last_audio_text = ""
        logger.info("AIService initialized")

    async def process_frame(
        self, frame_data: str, user_id: Optional[str] = None
    ) -> Dict:
        try:
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {
                    "type": "error",
                    "message": "Could not decode frame",
                    "code": "DECODE_ERROR",
                }

            return self._process_image(img, user_id)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {"type": "error", "message": str(e), "code": "PROCESSING_ERROR"}

    async def process_frame_binary(
        self, frame_bytes: bytes, user_id: Optional[str] = None
    ) -> Dict:
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {
                    "type": "error",
                    "message": "Could not decode frame",
                    "code": "DECODE_ERROR",
                }

            return self._process_image(img, user_id)

        except Exception as e:
            logger.error(f"Error processing binary frame: {e}")
            return {"type": "error", "message": str(e), "code": "PROCESSING_ERROR"}

    def _process_image(self, img: np.ndarray, user_id: Optional[str] = None) -> Dict:
        result = self.recognizer.process_frame(img)
        text = result.get("text", "")

        audio_data = None
        if text and text != self.last_audio_text:
            audio_data = self.tts.text_to_speech(text)
            self.last_audio_text = text
        elif not text:
            self.last_audio_text = ""

        response = {
            "type": "translation",
            "text": text,
            "confidence": result.get("confidence", 0.0),
            "has_keypoints": result.get("keypoints") is not None,
            "phrase": result.get("phrase", ""),
            "is_recording": result.get("is_recording", False),
            "candidate": result.get("candidate", ""),
            "candidate_confidence": result.get("candidate_confidence", 0.0),
        }

        if audio_data:
            response["audio"] = audio_data

        if text and user_id:
            response["sign_detected"] = True

        return response

    def reset(self):
        self.recognizer.reset_sequence()
        self.last_audio_text = ""


ai_service = None


def get_ai_service() -> AIService:
    global ai_service
    if ai_service is None:
        ai_service = AIService()
    return ai_service
