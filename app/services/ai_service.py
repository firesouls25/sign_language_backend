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

    async def process_landmarks(
        self, landmarks: dict, user_id: Optional[str] = None
    ) -> Dict:
        """Procesar landmarks directamente sin frames"""
        try:
            from ml.processor import MediapipeResults

            logger.info(f"[AIService] Processing landmarks request")
            logger.info(f"[AIService] Landmarks keys: {list(landmarks.keys())}")

            left_landmarks_data = landmarks.get("left_hand")
            right_landmarks_data = landmarks.get("right_hand")

            logger.info(
                f"[AIService] Left hand: {len(left_landmarks_data) if left_landmarks_data else 0} points"
            )
            logger.info(
                f"[AIService] Right hand: {len(right_landmarks_data) if right_landmarks_data else 0} points"
            )

            logger.info(
                f"[AIService] Recognizer initialized: {self.recognizer._initialized}"
            )
            logger.info(
                f"[AIService] Recorder available: {self.recognizer.recorder is not None}"
            )

            left_landmarks = (
                self._create_landmark_points(left_landmarks_data)
                if left_landmarks_data
                else None
            )
            right_landmarks = (
                self._create_landmark_points(right_landmarks_data)
                if right_landmarks_data
                else None
            )

            results = MediapipeResults(
                left_landmarks=left_landmarks,
                right_landmarks=right_landmarks,
            )

            result = self.recognizer.process_landmarks_data(results)
            text = result.get("text", "")
            confidence = result.get("confidence", 0.0)

            logger.info(
                f"[AIService] Recognition complete: text='{text}', confidence={confidence:.2f}"
            )
            logger.info(f"[AIService] Full result: {result}")

            logger.info(
                f"[AIService] Recognized text: '{text}', confidence: {result.get('confidence', 0.0)}"
            )

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

        except Exception as e:
            import traceback

            logger.error(f"Error processing landmarks: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"type": "error", "message": str(e), "code": "LANDMARKS_ERROR"}

    def _create_landmark_points(self, landmarks_data):
        """Crear objetos de landmark simulando salida de MediaPipe"""
        if not landmarks_data:
            return None

        class LandmarkPoint:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        return [LandmarkPoint(p[0], p[1], p[2]) for p in landmarks_data]

    def reset(self):
        self.recognizer.reset_sequence()
        self.last_audio_text = ""


ai_service = None


def get_ai_service() -> AIService:
    global ai_service
    if ai_service is None:
        ai_service = AIService()
    return ai_service
