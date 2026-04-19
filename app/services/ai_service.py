from typing import Dict, Optional
import logging
from ml.sign_detector_manager import get_sign_detector_manager
from ml.text_normalizer import get_text_normalizer
from app.services.tts_service import get_tts_service

logger = logging.getLogger(__name__)


class AIService:
    def __init__(self):
        self.detector = get_sign_detector_manager()
        self.normalizer = get_text_normalizer()
        self.tts = get_tts_service()
        self.last_audio_text = ""
        self.current_mode = "handshape"
        self.last_context = ""
        logger.info("[AIService] Initialized with new SignDetectorManager")

    def set_mode(self, mode: str):
        """Set the recognition mode"""
        if mode not in ["handshape", "fingerspelling"]:
            logger.warning(
                f"[AIService] Unknown mode: {mode}, defaulting to 'handshape'"
            )
            mode = "handshape"

        self.current_mode = mode
        self.detector.set_mode(mode)
        logger.info(f"[AIService] Mode set to: {mode}")

    async def process_landmarks(
        self,
        landmarks: dict,
        mode: str,
        context: str = "",
        user_id: Optional[str] = None,
    ) -> Dict:
        """Process landmarks with mode selection and text normalization"""
        try:
            if mode != self.current_mode:
                self.set_mode(mode)

            logger.info(f"[AIService] Processing landmarks, mode: {mode}")

            left_landmarks_data = landmarks.get("left_hand")
            right_landmarks_data = landmarks.get("right_hand")

            logger.info(
                f"[AIService] Left hand: {len(left_landmarks_data) if left_landmarks_data else 0} points"
            )
            logger.info(
                f"[AIService] Right hand: {len(right_landmarks_data) if right_landmarks_data else 0} points"
            )

            left_landmarks = left_landmarks_data if left_landmarks_data else None
            right_landmarks = right_landmarks_data if right_landmarks_data else None

            raw_result = self.detector.process_landmarks(
                left_landmarks, right_landmarks
            )

            raw_text = raw_result.get("text", "")
            candidate = raw_result.get("candidate", "")
            is_recording = raw_result.get("is_recording", False)

            if raw_text:
                raw_text = raw_text.strip()

            normalized_text = raw_text
            if raw_text and len(raw_text) > 0:
                normalized_text = self.normalizer.normalize(
                    raw_text, mode, self.last_context
                )

                if (
                    normalized_text != "[entrada no reconocida]"
                    and not normalized_text.startswith("[error")
                ):
                    self.last_context = normalized_text

            logger.info(
                f"[AIService] Recognition: raw='{raw_text}', normalized='{normalized_text}', mode={mode}"
            )

            audio_data = None
            if (
                normalized_text
                and normalized_text != "[entrada no reconocida]"
                and normalized_text != self.last_audio_text
            ):
                if not normalized_text.startswith("[error"):
                    audio_data = self.tts.text_to_speech(normalized_text)
                    self.last_audio_text = normalized_text
            elif not normalized_text or normalized_text == "[entrada no reconocida]":
                self.last_audio_text = ""

            response = {
                "type": "translation",
                "text": normalized_text
                if not normalized_text.startswith("[error")
                else "",
                "confidence": raw_result.get("confidence", 0.0),
                "has_keypoints": raw_text is not None and len(raw_text) > 0,
                "phrase": normalized_text
                if normalized_text and not normalized_text.startswith("[error")
                else "",
                "is_recording": is_recording,
                "candidate": candidate,
                "candidate_confidence": raw_result.get("confidence", 0.0),
                "mode": mode,
            }

            if audio_data:
                response["audio"] = audio_data

            if normalized_text and user_id and not normalized_text.startswith("[error"):
                response["sign_detected"] = True

            return response

        except Exception as e:
            import traceback

            logger.error(f"[AIService] Error processing landmarks: {e}")
            logger.error(f"[AIService] Traceback: {traceback.format_exc()}")
            return {"type": "error", "message": str(e), "code": "LANDMARKS_ERROR"}

    def reset(self):
        self.detector.reset_sequence()
        self.last_audio_text = ""
        self.last_context = ""


ai_service = None


def get_ai_service() -> AIService:
    global ai_service
    if ai_service is None:
        ai_service = AIService()
    return ai_service
