from typing import Dict, Optional
import logging
import numpy as np
from ml.sign_detector_manager import get_sign_detector_manager
from ml.text_normalizer import get_text_normalizer
from ml.processor import get_keypoint_extractor
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
            logger.info(
                f"[AIService] left_hand type: {type(left_landmarks_data)}, is list: {isinstance(left_landmarks_data, list)}"
            )
            logger.info(
                f"[AIService] right_hand type: {type(right_landmarks_data)}, is list: {isinstance(right_landmarks_data, list)}"
            )

            # Validar datos directamente - NO aplicar espejo
            # Los landmarks del móvil ya vienen en formato MediaPipe estándar
            left_landmarks_data, right_landmarks_data = (
                left_landmarks_data,
                right_landmarks_data,
            )

            # Validate data format before passing to model
            left_landmarks = None
            right_landmarks = None

            if left_landmarks_data and isinstance(left_landmarks_data, list):
                if len(left_landmarks_data) >= 21:
                    left_landmarks = left_landmarks_data
                    logger.info(
                        f"[AIService] Valid left_hand with {len(left_landmarks)} points"
                    )
                else:
                    logger.warning(
                        f"[AIService] left_hand has only {len(left_landmarks_data)} points, expected 21"
                    )

            if right_landmarks_data and isinstance(right_landmarks_data, list):
                if len(right_landmarks_data) >= 21:
                    right_landmarks = right_landmarks_data
                    logger.info(
                        f"[AIService] Valid right_hand with {len(right_landmarks)} points"
                    )
                else:
                    logger.warning(
                        f"[AIService] right_hand has only {len(right_landmarks_data)} points, expected 21"
                    )

            # Show detector status
            detector = self.detector
            logger.info(f"[AIService] Detector mode: {detector.mode}")
            logger.info(
                f"[AIService] Detector has handshape: {detector._handshape_recognizer is not None}"
            )
            logger.info(
                f"[AIService] Detector has fingerspelling: {detector._fingerspelling_recognizer is not None}"
            )
            if (
                hasattr(detector, "_fingerspelling_recognizer")
                and detector._fingerspelling_recognizer
            ):
                logger.info(
                    f"[AIService] Fingerspelling is_loaded: {detector._fingerspelling_recognizer.is_loaded}"
                )

            raw_result = self.detector.process_landmarks(
                left_landmarks, right_landmarks
            )

            logger.warning(
                f"[AIService] raw_result keys: {raw_result.keys() if raw_result else 'None'}"
            )
            logger.warning(f"[AIService] raw_result: {raw_result}")

            raw_text = raw_result.get("text", "")
            candidate = raw_result.get("candidate", "")
            is_recording = raw_result.get("is_recording", False)
            sequence = raw_result.get("sequence", "")

            logger.warning(
                f"[AIService] fingerspell - raw_text='{raw_text}', candidate='{candidate}', sequence='{sequence}', is_recording={is_recording}"
            )

            if candidate:
                logger.warning(f"[AIService] LETTER DETECTED: {candidate}")

            if raw_text:
                raw_text = raw_text.strip()

            normalized_text = raw_text
            if raw_text and len(raw_text) > 0:
                if mode == "fingerspelling":
                    normalized_text = ""
                else:
                    normalized_text = await self.normalizer.normalize(
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
            # No generamos audio en backend - el frontend usa flutter_tts
            logger.info(f"[AIService] TTS skipped - frontend handles it")

            response = {
                "type": "translation",
                "text": raw_text,
                "audio": audio_data,
                "confidence": raw_result.get("confidence", 0.0),
                "has_keypoints": (
                    left_landmarks is not None or right_landmarks is not None
                ),
                "phrase": raw_text,
                "is_recording": is_recording,
                "candidate": candidate,
                "candidate_confidence": raw_result.get("candidate_confidence", 0.0),
                "mode": mode,
                "sequence": sequence,
            }

            return response

        except Exception as e:
            import traceback

            logger.error(f"[AIService] Error processing landmarks: {e}")
            logger.error(f"[AIService] Traceback: {traceback.format_exc()}")
            return {"type": "error", "message": str(e), "code": "LANDMARKS_ERROR"}

    async def process_frame(
        self,
        frame_data: list,
        width: int,
        height: int,
        mode: str,
        user_id: Optional[str] = None,
    ) -> Dict:
        """Process a raw frame by extracting landmarks then processing them."""
        try:
            if mode != self.current_mode:
                self.set_mode(mode)

            logger.info(f"[AIService] Processing frame, mode: {mode}")

            if not frame_data or width == 0 or height == 0:
                logger.warning("[AIService] Empty frame data received")
                return {
                    "type": "translation",
                    "text": "",
                    "confidence": 0.0,
                    "has_keypoints": False,
                    "phrase": "",
                    "is_recording": False,
                    "candidate": "",
                    "candidate_confidence": 0.0,
                    "mode": mode,
                    "sequence": "",
                }

            frame_array = np.array(frame_data, dtype=np.uint8)
            if len(frame_array) != width * height * 3:
                logger.error(
                    f"[AIService] Frame data size mismatch: expected {width * height * 3}, got {len(frame_array)}"
                )
                return {
                    "type": "error",
                    "message": "Frame size mismatch",
                    "code": "FRAME_SIZE_ERROR",
                }

            frame = frame_array.reshape((height, width, 3))

            keypoint_extractor = get_keypoint_extractor()
            keypoints = keypoint_extractor.extract_keypoints(frame)

            landmarks = {
                "left_hand": keypoints.get("left_hand"),
                "right_hand": keypoints.get("right_hand"),
            }

            logger.info(
                f"[AIService] Frame extracted: left={len(landmarks['left_hand']) if landmarks['left_hand'] else 0} points, "
                f"right={len(landmarks['right_hand']) if landmarks['right_hand'] else 0} points"
            )

            return await self.process_landmarks(landmarks, mode=mode, user_id=user_id)

        except Exception as e:
            import traceback

            logger.error(f"[AIService] Error processing frame: {e}")
            logger.error(f"[AIService] Traceback: {traceback.format_exc()}")
            return {"type": "error", "message": str(e), "code": "FRAME_ERROR"}

    def reset(self):
        self.detector.reset_sequence()
        self.last_audio_text = ""
        self.last_context = ""

    async def finalize(self, mode: str, user_id: Optional[str] = None) -> Dict:
        """Finalize fingerspelling sequence and normalize with Groq."""
        logger.warning(
            f"[AIService] ================= FINALIZE CALLED ================="
        )
        logger.warning(f"[AIService] finalize called, mode: {mode}")

        try:
            logger.warning(f"[AIService] finalize called, mode: {mode}")

            sequence = ""
            confidence = 0.0

            if mode == "fingerspelling":
                fingerspelling = self.detector._fingerspelling_recognizer
                logger.warning(
                    f"[AIService] fingerspelling recognizer: {fingerspelling}"
                )

                if fingerspelling:
                    logger.warning(
                        f"[AIService] letter_history: {getattr(fingerspelling, 'letter_history', [])}"
                    )
                    logger.warning(
                        f"[AIService] letter_buffer: {getattr(fingerspelling, 'letter_buffer', [])}"
                    )

                    if hasattr(fingerspelling, "letter_history"):
                        sequence = "".join(fingerspelling.letter_history)
                        logger.warning(f"[AIService] Acquired sequence: '{sequence}'")

                        if len(fingerspelling.letter_buffer) > 0:
                            confidences = [
                                c for _, c in fingerspelling.letter_buffer if c > 0
                            ]
                            if confidences:
                                confidence = sum(confidences) / len(confidences)
                                logger.warning(
                                    f"[AIService] Average confidence: {confidence}"
                                )

            logger.warning(
                f"[AIService] Finalizing: sequence='{sequence}', mode={mode}"
            )

            normalized_text = sequence
            if sequence and len(sequence) > 0:
                logger.warning(f"[AIService] Calling Groq with sequence: '{sequence}'")
                normalized_text = await self.normalizer.normalize(
                    sequence, mode, self.last_context
                )
                logger.warning(f"[AIService] Groq returned: '{normalized_text}'")

                if (
                    normalized_text != "[entrada no reconocida]"
                    and not normalized_text.startswith("[error")
                ):
                    self.last_context = normalized_text

            logger.warning(
                f"[AIService] Finalize result: normalized='{normalized_text}'"
            )

            logger.warning(
                f"[AIService] ================= END FINALIZE ================="
            )

            audio_data = None
            # No generamos audio en backend - el frontend usa flutter_tts
            logger.info(
                f"[AIService] Finalize - Text ready for TTS: '{normalized_text}'"
            )

            return {
                "type": "translation",
                "text": normalized_text
                if not normalized_text.startswith("[error")
                else "",
                "audio": audio_data,
                "confidence": confidence,
                "has_keypoints": sequence is not None and len(sequence) > 0,
                "phrase": normalized_text
                if normalized_text and not normalized_text.startswith("[error")
                else "",
                "is_recording": False,
                "candidate": "",
                "candidate_confidence": confidence,
                "mode": mode,
                "is_finalized": True,
            }

        except Exception as e:
            import traceback

            logger.error(f"[AIService] Error in finalize: {e}")
            logger.error(f"[AIService] Traceback: {traceback.format_exc()}")
            return {"type": "error", "message": str(e), "code": "FINALIZE_ERROR"}


def _mirror_landmarks(left_landmarks, right_landmarks):
    """
    Mirror landmarks and swap handedness to match Python scripts behavior.
    Python scripts use cv2.flip(frame, 1) before MediaPipe detection.
    This function mirrors the x-coordinates and swaps left/right labels.
    """

    def _flip_and_swap(data):
        if data and isinstance(data, list) and len(data) >= 21:
            return [
                [1.0 - pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0] for pt in data[:21]
            ]
        return None

    # Swap left and right, flip x-coordinates
    return _flip_and_swap(right_landmarks), _flip_and_swap(left_landmarks)


ai_service = None


def get_ai_service() -> AIService:
    global ai_service
    if ai_service is None:
        ai_service = AIService()
    return ai_service
