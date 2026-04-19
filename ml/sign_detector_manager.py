import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

SIGN_LANGUAGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "sign_language_model")

if SIGN_LANGUAGE_MODEL_DIR not in sys.path:
    sys.path.insert(0, SIGN_LANGUAGE_MODEL_DIR)


class SignDetectorManager:
    """
    Unified wrapper for sign language detection.
    Supports two modes:
    - handshape: Static handshapes (ASL signs) -> concepts
    - fingerspelling: Continuous fingerspelling (A-Z) -> letters
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.mode = "handshape"
        self._handshape_recognizer = None
        self._fingerspelling_recognizer = None

        self._load_models()
        self._initialized = True
        logger.info("[SignDetectorManager] Initialized")

    def _load_models(self):
        """Load both models (lazy loading)"""
        try:
            from models.recognizers.handshape_model import HandshapeRecognizer

            handshape_path = os.path.join(
                SIGN_LANGUAGE_MODEL_DIR, "data", "reference", "handshape"
            )
            if os.path.exists(handshape_path):
                self._handshape_recognizer = HandshapeRecognizer(handshape_path)
                logger.info("[SignDetectorManager] Handshape model loaded")
            else:
                logger.warning(
                    f"[SignDetectorManager] Handshape model not found at {handshape_path}"
                )
        except Exception as e:
            logger.error(f"[SignDetectorManager] Error loading handshape model: {e}")

        try:
            from models.recognizers.fingerspelling_model import FingerspellingRecognizer

            fingerspelling_path = os.path.join(
                SIGN_LANGUAGE_MODEL_DIR, "data", "reference", "fingerspelling"
            )
            if os.path.exists(fingerspelling_path):
                self._fingerspelling_recognizer = FingerspellingRecognizer(
                    fingerspelling_path
                )
                logger.info("[SignDetectorManager] Fingerspelling model loaded")
            else:
                logger.warning(
                    f"[SignDetectorManager] Fingerspelling model not found at {fingerspelling_path}"
                )
        except Exception as e:
            logger.error(
                f"[SignDetectorManager] Error loading fingerspelling model: {e}"
            )

    def set_mode(self, mode: str):
        """Set the detection mode"""
        if mode not in ["handshape", "fingerspelling"]:
            logger.warning(
                f"[SignDetectorManager] Unknown mode: {mode}, using 'handshape'"
            )
            mode = "handshape"
        self.mode = mode
        logger.info(f"[SignDetectorManager] Mode set to: {mode}")

    def process_landmarks(self, left_landmarks, right_landmarks) -> Dict:
        """
        Process landmarks and return raw text based on current mode.

        Args:
            left_landmarks: List of [x, y, z] points for left hand (21 points)
            right_landmarks: List of [x, y, z] points for right hand (21 points)

        Returns:
            Dict with keys:
                - text: raw text output (concepts or letters)
                - confidence: confidence score
                - is_recording: whether recording is in progress
        """
        if self.mode == "handshape":
            return self._process_handshape(left_landmarks, right_landmarks)
        elif self.mode == "fingerspelling":
            return self._process_fingerspelling(left_landmarks, right_landmarks)
        else:
            return {"text": "", "confidence": 0.0, "is_recording": False}

    def _process_handshape(self, left_landmarks, right_landmarks) -> Dict:
        """Process static handshapes"""
        if self._handshape_recognizer is None:
            logger.warning("[SignDetectorManager] Handshape recognizer not loaded")
            return {
                "text": "",
                "confidence": 0.0,
                "is_recording": False,
                "candidate": "",
            }

        try:
            result = self._handshape_recognizer.predict(left_landmarks, right_landmarks)
            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "is_recording": result.get("is_recording", False),
                "candidate": result.get("candidate", ""),
            }
        except Exception as e:
            logger.error(f"[SignDetectorManager] Error in handshape recognition: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "is_recording": False,
                "candidate": "",
            }

    def _process_fingerspelling(self, left_landmarks, right_landmarks) -> Dict:
        """Process continuous fingerspelling"""
        if self._fingerspelling_recognizer is None:
            logger.warning("[SignDetectorManager] Fingerspelling recognizer not loaded")
            return {
                "text": "",
                "confidence": 0.0,
                "is_recording": False,
                "candidate": "",
            }

        try:
            result = self._fingerspelling_recognizer.predict(
                left_landmarks, right_landmarks
            )
            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "is_recording": result.get("is_recording", False),
                "candidate": result.get("candidate", ""),
            }
        except Exception as e:
            logger.error(
                f"[SignDetectorManager] Error in fingerspelling recognition: {e}"
            )
            return {
                "text": "",
                "confidence": 0.0,
                "is_recording": False,
                "candidate": "",
            }

    def reset_sequence(self):
        """Reset the recognition sequence"""
        if self._handshape_recognizer:
            self._handshape_recognizer.reset()
        if self._fingerspelling_recognizer:
            self._fingerspelling_recognizer.reset()


sign_detector_manager = None


def get_sign_detector_manager() -> SignDetectorManager:
    global sign_detector_manager
    if sign_detector_manager is None:
        sign_detector_manager = SignDetectorManager()
    return sign_detector_manager
