import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Get absolute path - go from sign_detector_manager.py -> ml/ -> sign_language_model/
_current_file = os.path.abspath(__file__)
SIGN_LANGUAGE_MODEL_DIR = os.path.join(
    os.path.dirname(_current_file), "sign_language_model"
)
# Convert to absolute in case it's relative
SIGN_LANGUAGE_MODEL_DIR = os.path.abspath(SIGN_LANGUAGE_MODEL_DIR)

logger.info(f"[SignDetectorManager] Base dir: {os.path.dirname(_current_file)}")
logger.info(f"[SignDetectorManager] SIGN_LANGUAGE_MODEL_DIR: {SIGN_LANGUAGE_MODEL_DIR}")

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
        logger.info("[SignDetectorManager] Starting to load models...")
        logger.info(
            f"[SignDetectorManager] SIGN_LANGUAGE_MODEL_DIR: {SIGN_LANGUAGE_MODEL_DIR}"
        )

        # Check directories
        data_dir = os.path.join(SIGN_LANGUAGE_MODEL_DIR, "data", "reference")
        logger.info(
            f"[SignDetectorManager] Reference dir exists: {os.path.exists(data_dir)}"
        )

        if os.path.exists(data_dir):
            for subdir in ["handshape", "fingerspelling"]:
                subdir_path = os.path.join(data_dir, subdir)
                logger.info(
                    f"[SignDetectorManager] {subdir} dir exists: {os.path.exists(subdir_path)}"
                )
                if os.path.exists(subdir_path):
                    files = os.listdir(subdir_path)
                    logger.info(f"[SignDetectorManager] {subdir} files: {files}")

        try:
            from models.recognizers.handshape_model import HandshapeRecognizer

            handshape_path = os.path.join(
                SIGN_LANGUAGE_MODEL_DIR, "data", "reference", "handshape"
            )
            logger.info(
                f"[SignDetectorManager] Checking handshape at: {handshape_path}"
            )
            if os.path.exists(handshape_path):
                self._handshape_recognizer = HandshapeRecognizer(handshape_path)
                logger.info(
                    f"[SignDetectorManager] Handshape model loaded, is_loaded: {self._handshape_recognizer.is_loaded}"
                )
            else:
                logger.warning(
                    f"[SignDetectorManager] Handshape model not found at {handshape_path}"
                )
        except Exception as e:
            logger.error(f"[SignDetectorManager] Error loading handshape model: {e}")
            import traceback

            logger.error(f"[SignDetectorManager] Traceback: {traceback.format_exc()}")

        try:
            from models.recognizers.fingerspelling_model import FingerspellingRecognizer

            fingerspelling_path = os.path.join(
                SIGN_LANGUAGE_MODEL_DIR, "data", "reference", "fingerspelling"
            )
            logger.info(
                f"[SignDetectorManager] Checking fingerspelling at: {fingerspelling_path}"
            )
            if os.path.exists(fingerspelling_path):
                self._fingerspelling_recognizer = FingerspellingRecognizer(
                    fingerspelling_path
                )
                logger.info(
                    f"[SignDetectorManager] Fingerspelling model loaded, is_loaded: {self._fingerspelling_recognizer.is_loaded}"
                )
                if not self._fingerspelling_recognizer.is_loaded:
                    logger.warning(
                        f"[SignDetectorManager] Fingerspelling model loaded but is_loaded=False, error: {self._fingerspelling_recognizer.load_error}"
                    )
            else:
                logger.warning(
                    f"[SignDetectorManager] Fingerspelling model not found at {fingerspelling_path}"
                )
        except Exception as e:
            logger.error(
                f"[SignDetectorManager] Error loading fingerspelling model: {e}"
            )
            import traceback

            logger.error(f"[SignDetectorManager] Traceback: {traceback.format_exc()}")

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
