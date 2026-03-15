import cv2
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class KeypointExtractor:
    def __init__(self):
        logger.info("KeypointExtractor initialized (placeholder - needs model files)")
        
    def extract_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        # Return placeholder for now
        # To enable: download MediaPipe model files and initialize properly
        return {
            "hands": [],
            "pose": None,
            "face": None,
            "frame_shape": frame.shape
        }

    def close(self):
        pass


class SignRecognizer:
    def __init__(self):
        self.sequence_buffer: List[Dict] = []
        self.sequence_length = 30
        self.frame_count = 0
        logger.info("SignRecognizer initialized")

    def process_frame(self, frame: np.ndarray) -> Dict:
        self.frame_count += 1
        
        # Add to sequence buffer
        self.sequence_buffer.append({"frame": self.frame_count})
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
        
        result = self._recognize_sign()
        
        return {
            "keypoints": None,
            "text": result["text"],
            "confidence": result["confidence"],
            "sequence_length": len(self.sequence_buffer),
            "frame_count": self.frame_count
        }

    def _recognize_sign(self) -> Dict:
        if len(self.sequence_buffer) < 5:
            return {"text": "", "confidence": 0.0}
        
        return {
            "text": "Hand detected",
            "confidence": 0.8
        }

    def reset_sequence(self):
        self.sequence_buffer = []

    def close(self):
        pass


keypoint_extractor = None
sign_recognizer = None


def get_keypoint_extractor() -> KeypointExtractor:
    global keypoint_extractor
    if keypoint_extractor is None:
        keypoint_extractor = KeypointExtractor()
    return keypoint_extractor


def get_sign_recognizer() -> SignRecognizer:
    global sign_recognizer
    if sign_recognizer is None:
        sign_recognizer = SignRecognizer()
    return sign_recognizer
