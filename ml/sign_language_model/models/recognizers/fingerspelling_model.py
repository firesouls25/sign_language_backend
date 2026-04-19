import os
import logging
import numpy as np
from typing import Optional, Dict, List
from collections import deque
from datetime import datetime, timedelta
import joblib

logger = logging.getLogger(__name__)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_TOKEN = 26


class FingerspellingRecognizer:
    """
    Recognizer for continuous fingerspelling (A-Z).
    Uses sequence model with connectionist temporal classification (CTC).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.is_loaded = False

        self.sequence_buffer = deque(maxlen=60)
        self.last_sign = ""
        self.last_detection_time = datetime.min
        self.cooldown = timedelta(seconds=0.5)

        self._load_model()

    def _load_model(self):
        """Load trained model"""
        model_file = os.path.join(self.model_path, "model.joblib")

        if os.path.exists(model_file):
            try:
                self.model = joblib.load(model_file)
                self.is_loaded = True
                logger.info("[FingerspellingRecognizer] Model loaded")
            except Exception as e:
                logger.error(f"[FingerspellingRecognizer] Error loading model: {e}")
                self.is_loaded = False
        else:
            logger.warning(f"[FingerspellingRecognizer] Model not found at {self.model_path}")
            logger.warning(
                "[FingerspellingRecognizer] Using placeholder - need to train with ASL Fingerspelling dataset"
            )
            self.is_loaded = False

    def extract_features(self, left_landmarks, right_landmarks) -> np.ndarray:
        """Extract features from hand landmarks"""
        left_flat = []
        right_flat = []

        if left_landmarks and len(left_landmarks) == 21:
            for point in left_landmarks:
                left_flat.extend([point[0], point[1], point[2]])
        else:
            left_flat = [0.0] * 63

        if right_landmarks and len(right_landmarks) == 21:
            for point in right_landmarks:
                right_flat.extend([point[0], point[1], point[2]])
        else:
            right_flat = [0.0] * 63

        return np.array(left_flat + right_flat)

    def predict(self, left_landmarks, right_landmarks) -> Dict:
        """Predict fingerspelling sequence"""
        has_left = left_landmarks and len(left_landmarks) == 21
        has_right = right_landmarks and len(right_landmarks) == 21

        if not has_left and not has_right:
            if len(self.sequence_buffer) > 0:
                text = self._decode_sequence()
                if text and text != self.last_sign:
                    now = datetime.now()
                    if now - self.last_detection_time > self.cooldown:
                        self.last_sign = text
                        self.last_detection_time = now
                        result = text
                        self.sequence_buffer.clear()
                        return {
                            "text": result,
                            "confidence": 0.7,
                            "is_recording": False,
                            "candidate": "",
                        }
            return {"text": "", "confidence": 0.0, "is_recording": False, "candidate": ""}

        features = self.extract_features(left_landmarks, right_landmarks)
        self.sequence_buffer.append(features)

        if not self.is_loaded:
            candidate = self._placeholder_predict(left_landmarks, right_landmarks)
            return {"text": "", "confidence": 0.5, "is_recording": True, "candidate": candidate}

        try:
            features_array = np.array(self.sequence_buffer)
            features_array = features_array.reshape(1, features_array.shape[0], -1)

            prediction = self.model.predict(features_array)
            letter = self._decode_prediction(prediction[0])

            return {"text": "", "confidence": 0.6, "is_recording": True, "candidate": letter}

        except Exception as e:
            logger.error(f"[FingerspellingRecognizer] Prediction error: {e}")
            return {"text": "", "confidence": 0.0, "is_recording": True, "candidate": ""}

    def _placeholder_predict(self, left_landmarks, right_landmarks) -> str:
        """Placeholder: return letter based on simple heuristics"""
        if not left_landmarks and not right_landmarks:
            return ""

        landmarks = left_landmarks if left_landmarks else right_landmarks

        index_tip = landmarks[8] if len(landmarks) > 8 else [0, 0, 0]
        thumb_tip = landmarks[4] if len(landmarks) > 4 else [0, 0, 0]
        pinky_tip = landmarks[20] if len(landmarks) > 20 else [0, 0, 0]

        x_spread = abs(index_tip[0] - pinky_tip[0])

        if x_spread < 0.1:
            return "I"
        elif x_spread < 0.2:
            return "L"
        else:
            return "O"

    def _decode_prediction(self, prediction) -> str:
        """Convert model prediction to letter"""
        if hasattr(prediction, "argmax"):
            idx = prediction.argmax()
        else:
            idx = int(prediction)

        if 0 <= idx < len(ALPHABET):
            return ALPHABET[idx]
        return ""

    def _decode_sequence(self) -> str:
        """Decode the accumulated sequence"""
        if not self.sequence_buffer:
            return ""

        return self.last_sign

    def reset(self):
        """Reset recognizer state"""
        self.sequence_buffer.clear()
        self.last_sign = ""
        self.last_detection_time = datetime.min
