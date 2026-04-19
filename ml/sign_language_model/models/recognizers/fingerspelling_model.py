import os
import logging
import numpy as np
from typing import Optional, Dict, List
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_MAP = {i: letter for i, letter in enumerate(ALPHABET)}


class FingerspellingRecognizer:
    """
    Recognizer for ASL fingerspelling using Hugging Face model.
    Model: sid220/asl-now-fingerspelling

    Input: 21 landmarks × 3 coords (x, y, z) = 63 features
    Output: Probability for each letter A-Z
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False

        self.letter_buffer = deque(maxlen=10)
        self.last_letter = ""
        self.last_detection_time = datetime.min
        self.cooldown = timedelta(seconds=0.3)
        self.confidence_threshold = 0.7
        self.letter_history = []

        self._load_model()

    def _load_model(self):
        """Load Keras model from Hugging Face"""
        weights_file = os.path.join(self.model_path, "asl-now-weights.h5")

        try:
            import tensorflow as tf

            if os.path.exists(weights_file):
                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Input(shape=(21, 3)),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation="relu"),
                        tf.keras.layers.Dense(26, activation="softmax"),
                    ]
                )
                model.load_weights(weights_file)
                self.model = model
                self.is_loaded = True
                logger.info(f"[FingerspellingRecognizer] Loaded weights from {weights_file}")
                logger.info(
                    f"[FingerspellingRecognizer] Model input shape: {self.model.input_shape}"
                )
                return

        except Exception as e:
            logger.error(f"[FingerspellingRecognizer] Error loading model: {e}")

        logger.warning(f"[FingerspellingRecognizer] Model not found at {self.model_path}")
        self.is_loaded = False

    def extract_features(self, landmarks) -> np.ndarray:
        """Extract features from hand landmarks."""
        if not landmarks or len(landmarks) < 21:
            return None

        features = []
        for i in range(21):
            point = landmarks[i] if i < len(landmarks) else [0.0, 0.0, 0.0]
            features.append(
                [
                    float(point[0]) if len(point) > 0 else 0.0,
                    float(point[1]) if len(point) > 1 else 0.0,
                    float(point[2]) if len(point) > 2 else 0.0,
                ]
            )

        return np.array(features).reshape(1, 21, 3)

    def predict(self, left_landmarks, right_landmarks) -> Dict:
        """Predict letter from hand landmarks."""
        if not self.is_loaded or self.model is None:
            return self._placeholder_predict(left_landmarks, right_landmarks)

        try:
            landmarks = (
                right_landmarks
                if right_landmarks and len(right_landmarks) >= 21
                else left_landmarks
            )

            if not landmarks or len(landmarks) < 21:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "is_recording": False,
                    "candidate": "",
                    "sequence": "",
                }

            features = self.extract_features(landmarks)
            if features is None:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "is_recording": False,
                    "candidate": "",
                    "sequence": "",
                }

            predictions = self.model.predict(features, verbose=0)[0]
            letter_idx = int(np.argmax(predictions))
            confidence = float(predictions[letter_idx])

            letter = LABEL_MAP.get(letter_idx, "?")

            self.letter_buffer.append((letter, confidence))

            if len(self.letter_buffer) >= 5:
                letter_counts = {}
                for l, c in self.letter_buffer:
                    if c >= self.confidence_threshold:
                        letter_counts[l] = letter_counts.get(l, 0) + 1

                if letter_counts:
                    best_letter = max(letter_counts, key=letter_counts.get)

                    now = datetime.now()
                    if (
                        best_letter != self.last_letter
                        or (now - self.last_detection_time) > self.cooldown
                    ):
                        self.last_letter = best_letter
                        self.last_detection_time = now

                        if best_letter != "?" and confidence >= self.confidence_threshold:
                            self.letter_history.append(best_letter)
                            return {
                                "text": "",
                                "confidence": confidence,
                                "is_recording": True,
                                "candidate": best_letter,
                                "sequence": "".join(self.letter_history[-10:]),
                            }

            return {
                "text": "",
                "confidence": confidence,
                "is_recording": True,
                "candidate": letter if confidence >= self.confidence_threshold else "",
                "sequence": "".join(self.letter_history[-10:]),
            }

        except Exception as e:
            logger.error(f"[FingerspellingRecognizer] Prediction error: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "is_recording": False,
                "candidate": "",
                "sequence": "",
            }

    def _placeholder_predict(self, left_landmarks, right_landmarks) -> Dict:
        """Placeholder when model not loaded"""
        has_left = left_landmarks and len(left_landmarks) >= 21
        has_right = right_landmarks and len(right_landmarks) >= 21

        if not has_left and not has_right:
            return {
                "text": "",
                "confidence": 0.0,
                "is_recording": False,
                "candidate": "",
                "sequence": "",
            }

        return {
            "text": "",
            "confidence": 0.5,
            "is_recording": True,
            "candidate": "A",
            "sequence": "",
        }

    def reset(self):
        """Reset recognizer state"""
        self.letter_buffer.clear()
        self.last_letter = ""
        self.last_detection_time = datetime.min
        self.letter_history = []
